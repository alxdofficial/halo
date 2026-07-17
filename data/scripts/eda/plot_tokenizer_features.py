"""Visualize pre-projection features from the reference HALO v2 PHz tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from data.scripts.eda.grid_io import (
    REPO,
    GridRef,
    discover_grids,
    matching_refs,
    output_dir,
    output_subdir,
    sample_indices,
    triad_indices,
)
from data.scripts.eda.orientation import gravity_alignment, rotate_vectors
from data.scripts.eda.plot_gravity_aligned_overlays import GRAVITY_STATE
from data.scripts.eda.plot_sensor_trajectories import (
    DEFAULT_STREAMS,
    DISPLAY_NAMES,
    INK,
    KNOWN_GYRO_UNIT_ISSUES,
    MUTED,
)


DEFAULT_LEGACY_ROOT = REPO.parent / "legacy_code"
PATCH_SECONDS = 1.5
DFT_SIZE = 256


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


def _display_ref(ref: GridRef) -> str:
    dataset = DISPLAY_NAMES.get(ref.dataset, ref.dataset.replace("_", " ").title())
    return f"{dataset}\n{ref.stream.replace('_', ' ')}"


def _load_tokenizer_class(legacy_root: Path):
    source = legacy_root / "model" / "feature_extractor.py"
    if not source.exists():
        raise FileNotFoundError(
            f"HALO v2 tokenizer implementation not found at {source}. The clean repo "
            "does not yet contain model code; pass --legacy-root explicitly."
        )
    spec = importlib.util.spec_from_file_location("halo_v2_reference_feature_extractor", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load tokenizer module from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PhysicalFilterbankTokenizer, source


def _prepare_window(
    ref: GridRef, window: np.ndarray, orientation: str
) -> tuple[np.ndarray, dict]:
    if ref.dataset in KNOWN_GYRO_UNIT_ISSUES and triad_indices(ref, "gyro"):
        raise ValueError(KNOWN_GYRO_UNIT_ISSUES[ref.dataset])
    acc_indices = triad_indices(ref, "acc")
    if acc_indices is None:
        raise ValueError(f"{ref.key}: missing accelerometer")

    prepared = np.asarray(window, dtype=np.float32).copy()
    if orientation == "raw":
        return prepared, {}
    if GRAVITY_STATE.get((ref.dataset, ref.stream)) != "present":
        raise ValueError(f"{ref.key}: gravity alignment requires gravity-present acceleration")

    estimate = gravity_alignment(prepared[:, acc_indices])
    prepared[:, acc_indices] = rotate_vectors(prepared[:, acc_indices], estimate.rotation)
    gyro_indices = triad_indices(ref, "gyro")
    if gyro_indices is not None:
        prepared[:, gyro_indices] = rotate_vectors(prepared[:, gyro_indices], estimate.rotation)
    return prepared, {
        "gravity_norm_g": round(estimate.gravity_norm, 6),
        "alignment_error": estimate.alignment_error,
    }


def extract_ref_features(
    ref: GridRef,
    label: str,
    count: int,
    seed: int,
    tokenizer_class,
    orientation: str,
) -> dict:
    import torch

    indices = sample_indices(ref, label, count, seed)
    data = ref.load_data()
    patch_len = round(PATCH_SECONDS * ref.rate_hz)
    if patch_len > DFT_SIZE:
        raise ValueError(f"{ref.key}: {patch_len} samples exceed tokenizer dft_size={DFT_SIZE}")

    sample_patches = []
    sample_meta = []
    for index in indices:
        window, alignment_meta = _prepare_window(
            ref, np.asarray(data[index], dtype=np.float32), orientation
        )
        n_patches = window.shape[0] // patch_len
        if n_patches < 1:
            raise ValueError(f"{ref.key}: window is shorter than {PATCH_SECONDS}s")
        patches = window[:n_patches * patch_len].reshape(n_patches, patch_len, window.shape[1])
        padded = np.zeros((n_patches, DFT_SIZE, window.shape[1]), dtype=np.float32)
        padded[:, :patch_len] = patches
        sample_patches.append(padded)
        sample_meta.append({
            "window_index": int(index),
            "subject": ref.subjects[int(index)],
            **alignment_meta,
        })

    patch_count = min(value.shape[0] for value in sample_patches)
    batch = np.stack([value[:patch_count] for value in sample_patches])
    tokenizer = tokenizer_class(d_model=1, dft_size=DFT_SIZE, norm="none", learnable=False)
    tokenizer.eval()
    with torch.no_grad():
        patches = torch.from_numpy(batch)
        rates, lengths = tokenizer._prep_rate_len(
            ref.rate_hz, patch_len, len(batch), patches.device, patches.dtype
        )
        energy, centers, sigma, dc = tokenizer._band_energy(patches, rates, lengths)
        observable, resolution = tokenizer._observability_masks(rates, lengths, centers, sigma)
        log_energy = torch.log1p(energy).cpu().numpy()
        amplitude = torch.log1p(energy.sum(dim=-1)).cpu().numpy()

    median_energy = np.median(log_energy, axis=(0, 1))
    median_amplitude = np.median(amplitude, axis=(0, 1))
    median_dc = np.median(dc.cpu().numpy(), axis=(0, 1))
    return {
        "median_log_energy": median_energy,
        "median_log_amplitude": median_amplitude,
        "median_dc": median_dc,
        "centers_hz": centers.cpu().numpy(),
        "observable": observable[0].cpu().numpy(),
        "resolution": resolution[0].cpu().numpy(),
        "sample_meta": sample_meta,
        "patch_count": patch_count,
        "patch_len": patch_len,
    }


def plot_features(
    refs: Sequence[GridRef], features: dict[str, dict], label: str,
    destination: Path, orientation: str,
) -> Path:
    import matplotlib.pyplot as plt

    modality_values = {"acc": [], "gyro": []}
    for ref in refs:
        for modality in modality_values:
            indices = triad_indices(ref, modality)
            if indices is not None:
                modality_values[modality].append(features[ref.key]["median_log_energy"][list(indices)])
    limits = {}
    for modality, values in modality_values.items():
        combined = np.concatenate([value.ravel() for value in values]) if values else np.asarray([0.0])
        limits[modality] = (float(np.min(combined)), float(np.percentile(combined, 99)))

    fig, axes = plt.subplots(
        len(refs), 2, figsize=(14.0, 2.25 * len(refs) + 2.3),
        squeeze=False, facecolor="white",
    )
    images = {"acc": [], "gyro": []}
    for row, ref in enumerate(refs):
        centers = features[ref.key]["centers_hz"]
        tick_freqs = np.asarray([0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0])
        tick_indices = [int(np.argmin(np.abs(centers - value))) for value in tick_freqs]
        for column, (modality, title) in enumerate((
            ("acc", "Accelerometer filterbank"), ("gyro", "Gyroscope filterbank")
        )):
            ax = axes[row, column]
            indices = triad_indices(ref, modality)
            if indices is None:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "Gyroscope not recorded", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color=MUTED)
            else:
                matrix = features[ref.key]["median_log_energy"][list(indices)]
                image = ax.imshow(matrix, aspect="auto", cmap="cividis",
                                  vmin=limits[modality][0], vmax=limits[modality][1])
                images[modality].append(image)
                ax.set_yticks([0, 1, 2], labels=["x", "y", "z"])
                ax.set_xticks(tick_indices, labels=[f"{value:g}" for value in tick_freqs])
                ax.set_xlabel("Band center (Hz)", fontsize=8)
                ax.tick_params(labelsize=8)
                amplitude = features[ref.key]["median_log_amplitude"][list(indices)]
                dc = features[ref.key]["median_dc"][list(indices)]
                ax.text(0.99, 0.05,
                        f"log total E {float(np.median(amplitude)):.2f}\n"
                        f"DC [{dc[0]:+.2f}, {dc[1]:+.2f}, {dc[2]:+.2f}]",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                        color="white", bbox={"facecolor": INK, "edgecolor": "none", "alpha": 0.72})
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight="semibold", color=INK)
            if column == 0:
                ax.text(-0.14, 0.5, _display_ref(ref), transform=ax.transAxes,
                        ha="right", va="center", fontsize=9, color=INK,
                        fontweight="semibold")

    colorbar_positions = {"acc": [0.22, 0.04, 0.27, 0.012],
                          "gyro": [0.62, 0.04, 0.27, 0.012]}
    for modality, column in (("acc", 0), ("gyro", 1)):
        if images[modality]:
            color_axis = fig.add_axes(colorbar_positions[modality])
            colorbar = fig.colorbar(images[modality][0], cax=color_axis,
                                    orientation="horizontal")
            colorbar.set_label("Median log(1 + band energy)", fontsize=8, labelpad=2)
            colorbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"HALO v2 tokenizer features: {label.replace('_', ' ')}",
                 x=0.055, y=0.988, ha="left", fontsize=18, fontweight="bold", color=INK)
    orientation_text = ("raw device-frame" if orientation == "raw"
                        else "gravity-aligned sensitivity view")
    alignment_text = refs[0].alignment.replace("_", " ")
    fig.text(
        0.055, 0.959,
        f"{orientation_text} | {alignment_text} 1.5 s patches | "
        "median across four subjects and patches | "
        "reference physical-Hz filterbank before normalization or learned projection",
        ha="left", va="top", fontsize=9.5, color=MUTED,
    )
    fig.subplots_adjust(left=0.2, right=0.96, top=0.89, bottom=0.1, hspace=0.46, wspace=0.28)
    suffix = "" if orientation == "raw" else "_gravity_aligned"
    path = destination / f"{_slug(label)}_tokenizer_features{suffix}.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def run(
    label: str,
    alignment_name: str,
    selectors: Sequence[str] | None,
    count: int,
    seed: int,
    max_streams: int,
    destination: Path,
    legacy_root: Path,
    orientation: str,
) -> tuple[Path, Path]:
    tokenizer_class, source = _load_tokenizer_class(legacy_root)
    refs = discover_grids(alignment_name)
    selected = matching_refs(refs, label, tuple(selectors) if selectors else DEFAULT_STREAMS)
    selected = selected[:max_streams]
    if not selected:
        raise ValueError(f"No selected grids contain {label!r}")

    features = {
        ref.key: extract_ref_features(ref, label, count, seed, tokenizer_class, orientation)
        for ref in selected
    }
    destination = output_subdir(destination, orientation)
    figure = plot_features(selected, features, label, destination, orientation)
    suffix = "" if orientation == "raw" else "_gravity_aligned"
    manifest = destination / f"{_slug(label)}_tokenizer_features{suffix}.json"
    manifest.write_text(json.dumps({
        "label": label,
        "seed": seed,
        "alignment": alignment_name,
        "input_orientation": orientation,
        "tokenizer_source": str(source),
        "tokenizer_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tokenizer_config": {
            "type": "PhysicalFilterbankTokenizer",
            "n_bands": 32,
            "f_min_hz": 0.3,
            "f_max_hz": 15.0,
            "Q": 4.0,
            "dft_size": DFT_SIZE,
            "patch_seconds": PATCH_SECONDS,
            "normalization": "none",
        },
        "checkpoint_loaded": False,
        "learned_projection_visualized": False,
        "note": "The clean repository has no HALO v2 model/checkpoint. These are pre-projection "
                "features from the sibling legacy implementation, not learned encoder embeddings.",
        "streams": {
            ref.key: {
                "channels": list(ref.channels),
                "channel_mask": list(ref.mask),
                "band_centers_hz": features[ref.key]["centers_hz"].tolist(),
                "nyquist_observable": features[ref.key]["observable"].tolist(),
                "resolution": features[ref.key]["resolution"].tolist(),
                "median_log_band_energy": features[ref.key]["median_log_energy"].tolist(),
                "median_log_amplitude": features[ref.key]["median_log_amplitude"].tolist(),
                "median_dc": features[ref.key]["median_dc"].tolist(),
                "samples": features[ref.key]["sample_meta"],
            }
            for ref in selected
        },
    }, indent=2) + "\n")
    return figure, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", nargs="+", required=True)
    parser.add_argument("--alignment", default="harmonised",
                        choices=("harmonised", "non_harmonised"))
    parser.add_argument("--streams", nargs="*", default=None)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-streams", type=int, default=6)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--orientation", choices=("raw", "gravity_aligned"), default="raw")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    destination = output_subdir(output_dir(args.output_dir), "tokenizer_features")
    for label in args.label:
        for path in run(label, args.alignment, args.streams, args.samples, args.seed,
                        args.max_streams, destination, args.legacy_root, args.orientation):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
