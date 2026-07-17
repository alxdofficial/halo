"""Aggregate robust, rotation-invariant activity signatures across datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from data.scripts.eda.grid_io import (
    GridRef,
    discover_grids,
    matching_refs,
    output_dir,
    output_subdir,
    sample_indices,
    triad_indices,
)
from data.scripts.eda.plot_sensor_trajectories import (
    DEFAULT_STREAMS,
    DISPLAY_NAMES,
    GRID,
    INK,
    KNOWN_GYRO_UNIT_ISSUES,
    MUTED,
)
from data.scripts.eda.plot_tokenizer_features import (
    DEFAULT_LEGACY_ROOT,
    DFT_SIZE,
    PATCH_SECONDS,
    _load_tokenizer_class,
)


DEFAULT_LABELS = (
    "sitting",
    "standing",
    "cycling",
    "walking",
    "walking_upstairs",
    "walking_downstairs",
    "jogging",
)

PALETTE = ("#2f6b9a", "#c38a24", "#d06f2c", "#6e7f3b", "#b04a67", "#4f5964")


@dataclass(frozen=True)
class SignatureSamples:
    ref: GridRef
    label: str
    centers_hz: np.ndarray
    resolution: np.ndarray
    window_indices: np.ndarray
    acc_share: np.ndarray
    acc_total: np.ndarray
    gyro_share: np.ndarray | None
    gyro_total: np.ndarray | None


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


def _display_dataset(ref: GridRef) -> str:
    dataset = DISPLAY_NAMES.get(ref.dataset, ref.dataset.replace("_", " ").title())
    return f"{dataset} ({ref.stream.replace('_', ' ')})"


def _band_area(tokenizer, rates, centers, sigma, dtype, device):
    import torch

    bins = torch.arange(tokenizer.M + 1, device=device, dtype=dtype)
    frequency = bins.unsqueeze(0) * rates.unsqueeze(1) / tokenizer.S
    difference = frequency.unsqueeze(1) - centers.view(1, -1, 1)
    weights = torch.exp(-0.5 * (difference / sigma.view(1, -1, 1)) ** 2)
    return weights.sum(dim=-1).clamp(min=1e-8)


def _triad_summary(energy, area, indices: tuple[int, int, int]):
    """Return per-window relative shape and absolute energy for one xyz triad."""
    triad = energy[:, :, list(indices), :].sum(dim=2)
    density = triad / area[:, None, :]
    share = density / density.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    share = share.median(dim=1).values
    share = share / share.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    total = np.log1p(triad.sum(dim=-1).cpu().numpy())
    total = np.median(total, axis=1)
    return share.cpu().numpy(), total


def extract_signatures(
    ref: GridRef,
    label: str,
    max_windows: int,
    seed: int,
    tokenizer_class,
) -> SignatureSamples:
    import torch

    if ref.dataset in KNOWN_GYRO_UNIT_ISSUES and triad_indices(ref, "gyro"):
        raise ValueError(KNOWN_GYRO_UNIT_ISSUES[ref.dataset])
    acc_indices = triad_indices(ref, "acc")
    if acc_indices is None:
        raise ValueError(f"{ref.key}: no accelerometer triad")

    available = sum(value == label for value in ref.labels)
    count = min(max_windows, available)
    indices = sample_indices(ref, label, count, seed)
    data = ref.load_data()
    patch_len = round(PATCH_SECONDS * ref.rate_hz)
    if patch_len > DFT_SIZE:
        raise ValueError(f"{ref.key}: patch length {patch_len} exceeds dft_size={DFT_SIZE}")
    patch_count = ref.shape[1] // patch_len
    if patch_count < 1:
        raise ValueError(f"{ref.key}: windows are shorter than {PATCH_SECONDS}s")

    windows = np.asarray(data[indices], dtype=np.float32)
    windows = windows[:, :patch_count * patch_len]
    patches = windows.reshape(len(indices), patch_count, patch_len, ref.shape[2])
    padded = np.zeros((len(indices), patch_count, DFT_SIZE, ref.shape[2]), dtype=np.float32)
    padded[:, :, :patch_len] = patches

    tokenizer = tokenizer_class(d_model=1, dft_size=DFT_SIZE, norm="none", learnable=False)
    tokenizer.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(padded)
        rates, lengths = tokenizer._prep_rate_len(
            ref.rate_hz, patch_len, len(indices), tensor.device, tensor.dtype
        )
        energy, centers, sigma, _ = tokenizer._band_energy(tensor, rates, lengths)
        _, resolution = tokenizer._observability_masks(rates, lengths, centers, sigma)
        area = _band_area(tokenizer, rates, centers, sigma, tensor.dtype, tensor.device)
        acc_share, acc_total = _triad_summary(energy, area, acc_indices)
        gyro_indices = triad_indices(ref, "gyro")
        if gyro_indices is None:
            gyro_share = gyro_total = None
        else:
            gyro_share, gyro_total = _triad_summary(energy, area, gyro_indices)

    return SignatureSamples(
        ref=ref,
        label=label,
        centers_hz=centers.cpu().numpy(),
        resolution=resolution[0].cpu().numpy(),
        window_indices=indices,
        acc_share=acc_share,
        acc_total=acc_total,
        gyro_share=gyro_share,
        gyro_total=gyro_total,
    )


def _profile_axis(ax, records: Sequence[SignatureSamples], modality: str) -> None:
    unresolved_end = max(
        float(record.centers_hz[np.flatnonzero(record.resolution < 1)[-1]])
        for record in records if np.any(record.resolution < 1)
    )
    ax.axvspan(records[0].centers_hz[0], unresolved_end, color="#eef1f4", zorder=0)
    for index, record in enumerate(records):
        values = getattr(record, f"{modality}_share")
        if values is None:
            continue
        median = np.percentile(values, 50, axis=0) * 100.0
        lower = np.percentile(values, 25, axis=0) * 100.0
        upper = np.percentile(values, 75, axis=0) * 100.0
        color = PALETTE[index % len(PALETTE)]
        ax.plot(record.centers_hz, median, color=color, linewidth=1.7,
                label=_display_dataset(record.ref))
        ax.fill_between(record.centers_hz, lower, upper, color=color, alpha=0.12, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlim(records[0].centers_hz[0], records[0].centers_hz[-1])
    ax.set_ylabel("Bandwidth-corrected band share (%)")
    ax.grid(color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(records[0].centers_hz[0] * 1.04, 0.97, "<1 cycle/patch",
            transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=7, color=MUTED)


def plot_cross_dataset_profile(
    label: str, records: Sequence[SignatureSamples], destination: Path
) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.2), sharex=True, facecolor="white")
    _profile_axis(axes[0], records, "acc")
    _profile_axis(axes[1], records, "gyro")
    axes[0].set_title("Accelerometer triad", fontsize=11, fontweight="semibold", color=INK)
    axes[1].set_title("Gyroscope triad", fontsize=11, fontweight="semibold", color=INK)
    axes[1].set_xlabel("Physical frequency (Hz)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.62, 0.905),
                   frameon=False, ncol=3, fontsize=8)
    fig.suptitle(f"Cross-dataset activity signature: {label.replace('_', ' ')}",
                 x=0.06, y=0.985, ha="left", fontsize=18, fontweight="bold", color=INK)
    fig.text(
        0.06, 0.95,
        "Window median and IQR. Summing each xyz triad before normalization makes the "
        "profile invariant to rigid 3D rotation.\n"
        "Use the energy plot to assess near-static activities.",
        ha="left", va="top", fontsize=9.5, color=MUTED,
    )
    fig.subplots_adjust(left=0.1, right=0.97, top=0.77, bottom=0.09, hspace=0.34)
    path = destination / f"{_slug(label)}_cross_dataset_signature.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def _balanced_activity_signature(
    records: Sequence[SignatureSamples], modality: str
) -> np.ndarray | None:
    dataset_medians = []
    for record in records:
        values = getattr(record, f"{modality}_share")
        if values is not None:
            dataset_medians.append(np.median(values, axis=0))
    return None if not dataset_medians else np.median(np.stack(dataset_medians), axis=0)


def plot_cross_activity_heatmap(
    labels: Sequence[str], by_label: dict[str, list[SignatureSamples]], destination: Path
) -> Path:
    import matplotlib.pyplot as plt

    centers = by_label[labels[0]][0].centers_hz
    matrices = {}
    support = {}
    for modality in ("acc", "gyro"):
        rows = []
        support[modality] = []
        for label in labels:
            value = _balanced_activity_signature(by_label[label], modality)
            rows.append(np.zeros_like(centers) if value is None else value * 100.0)
            support[modality].append(sum(
                getattr(record, f"{modality}_share") is not None for record in by_label[label]
            ))
        matrices[modality] = np.stack(rows)

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.6), facecolor="white")
    images = []
    for axis, modality, title in zip(
        axes, ("acc", "gyro"), ("Accelerometer signature", "Gyroscope signature")
    ):
        vmax = float(np.percentile(matrices[modality], 99))
        image = axis.imshow(matrices[modality], aspect="auto", cmap="cividis", vmin=0, vmax=vmax)
        images.append(image)
        axis.set_yticks(
            np.arange(len(labels)),
            labels=[f"{label.replace('_', ' ')} ({support[modality][i]} ds)"
                    for i, label in enumerate(labels)],
        )
        tick_freqs = np.asarray([0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0])
        tick_indices = [int(np.argmin(np.abs(centers - value))) for value in tick_freqs]
        axis.set_xticks(tick_indices, labels=[f"{value:g}" for value in tick_freqs])
        axis.set_xlabel("Band center (Hz)")
        axis.set_title(title, fontsize=11, fontweight="semibold", color=INK)
        colorbar = fig.colorbar(image, ax=axis, orientation="horizontal", fraction=0.055, pad=0.15)
        colorbar.set_label("Dataset-balanced median band share (%)", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

    fig.suptitle("Rotation-invariant activity signatures",
                 x=0.055, y=0.985, ha="left", fontsize=18, fontweight="bold", color=INK)
    fig.text(
        0.055, 0.948,
        "Each row is normalized independently; dataset medians receive equal weight. "
        "Brightness shows spectral shape, not absolute intensity.\n"
        "Near-static rows may primarily describe residual sensor noise; pair this view "
        "with the energy plot.",
        ha="left", va="top", fontsize=9.5, color=MUTED,
    )
    fig.subplots_adjust(left=0.17, right=0.98, top=0.82, bottom=0.2, wspace=0.55)
    path = destination / "activity_signature_heatmap.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def plot_energy_distributions(
    labels: Sequence[str], by_label: dict[str, list[SignatureSamples]], destination: Path
) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.3), facecolor="white")
    for axis, modality, title in zip(
        axes, ("acc", "gyro"), ("Accelerometer total energy", "Gyroscope total energy")
    ):
        distributions = []
        dataset_medians = []
        for label in labels:
            arrays = [getattr(record, f"{modality}_total") for record in by_label[label]
                      if getattr(record, f"{modality}_total") is not None]
            distributions.append(np.concatenate(arrays) if arrays else np.asarray([np.nan]))
            dataset_medians.append([float(np.median(values)) for values in arrays])
        boxes = axis.boxplot(distributions, patch_artist=True, showfliers=False, whis=(10, 90),
                             medianprops={"color": INK, "linewidth": 1.4},
                             whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        for index, box in enumerate(boxes["boxes"]):
            box.set_facecolor(PALETTE[index % len(PALETTE)])
            box.set_alpha(0.38)
            box.set_edgecolor(INK)
        for index, medians in enumerate(dataset_medians, start=1):
            if medians:
                offsets = np.linspace(-0.12, 0.12, len(medians))
                axis.scatter(index + offsets, medians, s=23, facecolor="white",
                             edgecolor=INK, linewidth=0.8, zorder=4)
        axis.set_xticks(np.arange(1, len(labels) + 1),
                        labels=[label.replace("_", "\n") for label in labels])
        axis.set_ylabel("Median patch log(1 + total triad band energy)")
        axis.set_title(title, fontsize=11, fontweight="semibold", color=INK)
        axis.grid(axis="y", color=GRID, linewidth=0.65)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Activity energy distributions",
                 x=0.055, y=0.985, ha="left", fontsize=18, fontweight="bold", color=INK)
    fig.text(
        0.055, 0.948,
        "Boxes show window-level IQR with 10th-90th percentile whiskers; open circles are "
        "per-dataset medians and expose dataset/placement variation.",
        ha="left", va="top", fontsize=9.5, color=MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.17, wspace=0.22)
    path = destination / "activity_energy_distributions.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def plot_within_dataset_heatmap(
    ref: GridRef,
    labels: Sequence[str],
    by_label: dict[str, list[SignatureSamples]],
    destination: Path,
) -> Path:
    import matplotlib.pyplot as plt

    records = {label: next((record for record in by_label[label] if record.ref.key == ref.key), None)
               for label in labels}
    supported = [label for label in labels if records[label] is not None]
    centers = records[supported[0]].centers_hz
    fig, axes = plt.subplots(1, 2, figsize=(14.0, max(4.6, 0.58 * len(supported) + 2.8)),
                             facecolor="white")
    for axis, modality, title in zip(
        axes, ("acc", "gyro"), ("Accelerometer", "Gyroscope")
    ):
        rows = []
        shown = []
        for label in supported:
            values = getattr(records[label], f"{modality}_share")
            if values is not None:
                rows.append(np.median(values, axis=0) * 100.0)
                shown.append(label)
        if not rows:
            axis.set_axis_off()
            axis.text(0.5, 0.5, "Modality not recorded", transform=axis.transAxes,
                      ha="center", va="center", color=MUTED)
            continue
        matrix = np.stack(rows)
        image = axis.imshow(matrix, aspect="auto", cmap="cividis", vmin=0,
                            vmax=float(np.percentile(matrix, 99)))
        axis.set_yticks(np.arange(len(shown)), labels=[value.replace("_", " ") for value in shown])
        tick_freqs = np.asarray([0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0])
        tick_indices = [int(np.argmin(np.abs(centers - value))) for value in tick_freqs]
        axis.set_xticks(tick_indices, labels=[f"{value:g}" for value in tick_freqs])
        axis.set_xlabel("Band center (Hz)")
        axis.set_title(title, fontsize=11, fontweight="semibold", color=INK)
        colorbar = fig.colorbar(image, ax=axis, orientation="horizontal", fraction=0.065, pad=0.17)
        colorbar.set_label("Median band share (%)", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"Within-dataset activity signatures: {_display_dataset(ref)}",
                 x=0.055, y=0.985, ha="left", fontsize=17, fontweight="bold", color=INK)
    fig.text(
        0.055, 0.945,
        "Rotation-invariant triad energy; each activity row is normalized independently. "
        "Near-static rows may primarily describe residual sensor noise.",
        ha="left", va="top", fontsize=9.5, color=MUTED,
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.84, bottom=0.22, wspace=0.33)
    path = destination / f"{_slug(ref.key)}_within_dataset_signatures.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def run(
    labels: Sequence[str],
    alignment: str,
    selectors: Sequence[str] | None,
    max_windows: int,
    seed: int,
    destination: Path,
    legacy_root: Path,
) -> list[Path]:
    tokenizer_class, source = _load_tokenizer_class(legacy_root)
    refs = discover_grids(alignment)
    requested = tuple(selectors) if selectors else DEFAULT_STREAMS
    by_label: dict[str, list[SignatureSamples]] = {}
    for label in labels:
        selected = matching_refs(refs, label, requested)
        if not selected:
            raise ValueError(f"No selected streams contain {label!r}")
        by_label[label] = [
            extract_signatures(ref, label, max_windows, seed, tokenizer_class)
            for ref in selected
        ]

    cross_dataset = output_subdir(destination, "cross_dataset")
    within_dataset = output_subdir(destination, "within_dataset")
    summaries = output_subdir(destination, "summaries")

    paths = [plot_cross_dataset_profile(label, by_label[label], cross_dataset) for label in labels]
    paths.append(plot_cross_activity_heatmap(labels, by_label, summaries))
    paths.append(plot_energy_distributions(labels, by_label, summaries))
    all_refs = []
    for ref in refs:
        if any(ref.key == record.ref.key for records in by_label.values() for record in records):
            all_refs.append(ref)
    paths.extend(
        plot_within_dataset_heatmap(ref, labels, by_label, within_dataset)
        for ref in all_refs
    )

    manifest = summaries / "activity_signatures.json"
    manifest.write_text(json.dumps({
        "labels": list(labels),
        "alignment": alignment,
        "seed": seed,
        "max_windows_per_dataset_activity": max_windows,
        "patch_seconds": PATCH_SECONDS,
        "tokenizer_source": str(source),
        "tokenizer_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "signature": "sum xyz band energy, divide by Gaussian filter area, normalize over bands",
        "rotation_invariant": True,
        "absolute_energy": "log1p of tokenizer band energy summed over xyz and bands",
        "records": {
            label: {
                record.ref.key: {
                    "n_windows": len(record.window_indices),
                    "window_indices": record.window_indices.tolist(),
                    "acc_percentiles": {
                        str(percentile): np.percentile(record.acc_share, percentile, axis=0).tolist()
                        for percentile in (10, 25, 50, 75, 90)
                    },
                    "gyro_percentiles": None if record.gyro_share is None else {
                        str(percentile): np.percentile(record.gyro_share, percentile, axis=0).tolist()
                        for percentile in (10, 25, 50, 75, 90)
                    },
                }
                for record in records
            }
            for label, records in by_label.items()
        },
    }, indent=2) + "\n")
    paths.append(manifest)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--alignment", default="harmonised",
                        choices=("harmonised", "non_harmonised"))
    parser.add_argument("--streams", nargs="*", default=None)
    parser.add_argument("--windows-per-stream", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    destination = output_subdir(output_dir(args.output_dir), "activity_signatures")
    for path in run(args.labels, args.alignment, args.streams, args.windows_per_stream,
                    args.seed, destination, args.legacy_root):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
