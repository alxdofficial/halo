"""Plot four-sample gravity-aligned accel/gyro overlays for one activity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from data.scripts.curate.deployment_policy import STREAM_SPECS
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
from data.scripts.eda.orientation import (
    centered_rms_normalize,
    gravity_alignment,
    rotate_vectors,
)
from data.scripts.eda.plot_sensor_trajectories import (
    BLUE,
    DEFAULT_STREAMS,
    DISPLAY_NAMES,
    GOLD,
    GRID,
    INK,
    KNOWN_GYRO_UNIT_ISSUES,
    MUTED,
)


GRAVITY_STATE = {(spec.dataset, spec.stream_id): spec.gravity_state for spec in STREAM_SPECS}


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


def _display_ref(ref: GridRef) -> str:
    dataset = DISPLAY_NAMES.get(ref.dataset, ref.dataset.replace("_", " ").title())
    if ref.dataset == "mhealth":
        dataset += "*"
    return f"{dataset}\n{ref.stream.replace('_', ' ')}"


def aligned_samples(
    refs: Sequence[GridRef], label: str, count: int, seed: int
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Load, gravity-align, and normalize selected windows."""
    result: dict[str, list[dict]] = {}
    manifest: list[dict] = []
    for ref in refs:
        state = GRAVITY_STATE.get((ref.dataset, ref.stream))
        if state != "present":
            raise ValueError(f"{ref.key}: gravity alignment requires gravity_state='present', got {state!r}")
        if ref.dataset in KNOWN_GYRO_UNIT_ISSUES and triad_indices(ref, "gyro"):
            raise ValueError(KNOWN_GYRO_UNIT_ISSUES[ref.dataset])

        acc_indices = triad_indices(ref, "acc")
        gyro_indices = triad_indices(ref, "gyro")
        if acc_indices is None:
            raise ValueError(f"{ref.key}: no valid accelerometer triad")

        data = ref.load_data()
        indices = sample_indices(ref, label, count, seed)
        samples = []
        for ordinal, index in enumerate(indices, start=1):
            window = np.asarray(data[index], dtype=np.float32)
            estimate = gravity_alignment(window[:, acc_indices])
            acc_aligned = rotate_vectors(window[:, acc_indices], estimate.rotation)
            acc_shape, acc_rms = centered_rms_normalize(acc_aligned)
            gyro_shape = None
            gyro_rms = None
            if gyro_indices is not None:
                gyro_aligned = rotate_vectors(window[:, gyro_indices], estimate.rotation)
                gyro_shape, gyro_rms = centered_rms_normalize(gyro_aligned)

            samples.append({
                "acc": acc_shape,
                "gyro": gyro_shape,
                "subject": ref.subjects[int(index)],
                "window_index": int(index),
                "acc_rms": acc_rms,
                "gyro_rms": gyro_rms,
                "alignment": estimate,
            })
            manifest.append({
                "dataset": ref.dataset,
                "stream": ref.stream,
                "label": label,
                "sample": ordinal,
                "window_index": int(index),
                "subject": ref.subjects[int(index)],
                "rate_hz": ref.rate_hz,
                "gravity_vector_g": estimate.gravity_vector.tolist(),
                "gravity_norm_g": round(estimate.gravity_norm, 6),
                "rotation_matrix": estimate.rotation.tolist(),
                "alignment_error": estimate.alignment_error,
                "acc_dynamic_rms_g": round(acc_rms, 6),
                "gyro_dynamic_rms_rad_s": None if gyro_rms is None else round(gyro_rms, 6),
                "grid_dir": str(ref.grid_dir.relative_to(REPO)),
            })
        result[ref.key] = samples
    return result, manifest


def _plot_shape(ax, values: np.ndarray, color: str) -> None:
    ax.plot(values[:, 0], values[:, 1], values[:, 2], color=color, linewidth=1.15, alpha=0.88)
    ax.scatter(*values[0], s=25, facecolor="white", edgecolor=color, linewidth=1.0,
               depthshade=False, zorder=5)


def plot_overlays(
    refs: Sequence[GridRef], samples: dict[str, list[dict]], label: str, destination: Path
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    count = len(next(iter(samples.values())))
    all_values = [sample[modality] for ref in refs for sample in samples[ref.key]
                  for modality in ("acc", "gyro") if sample[modality] is not None]
    limit = max(float(np.max(np.abs(values))) for values in all_values)
    limit = max(1.0, math.ceil(limit * 10.0) / 10.0)

    fig = plt.figure(figsize=(4.0 * count, 3.35 * len(refs) + 1.8), facecolor="white")
    grid = fig.add_gridspec(len(refs), count, wspace=0.02, hspace=0.18)
    for row, ref in enumerate(refs):
        for column, sample in enumerate(samples[ref.key]):
            ax = fig.add_subplot(grid[row, column], projection="3d")
            _plot_shape(ax, sample["acc"], BLUE)
            if sample["gyro"] is not None:
                _plot_shape(ax, sample["gyro"], GOLD)
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
            ax.set_box_aspect((1, 1, 1))
            ax.set_xlabel("x", labelpad=-7, fontsize=7)
            ax.set_ylabel("y", labelpad=-7, fontsize=7)
            ax.set_zlabel("z", labelpad=-7, fontsize=7)
            ax.tick_params(labelsize=6, pad=-2)
            ax.grid(True, color=GRID, linewidth=0.5)
            ax.view_init(elev=21, azim=-58)
            gyro_text = "gyro absent" if sample["gyro_rms"] is None else f"gyro {sample['gyro_rms']:.2f} rad/s"
            ax.text2D(0.5, -0.03,
                      f"subject {sample['subject']} | acc {sample['acc_rms']:.2f} g | {gyro_text}",
                      transform=ax.transAxes, ha="center", va="top", fontsize=7, color=MUTED)
            if row == 0:
                ax.set_title(f"Sample {column + 1}",
                             fontsize=9, fontweight="semibold", color=INK, pad=5)
            if column == 0:
                ax.text2D(-0.27, 0.5, _display_ref(ref), transform=ax.transAxes,
                          ha="right", va="center", fontsize=9, color=INK,
                          fontweight="semibold")

    handles = [
        Line2D([0], [0], color=BLUE, linewidth=1.8, label="Accelerometer"),
        Line2D([0], [0], color=GOLD, linewidth=1.8, label="Gyroscope"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.975),
               frameon=False, ncol=2, fontsize=9)
    fig.suptitle(f"Gravity-aligned sensor shapes: {label.replace('_', ' ')}",
                 x=0.055, y=0.988, ha="left", fontsize=18, fontweight="bold", color=INK)
    fig.text(
        0.055, 0.963,
        "One rigid rotation maps mean acceleration to +z and is shared by both sensors. "
        "Each modality is then centered and RMS-normalized; yaw remains unresolved.",
        ha="left", va="top", fontsize=10, color=MUTED,
    )
    if any(ref.dataset == "mhealth" for ref in refs):
        fig.text(0.055, 0.012, "* MHEALTH gyro is a known low-reliability, sample-and-hold stream.",
                 ha="left", va="bottom", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.18, right=0.99, top=0.92, bottom=0.045)
    path = destination / f"{_slug(label)}_gravity_aligned_overlays.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def run(
    label: str,
    alignment_name: str,
    selectors: Sequence[str] | None,
    samples_per_stream: int,
    seed: int,
    max_streams: int,
    destination: Path,
) -> tuple[Path, Path]:
    refs = discover_grids(alignment_name)
    selected = matching_refs(refs, label, tuple(selectors) if selectors else DEFAULT_STREAMS)
    selected = selected[:max_streams]
    if not selected:
        raise ValueError(f"No selected grids contain {label!r}")
    samples, records = aligned_samples(selected, label, samples_per_stream, seed)
    figure = plot_overlays(
        selected, samples, label, output_subdir(destination, "overlays")
    )
    manifest = (
        output_subdir(destination, "samples")
        / f"{_slug(label)}_gravity_aligned_samples.json"
    )
    manifest.write_text(json.dumps({
        "label": label,
        "seed": seed,
        "alignment": alignment_name,
        "normalization": "gravity-to-positive-z, temporal centering, per-modality vector RMS",
        "yaw_normalized": False,
        "samples": records,
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
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    destination = output_subdir(output_dir(args.output_dir), "gravity_alignment")
    for label in args.label:
        for path in run(label, args.alignment, args.streams, args.samples, args.seed,
                        args.max_streams, destination):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
