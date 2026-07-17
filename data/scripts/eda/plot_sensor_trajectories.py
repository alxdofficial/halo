"""Plot deterministic cross-dataset accelerometer and gyroscope examples."""

from __future__ import annotations

import argparse
import json
import math
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


INK = "#20262e"
BLUE = "#2f6b9a"
GOLD = "#c38a24"
GRID = "#d9dee5"
MUTED = "#69727d"
SAMPLE_COLORS = ("#2f6b9a", "#c38a24", "#b04a67")
SAMPLE_STYLES = ("-", "--", ":")

KNOWN_GYRO_UNIT_ISSUES = {
    "usc_had": "USC-HAD gyro is currently stored in degrees/s instead of rad/s",
}

DEFAULT_STREAMS = (
    "motionsense/phone_front_pocket",
    "hhar/phone_waist",
    "shoaib/phone_right_pocket",
    "pamap2/watch_wrist",
    "mhealth/watch_wrist",
    "capture24/watch_wrist",
)

DISPLAY_NAMES = {
    "capture24": "Capture-24",
    "hhar": "HHAR",
    "mhealth": "MHEALTH",
    "motionsense": "MotionSense",
    "pamap2": "PAMAP2",
    "shoaib": "Shoaib",
    "uci_har": "UCI HAR",
    "unimib_shar": "UniMiB SHAR",
    "wisdm": "WISDM",
}


def _display_ref(ref: GridRef) -> str:
    dataset = DISPLAY_NAMES.get(ref.dataset, ref.dataset.replace("_", " ").title())
    if ref.dataset == "mhealth":
        dataset += "*"
    placement = ref.stream.replace("_", " ")
    return f"{dataset}\n{placement}"


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


def _selected_windows(
    refs: Sequence[GridRef], label: str, samples: int, seed: int
) -> tuple[dict[str, list[np.ndarray]], list[dict]]:
    windows: dict[str, list[np.ndarray]] = {}
    manifest: list[dict] = []
    for ref in refs:
        data = ref.load_data()
        indices = sample_indices(ref, label, samples, seed)
        windows[ref.key] = [np.asarray(data[index], dtype=np.float32) for index in indices]
        for ordinal, index in enumerate(indices, start=1):
            window = windows[ref.key][ordinal - 1]
            summaries = {}
            for modality in ("acc", "gyro"):
                triad = triad_indices(ref, modality)
                if triad is None:
                    continue
                magnitude = np.linalg.norm(window[:, triad], axis=1)
                summaries[f"{modality}_magnitude_mean"] = round(float(magnitude.mean()), 6)
                summaries[f"{modality}_magnitude_std"] = round(float(magnitude.std()), 6)
            manifest.append({
                "dataset": ref.dataset,
                "stream": ref.stream,
                "alignment": ref.alignment,
                "label": label,
                "window_index": int(index),
                "sample": ordinal,
                "subject": ref.subjects[int(index)],
                "rate_hz": ref.rate_hz,
                "window_samples": ref.shape[1],
                "duration_seconds": ref.shape[1] / ref.rate_hz,
                "channels": list(ref.channels),
                "channel_mask": list(ref.mask),
                "grid_dir": str(ref.grid_dir.relative_to(REPO)),
                **summaries,
            })
    return windows, manifest


def _modality_values(
    refs: Sequence[GridRef], windows: dict[str, list[np.ndarray]], modality: str
) -> list[np.ndarray]:
    values = []
    for ref in refs:
        indices = triad_indices(ref, modality)
        if indices is None:
            continue
        values.extend(window[:, indices] for window in windows[ref.key])
    return values


def _axis_limit(values: Sequence[np.ndarray]) -> float:
    if not values:
        return 1.0
    maximum = max(float(np.max(np.abs(value))) for value in values)
    return max(0.1, math.ceil(maximum * 10.0) / 10.0)


def _plot_trajectory(
    ax, xyz: np.ndarray, color: str, limit: float, unit: str
) -> None:
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=1.15, alpha=0.9)
    ax.scatter(*xyz[0], s=31, facecolor="white", edgecolor=INK, linewidth=0.9,
               depthshade=False, zorder=5)
    ax.scatter(*xyz[-1], s=34, marker="^", color=color, edgecolor=INK,
               linewidth=0.6, depthshade=False, zorder=5)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(f"x ({unit})", labelpad=-7, fontsize=7)
    ax.set_ylabel(f"y ({unit})", labelpad=-7, fontsize=7)
    ax.set_zlabel(f"z ({unit})", labelpad=-7, fontsize=7)
    ax.tick_params(labelsize=6, pad=-2)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.view_init(elev=21, azim=-58)


def plot_trajectories(
    refs: Sequence[GridRef],
    windows: dict[str, list[np.ndarray]],
    label: str,
    destination: Path,
) -> Path:
    import matplotlib.pyplot as plt

    samples = len(next(iter(windows.values())))
    columns = samples * 2
    acc_values = _modality_values(refs, windows, "acc")
    gyro_values = _modality_values(refs, windows, "gyro")
    limits = {"acc": _axis_limit(acc_values), "gyro": _axis_limit(gyro_values)}

    fig = plt.figure(figsize=(3.55 * columns, 3.25 * len(refs) + 1.65), facecolor="white")
    grid = fig.add_gridspec(len(refs), columns, wspace=0.05, hspace=0.22)
    for row, ref in enumerate(refs):
        for sample in range(samples):
            for modality_index, (modality, color, unit) in enumerate((
                ("acc", BLUE, "g"), ("gyro", GOLD, "rad/s")
            )):
                column = sample * 2 + modality_index
                ax = fig.add_subplot(grid[row, column], projection="3d")
                indices = triad_indices(ref, modality)
                if indices is None:
                    ax.set_axis_off()
                    ax.text2D(0.5, 0.5, "Not recorded", transform=ax.transAxes,
                              ha="center", va="center", color=MUTED, fontsize=10)
                else:
                    xyz = windows[ref.key][sample][:, indices]
                    _plot_trajectory(ax, xyz, color, limits[modality], unit)
                if row == 0:
                    name = "Accelerometer" if modality == "acc" else "Gyroscope"
                    ax.set_title(f"Sample {sample + 1} | {name}", fontsize=10,
                                 fontweight="semibold", color=INK, pad=5)
                if column == 0:
                    ax.text2D(-0.31, 0.5, _display_ref(ref), transform=ax.transAxes,
                              ha="right", va="center", fontsize=9, color=INK,
                              fontweight="semibold")

    fig.suptitle(f"Sensor-state trajectories: {label.replace('_', ' ')}",
                 x=0.055, y=0.987, ha="left", fontsize=18, fontweight="bold", color=INK)
    fig.text(
        0.055, 0.962,
        "Each curve is (x(t), y(t), z(t)) in measurement space, not physical position. "
        "Open circle = start; triangle = end; axes are shared within each modality.",
        ha="left", va="top", fontsize=10, color=MUTED,
    )
    if any(ref.dataset == "mhealth" for ref in refs):
        fig.text(0.055, 0.012, "* MHEALTH gyro is a known low-reliability, sample-and-hold stream.",
                 ha="left", va="bottom", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.18, right=0.985, top=0.92, bottom=0.035)
    path = destination / f"{_slug(label)}_sensor_trajectories.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def plot_magnitudes(
    refs: Sequence[GridRef],
    windows: dict[str, list[np.ndarray]],
    label: str,
    destination: Path,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    samples = len(next(iter(windows.values())))
    magnitudes: dict[tuple[str, str, int], np.ndarray] = {}
    modality_max = {"acc": 0.0, "gyro": 0.0}
    duration_max = 0.0
    for ref in refs:
        duration_max = max(duration_max, ref.shape[1] / ref.rate_hz)
        for modality in ("acc", "gyro"):
            indices = triad_indices(ref, modality)
            if indices is None:
                continue
            for sample, window in enumerate(windows[ref.key]):
                values = np.linalg.norm(window[:, indices], axis=1)
                magnitudes[(ref.key, modality, sample)] = values
                modality_max[modality] = max(modality_max[modality], float(values.max()))

    fig, axes = plt.subplots(
        len(refs), 2, figsize=(13.8, 2.15 * len(refs) + 2.0),
        sharex=True, squeeze=False, facecolor="white",
    )
    for row, ref in enumerate(refs):
        for column, (modality, title, unit) in enumerate((
            ("acc", "Acceleration magnitude", "g"),
            ("gyro", "Angular-rate magnitude", "rad/s"),
        )):
            ax = axes[row, column]
            if triad_indices(ref, modality) is None:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "Gyroscope not recorded", transform=ax.transAxes,
                        ha="center", va="center", color=MUTED, fontsize=10)
            else:
                for sample in range(samples):
                    values = magnitudes[(ref.key, modality, sample)]
                    time = np.arange(len(values)) / ref.rate_hz
                    ax.plot(time, values, color=SAMPLE_COLORS[sample],
                            linestyle=SAMPLE_STYLES[sample], linewidth=1.15)
                if modality == "acc":
                    ax.axhline(1.0, color=INK, linestyle=":", linewidth=0.8, alpha=0.65)
                ax.set_xlim(0.0, duration_max)
                ax.set_ylim(0.0, max(0.1, modality_max[modality] * 1.06))
                ax.set_ylabel(unit, fontsize=8)
                ax.grid(color=GRID, linewidth=0.65)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight="semibold", color=INK)
            if column == 0:
                ax.text(-0.13, 0.5, _display_ref(ref), transform=ax.transAxes,
                        ha="right", va="center", fontsize=9, color=INK,
                        fontweight="semibold")
            if row == len(refs) - 1 and ax.axison:
                ax.set_xlabel("Time (seconds)")

    handles = [
        Line2D([0], [0], color=SAMPLE_COLORS[index], linestyle=SAMPLE_STYLES[index],
               label=f"Sample {index + 1}")
        for index in range(samples)
    ]
    handles.append(Line2D([0], [0], color=INK, linestyle=":", linewidth=0.8,
                          label="1 g reference"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.975, 0.975),
               ncol=len(handles), frameon=False, fontsize=9)
    fig.suptitle(f"Rotation-invariant sensor magnitudes: {label.replace('_', ' ')}",
                 x=0.06, y=0.985, ha="left", fontsize=18, fontweight="bold", color=INK)
    fig.text(
        0.06, 0.955,
        "Shared scales make amplitude and cadence comparable; magnitude removes axis orientation "
        "but not placement, device, gravity-state, or calibration differences.",
        ha="left", va="top", fontsize=10, color=MUTED,
    )
    if any(ref.dataset == "mhealth" for ref in refs):
        fig.text(0.06, 0.014, "* MHEALTH gyro is a known low-reliability, sample-and-hold stream.",
                 ha="left", va="bottom", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.89, bottom=0.085, hspace=0.34, wspace=0.18)
    path = destination / f"{_slug(label)}_sensor_magnitudes.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def run(
    label: str,
    alignment: str,
    selectors: Sequence[str] | None,
    samples: int,
    seed: int,
    max_streams: int,
    destination: Path,
) -> tuple[Path, Path, Path]:
    refs = discover_grids(alignment)
    requested = tuple(selectors) if selectors else DEFAULT_STREAMS
    selected = matching_refs(refs, label, requested)
    if not selected and not selectors:
        selected = matching_refs(refs, label)
    selected = selected[:max_streams]
    if not selected:
        available = sorted({value for ref in refs for value in ref.labels})
        raise ValueError(f"No grids contain {label!r}. Available labels: {available}")
    unit_issues = [KNOWN_GYRO_UNIT_ISSUES[ref.dataset] for ref in selected
                   if ref.dataset in KNOWN_GYRO_UNIT_ISSUES and triad_indices(ref, "gyro")]
    if unit_issues:
        raise ValueError("Cannot make a unit-comparable figure: " + "; ".join(unit_issues))

    windows, manifest = _selected_windows(selected, label, samples, seed)
    trajectory_path = plot_trajectories(
        selected, windows, label, output_subdir(destination, "trajectories")
    )
    magnitude_path = plot_magnitudes(
        selected, windows, label, output_subdir(destination, "magnitudes")
    )
    manifest_path = output_subdir(destination, "samples") / f"{_slug(label)}_samples.json"
    manifest_path.write_text(json.dumps({
        "seed": seed,
        "label": label,
        "alignment": alignment,
        "samples_per_stream": samples,
        "streams": [ref.key for ref in selected],
        "samples": manifest,
    }, indent=2) + "\n")
    return trajectory_path, magnitude_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", nargs="+", required=True,
                        help="Canonical activity labels, e.g. walking sitting")
    parser.add_argument("--alignment", default="harmonised",
                        choices=("harmonised", "non_harmonised"))
    parser.add_argument("--streams", nargs="*", default=None,
                        help="Optional dataset or dataset/stream selectors")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-streams", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.samples < 1 or args.samples > len(SAMPLE_COLORS):
        parser.error(f"--samples must be between 1 and {len(SAMPLE_COLORS)}")
    if args.max_streams < 1:
        parser.error("--max-streams must be positive")

    destination = output_subdir(output_dir(args.output_dir), "signals")
    for label in args.label:
        for path in run(
            label, args.alignment, args.streams, args.samples, args.seed,
            args.max_streams, destination,
        ):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
