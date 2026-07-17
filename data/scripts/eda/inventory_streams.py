"""Create a table and visual inventory of generated deployment-stream grids."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from data.scripts.eda.grid_io import discover_grids, output_dir, output_subdir, triad_indices


INK = "#20262e"
BLUE = "#2f6b9a"
GOLD = "#c38a24"
GRID = "#d9dee5"


def build_inventory(alignment: str, destination: Path) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    refs = discover_grids(alignment)
    if not refs:
        raise FileNotFoundError(f"No generated {alignment!r} grids were found")

    csv_path = destination / f"stream_inventory_{alignment}.csv"
    fields = [
        "dataset", "stream", "alignment", "rate_hz", "window_samples",
        "windows", "duration_hours", "subjects", "labels", "accelerometer", "gyroscope",
    ]
    rows = []
    for ref in refs:
        rows.append({
            "dataset": ref.dataset,
            "stream": ref.stream,
            "alignment": ref.alignment,
            "rate_hz": ref.rate_hz,
            "window_samples": ref.shape[1],
            "windows": ref.n_windows,
            "duration_hours": round(ref.duration_seconds / 3600.0, 3),
            "subjects": len(set(ref.subjects)),
            "labels": len(Counter(ref.labels)),
            "accelerometer": triad_indices(ref, "acc") is not None,
            "gyroscope": triad_indices(ref, "gyro") is not None,
        })
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    labels = [ref.key for ref in refs]
    windows = np.asarray([ref.n_windows for ref in refs])
    modalities = np.asarray([
        [triad_indices(ref, "acc") is not None, triad_indices(ref, "gyro") is not None]
        for ref in refs
    ], dtype=int)

    height = max(7.0, 0.44 * len(refs) + 2.2)
    fig = plt.figure(figsize=(13.5, height), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=(4.7, 1.2), wspace=0.12)
    ax_count = fig.add_subplot(grid[0, 0])
    ax_modality = fig.add_subplot(grid[0, 1], sharey=ax_count)
    y = np.arange(len(refs))

    ax_count.barh(y, windows, color=BLUE, edgecolor=INK, linewidth=0.45)
    ax_count.set_xscale("log")
    ax_count.set_yticks(y, labels=labels)
    ax_count.invert_yaxis()
    ax_count.set_xlabel("Generated windows (log scale)")
    ax_count.grid(axis="x", color=GRID, linewidth=0.7)
    ax_count.set_axisbelow(True)
    for index, value in enumerate(windows):
        ax_count.text(value * 1.07, index, f"{value:,}", va="center", fontsize=8, color=INK)

    for row in range(len(refs)):
        for column, color in enumerate((BLUE, GOLD)):
            available = bool(modalities[row, column])
            ax_modality.scatter(
                column, row, s=82, marker="s", color=color if available else "white",
                edgecolor=INK if available else "#a8afb8", linewidth=0.8,
            )
            if not available:
                ax_modality.plot(
                    [column - 0.11, column + 0.11], [row - 0.11, row + 0.11],
                    color="#a8afb8", linewidth=1.0,
                )
    ax_modality.set_xlim(-0.6, 1.6)
    ax_modality.set_xticks([0, 1], labels=["Accel", "Gyro"])
    ax_modality.tick_params(axis="y", left=False, labelleft=False)
    ax_modality.set_title("Recorded triads", fontsize=10)
    for axis in (ax_count, ax_modality):
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Deployable sensor-grid inventory", x=0.08, y=0.985, ha="left",
                 fontsize=17, fontweight="bold", color=INK)
    fig.text(
        0.08, 0.955,
        f"{alignment.replace('_', ' ')} grids | filled square = recorded xyz triad; open square = absent/zero-padded",
        ha="left", va="top", fontsize=10, color="#59616b",
    )
    fig.subplots_adjust(left=0.28, right=0.97, top=0.91, bottom=0.08)
    png_path = destination / f"stream_inventory_{alignment}.png"
    fig.savefig(png_path, dpi=180, facecolor="white")
    plt.close(fig)
    return csv_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", default="harmonised",
                        choices=("harmonised", "non_harmonised"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    destination = output_subdir(output_dir(args.output_dir), "inventory")
    csv_path, png_path = build_inventory(args.alignment, destination)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
