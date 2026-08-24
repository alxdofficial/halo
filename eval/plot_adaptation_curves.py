"""Plot current matched adaptation k-curves from an assembled cells table.

The two outputs answer different questions and must remain separate:

1. ``knn_representation_curves`` applies the same 1-NN readout to every representation.
2. ``primary_adaptation_curves`` shows HALO retrieve-mix-vote beside the same no-fitting 1-NN rule
   on every frozen representation.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_NAMES = {
    "halo_compact": "HALO",
    "harnet": "HARNet",
    "crosshar": "CrossHAR",
    "unimts": "UniMTS",
    "limubert": "LIMU-BERT",
    "normwear": "NormWear",
    "imagebind": "ImageBind",
}
KS = (1, 2, 4, 8, 16)
REGIMES = (("ordinary", "Ordinary activities"),
           ("specialized_novel", "Specialized novel activities"))
COLORS = {
    "HALO / retrieve-mix-vote": "#c43c39",
    "HALO / 1-NN": "#111111",
    "HALO": "#111111",
    "HARNet": "#2878b5",
    "CrossHAR": "#e07b20",
    "UniMTS": "#5a9f4b",
    "LIMU-BERT": "#7655a3",
    "ImageBind": "#8c6d5c",
    "NormWear": "#6f7b83",
}


def _load(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def _dataset_macro(
    rows: list[dict], *, model: str, method: str, regime: str, k: int,
) -> float:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if (row["model"] == model and row["method"] == method
                and row["regime"] == regime and row["label_mode"] == "coherent"
                and int(row["k"]) == k):
            by_dataset[row["dataset"]].append(float(row["f1_macro"]))
    if not by_dataset:
        return float("nan")
    return float(np.mean([np.mean(values) for values in by_dataset.values()]))


def _curve(rows: list[dict], model: str, method: str, regime: str) -> list[float]:
    return [_dataset_macro(rows, model=model, method=method, regime=regime, k=k) for k in KS]


def _style_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="semibold", pad=8)
    ax.set_xticks(range(len(KS)), [str(k) for k in KS])
    ax.set_xlabel("Enrolled executions per candidate (k)")
    ax.set_ylim(15, 70)
    ax.set_yticks(np.arange(20, 71, 10))
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_knn(rows: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=True)
    for ax, (regime, title) in zip(axes, REGIMES, strict=True):
        for model, name in MODEL_NAMES.items():
            values = _curve(rows, model, "nearest", regime)
            ax.plot(
                range(len(KS)), values, marker="o", markersize=4,
                linewidth=2.8 if model == "halo_compact" else 1.5,
                color=COLORS[name], label=name, zorder=5 if model == "halo_compact" else 2,
            )
        _style_axes(ax, title)
    axes[0].set_ylabel("Macro F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=8.5)
    fig.suptitle("Matched 1-NN comparison of frozen representations", fontsize=13,
                 fontweight="semibold", y=1.02)
    fig.subplots_adjust(bottom=0.23, wspace=0.09)
    _save(fig, out_dir, "knn_representation_curves")


def plot_primary(rows: list[dict], out_dir: Path) -> None:
    series = [
        ("HALO / retrieve-mix-vote", "halo_compact", "evidence_engine"),
        ("HALO / 1-NN", "halo_compact", "nearest"),
        *((MODEL_NAMES[model], model, "nearest")
          for model in MODEL_NAMES if model != "halo_compact"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=True)
    for ax, (regime, title) in zip(axes, REGIMES, strict=True):
        for name, model, method in series:
            halo = model == "halo_compact"
            ax.plot(
                range(len(KS)), _curve(rows, model, method, regime),
                marker="o", markersize=4, linewidth=2.8 if halo else 1.4,
                linestyle="--" if method == "evidence_engine" else "-",
                color=COLORS[name], label=name, zorder=5 if halo else 2,
            )
        _style_axes(ax, title)
    axes[0].set_ylabel("Macro F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.08), fontsize=8.5)
    fig.suptitle("Label-efficient adaptation: HALO retrieve-mix-vote and matched 1-NN", fontsize=13,
                 fontweight="semibold", y=1.02)
    fig.subplots_adjust(bottom=0.27, wspace=0.09)
    _save(fig, out_dir, "primary_adaptation_curves")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = _load(args.cells)
    plot_knn(rows, args.out_dir)
    plot_primary(rows, args.out_dir)
    print(f"wrote k-curve figures to {args.out_dir}")


if __name__ == "__main__":
    main()
