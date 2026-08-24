"""Plot HALO's matched retrieve-mix-vote decomposition."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


KS = (1, 2, 4, 8, 16)
REGIMES = (("ordinary", "Ordinary activities"),
           ("specialized_novel", "Specialized novel activities"))
SERIES = (
    ("Pooled execution 1-NN", "pooled_execution_1nn", "#777777", ":"),
    ("Patch 1-NN", "support_patch_1nn", "#111111", "-"),
    ("Support soft vote", "support_soft_vote", "#2878b5", "-"),
    ("Support-only mixer", "support_mixer", "#c43c39", "--"),
    ("Semantic top-64 vote", "full_semantic_topk_vote", "#4f9147", "-"),
    ("Semantic full-bank vote", "full_semantic_bank_vote", "#d17a22", "-"),
    ("Full engine", "full_engine", "#c43c39", "-"),
)


def _decomposition(path: Path) -> dict[tuple[str, str, int], float]:
    payload = json.loads(path.read_text())
    per_dataset: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for result in payload["results"].values():
        for method in payload["methods"]:
            per_dataset[
                method, result["regime"], int(result["support_count"]), result["dataset"]
            ].append(float(result[method]["f1_macro"]))
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for (method, regime, k, _), values in per_dataset.items():
        grouped[method, regime, k].append(float(np.mean(values)))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _pooled_reference(path: Path) -> dict[tuple[str, str, int], float]:
    per_dataset: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for row in csv.DictReader(path.open()):
        if not (
            row["model"] == "halo_compact"
            and row["method"] == "nearest"
            and row["label_mode"] == "coherent"
        ):
            continue
        per_dataset[
            "pooled_execution_1nn", row["regime"], int(row["k"]), row["dataset"]
        ].append(float(row["f1_macro"]))
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for (method, regime, k, _), values in per_dataset.items():
        grouped[method, regime, k].append(float(np.mean(values)))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decomposition", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    values = {**_decomposition(args.decomposition), **_pooled_reference(args.cells)}

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True)
    for ax, (regime, title) in zip(axes, REGIMES, strict=True):
        for label, method, color, linestyle in SERIES:
            curve = [values[method, regime, k] for k in KS]
            emphasis = method in {"support_patch_1nn", "full_engine"}
            ax.plot(
                range(len(KS)), curve, marker="o", markersize=4,
                linewidth=2.7 if emphasis else 1.7, linestyle=linestyle,
                color=color, label=label, zorder=5 if emphasis else 2,
            )
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.set_xticks(range(len(KS)), [str(k) for k in KS])
        ax.set_xlabel("Enrolled executions per candidate (k)")
        ax.grid(axis="y", color="#d8d8d8", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Macro F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.08), fontsize=8.5)
    fig.suptitle("HALO retrieve-mix-vote decomposition", fontsize=13,
                 fontweight="semibold")
    fig.subplots_adjust(bottom=0.27, top=0.88, wspace=0.10)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            args.out_dir / f"halo_engine_decomposition.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)
    print(f"wrote HALO engine decomposition figure to {args.out_dir}")


if __name__ == "__main__":
    main()
