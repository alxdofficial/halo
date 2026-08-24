"""Plot HALO's matched recording-level retrieval and reranking decomposition."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


KS = (1, 2, 4, 8, 16)
REGIMES = (("ordinary", "Ordinary activities"),
           ("specialized_novel", "Specialized novel activities"))
SERIES = (
    ("Support raw 1-NN", "support_raw_1nn", "#777777", ":"),
    ("Support reranked 1-NN", "support_reranked_1nn", "#111111", "-"),
    ("Corpus raw 1-NN", "corpus_raw_1nn", "#68a0cf", ":"),
    ("Corpus reranked 1-NN", "corpus_reranked_1nn", "#2878b5", "-"),
    ("Full-memory raw 1-NN", "full_raw_1nn", "#d88986", ":"),
    ("Full-memory reranked 1-NN", "full_reranked_1nn", "#c43c39", "-"),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decomposition", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    values = _decomposition(args.decomposition)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True)
    for ax, (regime, title) in zip(axes, REGIMES, strict=True):
        for label, method, color, linestyle in SERIES:
            curve = [values[method, regime, k] for k in KS]
            emphasis = method in {"support_reranked_1nn", "full_reranked_1nn"}
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
    fig.suptitle("HALO recording-level reranking decomposition", fontsize=13,
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
