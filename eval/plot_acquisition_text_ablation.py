"""Plot the matched acquisition-description ablation from raw evaluation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from eval.assemble_adaptation import dataset_macro, load_rows
from eval.enrollment_protocol import load_manifest


ARMS = (
    ("Full descriptions", "full", "#111111", "-"),
    ("Neutralized at inference", "inference_neutral", "#d17a22", "--"),
    ("Trained with neutral descriptions", "trained_neutral", "#2878b5", "-"),
)
REGIMES = (
    ("ordinary", "Ordinary activities"),
    ("specialized_novel", "Specialized novel activities"),
)


def _value(rows: list[dict], *, method: str, regime: str, k: int) -> float:
    matches = [
        row["f1_macro"]
        for row in rows
        if row["method"] == method
        and row["regime"] == regime
        and row["label_mode"] == "coherent"
        and int(row["k"]) == k
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one aggregate for method={method}, regime={regime}, k={k}; "
            f"found {len(matches)}"
        )
    return float(matches[0])


def _load(path: Path, manifest: dict) -> list[dict]:
    rows, _ = load_rows([path], manifest)
    return dataset_macro(rows)


def _style(ax, title: str, ks: tuple[int, ...]) -> None:
    ax.set_title(title, fontsize=10.5, fontweight="semibold")
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--inference-neutral", type=Path, required=True)
    parser.add_argument("--trained-neutral", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    paths = {
        "full": args.full,
        "inference_neutral": args.inference_neutral,
        "trained_neutral": args.trained_neutral,
    }
    rows = {name: _load(path, manifest) for name, path in paths.items()}

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharey="row")
    for col, (regime, regime_title) in enumerate(REGIMES):
        engine_ks = (0, 1, 2, 4, 8, 16)
        for label, arm, color, linestyle in ARMS:
            values = [
                _value(
                    rows[arm], method="zero_shot" if k == 0 else "evidence_engine",
                    regime=regime, k=k,
                )
                for k in engine_ks
            ]
            axes[0, col].plot(
                range(len(engine_ks)), values, marker="o", linewidth=2,
                color=color, linestyle=linestyle, label=label,
            )
        _style(axes[0, col], f"Native engine: {regime_title}", engine_ks)

        nearest_ks = (1, 2, 4, 8, 16)
        for label, arm, color, linestyle in ARMS:
            values = [
                _value(rows[arm], method="nearest", regime=regime, k=k)
                for k in nearest_ks
            ]
            axes[1, col].plot(
                range(len(nearest_ks)), values, marker="o", linewidth=2,
                color=color, linestyle=linestyle, label=label,
            )
        _style(axes[1, col], f"1-NN readout: {regime_title}", nearest_ks)
        axes[1, col].set_xlabel("Enrolled executions per candidate (k)")

    axes[0, 0].set_ylabel("Macro F1")
    axes[1, 0].set_ylabel("Macro F1")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Acquisition-description conditioning ablation", fontsize=13,
                 fontweight="semibold")
    fig.subplots_adjust(bottom=0.13, top=0.91, hspace=0.31, wspace=0.12)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            args.out_dir / f"acquisition_text_ablation.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(f"wrote acquisition-text ablation figure to {args.out_dir}")


if __name__ == "__main__":
    main()
