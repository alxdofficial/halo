"""Emit current headline and diagnostic adaptation tables from assembled cells.

    python -m eval.build_results_tables \
      --cells eval/adaptation_tables/<run>/cells.csv \
      --out docs/results/TABLES.md

1. Zero-shot: HALO against every baseline, no labelled examples.
2. Label efficiency: the same no-fitting 1-NN readout for every representation, with HALO's native
   evidence engine shown separately.
3. Additional matched readouts: support-only prototype and fitted linear head for every
   representation, plus ridge as a diagnostic.

The input must come from :mod:`eval.assemble_adaptation`, which validates the manifest, source and
checkpoint fingerprints before writing it. Aggregation is the mean over datasets within a regime,
after averaging seeds within a dataset, so a dataset with more seeds does not outvote one with fewer.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

MODEL_NAMES = {
    "halo_compact": "HALO (ours, compact engine)",
    "harnet": "HARNet", "crosshar": "CrossHAR", "unimts": "UniMTS",
    "limubert": "LIMU-BERT", "normwear": "NormWear", "imagebind": "ImageBind",
}
REGIMES = ("ordinary", "specialized_novel")


def _emit(header: list[str], rows: list[tuple[str, list[float]]], higher_better=True) -> list[str]:
    """Markdown table with the best value in each column bolded, matching the house format."""
    best = []
    for column in range(len(header) - 1):
        values = [r[1][column] for r in rows if r[1][column] == r[1][column]]
        best.append(max(values) if values and higher_better else (min(values) if values else None))
    out = ["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)]
    for name, values in rows:
        cells = []
        for column, value in enumerate(values):
            text = "n/a" if value != value else f"{value:.2f}"
            if value == value and best[column] is not None and abs(value - best[column]) < 1e-9:
                text = f"**{text}**"
            cells.append(text)
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return out


def _dataset_macro(rows: list[dict]) -> tuple[float, int]:
    """Mean over datasets of the per-dataset seed mean."""
    per_dataset = defaultdict(list)
    for row in rows:
        per_dataset[row["dataset"]].append(float(row["f1_macro"]))
    if not per_dataset:
        return float("nan"), 0
    return float(np.mean([np.mean(v) for v in per_dataset.values()])), len(per_dataset)


def _cells(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def _validate_current_cells(cells: list[dict]) -> None:
    present = {row["model"] for row in cells}
    missing = set(MODEL_NAMES) - present
    if missing:
        raise ValueError(
            "current report requires the complete model roster; missing: "
            + ", ".join(sorted(missing))
        )
    if not any(
        row["model"] == "halo_compact" and row["method"] == "evidence_engine"
        and int(row["k"]) > 0
        for row in cells
    ):
        raise ValueError(
            "HALO evidence-engine enrollment rows are missing; pooled-feature controls cannot be "
            "used as the current HALO headline"
        )


def table_zero_shot(cells: list[dict]) -> str:
    out = ["## 1. Zero-shot", "",
           "No labelled examples. Macro F1, mean over datasets.", ""]
    scored = []
    for model in MODEL_NAMES:
        per_regime = [
            _dataset_macro([c for c in cells if c["model"] == model
                            and c["method"] == "zero_shot" and c["regime"] == regime
                            and c["label_mode"] == "coherent"])[0]
            for regime in REGIMES
        ]
        scored.append((MODEL_NAMES[model], per_regime))
    scored.sort(key=lambda r: -np.nanmean(r[1]))
    out += _emit(["model"] + [r.replace("_", " ") for r in REGIMES], scored)
    counts = {r: _dataset_macro([c for c in cells if c["method"] == "zero_shot"
                                 and c["regime"] == r
                                 and c["label_mode"] == "coherent"])[1]
              for r in REGIMES}
    out += ["", f"Datasets per regime: " + ", ".join(f"{r} {n}" for r, n in counts.items()), ""]
    return "\n".join(out)


def table_label_efficiency(cells: list[dict]) -> str:
    ks = sorted({int(c["k"]) for c in cells if int(c["k"]) > 0})
    out = ["## 2. Label efficiency", "",
           "`k` is the number of independent enrolled executions per candidate. HALO is shown "
           "with both its native evidence engine and one-nearest-neighbor over the same learned "
           "representation. Every external model uses the same one-nearest-neighbor rule, which "
           "requires no fitting and sees only the enrolled support executions. Macro F1, mean "
           "over datasets.", ""]
    for regime in REGIMES:
        scored = []
        model_methods = [
            ("HALO (ours, native engine)", "halo_compact", "evidence_engine"),
            ("HALO (ours, 1-NN)", "halo_compact", "nearest"),
            *((f"{MODEL_NAMES[model]} / 1-NN", model, "nearest")
              for model in MODEL_NAMES if model != "halo_compact"),
        ]
        for display_name, model, method in model_methods:
            row = [
                _dataset_macro([c for c in cells if c["model"] == model
                                and c["method"] == method and c["regime"] == regime
                                and c["label_mode"] == "coherent"
                                and int(c["k"]) == k])[0]
                for k in ks
            ]
            scored.append((display_name, row))
        scored.sort(key=lambda r: -np.nanmean(r[1]))
        out += [f"### {regime.replace('_', ' ')}", ""]
        out += _emit(["model"] + [f"k={k}" for k in ks], scored)
        out.append("")
    return "\n".join(out)


def table_representation_controls(cells: list[dict]) -> str:
    methods = ("prototype", "linear_head", "ridge")
    ks = sorted({int(c["k"]) for c in cells if int(c["k"]) > 0})
    out = [
        "## 3. Additional matched readouts", "",
        "These comparisons apply the same readout to each model's exposed latent representation. "
        "Each enrolled execution contributes one equally weighted, normalized pooled vector. "
        "Prototype forms class centroids from enrolled supports only and never accesses query "
        "examples. `linear_head` fits only a linear classifier on those supports. Ridge is retained "
        "as an additional diagnostic.", "",
    ]
    for regime in REGIMES:
        for method in methods:
            scored = []
            for model in MODEL_NAMES:
                values = [
                    _dataset_macro([
                        c for c in cells
                        if c["model"] == model and c["method"] == method
                        and c["regime"] == regime and c["label_mode"] == "coherent"
                        and int(c["k"]) == k
                    ])[0]
                    for k in ks
                ]
                scored.append((MODEL_NAMES[model], values))
            if not any(any(value == value for value in values) for _, values in scored):
                continue
            scored.sort(key=lambda row: -np.nanmean(row[1]))
            out += [f"### {regime.replace('_', ' ')}: {method}", ""]
            out += _emit(["model"] + [f"k={k}" for k in ks], scored)
            out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cells = _cells(args.cells)
    _validate_current_cells(cells)
    text = "\n".join([
        "# Results", "",
        table_zero_shot(cells),
        table_label_efficiency(cells),
        table_representation_controls(cells),
    ])
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
