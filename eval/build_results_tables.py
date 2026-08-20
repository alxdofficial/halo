"""Emit the three headline results tables.

    python -m eval.build_results_tables --out docs/results/TABLES.md

1. Zero-shot: HALO against every baseline, no labelled examples.
2. Label efficiency: the same models at k = 1..16 novel labelled examples per class.
3. Adaptation mechanism: HALO's native retrieval/enrollment against prototype and ridge fitted on
   the same frozen features, plus the mechanism ablations.

Tables 1 and 2 read `eval/adaptation_tables/<version>/cells.csv`; table 3a reads the enrollment
evaluation payload; table 3b reads the Phase-B episodic run directories. Aggregation is the mean
over datasets within a regime, after averaging seeds within a dataset, so a dataset with more seeds
does not outvote one with fewer.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MODEL_NAMES = {
    "halo": "HALO (ours)", "harnet": "HARNet", "crosshar": "CrossHAR", "unimts": "UniMTS",
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


def table_zero_shot(cells: list[dict]) -> str:
    out = ["## 1. Zero-shot", "",
           "No labelled examples. Macro F1, mean over datasets.", ""]
    scored = []
    for model in MODEL_NAMES:
        per_regime = [
            _dataset_macro([c for c in cells if c["model"] == model
                            and c["method"] == "zero_shot" and c["regime"] == regime])[0]
            for regime in REGIMES
        ]
        scored.append((MODEL_NAMES[model], per_regime))
    scored.sort(key=lambda r: -np.nanmean(r[1]))
    out += _emit(["model"] + [r.replace("_", " ") for r in REGIMES], scored)
    counts = {r: _dataset_macro([c for c in cells if c["method"] == "zero_shot"
                                 and c["regime"] == r])[1] for r in REGIMES}
    out += ["", f"Datasets per regime: " + ", ".join(f"{r} {n}" for r, n in counts.items()), ""]
    return "\n".join(out)


def table_label_efficiency(cells: list[dict], method: str = "linear_head") -> str:
    ks = sorted({int(c["k"]) for c in cells if c["method"] == method}, key=int)
    out = ["## 2. Label efficiency", "",
           f"Frozen encoder, `{method}` fitted on k novel labelled examples per class. "
           "Macro F1, mean over datasets.", ""]
    for regime in REGIMES:
        scored = []
        for model in MODEL_NAMES:
            row = [
                _dataset_macro([c for c in cells if c["model"] == model
                                and c["method"] == method and c["regime"] == regime
                                and int(c["k"]) == k])[0]
                for k in ks
            ]
            scored.append((MODEL_NAMES[model], row))
        scored.sort(key=lambda r: -np.nanmean(r[1]))
        out += [f"### {regime.replace('_', ' ')}", ""]
        out += _emit(["model"] + [f"k={k}" for k in ks], scored)
        out.append("")
    return "\n".join(out)


def table_mechanism(enrollment: Path) -> str:
    payload = json.loads(enrollment.read_text())
    fields = {"native retrieval": "f1_macro", "identity": "identity_f1_macro",
              "prototype": "prototype_f1_macro", "ridge": "ridge_head_f1_macro",
              "support removed": "support_removed_f1_macro",
              "support labels shuffled": "support_label_shuffled_f1_macro"}
    finite = lambda v: isinstance(v, (int, float)) and v == v
    matched = {key: r for key, r in payload["results"].items()
               if int(r["support_count"]) > 0 and all(finite(r.get(f)) for f in fields.values())}
    ks = sorted({int(r["support_count"]) for r in matched.values()})
    by = defaultdict(lambda: defaultdict(list))
    for r in matched.values():
        for name, field in fields.items():
            by[name][int(r["support_count"])].append(float(r[field]))
    compared = ["native retrieval", "prototype", "ridge", "identity"]
    controls = ["support removed", "support labels shuffled"]
    out = ["## 3. Adaptation mechanism", "",
           "HALO only. The native retrieval/enrollment mechanism against heads fitted on the same "
           "frozen features, same cells. Macro F1.", ""]
    out += _emit(["mechanism"] + [f"k={k}" for k in ks],
                 [(n, [float(np.mean(by[n][k])) for k in ks]) for n in compared])
    out += ["", "Sanity controls (not competitors):", ""]
    out += _emit(["control"] + [f"k={k}" for k in ks],
                 [(n, [float(np.mean(by[n][k])) for k in ks]) for n in controls])
    datasets = sorted({k.split("/")[0] for k in matched})
    out += ["", f"{len(matched)} matched cells over {', '.join(datasets)}. "
                "Italic rows are sanity controls, not competitors.", ""]
    return "\n".join(out)


def table_ablations(runs: dict[str, Path]) -> str:
    out = ["### 3b. Mechanism ablations", "",
           "Phase-B held-out-concept validation (a different protocol from 3a: held-out concepts "
           "and subjects from the training corpus, deployment top-k rule). Macro F1 by support "
           "count.", "",
           "| arm | k=0 | k=1 | k=2 | k=4 | mean |", "|---|---:|---:|---:|---:|---:|"]
    rows = []
    for label, path in runs.items():
        summary = path / "summary.json"
        if not summary.exists():
            continue
        curve = json.loads(summary.read_text())["final_validation"]
        cells = curve["hard_curve"]
        rows.append((label, [cells.get(f"k={k}", {}).get("macro_f1", float("nan"))
                             for k in (0, 1, 2, 4)], curve["hard_mean_macro_f1"]))
    for label, cells, mean in rows:
        out.append(f"| {label} | " + " | ".join(f"{v:.4f}" for v in cells) + f" | {mean:.4f} |")
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path,
                    default=Path("eval/adaptation_tables/v2_20260819/cells.csv"))
    ap.add_argument("--enrollment", type=Path,
                    default=Path("training/evidence/outputs/eval_enrollment_h_mae_fixes.json"))
    ap.add_argument("--runs", type=Path, default=Path("training/tokenizer/outputs"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cells = _cells(args.cells)
    ablations = {
        "evidence mixer (adopted)": args.runs / "best_base",
        "+ semantic text refinement": args.runs / "best_refine_evidence",
        "mixer, post-trunk rows": args.runs / "ladder_mixer_full",
        "mixer, candidate-blind form only": args.runs / "sem_forms_qm",
        "mixer, scrambled label vocabulary": args.runs / "sem_scrambled",
        "NO mixer, encoder fine-tuned": args.runs / "ladder_encoder_isolated",
        "NO mixer, nothing trained": args.runs / "ladder_control_isolated",
    }
    text = "\n".join([
        "# Results", "",
        table_zero_shot(cells),
        table_label_efficiency(cells),
        table_mechanism(args.enrollment),
        table_ablations(ablations),
    ])
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
