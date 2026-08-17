"""Assemble matched adaptation outputs into dataset-macro tables and paired comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.enrollment_protocol import load_manifest


def _external_rows(payload: dict, manifest: dict) -> tuple[list[dict], list[dict]]:
    model = payload["baseline"]
    rows, subjects = [], []
    for key, result in payload["results"].items():
        if result.get("status"):
            continue
        dataset = key.split("/", 1)[0]
        common = {
            "model": model,
            "dataset": dataset,
            "regime": result["regime"],
            "label_mode": result.get("label_mode", "coherent"),
            "k": int(result["support_count"]),
            "seed": int(result.get("seed", 0)),
            "cell": key.rsplit("/", 3)[0] if result.get("kind") == "enrollment" else key,
        }
        methods = ["zero_shot"] if result["kind"] == "zero_shot" else payload["methods"]
        for method in methods:
            metric = result.get(method)
            if not isinstance(metric, dict):
                continue
            rows.append({**common, "method": method, "f1_macro": float(metric["f1_macro"])})
            field = f"{method}_f1_macro"
            for subject, record in result.get("subject_results", {}).items():
                if field in record:
                    subjects.append({
                        **common, "method": method, "subject": f"{dataset}:{subject}",
                        "f1_macro": float(record[field]),
                    })
    return rows, subjects


def _halo_rows(payload: dict, manifest: dict) -> tuple[list[dict], list[dict]]:
    model = "halo_learned_gate"
    label_mode = "random_alias" if payload.get("random_aliases") else "coherent"
    seed = int(payload.get("manifest_seed") or payload.get("seed", 0))
    metric_fields = {
        "learned": "f1_macro",
        "identity": "identity_f1_macro",
        "prototype": "prototype_f1_macro",
        "ridge": "ridge_head_f1_macro",
        "support_removed": "support_removed_f1_macro",
        "support_shuffled": "support_label_shuffled_f1_macro",
    }
    rows, subjects = [], []
    for key, result in payload["results"].items():
        if result.get("status"):
            continue
        dataset = key.split("/", 1)[0]
        regime = next(
            name for name, datasets in manifest["action_regimes"].items() if dataset in datasets
        )
        k = int(result["support_count"])
        common = {
            "model": model,
            "dataset": dataset,
            "regime": regime,
            "label_mode": label_mode,
            "k": k,
            "seed": seed,
            "cell": "/".join(key.split("/")[:5]),
        }
        for method, field in metric_fields.items():
            value = result.get(field)
            if value is None:
                continue
            rows.append({**common, "method": method, "f1_macro": float(value)})
            for subject, record in result.get("subject_results", {}).items():
                if field in record and record[field] is not None:
                    subjects.append({
                        **common, "method": method, "subject": f"{dataset}:{subject}",
                        "f1_macro": float(record[field]),
                    })
    return rows, subjects


def load_rows(paths: list[Path], manifest: dict) -> tuple[list[dict], list[dict]]:
    rows, subjects = [], []
    for path in paths:
        payload = json.loads(path.read_text())
        actual = payload.get("manifest_fingerprint")
        if actual != manifest["manifest_fingerprint"]:
            raise ValueError(
                f"{path}: manifest mismatch ({actual} != {manifest['manifest_fingerprint']})"
            )
        parsed = (
            _external_rows(payload, manifest) if "baseline" in payload
            else _halo_rows(payload, manifest)
        )
        rows.extend(parsed[0]); subjects.extend(parsed[1])
    return rows, subjects


def dataset_macro(rows: list[dict]) -> list[dict]:
    per_dataset = defaultdict(list)
    for row in rows:
        key = (
            row["model"], row["method"], row["regime"], row["label_mode"],
            row["k"], row["dataset"],
        )
        per_dataset[key].append(row["f1_macro"])
    dataset_values = [
        {
            "model": key[0], "method": key[1], "regime": key[2], "label_mode": key[3],
            "k": key[4], "dataset": key[5], "f1_macro": float(np.mean(values)),
            "protocol_cells": len(values),
        }
        for key, values in per_dataset.items()
    ]
    groups = defaultdict(list)
    for row in dataset_values:
        key = (row["model"], row["method"], row["regime"], row["label_mode"], row["k"])
        groups[key].append(row)
    return [
        {
            "model": key[0], "method": key[1], "regime": key[2], "label_mode": key[3],
            "k": key[4], "f1_macro": float(np.mean([row["f1_macro"] for row in values])),
            "datasets": len(values),
        }
        for key, values in groups.items()
    ]


def paired_deltas(subject_rows: list[dict], samples: int = 5_000) -> list[dict]:
    """Paired subject bootstrap within each evaluation condition."""
    indexed = defaultdict(dict)
    for row in subject_rows:
        key = (
            row["regime"], row["label_mode"], row["k"], row["cell"],
            row["seed"], row["subject"],
        )
        indexed[(row["model"], row["method"])][key] = row["f1_macro"]
    target_key = ("halo_learned_gate", "learned")
    if target_key not in indexed:
        return []
    target = indexed[target_key]
    rng = np.random.default_rng(20260817)
    output = []
    for comparator, values in indexed.items():
        if comparator == target_key:
            continue
        common = sorted(set(target) & set(values))
        if not common:
            continue
        # Keep unlike protocols separate, then average repeated stream/seed cells
        # within each independent dataset-subject unit.
        by_condition = defaultdict(lambda: defaultdict(list))
        for key in common:
            condition = key[:3]
            by_condition[condition][key[-1]].append(target[key] - values[key])
        for (regime, label_mode, k), by_subject in by_condition.items():
            deltas = np.asarray(
                [np.mean(value) for value in by_subject.values()], dtype=np.float64
            )
            draws = rng.choice(deltas, size=(samples, len(deltas)), replace=True).mean(1)
            output.append({
                "target": "halo_learned_gate/learned",
                "comparator": f"{comparator[0]}/{comparator[1]}",
                "regime": regime,
                "label_mode": label_mode,
                "k": k,
                "paired_subjects": len(deltas),
                "delta_f1_macro": float(deltas.mean()),
                "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
            })
    return sorted(
        output,
        key=lambda row: (
            row["regime"], row["label_mode"], row["k"], row["comparator"]
        ),
    )


def _markdown(aggregates: list[dict]) -> str:
    lines = ["# Matched adaptation results", ""]
    panels = [
        ("Semantic zero-shot", lambda row: (
            row["label_mode"] == "coherent" and row["k"] == 0
            and (row["method"] == "zero_shot" or (
                row["model"] == "halo_learned_gate" and row["method"] in {"learned", "identity"}
            ))
        )),
        ("Coherent label efficiency", lambda row: (
            row["label_mode"] == "coherent" and row["k"] > 0
            and ((row["model"] == "halo_learned_gate" and row["method"] == "learned")
                 or (row["model"] != "halo_learned_gate" and row["method"] == "linear_head"))
        )),
        ("HALO coherent mechanism ablation", lambda row: (
            row["model"] == "halo_learned_gate" and row["label_mode"] == "coherent"
            and row["k"] > 0
        )),
        ("Random-label binding", lambda row: (
            row["label_mode"] == "random_alias" and row["k"] > 0
        )),
    ]
    for title, include in panels:
        selected = sorted(
            (row for row in aggregates if include(row)),
            key=lambda row: (row["regime"], row["model"], row["method"], row["k"]),
        )
        lines.extend([f"## {title}", "", "| regime | model | method | k | macro F1 | datasets |",
                      "|---|---|---:|---:|---:|---:|"])
        for row in selected:
            lines.append(
                f"| {row['regime']} | {row['model']} | {row['method']} | {row['k']} | "
                f"{row['f1_macro']:.2f} | {row['datasets']} |"
            )
        if not selected:
            lines.append("| - | - | - | - | - | - |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest, validate_grids=True)
    rows, subjects = load_rows(args.inputs, manifest)
    aggregates = dataset_macro(rows)
    paired = paired_deltas(subjects)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("cells.csv", rows), ("dataset_macro.csv", aggregates)):
        path = args.out_dir / name
        fields = sorted({key for row in values for key in row})
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(values)
    (args.out_dir / "paired_deltas.json").write_text(
        json.dumps(paired, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "tables.md").write_text(_markdown(aggregates) + "\n")
    print(f"assembled {len(rows)} cells from {len(args.inputs)} artifacts -> {args.out_dir}")


if __name__ == "__main__":
    main()
