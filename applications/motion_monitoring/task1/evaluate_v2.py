"""Task-1 V2 evaluation: a-priori operating point, then the sealed test set.

Runs the untrained direct matcher or a trained head (``--head-directory``)
through the frozen protocol (TASK1_REFERENCE_RESOLUTION_SPEC.md section F):

* the threshold is the operating point fixed on the synthetic train corpus's
  held-out background subjects at the a-priori false-alarm budget — read from
  the head checkpoint, or computed here for the direct floor;
* every test dataset is reported separately and under each declared reference
  relation (``cross_subject``, ``same_subject``); event average precision is the
  threshold-free primary number, event F1 / precision / recall / FA/h at the
  fixed operating point sit beside it, and F1 at the oracle threshold is a
  labelled upper bound only.

Nothing is tuned on natural data. There is no development split.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.data.manifests import read_cohort_manifest
from applications.motion_monitoring.evaluation import fingerprint_protocol
from applications.motion_monitoring.evaluation_manifests import (
    Task1EvaluationUnit,
    read_task_manifest,
    validate_task_manifest,
)
from applications.motion_monitoring.representation_cache import open_representations
from applications.motion_monitoring.task1.full_evaluation import evaluate_task1_test
from applications.motion_monitoring.task1.model import DifferentiableSubsequenceMatcher
from applications.motion_monitoring.task1.train_full import (
    OPERATING_POINT_PROTOCOL,
    _pooled_metrics,
    _rows_for_threshold,
    average_precision,
    detection_curve,
    fix_operating_point,
    select_common_units,
    split_by_subject,
)


def load_head(
    directory: Path,
    device: str,
    *,
    cohort_fingerprint: str,
    train_manifest_fingerprint: str,
    representation_provenance: Any,
    train_common_fingerprint: str | None,
) -> tuple[DifferentiableSubsequenceMatcher, dict[str, Any]]:
    checkpoint = torch.load(directory / "task1_head.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("cohort_fingerprint") != cohort_fingerprint:
        raise ValueError("Task-1 checkpoint belongs to another cohort")
    if checkpoint.get("train_manifest_fingerprint") != train_manifest_fingerprint:
        raise ValueError("Task-1 checkpoint was fitted on another train manifest")
    if checkpoint.get("representation_provenance") != representation_provenance:
        raise ValueError("Task-1 checkpoint used another encoder representation")
    if checkpoint.get("train_common_unit_fingerprint") != train_common_fingerprint:
        raise ValueError("Task-1 checkpoint used another common train-unit set")
    if checkpoint.get("operating_point_protocol") != OPERATING_POINT_PROTOCOL:
        raise ValueError("Task-1 checkpoint was fixed under another operating-point protocol")
    model = DifferentiableSubsequenceMatcher(
        int(checkpoint["feature_dim"]), projection_dim=int(checkpoint["projection_dim"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, dict(checkpoint["operating_point"])


def _summary(results) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        metrics = dict(result.metrics)
        rows.append(
            {
                "dataset": result.dataset,
                "units": result.units,
                "target_events": result.target_events,
                "event_f1": metrics.get("event_f1"),
                "event_precision": metrics.get("event_precision"),
                "event_recall": metrics.get("event_recall"),
                "false_alarms_per_hour": metrics.get("false_alarms_per_hour"),
                "subject_uncertainty": metrics.get("subject_uncertainty"),
                "strata": metrics.get("strata"),
                "rejected_units": metrics.get("rejected_units"),
            }
        )
    return rows


def _oracle_f1(evaluated) -> dict[str, float]:
    """Labelled upper bound: best pooled F1 over every candidate threshold."""

    scores = np.unique(np.concatenate([item["scores"] for item in evaluated]))
    if not len(scores):
        return {"event_f1": 0.0, "threshold": float("nan")}
    if len(scores) > 256:
        scores = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 256)))
    best = None
    for threshold in scores:
        metrics = _pooled_metrics(_rows_for_threshold(evaluated, float(threshold)))
        if best is None or metrics["event_f1"] > best["event_f1"]:
            best = {"event_f1": metrics["event_f1"], "threshold": float(threshold)}
    return best


def evaluate_by_relation(
    test,
    caches,
    representations,
    *,
    threshold: float,
    model,
) -> list[dict[str, Any]]:
    """One row per (dataset, relation): AP, fixed-threshold metrics with CIs, oracle F1."""

    rows: list[dict[str, Any]] = []
    units = [Task1EvaluationUnit(**row) for row in test.units]
    relations = sorted({unit.reference_relation for unit in units})
    for relation in relations:
        subset = tuple(row for row in test.units if row.get("reference_relation") == relation)
        if not subset:
            continue
        manifest = replace(test, units=subset)
        results = evaluate_task1_test(
            manifest,
            caches,
            representations,
            score_threshold=threshold,
            model=model,
            nms_iou=float(OPERATING_POINT_PROTOCOL["nms_iou"]),
            match_iou=float(OPERATING_POINT_PROTOCOL["match_iou"]),
        )
        subset_units = [unit for unit in units if unit.reference_relation == relation]
        curve, _ = detection_curve(
            subset_units,
            caches,
            representations,
            model,
            nms_iou=float(OPERATING_POINT_PROTOCOL["nms_iou"]),
            match_iou=float(OPERATING_POINT_PROTOCOL["match_iou"]),
        )
        for row in _summary(results):
            per_dataset = [item for item in curve if item["dataset"] == row["dataset"]]
            rows.append(
                {
                    "relation": relation,
                    **row,
                    "event_average_precision": average_precision(per_dataset),
                    "oracle_upper_bound": _oracle_f1(per_dataset),
                }
            )
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=root / "manifests/COHORT_TASK1_V2.json")
    parser.add_argument("--train-manifest", type=Path, default=root / "manifests/TASK1_TRAIN_V2.json")
    parser.add_argument("--test-manifest", type=Path, default=root / "manifests/TASK1_TEST_V2.json")
    parser.add_argument("--representations", type=Path, nargs="+", required=True)
    parser.add_argument("--head-directory", type=Path, help="omit for the untrained direct matcher")
    parser.add_argument(
        "--common-train-units",
        type=Path,
        required=True,
        help="intersection used for every encoder's head fit and operating point",
    )
    parser.add_argument("--common-test-units", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    cohort = read_cohort_manifest(args.cohort)
    representations = open_representations(args.representations, cohort=cohort)
    provenance = representations.metadata["encoder_provenance"]
    train = read_task_manifest(args.train_manifest)
    test = read_task_manifest(args.test_manifest)
    train, train_common = select_common_units(
        train, args.common_train_units, representation_provenance=provenance
    )
    test, test_common = select_common_units(
        test, args.common_test_units, representation_provenance=provenance
    )
    for manifest in (train, test):
        if manifest.cohort_fingerprint != cohort.fingerprint:
            raise ValueError(f"{manifest.name} belongs to another cohort")
    caches = {
        dataset: open_cache(dataset)
        for dataset in sorted({str(row["dataset"]) for m in (train, test) for row in m.units})
    }
    validate_task_manifest(train, cohort, caches)
    validate_task_manifest(test, cohort, caches)
    if test.protocol.get("split") != "test" or train.protocol.get("split") != "train":
        raise ValueError("evaluate_v2 needs a Task-1 train manifest and a Task-1 test manifest")
    train_common_fingerprint = None if train_common is None else fingerprint_protocol(train_common)

    started = time.time()
    if args.head_directory is None:
        model = None
        units = [Task1EvaluationUnit(**row) for row in train.units]
        _, heldout = split_by_subject(
            units, seed=args.seed, heldout_fraction=float(OPERATING_POINT_PROTOCOL["holdout_fraction"])
        )
        operating_point = fix_operating_point(
            heldout,
            caches,
            representations,
            None,
            false_alarm_budget_per_hour=float(OPERATING_POINT_PROTOCOL["false_alarm_budget_per_hour"]),
            nms_iou=float(OPERATING_POINT_PROTOCOL["nms_iou"]),
            match_iou=float(OPERATING_POINT_PROTOCOL["match_iou"]),
        )
        operating_point.pop("rejections", None)
    else:
        model, operating_point = load_head(
            args.head_directory,
            args.device,
            cohort_fingerprint=cohort.fingerprint,
            train_manifest_fingerprint=train.fingerprint,
            representation_provenance=provenance,
            train_common_fingerprint=train_common_fingerprint,
        )
    threshold = float(operating_point["threshold"])
    print(
        f"operating point ({time.time() - started:.0f}s): thr={threshold:.4f} at "
        f"{operating_point['false_alarm_budget_per_hour']:.0f} FA/h budget "
        f"(hold-out FA/h={operating_point['metrics']['false_alarms_per_hour']:.1f}, "
        f"F1={operating_point['metrics']['event_f1']:.3f})"
    )

    started = time.time()
    rows = evaluate_by_relation(test, caches, representations, threshold=threshold, model=model)
    print(f"TEST evaluation ({time.time() - started:.0f}s):")
    for row in rows:
        ci = (row.get("subject_uncertainty") or {}).get("event_f1_ci95") or []
        print(
            f"  {row['dataset']:10s} {row['relation']:13s} units={row['units']:4d} "
            f"AP={row['event_average_precision']:.3f} F1={row['event_f1']:.3f} "
            f"P={row['event_precision']:.3f} R={row['event_recall']:.3f} "
            f"FA/h={row['false_alarms_per_hour']:.1f} oracleF1={row['oracle_upper_bound']['event_f1']:.3f}"
            + (f"  CI95=[{ci[0]:.3f}, {ci[1]:.3f}]" if len(ci) == 2 else "")
        )
    report: dict[str, Any] = {
        "task": "task1",
        "protocol": "TASK1_REFERENCE_RESOLUTION_SPEC.md section F (train/evaluation only)",
        "cohort_fingerprint": cohort.fingerprint,
        "train_manifest_fingerprint": train.fingerprint,
        "test_manifest_fingerprint": test.fingerprint,
        "train_common_units": train_common,
        "test_common_units": test_common,
        "representation_provenance": provenance,
        "representation_roots": [str(path) for path in args.representations],
        "head_directory": None if args.head_directory is None else str(args.head_directory),
        "arm": "direct_untrained" if model is None else "learned_head",
        "operating_point_protocol": dict(OPERATING_POINT_PROTOCOL),
        "operating_point": operating_point,
        "test": {"threshold": threshold, "rows": rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
