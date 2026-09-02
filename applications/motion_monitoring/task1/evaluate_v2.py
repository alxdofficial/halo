"""Task-1 V2 evaluation: calibrate once on natural development, then test.

Runs the untrained direct matcher or a trained head (``--head-directory``)
through the frozen protocol: the threshold is chosen on the development
manifest (natural, per-execution sources only) and applied unchanged to the
test manifest. Per-dataset event metrics, subject-cluster bootstrap CIs and
strata are reported; nothing is tuned on test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.data.manifests import read_cohort_manifest
from applications.motion_monitoring.evaluation_manifests import read_task_manifest
from applications.motion_monitoring.representation_cache import open_representations
from applications.motion_monitoring.task1.full_evaluation import evaluate_task1_test
from applications.motion_monitoring.task1.model import DifferentiableSubsequenceMatcher
from applications.motion_monitoring.task1.train_full import calibrate


def load_head(directory: Path, device: str) -> DifferentiableSubsequenceMatcher:
    checkpoint = torch.load(directory / "task1_head.pt", map_location="cpu", weights_only=False)
    model = DifferentiableSubsequenceMatcher(
        int(checkpoint["feature_dim"]), projection_dim=int(checkpoint["projection_dim"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=root / "manifests/COHORT_TASK1_V2.json")
    parser.add_argument(
        "--development-manifest", type=Path, default=root / "manifests/TASK1_DEVELOPMENT_V2.json"
    )
    parser.add_argument("--test-manifest", type=Path, default=root / "manifests/TASK1_TEST_V2.json")
    parser.add_argument("--representations", type=Path, nargs="+", required=True)
    parser.add_argument("--head-directory", type=Path, help="omit for the untrained direct matcher")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-test", action="store_true", help="development calibration only")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    cohort = read_cohort_manifest(args.cohort)
    representations = open_representations(args.representations, cohort=cohort)
    development = read_task_manifest(args.development_manifest)
    test = read_task_manifest(args.test_manifest)
    for manifest in (development, test):
        if manifest.cohort_fingerprint != cohort.fingerprint:
            raise ValueError(f"{manifest.name} belongs to another cohort")
    model = None if args.head_directory is None else load_head(args.head_directory, args.device)
    caches = {
        dataset: open_cache(dataset)
        for dataset in sorted(
            {str(row["dataset"]) for m in (development, test) for row in m.units}
        )
    }

    started = time.time()
    calibration = calibrate(development, caches, representations, model)
    calibration_seconds = time.time() - started
    threshold = float(calibration["threshold"])
    print(
        f"DEV calibration ({calibration_seconds:.0f}s): thr={threshold:.4f} "
        f"F1={calibration['metrics']['event_f1']:.3f} "
        f"FA/h={calibration['metrics']['false_alarms_per_hour']:.1f} "
        f"(eligible={calibration['eligible_units']}, rejected={calibration['rejected_units']})"
    )
    for dataset, metrics in sorted(calibration["per_dataset"].items()):
        print(f"  dev {dataset:12s} F1={metrics['event_f1']:.3f} FA/h={metrics['false_alarms_per_hour']:.1f}")

    report: dict[str, Any] = {
        "task": "task1",
        "protocol": "TASK1_REFERENCE_RESOLUTION_SPEC.md section C (V2 split)",
        "cohort_fingerprint": cohort.fingerprint,
        "development_manifest_fingerprint": development.fingerprint,
        "test_manifest_fingerprint": test.fingerprint,
        "representation_provenance": representations.metadata["encoder_provenance"],
        "representation_roots": [str(path) for path in args.representations],
        "head_directory": None if args.head_directory is None else str(args.head_directory),
        "arm": "direct_untrained" if model is None else "learned_head",
        "development": {
            "threshold": threshold,
            "metrics": calibration["metrics"],
            "per_dataset": calibration["per_dataset"],
            "eligible_units": calibration["eligible_units"],
            "rejected_units": calibration["rejected_units"],
        },
    }
    if not args.skip_test:
        started = time.time()
        results = evaluate_task1_test(test, caches, representations, score_threshold=threshold, model=model)
        print(f"TEST evaluation ({time.time() - started:.0f}s):")
        for row in _summary(results):
            ci = (row.get("subject_uncertainty") or {}).get("event_f1") or {}
            print(
                f"  {row['dataset']:10s} units={row['units']:4d} F1={row['event_f1']:.3f} "
                f"P={row['event_precision']:.3f} R={row['event_recall']:.3f} "
                f"FA/h={row['false_alarms_per_hour']:.1f}"
                + (f"  CI95=[{ci['low']:.3f}, {ci['high']:.3f}]" if {"low", "high"} <= set(ci) else "")
            )
        report["test"] = {"threshold": threshold, "datasets": _summary(results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
