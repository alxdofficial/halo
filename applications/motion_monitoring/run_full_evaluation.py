"""Run complete Task-1 and Task-3 evaluation for one frozen encoder cache."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.data.manifests import read_cohort_manifest
from applications.motion_monitoring.evaluation import (
    ApplicationEvaluation,
    fingerprint_protocol,
)
from applications.motion_monitoring.evaluation_manifests import read_task_manifest
from applications.motion_monitoring.representation_cache import CachedMotionSequenceDataset
from applications.motion_monitoring.task1.full_evaluation import evaluate_task1_test
from applications.motion_monitoring.task1.model import DifferentiableSubsequenceMatcher
from applications.motion_monitoring.task3.full_evaluation import evaluate_task3_test
from applications.motion_monitoring.task3.model import RecurrentMotionMetric


def _load_task1(path: Path, device: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DifferentiableSubsequenceMatcher(
        int(payload["feature_dim"]), projection_dim=int(payload["projection_dim"])
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def _load_task3(path: Path, device: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = RecurrentMotionMetric(
        int(payload["feature_dim"]), projection_dim=int(payload["projection_dim"])
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload


def _write_records(
    *,
    output: Path,
    task: str,
    encoder: str,
    readout: str,
    results,
    cohort_fingerprint: str,
    provenance,
    protocol,
    threshold: float,
) -> None:
    protocol_fingerprint = fingerprint_protocol(protocol)
    directory = output.with_suffix("").parent / f"{output.stem}_records"
    for result in results:
        metrics = dict(result.metrics)
        metrics["development_score_threshold"] = float(threshold)
        counts = (
            {"units": int(result.units), "target_events": int(result.target_events)}
            if task == "task1"
            else {
                "streams": int(result.streams),
                "target_occurrences": int(result.target_occurrences),
            }
        )
        record = ApplicationEvaluation(
            task=task,
            encoder=f"{encoder} / {readout}",
            dataset=result.dataset,
            split="test",
            cohort_fingerprint=cohort_fingerprint,
            protocol_fingerprint=protocol_fingerprint,
            protocol=protocol,
            metrics=metrics,
            counts=counts,
            encoder_provenance=provenance,
        )
        record.to_json(directory / f"{task}_{readout}_{result.dataset}.json")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--cohort", type=Path, default=root / "manifests/COHORT_V1.json")
    parser.add_argument(
        "--task1-manifest", type=Path, default=root / "manifests/TASK1_TEST_V1.json"
    )
    parser.add_argument(
        "--task1-development-manifest",
        type=Path,
        default=root / "manifests/TASK1_DEVELOPMENT_V1.json",
    )
    parser.add_argument(
        "--task3-manifest", type=Path, default=root / "manifests/TASK3_TEST_V1.json"
    )
    parser.add_argument(
        "--task3-development-manifest",
        type=Path,
        default=root / "manifests/TASK3_DEVELOPMENT_V1.json",
    )
    parser.add_argument("--common-task1-units", type=Path, required=True)
    parser.add_argument(
        "--common-task1-development-units", type=Path, required=True
    )
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--head-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task1-device", default="cpu")
    parser.add_argument("--task3-device", default="cuda")
    parser.add_argument("--task3-block-size", type=int, default=1024)
    args = parser.parse_args()

    cohort = read_cohort_manifest(args.cohort)
    representations = CachedMotionSequenceDataset(
        args.representations, manifest_fingerprint=cohort.fingerprint
    )
    task1_manifest = read_task_manifest(args.task1_manifest)
    task1_development_manifest = read_task_manifest(args.task1_development_manifest)
    common_development = json.loads(
        args.common_task1_development_units.read_text(encoding="utf-8")
    )
    if (
        common_development["task_manifest_fingerprint"]
        != task1_development_manifest.fingerprint
    ):
        raise ValueError("common Task-1 development units belong to another manifest")
    common = json.loads(args.common_task1_units.read_text())
    if common["task_manifest_fingerprint"] != task1_manifest.fingerprint:
        raise ValueError("common Task-1 units belong to another task manifest")
    selected = [int(index) for index in common["selected_unit_indices"]]
    task1_manifest = replace(
        task1_manifest, units=tuple(task1_manifest.units[index] for index in selected)
    )
    task3_manifest = read_task_manifest(args.task3_manifest)
    task3_development_manifest = read_task_manifest(args.task3_development_manifest)
    datasets = sorted(
        {str(row["dataset"]) for row in (*task1_manifest.units, *task3_manifest.units)}
    )
    recording_caches = {dataset: open_cache(dataset) for dataset in datasets}

    task1_training = json.loads(
        (args.head_directory / "task1_training.json").read_text()
    )
    common_development_fingerprint = fingerprint_protocol(common_development)
    if (
        task1_training.get("development_common_unit_fingerprint")
        != common_development_fingerprint
    ):
        raise ValueError("Task-1 head was not calibrated on the requested common units")
    task1_model = _load_task1(
        args.head_directory / "task1_head.pt", args.task1_device
    )
    task1_direct = evaluate_task1_test(
        task1_manifest,
        recording_caches,
        representations,
        score_threshold=float(task1_training["direct"]["threshold"]),
    )
    task1_learned = evaluate_task1_test(
        task1_manifest,
        recording_caches,
        representations,
        score_threshold=float(task1_training["learned"]["threshold"]),
        model=task1_model,
    )

    task3_training = json.loads(
        (args.head_directory / "task3_training.json").read_text()
    )
    task3_model, task3_checkpoint = _load_task3(
        args.head_directory / "task3_head.pt", args.task3_device
    )
    task3_direct = evaluate_task3_test(
        task3_manifest,
        recording_caches,
        representations,
        score_threshold=float(task3_training["calibration"]["direct"]["threshold"]),
        durations_sec=task3_checkpoint["durations"],
        candidate_stride_sec=float(task3_checkpoint["candidate_stride"]),
        block_size=args.task3_block_size,
        device=args.task3_device,
    )
    task3_learned = evaluate_task3_test(
        task3_manifest,
        recording_caches,
        representations,
        score_threshold=float(task3_training["calibration"]["learned"]["threshold"]),
        model=task3_model,
        durations_sec=task3_checkpoint["durations"],
        candidate_stride_sec=float(task3_checkpoint["candidate_stride"]),
        block_size=args.task3_block_size,
        device=args.task3_device,
    )
    provenance = representations.metadata["encoder_provenance"]
    task1_protocol_base = {
        "schema_version": 1,
        "task": "task1",
        "test_manifest_fingerprint": task1_manifest.fingerprint,
        "common_unit_fingerprint": fingerprint_protocol(common),
        "common_unit_count": len(selected),
        "development_manifest_fingerprint": task1_development_manifest.fingerprint,
        "common_development_unit_fingerprint": common_development_fingerprint,
        "common_development_unit_count": len(
            common_development["selected_unit_indices"]
        ),
        "threshold_selection": "pooled_event_f1_then_false_alarms_per_hour",
        "nms_iou": 0.3,
        "match_iou": 0.5,
    }
    task3_protocol_base = {
        "schema_version": 1,
        "task": "task3",
        "test_manifest_fingerprint": task3_manifest.fingerprint,
        "development_manifest_fingerprint": task3_development_manifest.fingerprint,
        "threshold_selection": "balanced_pair_accuracy",
        "durations_sec": list(task3_checkpoint["durations"]),
        "candidate_stride_sec": float(task3_checkpoint["candidate_stride"]),
        "mutual_k": 5,
        "minimum_occurrences": 2,
        "match_iou": 0.5,
        "multiscale_nms_iou": 0.5,
    }
    for readout, results, threshold in (
        ("direct_dtw", task1_direct, task1_training["direct"]["threshold"]),
        ("learned_metric_dtw", task1_learned, task1_training["learned"]["threshold"]),
    ):
        _write_records(
            output=args.output,
            task="task1",
            encoder=args.encoder,
            readout=readout,
            results=results,
            cohort_fingerprint=cohort.fingerprint,
            provenance=provenance,
            protocol={**task1_protocol_base, "readout": readout},
            threshold=float(threshold),
        )
    for readout, results, threshold in (
        (
            "direct_cosine_recurrence",
            task3_direct,
            task3_training["calibration"]["direct"]["threshold"],
        ),
        (
            "learned_metric_recurrence",
            task3_learned,
            task3_training["calibration"]["learned"]["threshold"],
        ),
    ):
        _write_records(
            output=args.output,
            task="task3",
            encoder=args.encoder,
            readout=readout,
            results=results,
            cohort_fingerprint=cohort.fingerprint,
            provenance=provenance,
            protocol={**task3_protocol_base, "readout": readout},
            threshold=float(threshold),
        )
    payload = {
        "schema_version": 1,
        "status": "complete_test_evaluation",
        "encoder": args.encoder,
        "cohort_fingerprint": cohort.fingerprint,
        "representation_provenance": provenance,
        "task1_common_units": len(selected),
        "task1": {
            "direct_dtw": [asdict(row) for row in task1_direct],
            "learned_metric_dtw": [asdict(row) for row in task1_learned],
        },
        "task3": {
            "direct_cosine_recurrence": [asdict(row) for row in task3_direct],
            "learned_metric_recurrence": [asdict(row) for row in task3_learned],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
