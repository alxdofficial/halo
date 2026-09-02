"""Fail-closed preflight for complete application test-set evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping

from applications.motion_monitoring.data.cache import CachedRecordingDataset
from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.data.manifests import (
    CohortManifest,
    read_cohort_manifest,
)
from applications.motion_monitoring.evaluation_manifests import (
    Task1EvaluationUnit,
    Task3EvaluationUnit,
    read_task_manifest,
    validate_task_manifest,
)
from applications.motion_monitoring.representation_cache import (
    CachedMotionSequenceDataset,
)


def _event_patch_count(sequence, event) -> int:
    centers = sequence.intervals_sec.mean(dim=1)
    selected = (
        sequence.valid
        & (centers >= float(event.start_sec))
        & (centers < float(event.end_sec))
    )
    return int(selected.sum())


def _representation_keys(
    cache: CachedMotionSequenceDataset, *, split: str | None = None
) -> set[tuple[str, str, str]]:
    return {
        (row["dataset"], row["recording_id"], row["stream_id"])
        for row in cache.rows
        if split is None or row.get("split") == split
    }


def evaluate_readiness(
    cohort: CohortManifest,
    recording_caches: Mapping[str, CachedRecordingDataset],
    representation_caches: Mapping[str, CachedMotionSequenceDataset],
    *,
    manifest_directory: Path,
) -> dict[str, Any]:
    manifests = {
        task: read_task_manifest(manifest_directory / f"TASK{index}_TEST_V1.json")
        for index, task in enumerate(("task1", "task2", "task3"), 1)
    }
    for manifest in manifests.values():
        validate_task_manifest(manifest, cohort, recording_caches)

    expected_keys = {
        (entry.dataset, entry.recording_id, stream_id)
        for entry in cohort.entries
        if entry.split == "test"
        for stream_id in entry.stream_ids
    }
    report: dict[str, Any] = {
        "cohort": cohort.name,
        "cohort_fingerprint": cohort.fingerprint,
        "test_recordings": sum(entry.split == "test" for entry in cohort.entries),
        "test_streams": len(expected_keys),
        "task_units": {
            task: len(manifest.units) for task, manifest in manifests.items()
        },
        "task2_status": "blocked_no_controlled_change_ground_truth",
        "encoders": {},
    }
    for name, cache in representation_caches.items():
        if cache.metadata.get("cohort_fingerprint") != cohort.fingerprint:
            raise ValueError(f"{name} representation cache belongs to another cohort")
        keys = _representation_keys(cache)
        test_keys = _representation_keys(cache, split="test")
        missing = expected_keys - test_keys
        extra = test_keys - expected_keys
        task1_eligible = Counter()
        task1_ineligible = Counter()
        for row in manifests["task1"].units:
            unit = Task1EvaluationUnit(**row)
            reference_key = (
                unit.dataset,
                unit.reference_recording_id,
                unit.reference_stream_id,
            )
            query_key = (
                unit.dataset,
                unit.query_recording_id,
                unit.query_stream_id,
            )
            if reference_key not in keys or query_key not in keys:
                task1_ineligible[unit.dataset] += 1
                continue
            sequence = cache.get(*reference_key)
            recording = recording_caches[unit.dataset][unit.reference_cache_index]
            event = recording.events[unit.reference_event_index]
            if _event_patch_count(sequence, event) < 1:
                task1_ineligible[unit.dataset] += 1
            else:
                task1_eligible[unit.dataset] += 1

        task3_patch_counts: dict[str, list[int]] = defaultdict(list)
        for row in manifests["task3"].units:
            unit = Task3EvaluationUnit(**row)
            key = (unit.dataset, unit.recording_id, unit.stream_id)
            if key in keys:
                task3_patch_counts[unit.dataset].append(len(cache.get(*key).embeddings))
        encoder_report = {
            "complete_test_cache": not missing and not extra,
            "missing_test_streams": len(missing),
            "extra_streams": len(extra),
            "task1_eligible_units": dict(sorted(task1_eligible.items())),
            "task1_ineligible_units": dict(sorted(task1_ineligible.items())),
            "task3_patch_counts": {
                dataset: {
                    "streams": len(counts),
                    "total": sum(counts),
                    "max": max(counts),
                }
                for dataset, counts in sorted(task3_patch_counts.items())
            },
        }
        encoder_report["ready_for_task1"] = (
            encoder_report["complete_test_cache"]
            and sum(task1_ineligible.values()) == 0
        )
        encoder_report["ready_for_task3_representation"] = (
            encoder_report["complete_test_cache"]
        )
        report["encoders"][name] = encoder_report
    report["ready_for_full_task1"] = bool(representation_caches) and all(
        row["ready_for_task1"] for row in report["encoders"].values()
    )
    report["ready_for_full_task2"] = False
    report["ready_for_full_task3"] = bool(representation_caches) and all(
        row["ready_for_task3_representation"]
        for row in report["encoders"].values()
    )
    return report


def _parse_representation(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use NAME=PATH") from error
    if not name or not path:
        raise argparse.ArgumentTypeError("use NAME=PATH")
    return name, Path(path)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", type=Path, default=root / "manifests" / "COHORT_V1.json"
    )
    parser.add_argument(
        "--manifest-directory", type=Path, default=root / "manifests"
    )
    parser.add_argument(
        "--representation",
        action="append",
        type=_parse_representation,
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cohort = read_cohort_manifest(args.cohort)
    datasets = sorted({entry.dataset for entry in cohort.entries if entry.split == "test"})
    recording_caches = {dataset: open_cache(dataset) for dataset in datasets}
    representation_caches = {
        name: CachedMotionSequenceDataset(path, manifest_fingerprint=cohort.fingerprint)
        for name, path in args.representation
    }
    report = evaluate_readiness(
        cohort,
        recording_caches,
        representation_caches,
        manifest_directory=args.manifest_directory,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
