"""Build the authoritative temporal-annotation capability inventory."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOURCE_ROOT = HERE / "sources"
OUTPUT_PATH = HERE / "ANNOTATION_INVENTORY.json"


# These are source/adapter contracts, not conclusions inferred from label names.
SOURCE_CONTRACTS: dict[str, dict[str, Any]] = {
    "openpack": {
        "study_role": "train_development",
        "current_data_form": "continuous_application_timeline",
        "continuous_timeline": True,
        "exact_instance_intervals": True,
        "instance_granularity": ["fine_action", "operation", "box_cycle"],
        "repeated_instances_in_timeline": True,
        "background_contract": "NULL and outlier annotations; verify completeness per track",
        "known_execution_change": False,
        "video_reference": "source annotations include video-derived fine-action labels",
        "recommended_use": "Task 1/3 training and occupational within-source evaluation",
    },
    "crossfit": {
        "study_role": "train_development",
        "current_data_form": "parent exercise arrays plus derived repetition excerpts",
        "continuous_timeline": True,
        "exact_instance_intervals": True,
        "instance_granularity": ["exercise_sequence", "repetition"],
        "repeated_instances_in_timeline": True,
        "background_contract": "NULL arrays exist but are not an exhaustive natural timeline",
        "known_execution_change": False,
        "video_reference": "not required by the selected release",
        "recommended_use": "controlled Task 1/3 repetition training",
    },
    "aidlab_har": {
        "study_role": "train_development_control",
        "current_data_form": "short recording with series interval and marker windows",
        "continuous_timeline": True,
        "exact_instance_intervals": False,
        "instance_granularity": ["series", "repetition_fiducial"],
        "repeated_instances_in_timeline": True,
        "background_contract": "three background-like recording classes; not exhaustive per series",
        "known_execution_change": False,
        "video_reference": "none in selected payload",
        "recommended_use": "series-boundary and temporal-anchor control only",
    },
    "recofit": {
        "study_role": "train_development_weak_supervision",
        "current_data_form": "continuous visit timeline",
        "continuous_timeline": True,
        "exact_instance_intervals": False,
        "instance_granularity": ["exercise_set", "count"],
        "repeated_instances_in_timeline": True,
        "background_contract": "explicit non-exercise and source-junk intervals",
        "known_execution_change": False,
        "video_reference": "none",
        "recommended_use": "background, set-level matching, and count supervision",
    },
    "c_mhad": {
        "study_role": "sealed_evaluation",
        "current_data_form": "continuous application timeline",
        "continuous_timeline": True,
        "exact_instance_intervals": True,
        "instance_granularity": ["action_instance"],
        "repeated_instances_in_timeline": True,
        "background_contract": "target events are scored; unlabeled background is not exhaustive",
        "known_execution_change": False,
        "video_reference": "manually inspected synchronized-video boundaries",
        "recommended_use": "primary sealed Task 1 and Task 3 event-level evaluation",
    },
    "wear": {
        "study_role": "sealed_evaluation",
        "current_data_form": "continuous activity timeline",
        "continuous_timeline": True,
        "exact_instance_intervals": False,
        "instance_granularity": ["activity_bout"],
        "repeated_instances_in_timeline": True,
        "background_contract": "activity intervals plus explicit complement as NULL",
        "known_execution_change": False,
        "video_reference": "synchronized egocentric video in source; not downloaded",
        "recommended_use": "long-duration activity-bout and false-alarm evaluation",
    },
    "oca": {
        "study_role": "sealed_evaluation",
        "current_data_form": "continuous sample-labeled assembly timeline",
        "continuous_timeline": True,
        "exact_instance_intervals": True,
        "instance_granularity": ["assembly_phase_run"],
        "repeated_instances_in_timeline": True,
        "background_contract": "sample-level Null labels cover the selected timeline",
        "known_execution_change": False,
        "video_reference": "none in release",
        "recommended_use": "occupational Task 1 transfer and Task 3 evaluation",
    },
    "xrf_v2": {
        "study_role": "train_development",
        "current_data_form": "Phase-A action excerpts; raw scene timeline retained locally",
        "continuous_timeline": True,
        "exact_instance_intervals": True,
        "instance_granularity": ["action_instance"],
        "repeated_instances_in_timeline": True,
        "background_contract": "temporal-action annotations do not prove exhaustive background",
        "known_execution_change": False,
        "video_reference": "video-aligned ActivityNet-style intervals",
        "recommended_use": "reconstruct raw scenes for Task 1/3 training; includes pouring_water",
        "adapter_gap": "application timeline adapter not yet implemented",
    },
    "harmes": {
        "study_role": "train_development",
        "current_data_form": "Phase-A action excerpts; raw recording and event log retained locally",
        "continuous_timeline": True,
        "exact_instance_intervals": True,
        "instance_granularity": ["activity_instance"],
        "repeated_instances_in_timeline": True,
        "background_contract": "inter-event time exists in source but is omitted by Phase-A converter",
        "known_execution_change": False,
        "video_reference": "event log; multimodal source exists but selected payload is wrist IMU",
        "recommended_use": "reconstruct wrist-ADL timelines for Task 1/3 training",
        "adapter_gap": "application timeline adapter not yet implemented",
    },
    "monipar": {
        "study_role": "sealed_task2_evaluation",
        "current_data_form": "continuous weekly protocol timeline",
        "continuous_timeline": True,
        "exact_instance_intervals": False,
        "instance_granularity": ["protocol_state_run"],
        "repeated_instances_in_timeline": False,
        "background_contract": "protocol transitions, not free-living exhaustive background",
        "known_execution_change": True,
        "video_reference": "neurologist-reviewed clinical protocol; severity adapter remains open",
        "recommended_use": (
            "Task 2 longitudinal association after severity alignment and active-state "
            "aggregation audits"
        ),
    },
    "phytmo": {
        "study_role": "task2_development",
        "current_data_form": "bounded exercise series",
        "continuous_timeline": False,
        "exact_instance_intervals": False,
        "instance_granularity": ["exercise_series"],
        "repeated_instances_in_timeline": True,
        "background_contract": "no natural continuous background",
        "known_execution_change": True,
        "video_reference": "optical reference in source protocol",
        "recommended_use": "Task 2 correct-versus-incorrect development",
    },
    "kneepad": {
        "study_role": "task2_evaluation_candidate",
        "current_data_form": "short bounded trials",
        "continuous_timeline": False,
        "exact_instance_intervals": False,
        "instance_granularity": ["trial"],
        "repeated_instances_in_timeline": False,
        "background_contract": "no continuous background",
        "known_execution_change": True,
        "video_reference": "source-defined correct and incorrect variants",
        "recommended_use": "Task 2 known-variant evaluation after provenance check",
    },
    "spar": {
        "study_role": "task2_development",
        "current_data_form": "exercise bouts with repeated shoulder movements",
        "continuous_timeline": False,
        "exact_instance_intervals": False,
        "instance_granularity": ["exercise_bout"],
        "repeated_instances_in_timeline": True,
        "background_contract": "no natural continuous background",
        "known_execution_change": False,
        "video_reference": "none in current representation",
        "recommended_use": "Task 2 within-bout repeatability, not event localization",
    },
    "mmfit": {
        "study_role": "train_development",
        "current_data_form": "Phase-A set excerpts; full workouts recoverable from source",
        "continuous_timeline": True,
        "exact_instance_intervals": False,
        "instance_granularity": ["exercise_set", "count"],
        "repeated_instances_in_timeline": True,
        "background_contract": "between-set timeline recoverable from source",
        "known_execution_change": False,
        "video_reference": "synchronized video and pose",
        "recommended_use": "Task 1/3 weak supervision and Task 2 phase development",
        "adapter_gap": "full application timeline adapter not yet implemented",
    },
}


def _summary_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _measure_application_cache(dataset: str) -> dict[str, Any] | None:
    paths = sorted(
        (SOURCE_ROOT / dataset / "processed" / "canonical_v1").glob("*/recording.json")
    )
    if not paths:
        return None

    subjects: set[str] = set()
    labels: set[str] = set()
    event_kinds: Counter[str] = Counter()
    durations: list[float] = []
    events_per_recording: list[int] = []
    background_events = 0
    background_seconds = 0.0
    repeated_label_recordings = 0
    multi_event_recordings = 0

    for path in paths:
        payload = json.loads(path.read_text())
        subjects.add(str(payload["subject_id"]))
        events = payload.get("events", [])
        events_per_recording.append(len(events))
        multi_event_recordings += len(events) > 1
        per_label: defaultdict[str, int] = defaultdict(int)
        for event in events:
            label = str(event["label"])
            kind = str(event.get("annotation_kind", "event"))
            duration = float(event["end_sec"]) - float(event["start_sec"])
            labels.add(label)
            event_kinds[kind] += 1
            durations.append(duration)
            per_label[label] += 1
            if kind in {"background", "source_junk"} or label.casefold() in {
                "null",
                "non-exercise",
            }:
                background_events += 1
                background_seconds += duration
        repeated_label_recordings += any(count > 1 for count in per_label.values())

    return {
        "cache_status": "measured",
        "recordings": len(paths),
        "subjects": len(subjects),
        "source_scoped_labels": len(labels),
        "event_count": int(sum(event_kinds.values())),
        "event_kinds": dict(sorted(event_kinds.items())),
        "event_duration_seconds": _summary_stats(durations),
        "events_per_recording": _summary_stats(events_per_recording),
        "recordings_with_multiple_events": multi_event_recordings,
        "recordings_with_repeated_label_instances": repeated_label_recordings,
        "background_event_count": background_events,
        "background_hours": background_seconds / 3600.0,
    }


def build_inventory() -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset, contract in SOURCE_CONTRACTS.items():
        row = dict(contract)
        measured = _measure_application_cache(dataset)
        row["application_cache"] = measured or {"cache_status": "not_applicable"}
        datasets[dataset] = row
    return {
        "schema_version": 1,
        "schema_updated_on": "2026-08-30",
        "definitions": {
            "exact_instance_intervals": (
                "Each source action execution or phase has both a start and an end; "
                "set boundaries, counts, and fiducial markers do not qualify."
            ),
            "continuous_timeline": (
                "The original source retains surrounding time, even when the current "
                "Phase-A converter emits only excerpts."
            ),
            "background_contract": (
                "States whether unlabeled time can legitimately be scored as negative."
            ),
        },
        "primary_natural_event_sources": ["c_mhad", "openpack", "oca", "xrf_v2"],
        "primary_isolated_instance_source": ["crossfit"],
        "primary_task2_change_sources": ["phytmo", "kneepad"],
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(build_inventory(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text() != rendered:
            raise SystemExit(f"annotation inventory is stale: regenerate {args.output}")
        return
    args.output.write_text(rendered)
    print(args.output)


if __name__ == "__main__":
    main()
