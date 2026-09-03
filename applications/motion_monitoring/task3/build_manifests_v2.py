"""Build the Task-3 V2 cohort and its train/evaluation manifests.

Two data roles only, matching Tasks 1 and 2: train and evaluation, no development
split. The recurrence threshold is fixed a priori on held-out *training
identities* (``task3/train_full.py::OPERATING_POINT_PROTOCOL``).

Roles follow the audit in ``docs/tasks/TASK3_RECURRENT_MOTION_DISCOVERY.md``
section 10:

* train -- Opportunity mid-level gestures (natural long recordings with real
  boundaries, background and execution variability) and ``synth_long_v1``
  (assembled recordings that force the matcher to locate boundaries it was not
  given).
* evaluation -- OpenPack and OCA, the only available sources whose recordings
  actually contain recurrence: a median of 20 and 26 occurrences per identity
  respectively, against a median of 1 for C-MHAD and WEAR.

Moving OpenPack out of training also settles the standing cross-task conflict:
all 16 of its subjects were simultaneously Task-3 training and Task-1 sealed test.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.data.manifests import (
    build_cohort_manifest,
    write_cohort_manifest,
)
from applications.motion_monitoring.evaluation_manifests import (
    Task3EvaluationUnit,
    _manifest,
    write_task_manifest,
)


COHORT_NAME = "cohort_task3_v2"
TRAIN_ONLY = ("opportunity", "synth_long_v1")
EVALUATION = ("openpack", "oca")

# Annotation kind carrying one bounded execution, and the labels that are not a
# motion identity, per source.
SOURCES: dict[str, dict[str, Any]] = {
    "opportunity": {"kind": "gesture", "background": (), "exhaustive": False},
    "synth_long_v1": {"kind": "inserted_execution", "background": (), "exhaustive": False},
    "openpack": {
        "kind": "fine_action",
        "background": ("Ignore", "Others", "System Error", "Unknown"),
        # 0.63 of the timeline carries a fine action, so the complement is
        # unlabelled rather than known-empty; section 10.5 requires review before
        # a motif found there is counted as a false positive.
        "exhaustive": False,
    },
    "oca": {"kind": "sample_label_run", "background": ("Null",), "exhaustive": True},
}


def _units(cohort, caches, split: str) -> tuple[list[Task3EvaluationUnit], list[dict[str, Any]]]:
    units: list[Task3EvaluationUnit] = []
    exclusions: list[dict[str, Any]] = []
    for entry in cohort.entries:
        if entry.split != split:
            continue
        config = SOURCES[entry.dataset]
        recording = caches[entry.dataset][entry.cache_index]
        events = [
            event
            for event in recording.events
            if event.annotation_kind == config["kind"]
            and event.label not in set(config["background"])
        ]
        if len(events) < 2:
            exclusions.append(
                {
                    "dataset": entry.dataset,
                    "recording_id": entry.recording_id,
                    "reason": "fewer than two bounded executions to compare",
                    "events": len(events),
                }
            )
            continue
        for stream_id in entry.stream_ids:
            units.append(
                Task3EvaluationUnit(
                    dataset=entry.dataset,
                    cache_index=entry.cache_index,
                    recording_id=entry.recording_id,
                    subject_id=entry.subject_id,
                    stream_id=stream_id,
                    annotation_kind=str(config["kind"]),
                    background_labels=tuple(config["background"]),
                    exhaustive=bool(config["exhaustive"]),
                )
            )
    return units, exclusions


def _recurrence(caches, units) -> dict[str, Any]:
    """Report the property the task depends on, per source, in the manifest itself."""

    per_source: dict[str, list[int]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for unit in units:
        key = (unit.dataset, unit.cache_index)
        if key in seen:
            continue
        seen.add(key)
        recording = caches[unit.dataset][unit.cache_index]
        counts: dict[str, int] = defaultdict(int)
        for event in recording.events:
            if event.annotation_kind != unit.annotation_kind:
                continue
            if event.label in set(unit.background_labels):
                continue
            counts[event.label] += 1
        per_source[unit.dataset].extend(counts.values())
    summary = {}
    for dataset, counts in sorted(per_source.items()):
        ordered = sorted(counts)
        summary[dataset] = {
            "identity_occurrences_median": ordered[len(ordered) // 2] if ordered else 0,
            "identities_recurring_3_or_more": sum(1 for value in counts if value >= 3),
            "identities": len(counts),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "manifests"
    )
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    datasets = sorted((*TRAIN_ONLY, *EVALUATION))
    caches = {dataset: open_cache(dataset) for dataset in datasets}
    cohort = build_cohort_manifest(
        caches,
        name=COHORT_NAME,
        seed=args.seed,
        train_only_datasets=TRAIN_ONLY,
        evaluation_datasets=EVALUATION,
        # CrossFit repetition arrays are duplicate views of their parent set and
        # are excluded by default; synth_long_v1 consumes them upstream, so this
        # cohort holds only assembled recordings and needs the default rule.
        exclude_duplicate_views=True,
    )
    cohort_path = args.output_dir / "COHORT_TASK3_V2.json"
    write_cohort_manifest(cohort, cohort_path)
    print(f"cohort: {len(cohort.entries)} entries, {cohort.fingerprint} -> {cohort_path}")

    for split, name in (("train", "task3_train_v2"), ("test", "task3_test_v2")):
        units, exclusions = _units(cohort, caches, split)
        manifest = _manifest(
            name=name,
            task="task3",
            cohort=cohort,
            seed=args.seed,
            protocol={
                "split": split,
                "unit": "one complete recording-stream timeline",
                "sources": {
                    dataset: SOURCES[dataset]
                    for dataset in sorted({unit.dataset for unit in units})
                },
                "pair_sampling": "event-anchored (design doc section 2.3)",
                "threshold": "a priori, on held-out training identities; no development split",
                "recurrence": _recurrence(caches, units),
                "reported_separately": sorted({unit.dataset for unit in units}),
            },
            units=units,
            exclusions=exclusions,
        )
        path = args.output_dir / f"TASK3_{split.upper()}_V2.json"
        write_task_manifest(manifest, path)
        counts: dict[str, int] = defaultdict(int)
        for unit in units:
            counts[unit.dataset] += 1
        print(f"{name}: {len(units)} timelines {dict(counts)}, {len(exclusions)} exclusions, {manifest.fingerprint}")


if __name__ == "__main__":
    main()
