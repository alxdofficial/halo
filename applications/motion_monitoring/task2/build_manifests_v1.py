"""Build the Task-2 cohort and its train/evaluation manifests.

Two data roles only, per the standing rule: train and evaluation, no development
split. Training is a declared *pool* rather than a fixed list of episodes, because
episodes are drawn under a seed at run time; the manifest freezes which executions
may be drawn and under what mixture. Evaluation is the opposite: every comparison
is frozen explicitly, so a reported number is reproducible from the manifest alone.

Evaluation episodes are built deterministically per source:

MoniPar serves two cells, because one reference rule cannot answer both questions.

* **MoniPar, ``between_week_reliability``** -- every subject and single-run
  protocol item with enough visits, whether or not a clinician scored them. The
  earliest ``reference_visits`` weekly visits are the reference set and every
  later visit is an ``accepted_query``. A weekly repeat of the same task by the
  same person is accepted by construction, so this needs no clinical label, and
  it is the only measurement of the between-week noise floor the whole protocol
  has: KneE-PAD is a single visit and cannot supply one. Restricting this cell to
  clinician-scored visits would discard every healthy control and every remote
  patient, which is 178 of the 196 series.
* **MoniPar, ``clinician_rated_change``** -- for each subject and item, the earliest
  ``reference_visits`` clinician-scored visits that share one score form a stable
  reference set. Every later clinician-scored visit is one query. A query is
  ``changed_query`` when its released MDS-UPDRS bradykinesia score differs from
  that reference score by at least ``CHANGE_MARGIN`` points; otherwise it is an
  ``accepted_query``. Unscored visits never receive an inferred label. The exact
  margin is carried per unit as ``score_margin`` so a predeclared two-point
  subgroup can be reported without rebuilding the manifest.
* **KneE-PAD, ``known_difference``** -- for each patient and exercise, the earliest ``reference_visits``
  correct trials form the reference set, the remaining correct trials are
  ``accepted_query``, and every incorrect variant trial is a ``changed_query``.
  All within one visit, so the relation is ``same_session`` throughout.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.data.manifests import (
    build_cohort_manifest,
    write_cohort_manifest,
)
from applications.motion_monitoring.evaluation_manifests import (
    Task2EvaluationUnit,
    _manifest,
    write_task_manifest,
)
from applications.motion_monitoring.task2.data_sources import (
    TASK2_READY_EVALUATION_DATASETS,
    TASK2_TRAIN_DATASETS,
    is_selected_event,
)
from applications.motion_monitoring.task2.records import EXECUTION_KINDS


COHORT_NAME = "cohort_task2_v1"
TRAIN_ONLY = (*TASK2_TRAIN_DATASETS, "task2_modified_v1")
EVALUATION = TASK2_READY_EVALUATION_DATASETS
# Four is the minimum for a leave-one-out operating limit whose fit folds still
# contain three independent executions.
REFERENCE_VISITS = 4
# Declared a priori. See the module docstring for why both are reported.
CHANGE_MARGIN = 1
STRICT_CHANGE_MARGIN = 2
# Declared source mixture for training (design doc section 4). CrossFit records one
# continuous set per subject and exercise, so all of its positives are
# same-session; HARMES carries every cross-day pair. The weights are the knob that
# keeps the same-session share visible and adjustable.
DATASET_WEIGHTS = {"harmes": 2.0, "crossfit": 1.0}


@dataclass(frozen=True)
class _ExecutionRow:
    cache_index: int
    recording_id: str
    subject_id: str
    recording_metadata: dict[str, Any]
    event_index: int
    event: Any
    stream_id: str


def _executions(cache, dataset: str):
    """Return grouped lightweight execution descriptors.

    Do not retain RawRecording objects here: their sensor arrays are memory maps,
    and KneE-PAD has enough recordings to exhaust the process descriptor limit.
    """

    kind = EXECUTION_KINDS[dataset]
    grouped: dict[tuple[str, str, str], list[_ExecutionRow]] = defaultdict(list)
    for cache_index, recording in enumerate(cache):
        for index, event in enumerate(recording.events):
            if event.annotation_kind != kind or not is_selected_event(recording, event):
                continue
            for stream in recording.streams:
                grouped[(recording.subject_id, event.label, stream.stream_id)].append(
                    _ExecutionRow(
                        cache_index=cache_index,
                        recording_id=recording.recording_id,
                        subject_id=recording.subject_id,
                        recording_metadata=dict(recording.metadata),
                        event_index=index,
                        event=event,
                        stream_id=stream.stream_id,
                    )
                )
    return grouped


def _monipar_reliability_units(
    grouped, *, reference_visits: int
) -> tuple[list[Task2EvaluationUnit], list[dict[str, Any]]]:
    """Between-week stability over every weekly repeat, scored or not."""

    units: list[Task2EvaluationUnit] = []
    exclusions: list[dict[str, Any]] = []
    for (subject, task, stream), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row.recording_metadata["week"]))
        if len(rows) <= reference_visits:
            exclusions.append(
                {
                    "dataset": "monipar",
                    "cell": "between_week_reliability",
                    "subject_id": subject,
                    "task_id": task,
                    "stream_id": stream,
                    "reason": "fewer weekly visits than one reference set plus a query",
                    "available_visits": len(rows),
                }
            )
            continue
        references = rows[:reference_visits]
        for row in rows[reference_visits:]:
            evidence: dict[str, Any] = {
                "week": int(row.recording_metadata["week"]),
                "cohort": row.recording_metadata["cohort"],
                "reference_weeks": [
                    int(item.recording_metadata["week"]) for item in references
                ],
            }
            score = row.event.metadata.get("mds_updrs_bradykinesia")
            if score is not None:
                # Carried for audit only. This cell never labels change; the
                # clinician-rated cell owns that question.
                evidence["mds_updrs_bradykinesia"] = int(score)
            units.append(
                Task2EvaluationUnit(
                    dataset="monipar",
                    subject_id=subject,
                    task_id=task,
                    stream_id=stream,
                    reference_cache_indices=tuple(item.cache_index for item in references),
                    reference_recording_ids=tuple(item.recording_id for item in references),
                    reference_event_indices=tuple(item.event_index for item in references),
                    query_cache_index=row.cache_index,
                    query_recording_id=row.recording_id,
                    query_event_index=row.event_index,
                    role="accepted_query",
                    relation="different_day",
                    change_evidence=evidence,
                    cell="between_week_reliability",
                )
            )
    return units, exclusions


def _monipar_units(
    cache, *, reference_visits: int
) -> tuple[list[Task2EvaluationUnit], list[dict[str, Any]]]:
    grouped = _executions(cache, "monipar")
    units, exclusions = _monipar_reliability_units(
        grouped, reference_visits=reference_visits
    )
    for (subject, task, stream), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row.recording_metadata["week"]))
        scored = [
            row
            for row in rows
            if "mds_updrs_bradykinesia" in row.event.metadata
        ]
        if not scored:
            exclusions.append(
                {
                    "dataset": "monipar",
                    "cell": "clinician_rated_change",
                    "subject_id": subject,
                    "task_id": task,
                    "stream_id": stream,
                    "reason": "no clinician-scored visits",
                }
            )
            continue
        baseline = int(scored[0].event.metadata["mds_updrs_bradykinesia"])
        references = [
            row
            for row in scored
            if int(row.event.metadata["mds_updrs_bradykinesia"]) == baseline
        ][:reference_visits]
        if len(references) < reference_visits:
            exclusions.append(
                {
                    "dataset": "monipar",
                    "subject_id": subject,
                    "task_id": task,
                    "stream_id": stream,
                    "cell": "clinician_rated_change",
                    "reason": "fewer than the required stable clinician-scored reference visits",
                    "available_stable_references": len(references),
                }
            )
            continue
        final_reference_week = int(references[-1].recording_metadata["week"])
        queries = [
            row
            for row in scored
            if int(row.recording_metadata["week"]) > final_reference_week
        ]
        if not queries:
            exclusions.append(
                {
                    "dataset": "monipar",
                    "subject_id": subject,
                    "task_id": task,
                    "stream_id": stream,
                    "cell": "clinician_rated_change",
                    "reason": "no clinician-scored visit after the stable reference period",
                }
            )
            continue
        reference_scores = [baseline] * reference_visits
        for row in queries:
            score = int(row.event.metadata["mds_updrs_bradykinesia"])
            margin = abs(score - baseline)
            changed = margin >= CHANGE_MARGIN
            evidence: dict[str, Any] = {
                "week": int(row.recording_metadata["week"]),
                "mds_updrs_bradykinesia": score,
                "reference_scores": reference_scores,
                "reference_median": baseline,
                "score_margin": margin,
                "strict_change": bool(margin >= STRICT_CHANGE_MARGIN),
            }
            if "mds_updrs_rest_tremor" in row.event.metadata:
                evidence["mds_updrs_rest_tremor"] = int(
                    row.event.metadata["mds_updrs_rest_tremor"]
                )
            evidence["cohort"] = row.recording_metadata["cohort"]
            units.append(
                Task2EvaluationUnit(
                    dataset="monipar",
                    subject_id=subject,
                    task_id=task,
                    stream_id=stream,
                    reference_cache_indices=tuple(item.cache_index for item in references),
                    reference_recording_ids=tuple(item.recording_id for item in references),
                    reference_event_indices=tuple(item.event_index for item in references),
                    query_cache_index=row.cache_index,
                    query_recording_id=row.recording_id,
                    query_event_index=row.event_index,
                    role="changed_query" if changed else "accepted_query",
                    relation="different_day",
                    change_evidence=evidence,
                    cell="clinician_rated_change",
                )
            )
    return units, exclusions


def _kneepad_units(cache, *, reference_visits: int) -> list[Task2EvaluationUnit]:
    units: list[Task2EvaluationUnit] = []
    for (subject, task, stream), rows in sorted(_executions(cache, "kneepad").items()):
        rows.sort(key=lambda row: row.recording_id)
        accepted = [row for row in rows if row.event.metadata["accepted"]]
        variants = [row for row in rows if not row.event.metadata["accepted"]]
        if len(accepted) <= reference_visits or not variants:
            continue
        references = accepted[:reference_visits]
        for row in accepted[reference_visits:] + variants:
            trial = str(row.recording_metadata["trial"])
            # Trial names are subject/task-qualified (for example
            # s01_seated_leg_extension_t05). Reliability needs a repeat index
            # shared across subjects, not the full recording identity.
            trial_index = trial.rsplit("_t", 1)[-1]
            if not trial_index.isdigit():
                raise ValueError(f"cannot parse KneE-PAD trial index from {trial!r}")
            units.append(
                Task2EvaluationUnit(
                    dataset="kneepad",
                    subject_id=subject,
                    task_id=task,
                    stream_id=stream,
                    reference_cache_indices=tuple(item.cache_index for item in references),
                    reference_recording_ids=tuple(item.recording_id for item in references),
                    reference_event_indices=tuple(item.event_index for item in references),
                    query_cache_index=row.cache_index,
                    query_recording_id=row.recording_id,
                    query_event_index=row.event_index,
                    role="accepted_query" if row.event.metadata["accepted"] else "changed_query",
                    relation="same_session",
                    cell="known_difference",
                    change_evidence={
                        "execution_variant": row.event.metadata["execution_variant"],
                        "released_label": row.event.metadata["released_label"],
                        "trial": trial,
                        "trial_index": int(trial_index),
                    },
                )
            )
    return units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "manifests"
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--reference-visits", type=int, default=REFERENCE_VISITS)
    args = parser.parse_args()

    datasets = sorted((*TRAIN_ONLY, *EVALUATION))
    caches = {dataset: open_cache(dataset) for dataset in datasets}
    cohort = build_cohort_manifest(
        caches,
        name=COHORT_NAME,
        seed=args.seed,
        train_only_datasets=TRAIN_ONLY,
        evaluation_datasets=EVALUATION,
        # Task 2's unit IS the repetition and it splits by subject, so CrossFit's
        # per-repetition arrays must stay in the cohort.
        exclude_duplicate_views=False,
    )
    cohort_path = args.output_dir / "COHORT_TASK2_V1.json"
    write_cohort_manifest(cohort, cohort_path)
    print(f"cohort: {len(cohort.entries)} entries, {cohort.fingerprint} -> {cohort_path}")

    train_pool = []
    for dataset in TRAIN_ONLY:
        kind = EXECUTION_KINDS[dataset]
        count = sum(
            1
            for recording in caches[dataset]
            for event in recording.events
            if event.annotation_kind == kind
        )
        train_pool.append({"dataset": dataset, "execution_kind": kind, "executions": count})
    train = _manifest(
        name="task2_train_v1",
        task="task2",
        cohort=cohort,
        seed=args.seed,
        protocol={
            "split": "train",
            "unit": "declared pool; episodes are drawn under a seed at run time",
            "sources": train_pool,
            "dataset_weights": DATASET_WEIGHTS,
            "reference_count": args.reference_visits,
            "variant_corpus": "task2_modified_v1",
            "batch_contract": "TASK2_CHANGE_QUANTIFICATION.md section 4",
        },
        units=(),
        exclusions=(),
    )
    write_task_manifest(train, args.output_dir / "TASK2_TRAIN_V1.json")
    print(f"{train.name}: pool {train_pool}, {train.fingerprint}")

    units, exclusions = _monipar_units(
        caches["monipar"], reference_visits=args.reference_visits
    )
    units += _kneepad_units(caches["kneepad"], reference_visits=args.reference_visits)
    test = _manifest(
        name="task2_test_v1",
        task="task2",
        cohort=cohort,
        seed=args.seed,
        protocol={
            "split": "test",
            "reference_rule": (
                f"earliest {args.reference_visits} accepted executions per subject, task and stream"
            ),
            "monipar_reference_rule": (
                f"earliest {args.reference_visits} visits sharing one released clinician "
                "bradykinesia score; unscored visits are excluded"
            ),
            "monipar_change_label": (
                f"released weekly clinician MDS-UPDRS bradykinesia score differs from the "
                f"reference median by >= {CHANGE_MARGIN} point"
            ),
            "monipar_strict_change_label": (
                f"the same, at >= {STRICT_CHANGE_MARGIN} points; reported as a subgroup via "
                f"the per-unit strict_change flag"
            ),
            "kneepad_change_label": "released incorrect execution variant",
            "threshold": "per-person reference-only limit; never fitted on this manifest",
            "reported_separately": ["monipar", "kneepad"],
        },
        units=units,
        exclusions=exclusions,
    )
    write_task_manifest(test, args.output_dir / "TASK2_TEST_V1.json")
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for unit in units:
        counts[(unit.dataset, unit.role)] += 1
    print(f"{test.name}: {len(units)} units {dict(counts)}, {test.fingerprint}")


if __name__ == "__main__":
    main()
