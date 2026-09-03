"""Immutable task-level evaluation units derived from the recording cohort."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from applications.motion_monitoring.data.cache import CachedRecordingDataset
from applications.motion_monitoring.data.compatibility import sensor_compatibility_key
from applications.motion_monitoring.data.contracts import EventInterval, RawRecording, SensorStream
from applications.motion_monitoring.data.manifests import CohortEntry, CohortManifest


TASK_MANIFEST_SCHEMA_VERSION = 2

# Single-execution enrollment bounds (TASK1_REFERENCE_RESOLUTION_SPEC.md section A).
MIN_REFERENCE_SECONDS = 0.3
MAX_REFERENCE_SECONDS = 20.0
EXEMPLAR_TARGET_SECONDS = 2.0
EXEMPLAR_MAX_PERIODS = 3
WEAR_REFERENCE_EXCERPT_SECONDS = 12.0


@dataclass(frozen=True)
class Task1EvaluationUnit:
    dataset: str
    query_cache_index: int
    query_recording_id: str
    query_subject_id: str
    query_stream_id: str
    reference_cache_index: int
    reference_recording_id: str
    reference_subject_id: str
    reference_stream_id: str
    reference_event_index: int
    label: str
    target_intervals_sec: tuple[tuple[float, float], ...]
    target_present: bool
    reference_interval_sec: tuple[float, float]
    reference_rule: str
    query_interval_sec: tuple[float, float] | None = None
    guard_intervals_sec: tuple[tuple[float, float], ...] = ()
    # ``cross_subject``: the reference is another person's execution;
    # ``same_subject``: another recording of the same person (deployment
    # enrollment). Donor-identity sources label the relation by donor subject.
    reference_relation: str = "cross_subject"


@dataclass(frozen=True)
class Task2EvaluationUnit:
    """One frozen Task-2 comparison: a personal reference set and one query.

    Every member is the same person, task and sensor configuration. ``role`` says
    what the query is: ``accepted_query`` is another execution the source calls
    acceptable, ``changed_query`` is one the source labels as different. The
    evidence for that label is carried in ``change_evidence`` so a reader can see
    why, rather than having to trust the flag.
    """

    dataset: str
    subject_id: str
    task_id: str
    stream_id: str
    reference_cache_indices: tuple[int, ...]
    reference_recording_ids: tuple[str, ...]
    reference_event_indices: tuple[int, ...]
    query_cache_index: int
    query_recording_id: str
    query_event_index: int
    role: str
    relation: str
    change_evidence: Mapping[str, Any] = field(default_factory=dict)
    # Which declared analysis this comparison was constructed for. One source can
    # serve more than one: MoniPar's unscored weekly repeats measure between-week
    # stability, while its clinician-scored visits measure responsiveness, and the
    # two need different reference rules. Cells are reported separately and never
    # pooled.
    cell: str = "unspecified"


@dataclass(frozen=True)
class Task3EvaluationUnit:
    dataset: str
    cache_index: int
    recording_id: str
    subject_id: str
    stream_id: str
    annotation_kind: str
    background_labels: tuple[str, ...]
    exhaustive: bool


@dataclass(frozen=True)
class TaskEvaluationManifest:
    schema_version: int
    name: str
    task: Literal["task1", "task2", "task3"]
    cohort_fingerprint: str
    seed: int
    protocol: Mapping[str, Any]
    units: tuple[Mapping[str, Any], ...]
    exclusions: tuple[Mapping[str, Any], ...]
    fingerprint: str


_TEST_CONFIG = {
    "c_mhad": {
        "task1_kind": "event",
        "task3_kind": "event",
        "background": (),
        # Action instances are bounded, but the complement is not published as
        # an exhaustive background annotation track. It must remain unassigned
        # during Task-3 training/evaluation rather than become a negative class.
        "exhaustive": False,
    },
    "wear": {
        "task1_kind": "activity",
        "task3_kind": "activity",
        "background": ("NULL",),
        "exhaustive": True,
        # Sessions are ~50 min and contain nearly every label; whole-session units
        # produced 2 target-absent units in the entire test manifest. Ten-minute
        # blocks yield natural target-absent enrollments (spec section D.1).
        "query_blocks_sec": 600.0,
    },
    "oca": {
        "task1_kind": "sample_label_run",
        "task3_kind": "sample_label_run",
        "background": ("Null",),
        "exhaustive": True,
    },
}

_TASK1_SYNTHETIC_CONFIG = {
    # Spec section C: execution-level targets by construction. Reference
    # identity is the donor clip, not the background subject, so a reference
    # can never be the inserted instance it is asked to find and is preferably
    # performed by a different person than every inserted target.
    "synth_wrist_v1": {
        "task1_kind": "inserted_execution",
        "task1_reference_kind": "enrollment_execution",
        "background": (),
        "exhaustive": False,
        "reference_identity": "donor",
        "reference_only_metadata_key": "task1_reference_only",
    },
}

_DEVELOPMENT_CONFIG = {
    "aidlab_har": {
        "task1_kind": "repetition_fiducial",
        "task3_kind": "repetition_fiducial",
        "background": (),
        "exhaustive": False,
    },
    "crossfit": {
        "task1_kind": "exercise_sequence",
        "task3_kind": "exercise_sequence",
        "background": (),
        "exhaustive": False,
    },
    "openpack": {
        "task1_kind": "fine_action",
        "task3_kind": "fine_action",
        "background": ("Ignore", "Others", "System Error", "Unknown"),
        "exhaustive": False,
    },
    "recofit": {
        "task1_kind": "set",
        "task3_kind": "set",
        "background": (),
        "exhaustive": False,
    },
}


# Task 1 partitions by cohort split, not by dataset, so one config table serves
# every split: a dataset such as OpenPack may hold development and test subjects.
_TASK1_CONFIG = {**_DEVELOPMENT_CONFIG, **_TEST_CONFIG, **_TASK1_SYNTHETIC_CONFIG}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _stream(recording: RawRecording, stream_id: str) -> SensorStream:
    matches = [item for item in recording.streams if item.stream_id == stream_id]
    if len(matches) != 1:
        raise ValueError(f"recording {recording.recording_id!r} has no unique stream {stream_id!r}")
    return matches[0]


def _stream_bounds(stream: SensorStream) -> tuple[float, float]:
    timestamps = stream.timestamps_sec
    if len(timestamps) > 1:
        period = float(np.median(np.diff(timestamps)))
    elif stream.nominal_rate_hz:
        period = 1.0 / float(stream.nominal_rate_hz)
    else:
        period = 0.0
    return float(timestamps[0]), float(timestamps[-1]) + period


def _intervals_inside_stream(
    intervals: Sequence[tuple[float, float]], stream: SensorStream
) -> bool:
    start, end = _stream_bounds(stream)
    return all(left >= start - 1e-6 and right <= end + 1e-6 for left, right in intervals)


def _interval_valid_fraction(
    stream: SensorStream, start_sec: float, end_sec: float
) -> float:
    """Fraction of samples inside the interval with every channel valid."""

    timestamps = np.asarray(stream.timestamps_sec)
    left = int(np.searchsorted(timestamps, start_sec, side="left"))
    right = int(np.searchsorted(timestamps, end_sec, side="left"))
    if right <= left:
        return 0.0
    return float(np.asarray(stream.valid)[left:right].all(axis=1).mean())


MIN_REFERENCE_VALID_FRACTION = 0.8


def _config_key(stream: SensorStream):
    return sensor_compatibility_key(
        device=stream.device,
        placement=stream.placement,
        channels=stream.channels,
        gravity_state=stream.gravity_state,
    )


def _events(recording: RawRecording, *, kind: str, background: set[str]) -> list[tuple[int, EventInterval]]:
    return [
        (index, event)
        for index, event in enumerate(recording.events)
        if event.annotation_kind == kind
        and event.label not in background
        and not bool(event.metadata.get("clipped_by_recording_crop", False))
    ]


def _event_seam_guards(events: Sequence[EventInterval]) -> tuple[tuple[float, float], ...]:
    """Return join regions that may be aligned through but never supervised."""

    guards: list[tuple[float, float]] = []
    for event in events:
        guard = float(event.metadata.get("guard_sec", 0.0) or 0.0)
        if guard <= 0:
            continue
        guards.extend(
            [
                (float(event.start_sec) - guard, float(event.start_sec) + guard),
                (float(event.end_sec) - guard, float(event.end_sec) + guard),
            ]
        )
    return tuple(guards)


def _block_targets_and_guards(
    targets_by_label: Mapping[str, tuple[tuple[float, float], ...]],
    events: Sequence[EventInterval],
    block: tuple[float, float] | None,
) -> tuple[
    dict[str, tuple[tuple[float, float], ...]],
    tuple[tuple[float, float], ...],
]:
    """Keep only complete targets and mask source events cut by a query block."""

    seam_guards = _event_seam_guards(events)
    if block is None:
        return dict(targets_by_label), seam_guards
    block_start, block_end = map(float, block)
    complete = {
        label: tuple(
            (left, right)
            for left, right in targets
            if left >= block_start - 1e-6 and right <= block_end + 1e-6
        )
        for label, targets in targets_by_label.items()
    }
    crossing = tuple(
        (max(float(event.start_sec), block_start), min(float(event.end_sec), block_end))
        for event in events
        if event.start_sec < block_end
        and event.end_sec > block_start
        and not (
            event.start_sec >= block_start - 1e-6
            and event.end_sec <= block_end + 1e-6
        )
    )
    clipped_seams = tuple(
        (max(left, block_start), min(right, block_end))
        for left, right in seam_guards
        if min(right, block_end) > max(left, block_start)
    )
    return complete, crossing + clipped_seams


def single_execution_interval(
    dataset: str, recording: RawRecording, event: EventInterval
) -> tuple[tuple[float, float], str]:
    """Derive a one-short-execution enrollment interval from a source annotation.

    Implements TASK1_REFERENCE_RESOLUTION_SPEC.md section A. Raises ``ValueError``
    with the exclusion reason when no defensible single execution exists.
    """

    start = float(event.start_sec)
    end = float(event.end_sec)
    duration = end - start
    rule = "source_event"
    if dataset == "recofit":
        count = event.metadata.get("repetition_count")
        if isinstance(count, (int, float)) and count and count >= 1:
            period = duration / float(count)
            periods = min(
                EXEMPLAR_MAX_PERIODS,
                max(1, int(np.ceil(EXEMPLAR_TARGET_SECONDS / max(period, 1e-6)))),
            )
            end = start + min(periods * period, duration)
            rule = f"exemplar_prefix_{periods}x{period:.2f}s"
        else:
            end = start + min(duration, MAX_REFERENCE_SECONDS)
            rule = "leading_no_count"
    elif dataset == "crossfit":
        repetition_indices = recording.metadata.get("available_repetition_indices") or ()
        count = len(repetition_indices)
        if count >= 1:
            period = duration / float(count)
            periods = min(
                EXEMPLAR_MAX_PERIODS,
                max(1, int(np.ceil(EXEMPLAR_TARGET_SECONDS / max(period, 1e-6)))),
            )
            end = start + min(periods * period, duration)
            rule = f"exemplar_prefix_{periods}x{period:.2f}s"
        else:
            end = start + min(duration, MAX_REFERENCE_SECONDS)
            rule = "leading_no_count"
    elif dataset == "wear":
        end = start + min(duration, WEAR_REFERENCE_EXCERPT_SECONDS)
        rule = "leading_excerpt" if end - start < duration else "source_event"
    derived = end - start
    if derived > MAX_REFERENCE_SECONDS:
        end = start + MAX_REFERENCE_SECONDS
        rule = f"{rule}+capped"
        derived = MAX_REFERENCE_SECONDS
    if derived < MIN_REFERENCE_SECONDS:
        raise ValueError(
            f"reference below the {MIN_REFERENCE_SECONDS}s duration gate ({derived:.3f}s)"
        )
    return (start, end), rule


REFERENCE_RELATIONS = ("cross_subject", "same_subject")


def _relation_trial_key(unit: Task1EvaluationUnit) -> tuple[object, ...]:
    """Identify a query-label trial independently of its enrolled reference."""

    return (
        unit.dataset,
        unit.query_recording_id,
        unit.query_stream_id,
        unit.query_interval_sec,
        unit.label,
        unit.target_present,
    )


def _retain_complete_relation_sets(
    units: Sequence[Task1EvaluationUnit], relations: Sequence[str]
) -> tuple[list[Task1EvaluationUnit], list[dict[str, Any]]]:
    """Keep only trials available under every requested subject relation.

    This makes a same-subject versus cross-subject comparison paired: a relation
    can change the enrolled reference, but cannot change the query, label, or
    target condition being measured.
    """

    required = frozenset(relations)
    grouped: dict[tuple[object, ...], list[Task1EvaluationUnit]] = defaultdict(list)
    for unit in units:
        grouped[_relation_trial_key(unit)].append(unit)
    retained: list[Task1EvaluationUnit] = []
    exclusions: list[dict[str, Any]] = []
    for key, group in grouped.items():
        observed = {unit.reference_relation for unit in group}
        if observed == required and len(group) == len(required):
            retained.extend(group)
            continue
        dataset, recording_id, stream_id, interval, label, target_present = key
        exclusions.append(
            {
                "dataset": dataset,
                "recording_id": recording_id,
                "stream_id": stream_id,
                "query_interval_sec": interval,
                "label": label,
                "target_present": target_present,
                "reason": "incomplete reference-relation set",
                "available_relations": sorted(observed),
                "required_relations": sorted(required),
            }
        )
    return retained, exclusions


def _choose_reference(
    candidates: Sequence[tuple[CohortEntry, str, int, EventInterval]],
    *,
    query: CohortEntry,
    label: str,
    seed: int,
    excluded_identities: frozenset[str] = frozenset(),
    target_subjects: frozenset[str] | None = None,
    relation: str = "cross_subject",
) -> tuple[CohortEntry, str, int, EventInterval] | None:
    """Pick one reference under the declared subject relation.

    ``cross_subject`` draws from other people only; ``same_subject`` draws from
    other recordings of the same person only (no fallback between the two, so a
    unit's relation is exact). ``excluded_identities``/``target_subjects`` carry
    donor identities for the synthetic corpus, where the recording's subject is
    the background wearer and the performed execution belongs to the donor clip;
    there the draw prefers a different donor subject and falls back to any
    independent donor clip.
    """

    if relation not in REFERENCE_RELATIONS:
        raise ValueError(f"unknown reference relation: {relation!r}")
    independent = [item for item in candidates if item[0].recording_id != query.recording_id]
    if excluded_identities:
        independent = [
            item
            for item in independent
            if str(item[3].metadata.get("donor_clip_id")) not in excluded_identities
        ]
    if not independent:
        return None
    if target_subjects is None:
        if relation == "cross_subject":
            pool = [item for item in independent if item[0].subject_id != query.subject_id]
        else:
            pool = [item for item in independent if item[0].subject_id == query.subject_id]
    else:
        cross_subject = [
            item
            for item in independent
            if str(item[3].metadata.get("donor_subject_id")) not in target_subjects
        ]
        pool = cross_subject or independent
    if not pool:
        return None
    return min(
        pool,
        key=lambda item: sha256(
            f"{seed}:{query.dataset}:{query.recording_id}:{label}:"
            f"{item[0].recording_id}:{item[1]}:{item[2]}".encode("utf-8")
        ).hexdigest(),
    )


def _manifest(
    *,
    name: str,
    task: Literal["task1", "task2", "task3"],
    cohort: CohortManifest,
    seed: int,
    protocol: Mapping[str, Any],
    units: Sequence[Any],
    exclusions: Sequence[Mapping[str, Any]],
) -> TaskEvaluationManifest:
    rows = [asdict(item) if hasattr(item, "__dataclass_fields__") else dict(item) for item in units]
    unsigned = {
        "schema_version": TASK_MANIFEST_SCHEMA_VERSION,
        "name": name,
        "task": task,
        "cohort_fingerprint": cohort.fingerprint,
        "seed": int(seed),
        "protocol": dict(protocol),
        "units": rows,
        "exclusions": [dict(item) for item in exclusions],
    }
    fingerprint = sha256(_canonical_json(unsigned)).hexdigest()
    return TaskEvaluationManifest(
        schema_version=TASK_MANIFEST_SCHEMA_VERSION,
        name=name,
        task=task,
        cohort_fingerprint=cohort.fingerprint,
        seed=int(seed),
        protocol=dict(protocol),
        units=tuple(rows),
        exclusions=tuple(dict(item) for item in exclusions),
        fingerprint=fingerprint,
    )


def _build_task1_manifest(
    cohort: CohortManifest,
    caches: Mapping[str, CachedRecordingDataset],
    *,
    split: Literal["train", "development", "test"],
    configs: Mapping[str, Mapping[str, Any]],
    name: str,
    seed: int = 20260831,
    relations: Sequence[str] = ("cross_subject",),
) -> TaskEvaluationManifest:
    """Pair every recording with present and matched-count absent enrollments.

    One unit is emitted per (query, label, relation). Donor-identity sources
    ignore ``relations`` and label each unit by the drawn donor's subject.
    """

    relations = tuple(relations)
    if not relations or any(item not in REFERENCE_RELATIONS for item in relations):
        raise ValueError(f"relations must be a non-empty subset of {REFERENCE_RELATIONS}")

    entries = tuple(entry for entry in cohort.entries if entry.split == split)
    references: dict[
        tuple[str, object, str],
        list[tuple[CohortEntry, str, int, EventInterval, tuple[float, float], str]],
    ] = defaultdict(list)
    labels_by_config: dict[tuple[str, object], set[str]] = defaultdict(set)
    gate_exclusions: dict[tuple[str, str], int] = defaultdict(int)
    for entry in entries:
        recording = caches[entry.dataset][entry.cache_index]
        config = configs[entry.dataset]
        events = _events(
            recording,
            kind=str(config.get("task1_reference_kind", config["task1_kind"])),
            background=set(config["background"]),
        )
        for stream_id in entry.stream_ids:
            stream = _stream(recording, stream_id)
            key = (entry.dataset, _config_key(stream))
            for event_index, event in events:
                if not _intervals_inside_stream(
                    ((float(event.start_sec), float(event.end_sec)),), stream
                ):
                    continue
                try:
                    interval, rule = single_execution_interval(
                        entry.dataset, recording, event
                    )
                except ValueError:
                    gate_exclusions[(entry.dataset, event.label)] += 1
                    continue
                if (
                    _interval_valid_fraction(stream, *interval)
                    < MIN_REFERENCE_VALID_FRACTION
                ):
                    # A reference inside a dead-sensor region (for example the
                    # missing WEAR sbj_10 left-arm rows) cannot enroll anything.
                    gate_exclusions[(entry.dataset, event.label)] += 1
                    continue
                references[(*key, event.label)].append(
                    (entry, stream_id, event_index, event, interval, rule)
                )
                labels_by_config[key].add(event.label)

    units: list[Task1EvaluationUnit] = []
    exclusions: list[dict[str, Any]] = []
    for entry in entries:
        recording = caches[entry.dataset][entry.cache_index]
        config = configs[entry.dataset]
        reference_only_key = config.get("reference_only_metadata_key")
        if reference_only_key and bool(recording.metadata.get(reference_only_key)):
            continue
        background = set(config["background"])
        kind = str(config["task1_kind"])
        block_seconds = config.get("query_blocks_sec")
        present_events = _events(recording, kind=kind, background=background)
        present_labels = sorted({event.label for _, event in present_events})
        targets_by_label = {
            label: tuple(
                (float(event.start_sec), float(event.end_sec))
                for _, event in present_events
                if event.label == label
            )
            for label in present_labels
        }
        for stream_id in entry.stream_ids:
            stream = _stream(recording, stream_id)
            key = (entry.dataset, _config_key(stream))
            stream_targets_by_label = {
                label: tuple(
                    interval
                    for interval in targets
                    if _intervals_inside_stream((interval,), stream)
                )
                for label, targets in targets_by_label.items()
            }
            for block_index, block in enumerate(
                _query_blocks(stream, block_seconds)
            ):
                block_targets_by_label, block_guards = _block_targets_and_guards(
                    stream_targets_by_label,
                    [event for _, event in present_events],
                    block,
                )
                block_present_labels = sorted(
                    label
                    for label, targets in block_targets_by_label.items()
                    if targets
                )
                absent_labels = sorted(
                    labels_by_config[key] - set(block_present_labels)
                )
                desired_absent = min(
                    len(absent_labels), max(1, len(block_present_labels))
                )
                absent_labels = sorted(
                    absent_labels,
                    key=lambda label: sha256(
                        f"{seed}:{entry.dataset}:{entry.recording_id}:{stream_id}:"
                        f"{block_index}:{label}".encode("utf-8")
                    ).hexdigest(),
                )[:desired_absent]
                for label in (*block_present_labels, *absent_labels):
                    label_events = [
                        event for _, event in present_events if event.label == label
                    ]
                    donor_identity = config.get("reference_identity") == "donor"
                    target_subjects = frozenset(
                        str(event.metadata.get("donor_subject_id"))
                        for event in label_events
                    )
                    for relation in (("donor",) if donor_identity else relations):
                        if donor_identity:
                            reference = _choose_reference(
                                references.get((*key, label), ()),
                                query=entry,
                                label=label,
                                seed=seed,
                                excluded_identities=frozenset(
                                    str(event.metadata.get("donor_clip_id"))
                                    for event in label_events
                                ),
                                target_subjects=target_subjects,
                            )
                        else:
                            reference = _choose_reference(
                                references.get((*key, label), ()),
                                query=entry,
                                label=label,
                                seed=seed,
                                relation=relation,
                            )
                        if reference is None:
                            exclusions.append(
                                {
                                    "dataset": entry.dataset,
                                    "recording_id": entry.recording_id,
                                    "stream_id": stream_id,
                                    "label": label,
                                    "relation": relation,
                                    "reason": "no independent compatible reference",
                                }
                            )
                            continue
                        (
                            reference_entry,
                            reference_stream,
                            event_index,
                            reference_event,
                            reference_interval,
                            reference_rule,
                        ) = reference
                        if donor_identity:
                            unit_relation = (
                                "same_subject"
                                if str(reference_event.metadata.get("donor_subject_id"))
                                in target_subjects
                                else "cross_subject"
                            )
                        else:
                            unit_relation = relation
                        targets = block_targets_by_label.get(label, ())
                        units.append(
                            Task1EvaluationUnit(
                                dataset=entry.dataset,
                                query_cache_index=entry.cache_index,
                                query_recording_id=entry.recording_id,
                                query_subject_id=entry.subject_id,
                                query_stream_id=stream_id,
                                reference_cache_index=reference_entry.cache_index,
                                reference_recording_id=reference_entry.recording_id,
                                reference_subject_id=reference_entry.subject_id,
                                reference_stream_id=reference_stream,
                                reference_event_index=event_index,
                                label=label,
                                target_intervals_sec=targets,
                                target_present=bool(targets),
                                reference_interval_sec=reference_interval,
                                reference_rule=reference_rule,
                                query_interval_sec=block,
                                guard_intervals_sec=block_guards,
                                reference_relation=unit_relation,
                            )
                        )
    # When reporting several subject-reference relations, retain the common
    # query-label trials only. Otherwise relation columns would differ in their
    # mix of labels and target-absent timelines, confounding the comparison.
    if len(relations) > 1:
        units, relation_exclusions = _retain_complete_relation_sets(units, relations)
        exclusions.extend(relation_exclusions)

    units.sort(
        key=lambda item: (
            item.dataset,
            item.query_recording_id,
            item.query_stream_id,
            item.query_interval_sec or (float("-inf"), float("-inf")),
            item.label,
            item.reference_relation,
        )
    )
    for (dataset, label), count in sorted(gate_exclusions.items()):
        exclusions.append(
            {
                "dataset": dataset,
                "label": label,
                "reason": "reference below duration gate",
                "count": count,
            }
        )
    return _manifest(
        name=name,
        task="task1",
        cohort=cohort,
        seed=seed,
        protocol={
            "split": split,
            "reference_policy": (
                "cross_subject: another person; same_subject: another recording of the same "
                "person; donor sources: prefer a different donor subject, else any independent clip"
            ),
            "reference_relations": list(relations),
            "relation_comparison": (
                "paired query-label trials across every declared relation"
                if len(relations) > 1
                else "single declared relation"
            ),
            "reference_identity": {
                dataset: config.get("reference_identity", "recording_subject")
                for dataset, config in sorted(configs.items())
            },
            "reference_semantics": "one short execution per TASK1_REFERENCE_RESOLUTION_SPEC.md section A",
            "reference_bounds_sec": [MIN_REFERENCE_SECONDS, MAX_REFERENCE_SECONDS],
            "positive_labels": "all labels present in each query recording (or block)",
            "absent_labels": "deterministic matched count up to number of present labels",
            "query_blocks_sec": {
                dataset: config.get("query_blocks_sec")
                for dataset, config in sorted(configs.items())
            },
            "sensor_compatibility": "exact device family, placement, channels, gravity state",
            "guard_semantics": "query-valid for alignment; excluded from endpoint supervision",
        },
        units=units,
        exclusions=exclusions,
    )


def _query_blocks(
    stream: SensorStream, block_seconds: float | None
) -> list[tuple[float, float] | None]:
    """Non-overlapping query blocks; the trailing partial kept when >= half a block."""

    if block_seconds is None:
        return [None]
    if block_seconds <= 0:
        raise ValueError("query_blocks_sec must be positive when configured")
    start, end = _stream_bounds(stream)
    blocks: list[tuple[float, float] | None] = []
    cursor = start
    while cursor < end - 1e-9:
        block_end = min(cursor + block_seconds, end)
        if block_end - cursor >= block_seconds / 2:
            blocks.append((cursor, block_end))
        cursor += block_seconds
    return blocks or [None]


def build_task1_test_manifest(
    cohort: CohortManifest,
    caches: Mapping[str, CachedRecordingDataset],
    *,
    seed: int = 20260831,
    name: str = "task1_test_v1",
    relations: Sequence[str] = REFERENCE_RELATIONS,
) -> TaskEvaluationManifest:
    """Sealed evaluation units under every declared subject relation.

    ``same_subject`` is the deployment condition (a person enrolls their own
    execution); ``cross_subject`` is the generalisation condition. Both are
    reported as separate columns; neither is ever tuned on.
    """

    return _build_task1_manifest(
        cohort,
        caches,
        split="test",
        configs=_TASK1_CONFIG,
        name=name,
        seed=seed,
        relations=relations,
    )


def build_task1_train_manifest(
    cohort: CohortManifest,
    caches: Mapping[str, CachedRecordingDataset],
    *,
    seed: int = 20260831,
    name: str = "task1_train_v1",
) -> TaskEvaluationManifest:
    return _build_task1_manifest(
        cohort,
        caches,
        split="train",
        configs=_TASK1_CONFIG,
        name=name,
        seed=seed,
    )


def _build_task3_manifest(
    cohort: CohortManifest,
    *,
    split: Literal["train", "development", "test"],
    configs: Mapping[str, Mapping[str, Any]],
    name: str,
    seed: int = 20260831,
) -> TaskEvaluationManifest:
    units: list[Task3EvaluationUnit] = []
    for entry in cohort.entries:
        if entry.split != split:
            continue
        config = configs[entry.dataset]
        for stream_id in entry.stream_ids:
            units.append(
                Task3EvaluationUnit(
                    dataset=entry.dataset,
                    cache_index=entry.cache_index,
                    recording_id=entry.recording_id,
                    subject_id=entry.subject_id,
                    stream_id=stream_id,
                    annotation_kind=str(config["task3_kind"]),
                    background_labels=tuple(config["background"]),
                    exhaustive=bool(config["exhaustive"]),
                )
            )
    units.sort(key=lambda item: (item.dataset, item.recording_id, item.stream_id))
    return _manifest(
        name=name,
        task="task3",
        cohort=cohort,
        seed=seed,
        protocol={
            "split": split,
            "candidate_source": "complete recording timeline",
            "labels_hidden_from_model": True,
            "reporting": "per dataset; no pooled headline",
        },
        units=units,
        exclusions=(),
    )


def build_task3_test_manifest(
    cohort: CohortManifest,
    *,
    seed: int = 20260831,
) -> TaskEvaluationManifest:
    return _build_task3_manifest(
        cohort,
        split="test",
        configs=_TEST_CONFIG,
        name="task3_test_v1",
        seed=seed,
    )


def build_task3_development_manifest(
    cohort: CohortManifest,
    *,
    seed: int = 20260831,
) -> TaskEvaluationManifest:
    return _build_task3_manifest(
        cohort,
        split="development",
        configs=_DEVELOPMENT_CONFIG,
        name="task3_development_v1",
        seed=seed,
    )


def build_task3_train_manifest(
    cohort: CohortManifest,
    *,
    seed: int = 20260831,
) -> TaskEvaluationManifest:
    return _build_task3_manifest(
        cohort,
        split="train",
        configs=_DEVELOPMENT_CONFIG,
        name="task3_train_v1",
        seed=seed,
    )


def write_task_manifest(manifest: TaskEvaluationManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_task_manifest(path: Path) -> TaskEvaluationManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = payload.pop("fingerprint", None)
    expected = sha256(_canonical_json(payload)).hexdigest()
    if fingerprint != expected:
        raise ValueError(f"task manifest fingerprint mismatch in {path}")
    if payload.get("schema_version") != TASK_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported task manifest schema in {path}")
    units = tuple(payload.pop("units"))
    exclusions = tuple(payload.pop("exclusions"))
    return TaskEvaluationManifest(
        **payload,
        units=units,
        exclusions=exclusions,
        fingerprint=fingerprint,
    )


def validate_task_manifest(
    manifest: TaskEvaluationManifest,
    cohort: CohortManifest,
    caches: Mapping[str, CachedRecordingDataset],
) -> None:
    """Resolve every persisted identity against the current immutable caches."""

    if manifest.cohort_fingerprint != cohort.fingerprint:
        raise ValueError("task manifest belongs to a different cohort")
    split = manifest.protocol.get("split")
    if manifest.task in {"task1", "task3"} and split not in {
        "train",
        "development",
        "test",
    }:
        raise ValueError("task manifest must declare development or test split")
    configs = _TASK1_CONFIG if manifest.task == "task1" else (
        _TEST_CONFIG if split == "test" else _DEVELOPMENT_CONFIG
    )
    cohort_keys = {
        (entry.dataset, entry.cache_index, entry.recording_id, entry.split)
        for entry in cohort.entries
    }
    if manifest.task == "task1":
        for row in manifest.units:
            unit = Task1EvaluationUnit(**row)
            query_key = (
                unit.dataset,
                unit.query_cache_index,
                unit.query_recording_id,
                split,
            )
            reference_key = (
                unit.dataset,
                unit.reference_cache_index,
                unit.reference_recording_id,
                split,
            )
            if query_key not in cohort_keys or reference_key not in cohort_keys:
                raise ValueError("Task-1 unit points outside its declared cohort split")
            cache = caches[unit.dataset]
            query = cache[unit.query_cache_index]
            reference = cache[unit.reference_cache_index]
            if (
                query.subject_id != unit.query_subject_id
                or reference.subject_id != unit.reference_subject_id
            ):
                raise ValueError("Task-1 subject identity no longer matches its cache")
            query_stream = _stream(query, unit.query_stream_id)
            reference_stream = _stream(reference, unit.reference_stream_id)
            if _config_key(query_stream) != _config_key(reference_stream):
                raise ValueError("Task-1 reference and query sensor configurations differ")
            try:
                reference_event = reference.events[unit.reference_event_index]
            except IndexError as error:
                raise ValueError("Task-1 reference event index is invalid") from error
            if reference_event.label != unit.label:
                raise ValueError("Task-1 reference label no longer matches its event")
            expected_interval, expected_rule = single_execution_interval(
                unit.dataset, reference, reference_event
            )
            if (
                tuple(map(float, unit.reference_interval_sec)) != expected_interval
                or unit.reference_rule != expected_rule
            ):
                raise ValueError(
                    "Task-1 reference enrollment interval no longer matches its derivation rule"
                )
            config = configs[unit.dataset]
            if unit.reference_relation not in REFERENCE_RELATIONS:
                raise ValueError("Task-1 unit declares an unknown reference relation")
            if config.get("reference_identity") != "donor":
                same_person = unit.reference_subject_id == unit.query_subject_id
                if unit.reference_relation == "same_subject" and (
                    not same_person
                    or unit.reference_recording_id == unit.query_recording_id
                ):
                    raise ValueError(
                        "Task-1 same-subject reference must be another recording of the same person"
                    )
                if unit.reference_relation == "cross_subject" and same_person:
                    raise ValueError("Task-1 cross-subject reference must be another person")
            query_events = _events(
                query,
                kind=str(config["task1_kind"]),
                background=set(config["background"]),
            )
            targets_by_label = {
                label: tuple(
                    (float(event.start_sec), float(event.end_sec))
                    for _, event in query_events
                    if event.label == label
                    and _intervals_inside_stream(
                        ((float(event.start_sec), float(event.end_sec)),), query_stream
                    )
                )
                for label in {event.label for _, event in query_events}
            }
            if unit.query_interval_sec is not None:
                block_start, block_end = map(float, unit.query_interval_sec)
                if block_end <= block_start:
                    raise ValueError("Task-1 query block is invalid")
            blocked_targets, expected_guards = _block_targets_and_guards(
                targets_by_label,
                [event for _, event in query_events],
                unit.query_interval_sec,
            )
            expected = blocked_targets.get(unit.label, ())
            if expected != tuple(map(tuple, unit.target_intervals_sec)):
                raise ValueError("Task-1 target intervals no longer match source annotations")
            if unit.target_present != bool(expected):
                raise ValueError("Task-1 target-presence flag is inconsistent")
            if expected_guards != tuple(map(tuple, unit.guard_intervals_sec)):
                raise ValueError("Task-1 guard intervals no longer match source annotations")
    elif manifest.task == "task3":
        for row in manifest.units:
            unit = Task3EvaluationUnit(**row)
            if (
                unit.dataset,
                unit.cache_index,
                unit.recording_id,
                split,
            ) not in cohort_keys:
                raise ValueError("Task-3 unit points outside its declared cohort split")
            recording = caches[unit.dataset][unit.cache_index]
            if recording.subject_id != unit.subject_id:
                raise ValueError("Task-3 subject identity no longer matches its cache")
            _stream(recording, unit.stream_id)
            config = configs[unit.dataset]
            expected = (
                str(config["task3_kind"]),
                tuple(config["background"]),
                bool(config["exhaustive"]),
            )
            observed = (
                unit.annotation_kind,
                tuple(unit.background_labels),
                unit.exhaustive,
            )
            if observed != expected:
                raise ValueError("Task-3 annotation contract is stale")
    elif manifest.task == "task2":
        # A RawRecording owns memory-mapped arrays. Retaining one object per cache
        # row eventually exhausts the process file-descriptor limit on KneE-PAD.
        # Cache only the small immutable fields needed by this validator so each
        # RawRecording (and its mmap handles) can be released immediately.
        seen: dict[
            tuple[str, int],
            tuple[str, str, dict[str, object], tuple[tuple[str, str, dict[str, Any]], ...]],
        ] = {}

        def recording_at(
            dataset: str, index: int
        ) -> tuple[
            str,
            str,
            dict[str, object],
            tuple[tuple[str, str, dict[str, Any]], ...],
        ]:
            key = (dataset, index)
            if key not in seen:
                recording = caches[dataset][index]
                seen[key] = (
                    recording.recording_id,
                    recording.subject_id,
                    {
                        stream.stream_id: _config_key(stream)
                        for stream in recording.streams
                    },
                    tuple(
                        (event.label, event.annotation_kind, dict(event.metadata))
                        for event in recording.events
                    ),
                )
            return seen[key]

        for row in manifest.units:
            unit = Task2EvaluationUnit(**row)
            if unit.role not in {"accepted_query", "changed_query"}:
                raise ValueError(f"Task-2 unit declares an unknown role: {unit.role!r}")
            if not unit.cell or unit.cell == "unspecified":
                raise ValueError("a Task-2 unit must declare which analysis cell it serves")
            query_id, query_subject, query_streams, query_events = recording_at(
                unit.dataset, unit.query_cache_index
            )
            if query_id != unit.query_recording_id:
                raise ValueError("Task-2 query recording no longer matches its cache")
            if query_subject != unit.subject_id:
                raise ValueError("Task-2 subject identity no longer matches its cache")
            if not (
                len(unit.reference_cache_indices)
                == len(unit.reference_recording_ids)
                == len(unit.reference_event_indices)
            ):
                raise ValueError("Task-2 reference identities are inconsistent")
            if unit.query_cache_index in unit.reference_cache_indices:
                raise ValueError("a Task-2 query may not appear in its own reference set")
            try:
                query_config = query_streams[unit.stream_id]
            except KeyError as error:
                raise ValueError(
                    f"Task-2 query recording has no stream {unit.stream_id!r}"
                ) from error
            try:
                query_label, query_kind, query_metadata = query_events[unit.query_event_index]
            except IndexError as error:
                raise ValueError("Task-2 query event index is stale") from error
            if query_label != unit.task_id or query_kind != "bounded_execution":
                raise ValueError("Task-2 query event no longer matches its declared task")
            reference_metadata: list[dict[str, Any]] = []
            for index, recording_id in zip(
                unit.reference_cache_indices, unit.reference_recording_ids
            ):
                reference_id, reference_subject, reference_streams, _ = recording_at(
                    unit.dataset, index
                )
                if reference_id != recording_id:
                    raise ValueError("Task-2 reference recording no longer matches its cache")
                if reference_subject != unit.subject_id:
                    raise ValueError("Task-2 reference belongs to another subject")
                try:
                    reference_config = reference_streams[unit.stream_id]
                except KeyError as error:
                    raise ValueError(
                        f"Task-2 reference recording has no stream {unit.stream_id!r}"
                    ) from error
                if reference_config != query_config:
                    raise ValueError("Task-2 reference and query sensor configurations differ")
            for index, event_index in zip(
                unit.reference_cache_indices, unit.reference_event_indices
            ):
                _, _, _, events = recording_at(unit.dataset, index)
                try:
                    label, kind, metadata = events[event_index]
                except IndexError as error:
                    raise ValueError("Task-2 reference event index is stale") from error
                if label != unit.task_id or kind != "bounded_execution":
                    raise ValueError("Task-2 reference no longer matches its declared task")
                reference_metadata.append(metadata)
            if unit.dataset == "monipar" and unit.cell == "between_week_reliability":
                # This cell measures the between-week noise floor over weekly
                # repeats, which are accepted by construction. It carries no
                # clinical label and must never infer one, so the clinician-score
                # rules below do not apply to it.
                if unit.relation != "different_day":
                    raise ValueError("MoniPar comparisons must be between visits")
                if unit.role != "accepted_query":
                    raise ValueError(
                        "the MoniPar reliability cell never labels a query as changed"
                    )
                if "week" not in unit.change_evidence:
                    raise ValueError("a MoniPar reliability unit must record its visit week")
            elif unit.dataset == "monipar":
                if unit.cell != "clinician_rated_change":
                    raise ValueError(f"unknown MoniPar analysis cell: {unit.cell!r}")
                if unit.relation != "different_day":
                    raise ValueError("MoniPar comparisons must be between visits")
                if "mds_updrs_bradykinesia" not in query_metadata:
                    raise ValueError("MoniPar query has no clinician score")
                scores = [
                    int(metadata["mds_updrs_bradykinesia"])
                    for metadata in reference_metadata
                    if "mds_updrs_bradykinesia" in metadata
                ]
                if len(scores) != len(reference_metadata) or len(set(scores)) != 1:
                    raise ValueError("MoniPar references do not define one stable clinician score")
                margin = abs(int(query_metadata["mds_updrs_bradykinesia"]) - scores[0])
                expected_role = "changed_query" if margin >= 1 else "accepted_query"
                evidence = unit.change_evidence
                if (
                    unit.role != expected_role
                    or evidence.get("score_margin") != margin
                    or evidence.get("mds_updrs_bradykinesia")
                    != int(query_metadata["mds_updrs_bradykinesia"])
                    or tuple(evidence.get("reference_scores", ())) != tuple(scores)
                    or evidence.get("strict_change") != (margin >= 2)
                ):
                    raise ValueError("MoniPar role no longer matches its clinician score")
            elif unit.dataset == "kneepad":
                if unit.cell != "known_difference":
                    raise ValueError(f"unknown KneE-PAD analysis cell: {unit.cell!r}")
                if unit.relation != "same_session":
                    raise ValueError("KneE-PAD comparisons must remain within one visit")
                expected_role = (
                    "accepted_query" if bool(query_metadata.get("accepted")) else "changed_query"
                )
                if unit.role != expected_role:
                    raise ValueError("KneE-PAD role no longer matches its released variant")
                if not all(bool(metadata.get("accepted")) for metadata in reference_metadata):
                    raise ValueError("KneE-PAD reference set contains an incorrect execution")
                trial = query_id.removeprefix("kneepad:")
                trial_index = trial.rsplit("_t", 1)[-1]
                if (
                    not trial_index.isdigit()
                    or unit.change_evidence.get("trial") != trial
                    or unit.change_evidence.get("trial_index") != int(trial_index)
                ):
                    raise ValueError("KneE-PAD trial identity is stale")
    else:
        raise ValueError(f"unknown task manifest: {manifest.task!r}")
