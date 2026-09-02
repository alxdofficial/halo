"""Immutable task-level evaluation units derived from the recording cohort."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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
        "background": (),
        "exhaustive": False,
        "reference_identity": "donor",
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


def _choose_reference(
    candidates: Sequence[tuple[CohortEntry, str, int, EventInterval]],
    *,
    query: CohortEntry,
    label: str,
    seed: int,
    excluded_identities: frozenset[str] = frozenset(),
    target_subjects: frozenset[str] | None = None,
) -> tuple[CohortEntry, str, int, EventInterval] | None:
    """Pick one reference, preferring a different person than the targets.

    ``excluded_identities``/``target_subjects`` carry donor identities for the
    synthetic corpus, where the recording's subject is the background wearer and
    the performed execution belongs to the donor clip instead.
    """

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
        cross_subject = [item for item in independent if item[0].subject_id != query.subject_id]
    else:
        cross_subject = [
            item
            for item in independent
            if str(item[3].metadata.get("donor_subject_id")) not in target_subjects
        ]
    pool = cross_subject or independent
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
) -> TaskEvaluationManifest:
    """Pair every recording with present and matched-count absent enrollments."""

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
            kind=str(config["task1_kind"]),
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
                if block is None:
                    block_targets_by_label = stream_targets_by_label
                else:
                    block_start, block_end = block
                    block_targets_by_label = {
                        label: tuple(
                            (max(left, block_start), min(right, block_end))
                            for left, right in targets
                            if min(right, block_end) > max(left, block_start)
                        )
                        for label, targets in stream_targets_by_label.items()
                    }
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
                    if config.get("reference_identity") == "donor":
                        label_events = [
                            event for _, event in present_events if event.label == label
                        ]
                        reference = _choose_reference(
                            references.get((*key, label), ()),
                            query=entry,
                            label=label,
                            seed=seed,
                            excluded_identities=frozenset(
                                str(event.metadata.get("donor_clip_id"))
                                for event in label_events
                            ),
                            target_subjects=frozenset(
                                str(event.metadata.get("donor_subject_id"))
                                for event in label_events
                            ),
                        )
                    else:
                        reference = _choose_reference(
                            references.get((*key, label), ()),
                            query=entry,
                            label=label,
                            seed=seed,
                        )
                    if reference is None:
                        exclusions.append(
                            {
                                "dataset": entry.dataset,
                                "recording_id": entry.recording_id,
                                "stream_id": stream_id,
                                "label": label,
                                "reason": "no independent compatible reference",
                            }
                        )
                        continue
                    (
                        reference_entry,
                        reference_stream,
                        event_index,
                        _,
                        reference_interval,
                        reference_rule,
                    ) = reference
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
                        )
                    )
    units.sort(
        key=lambda item: (
            item.dataset,
            item.query_recording_id,
            item.query_stream_id,
            item.query_interval_sec or (float("-inf"), float("-inf")),
            item.label,
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
            "reference_policy": "prefer different subject; otherwise different recording",
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
) -> TaskEvaluationManifest:
    return _build_task1_manifest(
        cohort,
        caches,
        split="test",
        configs=_TASK1_CONFIG,
        name=name,
        seed=seed,
    )


def build_task1_development_manifest(
    cohort: CohortManifest,
    caches: Mapping[str, CachedRecordingDataset],
    *,
    seed: int = 20260831,
    name: str = "task1_development_v1",
) -> TaskEvaluationManifest:
    return _build_task1_manifest(
        cohort,
        caches,
        split="development",
        configs=_TASK1_CONFIG,
        name=name,
        seed=seed,
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


def unsupported_task2_manifest(
    cohort: CohortManifest, *, seed: int = 20260831
) -> TaskEvaluationManifest:
    """Record the current scientific blocker instead of inventing change labels."""

    return _manifest(
        name="task2_test_v1_blocked",
        task="task2",
        cohort=cohort,
        seed=seed,
        protocol={
            "status": "blocked",
            "required_unit": "same-person, same-task bounded executions with accepted/change truth",
        },
        units=(),
        exclusions=(
            {
                "dataset": "alameda",
                "reason": "free-living days have clinical linkage but no bounded task executions",
            },
            {
                "dataset": "cops",
                "reason": "hourly symptom states have no bounded task executions",
            },
            {
                "dataset": "cohort_v1_test",
                "reason": "C-MHAD, WEAR, and OCA do not label accepted versus changed executions",
            },
        ),
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
    configs = _TEST_CONFIG if split == "test" else _DEVELOPMENT_CONFIG
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
            expected = tuple(
                (float(event.start_sec), float(event.end_sec))
                for _, event in _events(
                    query,
                    kind=str(config["task1_kind"]),
                    background=set(config["background"]),
                )
                if event.label == unit.label
                and _intervals_inside_stream(
                    ((float(event.start_sec), float(event.end_sec)),), query_stream
                )
            )
            if unit.query_interval_sec is not None:
                block_start, block_end = map(float, unit.query_interval_sec)
                if block_end <= block_start:
                    raise ValueError("Task-1 query block is invalid")
                expected = tuple(
                    (max(left, block_start), min(right, block_end))
                    for left, right in expected
                    if min(right, block_end) > max(left, block_start)
                )
            if expected != tuple(map(tuple, unit.target_intervals_sec)):
                raise ValueError("Task-1 target intervals no longer match source annotations")
            if unit.target_present != bool(expected):
                raise ValueError("Task-1 target-presence flag is inconsistent")
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
        if manifest.units:
            raise ValueError("Task-2 is blocked but contains evaluation units")
    else:
        raise ValueError(f"unknown task manifest: {manifest.task!r}")
