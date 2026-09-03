"""Turn canonical and representation caches into Task-2 execution records.

This is Task 2's own data path, deliberately separate from the shared cohort
builder. That builder drops recordings marked as duplicate signal views, which is
correct for Tasks 1 and 3 because a CrossFit repetition and its parent set are the
same recorded motion and must never land on opposite sides of a split. Task 2's
unit *is* the repetition and it splits by subject, so the guard has no purchase
here and applying it would silently empty the source.

A record carries the provenance the batch rules need: which physical execution it
came from (so a query can never be a transformed descendant of its own reference),
which day it was recorded on (so different-day positives can be over-represented),
and which variant it is (clean, nuisance, or a declared modification).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from applications.motion_monitoring.data.cache import CachedRecordingDataset
from applications.motion_monitoring.data.compatibility import sensor_compatibility_key
from applications.motion_monitoring.data.contracts import RawRecording
from applications.motion_monitoring.sequence import MotionSequence
from applications.motion_monitoring.representation_cache import bounded_representation_id
from .contracts import BoundedExecution
from .data_sources import is_selected_event, spec_for
from .episodes import ExecutionRecord


# Which annotation kind is one bounded execution, per source.
EXECUTION_KINDS: Mapping[str, str] = {
    "harmes": "bounded_execution",
    "crossfit": "repetition",
    "monipar": "bounded_execution",
    "kneepad": "bounded_execution",
    "task2_modified_v1": "bounded_execution",
}


def record_identity(record: ExecutionRecord) -> str:
    """Stable identity of one encoded execution/stream across encoder caches."""

    key = record.key
    return "|".join(
        (
            record.dataset,
            record.execution.execution_id,
            key.device_family,
            key.placement,
            ",".join(key.channels),
            key.gravity_state,
        )
    )


def _harmes_day(recording: RawRecording) -> str | None:
    start = recording.streams[0].metadata.get("source_epoch_start_sec")
    if start is None:
        return None
    return datetime.fromtimestamp(float(start), tz=timezone.utc).date().isoformat()


def _monipar_day(recording: RawRecording) -> str | None:
    week = recording.metadata.get("week")
    return None if week is None else f"week_{int(week):02d}"


def _derived_day(recording: RawRecording) -> str | None:
    """A variant inherits the acquisition day of the execution it was made from.

    The variant adapter copies the source stream's metadata verbatim, so the
    origin's own resolver still works on it. Resolving through the origin matters
    because a nuisance variant may serve as an accepted query, and a variant with
    no day would be scored as same-session, undercounting exactly the cross-day
    positives the objective over-weights.
    """

    origin = recording.metadata.get("origin_dataset")
    if not origin or origin == "task2_modified_v1":
        return None
    resolver = DAY_RESOLVERS.get(str(origin))
    return None if resolver is None else resolver(recording)


DAY_RESOLVERS: Mapping[str, Callable[[RawRecording], str | None]] = {
    "harmes": _harmes_day,
    "monipar": _monipar_day,
    # CrossFit records one continuous set per subject and exercise, so it has no
    # day axis at all. Left as None deliberately: every CrossFit positive pair is
    # same-session, and the relation telemetry must be able to say so.
    "crossfit": lambda recording: None,
    # One visit per patient, so every KneE-PAD repeat is within-session.
    "kneepad": lambda recording: None,
    "task2_modified_v1": _derived_day,
}


def crop_sequence(sequence: MotionSequence, start_sec: float, end_sec: float) -> MotionSequence:
    """Keep the patches whose centres fall inside a bounded execution."""

    centres = sequence.intervals_sec.mean(dim=1)
    selected = ((centres >= start_sec) & (centres < end_sec)).nonzero().flatten()
    if not len(selected):
        raise ValueError("bounded execution contains no representation patches")
    left, right = int(selected[0]), int(selected[-1]) + 1
    return MotionSequence(
        embeddings=sequence.embeddings[left:right],
        intervals_sec=sequence.intervals_sec[left:right],
        valid=sequence.valid[left:right],
        physical_features=sequence.physical_features[left:right],
        physical_feature_mask=sequence.physical_feature_mask[left:right],
        physical_feature_names=sequence.physical_feature_names,
        dataset=sequence.dataset,
        recording_id=sequence.recording_id,
        subject_id=sequence.subject_id,
        session_id=sequence.session_id,
        stream_id=sequence.stream_id,
        placement=sequence.placement,
        device=sequence.device,
        channels=sequence.channels,
        gravity_state=sequence.gravity_state,
        sampling_rate_hz=sequence.sampling_rate_hz,
    )


def iter_execution_records(
    dataset: str,
    cache: CachedRecordingDataset,
    representations,
    *,
    task_of: Callable[[RawRecording, object], str] | None = None,
    limit: int | None = None,
    strict: bool = False,
) -> Iterator[ExecutionRecord]:
    """Yield one record per bounded execution that has usable embeddings."""

    kind = EXECUTION_KINDS.get(dataset)
    if kind is None:
        raise KeyError(f"{dataset!r} has no declared Task-2 execution kind")
    # The derived variant corpus is generated, not a declared source, so it has no
    # entry in the source contract; everything else is filtered through it so a
    # record pool cannot pick up an execution the protocol excludes (MoniPar's
    # tremor-graded resting runs are the case this catches).
    try:
        spec_for(dataset)
        contracted = True
    except KeyError:
        contracted = False
    resolve_day = DAY_RESOLVERS.get(dataset, lambda recording: None)
    yielded = 0
    for recording in cache:
        if limit is not None and yielded >= limit:
            return
        day = resolve_day(recording)
        for stream in recording.streams:
            key = sensor_compatibility_key(
                device=stream.device,
                placement=stream.placement,
                channels=stream.channels,
                gravity_state=stream.gravity_state,
            )
            for index, event in enumerate(recording.events):
                if event.annotation_kind != kind:
                    continue
                if contracted and not is_selected_event(recording, event):
                    continue
                if limit is not None and yielded >= limit:
                    return
                try:
                    sequence = representations.get(
                        dataset,
                        bounded_representation_id(recording.recording_id, index),
                        stream.stream_id,
                    )
                except (KeyError, FileNotFoundError) as error:
                    if strict:
                        raise ValueError(
                            f"missing bounded representation for "
                            f"{dataset}/{recording.recording_id}:{index}/{stream.stream_id}"
                        ) from error
                    continue
                try:
                    cropped = crop_sequence(sequence, float(event.start_sec), float(event.end_sec))
                except ValueError as error:
                    if strict:
                        raise ValueError(
                            f"cannot crop {dataset}/{recording.recording_id}:{index}/{stream.stream_id}"
                        ) from error
                    continue
                # A variant belongs to the source it was recorded from, not to the
                # cache it is stored in, so it groups with its clean siblings and a
                # batch still holds one source.
                source_dataset = str(recording.metadata.get("origin_dataset", dataset))
                origin = str(
                    recording.metadata.get(
                        "origin_execution_id", f"{source_dataset}:{recording.recording_id}:{index}"
                    )
                )
                execution_id = f"{recording.recording_id}:{index}"
                task_id = event.label if task_of is None else task_of(recording, event)
                try:
                    execution = BoundedExecution(
                        embeddings=cropped.embeddings,
                        patch_intervals_sec=cropped.intervals_sec,
                        patch_mask=cropped.valid,
                        dataset=source_dataset,
                        subject_id=str(recording.subject_id),
                        session_id=str(recording.session_id),
                        execution_id=execution_id,
                        task_id=str(task_id),
                        sensor_config=key,
                        physical_features=cropped.physical_features,
                        physical_feature_mask=cropped.physical_feature_mask,
                        physical_feature_names=cropped.physical_feature_names,
                    )
                except ValueError as error:
                    # An execution spanning an invalid patch gap is rejected, not
                    # interpolated across.
                    if strict:
                        raise ValueError(
                            f"invalid execution {dataset}/{recording.recording_id}:{index}/{stream.stream_id}"
                        ) from error
                    continue
                yield ExecutionRecord(
                    execution=execution,
                    key=key,
                    day=day,
                    origin_execution_id=origin,
                    source_dataset=source_dataset,
                    variant=str(recording.metadata.get("variant", "clean")),
                    modification_kind=recording.metadata.get("modification_kind"),
                    severity=float(recording.metadata.get("severity", 0.0) or 0.0),
                    nuisance_kind=recording.metadata.get("nuisance_kind"),
                    source_subject_id=str(
                        recording.metadata.get("origin_subject_id", recording.subject_id)
                    ),
                )
                yielded += 1


def build_record_pool(
    datasets: Sequence[str],
    representations,
    *,
    source_root: Path | None = None,
    limit_per_dataset: int | None = None,
    strict: bool = False,
) -> list[ExecutionRecord]:
    """Assemble the clean and pre-materialised variant records for a run."""

    from applications.motion_monitoring.data.examples import open_cache

    records: list[ExecutionRecord] = []
    for dataset in datasets:
        cache = (
            open_cache(dataset)
            if source_root is None
            else CachedRecordingDataset(Path(source_root) / dataset / "processed" / "canonical_v1")
        )
        records.extend(
            iter_execution_records(
                dataset,
                cache,
                representations,
                limit=limit_per_dataset,
                strict=strict,
            )
        )
    return records
