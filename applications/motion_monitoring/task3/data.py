"""Bridge shared complete-timeline representations to Task-3 supervision."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from applications.motion_monitoring.data.compatibility import sensor_compatibility_key
from applications.motion_monitoring.data.contracts import RawRecording
from applications.motion_monitoring.sequence import MotionSequence, localization_intervals

from .contracts import EventBatch


@dataclass(frozen=True)
class TimelineBatch:
    embeddings: torch.Tensor
    intervals_sec: torch.Tensor
    valid: torch.Tensor

    def to(self, device: torch.device | str) -> "TimelineBatch":
        return TimelineBatch(
            self.embeddings.to(device),
            self.intervals_sec.to(device),
            self.valid.to(device),
        )


def collate_motion_sequences(sequences: Sequence[MotionSequence]) -> TimelineBatch:
    if not sequences:
        raise ValueError("cannot collate an empty timeline batch")
    if len({sequence.embeddings.shape[1] for sequence in sequences}) != 1:
        raise ValueError("all timeline embeddings must have the same width")
    return TimelineBatch(
        embeddings=pad_sequence(
            [sequence.embeddings for sequence in sequences], batch_first=True
        ),
        intervals_sec=pad_sequence(
            [localization_intervals(sequence) for sequence in sequences], batch_first=True
        ),
        valid=pad_sequence(
            [sequence.valid for sequence in sequences],
            batch_first=True,
            padding_value=False,
        ),
    )


def event_batch_from_recordings(
    recordings: Sequence[RawRecording],
    sequences: Sequence[MotionSequence],
    *,
    annotation_kind: str,
    exhaustive: bool | Sequence[bool] = False,
    background_labels: frozenset[str] = frozenset(),
) -> EventBatch:
    """Create equality-only IDs; event label strings never enter the model."""

    if len(recordings) != len(sequences) or not recordings:
        raise ValueError("recordings and sequences must be non-empty and aligned")
    exhaustive_rows = (
        [bool(exhaustive)] * len(recordings)
        if isinstance(exhaustive, bool)
        else [bool(value) for value in exhaustive]
    )
    if len(exhaustive_rows) != len(recordings):
        raise ValueError("exhaustive flags must align with recordings")

    selected: list[list[tuple[float, float, str, tuple[object, ...]]]] = []
    scope_keys: list[tuple[object, ...]] = []
    for row_index, (recording, sequence) in enumerate(
        zip(recordings, sequences, strict=True)
    ):
        if sequence.recording_id != recording.recording_id:
            raise ValueError("recording and MotionSequence provenance do not align")
        timeline_start = float(sequence.intervals_sec[0, 0])
        timeline_end = float(sequence.intervals_sec[-1, 1])
        clipped_events = [
            event
            for event in recording.events
            if event.annotation_kind == annotation_kind
            and bool(event.metadata.get("clipped_by_recording_crop", False))
        ]
        if clipped_events and exhaustive_rows[row_index]:
            raise ValueError(
                "exhaustive Task-3 supervision cannot use events clipped by a timeline crop"
            )
        source_recording_id = str(
            recording.metadata.get("source_recording_id", recording.recording_id)
        )
        rows = [
            (
                max(timeline_start, event.start_sec),
                min(timeline_end, event.end_sec),
                event.label,
                (
                    recording.dataset,
                    source_recording_id,
                    event.metadata.get("execution_id", event.start_sec),
                    event.end_sec,
                    event.label,
                ),
            )
            for event in recording.events
            if event.annotation_kind == annotation_kind
            and event.label not in background_labels
            and not bool(event.metadata.get("clipped_by_recording_crop", False))
            and min(timeline_end, event.end_sec)
            > max(timeline_start, event.start_sec)
        ]
        selected.append(rows)
        scope_keys.append(
            (
                recording.dataset,
                annotation_kind,
                sensor_compatibility_key(
                    device=sequence.device,
                    placement=sequence.placement,
                    channels=sequence.channels,
                    gravity_state=sequence.gravity_state,
                ),
            )
        )

    scope_lookup = {key: index for index, key in enumerate(sorted(set(scope_keys)))}
    labels_by_scope: dict[tuple[object, ...], dict[str, int]] = {}
    for scope in set(scope_keys):
        labels = sorted(
            {
                label
                for row_scope, rows in zip(scope_keys, selected, strict=True)
                if row_scope == scope
                for _, _, label, _ in rows
            }
        )
        labels_by_scope[scope] = {label: index for index, label in enumerate(labels)}

    max_events = max(map(len, selected), default=0)
    batch_size = len(recordings)
    starts = torch.zeros((batch_size, max_events), dtype=torch.float64)
    ends = torch.zeros_like(starts)
    label_id = torch.full((batch_size, max_events), -1, dtype=torch.long)
    instance_id = torch.full_like(label_id, -1)
    scope_id = torch.full_like(label_id, -1)
    event_mask = torch.zeros((batch_size, max_events), dtype=torch.bool)
    instance_keys = sorted(
        {instance_key for rows in selected for _, _, _, instance_key in rows},
        key=repr,
    )
    instance_lookup = {key: index for index, key in enumerate(instance_keys)}
    for row_index, (scope, rows) in enumerate(zip(scope_keys, selected, strict=True)):
        for event_index, (start, end, label, instance_key) in enumerate(rows):
            starts[row_index, event_index] = start
            ends[row_index, event_index] = end
            label_id[row_index, event_index] = labels_by_scope[scope][label]
            instance_id[row_index, event_index] = instance_lookup[instance_key]
            scope_id[row_index, event_index] = scope_lookup[scope]
            event_mask[row_index, event_index] = True
    return EventBatch(
        start_sec=starts,
        end_sec=ends,
        label_id=label_id,
        instance_id=instance_id,
        scope_id=scope_id,
        event_mask=event_mask,
        exhaustive=torch.tensor(exhaustive_rows, dtype=torch.bool),
    )
