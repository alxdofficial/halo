from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)
from applications.motion_monitoring.sequence import MotionSequence
from applications.motion_monitoring.task3.data import (
    collate_motion_sequences,
    event_batch_from_recordings,
)


def _recording(dataset: str, recording_id: str) -> RawRecording:
    timestamps = np.arange(1_634_178_333.0, 1_634_178_337.0, 0.1)
    stream = SensorStream(
        stream_id="watch",
        placement="wrist",
        device="watch",
        timestamps_sec=timestamps,
        values=np.ones((len(timestamps), 3), dtype=np.float32),
        channels=("acc_x", "acc_y", "acc_z"),
        valid=np.ones((len(timestamps), 3), dtype=np.bool_),
        gravity_state="present",
        nominal_rate_hz=10.0,
    )
    events = (
        EventInterval(timestamps[0], timestamps[0] + 1.0, "reach", "action"),
        EventInterval(timestamps[0] + 1.0, timestamps[0] + 2.0, "idle", "action"),
        EventInterval(timestamps[0] + 2.0, timestamps[0] + 3.0, "reach", "action"),
    )
    return RawRecording(
        dataset=dataset,
        recording_id=recording_id,
        subject_id=f"{dataset}-subject",
        session_id=f"{dataset}-session",
        streams=(stream,),
        events=events,
    )


def _sequence(recording: RawRecording, *, length: int) -> MotionSequence:
    start = recording.streams[0].timestamps_sec[0]
    intervals = torch.tensor(
        [[start + index, start + index + 1.0] for index in range(length)],
        dtype=torch.float64,
    )
    embeddings = torch.eye(length, 4, dtype=torch.float32)
    physical = torch.zeros(length, 1)
    return MotionSequence(
        embeddings=embeddings,
        intervals_sec=intervals,
        valid=torch.ones(length, dtype=torch.bool),
        physical_features=physical,
        physical_feature_mask=torch.ones_like(physical, dtype=torch.bool),
        physical_feature_names=("test",),
        dataset=recording.dataset,
        recording_id=recording.recording_id,
        subject_id=recording.subject_id,
        session_id=recording.session_id,
        stream_id="watch",
        placement="wrist",
        device="watch",
        channels=("acc_x", "acc_y", "acc_z"),
        gravity_state="present",
        sampling_rate_hz=10.0,
    )


def test_task3_bridge_preserves_clock_and_padding_masks() -> None:
    first = _recording("dataset_a", "first")
    second = _recording("dataset_b", "second")
    first_sequence = _sequence(first, length=4)
    second_sequence = _sequence(second, length=3)

    timeline = collate_motion_sequences((first_sequence, second_sequence))

    assert timeline.intervals_sec.dtype == torch.float64
    assert timeline.valid.tolist() == [[True, True, True, True], [True, True, True, False]]
    assert timeline.embeddings.shape == (2, 4, 4)


def test_task3_event_ids_are_scope_local_and_background_is_not_an_identity() -> None:
    first = _recording("dataset_a", "first")
    second = _recording("dataset_b", "second")
    events = event_batch_from_recordings(
        (first, second),
        (_sequence(first, length=4), _sequence(second, length=4)),
        annotation_kind="action",
        exhaustive=(True, False),
        background_labels=frozenset({"idle"}),
    )

    assert events.start_sec.dtype == torch.float64
    assert events.event_mask.sum(dim=1).tolist() == [2, 2]
    assert events.label_id[:, :2].tolist() == [[0, 0], [0, 0]]
    assert events.scope_id[:, :2].tolist() == [[0, 0], [1, 1]]
    assert len(set(events.instance_id[events.event_mask].tolist())) == 4
    assert events.exhaustive.tolist() == [True, False]


def test_task3_scopes_do_not_merge_cross_placement_events() -> None:
    recording = _recording("dataset_a", "first")
    wrist = _sequence(recording, length=4)
    ankle = replace(wrist, stream_id="ankle", placement="ankle")
    events = event_batch_from_recordings(
        (recording, recording),
        (wrist, ankle),
        annotation_kind="action",
        background_labels=frozenset({"idle"}),
    )

    assert events.scope_id[0, 0] != events.scope_id[1, 0]
    assert events.instance_id[0, :2].tolist() == events.instance_id[1, :2].tolist()


def test_exhaustive_task3_supervision_rejects_crop_clipped_events() -> None:
    recording = _recording("dataset_a", "first")
    clipped = EventInterval(
        recording.events[0].start_sec,
        recording.events[0].end_sec,
        "reach",
        "action",
        {"clipped_by_recording_crop": True},
    )
    recording = RawRecording(
        **{**recording.__dict__, "events": (clipped,)}
    )
    with pytest.raises(ValueError, match="events clipped by a timeline crop"):
        event_batch_from_recordings(
            (recording,),
            (_sequence(recording, length=4),),
            annotation_kind="action",
            exhaustive=True,
        )
