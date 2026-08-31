from __future__ import annotations

import numpy as np
import pytest

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)
from applications.motion_monitoring.data.examples import (
    EventExample,
    crop_event,
    crop_query_around_event,
    crop_recording,
)


def _example() -> EventExample:
    timestamps = np.arange(100.0, 110.0, 0.1)
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
    event = EventInterval(104.0, 106.0, "movement")
    recording = RawRecording("test", "recording", "subject", "session", (stream,), (event,))
    return EventExample(recording, event, "test/recording/event/0")


def test_crop_keeps_source_clock_and_clips_events() -> None:
    cropped = crop_recording(_example().recording, 103.0, 105.0)
    assert cropped.streams[0].timestamps_sec[[0, -1]].tolist() == pytest.approx([103.0, 104.9])
    assert [(event.start_sec, event.end_sec) for event in cropped.events] == [(104.0, 105.0)]
    assert cropped.events[0].metadata["clipped_by_recording_crop"]
    assert cropped.events[0].metadata["source_event_end_sec"] == 106.0


def test_event_and_query_crops_have_distinct_contracts() -> None:
    example = _example()
    event = crop_event(example)
    query = crop_query_around_event(example, duration_sec=6.0)
    assert event.streams[0].timestamps_sec[0] == pytest.approx(104.0)
    assert event.streams[0].timestamps_sec[-1] == pytest.approx(105.9)
    assert query.streams[0].timestamps_sec[0] == pytest.approx(102.0)
    assert query.streams[0].timestamps_sec[-1] == pytest.approx(107.9)
    assert event.metadata["source_recording_id"] == "recording"
    assert query.metadata["source_recording_id"] == "recording"


def test_nested_crops_preserve_the_root_recording_identity() -> None:
    first = crop_recording(_example().recording, 101.0, 109.0, recording_suffix="first")
    second = crop_recording(first, 102.0, 108.0, recording_suffix="second")
    assert second.metadata["source_recording_id"] == "recording"
