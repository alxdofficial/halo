import numpy as np
import pytest

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
    split_at_clock_gaps,
)


def _stream() -> SensorStream:
    return SensorStream(
        stream_id="watch",
        placement="wrist",
        device="watch",
        timestamps_sec=np.asarray([0.0, 0.02, 0.04]),
        values=np.ones((3, 3), dtype=np.float32),
        channels=("acc_x", "acc_y", "acc_z"),
        valid=np.ones((3, 3), dtype=bool),
        gravity_state="present",
        nominal_rate_hz=50.0,
    )


def test_recording_contract_accepts_native_accelerometer() -> None:
    recording = RawRecording(
        dataset="example",
        recording_id="r1",
        subject_id="s1",
        session_id="visit1",
        streams=(_stream(),),
        events=(EventInterval(0.0, 0.04, "movement"),),
    )
    assert recording.streams[0].values.dtype == np.float32


def test_stream_rejects_non_monotonic_time() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        SensorStream(
            stream_id="watch",
            placement="wrist",
            device="watch",
            timestamps_sec=np.asarray([0.0, 0.02, 0.01]),
            values=np.ones((3, 3), dtype=np.float32),
            channels=("acc_x", "acc_y", "acc_z"),
            valid=np.ones((3, 3), dtype=bool),
            gravity_state="present",
        )


def test_split_at_clock_gaps_keeps_hard_boundaries() -> None:
    slices = split_at_clock_gaps(
        np.asarray([0.0, 0.02, 0.04, 2.0, 2.02]), max_gap_sec=0.1
    )
    assert [(item.start, item.stop) for item in slices] == [(0, 3), (3, 5)]
