from __future__ import annotations

import numpy as np
import pytest

from applications.motion_monitoring.data.contracts import RawRecording, SensorStream
from applications.motion_monitoring.data.verify_adapters import _validate_recording


def test_invalid_fraction_counts_scalar_values_not_timestamps() -> None:
    timestamps = np.arange(2, dtype=np.float64)
    one_channel = SensorStream(
        stream_id="one",
        placement="wrist",
        device="fixture",
        timestamps_sec=timestamps,
        values=np.zeros((2, 1), dtype=np.float32),
        channels=("acc_x",),
        valid=np.asarray([[False], [True]]),
        gravity_state="present",
        nominal_rate_hz=1.0,
    )
    three_channels = SensorStream(
        stream_id="three",
        placement="wrist",
        device="fixture",
        timestamps_sec=timestamps,
        values=np.zeros((2, 3), dtype=np.float32),
        channels=("acc_x", "acc_y", "acc_z"),
        valid=np.ones((2, 3), dtype=bool),
        gravity_state="present",
        nominal_rate_hz=1.0,
    )
    row = _validate_recording(
        RawRecording(
            dataset="fixture",
            recording_id="recording",
            subject_id="subject",
            session_id="session",
            streams=(one_channel, three_channels),
        )
    )

    assert row["invalid_values"] == 1
    assert row["value_count"] == 8
    assert row["invalid_fraction"] == pytest.approx(1 / 8)
