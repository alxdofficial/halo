"""Acquisition compatibility policy tests."""

from __future__ import annotations

import numpy as np

from applications.motion_monitoring.data.compatibility import (
    sensor_compatibility_key,
    stream_compatibility_key,
)
from applications.motion_monitoring.data.contracts import SensorStream


def _stream(*, device: str, rate: float, placement: str = "wrist") -> SensorStream:
    return SensorStream(
        stream_id="imu",
        placement=placement,
        device=device,
        timestamps_sec=np.arange(4, dtype=np.float64) / rate,
        values=np.ones((4, 6), dtype=np.float32),
        channels=("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
        valid=np.ones((4, 6), dtype=bool),
        gravity_state="present",
        nominal_rate_hz=rate,
    )


def test_device_model_and_native_rate_are_declared_nuisance_factors() -> None:
    first = _stream(device="Apple smartwatch", rate=50.0)
    second = _stream(device="Bangle.js smartwatch", rate=100.0)

    assert stream_compatibility_key(first) == stream_compatibility_key(second)


def test_phone_and_watch_are_not_compatible_even_at_one_placement() -> None:
    watch = _stream(device="smartwatch", rate=50.0)
    phone = _stream(device="smartphone", rate=50.0)

    assert stream_compatibility_key(watch) != stream_compatibility_key(phone)


def test_channel_order_is_canonical_but_placement_is_not_erased() -> None:
    first = sensor_compatibility_key(
        device="smartwatch",
        placement="Right-Wrist",
        channels=("gyro_z", "acc_y", "acc_x", "gyro_x", "acc_z", "gyro_y"),
        gravity_state="present",
    )
    reordered = sensor_compatibility_key(
        device="smartwatch",
        placement="right wrist",
        channels=("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
        gravity_state="present",
    )
    ankle = sensor_compatibility_key(
        device="smartwatch",
        placement="right ankle",
        channels=reordered.channels,
        gravity_state="present",
    )

    assert first == reordered
    assert first != ankle
