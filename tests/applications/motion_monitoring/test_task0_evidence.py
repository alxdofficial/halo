from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from applications.motion_monitoring.data.contracts import RawRecording, SensorStream
from applications.motion_monitoring.task0.contracts import EvidenceConfig
from applications.motion_monitoring.task0.evidence import (
    extract_physical_evidence,
    fit_robust_scaler,
)


def _recording(
    *,
    rate_hz: float,
    duration: float = 4.0,
    gravity_state: str = "removed",
    include_gyro: bool = True,
    missing: slice | None = None,
    static: bool = False,
) -> tuple[RawRecording, SensorStream]:
    timestamps = np.arange(0.0, duration, 1.0 / rate_hz)
    if static:
        acceleration = np.tile([0.0, 0.0, 1.0], (len(timestamps), 1))
    else:
        acceleration = np.column_stack(
            [
                np.sin(2 * np.pi * 2.0 * timestamps),
                np.zeros_like(timestamps),
                np.zeros_like(timestamps),
            ]
        )
    values = acceleration
    channels = ["acc_x", "acc_y", "acc_z"]
    if include_gyro:
        gyro = np.column_stack(
            [
                np.zeros_like(timestamps),
                np.cos(2 * np.pi * timestamps),
                np.zeros_like(timestamps),
            ]
        )
        values = np.column_stack([values, gyro])
        channels.extend(["gyro_x", "gyro_y", "gyro_z"])
    valid = np.ones_like(values, dtype=bool)
    if missing is not None:
        valid[missing] = False
        values[missing] = np.nan
    stream = SensorStream(
        stream_id="watch",
        placement="left wrist",
        device="watch",
        timestamps_sec=timestamps,
        values=values,
        channels=channels,
        valid=valid,
        gravity_state=gravity_state,
        nominal_rate_hz=rate_hz,
    )
    recording = RawRecording(
        dataset="synthetic",
        recording_id=f"r{rate_hz}",
        subject_id="s1",
        session_id="session1",
        streams=(stream,),
    )
    return recording, stream


def test_native_rate_evidence_is_consistent() -> None:
    config = EvidenceConfig(window_seconds=1.0, stride_seconds=0.5)
    low_recording, low_stream = _recording(rate_hz=50.0)
    high_recording, high_stream = _recording(rate_hz=100.0)
    low = extract_physical_evidence(low_recording, low_stream, config)
    high = extract_physical_evidence(high_recording, high_stream, config)
    assert np.allclose(low.features, high.features, atol=2e-3)
    assert np.array_equal(low.feature_valid, high.feature_valid)


def test_static_gravity_is_removed_only_for_evidence() -> None:
    recording, stream = _recording(rate_hz=50.0, gravity_state="present", static=True)
    sequence = extract_physical_evidence(recording, stream)
    assert sequence.feature_valid[:, 0].all()
    assert np.max(sequence.features[:, 0]) < 1e-10
    assert np.allclose(stream.values[:, 2], 1.0)


def test_acceleration_only_keeps_gyro_explicitly_unavailable() -> None:
    recording, stream = _recording(rate_hz=50.0, include_gyro=False)
    sequence = extract_physical_evidence(recording, stream)
    assert sequence.feature_valid[:, 0].all()
    assert not sequence.feature_valid[:, 1].any()
    assert np.all(sequence.features[:, 1] == 0.0)


def test_missing_data_does_not_become_signal() -> None:
    recording, stream = _recording(rate_hz=50.0, missing=slice(50, 100))
    sequence = extract_physical_evidence(recording, stream)
    affected = (sequence.window_start_sec >= 1.0) & (sequence.window_end_sec <= 2.0)
    assert (~sequence.feature_valid[affected]).all()
    assert np.all(sequence.features[affected] == 0.0)


def test_scaler_is_robust_and_requires_observations() -> None:
    recording, stream = _recording(rate_hz=50.0)
    sequence = extract_physical_evidence(recording, stream)
    scaler = fit_robust_scaler([sequence])
    assert scaler.observed.all()
    assert np.all(scaler.scale > 0)
    with pytest.raises(ValueError, match="no valid physical evidence"):
        fit_robust_scaler(
            [
                type(sequence)(
                    **{
                        **sequence.__dict__,
                        "feature_valid": np.zeros_like(sequence.feature_valid),
                    }
                )
            ]
        )


def test_measured_timestamps_control_window_completeness() -> None:
    recording, stream = _recording(
        rate_hz=100.0 / 3.0,
        duration=1.0,
        missing=slice(0, 1),
    )
    stream = replace(stream, nominal_rate_hz=30.0)
    recording = replace(recording, streams=(stream,))
    sequence = extract_physical_evidence(
        recording,
        stream,
        EvidenceConfig(min_valid_fraction=0.95),
    )
    assert not sequence.feature_valid[0].any()


def test_quiet_second_modality_does_not_reduce_motion_score() -> None:
    from applications.motion_monitoring.task0.contracts import RobustFeatureScaler
    from applications.motion_monitoring.task0.evidence import standardized_motion_score

    recording, stream = _recording(rate_hz=50.0)
    sequence = extract_physical_evidence(recording, stream)
    features = np.asarray([[4.0, 0.0]])
    common = {
        **sequence.__dict__,
        "window_start_sec": np.asarray([0.0]),
        "window_end_sec": np.asarray([0.5]),
        "features": features,
        "valid_fraction": np.ones(1),
        "constant_fraction": np.zeros(1),
    }
    six_axis = type(sequence)(**{**common, "feature_valid": np.asarray([[True, True]])})
    acceleration_only = type(sequence)(
        **{**common, "feature_valid": np.asarray([[True, False]])}
    )
    scaler = RobustFeatureScaler(
        center=np.zeros(2), scale=np.ones(2), observed=np.ones(2, dtype=bool)
    )
    assert standardized_motion_score(six_axis, scaler)[0][0] == 4.0
    assert standardized_motion_score(acceleration_only, scaler)[0][0] == 4.0
