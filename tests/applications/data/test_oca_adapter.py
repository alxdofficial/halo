from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from applications.motion_monitoring.data.adapters.oca import iter_recordings


OCA_ROOT = (
    Path(__file__).resolve().parents[3]
    / "applications"
    / "motion_monitoring"
    / "data"
    / "sources"
    / "oca"
)


def _require_payload() -> None:
    if not (OCA_ROOT / "downloads" / "OCA.zip").is_file():
        pytest.skip("local OCA payload is not available")


def test_oca_adapter_loads_real_native_rate_recordings() -> None:
    _require_payload()
    recordings = list(iter_recordings(OCA_ROOT, limit=3))

    assert [recording.recording_id for recording in recordings] == [
        "P0-R0-part00",
        "P0-R0-part01",
        "P0-R1",
    ]
    assert [recording.subject_id for recording in recordings] == ["P0", "P0", "P0"]
    assert all(recording.session_id.startswith("P0-R") for recording in recordings)
    assert all(recording.split == "training" for recording in recordings)
    assert all(len(recording.streams) == 4 for recording in recordings)

    before_gap, after_gap, rate_20hz = recordings
    gap_sec = (
        after_gap.streams[0].timestamps_sec[0]
        - before_gap.streams[0].timestamps_sec[-1]
    )
    assert gap_sec == pytest.approx(81.296, abs=0.001)
    assert before_gap.streams[0].nominal_rate_hz == pytest.approx(27.027, rel=1e-3)
    assert rate_20hz.streams[0].nominal_rate_hz == pytest.approx(20.0, rel=1e-3)

    for recording in recordings:
        first_t = recording.streams[0].timestamps_sec[0]
        last_t = recording.streams[0].timestamps_sec[-1]
        assert [stream.stream_id for stream in recording.streams] == [
            "imu0",
            "imu2",
            "imu1",
            "imu3",
        ]
        assert [stream.placement for stream in recording.streams] == [
            "right upper arm",
            "left upper arm",
            "right chest",
            "left chest",
        ]
        for stream in recording.streams:
            assert stream.values.shape == stream.valid.shape
            assert stream.values.shape[1] == 6
            assert stream.channels == (
                "acc_x",
                "acc_y",
                "acc_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
            )
            assert stream.valid.dtype == np.bool_
            assert stream.valid.all()
            assert np.all(np.diff(stream.timestamps_sec) > 0)
            assert stream.gravity_state == "present"
            median_acc_norm_g = np.median(np.linalg.norm(stream.values[:, :3], axis=1))
            assert 0.85 < median_acc_norm_g < 1.15
            gyro_quantization = stream.values[:, 3:] * (180.0 / np.pi) * 16.0
            assert np.max(np.abs(gyro_quantization - np.rint(gyro_quantization))) < 2e-3

        assert recording.events[0].start_sec == pytest.approx(first_t)
        assert recording.events[-1].end_sec == pytest.approx(last_t)
        assert all(
            first_t <= event.start_sec < event.end_sec <= last_t
            for event in recording.events
        )
        assert all(
            left.end_sec == pytest.approx(right.start_sec)
            for left, right in zip(recording.events[:-1], recording.events[1:])
        )
        assert {event.label for event in recording.events} == {
            "Null",
            "Mount Cover Panel",
            "Take Cover Panel Off",
            "Take Screwdriver",
            "Place Screwdriver Down",
            "Screw Unscrew Cover Panel",
            "Pick Up Screw",
        }


def test_oca_adapter_preserves_official_splits_and_source_metadata() -> None:
    _require_payload()
    recordings_by_session = {}
    recording_count = 0
    for recording in iter_recordings(OCA_ROOT):
        recordings_by_session[recording.session_id] = recording
        recording_count += 1

    assert recording_count == 13  # Twelve sessions and one hard-gap split.
    assert recordings_by_session["P1-R1"].split == "validation"
    assert recordings_by_session["P2-R2"].split == "validation"
    assert recordings_by_session["P3-R0"].split == "testing"
    assert recordings_by_session["P4-R0"].split == "testing"
    assert recordings_by_session["P4-R0"].metadata["arm_support"] == "entire_session"
    assert recordings_by_session["P3-R0"].metadata["source_placements"] == {
        "imu0": "right upper arm",
        "imu1": "right chest",
        "imu2": "left upper arm",
        "imu3": "left chest",
    }
    assert recordings_by_session["P3-R0"].metadata["preferred_application_streams"] == (
        "imu0",
        "imu2",
    )


def test_oca_adapter_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        next(iter_recordings(OCA_ROOT, limit=-1))
