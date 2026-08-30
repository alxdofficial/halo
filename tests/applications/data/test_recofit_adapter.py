from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from applications.motion_monitoring.data.adapters.recofit import iter_recordings


SOURCE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "applications"
    / "motion_monitoring"
    / "data"
    / "sources"
    / "recofit"
)


def _require_source() -> None:
    if not (
        SOURCE_ROOT / "downloads" / "exercise_data.50.0000_multionly.mat"
    ).is_file():
        pytest.skip("local RecoFit payload is not available")


def test_recofit_adapter_loads_real_visits_in_canonical_units() -> None:
    _require_source()
    recordings = list(iter_recordings(SOURCE_ROOT, limit=6))

    assert len(recordings) == 6
    assert [recording.subject_id for recording in recordings] == ["3"] * 5 + ["62"]
    assert [recording.metadata["source_file_index"] for recording in recordings] == [
        1,
        51,
        87,
        110,
        146,
        2,
    ]

    for recording in recordings:
        assert recording.dataset == "recofit"
        assert recording.metadata["clock_segment_count"] == 1
        assert recording.metadata["documented_master_stream_only"] is True
        assert len(recording.streams) == 1

        stream = recording.streams[0]
        assert stream.stream_id == "right_forearm_imu"
        assert stream.placement == "right_forearm"
        assert stream.gravity_state == "present"
        assert stream.nominal_rate_hz == 50.0
        assert stream.channels == (
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
        )
        assert stream.values.shape == stream.valid.shape
        assert stream.values.shape[1] == 6
        assert stream.values.dtype == np.float32
        assert stream.valid.dtype == np.bool_
        assert stream.valid.all()
        assert stream.timestamps_sec.flags.c_contiguous
        timestamp_base = stream.timestamps_sec.base
        assert timestamp_base is None or (
            timestamp_base.ndim == 1
            and timestamp_base.nbytes == stream.timestamps_sec.nbytes
        )
        assert np.all(np.diff(stream.timestamps_sec) > 0)
        assert np.median(np.diff(stream.timestamps_sec)) == pytest.approx(
            0.02, abs=2e-5
        )

        acceleration_norm = np.linalg.norm(stream.values[:, :3], axis=1)
        gyro_norm = np.linalg.norm(stream.values[:, 3:], axis=1)
        assert 0.8 < float(np.median(acceleration_norm)) < 1.2
        assert 0.1 < float(np.percentile(gyro_norm, 95)) < 10.0

        assert recording.events
        assert any(event.annotation_kind == "background" for event in recording.events)
        assert any(event.annotation_kind == "set" for event in recording.events)
        for event in recording.events:
            assert stream.timestamps_sec[0] <= event.start_sec < event.end_sec
            assert event.end_sec <= stream.timestamps_sec[-1]

    np.testing.assert_allclose(
        recordings[0].streams[0].values[0, :3],
        [-1.05339641, -0.240851366, 0.00764406976],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        recordings[0].streams[0].values[0, 3:],
        np.deg2rad([-6.3203296, -38.3379734, -16.0673508]),
        rtol=1e-6,
    )

    first_event = recordings[0].events[0]
    assert first_event.label == "<Initial Activity>"
    assert first_event.annotation_kind == "source_junk"
    assert first_event.start_sec == 0.0
    assert first_event.metadata["source_start_sec"] == pytest.approx(-15.207)
    assert first_event.metadata["clipped_to_clock_segment"] is True

    jumping_jacks = next(
        event for event in recordings[0].events if event.label == "Jumping Jacks"
    )
    assert jumping_jacks.annotation_kind == "set"
    assert jumping_jacks.metadata["repetition_count"] == 20
    assert jumping_jacks.metadata["source_startSequenceNumberMaster"] == 29792.0

    device_on_table = recordings[1].events[-1]
    assert device_on_table.label == "Device on Table"
    assert device_on_table.annotation_kind == "background"
    assert "Non-exercise" in device_on_table.metadata["source_activity_groups"]


def test_recofit_adapter_preserves_corpus_visit_and_completeness_metadata() -> None:
    _require_source()
    visit_count = 0
    incomplete_count = 0
    subject_ids: set[str] = set()

    for recording in iter_recordings(SOURCE_ROOT):
        visit_count += 1
        incomplete_count += int(recording.metadata["source_incomplete"])
        subject_ids.add(recording.subject_id)

    assert visit_count == 126
    assert incomplete_count == 2
    assert len(subject_ids) == 94


def test_recofit_adapter_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        next(iter_recordings(SOURCE_ROOT, limit=-1))
