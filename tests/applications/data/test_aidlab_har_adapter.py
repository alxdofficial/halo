from pathlib import Path

import numpy as np
import pytest

from applications.motion_monitoring.data.adapters.aidlab_har import iter_recordings


_SOURCE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "applications"
    / "motion_monitoring"
    / "data"
    / "sources"
    / "aidlab_har"
)
_ARCHIVE = _SOURCE_ROOT / "downloads" / "AIDLAB-HAR-DATASET_v3.zip"


@pytest.mark.skipif(not _ARCHIVE.is_file(), reason="local AIDLAB-HAR archive is absent")
def test_real_aidlab_har_release_satisfies_raw_recording_contract() -> None:
    recordings = list(iter_recordings(_SOURCE_ROOT))

    assert len(recordings) == 180
    assert len({recording.recording_id for recording in recordings}) == 180
    assert len({recording.subject_id for recording in recordings}) == 180
    assert len({recording.session_id for recording in recordings}) == 180
    assert all(recording.dataset == "aidlab_har" for recording in recordings)
    assert all(
        recording.metadata["subject_identity_scope"] == "recording_only"
        for recording in recordings
    )

    annotation_recordings = 0
    series_intervals = 0
    fiducial_intervals = 0
    recordings_with_invalid_acceleration = 0
    valid_norm_medians = []
    for recording in recordings:
        assert len(recording.streams) == 1
        stream = recording.streams[0]
        assert stream.stream_id == "chest_accelerometer"
        assert stream.placement == "chest"
        assert stream.channels == ("acc_x", "acc_y", "acc_z")
        assert stream.values.ndim == 2 and stream.values.shape[1] == 3
        assert stream.values.shape == stream.valid.shape
        assert stream.valid.dtype == np.bool_
        assert stream.nominal_rate_hz == pytest.approx(50.0)
        assert stream.gravity_state == "present"
        assert stream.metadata["source_unit"] == "g"
        assert stream.metadata["raw_gyroscope_available"] is False
        assert np.allclose(np.diff(stream.timestamps_sec), 0.02, rtol=0.0, atol=1e-12)
        assert np.isfinite(stream.values[stream.valid]).all()
        assert np.max(np.abs(stream.values[stream.valid])) <= 8.001

        sample_valid = stream.valid.all(axis=1)
        if not sample_valid.all():
            recordings_with_invalid_acceleration += 1
        valid_norm_medians.append(
            float(np.median(np.linalg.norm(stream.values[sample_valid], axis=1)))
        )

        if recording.events:
            annotation_recordings += 1
        duration_sec = stream.timestamps_sec[-1] + 1.0 / stream.nominal_rate_hz
        for event in recording.events:
            assert 0.0 <= event.start_sec < event.end_sec <= duration_sec + 1e-9
            assert event.label == recording.metadata["activity"]
            assert event.annotation_kind in {"series", "repetition_fiducial"}
            if event.annotation_kind == "series":
                series_intervals += 1
            else:
                assert event.metadata["source_interval_type"] == "repetition_marker"
                fiducial_intervals += 1

    assert annotation_recordings == 130
    assert series_intervals == 130
    assert fiducial_intervals == 1_486
    assert recordings_with_invalid_acceleration > 0
    assert 0.5 < float(np.median(valid_norm_medians)) < 1.5


@pytest.mark.skipif(not _ARCHIVE.is_file(), reason="local AIDLAB-HAR archive is absent")
def test_aidlab_har_limit_is_lazy_and_deterministic() -> None:
    first = list(iter_recordings(_ARCHIVE, limit=5))
    second = list(iter_recordings(_ARCHIVE, limit=5))

    assert len(first) == 5
    assert [item.recording_id for item in first] == [
        item.recording_id for item in second
    ]
    assert all(
        item.subject_id != item.metadata["source_subject_code"] for item in first
    )
