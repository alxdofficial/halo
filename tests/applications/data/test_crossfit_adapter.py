from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from applications.motion_monitoring.data.adapters.crossfit import iter_recordings


RAW_ROOT = (
    Path(__file__).resolve().parents[3]
    / "applications"
    / "motion_monitoring"
    / "data"
    / "sources"
    / "crossfit"
    / "raw"
)
DATA_ROOT = (
    RAW_ROOT
    / "HAR_Crossfit_Sensors_Data"
    / "data"
    / "constrained_workout"
    / "preprocessed_numpy_data"
)

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "np_exercise_data").is_dir(),
    reason="local CrossFit authored release is absent",
)


def test_crossfit_loads_real_parent_and_repetitions_in_canonical_units() -> None:
    recordings = list(iter_recordings(RAW_ROOT, limit=4))
    assert len(recordings) == 4
    parent, repetition = recordings[:2]

    assert parent.dataset == "crossfit"
    assert parent.recording_id == "crossfit:exercise:137"
    assert parent.subject_id == "P1"
    assert parent.session_id == repetition.session_id == "crossfit:exercise:137"
    assert parent.metadata["source_level"] == "exercise"
    assert repetition.metadata["source_level"] == "repetition"
    assert repetition.metadata["parent_exercise_recording_id"] == parent.recording_id
    assert repetition.metadata["repetition_index"] == 0
    assert repetition.metadata["duplicates_parent_exercise_signal"] is True

    stream = parent.streams[0]
    source = np.load(
        DATA_ROOT / "np_exercise_data" / "Box jumps" / "Box jumps_137.npy",
        allow_pickle=False,
    )
    assert stream.channels == (
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    )
    assert stream.values.shape == stream.valid.shape == (source.shape[1], 6)
    assert stream.values.dtype == np.float32
    assert stream.valid.dtype == np.bool_
    assert stream.valid.all()
    np.testing.assert_allclose(np.diff(stream.timestamps_sec), 0.01, atol=1e-12)
    np.testing.assert_allclose(stream.values[0, :3], source[:3, 0] / 9.80665)
    np.testing.assert_allclose(stream.values[0, 3:], source[3:6, 0])
    assert stream.gravity_state == "present"
    assert stream.nominal_rate_hz == 100.0
    assert 0.5 < np.median(np.linalg.norm(stream.values[:, :3], axis=1)) < 3.0
    assert parent.events[0].label == "Box jumps"
    assert parent.events[0].annotation_kind == "exercise_sequence"
    assert repetition.events[0].annotation_kind == "repetition"


def test_crossfit_intervals_and_release_caveats_are_explicit() -> None:
    for recording in iter_recordings(RAW_ROOT, limit=32):
        stream = recording.streams[0]
        assert np.isfinite(stream.values[stream.valid]).all()
        assert np.all(np.diff(stream.timestamps_sec) > 0)
        for event in recording.events:
            assert event.start_sec >= stream.timestamps_sec[0]
            assert event.end_sec > event.start_sec
            assert event.end_sec <= stream.timestamps_sec[-1] + 0.010000000001

        assert recording.metadata["paper_participant_count"] == 54
        assert recording.metadata["release_participant_count"] == 57
        assert recording.metadata["participant_count_mismatch"] is True
        assert recording.metadata["null_only_participant_codes"] == (
            "P52",
            "P53",
            "P54",
            "P55",
            "P56",
            "P57",
            "P58",
        )


def test_crossfit_pseudo_repetition_fragments_are_not_events() -> None:
    fragments = [
        recording
        for recording in iter_recordings(RAW_ROOT)
        if recording.metadata["pseudo_repetition_fragment"]
    ]
    assert {
        Path(recording.metadata["source_file"]).name for recording in fragments
    } == {
        "Burpees_495_9.npy",
        "Push ups_276_13.npy",
        "Squats_158_10.npy",
        "Squats_261_10.npy",
        "Squats_281_13.npy",
        "Squats_310_9.npy",
    }
    for fragment in fragments:
        assert fragment.metadata["source_level"] == "repetition"
        assert 8 <= len(fragment.streams[0].timestamps_sec) <= 16
        assert fragment.events == ()
        assert fragment.metadata["event_excluded"] is True
        assert "pseudo-repetition" in fragment.metadata["event_exclusion_reason"]


def test_crossfit_limit_zero_is_lazy_and_empty() -> None:
    assert list(iter_recordings(RAW_ROOT / "does-not-need-to-exist", limit=0)) == []


def test_crossfit_accepts_release_or_preprocessed_root() -> None:
    release_root = RAW_ROOT / "HAR_Crossfit_Sensors_Data"
    from_release = next(iter_recordings(release_root, limit=1))
    from_preprocessed = next(iter_recordings(DATA_ROOT, limit=1))
    assert from_release.recording_id == from_preprocessed.recording_id
