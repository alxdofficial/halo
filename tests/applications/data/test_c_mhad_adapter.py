from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from applications.motion_monitoring.data.adapters.c_mhad import iter_recordings
from applications.motion_monitoring.data.contracts import CANONICAL_CHANNELS


ROOT = (
    Path(__file__).resolve().parents[3]
    / "applications"
    / "motion_monitoring"
    / "data"
    / "sources"
    / "c_mhad"
    / "raw"
)


pytestmark = pytest.mark.skipif(
    not ROOT.is_dir(), reason="local C-MHAD payload is absent"
)


def _assert_recording(recording, expected_placement: str) -> None:
    assert recording.dataset == "c_mhad"
    assert recording.split is None
    assert recording.metadata["application_role"] == "sealed_external_evaluation"
    assert recording.subject_id.startswith("subject_")
    assert recording.session_id in recording.recording_id
    assert recording.metadata["annotation_usage"] == "scoring_only"
    assert recording.metadata["alignment_convention_uncertainty_samples"] == 2
    assert recording.metadata["alignment_convention_uncertainty_sec"] == pytest.approx(
        0.04
    )

    assert len(recording.streams) == 1
    stream = recording.streams[0]
    assert stream.placement == expected_placement
    assert stream.device == "Shimmer3"
    assert stream.channels == CANONICAL_CHANNELS
    assert stream.gravity_state == "present"
    assert stream.nominal_rate_hz == 50.0
    assert stream.values.shape == stream.valid.shape == (len(stream.timestamps_sec), 6)
    assert stream.values.dtype == np.float32
    assert stream.timestamps_sec.dtype == np.float64
    assert stream.valid.dtype == np.bool_
    assert stream.valid.all()
    assert np.all(np.diff(stream.timestamps_sec) > 0)
    assert 49.5 < 1.0 / np.median(np.diff(stream.timestamps_sec)) < 50.5
    assert 0.5 < np.median(np.linalg.norm(stream.values[:, :3], axis=1)) < 1.5
    assert np.quantile(np.abs(stream.values[:, 3:]), 0.99) < 20.0

    assert recording.events
    assert all(event.metadata["usage"] == "scoring_only" for event in recording.events)
    assert all(0.0 <= event.start_sec < event.end_sec for event in recording.events)
    assert all(
        event.start_sec >= stream.timestamps_sec[0]
        and event.end_sec <= stream.timestamps_sec[-1]
        for event in recording.events
    )
    assert all(
        0.0
        <= event.metadata["source_start_elapsed_sec"]
        < event.metadata["source_end_elapsed_sec"]
        <= 120.0
        for event in recording.events
    )
    assert not any("label" in key.lower() for key in stream.metadata)


def test_real_c_mhad_timelines_units_annotations_and_identity() -> None:
    wrist = list(iter_recordings(ROOT / "TVGestureApplication", limit=3))
    waist = list(iter_recordings(ROOT / "TransitionMovementsApplication", limit=3))
    assert len(wrist) == len(waist) == 3
    for recording in wrist:
        _assert_recording(recording, "right_wrist")
    for recording in waist:
        _assert_recording(recording, "middle_waist")

    first = wrist[0]
    assert first.recording_id == "tv_gestures_subject_01_run_01"
    assert first.subject_id == "subject_01"
    assert [
        (
            event.label,
            event.metadata["source_start_elapsed_sec"],
            event.metadata["source_end_elapsed_sec"],
        )
        for event in first.events[:2]
    ] == [
        ("circle_clockwise", 1.4, 4.2),
        ("swipe_right", 9.7, 11.0),
    ]
    video_origin = first.metadata["video_clock_origin_sec"]
    assert first.events[0].start_sec == pytest.approx(video_origin + 1.4)
    assert first.events[0].end_sec == pytest.approx(video_origin + 4.2)


def test_real_c_mhad_preserves_observed_rows_and_converts_source_units() -> None:
    recording = next(iter_recordings(ROOT / "TVGestureApplication", limit=1))
    source = ROOT / "TVGestureApplication" / "Subject1" / "inertial_sub1_tv1.csv"
    frame = pd.read_csv(source, skiprows=[1, 2])
    source_values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    stream = recording.streams[0]

    assert len(stream.timestamps_sec) == len(frame)
    expected_offset = 6003 - len(frame)
    assert recording.metadata["source_frame_offset_samples"] == expected_offset
    assert recording.metadata["initial_missing_samples"] == max(0, expected_offset)
    assert recording.metadata["extra_prefix_samples"] == max(0, -expected_offset)
    assert stream.timestamps_sec[0] == pytest.approx(frame.iloc[0, 0] / 1000.0)
    assert recording.metadata["video_clock_origin_sec"] == pytest.approx(
        stream.timestamps_sec[0] - expected_offset / 50.0
    )
    assert np.allclose(stream.values[0, :3], source_values[0, :3] / 9.80665)
    assert np.allclose(stream.values[0, 3:], np.deg2rad(source_values[0, 3:]))
    assert not np.allclose(stream.values[0], 0.0)


def test_c_mhad_limit_and_missing_root_contract() -> None:
    assert list(iter_recordings(ROOT, limit=0)) == []
    assert len(list(iter_recordings(ROOT, limit=2))) == 2
    with pytest.raises(ValueError, match="limit"):
        list(iter_recordings(ROOT, limit=-1))
    with pytest.raises(FileNotFoundError):
        list(iter_recordings(ROOT / "does-not-exist", limit=1))
