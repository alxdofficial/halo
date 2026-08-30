from itertools import islice
from pathlib import Path

import numpy as np
import pytest

from applications.motion_monitoring.data.adapters.openpack import iter_recordings


SOURCE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "applications"
    / "motion_monitoring"
    / "data"
    / "sources"
    / "openpack"
)
DOWNLOADS = SOURCE_ROOT / "downloads"

pytestmark = pytest.mark.skipif(
    not (DOWNLOADS / "U0101.zip").exists(),
    reason="local OpenPack payload is not available",
)


def test_real_openpack_recordings_have_native_clock_units_and_annotations() -> None:
    recordings = list(islice(iter_recordings(DOWNLOADS / "U0101.zip"), 3))
    assert len(recordings) == 3

    recording = recordings[0]
    stream = recording.streams[0]
    assert recording.dataset == "openpack"
    assert recording.subject_id == "U0101"
    assert recording.session_id == "U0101/S0100"
    assert stream.stream_id == "atr01"
    assert stream.placement == "right_wrist"
    assert stream.device == "ATR TSND151"
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
    assert stream.valid.dtype == np.bool_
    assert stream.valid.all()
    assert np.all(np.diff(stream.timestamps_sec) > 0)
    assert 30.0 <= stream.metadata["measured_rate_hz"] <= 34.0
    assert np.isclose(stream.values[0, 0], 0.2838, atol=1e-6)
    assert np.isclose(stream.values[0, 3], np.deg2rad(-3.21), atol=1e-6)
    assert 0.8 < np.median(np.linalg.norm(stream.values[:, :3], axis=1)) < 1.3

    annotation_kinds = {event.annotation_kind for event in recording.events}
    assert annotation_kinds == {"fine_action", "operation", "box_cycle"}
    assert sum(event.annotation_kind == "box_cycle" for event in recording.events) == 20
    assert all(event.end_sec > event.start_sec for event in recording.events)
    assert all(
        event.end_sec >= stream.timestamps_sec[0]
        and event.start_sec <= stream.timestamps_sec[-1]
        for event in recording.events
    )


def test_real_openpack_gap_is_split_without_interpolation() -> None:
    recordings = list(iter_recordings(DOWNLOADS / "U0108.zip"))
    gap_parts = [
        recording for recording in recordings if recording.session_id.endswith("/S0400")
    ]
    assert len(gap_parts) == 2
    assert {recording.metadata["quality_part_count"] for recording in gap_parts} == {2}
    assert {
        recording.metadata["source_quality_part_count"] for recording in gap_parts
    } == {3}
    assert {
        recording.metadata["dropped_short_part_count"] for recording in gap_parts
    } == {1}
    assert {
        recording.metadata["dropped_short_sample_count"] for recording in gap_parts
    } == {1}

    for recording in gap_parts:
        timestamps = recording.streams[0].timestamps_sec
        assert np.all(np.diff(timestamps) > 0)
        assert np.all(np.diff(timestamps) <= 0.2)

    joined = np.concatenate(
        [recording.streams[0].timestamps_sec for recording in gap_parts]
    )
    hard_gaps = np.diff(joined)[np.diff(joined) > 0.2]
    assert np.allclose(hard_gaps, [7.14], atol=1e-6)

    assert all(
        event.start_sec >= recording.streams[0].timestamps_sec[0]
        and event.end_sec <= recording.streams[0].timestamps_sec[-1]
        for recording in gap_parts
        for event in recording.events
    )


def test_real_openpack_alias_and_invalid_annotation_are_explicit() -> None:
    aliased = next(iter_recordings(DOWNLOADS / "U0202.zip", limit=1))
    assert aliased.subject_id == "U0105"
    assert aliased.metadata["source_user_id"] == "U0202"
    assert aliased.metadata["identity_alias_applied"] is True

    recordings = list(iter_recordings(DOWNLOADS / "U0208.zip"))
    session = next(
        recording for recording in recordings if recording.session_id.endswith("/S0200")
    )
    assert session.metadata["excluded_zero_duration_annotations"] == 1
    assert session.metadata["excluded_other_invalid_annotations"] == 0
    assert all(event.end_sec > event.start_sec for event in session.events)
