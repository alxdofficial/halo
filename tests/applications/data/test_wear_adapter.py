from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from applications.motion_monitoring.data.adapters.wear import iter_recordings


ROOT = (
    Path(__file__).resolve().parents[3]
    / "applications"
    / "motion_monitoring"
    / "data"
    / "sources"
    / "wear"
)
RAW = ROOT / "raw"

pytestmark = pytest.mark.skipif(
    not (RAW / "inertial_50hz" / "sbj_0.csv").is_file(),
    reason="local WEAR payload is absent",
)


def test_wear_loads_real_arm_streams_at_release_rate_and_units() -> None:
    recording = next(iter_recordings(ROOT, limit=1))
    assert recording.dataset == "wear"
    assert recording.recording_id == "wear:sbj_0"
    assert recording.subject_id == recording.session_id == "sbj_0"
    assert recording.split is None
    assert recording.metadata["annotation_usage"] == "scoring_only"
    assert recording.metadata["excluded_source_placements"] == ("right_leg", "left_leg")

    assert [stream.stream_id for stream in recording.streams] == [
        "right_arm",
        "left_arm",
    ]
    assert [stream.placement for stream in recording.streams] == [
        "right_wrist",
        "left_wrist",
    ]
    source = pd.read_csv(
        RAW / "inertial_50hz" / "sbj_0.csv",
        usecols=[
            "right_arm_acc_x",
            "right_arm_acc_y",
            "right_arm_acc_z",
            "left_arm_acc_x",
            "left_arm_acc_y",
            "left_arm_acc_z",
        ],
        nrows=1,
    )
    for stream, prefix in zip(recording.streams, ("right_arm", "left_arm")):
        expected = source.loc[0, [f"{prefix}_acc_{axis}" for axis in "xyz"]].to_numpy(
            dtype=np.float32
        )
        assert stream.channels == ("acc_x", "acc_y", "acc_z")
        assert stream.values.shape == stream.valid.shape == (139_725, 3)
        assert stream.values.dtype == np.float32
        assert stream.timestamps_sec.dtype == np.float64
        assert stream.valid.dtype == np.bool_
        assert stream.valid.all()
        assert stream.gravity_state == "present"
        assert stream.nominal_rate_hz == 50.0
        np.testing.assert_allclose(stream.values[0], expected)
        np.testing.assert_allclose(np.diff(stream.timestamps_sec), 0.02, atol=1e-12)
        median_norm_g = np.median(np.linalg.norm(stream.values, axis=1))
        assert 0.8 < median_norm_g < 1.2


def test_wear_annotations_are_scoring_only_and_preserve_background() -> None:
    recording = next(iter_recordings(ROOT, limit=1))
    assert recording.events[0].label == "NULL"
    assert recording.events[0].annotation_kind == "background"
    first_activity = next(
        event for event in recording.events if event.annotation_kind == "activity"
    )
    assert (first_activity.label, first_activity.start_sec, first_activity.end_sec) == (
        "jogging",
        0.92,
        99.26,
    )
    assert all(event.metadata["usage"] == "scoring_only" for event in recording.events)
    assert all(
        left.end_sec == pytest.approx(right.start_sec)
        for left, right in zip(recording.events[:-1], recording.events[1:])
    )
    assert recording.events[-1].end_sec == pytest.approx(139_725 / 50.0)
    assert not any(
        "label" in key.lower()
        for stream in recording.streams
        for key in stream.metadata
    )


def test_wear_full_release_ids_masks_intervals_and_source_quirks() -> None:
    summaries = {}
    total_activity_intervals = 0
    total_samples = 0
    label_set = set()
    for recording in iter_recordings(ROOT):
        summaries[recording.session_id] = recording
        total_samples += len(recording.streams[0].timestamps_sec)
        total_activity_intervals += sum(
            event.annotation_kind == "activity" for event in recording.events
        )
        label_set.update(
            event.label
            for event in recording.events
            if event.annotation_kind == "activity"
        )
        duration_sec = recording.metadata["inertial_duration_sec"]

        assert len(recording.streams) == 2
        assert (
            recording.metadata["matched_activity_interval_count"]
            == recording.metadata["json_activity_interval_count"]
        )
        assert all(
            0.0 <= event.start_sec < event.end_sec <= duration_sec
            for event in recording.events
        )
        for stream in recording.streams:
            assert stream.values.shape == stream.valid.shape
            assert stream.values.shape[1] == 3
            assert np.isfinite(stream.values[stream.valid]).all()
            assert np.all(np.diff(stream.timestamps_sec) > 0)
            assert stream.metadata["source_acceleration_unit"] == "g"

    assert len(summaries) == 24
    assert total_samples == 3_466_400
    assert total_activity_intervals == 719
    assert len(label_set) == 18
    assert summaries["sbj_18"].subject_id == "sbj_18"
    assert summaries["sbj_19"].subject_id == "sbj_19"
    assert summaries["sbj_18"].session_id == "sbj_18"
    assert summaries["sbj_19"].session_id == "sbj_19"
    assert summaries["sbj_18"].metadata["identity_alias_applied"] is False
    assert summaries["sbj_19"].metadata["identity_alias_applied"] is False
    assert summaries["sbj_18"].metadata["identity_linkage_group"] == (
        "wear_repeat_pair_unresolved"
    )
    assert summaries["sbj_19"].metadata["identity_linkage_group"] == (
        "wear_repeat_pair_unresolved"
    )
    assert summaries["sbj_0"].metadata["identity_linkage_group"] == (
        "wear_repeat_pair_unresolved"
    )
    assert summaries["sbj_14"].metadata["identity_linkage_group"] == (
        "wear_repeat_pair_unresolved"
    )
    assert all(summaries[f"sbj_{index}"].split is None for index in range(24))
    assert all(
        summaries[f"sbj_{index}"].metadata["official_partition"] == "training"
        for index in range(18)
    )
    assert all(
        summaries[f"sbj_{index}"].metadata["official_partition"] == "testing"
        for index in range(18, 24)
    )
    assert all(
        recording.metadata["official_partition_subject_disjoint"] is False
        for recording in summaries.values()
    )

    left = summaries["sbj_10"].streams[1]
    assert int((~left.valid).all(axis=1).sum()) == 51_392
    assert not left.valid[13_708:65_100].any()
    assert np.isnan(left.values[13_708:65_100]).all()
    assert left.valid[:13_708].all() and left.valid[65_100:].all()

    assert summaries["sbj_7"].metadata["row_label_only_intervals"] == (
        ("bench-dips", 2840.9, 2853.0),
    )
    assert summaries["sbj_21"].metadata[
        "declared_duration_difference_sec"
    ] == pytest.approx(92.0)
    assert summaries["sbj_21"].events[-1].end_sec == pytest.approx(2109.0)


def test_wear_limit_is_lazy_and_validated() -> None:
    assert list(iter_recordings(ROOT / "does-not-exist", limit=0)) == []
    assert [recording.session_id for recording in iter_recordings(RAW, limit=2)] == [
        "sbj_0",
        "sbj_1",
    ]
    with pytest.raises(ValueError, match="limit"):
        next(iter_recordings(ROOT, limit=-1))
