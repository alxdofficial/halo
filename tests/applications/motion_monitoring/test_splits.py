from __future__ import annotations

import numpy as np
import pytest

from applications.motion_monitoring.data.contracts import RawRecording, SensorStream
from applications.motion_monitoring.data.splits import (
    recording_key,
    subject_leakage_group,
    validate_subject_disjoint_assignments,
)


def _recording(
    recording_id: str,
    subject_id: str,
    *,
    dataset: str = "test",
    linkage: str | None = None,
) -> RawRecording:
    stream = SensorStream(
        stream_id="watch",
        placement="wrist",
        device="watch",
        timestamps_sec=np.asarray([0.0, 0.1]),
        values=np.ones((2, 3), dtype=np.float32),
        channels=("acc_x", "acc_y", "acc_z"),
        valid=np.ones((2, 3), dtype=np.bool_),
        gravity_state="present",
        nominal_rate_hz=10.0,
    )
    return RawRecording(
        dataset=dataset,
        recording_id=recording_id,
        subject_id=subject_id,
        session_id=recording_id,
        streams=(stream,),
        metadata={"identity_linkage_group": linkage},
    )


def test_canonical_subject_id_cannot_cross_splits() -> None:
    recordings = (_recording("a", "person"), _recording("b", "person"))
    with pytest.raises(ValueError, match="subject leakage groups cross splits"):
        validate_subject_disjoint_assignments(
            recordings,
            {recording_key(recordings[0]): "train", recording_key(recordings[1]): "test"},
        )


def test_unresolved_identity_linkage_is_one_conservative_group() -> None:
    recordings = (
        _recording("sbj_0", "sbj_0", dataset="wear", linkage="unresolved"),
        _recording("sbj_18", "sbj_18", dataset="wear", linkage="unresolved"),
    )
    assert subject_leakage_group(recordings[0]) == subject_leakage_group(recordings[1])
    with pytest.raises(ValueError, match="wear:linkage:unresolved"):
        validate_subject_disjoint_assignments(
            recordings,
            {recording_key(recordings[0]): "train", recording_key(recordings[1]): "test"},
        )


def test_complete_grouped_manifest_returns_auditable_membership() -> None:
    recordings = (_recording("a", "one"), _recording("b", "two"))
    groups = validate_subject_disjoint_assignments(
        recordings,
        {recording_key(recordings[0]): "train", recording_key(recordings[1]): "test"},
    )
    assert set(groups) == {"test:subject:one", "test:subject:two"}


def test_manifest_rejects_missing_and_unknown_recording_keys() -> None:
    recording = _recording("a", "one")
    with pytest.raises(ValueError, match="no split assignment"):
        validate_subject_disjoint_assignments((recording,), {})
    with pytest.raises(ValueError, match="unknown recording assignments"):
        validate_subject_disjoint_assignments(
            (recording,),
            {recording_key(recording): "train", ("test", "extra"): "test"},
        )
