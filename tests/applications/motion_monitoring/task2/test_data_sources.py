from __future__ import annotations

import numpy as np
import pytest

from applications.motion_monitoring.data.contracts import EventInterval, RawRecording, SensorStream
from applications.motion_monitoring.task2.data_sources import (
    TASK2_DEFERRED_EVALUATION_DATASETS,
    TASK2_READY_EVALUATION_DATASETS,
    TASK2_TRAIN_DATASETS,
    is_selected_event,
    spec_for,
    validate_selected_recording,
)


def _recording(dataset: str, *, channels: tuple[str, ...]) -> RawRecording:
    times = np.arange(10, dtype=np.float64) / 50.0
    stream = SensorStream(
        stream_id="watch_wrist",
        placement="dominant_wrist" if dataset == "harmes" else "wrist",
        device="smartwatch",
        timestamps_sec=times,
        values=np.ones((len(times), len(channels)), dtype=np.float32),
        channels=channels,
        valid=np.ones((len(times), len(channels)), dtype=np.bool_),
        gravity_state="present",
        nominal_rate_hz=50.0,
    )
    return RawRecording(
        dataset=dataset,
        recording_id=f"{dataset}:one",
        subject_id="s1",
        session_id="session1",
        streams=(stream,),
    )


def test_task2_minimal_source_roles_are_fixed() -> None:
    assert TASK2_TRAIN_DATASETS == ("harmes", "crossfit")
    assert TASK2_READY_EVALUATION_DATASETS == ("monipar", "kneepad")
    assert TASK2_DEFERRED_EVALUATION_DATASETS == ()
    assert spec_for("kneepad").role == "evaluation"


def test_monipar_rest_extension_is_not_a_primary_evaluation_execution() -> None:
    recording = _recording("monipar", channels=("acc_x", "acc_y", "acc_z"))
    primary = EventInterval(0.0, 0.1, "gait", "bounded_execution")
    rest = EventInterval(0.0, 0.1, "resting", "bounded_execution")
    assert is_selected_event(recording, primary)
    assert not is_selected_event(recording, rest)
    validate_selected_recording(recording)


def test_source_guards_reject_wrong_configuration_and_pending_data() -> None:
    harmes = _recording("harmes", channels=("acc_x", "acc_y", "acc_z"))
    validate_selected_recording(harmes)
    wrong_monipar = _recording(
        "monipar", channels=("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
    )
    with pytest.raises(ValueError, match="acceleration-only"):
        validate_selected_recording(wrong_monipar)
    kneepad = _recording("kneepad", channels=("acc_x", "acc_y", "acc_z"))
    with pytest.raises(ValueError, match="within one visit"):
        validate_selected_recording(kneepad)


def test_record_path_excludes_events_the_source_contract_rejects(monkeypatch) -> None:
    """MoniPar's tremor-graded resting runs are bounded executions in the cache
    but not part of the primary protocol, so a record pool must not pick them up."""

    from applications.motion_monitoring.task2 import records

    recording = _recording("monipar", channels=("acc_x", "acc_y", "acc_z"))
    gait = EventInterval(0.0, 0.1, "gait", "bounded_execution")
    rest = EventInterval(0.1, 0.2, "resting", "bounded_execution")
    recording = RawRecording(
        dataset=recording.dataset,
        recording_id=recording.recording_id,
        subject_id=recording.subject_id,
        session_id=recording.session_id,
        streams=recording.streams,
        events=(gait, rest),
        metadata={"week": 1},
    )

    class _Representations:
        def __init__(self):
            self.calls = []

        def get(self, *args, **kwargs):
            self.calls.append(args)
            raise KeyError("no representation")

    # With no representation the loop skips the stream entirely; the point of the
    # test is that selection happens before any cropping is attempted.
    seen = []
    original = records.is_selected_event

    def spy(rec, event):
        seen.append(event.label)
        return original(rec, event)

    monkeypatch.setattr(records, "is_selected_event", spy)
    representations = _Representations()
    list(records.iter_execution_records("monipar", [recording], representations))
    assert len(representations.calls) == 1
    assert representations.calls[0][1].endswith("::bounded_event_0")
    assert original(recording, gait) and not original(recording, rest)
