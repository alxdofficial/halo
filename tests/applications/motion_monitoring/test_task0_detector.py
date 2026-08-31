from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from applications.motion_monitoring.data.contracts import RawRecording
from applications.motion_monitoring.task0.contracts import (
    EvidenceSequence,
    ProposalConfig,
    RefinementConfig,
    RobustFeatureScaler,
)
from applications.motion_monitoring.task0.detector import Task0Detector


def _sequence() -> EvidenceSequence:
    starts = np.arange(20, dtype=float) * 0.1
    features = np.zeros((20, 2), dtype=float)
    features[5:14, 0] = 4.0
    valid = np.zeros_like(features, dtype=bool)
    valid[:, 0] = True
    return EvidenceSequence(
        dataset="synthetic",
        recording_id="recording",
        subject_id="subject",
        session_id="session",
        stream_id="watch",
        placement="left wrist",
        window_start_sec=starts,
        window_end_sec=starts + 0.5,
        features=features,
        feature_valid=valid,
        valid_fraction=np.ones(20),
        constant_fraction=np.zeros(20),
    )


def _detector(*, refinement: bool = False) -> Task0Detector:
    return Task0Detector(
        RobustFeatureScaler(
            center=np.zeros(2), scale=np.ones(2), observed=np.array([True, False])
        ),
        proposal_config=ProposalConfig(
            start_threshold=3.0,
            continue_threshold=1.0,
            minimum_duration_seconds=0.5,
        ),
        refinement_config=RefinementConfig(enabled=refinement),
    )


def test_hysteresis_proposes_bounded_motion() -> None:
    proposals = _detector().detect_evidence(_sequence())
    assert len(proposals) == 1
    assert proposals[0].start_sec == 0.7
    assert proposals[0].end_sec == 1.6
    assert proposals[0].stream_ids == ("watch",)
    assert not proposals[0].uncertain


def test_detector_round_trip(tmp_path) -> None:
    path = tmp_path / "detector.json"
    detector = _detector()
    detector.save(path)
    loaded = Task0Detector.load(path)
    assert loaded.to_dict() == detector.to_dict()
    assert (
        loaded.detect_evidence(_sequence())[0].to_dict()
        == detector.detect_evidence(_sequence())[0].to_dict()
    )


def test_pelt_refinement_stays_inside_search_margin() -> None:
    detector = _detector(refinement=True)
    proposals = detector.detect_evidence(_sequence())
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.refinement == "pelt_both"
    assert proposal.start_sec == pytest.approx(0.7)
    assert proposal.end_sec == pytest.approx(1.6)


def test_low_evidence_gap_is_not_merged_despite_window_overlap() -> None:
    sequence = _sequence()
    features = sequence.features.copy()
    features[:] = 0.0
    features[2:8, 0] = 4.0
    features[9:15, 0] = 4.0
    proposals = _detector().detect_evidence(replace(sequence, features=features))
    assert len(proposals) == 2
    assert proposals[0].end_sec <= proposals[1].start_sec


def test_short_fragments_are_merged_before_minimum_duration_filtering() -> None:
    sequence = _sequence()
    features = np.zeros_like(sequence.features)
    features[2, 0] = 4.0
    features[3, 0] = 0.8
    features[4, 0] = 4.0
    detector = Task0Detector(
        _detector().scaler,
        proposal_config=ProposalConfig(
            start_threshold=3.0,
            continue_threshold=1.0,
            minimum_duration_seconds=0.25,
            merge_gap_seconds=0.25,
            merge_floor=0.75,
        ),
        refinement_config=RefinementConfig(enabled=False),
    )

    proposals = detector.detect_evidence(replace(sequence, features=features))

    assert len(proposals) == 1
    assert proposals[0].end_sec - proposals[0].start_sec >= 0.25


def test_multi_stream_recording_requires_explicit_selection() -> None:
    from tests.applications.motion_monitoring.test_task0_evidence import _recording

    recording, stream = _recording(rate_hz=50.0)
    second = replace(stream, stream_id="second_watch")
    recording = RawRecording(
        dataset=recording.dataset,
        recording_id=recording.recording_id,
        subject_id=recording.subject_id,
        session_id=recording.session_id,
        streams=(stream, second),
    )
    with pytest.raises(ValueError, match="select stream_ids explicitly"):
        _detector().detect_recording(recording)
    with pytest.raises(ValueError, match="exactly one selected stream"):
        _detector().detect_recording(recording, stream_ids=("watch", "second_watch"))


def test_pelt_input_is_bounded_around_boundaries(monkeypatch) -> None:
    observed_lengths = []

    class FakePelt:
        def __init__(self, **_kwargs) -> None:
            self.length = 0

        def fit(self, values):
            self.length = len(values)
            observed_lengths.append(self.length)
            return self

        def predict(self, **_kwargs):
            return [self.length]

    monkeypatch.setitem(
        __import__("sys").modules,
        "ruptures",
        SimpleNamespace(Pelt=FakePelt),
    )
    count = 10_000
    starts = np.arange(count, dtype=float) * 0.1
    features = np.zeros((count, 2), dtype=float)
    features[5_000:5_010, 0] = 4.0
    sequence = replace(
        _sequence(),
        window_start_sec=starts,
        window_end_sec=starts + 0.5,
        features=features,
        feature_valid=np.column_stack(
            [np.ones(count, dtype=bool), np.zeros(count, dtype=bool)]
        ),
        valid_fraction=np.ones(count),
        constant_fraction=np.zeros(count),
    )
    proposals = _detector(refinement=True).detect_evidence(sequence)
    assert len(proposals) == 1
    assert observed_lengths
    assert max(observed_lengths) <= 20
