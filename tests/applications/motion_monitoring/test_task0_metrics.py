import pytest

from applications.motion_monitoring.data.contracts import EventInterval
from applications.motion_monitoring.task0.contracts import MotionProposal
from applications.motion_monitoring.task0.metrics import (
    evaluate_intervals,
    interval_iou,
    match_intervals_one_to_one,
)


def _proposal(start: float, end: float, *, score: float = 1.0) -> MotionProposal:
    return MotionProposal(
        dataset="example",
        recording_id="recording",
        subject_id="subject",
        session_id="session",
        stream_ids=("watch",),
        placements=("wrist",),
        start_sec=start,
        end_sec=end,
        score=score,
        start_boundary_change_score=1.0,
        end_boundary_change_score=1.0,
        valid_fraction=1.0,
        uncertain=False,
        refinement="none",
        feature_summary={},
    )


def _event(start: float, end: float) -> EventInterval:
    return EventInterval(start, end, "movement")


def test_perfect_intervals_have_perfect_metrics() -> None:
    events = (_event(1.0, 2.0), _event(4.0, 6.0))
    proposals = (_proposal(1.0, 2.0, score=0.9), _proposal(4.0, 6.0, score=0.8))

    metrics = evaluate_intervals(
        proposals,
        events,
        exhaustive_background=True,
        recording_duration_sec=3600.0,
    )

    assert metrics.event_recall == 1.0
    assert metrics.event_precision == 1.0
    assert metrics.event_f1 == 1.0
    assert metrics.matched_iou_mean == 1.0
    assert metrics.start_boundary_mae_sec == 0.0
    assert metrics.end_boundary_mae_sec == 0.0
    assert metrics.boundary_f1 == 1.0
    assert metrics.oversegmented_event_count == 0
    assert metrics.undersegmented_proposal_count == 0
    assert metrics.average_precision == 1.0
    assert metrics.false_proposals_per_hour == 0.0


def test_one_to_one_matching_uses_best_iou_before_input_order() -> None:
    events = (_event(0.0, 4.0), _event(4.0, 6.0))
    proposals = (_proposal(0.0, 6.0), _proposal(0.0, 4.0))

    matches = match_intervals_one_to_one(proposals, events, iou_threshold=0.3)

    assert [(match.proposal_index, match.event_index) for match in matches] == [
        (0, 1),
        (1, 0),
    ]
    assert matches[0].iou == pytest.approx(1.0 / 3.0)
    assert matches[1].iou == 1.0


def test_boundary_f1_penalizes_extra_and_inaccurate_boundaries() -> None:
    events = (_event(1.0, 3.0),)
    proposals = (
        _proposal(1.1, 3.8, score=0.9),
        _proposal(1.5, 2.0, score=0.2),
    )

    metrics = evaluate_intervals(
        proposals,
        events,
        exhaustive_background=False,
        iou_threshold=0.25,
        boundary_tolerance_sec=0.25,
    )

    assert metrics.start_boundary_mae_sec == pytest.approx(0.1)
    assert metrics.end_boundary_mae_sec == pytest.approx(0.8)
    assert metrics.boundary_precision == pytest.approx(0.5)
    assert metrics.boundary_recall == pytest.approx(0.5)
    assert metrics.boundary_f1 == pytest.approx(0.5)


def test_overlap_structure_reports_over_and_under_segmentation() -> None:
    events = (_event(1.0, 3.0), _event(4.0, 6.0))
    proposals = (
        _proposal(0.5, 2.0),
        _proposal(2.0, 5.0),
        _proposal(4.5, 6.5),
    )

    metrics = evaluate_intervals(
        proposals,
        events,
        exhaustive_background=False,
        iou_threshold=0.25,
    )

    assert metrics.oversegmented_event_count == 2
    assert metrics.oversegmentation_excess_count == 2
    assert metrics.undersegmented_proposal_count == 1
    assert metrics.undersegmentation_excess_count == 1


def test_non_exhaustive_annotations_suppress_ap_and_false_rate() -> None:
    metrics = evaluate_intervals(
        (_proposal(1.0, 2.0), _proposal(8.0, 9.0)),
        (_event(1.0, 2.0),),
        exhaustive_background=False,
        recording_duration_sec=10.0,
    )

    assert metrics.average_precision is None
    assert metrics.false_proposals_per_hour is None
    assert metrics.event_precision is None
    assert metrics.event_f1 is None


def test_exhaustive_annotations_score_ranked_ap_and_false_rate() -> None:
    metrics = evaluate_intervals(
        (_proposal(8.0, 9.0, score=0.9), _proposal(1.0, 2.0, score=0.8)),
        (_event(1.0, 2.0),),
        exhaustive_background=True,
        recording_duration_sec=3600.0,
    )

    assert metrics.average_precision == pytest.approx(0.5)
    assert metrics.false_proposals_per_hour == 1.0


def test_exhaustive_evaluation_requires_observed_duration() -> None:
    with pytest.raises(ValueError, match="recording_duration_sec is required"):
        evaluate_intervals((), (), exhaustive_background=True)


def test_absolute_source_clock_uses_explicit_recording_origin() -> None:
    metrics = evaluate_intervals(
        (_proposal(1_000_001.0, 1_000_002.0),),
        (_event(1_000_001.0, 1_000_002.0),),
        exhaustive_background=True,
        recording_start_sec=1_000_000.0,
        recording_duration_sec=10.0,
    )
    assert metrics.event_recall == 1.0


def test_non_exhaustive_boundary_precision_does_not_score_unknown_background() -> None:
    metrics = evaluate_intervals(
        (_proposal(1.0, 2.0), _proposal(8.0, 9.0)),
        (_event(1.0, 2.0),),
        exhaustive_background=False,
    )
    assert metrics.boundary_precision == 1.0


def test_interval_iou_rejects_invalid_intervals() -> None:
    assert interval_iou(0.0, 2.0, 1.0, 3.0) == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError, match="greater than starts"):
        interval_iou(1.0, 1.0, 1.0, 2.0)


def test_matching_maximizes_valid_cardinality_before_iou(monkeypatch) -> None:
    import applications.motion_monitoring.task0.metrics as metrics_module

    proposals = (_proposal(0.0, 1.0), _proposal(2.0, 3.0))
    events = (_event(10.0, 11.0), _event(12.0, 13.0))
    monkeypatch.setattr(
        metrics_module,
        "_interval_iou_matrix",
        lambda _proposals, _events: __import__("numpy").asarray(
            [[0.9, 0.5], [0.5, 0.49]]
        ),
    )
    matches = match_intervals_one_to_one(proposals, events, iou_threshold=0.5)
    assert len(matches) == 2
