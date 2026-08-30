"""Interval-level evaluation metrics for Task 0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from applications.motion_monitoring.data.contracts import EventInterval
from applications.motion_monitoring.task0.contracts import MotionProposal


@dataclass(frozen=True)
class IntervalMatch:
    """One deterministic proposal-to-event assignment."""

    proposal_index: int
    event_index: int
    iou: float


@dataclass(frozen=True)
class Task0IntervalMetrics:
    """Metrics for one quality-contiguous recording."""

    matches: tuple[IntervalMatch, ...]
    proposal_count: int
    event_count: int
    event_precision: float | None
    event_recall: float | None
    event_f1: float | None
    matched_iou_mean: float | None
    start_boundary_mae_sec: float | None
    end_boundary_mae_sec: float | None
    boundary_mae_sec: float | None
    boundary_precision: float | None
    boundary_recall: float | None
    boundary_f1: float | None
    oversegmented_event_count: int
    oversegmentation_excess_count: int
    undersegmented_proposal_count: int
    undersegmentation_excess_count: int
    average_precision: float | None
    false_proposals_per_hour: float | None


def interval_iou(
    first_start_sec: float,
    first_end_sec: float,
    second_start_sec: float,
    second_end_sec: float,
) -> float:
    """Return temporal intersection over union for two valid half-open intervals."""

    values = np.asarray(
        [first_start_sec, first_end_sec, second_start_sec, second_end_sec],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("interval boundaries must be finite")
    if first_end_sec <= first_start_sec or second_end_sec <= second_start_sec:
        raise ValueError("interval ends must be greater than starts")

    intersection = max(
        0.0,
        min(first_end_sec, second_end_sec) - max(first_start_sec, second_start_sec),
    )
    union = (
        first_end_sec
        - first_start_sec
        + second_end_sec
        - second_start_sec
        - intersection
    )
    return float(intersection / union)


def match_intervals_one_to_one(
    proposals: Sequence[MotionProposal],
    events: Sequence[EventInterval],
    *,
    iou_threshold: float = 0.5,
) -> tuple[IntervalMatch, ...]:
    """Find a deterministic maximum-IoU one-to-one assignment."""

    _validate_iou_threshold(iou_threshold)
    if not proposals or not events:
        return ()
    return _match_from_iou_matrix(
        _interval_iou_matrix(proposals, events), iou_threshold=iou_threshold
    )


def _interval_iou_matrix(
    proposals: Sequence[MotionProposal], events: Sequence[EventInterval]
) -> np.ndarray:
    if not proposals or not events:
        return np.zeros((len(proposals), len(events)), dtype=np.float64)
    proposal_starts = np.asarray([item.start_sec for item in proposals])[:, None]
    proposal_ends = np.asarray([item.end_sec for item in proposals])[:, None]
    event_starts = np.asarray([item.start_sec for item in events])[None, :]
    event_ends = np.asarray([item.end_sec for item in events])[None, :]
    intersection = np.maximum(
        0.0,
        np.minimum(proposal_ends, event_ends)
        - np.maximum(proposal_starts, event_starts),
    )
    union = proposal_ends - proposal_starts + event_ends - event_starts - intersection
    return intersection / union


def _match_from_iou_matrix(
    ious: np.ndarray, *, iou_threshold: float
) -> tuple[IntervalMatch, ...]:
    if not ious.size:
        return ()
    proposal_count, event_count = ious.shape
    eligible = ious >= iou_threshold
    cardinality_reward = float(min(proposal_count, event_count) + 1)
    reward = eligible * (cardinality_reward + ious)
    # The tiny index perturbation gives reproducible tie-breaking without
    # changing either objective.
    tie_break = (
        np.arange(proposal_count)[:, None] * event_count
        + np.arange(event_count)[None, :]
    ) * np.finfo(np.float64).eps
    proposal_indices, event_indices = linear_sum_assignment(-(reward - tie_break))
    matches = [
        IntervalMatch(
            proposal_index=int(proposal_index),
            event_index=int(event_index),
            iou=float(ious[proposal_index, event_index]),
        )
        for proposal_index, event_index in zip(proposal_indices, event_indices)
        if ious[proposal_index, event_index] >= iou_threshold
    ]

    return tuple(sorted(matches, key=lambda match: match.proposal_index))


def evaluate_intervals(
    proposals: Sequence[MotionProposal],
    events: Sequence[EventInterval],
    *,
    exhaustive_background: bool,
    recording_duration_sec: float | None = None,
    recording_start_sec: float = 0.0,
    iou_threshold: float = 0.5,
    boundary_tolerance_sec: float = 0.5,
) -> Task0IntervalMetrics:
    """Evaluate Task-0 proposals against class-agnostic reference intervals.

    Average precision and false proposals per hour require exhaustive annotation
    of the monitored background. They are deliberately suppressed otherwise.
    """

    proposals = tuple(proposals)
    events = tuple(events)
    _validate_iou_threshold(iou_threshold)
    if not np.isfinite(boundary_tolerance_sec) or boundary_tolerance_sec < 0:
        raise ValueError("boundary_tolerance_sec must be finite and non-negative")
    if recording_duration_sec is not None and (
        not np.isfinite(recording_duration_sec) or recording_duration_sec <= 0
    ):
        raise ValueError("recording_duration_sec must be finite and positive")
    if not np.isfinite(recording_start_sec):
        raise ValueError("recording_start_sec must be finite")
    if exhaustive_background and recording_duration_sec is None:
        raise ValueError(
            "recording_duration_sec is required when background is exhaustive"
        )
    if recording_duration_sec is not None:
        earliest_start = min(
            (item.start_sec for item in (*proposals, *events)),
            default=recording_start_sec,
        )
        latest_end = max(
            (item.end_sec for item in (*proposals, *events)),
            default=recording_start_sec,
        )
        recording_end_sec = recording_start_sec + recording_duration_sec
        if (
            earliest_start < recording_start_sec - 1e-9
            or latest_end > recording_end_sec + 1e-9
        ):
            raise ValueError("an interval extends beyond the recording clock bounds")

    ious = _interval_iou_matrix(proposals, events)
    matches = _match_from_iou_matrix(ious, iou_threshold=iou_threshold)
    matched_count = len(matches)
    event_recall = matched_count / len(events) if events else None
    event_precision = None
    if exhaustive_background:
        event_precision = (
            matched_count / len(proposals) if proposals else 1.0 if not events else 0.0
        )
    event_f1 = _f1(event_precision, event_recall)
    matched_iou_mean = (
        float(np.mean([match.iou for match in matches])) if matches else None
    )

    start_errors = np.asarray(
        [
            abs(
                proposals[match.proposal_index].start_sec
                - events[match.event_index].start_sec
            )
            for match in matches
        ],
        dtype=np.float64,
    )
    end_errors = np.asarray(
        [
            abs(
                proposals[match.proposal_index].end_sec
                - events[match.event_index].end_sec
            )
            for match in matches
        ],
        dtype=np.float64,
    )
    start_boundary_mae = float(np.mean(start_errors)) if matches else None
    end_boundary_mae = float(np.mean(end_errors)) if matches else None
    boundary_mae = (
        float(np.mean(np.concatenate((start_errors, end_errors)))) if matches else None
    )

    boundary_true_positives = int(
        np.count_nonzero(start_errors <= boundary_tolerance_sec)
        + np.count_nonzero(end_errors <= boundary_tolerance_sec)
    )
    predicted_boundary_count = 2 * (
        len(proposals) if exhaustive_background else len(matches)
    )
    reference_boundary_count = 2 * len(events)
    boundary_precision = (
        boundary_true_positives / predicted_boundary_count
        if predicted_boundary_count
        else None
    )
    boundary_recall = (
        boundary_true_positives / reference_boundary_count
        if reference_boundary_count
        else None
    )
    boundary_f1 = _f1(boundary_precision, boundary_recall)

    proposal_overlaps_per_event = np.sum(ious > 0.0, axis=0).tolist()
    event_overlaps_per_proposal = np.sum(ious > 0.0, axis=1).tolist()
    oversegmented_event_count = sum(count > 1 for count in proposal_overlaps_per_event)
    oversegmentation_excess_count = sum(
        max(0, count - 1) for count in proposal_overlaps_per_event
    )
    undersegmented_proposal_count = sum(
        count > 1 for count in event_overlaps_per_proposal
    )
    undersegmentation_excess_count = sum(
        max(0, count - 1) for count in event_overlaps_per_proposal
    )

    average_precision: float | None = None
    false_proposals_per_hour: float | None = None
    if exhaustive_background:
        average_precision = _average_precision(
            proposals, events, ious=ious, iou_threshold=iou_threshold
        )
        false_count = len(proposals) - matched_count
        assert recording_duration_sec is not None
        false_proposals_per_hour = false_count / (recording_duration_sec / 3600.0)

    return Task0IntervalMetrics(
        matches=matches,
        proposal_count=len(proposals),
        event_count=len(events),
        event_precision=event_precision,
        event_recall=event_recall,
        event_f1=event_f1,
        matched_iou_mean=matched_iou_mean,
        start_boundary_mae_sec=start_boundary_mae,
        end_boundary_mae_sec=end_boundary_mae,
        boundary_mae_sec=boundary_mae,
        boundary_precision=boundary_precision,
        boundary_recall=boundary_recall,
        boundary_f1=boundary_f1,
        oversegmented_event_count=oversegmented_event_count,
        oversegmentation_excess_count=oversegmentation_excess_count,
        undersegmented_proposal_count=undersegmented_proposal_count,
        undersegmentation_excess_count=undersegmentation_excess_count,
        average_precision=average_precision,
        false_proposals_per_hour=false_proposals_per_hour,
    )


def _average_precision(
    proposals: Sequence[MotionProposal],
    events: Sequence[EventInterval],
    *,
    ious: np.ndarray,
    iou_threshold: float,
) -> float:
    if not events:
        return 1.0 if not proposals else 0.0

    ranked_proposals = sorted(
        enumerate(proposals), key=lambda item: (-item[1].score, item[0])
    )
    used_events: set[int] = set()
    true_positives: list[float] = []
    for proposal_index, _proposal in ranked_proposals:
        eligible = []
        for event_index, _event in enumerate(events):
            if event_index in used_events:
                continue
            iou = float(ious[proposal_index, event_index])
            if iou >= iou_threshold:
                eligible.append((-iou, event_index))
        if eligible:
            _, event_index = min(eligible)
            used_events.add(event_index)
            true_positives.append(1.0)
        else:
            true_positives.append(0.0)

    if not true_positives:
        return 0.0
    tp = np.cumsum(np.asarray(true_positives, dtype=np.float64))
    fp = np.cumsum(1.0 - np.asarray(true_positives, dtype=np.float64))
    recall = tp / len(events)
    precision = tp / (tp + fp)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.flatnonzero(recall[1:] != recall[:-1])
    return float(
        np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1])
    )


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _validate_iou_threshold(iou_threshold: float) -> None:
    if not np.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be finite and in (0, 1]")


__all__ = [
    "IntervalMatch",
    "Task0IntervalMetrics",
    "evaluate_intervals",
    "interval_iou",
    "match_intervals_one_to_one",
]
