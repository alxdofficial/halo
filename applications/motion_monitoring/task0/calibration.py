"""Development-only operating-point calibration for Task 0."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from applications.motion_monitoring.data.contracts import EventInterval
from applications.motion_monitoring.task0.contracts import (
    EvidenceSequence,
    ProposalConfig,
    RefinementConfig,
    RobustFeatureScaler,
)
from applications.motion_monitoring.task0.detector import Task0Detector
from applications.motion_monitoring.task0.metrics import match_intervals_one_to_one


@dataclass(frozen=True)
class CalibrationCase:
    evidence: EvidenceSequence
    events: Sequence[EventInterval]


@dataclass(frozen=True)
class CalibrationRow:
    start_threshold: float
    continue_threshold: float
    event_precision: float
    event_recall: float
    event_f1: float
    proposal_count: int
    event_count: int


def calibrate_thresholds(
    cases: Sequence[CalibrationCase],
    scaler: RobustFeatureScaler,
    base_config: ProposalConfig,
    *,
    start_thresholds: Sequence[float],
    continue_thresholds: Sequence[float],
    iou_threshold: float = 0.5,
) -> tuple[ProposalConfig, tuple[CalibrationRow, ...]]:
    """Choose an event-F1 operating point from exhaustively annotated development data."""

    if not cases:
        raise ValueError("threshold calibration requires at least one development case")
    if not any(case.events for case in cases):
        raise ValueError("threshold calibration requires at least one positive event")
    candidates = sorted(
        {
            (float(start), float(continuation))
            for start in start_thresholds
            for continuation in continue_thresholds
            if start > continuation >= 0
        }
    )
    if not candidates:
        raise ValueError("threshold grid contains no valid start/continuation pair")

    rows: list[CalibrationRow] = []
    for start, continuation in candidates:
        config = replace(
            base_config,
            start_threshold=start,
            continue_threshold=continuation,
            merge_floor=min(base_config.merge_floor, continuation),
        )
        detector = Task0Detector(
            scaler,
            proposal_config=config,
            refinement_config=RefinementConfig(enabled=False),
        )
        matched = 0
        proposal_count = 0
        event_count = 0
        for case in cases:
            proposals = detector.detect_evidence(case.evidence)
            matched += len(
                match_intervals_one_to_one(
                    proposals, case.events, iou_threshold=iou_threshold
                )
            )
            proposal_count += len(proposals)
            event_count += len(case.events)
        precision = matched / proposal_count if proposal_count else 0.0
        recall = matched / event_count if event_count else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append(
            CalibrationRow(
                start_threshold=start,
                continue_threshold=continuation,
                event_precision=precision,
                event_recall=recall,
                event_f1=f1,
                proposal_count=proposal_count,
                event_count=event_count,
            )
        )
    best = min(
        rows,
        key=lambda row: (
            -row.event_f1,
            -row.event_recall,
            row.proposal_count,
            row.start_threshold,
            row.continue_threshold,
        ),
    )
    selected = replace(
        base_config,
        start_threshold=best.start_threshold,
        continue_threshold=best.continue_threshold,
        merge_floor=min(base_config.merge_floor, best.continue_threshold),
    )
    return selected, tuple(rows)
