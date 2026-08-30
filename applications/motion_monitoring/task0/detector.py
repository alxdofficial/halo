"""Calibrated hysteresis and bounded PELT refinement for Task 0."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from applications.motion_monitoring.data.contracts import RawRecording, SensorStream
from applications.motion_monitoring.task0.contracts import (
    EvidenceConfig,
    EvidenceSequence,
    FEATURE_NAMES,
    MotionProposal,
    ProposalConfig,
    RefinementConfig,
    RobustFeatureScaler,
)
from applications.motion_monitoring.task0.evidence import (
    extract_physical_evidence,
    fit_robust_scaler,
    standardized_motion_score,
)


MODEL_SCHEMA_VERSION = 2


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    boundaries = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(boundaries[::2], boundaries[1::2]))


def _interval_times(
    sequence: EvidenceSequence, start: int, end: int
) -> tuple[float, float]:
    centers = sequence.centers_sec
    start_time = (
        0.5 * (centers[start - 1] + centers[start])
        if start > 0
        else sequence.window_start_sec[start]
    )
    end_time = (
        0.5 * (centers[end] + centers[end + 1])
        if end + 1 < len(centers)
        else sequence.window_end_sec[end]
    )
    return float(start_time), float(end_time)


def _rough_intervals(
    sequence: EvidenceSequence,
    scores: np.ndarray,
    valid: np.ndarray,
    config: ProposalConfig,
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    active_start: int | None = None
    for index, (score, is_valid) in enumerate(zip(scores, valid)):
        if active_start is None:
            if is_valid and score >= config.start_threshold:
                active_start = index
        elif not is_valid or score < config.continue_threshold:
            intervals.append((active_start, index - 1))
            active_start = None
            if is_valid and score >= config.start_threshold:
                active_start = index
    if active_start is not None:
        intervals.append((active_start, len(scores) - 1))

    intervals = [
        (start, end)
        for start, end in intervals
        if _interval_times(sequence, start, end)[1]
        - _interval_times(sequence, start, end)[0]
        >= config.minimum_duration_seconds
    ]
    if not intervals:
        return []

    merged = [intervals[0]]
    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]
        gap_seconds = max(
            0.0,
            _interval_times(sequence, start, end)[0]
            - _interval_times(sequence, previous_start, previous_end)[1],
        )
        gap_slice = slice(previous_end + 1, start)
        gap_valid = valid[gap_slice]
        gap_scores = scores[gap_slice]
        evidence_supports_merge = len(gap_scores) == 0 or (
            len(gap_scores) > 0
            and gap_valid.all()
            and float(np.mean(gap_scores)) >= config.merge_floor
        )
        if gap_seconds <= config.merge_gap_seconds and evidence_supports_merge:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def _change_points(
    sequence: EvidenceSequence,
    scaler: RobustFeatureScaler,
    valid: np.ndarray,
    intervals: Sequence[tuple[int, int]],
    config: RefinementConfig,
) -> list[tuple[float, float]]:
    if not config.enabled or len(sequence.features) < 3:
        return []
    try:
        import ruptures as rpt
    except ImportError as error:
        raise RuntimeError(
            "PELT refinement requires ruptures; install HALO with the task0 extra"
        ) from error

    standardized = scaler.transform(sequence.features, sequence.feature_valid)
    stride = (
        float(np.median(np.diff(sequence.window_start_sec)))
        if len(sequence.window_start_sec) > 1
        else float(sequence.window_end_sec[0] - sequence.window_start_sec[0])
    )
    minimum_size = max(2, int(np.ceil(config.minimum_segment_seconds / stride)))
    jump = max(1, int(round(config.jump_seconds / stride)))
    changes: list[tuple[float, float]] = []
    search_radius = config.boundary_margin_seconds + config.minimum_segment_seconds
    search_blocks: set[tuple[int, int]] = set()
    for start, end in intervals:
        for boundary in _interval_times(sequence, start, end):
            left = int(
                np.searchsorted(
                    sequence.centers_sec, boundary - search_radius, side="left"
                )
            )
            right = int(
                np.searchsorted(
                    sequence.centers_sec, boundary + search_radius, side="right"
                )
            )
            search_blocks.add((left, right))
    pattern_changes = np.zeros(len(valid), dtype=bool)
    if len(valid) > 1:
        pattern_changes[1:] = np.any(
            sequence.feature_valid[1:] != sequence.feature_valid[:-1], axis=1
        )
    quality_boundaries = set(np.flatnonzero(pattern_changes).tolist())
    blocks: set[tuple[int, int]] = set()
    for search_start, search_stop in search_blocks:
        for local_start, local_stop in _contiguous_runs(
            valid[search_start:search_stop]
        ):
            valid_start = search_start + local_start
            valid_stop = search_start + local_stop
            boundaries = [valid_start]
            boundaries.extend(
                index
                for index in sorted(quality_boundaries)
                if valid_start < index < valid_stop
            )
            boundaries.append(valid_stop)
            blocks.update(zip(boundaries[:-1], boundaries[1:]))

    for block_start, block_stop in sorted(blocks):
        available_columns = sequence.feature_valid[block_start]
        block = standardized[block_start:block_stop, available_columns]
        if len(block) < 2 * minimum_size:
            continue
        endpoints = (
            rpt.Pelt(model="l2", min_size=minimum_size, jump=jump)
            .fit(block)
            .predict(pen=config.penalty)
        )
        for local_endpoint in endpoints[:-1]:
            endpoint = block_start + int(local_endpoint)
            if endpoint <= block_start or endpoint >= block_stop:
                continue
            before = standardized[max(block_start, endpoint - minimum_size) : endpoint]
            after = standardized[endpoint : min(block_stop, endpoint + minimum_size)]
            magnitude = float(np.linalg.norm(after.mean(axis=0) - before.mean(axis=0)))
            if magnitude < config.minimum_change_magnitude:
                continue
            time = 0.5 * (
                sequence.centers_sec[endpoint - 1] + sequence.centers_sec[endpoint]
            )
            changes.append((float(time), magnitude))
    unique_changes: dict[float, float] = {}
    for time, magnitude in changes:
        unique_changes[time] = max(magnitude, unique_changes.get(time, 0.0))
    return sorted(unique_changes.items())


def _boundary_fallback(
    scores: np.ndarray,
    index: int,
    *,
    is_start: bool,
) -> tuple[float, float]:
    inside = float(scores[index])
    outside_index = index - 1 if is_start else index + 1
    outside = float(scores[outside_index]) if 0 <= outside_index < len(scores) else 0.0
    magnitude = max(0.0, inside - outside)
    return magnitude, magnitude / (1.0 + magnitude)


def _refine_boundary(
    rough_time: float,
    changes: Sequence[tuple[float, float]],
    margin: float,
) -> tuple[float, float, bool]:
    candidates = [item for item in changes if abs(item[0] - rough_time) <= margin]
    if not candidates:
        return rough_time, 0.0, False
    time, magnitude = min(
        candidates, key=lambda item: (abs(item[0] - rough_time), -item[1])
    )
    return time, magnitude / (1.0 + magnitude), True


class Task0Detector:
    """A fitted, serializable event-proposal detector."""

    def __init__(
        self,
        scaler: RobustFeatureScaler,
        *,
        evidence_config: EvidenceConfig | None = None,
        proposal_config: ProposalConfig | None = None,
        refinement_config: RefinementConfig | None = None,
        fit_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.scaler = scaler
        self.evidence_config = evidence_config or EvidenceConfig()
        self.proposal_config = proposal_config or ProposalConfig()
        self.refinement_config = refinement_config or RefinementConfig()
        self.fit_provenance = dict(fit_provenance or {})

    @classmethod
    def fit(
        cls,
        sequences: Iterable[EvidenceSequence],
        *,
        evidence_config: EvidenceConfig | None = None,
        proposal_config: ProposalConfig | None = None,
        refinement_config: RefinementConfig | None = None,
        fit_provenance: Mapping[str, Any] | None = None,
    ) -> Task0Detector:
        return cls(
            fit_robust_scaler(sequences),
            evidence_config=evidence_config,
            proposal_config=proposal_config,
            refinement_config=refinement_config,
            fit_provenance=fit_provenance,
        )

    def detect_evidence(self, sequence: EvidenceSequence) -> list[MotionProposal]:
        scores, valid = standardized_motion_score(sequence, self.scaler)
        intervals = _rough_intervals(sequence, scores, valid, self.proposal_config)
        changes = _change_points(
            sequence,
            self.scaler,
            valid,
            intervals,
            self.refinement_config,
        )
        proposals: list[MotionProposal] = []
        for start_index, end_index in intervals:
            rough_start, rough_end = _interval_times(sequence, start_index, end_index)
            start, start_change_score, refined_start = _refine_boundary(
                rough_start, changes, self.refinement_config.boundary_margin_seconds
            )
            end, end_change_score, refined_end = _refine_boundary(
                rough_end, changes, self.refinement_config.boundary_margin_seconds
            )
            if not refined_start:
                _, start_change_score = _boundary_fallback(
                    scores, start_index, is_start=True
                )
            if not refined_end:
                _, end_change_score = _boundary_fallback(
                    scores, end_index, is_start=False
                )
            if (
                end <= start
                or end - start < self.proposal_config.minimum_duration_seconds
            ):
                start, end = rough_start, rough_end
                refined_start = refined_end = False
                _, start_change_score = _boundary_fallback(
                    scores, start_index, is_start=True
                )
                _, end_change_score = _boundary_fallback(
                    scores, end_index, is_start=False
                )

            selected = slice(start_index, end_index + 1)
            selected_valid = sequence.feature_valid[selected]
            summary = {
                name: float(
                    np.mean(
                        sequence.features[selected, feature][selected_valid[:, feature]]
                    )
                )
                for feature, name in enumerate(FEATURE_NAMES)
                if selected_valid[:, feature].any()
            }
            peak_score = float(np.max(scores[selected]))
            mean_valid = float(np.mean(sequence.valid_fraction[selected]))
            mean_constant = float(np.mean(sequence.constant_fraction[selected]))
            threshold_sensitive = peak_score < (
                self.proposal_config.start_threshold
                + self.proposal_config.uncertainty_margin
            )
            quality_flags: list[str] = []
            if mean_valid < self.evidence_config.min_valid_fraction:
                quality_flags.append("low_valid_fraction")
            if mean_constant > 0.5:
                quality_flags.append("constant_signal")
            if threshold_sensitive:
                quality_flags.append("threshold_sensitive")
            if not (refined_start and refined_end):
                quality_flags.append("partially_unrefined_boundary")
            uncertain_quality = {
                "low_valid_fraction",
                "constant_signal",
                "threshold_sensitive",
            }
            uncertain = (
                any(flag in uncertain_quality for flag in quality_flags)
                or min(start_change_score, end_change_score)
                < self.proposal_config.minimum_boundary_change_score
            )
            proposals.append(
                MotionProposal(
                    dataset=sequence.dataset,
                    recording_id=sequence.recording_id,
                    subject_id=sequence.subject_id,
                    session_id=sequence.session_id,
                    stream_ids=(sequence.stream_id,),
                    placements=(sequence.placement,),
                    start_sec=start,
                    end_sec=end,
                    score=peak_score,
                    start_boundary_change_score=start_change_score,
                    end_boundary_change_score=end_change_score,
                    valid_fraction=mean_valid,
                    uncertain=uncertain,
                    refinement=(
                        "pelt_both"
                        if refined_start and refined_end
                        else (
                            "pelt_partial"
                            if refined_start or refined_end
                            else "hysteresis"
                        )
                    ),
                    feature_summary=summary,
                    quality_flags=quality_flags,
                    metadata={"evidence_windows": end_index - start_index + 1},
                )
            )
        return proposals

    def detect_stream(
        self, recording: RawRecording, stream: SensorStream
    ) -> list[MotionProposal]:
        sequence = extract_physical_evidence(recording, stream, self.evidence_config)
        return self.detect_evidence(sequence)

    def detect_recording(
        self, recording: RawRecording, *, stream_ids: Sequence[str] | None = None
    ) -> list[MotionProposal]:
        requested = set(stream_ids) if stream_ids is not None else None
        known = {stream.stream_id for stream in recording.streams}
        if requested is None and len(recording.streams) > 1:
            raise ValueError(
                f"{recording.recording_id}: select stream_ids explicitly for a "
                "multi-stream recording to avoid double-counting synchronized motion"
            )
        if requested is not None and len(requested) != 1:
            raise ValueError("Task 0 detection requires exactly one selected stream")
        if requested is not None and not requested <= known:
            missing = sorted(requested - known)
            raise KeyError(f"unknown stream ids in {recording.recording_id}: {missing}")
        proposals: list[MotionProposal] = []
        for stream in recording.streams:
            if requested is None or stream.stream_id in requested:
                proposals.extend(self.detect_stream(recording, stream))
        return sorted(
            proposals, key=lambda proposal: (proposal.start_sec, proposal.stream_ids)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "scaler": self.scaler.to_dict(),
            "evidence_config": asdict(self.evidence_config),
            "proposal_config": asdict(self.proposal_config),
            "refinement_config": asdict(self.refinement_config),
            "fit_provenance": self.fit_provenance,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Task0Detector:
        if payload.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError("unsupported Task-0 detector schema")
        return cls(
            RobustFeatureScaler.from_dict(payload["scaler"]),
            evidence_config=EvidenceConfig(**payload["evidence_config"]),
            proposal_config=ProposalConfig(**payload["proposal_config"]),
            refinement_config=RefinementConfig(**payload["refinement_config"]),
            fit_provenance=payload.get("fit_provenance", {}),
        )

    @classmethod
    def load(cls, path: Path) -> Task0Detector:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
