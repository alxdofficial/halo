"""Validated contracts for Task-0 evidence and event proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_NAMES = ("dynamic_acc_rms_g", "angular_speed_rms_rad_s")


@dataclass(frozen=True)
class EvidenceConfig:
    window_seconds: float = 0.5
    stride_seconds: float = 0.1
    gravity_cutoff_hz: float = 0.3
    min_valid_fraction: float = 0.8
    min_samples: int = 3
    constant_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        numeric = (
            self.window_seconds,
            self.stride_seconds,
            self.gravity_cutoff_hz,
            self.min_valid_fraction,
            self.constant_tolerance,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("evidence configuration values must be finite")
        if self.window_seconds <= 0 or self.stride_seconds <= 0:
            raise ValueError("window and stride durations must be positive")
        if self.stride_seconds > self.window_seconds:
            raise ValueError("evidence stride cannot exceed its window duration")
        if self.gravity_cutoff_hz <= 0:
            raise ValueError("gravity cutoff must be positive")
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError("minimum valid fraction must be in (0, 1]")
        if self.min_samples < 2:
            raise ValueError("minimum samples must be at least two")
        if self.constant_tolerance < 0:
            raise ValueError("constant tolerance cannot be negative")


@dataclass(frozen=True)
class ProposalConfig:
    start_threshold: float = 3.0
    continue_threshold: float = 1.5
    minimum_duration_seconds: float = 0.5
    merge_gap_seconds: float = 0.25
    merge_floor: float = 0.75
    uncertainty_margin: float = 0.2
    minimum_boundary_change_score: float = 0.25

    def __post_init__(self) -> None:
        if not np.isfinite(
            (
                self.start_threshold,
                self.continue_threshold,
                self.minimum_duration_seconds,
                self.merge_gap_seconds,
                self.merge_floor,
                self.uncertainty_margin,
                self.minimum_boundary_change_score,
            )
        ).all():
            raise ValueError("proposal configuration values must be finite")
        if self.start_threshold <= self.continue_threshold:
            raise ValueError("start threshold must exceed continuation threshold")
        if self.continue_threshold < 0:
            raise ValueError("continuation threshold cannot be negative")
        if self.minimum_duration_seconds <= 0 or self.merge_gap_seconds < 0:
            raise ValueError("proposal durations are invalid")
        if not 0 <= self.merge_floor <= self.continue_threshold:
            raise ValueError(
                "merge floor must be between zero and continuation threshold"
            )
        if self.uncertainty_margin < 0:
            raise ValueError("uncertainty margin cannot be negative")
        if not 0 <= self.minimum_boundary_change_score <= 1:
            raise ValueError("minimum boundary change score must be in [0, 1]")


@dataclass(frozen=True)
class RefinementConfig:
    enabled: bool = True
    penalty: float = 5.0
    minimum_segment_seconds: float = 0.3
    jump_seconds: float = 0.1
    boundary_margin_seconds: float = 0.5
    minimum_change_magnitude: float = 0.5

    def __post_init__(self) -> None:
        if not np.isfinite(
            (
                self.penalty,
                self.minimum_segment_seconds,
                self.jump_seconds,
                self.boundary_margin_seconds,
                self.minimum_change_magnitude,
            )
        ).all():
            raise ValueError("refinement configuration values must be finite")
        if self.penalty <= 0:
            raise ValueError("PELT penalty must be positive")
        if self.minimum_segment_seconds <= 0 or self.jump_seconds <= 0:
            raise ValueError("PELT durations must be positive")
        if self.boundary_margin_seconds < 0 or self.minimum_change_magnitude < 0:
            raise ValueError("boundary refinement parameters cannot be negative")


@dataclass(frozen=True)
class RobustFeatureScaler:
    center: np.ndarray
    scale: np.ndarray
    observed: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        observed = np.asarray(self.observed, dtype=bool)
        expected = (len(FEATURE_NAMES),)
        if (
            center.shape != expected
            or scale.shape != expected
            or observed.shape != expected
        ):
            raise ValueError(f"feature scaler arrays must have shape {expected}")
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError("feature scaler values must be finite")
        if np.any(scale <= 0):
            raise ValueError("feature scales must be positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "observed", observed)

    def transform(self, features: np.ndarray, feature_valid: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        feature_valid = np.asarray(feature_valid, dtype=bool)
        if features.shape != feature_valid.shape or features.shape[-1] != len(
            FEATURE_NAMES
        ):
            raise ValueError("features and masks disagree with the scaler")
        valid = feature_valid & self.observed
        standardized = np.zeros_like(features, dtype=np.float64)
        np.subtract(features, self.center, out=standardized, where=valid)
        np.divide(standardized, self.scale, out=standardized, where=valid)
        standardized[~valid] = 0.0
        return standardized

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "observed": self.observed.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RobustFeatureScaler:
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("feature scaler names do not match this Task-0 version")
        return cls(
            center=np.asarray(payload["center"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            observed=np.asarray(payload["observed"], dtype=bool),
        )


@dataclass(frozen=True)
class EvidenceSequence:
    dataset: str
    recording_id: str
    subject_id: str
    session_id: str
    stream_id: str
    placement: str
    window_start_sec: np.ndarray
    window_end_sec: np.ndarray
    features: np.ndarray
    feature_valid: np.ndarray
    valid_fraction: np.ndarray
    constant_fraction: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        starts = np.asarray(self.window_start_sec, dtype=np.float64)
        ends = np.asarray(self.window_end_sec, dtype=np.float64)
        features = np.asarray(self.features, dtype=np.float64)
        feature_valid = np.asarray(self.feature_valid, dtype=bool)
        valid_fraction = np.asarray(self.valid_fraction, dtype=np.float64)
        constant_fraction = np.asarray(self.constant_fraction, dtype=np.float64)
        count = len(starts)
        if not all(
            (
                self.dataset,
                self.recording_id,
                self.subject_id,
                self.session_id,
                self.stream_id,
            )
        ):
            raise ValueError("evidence provenance must be non-empty")
        if starts.shape != (count,) or ends.shape != (count,):
            raise ValueError("evidence window times must be vectors")
        if features.shape != (count, len(FEATURE_NAMES)):
            raise ValueError("evidence feature shape is invalid")
        if feature_valid.shape != features.shape:
            raise ValueError("evidence feature mask shape is invalid")
        if valid_fraction.shape != (count,) or constant_fraction.shape != (count,):
            raise ValueError("evidence quality vectors have invalid shapes")
        if count and (
            not np.isfinite(starts).all()
            or not np.isfinite(ends).all()
            or np.any(ends <= starts)
            or np.any(np.diff(starts) <= 0)
            or np.any(np.diff(ends) <= 0)
        ):
            raise ValueError("evidence windows must be finite and ordered")
        if not np.isfinite(features).all() or np.any(features < 0):
            raise ValueError("evidence features must be finite and non-negative")
        if (
            not np.isfinite(valid_fraction).all()
            or not np.isfinite(constant_fraction).all()
        ):
            raise ValueError("evidence quality fractions must be finite")
        if np.any((valid_fraction < 0) | (valid_fraction > 1)) or np.any(
            (constant_fraction < 0) | (constant_fraction > 1)
        ):
            raise ValueError("evidence quality fractions must be in [0, 1]")
        object.__setattr__(self, "window_start_sec", starts)
        object.__setattr__(self, "window_end_sec", ends)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_valid", feature_valid)
        object.__setattr__(self, "valid_fraction", valid_fraction)
        object.__setattr__(self, "constant_fraction", constant_fraction)

    @property
    def centers_sec(self) -> np.ndarray:
        return 0.5 * (self.window_start_sec + self.window_end_sec)


@dataclass(frozen=True)
class MotionProposal:
    dataset: str
    recording_id: str
    subject_id: str
    session_id: str
    stream_ids: Sequence[str]
    placements: Sequence[str]
    start_sec: float
    end_sec: float
    score: float
    start_boundary_change_score: float
    end_boundary_change_score: float
    valid_fraction: float
    uncertain: bool
    refinement: str
    feature_summary: Mapping[str, float]
    quality_flags: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all((self.dataset, self.recording_id, self.subject_id, self.session_id)):
            raise ValueError("proposal provenance must be non-empty")
        if self.end_sec <= self.start_sec or self.start_sec < 0:
            raise ValueError("proposal interval is invalid")
        if not np.isfinite(
            [
                self.start_sec,
                self.end_sec,
                self.score,
                self.start_boundary_change_score,
                self.end_boundary_change_score,
                self.valid_fraction,
            ]
        ).all():
            raise ValueError("proposal values must be finite")
        if not 0 <= self.valid_fraction <= 1:
            raise ValueError("proposal valid fraction must be in [0, 1]")
        if (
            not 0 <= self.start_boundary_change_score <= 1
            or not 0 <= self.end_boundary_change_score <= 1
        ):
            raise ValueError("proposal boundary change score must be in [0, 1]")
        if not self.stream_ids or len(self.stream_ids) != len(self.placements):
            raise ValueError("proposal stream provenance is invalid")
        if self.score < 0:
            raise ValueError("proposal score cannot be negative")
        if not np.isfinite(tuple(self.feature_summary.values())).all():
            raise ValueError("proposal feature summary must be finite")
        object.__setattr__(self, "stream_ids", tuple(self.stream_ids))
        object.__setattr__(self, "placements", tuple(self.placements))
        object.__setattr__(self, "quality_flags", tuple(self.quality_flags))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
