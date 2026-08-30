"""Arbitrary reference-to-stream movement detection."""

from applications.motion_monitoring.task1.episodes import (
    CachedEventPair,
    CachedEventPairDataset,
    DetectionBatch,
    DetectionEpisode,
    EmbeddingSequence,
    collate_detection_episodes,
    episode_from_recordings,
    from_motion_sequence,
)
from applications.motion_monitoring.task1.matcher import (
    TemporalMatch,
    best_full_timeline_match,
    full_timeline_matches,
)
from applications.motion_monitoring.task1.model import DifferentiableSubsequenceMatcher
from applications.motion_monitoring.task1.synthetic import SyntheticDetectionDataset
from applications.motion_monitoring.task1.training import (
    balanced_endpoint_loss,
    detection_metrics,
    event_detection_metrics,
    gradient_telemetry,
    train_step,
)

__all__ = [
    "CachedEventPair",
    "CachedEventPairDataset",
    "DetectionBatch",
    "DetectionEpisode",
    "DifferentiableSubsequenceMatcher",
    "EmbeddingSequence",
    "SyntheticDetectionDataset",
    "TemporalMatch",
    "balanced_endpoint_loss",
    "best_full_timeline_match",
    "collate_detection_episodes",
    "detection_metrics",
    "event_detection_metrics",
    "episode_from_recordings",
    "from_motion_sequence",
    "full_timeline_matches",
    "gradient_telemetry",
    "train_step",
]
