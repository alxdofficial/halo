"""Task 3: dense recurrent-motion discovery over complete timelines."""

from .candidates import assign_event_targets, pool_multiscale_candidates
from .controls import DirectAffinityOutput, direct_cosine_affinity
from .consolidation import (
    RecurrenceCluster,
    recurrence_clusters,
    recurrence_clusters_blockwise,
    temporal_nms,
)
from .contracts import CandidateBatch, CandidateTargets, EventBatch
from .data import TimelineBatch, collate_motion_sequences, event_batch_from_recordings
from .losses import PairLossOutput, scoped_pair_indices, scoped_pair_loss
from .model import RecurrentMotionMetric
from .training import initialize_affinity_threshold

__all__ = [
    "CandidateBatch",
    "DirectAffinityOutput",
    "CandidateTargets",
    "EventBatch",
    "PairLossOutput",
    "RecurrenceCluster",
    "RecurrentMotionMetric",
    "TimelineBatch",
    "assign_event_targets",
    "direct_cosine_affinity",
    "collate_motion_sequences",
    "event_batch_from_recordings",
    "initialize_affinity_threshold",
    "pool_multiscale_candidates",
    "recurrence_clusters",
    "recurrence_clusters_blockwise",
    "scoped_pair_loss",
    "scoped_pair_indices",
    "temporal_nms",
]
