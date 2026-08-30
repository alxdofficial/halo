"""Task 3: dense recurrent-motion discovery over complete timelines."""

from .candidates import assign_event_targets, pool_multiscale_candidates
from .consolidation import RecurrenceCluster, temporal_nms, recurrence_clusters
from .contracts import CandidateBatch, CandidateTargets, EventBatch
from .data import TimelineBatch, collate_motion_sequences, event_batch_from_recordings
from .losses import PairLossOutput, scoped_pair_indices, scoped_pair_loss
from .model import RecurrentMotionMetric
from .training import initialize_affinity_threshold

__all__ = [
    "CandidateBatch",
    "CandidateTargets",
    "EventBatch",
    "PairLossOutput",
    "RecurrenceCluster",
    "RecurrentMotionMetric",
    "TimelineBatch",
    "assign_event_targets",
    "collate_motion_sequences",
    "event_batch_from_recordings",
    "initialize_affinity_threshold",
    "pool_multiscale_candidates",
    "recurrence_clusters",
    "scoped_pair_loss",
    "scoped_pair_indices",
    "temporal_nms",
]
