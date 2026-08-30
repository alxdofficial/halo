"""Task 2: bounded-execution change quantification."""

from .contracts import (
    BoundedExecution,
    ChangeTargetSpec,
    ExecutionPair,
    ExecutionPairDataset,
    PairBatch,
    collate_execution_pairs,
    from_motion_sequence,
)
from .losses import ChangeLoss, ChangeLossConfig, change_quantification_loss
from .metrics import (
    RegressionMetrics,
    balanced_accuracy,
    binary_auroc,
    masked_regression_metrics,
)
from .model import ChangeHeadOutput, ChangeMetricHead, resample_to_phase
from .training import StepTelemetry, initialize_change_threshold, train_step

__all__ = [
    "BoundedExecution",
    "ChangeHeadOutput",
    "ChangeLoss",
    "ChangeLossConfig",
    "ChangeMetricHead",
    "ChangeTargetSpec",
    "ExecutionPair",
    "ExecutionPairDataset",
    "PairBatch",
    "RegressionMetrics",
    "StepTelemetry",
    "balanced_accuracy",
    "binary_auroc",
    "change_quantification_loss",
    "collate_execution_pairs",
    "from_motion_sequence",
    "initialize_change_threshold",
    "masked_regression_metrics",
    "resample_to_phase",
    "train_step",
]
