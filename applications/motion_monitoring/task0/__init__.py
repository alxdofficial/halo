"""Task-0 motion-event proposal and segmentation."""

from applications.motion_monitoring.task0.calibration import (
    CalibrationCase,
    calibrate_thresholds,
)
from applications.motion_monitoring.task0.contracts import (
    EvidenceConfig,
    EvidenceSequence,
    MotionProposal,
    ProposalConfig,
    RefinementConfig,
    RobustFeatureScaler,
)
from applications.motion_monitoring.task0.detector import Task0Detector
from applications.motion_monitoring.task0.evidence import (
    extract_physical_evidence,
    fit_robust_scaler,
    standardized_motion_score,
)
from applications.motion_monitoring.task0.metrics import evaluate_intervals
from applications.motion_monitoring.task0.plotting import plot_task0_timeline

__all__ = [
    "EvidenceConfig",
    "CalibrationCase",
    "EvidenceSequence",
    "MotionProposal",
    "ProposalConfig",
    "RefinementConfig",
    "RobustFeatureScaler",
    "Task0Detector",
    "extract_physical_evidence",
    "fit_robust_scaler",
    "standardized_motion_score",
    "calibrate_thresholds",
    "evaluate_intervals",
    "plot_task0_timeline",
]
