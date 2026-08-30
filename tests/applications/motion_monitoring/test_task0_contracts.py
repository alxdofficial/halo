import numpy as np
import pytest

from applications.motion_monitoring.task0.contracts import (
    EvidenceConfig,
    EvidenceSequence,
    ProposalConfig,
    RefinementConfig,
)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: EvidenceConfig(window_seconds=np.nan),
        lambda: ProposalConfig(start_threshold=np.nan),
        lambda: RefinementConfig(penalty=np.nan),
    ],
)
def test_configs_reject_non_finite_values(constructor) -> None:
    with pytest.raises(ValueError, match="finite"):
        constructor()


def test_evidence_rejects_crossing_window_geometry_and_nonfinite_quality() -> None:
    kwargs = dict(
        dataset="synthetic",
        recording_id="recording",
        subject_id="subject",
        session_id="session",
        stream_id="watch",
        placement="wrist",
        window_start_sec=np.asarray([0.0, 1.0]),
        window_end_sec=np.asarray([10.0, 2.0]),
        features=np.zeros((2, 2)),
        feature_valid=np.ones((2, 2), dtype=bool),
        valid_fraction=np.ones(2),
        constant_fraction=np.zeros(2),
    )
    with pytest.raises(ValueError, match="ordered"):
        EvidenceSequence(**kwargs)
    kwargs["window_end_sec"] = np.asarray([0.5, 1.5])
    kwargs["valid_fraction"] = np.asarray([1.0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        EvidenceSequence(**kwargs)
