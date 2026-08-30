import pytest

from applications.motion_monitoring.task0.policies import validate_policy


def test_c_mhad_cannot_be_scored_as_exhaustive() -> None:
    with pytest.raises(ValueError, match="not exhaustively annotated"):
        validate_policy(
            "c_mhad",
            stream_id="imu",
            annotation_kinds={"event"},
            excluded_labels=set(),
            exhaustive_background=True,
            calibration=False,
            allow_exploratory=False,
        )


def test_oca_exhaustive_scoring_requires_null_exclusion() -> None:
    with pytest.raises(ValueError, match="background labels"):
        validate_policy(
            "oca",
            stream_id="imu0",
            annotation_kinds={"sample_label_run"},
            excluded_labels=set(),
            exhaustive_background=True,
            calibration=False,
            allow_exploratory=False,
        )


def test_openpack_operation_calibration_requires_background_exclusions() -> None:
    with pytest.raises(ValueError, match="background labels"):
        validate_policy(
            "openpack",
            stream_id="atr01",
            annotation_kinds={"operation"},
            excluded_labels=set(),
            exhaustive_background=True,
            calibration=True,
            allow_exploratory=False,
        )
