from argparse import Namespace
from pathlib import Path

import pytest

from applications.motion_monitoring.task0.cli import _calibrate, _fit


def test_fit_refuses_evaluation_sources_before_loading_data() -> None:
    with pytest.raises(ValueError, match="refusing to fit"):
        _fit(
            Namespace(
                datasets=["c_mhad"],
                allow_evaluation_fit=False,
            )
        )


def test_calibration_refuses_evaluation_sources_before_loading_data() -> None:
    with pytest.raises(ValueError, match="refusing to calibrate"):
        _calibrate(
            Namespace(
                dataset="wear",
                allow_evaluation_calibration=False,
                confirm_exhaustive_background=True,
                model=Path("unused"),
            )
        )


def test_calibration_requires_explicit_exhaustive_background_confirmation() -> None:
    with pytest.raises(ValueError, match="confirm-exhaustive-background"):
        _calibrate(
            Namespace(
                dataset="recofit",
                allow_evaluation_calibration=False,
                confirm_exhaustive_background=False,
                model=Path("unused"),
            )
        )
