from dataclasses import replace

import numpy as np

from applications.motion_monitoring.data.contracts import EventInterval
from applications.motion_monitoring.task0.calibration import (
    CalibrationCase,
    calibrate_thresholds,
)
from applications.motion_monitoring.task0.contracts import ProposalConfig
from tests.applications.motion_monitoring.test_task0_detector import (
    _detector,
    _sequence,
)


def test_calibration_selects_development_f1_operating_point() -> None:
    sequence = _sequence()
    event = EventInterval(0.7, 1.6, "movement")
    selected, rows = calibrate_thresholds(
        [CalibrationCase(sequence, (event,))],
        _detector().scaler,
        replace(ProposalConfig(), minimum_duration_seconds=0.5),
        start_thresholds=(3.0, 5.0),
        continue_thresholds=(1.0, 2.0),
    )
    assert selected.start_threshold == 3.0
    assert len(rows) == 4
    assert max(row.event_f1 for row in rows) == 1.0


def test_background_only_recording_contributes_false_proposals() -> None:
    event_sequence = _sequence()
    event_sequence = replace(
        event_sequence,
        features=np.column_stack(
            [
                np.full(len(event_sequence.features), 5.0),
                np.zeros(len(event_sequence.features)),
            ]
        ),
    )
    background = replace(
        event_sequence,
        recording_id="background",
        features=np.column_stack(
            [
                np.full(len(event_sequence.features), 3.5),
                np.zeros(len(event_sequence.features)),
            ]
        ),
    )
    selected, _ = calibrate_thresholds(
        [
            CalibrationCase(event_sequence, (EventInterval(0.0, 2.4, "movement"),)),
            CalibrationCase(background, ()),
        ],
        _detector().scaler,
        replace(ProposalConfig(), minimum_duration_seconds=0.5),
        start_thresholds=(3.0, 4.0),
        continue_thresholds=(1.0,),
        iou_threshold=0.5,
    )
    assert selected.start_threshold == 4.0
