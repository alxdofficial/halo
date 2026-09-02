from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from applications.motion_monitoring.data.adapters.cops import _time_seconds
from applications.motion_monitoring.data.adapters.alameda import _relative_timestamps
from applications.motion_monitoring.data.adapters.alameda import _nearest_clinical_visit
from applications.motion_monitoring.data.adapters.monipar import _activity_events


def test_cops_clock_unwraps_midnight_without_losing_physical_spacing() -> None:
    values = pd.Series(["23:59:59.990", "00:00:00.000", "00:00:00.010"])
    seconds = _time_seconds(values)
    assert seconds.tolist() == pytest.approx([0.0, 0.01, 0.02])


def test_monipar_activity_runs_have_sample_exact_boundaries() -> None:
    times = np.arange(6, dtype=np.float64) / 50.0
    events = _activity_events(np.asarray(["a", "a", "b", "b", "b", "a"]), times)
    assert [event.label for event in events] == ["a", "b", "a"]
    assert [
        value for event in events for value in (event.start_sec, event.end_sec)
    ] == pytest.approx([0.0, 0.04, 0.04, 0.1, 0.1, 0.12])


def test_alameda_preserves_released_datetime_clock() -> None:
    index = pd.DatetimeIndex(["2023-01-02 03:04:05.000", "2023-01-02 03:04:05.010"])
    seconds, source_start = _relative_timestamps(index)
    assert seconds.tolist() == pytest.approx([0.0, 0.01])
    assert source_start == "2023-01-02T03:04:05"


def test_alameda_clinical_link_requires_nearby_documented_visit() -> None:
    visits = pd.DataFrame(
        {
            "VisitID": [1],
            "PID": [4],
            "VisitDate": pd.to_datetime(["2023-01-24"]),
            "Phase": ["Off"],
            "UPDRS_3_total": [17],
        }
    )
    assert _nearest_clinical_visit(
        visits, participant="4", source_date="2023-02-02"
    )["updrs_3_total"] == 17.0
    assert _nearest_clinical_visit(
        visits, participant="4", source_date="2023-06-14"
    ) == {
        "clinical_linkage_status": "no_visit_within_14_days",
        "nearest_clinical_visit_days": 141,
    }
