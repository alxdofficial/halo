from __future__ import annotations

import numpy as np

from data.datasets.mmfit.convert import _resample_modality


def _table(clock_ms: list[float], x: list[float]) -> np.ndarray:
    rows = len(clock_ms)
    return np.column_stack((
        np.arange(rows, dtype=np.float64),
        np.asarray(clock_ms, dtype=np.float64),
        np.asarray(x, dtype=np.float64),
        np.zeros(rows, dtype=np.float64),
        np.zeros(rows, dtype=np.float64),
    ))


def test_packet_samples_with_duplicate_timestamps_survive_resampling():
    # The nonzero row shares a timestamp with the following row. Direct np.interp on the raw packet
    # clock discards it; packet-aware sequence resampling must retain its energy.
    table = _table([0, 10, 10, 20, 30], [0, 4, 0, 0, 0])
    values, observed, rates, gaps = _resample_modality(table, np.arange(0, 31, 10.0))
    assert observed.all()
    assert gaps == []
    assert rates == []  # sub-10-second synthetic block is intentionally omitted from rate telemetry
    assert np.max(np.abs(values[:, 0])) > 0.5


def test_acquisition_gap_remains_unobserved():
    table = _table([0, 10, 20, 1000, 1010], [1, 1, 1, 2, 2])
    grid = np.arange(0, 1020, 10.0)
    _, observed, _, gaps = _resample_modality(table, grid)
    assert gaps == [0.98]
    assert observed[:3].all()
    assert not observed[3:100].any()
    assert observed[100:].all()


def test_isolated_single_timestamp_packet_is_not_fabricated():
    table = _table([0, 0, 1000, 1010], [7, 9, 2, 2])
    grid = np.arange(0, 1020, 10.0)
    values, observed, _, _ = _resample_modality(table, grid)
    assert not observed[:100].any()
    assert np.all(values[:100] == 0)
    assert observed[100:].all()
