"""Integration tests for the grid-build orchestrator's assembly core (`stream_grid`).

Uses synthetic in-memory sessions, so it exercises the accumulation + canonicalization + both
alignments without needing converted parquet on disk.
"""

import numpy as np
import pandas as pd

from data.scripts.build_grids import stream_grid
from data.scripts.curate.accel_units import GRAVITY_MS2
from data.scripts.curate.deployment_policy import all_source_channels, get_stream_spec


def _session_frame(dataset, spec, n, activity, acc=GRAVITY_MS2, gyro=0.3):
    cols = {"timestamp_sec": np.arange(n, dtype=float) / 50.0}
    for s in all_source_channels(dataset, role=spec.role):
        cols[s] = np.full(n, acc if "acc" in s else (gyro if "gyro" in s else 1.0), dtype=float)
    cols["activity"] = [activity] * n
    return pd.DataFrame(cols)


def test_stream_grid_harmonised_stacks_resamples_and_canonicalizes():
    spec = get_stream_spec("hhar", "phone_waist")
    sessions = [  # (frame, native_rate_hz, subject)
        (_session_frame("hhar", spec, 300, "laying"), 50.0, "s1"),   # laying -> lying (canonical)
        (_session_frame("hhar", spec, 300, "walking"), 50.0, "s2"),
    ]
    grid, subjects = stream_grid("hhar", spec, sessions, alignment="harmonised", harmonised=True)
    assert grid.rate_hz == 60.0                                       # resampled
    assert grid.channels == ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
    assert grid.data.shape == (2, 360, 6)                            # 300@50Hz -> 360@60Hz, one 6 s window each
    assert np.allclose(grid.data[:, 100:200, :3], 1.0, atol=0.05)    # m/s^2 -> g, gravity preserved
    assert set(grid.labels) == {"lying", "walking"}                 # 'laying' canonicalized to 'lying'
    assert len(subjects) == len(grid.data) and set(subjects) == {"s1", "s2"}


def test_stream_grid_non_harmonised_keeps_native_rate_and_labels():
    spec = get_stream_spec("hhar", "phone_waist")
    sessions = [(_session_frame("hhar", spec, 120, "laying"), 50.0, "s1")]
    grid, subjects = stream_grid("hhar", spec, sessions, alignment="non_harmonised",
                                 harmonised=False, window_seconds=1.0)
    assert grid.rate_hz == 50.0                                      # native rate kept
    assert grid.data.shape == (2, 50, 6)                            # 120@50Hz, 1 s windows, stride=window
    assert grid.labels == ["laying", "laying"]                     # native label, NOT canonicalized


def test_stream_grid_acc_only_dataset_harmonised_pads_gyro():
    spec = get_stream_spec("capture24", "watch_wrist")
    sessions = [(_session_frame("capture24", spec, 600, "walking", acc=1.0), 100.0, "p1")]
    grid, _ = stream_grid("capture24", spec, sessions, alignment="harmonised", harmonised=True)
    assert grid.channels == ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
    assert list(grid.mask) == [True, True, True, False, False, False]
    assert np.count_nonzero(grid.data[..., 3:]) == 0                # gyro zero-padded, never fabricated
