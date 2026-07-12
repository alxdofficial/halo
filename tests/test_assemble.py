"""End-to-end tests for the harmonised / non-harmonised assembly pipeline."""

import numpy as np
import pandas as pd
import pytest

from data.scripts.assembly.assemble import assemble
from data.scripts.curate.accel_units import GRAVITY_MS2
from data.scripts.curate.deployment_policy import all_source_channels, get_stream_spec

ACC = ("acc_x", "acc_y", "acc_z")
GYRO = ("gyro_x", "gyro_y", "gyro_z")


def make_frame(dataset, stream_id, n=40, fills=None, activity=None):
    """A synthetic raw session: every source channel filled per `fills` (substring → value)."""
    spec = get_stream_spec(dataset, stream_id)
    cols = {"timestamp_sec": np.arange(n, dtype=float) / 50.0}
    for src in all_source_channels(dataset, role=spec.role):
        v = 1.0
        for key, val in (fills or {}).items():
            if key in src:
                v = val
        cols[src] = np.full(n, v, dtype=float)
    if activity is not None:
        cols["activity"] = activity
    return pd.DataFrame(cols), spec


def test_hhar_ms2_accel_becomes_g_and_gyro_untouched():
    frame, spec = make_frame("hhar", "phone_waist", fills={"acc": GRAVITY_MS2, "gyro": 0.3})
    g = assemble(frame, "hhar", spec, alignment="harmonised", window=20, rate_hz=50)
    assert g.channels == ACC + GYRO and g.mask.all()          # real acc+gyro
    assert np.allclose(g.data[..., :3], 1.0, atol=1e-4)       # 9.80665 m/s^2 -> 1 g
    assert np.allclose(g.data[..., 3:], 0.3)                  # gyro NEVER scaled


def test_uci_har_uses_total_acc_in_g():
    frame, spec = make_frame("uci_har", "phone_waist", fills={"acc": 1.0, "gyro": 0.2})
    g = assemble(frame, "uci_har", spec, alignment="harmonised", window=20, rate_hz=50)
    assert np.allclose(g.data[..., :3], 1.0)                  # already g -> unchanged
    assert np.allclose(g.data[..., 3:], 0.2)


def test_motionsense_reconstructs_total_then_g():
    # curate sums userAcceleration + gravity (both g); accel_units leaves g as-is.
    frame, spec = make_frame("motionsense", "phone_front_pocket",
                             fills={"gravity": 0.6, "gyro": 0.1, "acc": 0.4})
    g = assemble(frame, "motionsense", spec, alignment="harmonised", window=20, rate_hz=50)
    assert np.allclose(g.data[..., :3], 1.0)                  # 0.4 + 0.6 = 1.0 g
    assert np.allclose(g.data[..., 3:], 0.1)


def test_capture24_acc_only_harmonised_pads_gyro_nonharmonised_keeps_three():
    frame, spec = make_frame("capture24", "watch_wrist", fills={"acc": 1.0})
    h = assemble(frame, "capture24", spec, alignment="harmonised", window=20, rate_hz=100)
    assert h.channels == ACC + GYRO
    assert list(h.mask) == [True, True, True, False, False, False]
    assert np.count_nonzero(h.data[..., 3:]) == 0            # gyro zero-padded, never fabricated
    n = assemble(frame, "capture24", spec, alignment="non_harmonised", window=20, rate_hz=100)
    assert n.channels == ACC and n.data.shape[-1] == 3 and n.mask.all()


def test_mhealth_now_has_real_six_channels():
    frame, spec = make_frame("mhealth", "watch_wrist", fills={"acc": GRAVITY_MS2, "gyro": 0.5})
    g = assemble(frame, "mhealth", spec, alignment="harmonised", window=20, rate_hz=50)
    assert g.channels == ACC + GYRO and g.mask.all()          # gyro kept (real)
    assert np.allclose(g.data[..., :3], 1.0, atol=1e-4)       # m/s^2 -> g
    assert np.allclose(g.data[..., 3:], 0.5)


def test_windowing_shapes_and_majority_labels():
    activity = np.array(["walk"] * 20 + ["run"] * 20)
    frame, spec = make_frame("hhar", "phone_waist", n=40, fills={"acc": GRAVITY_MS2}, activity=activity)
    g = assemble(frame, "hhar", spec, alignment="harmonised", window=20, rate_hz=50)  # stride=window
    assert g.data.shape == (2, 20, 6)
    assert g.labels == ["walk", "run"]
