"""Tests for accelerometer unit canonicalization + its per-dataset coverage invariant."""

import numpy as np
import pytest

from data.scripts.curate import accel_units as au
from data.scripts.curate import deployment_policy as dp


def test_unit_sets_are_disjoint():
    assert au.ACC_UNIT_G.isdisjoint(au.ACC_UNIT_MS2)


def test_every_deployment_dataset_is_classified_exactly_once():
    """A dataset cannot enter the deployment policy without a documented unit decision."""
    datasets = {s.dataset for s in dp.STREAM_SPECS}
    classified = au.ACC_UNIT_G | au.ACC_UNIT_MS2
    assert datasets <= classified, f"unclassified deployment datasets: {sorted(datasets - classified)}"


def test_scale_factors():
    assert au.accel_scale_factor("uci_har") == 1.0            # native g
    assert au.accel_scale_factor("capture24") == 1.0
    assert abs(au.accel_scale_factor("hhar") - 1.0 / au.GRAVITY_MS2) < 1e-12   # m/s^2 -> g
    assert abs(au.accel_scale_factor("kuhar") - 1.0 / au.GRAVITY_MS2) < 1e-12  # gravity-removed, still m/s^2
    with pytest.raises(KeyError):
        au.accel_scale_factor("not_a_dataset")


def test_to_g_rescales_ms2_and_leaves_g_alone():
    one_g_in_ms2 = np.full((4, 3), au.GRAVITY_MS2, np.float32)
    assert np.allclose(au.to_g("hhar", one_g_in_ms2), 1.0, atol=1e-4)       # m/s^2 -> g
    assert np.allclose(au.to_g("uci_har", one_g_in_ms2), au.GRAVITY_MS2)    # g-set: unchanged


def test_is_accel_channel():
    assert au.is_accel_channel("acc_x") and au.is_accel_channel("hand_acc16_x")
    assert not au.is_accel_channel("gyro_x") and not au.is_accel_channel("mag_x")
