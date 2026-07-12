"""Accelerometer UNIT canonicalization → g.

This module normalizes exactly ONE heterogeneity axis: the accelerometer UNIT, so that a still,
gravity-present window reads |acc| ≈ 1.0 g. (HALO's signed DC/gravity feature needs g — a per-channel
DC of 1.0 means "gravity", which is only true in g.) The gyroscope is NEVER touched.

Pipeline ordering (important — this is why the logic here is so small):

    raw session → deployment_policy.curate_*  →  accel_units.to_g  →  windowing  →  baseline_view

`deployment_policy` runs FIRST and owns GRAVITY: it selects the phone/watch device stream and, for the
iOS sets, reconstructs total acceleration = userAcceleration + gravity (both in g). So by the time a
stream reaches this module, gravity is already handled and each accelerometer is in its dataset's
NATIVE unit — all we do here is a scalar rescale (g stays; m/s² → ÷9.80665).

The per-dataset unit is a fixed property of how each dataset was recorded. The rationale for every
dataset is documented in `docs/DATA_HETEROGENEITY.md`; `tests/test_accel_units.py` asserts that every
dataset in the deployment policy is classified here exactly once, so a new dataset cannot be added
without a documented unit decision.
"""

from __future__ import annotations

import numpy as np

GRAVITY_MS2 = 9.80665

# --- Accelerometer already in g → scale 1.0 --------------------------------------------------------
# Native-g recordings: capture24 + harth (Axivity AX3, reported in g); uci_har (Android, exported as
# `total_acc` in g — NOT the gravity-removed `body_acc`); hapt (Android, ~1.02 g).
# iOS sets (motionsense, inclusivehar): deployment_policy already summed userAcceleration + gravity,
# both in g, so they ARRIVE here in g.
ACC_UNIT_G = frozenset({
    "uci_har", "hapt", "capture24", "harth", "motionsense", "inclusivehar",
    "usc_had",  # MotionNode accelerometer reported in g (USC-HAD Readme); |acc|~1.07 g still.
})

# --- Accelerometer in m/s² → scale 1/9.80665 -------------------------------------------------------
# Android / Shimmer recordings. kuhar is here too: it is m/s² but gravity-REMOVED (linear
# acceleration); the unit rescale is identical — gravity STATE is deployment_policy's concern and is
# never fabricated here.
# unimib_shar: the RAW UniMiB .npy release is m/s² (|acc|~9.8, gravity present); the converter reads
# that (not the z-scored Kaggle CSV). Reclassified from G on 2026-07-12 after the debug sweep found the
# CSV was z-score-normalized.
ACC_UNIT_MS2 = frozenset({
    "hhar", "pamap2", "wisdm", "kuhar", "mhealth", "realworld", "mobiact", "shoaib",
    "tnda_har", "ut_complex", "unimib_shar",  # accelerometer in m/s^2 (gravity present) -> rescale to g.
})


def accel_scale_factor(dataset: str) -> float:
    """Scalar that brings `dataset`'s (already gravity-handled) accelerometer to g."""
    if dataset in ACC_UNIT_G:
        return 1.0
    if dataset in ACC_UNIT_MS2:
        return 1.0 / GRAVITY_MS2
    raise KeyError(
        f"{dataset!r} has no accelerometer-unit classification. Add it to ACC_UNIT_G or ACC_UNIT_MS2 "
        f"in data/scripts/accel_units.py and document why in docs/DATA_HETEROGENEITY.md.")


def to_g(dataset: str, acc: np.ndarray) -> np.ndarray:
    """Return the accelerometer array rescaled to g (pure unit rescale; gravity already handled)."""
    return np.asarray(acc, dtype=np.float32) * accel_scale_factor(dataset)


def is_accel_channel(name: str) -> bool:
    """True iff a channel name is an accelerometer axis (never gyro/mag): `acc_x`, `hand_acc16_x`, ..."""
    return "acc" in name.lower()
