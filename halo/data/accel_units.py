#!/usr/bin/env python3
"""Single source of truth for accelerometer UNIT canonicalization -> g (gravity present).

Decided corpus policy (docs/baselines/EVALUATION_PROTOCOL_V2.md, "Heterogeneity policy"):
canonicalize exactly ONE heterogeneity axis — accelerometer UNITS — to **g** (a still,
gravity-present window has |acc| ~= 1.0 g). HALO's signed DC/gravity feature REQUIRES it
(a DC of 1.0 means gravity, and that is only true in g). Every path that feeds accelerometer
into a model imports these tables so the conversion cannot drift between paths:

  * HALO training corpus     -> datasets/imu_pretraining_dataset/multi_dataset_loader.py
  * HALO zero-shot eval      -> benchmark_data/scripts/preprocess_tsfm_eval.py
  * ssl-wearables baseline   -> benchmark_data/scripts/preprocess_ssl_wearables.py
  * UniMTS gravity guard     -> val_scripts/human_activity_recognition/evaluate_unimts.py

ONLY accelerometer channels are ever scaled. Gyroscope / magnetometer are never touched.
"""

import numpy as np

GRAVITY_MS2 = 9.80665

# iOS CoreMotion userAcceleration is g with GRAVITY REMOVED; the total specific force (in g) is
# userAcceleration + the separate unit-gravity vector. Needs gravity_{x,y,z} columns to rebuild.
IOS_USERACC_PLUS_GRAVITY = {"motionsense", "inclusivehar"}

# Already g WITH gravity -> leave as-is: Axivity raw (harth, capture24); hapt acc_* (~1.02 g);
# uci_har acc_* (remapped to total_acc, gravity present, in g); unimib_shar (g, dynamic set).
ACC_G_ASIS = {"harth", "capture24", "hapt", "uci_har", "unimib_shar"}

# milli-g with gravity -> divide by 1000.
ACC_MILLI_G = {"opportunity"}

# Gravity-REMOVED accelerometer (linear accel). Native unit is m/s^2, so it is scaled to g by
# /9.8 like any other m/s^2 set, BUT its |acc| never reaches 1 g and its signed DC feature is
# legitimately ~0 (no gravity / posture cue). NEVER fabricate gravity back. HALO KEEPS these
# (their dynamic/band features are valid); gravity-DEPENDENT baselines (ssl-wearables, UniMTS)
# must exclude or disclose them (see GRAVITY_ABSENT_IN_LIMU_GRID and
# preprocess_ssl_wearables.INCOMPATIBLE_ACCEL).
GRAVITY_REMOVED = {"kuhar"}

# Everything not listed above is m/s^2 WITH gravity -> divide by g:
#   hhar, mhealth, pamap2, wisdm, dsads, realworld, mobiact, shoaib (+ kuhar, gravity-removed).

# Non-physical accel that CANNOT be scaled to g at all (excluded from the corpus, not converted).
NON_PHYSICAL_ACCEL = {
    "recgym": "min-max [0,1] per-axis normalized; no recoverable physical scale or gravity",
}

# Datasets whose accel, AS STORED IN THE 20 Hz limubert grid (data_20_120.npy), has NO gravity.
# As of the 2026-07-11 parity fix, preprocess_limubert reconstructs total accel = userAcc + gravity
# for the iOS sets (matching the ssl 30 Hz and HALO tsfm_eval grids), so the LIMU grid is now
# gravity-PRESENT for motionsense/inclusivehar and ONLY the gravity-removed set (kuhar) is
# gravity-absent here. A gravity-DEPENDENT baseline (UniMTS) must not be scored on a gravity-absent
# set as if gravity were present — it must skip + disclose (#91b).
GRAVITY_ABSENT_IN_LIMU_GRID = frozenset(GRAVITY_REMOVED)


def correct_core_geometry(dataset: str, df, sensor_data: np.ndarray) -> np.ndarray:
    """Fix physically-incoherent / mis-merged core channels IN PLACE on a (T, 6) [acc_xyz, gyro_xyz]
    array (as produced by the preprocessors' extract_core_channels), using the raw dataframe `df`.

    Shared by every benchmark preprocessor (limubert / tsfm_eval) so their grids agree on the same
    physical sensor. Gravity/unit canonicalization stays the caller's job (it differs per grid).

      * wisdm  — phone accel and phone gyro are logged on DISJOINT interleaved rows (a phone-accel
                 row has NaN gyro, so extract_core_channels zero-fills it → 100% zero gyro). Rebuild
                 real gyro by interpolating the phone-gyro stream onto every row's timestamp (P0-2).
      * mhealth — the core paired CHEST acc with ANKLE gyro (different body sites — not one IMU).
                 Co-locate onto the ankle IMU: ankle_acc_* with the ankle gyro (the bare gyro_*) (P0-3).
    """
    if dataset == "wisdm":
        t = df["timestamp_sec"].values.astype(np.float64)
        for j, col in enumerate(("gyro_x", "gyro_y", "gyro_z")):
            g = df[col].values.astype(np.float64)
            m = ~np.isnan(g)
            if m.sum() >= 2:
                sensor_data[:, 3 + j] = np.interp(t, t[m], g[m])   # real phone gyro, not zeros
    elif dataset == "mhealth":
        sensor_data[:, 0:3] = df[["ankle_acc_x", "ankle_acc_y", "ankle_acc_z"]].values.astype(np.float64)
    return sensor_data


def is_accel_channel(name: str) -> bool:
    """True iff a channel name is an accelerometer axis (never gyro/mag). Matches every accel
    spelling in the corpus: acc_x, body_acc_x, total_acc_x, watch_accel_x, chest_acc16_x, ..."""
    return "acc" in name.lower()


def accel_scale_factor(dataset: str) -> float:
    """Scalar multiplier to bring a dataset's accelerometer to g, for paths that scale columns
    in place (the HALO train loader). Raises for iOS userAcc datasets, which need the gravity
    vector added (not a scalar) — none of the 10 TRAIN datasets are iOS, so this never fires
    in the loader; it fails loud if that assumption is ever violated."""
    if dataset in IOS_USERACC_PLUS_GRAVITY:
        raise ValueError(
            f"{dataset}: iOS userAcceleration needs gravity reconstruction (acc + gravity), "
            f"not a scalar. Use to_g(ds, acc3, grav3=...).")
    if dataset in NON_PHYSICAL_ACCEL:
        raise ValueError(f"{dataset}: {NON_PHYSICAL_ACCEL[dataset]} — excluded, do not scale.")
    if dataset in ACC_G_ASIS:
        return 1.0
    if dataset in ACC_MILLI_G:
        return 1.0 / 1000.0
    return 1.0 / GRAVITY_MS2   # m/s^2 (incl. gravity-removed kuhar) -> g


def to_g(dataset: str, acc3: np.ndarray, grav3=None, clip=None) -> np.ndarray:
    """Canonicalize an accelerometer array (..., 3) to g (gravity present).

    For iOS userAcc datasets, pass grav3 (..., 3) so gravity is added back. `clip` (e.g. 3.0)
    optionally clamps to +/-clip g (ssl-wearables' harnet input contract); HALO paths pass None
    to preserve the raw signal the DC/gravity feature needs.
    """
    if dataset in NON_PHYSICAL_ACCEL:
        raise ValueError(f"{dataset}: {NON_PHYSICAL_ACCEL[dataset]} — excluded, do not convert.")
    if dataset in IOS_USERACC_PLUS_GRAVITY:
        assert grav3 is not None, f"{dataset}: needs gravity_x/y/z columns to reconstruct gravity"
        out = acc3 + grav3
    elif dataset in ACC_G_ASIS:
        out = acc3
    elif dataset in ACC_MILLI_G:
        out = acc3 / 1000.0
    else:
        out = acc3 / GRAVITY_MS2
    if clip is not None:
        out = np.clip(out, -clip, clip)
    return out
