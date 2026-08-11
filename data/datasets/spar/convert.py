"""Convert SPAR (Burns et al., Physiol Meas 2018) to the HALO session format.

Source: github.com/dmbee/SPAR-dataset (GPL-3.0), `csv/S{n}_E{e}_{L|R}.csv`, 280 files =
20 subjects x 7 shoulder physiotherapy exercises x 2 sides.

  * Device is a consumer **Apple Watch 2/3** on the exercising wrist -- the deployment device
    class HALO actually claims, not a strapped research IMU.
  * Columns are `ax,ay,az,wx,wy,wz`. Verified at convert time: acceleration is already in **g
    with gravity present** (median |acc| ~ 1.0 g) and angular rate is already in **rad/s**
    (|w| p99 of a few rad/s; deg/s would be two orders larger). NOTHING is rescaled -- the
    guard below fails loudly if a future release changes units.
  * 50 Hz, uniform. The CSVs carry no timestamp column, so time is synthesised from the
    index at exactly 50 Hz; this is honest here because the watch wrote a fixed-rate buffer
    (row counts divide evenly into whole seconds and the paper states 50 Hz).
  * L and R are the LEFT and RIGHT shoulder performed **sequentially**, not simultaneously.
    They are emitted as two separate placements, and the event id written to `events.json`
    KEEPS the placement in it (`spar:s01:bent_over_row:left_wrist`) so the two sides never
    collapse onto one physical event: pairing them would assert a synchrony the protocol never
    claims. Every other simultaneous-placement source in the corpus does the opposite, leaving the
    placement OUT of the event id precisely because its placements ARE synchronous.

**Enrollment caveat (measured 2026-08-08, load-bearing -- read before using this for k-curves).**
One file is a single continuous bout of 20 repetitions. Measured over all 280 files: median
duration 42 s => ~2.3 s per repetition, 3.39 h total, 1,898 six-second windows dataset-wide.
So repetitions here are ~2 s apart and are NOT independent sessions. A same-subject enrollment
curve built by treating a repetition as an "execution" measures within-session binding, which is
the adjacent-window regime `eval_enrollment`'s `window_level_ids` gate exists to refuse. Execution
ids below are therefore emitted per FILE (one bout = one execution), which honestly yields k=1 per
(subject, exercise, side). See docs/data/DATASET_EXPANSION_2026-08.md Section 9.

Labels keep this dataset's published clinical wording (held-out eval sets are zero-shot targets and
are never merged into the training vocabulary).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "downloads" / "repo" / "csv"
NATIVE_RATE = 50.0
CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
SRC_COLS = ("ax", "ay", "az", "wx", "wy", "wz")

# E{n} -> published exercise name. Burns et al. 2018, Table 1.
EXERCISES = {
    0: "shoulder_pendulum",
    1: "shoulder_abduction",
    2: "shoulder_forward_elevation",
    3: "shoulder_internal_rotation",
    4: "shoulder_external_rotation",
    5: "lower_trapezius_row",
    6: "bent_over_row",
}
SIDES = {"L": "left_wrist", "R": "right_wrist"}

# Unit guards, applied to the WHOLE dataset rather than per file.
#
# The physically meaningful gravity reference is the acceleration during the quietest part of a
# recording, not its overall median: these are vigorous arm exercises, and per-file median |acc|
# legitimately ranges 0.99-2.47 g because dynamic acceleration adds to gravity. Measured over all
# 280 files, the lowest-gyro decile of each file has median |acc| = 1.026 g -- gravity, confirming
# the documented g units. Guarding on the per-file median instead would reject honest data.
#
# These bounds exist to catch a UNIT change (m/s^2 would land near 9.8, milli-g near 1000), which is
# the failure that silently corrupts the DC/gravity feature. They are deliberately wide.
QUIESCENT_G_RANGE = (0.7, 1.4)   # dataset median of each file's lowest-gyro-decile |acc|
GYRO_RADS_MAX = 40.0             # dataset max p99.9 |w|; deg/s would blow straight through this


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [c for c in SRC_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing source columns {missing}")
    return frame


def _quiescent_g(acc: np.ndarray, gyro: np.ndarray) -> float:
    """Median |acc| over the least-rotating decile — the file's gravity reference."""
    rotation = np.linalg.norm(gyro, axis=1)
    quiet = np.argsort(rotation)[:max(20, len(rotation) // 10)]
    return float(np.median(np.linalg.norm(acc[quiet], axis=1)))


def _assert_units(quiescent: list[float], gyro_peaks: list[float]) -> None:
    reference = float(np.median(quiescent))
    if not QUIESCENT_G_RANGE[0] <= reference <= QUIESCENT_G_RANGE[1]:
        raise ValueError(
            f"dataset quiescent |acc| = {reference:.3f} g, outside {QUIESCENT_G_RANGE}. The "
            "source units changed (m/s^2 would be ~9.8, milli-g ~1000); do not silently rescale."
        )
    peak = max(gyro_peaks)
    if peak > GYRO_RADS_MAX:
        raise ValueError(
            f"dataset |gyro| p99.9 max = {peak:.1f} exceeds {GYRO_RADS_MAX} rad/s; the source is "
            "probably deg/s now. Convert explicitly rather than assuming."
        )
    print(f"[spar] units verified: quiescent |acc| median {reference:.3f} g "
          f"(gravity present), |gyro| p99.9 max {peak:.1f} rad/s", flush=True)


def create_manifest() -> dict:
    return {
        "dataset_name": "SPAR",
        "description": (
            "Shoulder physiotherapy exercise recognition from a consumer Apple Watch 2/3. "
            "20 healthy subjects (40 shoulders) x 7 prescribed exercises x 20 repetitions per "
            "side, 6-axis IMU at 50 Hz. Burns et al., Physiological Measurement 39(7):075007, "
            "2018. Source: github.com/dmbee/SPAR-dataset (GPL-3.0)."
        ),
        "sampling_rate_hz": NATIVE_RATE,
        "channels": [
            {"name": f"acc_{a}", "description":
             f"accelerometer {a}-axis in g units (gravity present), wrist-worn Apple Watch",
             "sampling_rate_hz": NATIVE_RATE} for a in "xyz"
        ] + [
            {"name": f"gyro_{a}", "description":
             f"gyroscope {a}-axis in rad/s, wrist-worn Apple Watch",
             "sampling_rate_hz": NATIVE_RATE} for a in "xyz"
        ],
        "subjects": 20,
        "activities": sorted(EXERCISES.values()),
        "placements": sorted(SIDES.values()),
        "device_profile": "watch",
        "gravity_state": "present",
        "license": "GPL-3.0",
        "citation": "Burns DM et al. Physiol Meas 39(7):075007, 2018. doi:10.1088/1361-6579/aacfd9",
        "enrollment_note": (
            "One file is one continuous bout of 20 repetitions (median 42 s, ~2.3 s per "
            "repetition). Execution ids are per file, so genuine same-subject enrollment is k=1 "
            "per (subject, exercise, side). Repetition-level enrollment would be within-session."
        ),
    }


def main() -> None:
    if not RAW.is_dir():
        raise SystemExit(f"raw CSVs not found at {RAW}; clone github.com/dmbee/SPAR-dataset first")
    sessions_dir = HERE / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, list[str]] = {}
    events: dict[str, str] = {}
    quiescent: list[float] = []
    gyro_peaks: list[float] = []
    n_rows = 0
    files = sorted(RAW.glob("S*_E*_*.csv"))
    if len(files) != 280:
        print(f"[spar] WARNING: expected 280 csv files, found {len(files)}", flush=True)

    for path in files:
        stem = path.stem                       # e.g. S1_E0_R
        subject_token, exercise_token, side_token = stem.split("_")
        subject = f"s{int(subject_token[1:]):02d}"
        exercise = EXERCISES[int(exercise_token[1:])]
        if side_token not in SIDES:
            raise ValueError(f"{path.name}: unknown side token {side_token!r}")
        placement = SIDES[side_token]

        frame = _load(path)
        acc = frame[list(SRC_COLS[:3])].to_numpy(dtype=np.float64)
        gyro = frame[list(SRC_COLS[3:])].to_numpy(dtype=np.float64)
        if not np.isfinite(acc).all() or not np.isfinite(gyro).all():
            raise ValueError(f"{path.name}: non-finite samples in the source CSV")
        quiescent.append(_quiescent_g(acc, gyro))
        gyro_peaks.append(float(np.percentile(np.abs(gyro), 99.9)))

        session_id = f"{subject}_{exercise}_{placement}"
        out = pd.DataFrame({
            "timestamp_sec": np.arange(len(frame), dtype=np.float64) / NATIVE_RATE,
            **{f"acc_{a}": acc[:, i].astype(np.float32) for i, a in enumerate("xyz")},
            **{f"gyro_{a}": gyro[:, i].astype(np.float32) for i, a in enumerate("xyz")},
        })
        out["subject"] = subject
        target = sessions_dir / session_id
        target.mkdir(parents=True, exist_ok=True)
        out.to_parquet(target / "data.parquet", index=False)

        labels[session_id] = [exercise]
        # One bout = one execution. Deliberately NOT per repetition: see the module docstring.
        events[session_id] = f"spar:{subject}:{exercise}:{placement}"
        n_rows += len(out)

    _assert_units(quiescent, gyro_peaks)
    (HERE / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True))
    (HERE / "manifest.json").write_text(json.dumps(create_manifest(), indent=2))
    (HERE / "metadata.json").write_text(json.dumps({
        "dataset": "spar", "sampling_rate_hz": NATIVE_RATE, "pre_windowed": False,
    }, indent=2))
    (HERE / "events.json").write_text(json.dumps(events, indent=2, sort_keys=True))
    # Zero-shot candidate vocabulary: this dataset's own published wording.
    (HERE / "eval_labels.json").write_text(json.dumps(
        {"labels": sorted(EXERCISES.values())}, indent=2))

    hours = n_rows / NATIVE_RATE / 3600.0
    print(f"[spar] {len(labels)} sessions, {n_rows:,} samples, {hours:.2f} h, "
          f"{len(set(labels[k][0] for k in labels))} activities -> {sessions_dir}", flush=True)


if __name__ == "__main__":
    main()
