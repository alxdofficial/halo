"""Convert REALDISP (UCI #305) to the HALO session format.

Source: `subject{n}_{ideal|self|mutual{4..7}}.log`, 46 logs, 17 subjects. Nine Xsens MTx units,
50 Hz, one per limb plus the back.

Why this dataset is here: REALDISP is the input-side thesis as *ground truth* rather than as a
synthetic augmentation. The same subject performs the same 33 exercises under three placement
regimes recorded separately:

  * `ideal`   — the instructor positions every unit at a predefined landmark;
  * `self`    — the subject positions three units themselves;
  * `mutual4..7` — the instructor deliberately rotates/translates 4, 5, 6 or 7 units.

So a same-subject, same-exercise pair drawn from `ideal` and `self` is a genuine
change-of-configuration, not a rotation we applied. It is also one of the few sources where an
exercise recurs for the same subject in separate recordings, which is what session-level execution
ids need (the historical audit is indexed by docs/LEGACY.md).

Layout, verified against `downloads/x/dataset manual.pdf` (Banos & Toth, 2014) tables 3-5:

    col 0        timestamp, whole seconds
    col 1        timestamp, microseconds
    col 2..118   9 sensors x 13 modalities
                 sensor order  : RLA RUA BACK LUA LLA RC RT LT LC
                 modality order: ACC xyz, GYR xyz, MAG xyz, QUAT 1-4
    col 119      activity label, 1..33; 0 = no activity

All nine sensors are emitted into ONE parquet per activity block, with a per-placement column
prefix, so the nine deployment streams are exactly simultaneous: `build_grids` gives windows of the
same session id the same event id, and their window ordinals line up. Magnetometer and quaternion
columns are dropped at the canonical-channel boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "downloads" / "x"
NATIVE_RATE = 50.0

# Manual, Table 4. Index i occupies columns 2 + 13*i .. 2 + 13*i + 12.
SENSOR_ORDER = ("rla", "rua", "back", "lua", "lla", "rc", "rt", "lt", "lc")

# Manual, Table 2. Snake-cased; the parenthesised repetition counts are protocol, not label text.
ACTIVITIES = {
    1: "walking",
    2: "jogging",
    3: "running",
    4: "jump_up",
    5: "jump_front_and_back",
    6: "jump_sideways",
    7: "jump_with_legs_and_arms_open_and_closed",
    8: "jump_rope",
    9: "trunk_twist_with_arms_outstretched",
    10: "trunk_twist_with_elbows_bent",
    11: "waist_bends_forward",
    12: "waist_rotation",
    13: "waist_bend_reaching_the_foot_with_the_opposite_hand",
    14: "reach_heels_backwards",
    15: "lateral_bend",
    16: "lateral_bend_with_the_arm_up",
    17: "repetitive_forward_stretching",
    18: "upper_trunk_and_lower_body_opposite_twist",
    19: "arms_lateral_elevation",
    20: "arms_frontal_elevation",
    21: "frontal_hand_claps",
    22: "arms_frontal_crossing",
    23: "shoulders_high_amplitude_rotation",
    24: "shoulders_low_amplitude_rotation",
    25: "arms_inner_rotation",
    26: "knees_alternately_to_the_breast",
    27: "heels_alternately_to_the_backside",
    28: "knees_bending_crouching",
    29: "knees_alternately_bending_forward",
    30: "rotation_on_the_knees",
    31: "rowing",
    32: "elliptic_bike",
    33: "cycling",
}

# Unit guards, dataset-level. The manual does not state units; both are therefore MEASURED here and
# asserted, because a silent change would corrupt the DC/gravity feature (accel) or the frequency
# content the filterbank reads (gyro).
ACC_MS2_RANGE = (8.0, 11.5)      # median |acc| over every sensor and log
GYRO_RADS_MAX = 40.0             # p99.9 |w|; deg/s would be ~57x larger and blow through this
MIN_BLOCK_SEC = 6.0              # a block shorter than one window cannot produce a grid row

# Three of the 46 released logs are PLACEHOLDERS: every one of the 117 sensor columns is exactly 0.0
# on every row, and the timestamp never advances. They still carry a full activity-label track, so
# nothing upstream notices — a converter that trusts the labels emits ~400 windows of pure zeros into
# every one of the nine streams (6.6% of the corpus), which is what the grid sweep caught. Their
# names are recorded rather than hard-coded as the skip rule: the guard below is on the DATA, so a
# future release that fills them in would convert them without any change here.
#
# Cost of skipping them: subject06 and subject13 lose their `self` condition and subject15 loses
# `mutual4`, so those subjects have no ideal-vs-self displacement pair.
ALL_ZERO_TOLERANCE = 0.0


def recording_id(session_id: str) -> str:
    """The continuous capture a session was cut out of: one `.log`, i.e. `subject01_ideal`.

    Session ids are `{subject}_{config}_{activity}_{block}`. The ideal / self / mutual4..7 logs ARE
    separate recordings, which is what makes REALDISP the corpus's second genuine multi-execution
    source after monipar — but repeated blocks of one activity inside a single log are not.
    Note what a k>1 curve here therefore measures: the configurations differ by sensor placement, so
    REALDISP's enrollment curve is intrinsically a change-of-configuration curve.
    """
    return "_".join(session_id.split("_")[:2])


def _column_indices() -> tuple[list[int], dict[int, int], dict[str, list[int]]]:
    """Columns to read, and where each of them lands in the read frame.

    `pandas.read_csv(usecols=...)` returns the selected columns in FILE order, not in the order they
    were requested, so positions must be derived from the sorted selection. Indexing by request
    order instead silently reads RLA accel-x as the activity label.
    """
    source: dict[str, list[int]] = {}
    wanted = {0, 1, 119}
    for index, name in enumerate(SENSOR_ORDER):
        base = 2 + 13 * index
        source[name] = list(range(base, base + 6))    # ACC xyz + GYR xyz; MAG/QUAT dropped
        wanted.update(source[name])
    order = sorted(wanted)
    position = {column: slot for slot, column in enumerate(order)}
    slots = {name: [position[c] for c in columns] for name, columns in source.items()}
    return order, position, slots


def _runs(codes: np.ndarray):
    """Yield (label_code, start, stop) for each maximal run of one non-zero label."""
    boundaries = np.flatnonzero(np.diff(codes)) + 1
    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(codes)]):
        code = int(codes[start])
        if code != 0:
            yield code, int(start), int(stop)


def _sample_period(clock: np.ndarray) -> float:
    """Seconds per sample from the MEDIAN step, or NaN when the log's clock is unusable.

    Not `(n - 1) / span`: measured over all 46 logs, most files contain exactly one BACKWARDS jump
    of 1300-2100 s (a master-clock reset mid-recording) and no forward gaps at all. The span
    estimator therefore reads 79-98 Hz on a stream whose every single step is 0.0200 s.

    Three logs — subject6_self, subject13_self, subject15_mutual4 — carry the SAME timestamp on
    every row. They are also EMPTY (see EMPTY_LOGS below), so they are skipped entirely and take no
    part in the rate statistics.
    """
    step = float(np.median(np.diff(clock)))
    return step if step > 0 else float("nan")


def create_manifest() -> dict:
    return {
        "dataset_name": "REALDISP",
        "description": (
            "Realistic sensor displacement benchmark. 17 subjects performed 33 warm-up, fitness "
            "and cool-down exercises while wearing 9 Xsens MTx inertial units (both calves, both "
            "thighs, both lower arms, both upper arms, back) at 50 Hz, recorded under three "
            "placement regimes: instructor-placed (ideal), subject-placed (self), and deliberately "
            "displaced (mutual, 4 to 7 units rotated/translated). Banos et al., UbiComp 2012."
        ),
        "sampling_rate_hz": NATIVE_RATE,
        "channels": [
            {"name": f"{sensor}_{kind}_{axis}",
             "description": (f"{'accelerometer' if kind == 'acc' else 'gyroscope'} {axis}-axis, "
                             f"Xsens MTx on the {sensor.upper()} body location; "
                             f"{'m/s^2 (gravity present)' if kind == 'acc' else 'rad/s'}"),
             "sampling_rate_hz": NATIVE_RATE}
            for sensor in SENSOR_ORDER for kind in ("acc", "gyro") for axis in "xyz"
        ],
        "subjects": 17,
        "activities": sorted(ACTIVITIES.values()),
        "placements": list(SENSOR_ORDER),
        "device_profile": "device",
        "gravity_state": "present",
        "license": "open (UCI Machine Learning Repository #305)",
        "citation": ("Banos O, Damas M, Pomares H, Rojas I, Toth MA, Amft O. A benchmark dataset "
                     "to evaluate sensor displacement in activity recognition. UbiComp 2012, "
                     "1026-1035. doi:10.1145/2370216.2370437"),
        "excluded_logs": (
            "subject6_self, subject13_self and subject15_mutual4 are released as placeholders: "
            "every sensor column is identically zero on every row while the activity-label track is "
            "complete. They are skipped, so subject06 and subject13 have no `self` condition and "
            "subject15 has no `mutual4`."
        ),
        "displacement_note": (
            "The placement regime is encoded in the session id (`_ideal_`, `_self_`, `_mutual4_` "
            "..`_mutual7_`). ideal-vs-self for the same subject and exercise is a real "
            "change-of-configuration pair, not a synthetic rotation."
        ),
    }


def main() -> None:
    logs = sorted(RAW.glob("subject*_*.log"))
    if not logs:
        raise SystemExit(f"raw logs not found at {RAW}")
    sessions_dir = HERE / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    wanted, position, slots = _column_indices()

    labels: dict[str, list[str]] = {}
    magnitudes: list[float] = []
    gyro_peaks: list[float] = []
    rates: list[float] = []
    dropped_short = 0
    straddling_reset = 0
    broken_clocks: list[str] = []
    empty_logs: list[str] = []
    n_rows = 0

    for path in logs:
        subject_token, config = path.stem.split("_", 1)
        subject = f"subject{int(subject_token.removeprefix('subject')):02d}"
        table = pd.read_csv(path, sep=r"\s+", header=None, usecols=wanted,
                            dtype=np.float64).to_numpy()
        seconds = table[:, position[0]]
        micros = table[:, position[1]]
        codes = table[:, position[119]].astype(int)
        clock = seconds + micros * 1e-6
        sensor_columns = [slot for name in SENSOR_ORDER for slot in slots[name]]
        if np.abs(table[:, sensor_columns]).max() <= ALL_ZERO_TOLERANCE:
            empty_logs.append(path.name)
            print(f"  {path.name}: SKIPPED, every sensor column is identically zero", flush=True)
            continue
        period = _sample_period(clock)
        if np.isnan(period):
            broken_clocks.append(path.name)
            resets = np.empty(0, dtype=int)     # no usable clock: nothing to detect
        else:
            rates.append(1.0 / period)
            resets = np.flatnonzero(np.diff(clock) <= 0)
        unknown = set(np.unique(codes)) - set(ACTIVITIES) - {0}
        if unknown:
            raise ValueError(f"{path.name}: unknown activity codes {sorted(unknown)}")

        seen: dict[int, int] = {}
        for code, start, stop in _runs(codes):
            block = seen.get(code, 0)
            seen[code] = block + 1
            if (stop - start) < MIN_BLOCK_SEC * NATIVE_RATE:
                dropped_short += 1
                continue
            # A block that straddles the master-clock reset would carry a discontinuous timebase.
            if np.any((resets >= start) & (resets < stop - 1)):
                straddling_reset += 1
                continue
            activity = ACTIVITIES[code]
            session_id = f"{subject}_{config}_{activity}_{block:02d}"
            frame = pd.DataFrame({
                "timestamp_sec": np.arange(stop - start, dtype=np.float64) / NATIVE_RATE})
            for sensor, columns in slots.items():
                chunk = table[start:stop, columns]
                magnitudes.append(float(np.median(np.linalg.norm(chunk[:, :3], axis=1))))
                gyro_peaks.append(float(np.percentile(np.abs(chunk[:, 3:]), 99.9)))
                for offset, name in enumerate(("acc_x", "acc_y", "acc_z",
                                               "gyro_x", "gyro_y", "gyro_z")):
                    frame[f"{sensor}_{name}"] = chunk[:, offset].astype(np.float32)
            frame["activity"] = activity
            frame["subject"] = subject
            target = sessions_dir / session_id
            target.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(target / "data.parquet", index=False)
            labels[session_id] = [activity]
            n_rows += len(frame)
        print(f"  {path.name}: {len(seen)} activities", flush=True)

    reference = float(np.median(magnitudes))
    if not ACC_MS2_RANGE[0] <= reference <= ACC_MS2_RANGE[1]:
        raise ValueError(f"median |acc| = {reference:.3f}, outside the m/s^2 range {ACC_MS2_RANGE}")
    peak = float(np.max(gyro_peaks))
    if peak > GYRO_RADS_MAX:
        raise ValueError(f"|gyro| p99.9 max = {peak:.1f} exceeds {GYRO_RADS_MAX} rad/s; the source "
                         "is probably deg/s now. Convert explicitly rather than assuming.")
    measured_rate = float(np.median(rates))
    if abs(measured_rate - NATIVE_RATE) > 1.0:
        raise ValueError(f"median log rate {measured_rate:.3f} Hz != documented {NATIVE_RATE} Hz")
    if len(empty_logs) > len(logs) // 4:
        raise ValueError(f"{len(empty_logs)}/{len(logs)} logs are identically zero "
                         f"({empty_logs[:5]}...); the download is probably truncated.")
    if len(broken_clocks) > len(logs) // 4:
        raise ValueError(f"{len(broken_clocks)}/{len(logs)} logs have an unusable clock "
                         f"({broken_clocks[:5]}...); the release layout has changed.")

    (HERE / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True))
    (HERE / "recordings.json").write_text(json.dumps(
        {session: recording_id(session) for session in sorted(labels)}, indent=2, sort_keys=True))
    (HERE / "manifest.json").write_text(json.dumps(create_manifest(), indent=2))
    (HERE / "metadata.json").write_text(json.dumps({
        "dataset": "realdisp", "sampling_rate_hz": NATIVE_RATE, "pre_windowed": False,
    }, indent=2))

    hours = n_rows / NATIVE_RATE / 3600.0
    print(f"[realdisp] units verified: median |acc| {reference:.3f} m/s^2 (gravity present), "
          f"|gyro| p99.9 max {peak:.1f} rad/s, clock {measured_rate:.3f} Hz over "
          f"{len(rates)}/{len(logs)} logs; {len(empty_logs)} skipped as identically zero "
          f"({', '.join(empty_logs) or 'none'}); {len(broken_clocks)} with a constant timestamp "
          f"({', '.join(broken_clocks) or 'none'})", flush=True)
    print(f"[realdisp] {len(labels)} activity-block sessions from {len(logs)} logs, {n_rows:,} "
          f"samples, {hours:.2f} h labelled; {dropped_short} blocks shorter than "
          f"{MIN_BLOCK_SEC:g}s and {straddling_reset} blocks straddling a master-clock reset "
          f"dropped -> {sessions_dir}", flush=True)


if __name__ == "__main__":
    main()
