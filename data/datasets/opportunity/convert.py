"""Convert the OPPORTUNITY activity-recognition dataset (UCI) to the HALO session format.

Source: `OpportunityUCIDataset/dataset/S{1..4}-{ADL1..ADL5,Drill}.dat`, 24 runs from 4 subjects in a
sensor-rich kitchen. Only the five body-worn Xsens IMUs are read (BACK, RUA, RLA, LUA, LLA); the 12
loose accelerometers, the shoe sensors, the object sensors and the ambient sensors are not.

Column positions come from the release's own `dataset/column_names.txt`, which is 1-based; the
0-based offsets used here are BACK 37, RUA 50, RLA 63, LUA 76, LLA 89, each acc xyz then gyro xyz.

Units. The column file documents acceleration as **milli-g** (`round(original / 9.8 * 1000)`) and
leaves the gyroscope as `unit = unknown` with `round(original * 1000)`. Both are resolved by
measurement and asserted at convert time: a still IMU reads |a| = 1001 milli-g, i.e. **g once
divided by 1000**; and during labelled walking the forearm gyroscope reaches p99 = 2685, i.e.
**2.685 rad/s once divided by 1000** (as deg/s that would be 2.7 deg/s for a swinging forearm, which
is not physically possible). Both are therefore scaled by 1/1000 here, giving g and rad/s.

**Labels: the Locomotion track, not the mid-level gestures.** The gesture track (`ML_Both_Arms`,
17 classes) is what makes Opportunity attractive on paper — the Drill runs are 20 repetitions of each
gesture — but the gestures are short by construction against a 6 s analysis window, so a window
majority-vote over them mostly returns the null class. Locomotion (stand / walk / sit / lie) is the
label that survives at this window length. The gesture cell is reported as unsupported rather than
forced; the historical audit is indexed by docs/LEGACY.md.

Opportunity is in `EXCLUDED_PRIMARY_DATASETS` for training on placement grounds — back and
upper/lower-arm IMUs are not a phone-pocket or watch-wrist deployment.

**Enrollment caveat.** One session is one contiguous Locomotion block, and a single ADL run contains
many blocks of the same activity minutes apart. Those are not independent enrollment executions;
`recording_id` below groups them back onto the run they came from, which is the honest leakage unit.
The real ceiling is 6 recordings per subject (ADL1-5 + Drill), all on one day.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "downloads" / "x" / "OpportunityUCIDataset" / "dataset"
NATIVE_RATE = 30.0
MILLI = 1.0 / 1000.0

# 0-based column of each IMU's accelerometer x; gyro x is +3. (column_names.txt is 1-based.)
SENSORS = {
    "back": 37,
    "right_upper_arm": 50,
    "right_lower_arm": 63,
    "left_upper_arm": 76,
    "left_lower_arm": 89,
}
LOCOMOTION_COLUMN = 243          # 0-based; column_names.txt "Column: 244 Locomotion"
GESTURE_COLUMN = 249             # 0-based; "Column: 250 ML_Both_Arms"

# label_legend.txt, Locomotion track.
ACTIVITIES = {1: "standing", 2: "walking", 4: "sitting", 5: "lying"}

ACC_MILLI_G_RANGE = (700.0, 1400.0)      # median |acc| BEFORE the 1/1000 scaling
GYRO_MILLI_RADS_MIN = 500.0              # p99 |w| BEFORE scaling, during labelled walking
MAX_NAN_RUN = 3                          # longer holes split the session instead of interpolating
WINDOW_SECONDS = 6.0


def recording_id(session_id: str) -> str:
    """The continuous capture a session was cut out of: `S<n>-<run>.dat`, i.e. `s01_adl1`.

    Session ids here are `{subject}_{run}_{activity}_{block}`, one per contiguous Locomotion block.
    Blocks of the same activity inside one run are seconds or minutes apart in a single recording,
    so they are NOT independent enrollment executions — treating them as such would measure
    adjacent-window binding. Measured 2026-08-11: 30 blocks per (subject, label) against 6 real
    recordings, a 10.2x overcount. The historical audit is indexed by docs/LEGACY.md.
    """
    return "_".join(session_id.split("_")[:2])


def _runs(codes: np.ndarray):
    boundaries = np.flatnonzero(np.diff(codes)) + 1
    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(codes)]):
        yield int(codes[start]), int(start), int(stop)


def create_manifest(gesture_median_s: float) -> dict:
    return {
        "dataset_name": "OPPORTUNITY",
        "description": (
            "4 subjects performing morning-routine activities of daily living in a sensor-rich "
            "kitchen, 5 ADL runs plus one Drill run each. Only the five body-worn Xsens inertial "
            "units (back, both upper arms, both lower arms) are converted, at 30 Hz. Chavarriaga "
            "et al., Pattern Recognition Letters 34(15):2033-2042, 2013."
        ),
        "sampling_rate_hz": NATIVE_RATE,
        "channels": [
            {"name": f"{sensor}_{kind}_{axis}",
             "description": (f"{'accelerometer' if kind == 'acc' else 'gyroscope'} {axis}-axis, "
                             f"Xsens inertial unit on the {sensor.replace('_', ' ')}; "
                             f"{'g (gravity present)' if kind == 'acc' else 'rad/s'}"),
             "sampling_rate_hz": NATIVE_RATE}
            for sensor in SENSORS for kind in ("acc", "gyro") for axis in "xyz"
        ],
        "subjects": 4,
        "activities": sorted(ACTIVITIES.values()),
        "placements": list(SENSORS),
        "device_profile": "device",
        "gravity_state": "present",
        "license": "open (UCI Machine Learning Repository)",
        "citation": ("Chavarriaga R, Sagha H, Calatroni A, Digumarti ST, Troster G, Millan JdR, "
                     "Roggen D. The Opportunity challenge: A benchmark database for on-body "
                     "sensor-based activity recognition. Pattern Recognition Letters "
                     "34(15):2033-2042, 2013. doi:10.1016/j.patrec.2012.12.014"),
        "unit_note": (
            "Source accelerometer is milli-g and source gyroscope is milli-rad/s (the column file "
            "leaves the gyroscope unit 'unknown'; it was resolved by measurement). Both are scaled "
            "by 1/1000 in this converter, to g and rad/s."
        ),
        "label_note": (
            f"Labels are the Locomotion track. The ML_Both_Arms gesture track is not used: its "
            f"instances have a median duration of {gesture_median_s:.1f} s against a "
            f"{WINDOW_SECONDS:g} s analysis window, so a window majority vote over gestures returns "
            f"the null class almost everywhere. That cell is unsupported, not forced."
        ),
    }


def main() -> None:
    if not RAW.is_dir():
        raise SystemExit(f"raw dataset directory not found at {RAW}")
    sessions_dir = HERE / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, list[str]] = {}
    magnitudes: list[float] = []
    walking_gyro: list[float] = []
    rates: list[float] = []
    gesture_durations: list[float] = []
    interpolated = 0
    split_at_holes = 0
    dropped_hole_rows = 0
    short_blocks = 0
    n_rows = 0

    columns = [0, LOCOMOTION_COLUMN, GESTURE_COLUMN]
    for base in SENSORS.values():
        columns.extend(range(base, base + 6))
    order = sorted(set(columns))
    position = {column: slot for slot, column in enumerate(order)}

    for path in sorted(RAW.glob("S[0-9]-*.dat")):
        subject_token, run = path.stem.split("-", 1)
        subject = f"s{int(subject_token[1:]):02d}"
        table = pd.read_csv(path, sep=r"\s+", header=None, usecols=order,
                            dtype=np.float64).to_numpy()
        clock = table[:, position[0]]
        rates.append(1000.0 / float(np.median(np.diff(clock))))
        codes = np.nan_to_num(table[:, position[LOCOMOTION_COLUMN]], nan=0.0).astype(int)
        gestures = np.nan_to_num(table[:, position[GESTURE_COLUMN]], nan=0.0).astype(int)
        for code, start, stop in _runs(gestures):
            if code:
                gesture_durations.append((stop - start) / NATIVE_RATE)
        unknown = set(np.unique(codes)) - set(ACTIVITIES) - {0}
        if unknown:
            raise ValueError(f"{path.name}: unknown Locomotion codes {sorted(unknown)}")

        signals = {}
        holes = np.zeros(len(table), dtype=bool)
        for sensor, base in SENSORS.items():
            block = table[:, [position[c] for c in range(base, base + 6)]]
            magnitudes.append(float(np.median(np.linalg.norm(
                block[np.isfinite(block).all(axis=1), :3], axis=1))))
            walk = (codes == 2) & np.isfinite(block).all(axis=1)
            if walk.any() and sensor.endswith("lower_arm"):
                walking_gyro.append(float(np.percentile(np.abs(block[walk, 3:]), 99)))
            signals[sensor] = block
            holes |= ~np.isfinite(block).all(axis=1)

        # Short dropouts are interpolated; a longer hole splits the recording so no window spans it.
        edges = np.flatnonzero(np.diff(np.r_[0, holes.view(np.int8), 0]))
        long_holes = [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2])
                      if b - a > MAX_NAN_RUN]
        good = ~holes
        for sensor, block in signals.items():
            for column in range(block.shape[1]):
                bad = ~np.isfinite(block[:, column])
                if bad.any():
                    block[bad, column] = np.interp(clock[bad], clock[good], block[good, column])
        interpolated += int(holes.sum())
        split_at_holes += len(long_holes)
        # A long hole splits the recording so no window spans it, and the hole ITSELF is dropped
        # rather than emitted. Its sensor rows are pure interpolation across the gap while its
        # label track is real and independent of the sensors, so keeping it would ship entirely
        # synthetic labelled windows. Measured: S2-ADL1 has one 220.9 s hole covering 165.1 s of
        # labelled samples, which produced 14 fabricated windows per placement (70 stream-windows)
        # before this exclusion. `data/quality/scan_duplicates` caught them independently as
        # near-constant duplicates.
        pieces: list[tuple[int, int]] = []
        cursor = 0
        for hole_start, hole_stop in long_holes:
            if hole_start > cursor:
                pieces.append((cursor, hole_start))
            dropped_hole_rows += hole_stop - hole_start
            cursor = hole_stop
        if cursor < len(table):
            pieces.append((cursor, len(table)))

        seen: dict[int, int] = {}
        for piece_start, piece_stop in pieces:
            for code, offset, end in _runs(codes[piece_start:piece_stop]):
                if code == 0:
                    continue
                start, stop = piece_start + offset, piece_start + end
                activity = ACTIVITIES[code]
                block_index = seen.get(code, 0)
                seen[code] = block_index + 1
                if (stop - start) < WINDOW_SECONDS * NATIVE_RATE:
                    short_blocks += 1
                session_id = f"{subject}_{run.lower()}_{activity}_{block_index:03d}"
                frame = pd.DataFrame({
                    "timestamp_sec": np.arange(stop - start, dtype=np.float64) / NATIVE_RATE})
                for sensor, block in signals.items():
                    chunk = block[start:stop]
                    for offset_index, name in enumerate(("acc_x", "acc_y", "acc_z")):
                        frame[f"{sensor}_{name}"] = (chunk[:, offset_index] * MILLI
                                                     ).astype(np.float32)
                    for offset_index, name in enumerate(("gyro_x", "gyro_y", "gyro_z")):
                        frame[f"{sensor}_{name}"] = (chunk[:, 3 + offset_index] * MILLI
                                                     ).astype(np.float32)
                frame["activity"] = activity
                frame["subject"] = subject
                target = sessions_dir / session_id
                target.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(target / "data.parquet", index=False)
                labels[session_id] = [activity]
                n_rows += stop - start
        print(f"  {path.name}: {sum(seen.values())} locomotion blocks", flush=True)

    reference = float(np.median(magnitudes))
    if not ACC_MILLI_G_RANGE[0] <= reference <= ACC_MILLI_G_RANGE[1]:
        raise ValueError(f"median |acc| = {reference:.1f}, outside the milli-g range "
                         f"{ACC_MILLI_G_RANGE}; column_names.txt documents milli-g.")
    gyro_reference = float(np.median(walking_gyro))
    if gyro_reference < GYRO_MILLI_RADS_MIN:
        raise ValueError(
            f"forearm |gyro| p99 during walking = {gyro_reference:.1f} before scaling, below "
            f"{GYRO_MILLI_RADS_MIN}; the source is not milli-rad/s and the 1/1000 scaling here "
            "would be wrong.")
    measured_rate = float(np.median(rates))
    if abs(measured_rate - NATIVE_RATE) > 1.0:
        raise ValueError(f"median clock rate {measured_rate:.3f} Hz != documented {NATIVE_RATE} Hz")

    gesture_median = float(np.median(gesture_durations)) if gesture_durations else float("nan")
    (HERE / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True))
    (HERE / "recordings.json").write_text(json.dumps(
        {session: recording_id(session) for session in sorted(labels)}, indent=2, sort_keys=True))
    (HERE / "manifest.json").write_text(json.dumps(create_manifest(gesture_median), indent=2))
    (HERE / "metadata.json").write_text(json.dumps({
        "dataset": "opportunity", "sampling_rate_hz": NATIVE_RATE, "pre_windowed": False,
    }, indent=2))

    hours = n_rows / NATIVE_RATE / 3600.0
    print(f"[opportunity] units verified: median |acc| {reference:.1f} milli-g -> "
          f"{reference * MILLI:.3f} g (gravity present); forearm |gyro| p99 while walking "
          f"{gyro_reference:.0f} milli-rad/s -> {gyro_reference * MILLI:.2f} rad/s; clock "
          f"{measured_rate:.3f} Hz", flush=True)
    print(f"[opportunity] {len(labels)} locomotion sessions, {n_rows:,} samples, {hours:.2f} h; "
          f"{interpolated:,} dropout samples interpolated, {split_at_holes} long holes split and "
          f"dropped ({dropped_hole_rows:,} samples, {dropped_hole_rows / NATIVE_RATE:.1f} s); "
          f"{short_blocks} blocks shorter than {WINDOW_SECONDS:g}s; gesture instances median "
          f"{gesture_median:.1f}s (below the window) -> {sessions_dir}", flush=True)


if __name__ == "__main__":
    main()
