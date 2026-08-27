"""Convert DSADS (UCI "Daily and Sports Activities") to the HALO session format.

Source: `data/a{01..19}/p{1..8}/s{01..60}.txt`, 9,120 files. 19 activities x 8 subjects x 60
segments; each segment is 125 rows x 45 columns = 5 s at 25 Hz.

Layout (UCI dataset description): five Xsens MTx units, in column-block order
**T, RA, LA, RL, LL**, each contributing 9 columns in the order acc xyz, gyro xyz, mag xyz.
Magnetometer columns are dropped at the canonical-channel boundary.

The UCI block names are anatomically vague and this matters, because HALO conditions on the
placement TEXT. Barshan & Yuksek section 3 is explicit: "two of the +/-18g sensor units are placed
on the sides of the knees (the right-hand side of the right knee and the left-hand side of the left
knee), the remaining +/-18g unit is placed on the subject's chest, and the two +/-5g units on the
wrists". So T is the CHEST, RA/LA are the right/left WRIST, and RL/LL are the right/left KNEE — not
"arm" and "leg". The prefixes below use the paper's anatomy. Note also that the wrists use the
+/-5g range and the chest and knees the +/-18g range.

Segments are RE-JOINED here. UCI distributes each activity as "the 5-min signals divided into 5-sec
segments", so `s01..s60` in order reconstruct the original 5-minute recording. A 5 s segment cannot
produce a single 6 s analysis window, so converting segment-by-segment would yield an empty grid
from a perfectly good dataset. The join is VERIFIED rather than assumed: the converter compares the
sample-to-sample step across each segment boundary with the steps inside the segments, and fails if
the boundaries are discontinuous (which is what a non-temporal segment ordering would look like).

All five placements go into one parquet per (activity, subject) with a per-unit column prefix, so
the five deployment streams are simultaneous.

DSADS is in `EXCLUDED_PRIMARY_DATASETS` for training on placement grounds — torso and limb IMUs are
not a phone-pocket or watch-wrist deployment. It is converted for the placement-generalisation and
multi-configuration arms.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "downloads" / "x" / "data"
NATIVE_RATE = 25.0
SEGMENT_ROWS = 125

# UCI column-block order. Index i occupies columns 9*i .. 9*i+8 (acc xyz, gyro xyz, mag xyz).
UNIT_ORDER = ("chest", "right_wrist", "left_wrist", "right_knee", "left_knee")

# UCI dataset description, activities a01..a19, kept in the source's own wording.
ACTIVITIES = {
    1: "sitting",
    2: "standing",
    3: "lying_on_back",
    4: "lying_on_right_side",
    5: "ascending_stairs",
    6: "descending_stairs",
    7: "standing_still_in_an_elevator",
    8: "moving_around_in_an_elevator",
    9: "walking_in_a_parking_lot",
    10: "walking_on_a_treadmill_in_flat_position",
    11: "walking_on_a_treadmill_in_inclined_position",
    12: "running_on_a_treadmill",
    13: "exercising_on_a_stepper",
    14: "exercising_on_a_cross_trainer",
    15: "cycling_on_an_exercise_bike_in_horizontal_position",
    16: "cycling_on_an_exercise_bike_in_vertical_position",
    17: "rowing",
    18: "jumping",
    19: "playing_basketball",
}

ACC_MS2_RANGE = (8.0, 11.5)      # dataset median |acc|
GYRO_RADS_MAX = 40.0             # p99.9 |w|; deg/s would be ~57x larger

# A temporal segment ordering means the step ACROSS a join looks like the steps INSIDE a segment.
# Measured over all 8,968 joins: median ratio 1.09 against the interior median, so the ordering IS
# temporal (a shuffle would centre well above 1; an unrelated segment pair on a01/p1 scores 5x the
# interior step where the true neighbour scores 1.1x). Real discontinuities are not smoothed over
# and not thrown away: the session is SPLIT there, so no window ever straddles a gap.
#
# The test is a PERCENTILE within the recording's own interior-step distribution, not a fixed
# multiple of the median. That distinction is load-bearing and was measured on 2026-08-11: a quiet
# activity's median interior step is sensor noise, so any small real movement clears "3x the median"
# and the recording gets shredded even though its joins are statistically indistinguishable from
# continuous data. Mean join percentile per activity (0.50 = indistinguishable from an interior
# step) against the cuts a 3x-median rule made:
#
#     a05 ascending_stairs    0.923   254 cuts   <- genuinely discontinuous; you cannot ascend
#     a06 descending_stairs   0.862   160 cuts      stairs continuously for five minutes
#     a08 moving_in_elevator  0.507    88 cuts   <- CONTINUOUS, yet shredded by the ratio rule
#     a18 jumping             0.573    44 cuts
#     everything else      0.49-0.55  <=28 cuts
#
# The 3x rule cut 726 joins and lost 504 windows per placement; 312 of those cuts were on
# recordings whose joins average the 50th percentile, i.e. false positives. The percentile rule
# keeps a05/a06's real cuts and drops the rest. See
# The historical converter audit is indexed by docs/LEGACY.md.
JOIN_STEP_PERCENTILE = 99.9
# If the TYPICAL join sat this high in the interior distribution the ordering itself would be wrong
# and splitting would not save it — convert per segment instead.
MAX_MEDIAN_JOIN_PERCENTILE = 0.90


def recording_id(session_id: str) -> str:
    """The continuous capture a session was cut out of: one 5-minute (subject, activity) recording.

    UCI distributes that recording as 60 five-second segments, which this converter re-joins; a
    discontinuous join then splits it into `{subject}_{activity}_{piece:02d}` pieces. Those pieces
    are seconds apart inside one bout, so they are NOT independent enrollment executions. Measured
    2026-08-11: 2 pieces per (subject, label) against 1 real recording, a 3.2x overcount.
    """
    return re.sub(r"_\d{2}$", "", session_id)


def create_manifest() -> dict:
    return {
        "dataset_name": "DSADS",
        "description": (
            "Daily and Sports Activities. 8 subjects (4 female, 4 male, 20-30 years) performed 19 "
            "activities for 5 minutes each while wearing 5 Xsens MTx units on the chest, both "
            "wrists and the outer sides of both knees, at 25 Hz. Barshan & Yuksek. Source: UCI "
            "Machine Learning Repository, "
            "Daily and Sports Activities Data Set."
        ),
        "sampling_rate_hz": NATIVE_RATE,
        "channels": [
            {"name": f"{unit}_{kind}_{axis}",
             "description": (f"{'accelerometer' if kind == 'acc' else 'gyroscope'} {axis}-axis, "
                             f"Xsens MTx on the {unit.replace('_', ' ')}; "
                             f"{'m/s^2 (gravity present)' if kind == 'acc' else 'rad/s'}"),
             "sampling_rate_hz": NATIVE_RATE}
            for unit in UNIT_ORDER for kind in ("acc", "gyro") for axis in "xyz"
        ],
        "subjects": 8,
        "activities": sorted(ACTIVITIES.values()),
        "placements": list(UNIT_ORDER),
        "device_profile": "device",
        "gravity_state": "present",
        "license": "open (UCI Machine Learning Repository)",
        "citation": ("Barshan B, Yuksek MC. Recognizing daily and sports activities in two open "
                     "source machine learning environments using body-worn sensor units. The "
                     "Computer Journal 57(11):1649-1667, 2014. doi:10.1093/comjnl/bxt075"),
        "segment_note": (
            "The distributed 5-second segments are re-joined in filename order to reconstruct the "
            "original 5-minute recording; the join continuity is asserted at convert time."
        ),
    }


def main() -> None:
    if not RAW.is_dir():
        raise SystemExit(f"raw segment tree not found at {RAW}")
    sessions_dir = HERE / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, list[str]] = {}
    magnitudes: list[float] = []
    gyro_peaks: list[float] = []
    join_percentiles: list[float] = []
    n_discontinuities = 0
    n_rows = 0

    for activity_dir in sorted(RAW.glob("a[0-9][0-9]")):
        code = int(activity_dir.name[1:])
        if code not in ACTIVITIES:
            raise ValueError(f"unknown activity directory {activity_dir.name}")
        activity = ACTIVITIES[code]
        for subject_dir in sorted(activity_dir.glob("p[0-9]*")):
            subject = f"p{int(subject_dir.name[1:]):02d}"
            segments = sorted(subject_dir.glob("s[0-9][0-9].txt"))
            if len(segments) != 60:
                print(f"  WARNING {activity_dir.name}/{subject_dir.name}: {len(segments)} segments, "
                      "expected 60", flush=True)
            blocks = []
            for path in segments:
                block = np.loadtxt(path, delimiter=",", dtype=np.float64)
                if block.shape != (SEGMENT_ROWS, 45):
                    raise ValueError(f"{path}: expected ({SEGMENT_ROWS}, 45), got {block.shape}")
                blocks.append(block)
            table = np.concatenate(blocks, axis=0)
            if not np.isfinite(table).all():
                raise ValueError(f"{activity_dir.name}/{subject_dir.name}: non-finite samples")

            # Join continuity: |x[t+1] - x[t]| across segment boundaries vs inside segments.
            steps = np.abs(np.diff(table, axis=0)).mean(axis=1)
            join_at = np.arange(1, len(blocks)) * SEGMENT_ROWS - 1
            interior = np.sort(np.delete(steps, join_at))
            join_steps = steps[join_at]
            # Each join's rank within THIS recording's own interior-step distribution. Continuous
            # data puts joins uniformly through that distribution, so the median lands near 0.5.
            percentiles = np.searchsorted(interior, join_steps, side="left") / len(interior)
            join_percentiles.append(float(np.median(percentiles)))
            threshold = float(np.percentile(interior, JOIN_STEP_PERCENTILE))
            # Cut at the discontinuous joins so no window can straddle one.
            cuts = [0, *(int(join_at[i]) + 1 for i in np.flatnonzero(join_steps > threshold)),
                    len(table)]
            n_discontinuities += len(cuts) - 2

            for piece, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:])):
                chunk = table[start:stop]
                if len(chunk) < 2:
                    continue
                session_id = f"{subject}_{activity}" if len(cuts) == 2 \
                    else f"{subject}_{activity}_{piece:02d}"
                frame = pd.DataFrame({
                    "timestamp_sec": np.arange(len(chunk), dtype=np.float64) / NATIVE_RATE})
                for index, unit in enumerate(UNIT_ORDER):
                    base = 9 * index
                    magnitudes.append(
                        float(np.median(np.linalg.norm(chunk[:, base:base + 3], axis=1))))
                    gyro_peaks.append(
                        float(np.percentile(np.abs(chunk[:, base + 3:base + 6]), 99.9)))
                    for offset, name in enumerate(("acc_x", "acc_y", "acc_z",
                                                   "gyro_x", "gyro_y", "gyro_z")):
                        frame[f"{unit}_{name}"] = chunk[:, base + offset].astype(np.float32)
                frame["activity"] = activity
                frame["subject"] = subject
                target = sessions_dir / session_id
                target.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(target / "data.parquet", index=False)
                labels[session_id] = [activity]
                n_rows += len(frame)
        print(f"  {activity_dir.name} ({activity}) done", flush=True)

    reference = float(np.median(magnitudes))
    if not ACC_MS2_RANGE[0] <= reference <= ACC_MS2_RANGE[1]:
        raise ValueError(f"median |acc| = {reference:.3f}, outside the m/s^2 range {ACC_MS2_RANGE}")
    peak = float(np.max(gyro_peaks))
    if peak > GYRO_RADS_MAX:
        raise ValueError(f"|gyro| p99.9 max = {peak:.1f} exceeds {GYRO_RADS_MAX} rad/s; the source "
                         "is probably deg/s now.")
    typical_join = float(np.median(join_percentiles))
    if typical_join > MAX_MEDIAN_JOIN_PERCENTILE:
        raise ValueError(
            f"the TYPICAL segment join is discontinuous (median join percentile {typical_join:.2f} "
            f"> {MAX_MEDIAN_JOIN_PERCENTILE}); s01..s60 are not consecutive in time, so re-joining "
            "them would fabricate a continuous recording. Convert per segment instead.")

    (HERE / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True))
    (HERE / "recordings.json").write_text(json.dumps(
        {session: recording_id(session) for session in sorted(labels)}, indent=2, sort_keys=True))
    (HERE / "manifest.json").write_text(json.dumps(create_manifest(), indent=2))
    (HERE / "metadata.json").write_text(json.dumps({
        "dataset": "dsads", "sampling_rate_hz": NATIVE_RATE, "pre_windowed": False,
    }, indent=2))

    hours = n_rows / NATIVE_RATE / 3600.0
    print(f"[dsads] units verified: median |acc| {reference:.3f} m/s^2 (gravity present), "
          f"|gyro| p99.9 max {peak:.2f} rad/s; per-recording median segment-join percentile "
          f"{typical_join:.2f} (0.50 = indistinguishable from an interior step)", flush=True)
    print(f"[dsads] {len(labels)} sessions, {n_rows:,} samples, {hours:.2f} h, "
          f"{len(ACTIVITIES)} activities; split at {n_discontinuities} discontinuous joins "
          f"-> {sessions_dir}", flush=True)


if __name__ == "__main__":
    main()
