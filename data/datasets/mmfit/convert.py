"""Convert MM-Fit (Stromback, Huang & Radu, IMWUT 2020) to the HALO session format.

Source: mmfit.github.io, `mm-fit/w{00..20}/` — 21 full workouts, each recorded simultaneously on
**four inertial devices** plus video/pose (not read here):

    sw_l   smartwatch, left wrist       sp_r   smartphone, right trouser pocket
    sw_r   smartwatch, right wrist      eb_l   earbud, left ear

Why this dataset is here: it is the corpus's cleanest **cross-configuration** source. The same
repetition of the same exercise is captured on a watch, a phone in a pocket and an earbud at once,
so a support example enrolled on one device and a query drawn from another is a genuine
change-of-configuration pair rather than a synthetic augmentation.

Each `.npy` is (N, 5) = `[frame, unix_timestamp_ms, x, y, z]`. Measured over all 21 workouts:
acceleration is **m/s^2 with gravity present** (median |a| 9.82-10.07) and angular rate is **rad/s**
(p99.9 of 4-8). The four devices log at genuinely different mean rates — 103 Hz and 104 Hz for the
watches, 212 Hz for the phone, 85 Hz for the earbud, and none of them uniformly — so they are
resampled onto ONE uniform grid from their shared wall clock. That is what makes the four streams
simultaneous sample-for-sample; leaving them on their own clocks would silently misalign them.

Resampling is NOT the same as interpolating across a dropout, and this source has many: the earbud
alone drops out 601 times for more than half a second. Those intervals are marked unobserved and any
labelled set overlapping one is dropped, rather than shipped as a straight line under a real
exercise label. See `MAX_GAP_SEC`.

`w{NN}_labels.csv` is `start_frame, end_frame, repetitions, exercise`, indexing the same video frame
counter as column 0 of every `.npy`, so label frames are mapped to wall-clock milliseconds through
that column. One labelled set becomes one session, which makes a *set* the unit of execution — sets
of the same exercise by the same subject are separated by minutes of other exercises, unlike the
within-bout repetitions of SPAR.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "downloads" / "x" / "mm-fit"
# One grid for all four devices. 100 Hz is at or below every device's own mean rate except the
# phone's 212 Hz, so only the phone is decimated and nothing is upsampled beyond its native content.
NATIVE_RATE = 100.0

DEVICES = {
    "sw_l": "left_wrist",
    "sw_r": "right_wrist",
    "sp_r": "right_pocket",
    "eb_l": "left_ear",
}

ACC_MS2_RANGE = (8.0, 11.5)
GYRO_RADS_MAX = 40.0
MIN_SET_SECONDS = 6.0            # a set shorter than one window contributes no grid row
MAX_CLOCK_SKEW_SEC = 5.0         # device start times must agree to within this

# A hole longer than this is an acquisition gap, not jitter, and is NEVER interpolated across.
# `np.interp` has no notion of a gap: it draws a straight line through one, so a 3.7 s earbud
# dropout becomes 3.7 s of smooth fabricated motion carrying a real exercise label. Measured over
# all 21 workouts on 2026-08-11:
#
#     device        samples      duplicate ts        gaps > 0.5 s   largest gap
#     eb_l earbud   4,175,310    1,163,923 (27.9%)   601            3.66 s
#     sp_r phone   10,482,011    1,052,285 (10.0%)    55            2.17 s
#     sw_r watch    5,142,870       27,535            1             7.83 s
#     sw_l watch    5,087,619       20,945            1            74.47 s
#
# The earbud's duplicate timestamps are packetised delivery (several samples share one packet
# stamp), which is benign for interpolation; the gaps are not. Sets that overlap a gap on ANY device
# are dropped rather than emitted, because the four placements are only useful here if they describe
# the same moment on all four. See docs/data/DATASET_EXPANSION_AUDIT_2026-08-11.md section 1.2.
MAX_GAP_SEC = 0.5

# Paper section 3.1: "Ten subjects participated in the data collection; two participants carried out
# six workout sessions each, one participated in two sessions, and the remaining participants carried
# out one workout session each" (2x6 + 1x2 + 7x1 = 21). Section 5.1.2 then gives a reproducible
# split by workout id. The full workout->person map is NOT released, so the only person-disjoint
# boundary available is this one: the cross-subject test workouts come from participants who appear
# nowhere else.
PAPER_SPLITS = {
    "train": (1, 2, 3, 4, 6, 7, 8, 16, 17, 18),
    "validation": (14, 15, 19),
    "seen_subject_test": (9, 10, 11),
    "cross_subject_test": (0, 5, 12, 13, 20),
}
N_PARTICIPANTS = 10


def create_manifest(workouts: int, activities: list[str]) -> dict:
    return {
        "dataset_name": "MM-Fit",
        "description": (
            "21 full workouts of 10 exercises recorded simultaneously on two smartwatches (both "
            "wrists), a smartphone in the right trouser pocket and an earbud, with per-set "
            "repetition counts. Accelerometer and gyroscope only here; the RGB-D video, 2D/3D pose "
            "and heart-rate streams are not converted. Stromback, Huang & Radu, IMWUT 4(4):1-22, "
            "2020. Source: mmfit.github.io."
        ),
        "sampling_rate_hz": NATIVE_RATE,
        "channels": [
            {"name": f"{prefix}_{kind}_{axis}",
             "description": (f"{'accelerometer' if kind == 'acc' else 'gyroscope'} {axis}-axis, "
                             f"{'smartwatch' if prefix.endswith('wrist') else 'smartphone' if prefix == 'right_pocket' else 'earbud'} "
                             f"at the {prefix.replace('_', ' ')}; "
                             f"{'m/s^2 (gravity present)' if kind == 'acc' else 'rad/s'}"),
             "sampling_rate_hz": NATIVE_RATE}
            for prefix in DEVICES.values() for kind in ("acc", "gyro") for axis in "xyz"
        ],
        "subjects": workouts,
        "activities": activities,
        "placements": list(DEVICES.values()),
        "device_profile": "watch",
        "gravity_state": "present",
        "license": "see mmfit.github.io",
        "citation": ("Stromback D, Huang S, Radu V. MM-Fit: Multimodal Deep Learning for Automatic "
                     "Exercise Logging across Sensing Devices. Proc. ACM IMWUT 4(4):168, 2020. "
                     "doi:10.1145/3432701"),
        "synchronisation_note": (
            "The four devices log at different, non-uniform rates on a shared wall clock; every "
            "device is resampled onto one uniform grid so the placements are simultaneous."
        ),
        "subject_note": (
            f"WORKOUT IS NOT PERSON. The paper (section 3.1) states the 21 workouts come from "
            f"{N_PARTICIPANTS} participants: two did six sessions each, one did two, and seven did "
            "one. The full workout-to-person map is not released, so this converter still writes "
            "one subject id per workout — but a workout-disjoint split is NOT person-disjoint, and "
            "cross-subject enrollment over arbitrary workout pairs can enrol and query the same "
            "person. The only person-disjoint boundary the release supports is the paper's own "
            "split (section 5.1.2), recorded in `paper_splits` below; the cross-subject test "
            "workouts come from participants who appear in no other split."
        ),
        "paper_splits": {name: list(ids) for name, ids in PAPER_SPLITS.items()},
        "participants": N_PARTICIPANTS,
    }


def recording_id(session_id: str) -> str:
    """The continuous capture a session was cut out of: one workout, `w00`..`w20`.

    Session ids are `{workout}_{exercise}_{set_index}`. The three sets of one exercise are cut out
    of a SINGLE continuous per-workout recording — they are separated by minutes of other exercises
    rather than by seconds, which makes them a reasonable within-session enrollment unit but not an
    across-session one. Grouping them onto the workout keeps that distinction honest: MM-Fit
    supports cross-CONFIGURATION enrollment (four devices at once), not cross-session.
    """
    return session_id.split("_")[0]


def _load_device(workout: Path, workout_id: str, device: str):
    """Return (clock_ms, (N,6) acc+gyro) for one device, or None if either file is missing."""
    parts = {}
    for modality, key in (("acc", "acc"), ("gyr", "gyro")):
        path = workout / f"{workout_id}_{device}_{modality}.npy"
        if not path.exists():
            return None
        table = np.load(path).astype(np.float64)
        if table.ndim != 2 or table.shape[1] != 5:
            raise ValueError(f"{path}: expected (N, 5), got {table.shape}")
        parts[key] = table
    return parts


def main() -> None:
    if not RAW.is_dir():
        raise SystemExit(f"raw mm-fit tree not found at {RAW}")
    sessions_dir = HERE / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, list[str]] = {}
    magnitudes: list[float] = []
    gyro_peaks: list[float] = []
    skews: list[float] = []
    gap_seconds: list[float] = []
    short_sets = 0
    fabricated_samples = 0
    dropped_gap_sets = 0
    dropped_gap_seconds = 0.0
    n_rows = 0
    activities: set[str] = set()

    workouts = sorted(RAW.glob("w[0-9][0-9]"))
    for workout in workouts:
        workout_id = workout.name
        label_path = workout / f"{workout_id}_labels.csv"
        if not label_path.exists():
            print(f"  {workout_id}: no labels, skipped", flush=True)
            continue
        sets = pd.read_csv(label_path, header=None,
                           names=["start_frame", "end_frame", "reps", "exercise"])

        loaded = {}
        for device in DEVICES:
            parts = _load_device(workout, workout_id, device)
            if parts is None:
                raise ValueError(f"{workout_id}: device {device} is missing an acc or gyr file")
            loaded[device] = parts

        # One shared uniform grid over the span every device covers.
        starts = [p[k][0, 1] for p in loaded.values() for k in p]
        stops = [p[k][-1, 1] for p in loaded.values() for k in p]
        skews.append((max(starts) - min(starts)) / 1000.0)
        if skews[-1] > MAX_CLOCK_SKEW_SEC:
            raise ValueError(f"{workout_id}: device start times differ by {skews[-1]:.1f}s, more "
                             f"than {MAX_CLOCK_SKEW_SEC}s; the shared clock cannot be trusted.")
        lo, hi = max(starts), min(stops)
        grid = np.arange(lo, hi, 1000.0 / NATIVE_RATE)

        # Grid points that no device actually observed. A gap on ANY of the four devices poisons
        # that instant for all of them: the whole point of this source is that the four placements
        # describe the same moment, so a set is only usable where every device was recording.
        fabricated = np.zeros(len(grid), dtype=bool)
        signals: dict[str, np.ndarray] = {}
        for device, parts in loaded.items():
            block = np.empty((len(grid), 6), dtype=np.float32)
            for offset, key in ((0, "acc"), (3, "gyro")):
                table = parts[key]
                for axis in range(3):
                    block[:, offset + axis] = np.interp(grid, table[:, 1], table[:, 2 + axis])
                clock = table[:, 1]
                holes = np.flatnonzero(np.diff(clock) > MAX_GAP_SEC * 1000.0)
                for hole in holes:
                    start, stop = np.searchsorted(grid, [clock[hole], clock[hole + 1]])
                    fabricated[start:stop] = True
                    gap_seconds.append(float(clock[hole + 1] - clock[hole]) / 1000.0)
            magnitudes.append(float(np.median(np.linalg.norm(block[:, :3], axis=1))))
            gyro_peaks.append(float(np.percentile(np.abs(block[:, 3:]), 99.9)))
            signals[DEVICES[device]] = block
        fabricated_samples += int(fabricated.sum())

        # Frame -> wall clock, read off the (frame, timestamp) pairs the recordings already carry.
        reference = loaded["sw_l"]["acc"]
        frame_to_ms = lambda f: np.interp(f, reference[:, 0], reference[:, 1])

        seen: dict[str, int] = {}
        for _, row in sets.iterrows():
            activity = str(row["exercise"]).strip()
            activities.add(activity)
            start_ms, stop_ms = frame_to_ms(row["start_frame"]), frame_to_ms(row["end_frame"])
            start, stop = np.searchsorted(grid, [start_ms, stop_ms])
            if stop - start < 2:
                continue
            index = seen.get(activity, 0)
            seen[activity] = index + 1
            if fabricated[start:stop].any():
                # Emitting this would ship interpolated straight lines under a real exercise label.
                # Trimming to the clean part instead would silently shorten the set and misstate the
                # repetition count the annotation carries, so the set is dropped whole.
                dropped_gap_sets += 1
                dropped_gap_seconds += float(fabricated[start:stop].sum()) / NATIVE_RATE
                continue
            if (stop - start) < MIN_SET_SECONDS * NATIVE_RATE:
                short_sets += 1
            session_id = f"{workout_id}_{activity}_{index:02d}"
            frame = pd.DataFrame({
                "timestamp_sec": np.arange(stop - start, dtype=np.float64) / NATIVE_RATE})
            for prefix, block in signals.items():
                chunk = block[start:stop]
                for offset, name in enumerate(("acc_x", "acc_y", "acc_z",
                                               "gyro_x", "gyro_y", "gyro_z")):
                    frame[f"{prefix}_{name}"] = chunk[:, offset]
            frame["activity"] = activity
            frame["subject"] = workout_id
            target = sessions_dir / session_id
            target.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(target / "data.parquet", index=False)
            labels[session_id] = [activity]
            n_rows += stop - start
        print(f"  {workout_id}: {len(sets)} sets, {len(seen)} exercises", flush=True)

    reference_mag = float(np.median(magnitudes))
    if not ACC_MS2_RANGE[0] <= reference_mag <= ACC_MS2_RANGE[1]:
        raise ValueError(f"median |acc| = {reference_mag:.3f}, outside {ACC_MS2_RANGE} m/s^2")
    peak = float(np.max(gyro_peaks))
    if peak > GYRO_RADS_MAX:
        raise ValueError(f"|gyro| p99.9 max = {peak:.1f} exceeds {GYRO_RADS_MAX} rad/s; the source "
                         "is probably deg/s now.")

    ordered = sorted(activities)
    (HERE / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True))
    (HERE / "recordings.json").write_text(json.dumps(
        {session: recording_id(session) for session in sorted(labels)}, indent=2, sort_keys=True))
    (HERE / "manifest.json").write_text(
        json.dumps(create_manifest(len(workouts), ordered), indent=2))
    (HERE / "metadata.json").write_text(json.dumps({
        "dataset": "mmfit", "sampling_rate_hz": NATIVE_RATE, "pre_windowed": False,
    }, indent=2))
    (HERE / "eval_labels.json").write_text(json.dumps({"labels": ordered}, indent=2))

    hours = n_rows / NATIVE_RATE / 3600.0
    print(f"[mmfit] units verified: median |acc| {reference_mag:.3f} m/s^2 (gravity present), "
          f"|gyro| p99.9 max {peak:.2f} rad/s; worst device clock skew "
          f"{max(skews):.2f}s", flush=True)
    print(f"[mmfit] acquisition gaps > {MAX_GAP_SEC:g}s: {len(gap_seconds)} across all devices, "
          f"largest {max(gap_seconds) if gap_seconds else 0.0:.2f}s, "
          f"{fabricated_samples / NATIVE_RATE:.1f}s of grid marked unobserved; "
          f"{dropped_gap_sets} labelled sets dropped for overlapping one "
          f"({dropped_gap_seconds:.1f}s of that was fabricated)", flush=True)
    print(f"[mmfit] {len(labels)} set sessions from {len(workouts)} workouts, {n_rows:,} samples "
          f"per placement, {hours:.2f} h, {len(ordered)} exercises; {short_sets} sets shorter than "
          f"{MIN_SET_SECONDS:g}s will yield no grid rows -> {sessions_dir}", flush=True)


if __name__ == "__main__":
    main()
