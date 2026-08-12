"""Convert the upper-limb-use assessment dataset (Biorehab, CMC Vellore) to the HALO session format.

Source: github.com/biorehab/upper-limb-use-assessment, four CSVs —
`control/data/{left,right}.csv` (10 unimpaired controls) and
`patient/data/{affected,unaffected}.csv` (5 hemiparetic stroke survivors). Both arms of every
participant wore a wrist band; the two files of a cohort are the two arms of the SAME sessions.

Why this dataset is here: it is one of the few sources with real **stroke patients**, a bilateral
affected/unaffected wrist pair, and 15 functional ADLs annotated from video by therapists. Its
labels are functional-use categories and task identity rather than exercise identity, which is a
different question from the rest of the rehabilitation roster.

Columns (repository README): `time, ax, ay, az, gx, gy, gz, pitch, yaw, mx, my, mz, subject,
old_time, r1, r2, g1, g2, task, use_type, gnd`. Only the six inertial channels are kept. Measured
at convert time: acceleration is in **g with gravity present** (median |a| 1.01-1.07) and angular
rate is in **rad/s** (p99 ~2.3-2.9; deg/s would be ~57x larger). 50 Hz, per the README and the
20 ms median step of the corrected `time` column.

`task` is blank between tasks; those rows are dropped rather than relabelled. Each contiguous run of
one task by one subject on one arm becomes a session, so a repeat of the same task is a separate
execution.

The affected SIDE (left or right) is not recorded in the released CSVs, so the patient streams are
described as the wrist of the more- and less-affected arm rather than as a left/right placement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "downloads" / "repo"
NATIVE_RATE = 50.0

# (relative csv, cohort tag, subject-id prefix, arm token used for stream routing)
SOURCES = (
    ("control/data/left.csv", "control", "c", "left_wrist"),
    ("control/data/right.csv", "control", "c", "right_wrist"),
    ("patient/data/affected.csv", "patient", "p", "affected_wrist"),
    ("patient/data/unaffected.csv", "patient", "p", "unaffected_wrist"),
)

# The 15 annotated ADLs, kept in the repository's own CamelCase wording, snake_cased.
TASKS = {
    "ButtonShirt": "buttoning_a_shirt",
    "CallMobile": "making_a_call_on_a_mobile_phone",
    "CombHair": "combing_hair",
    "DrinkCup": "drinking_from_a_cup",
    "DrinkGlass": "drinking_from_a_glass",
    "EatBowl": "eating_from_a_bowl",
    "EatPlate": "eating_from_a_plate",
    "FoldTowel": "folding_a_towel",
    "OnSwitch": "switching_on_a_light_switch",
    "OpenBottle": "opening_a_bottle",
    "OpenDoor": "opening_a_door",
    "TypeKeyboard": "typing_on_a_keyboard",
    "Walk25": "walking_25_metres",
    "WipeTable": "wiping_a_table",
    "WritePen": "writing_with_a_pen",
}

ACC_G_RANGE = (0.7, 1.4)         # median |acc| per file
GYRO_RADS_MAX = 40.0             # p99.9 |w|; deg/s would blow straight through this
MAX_GAP_SEC = 0.5                # a longer hole in the clock splits the run
WINDOW_SECONDS = 6.0             # reporting only


def create_manifest() -> dict:
    return {
        "dataset_name": "upper-limb-use-assessment",
        "description": (
            "Bilateral custom wrist-worn IMU recordings of 15 activities of daily living performed by "
            "10 unimpaired controls and 5 hemiparetic stroke survivors, with therapist video "
            "annotations of upper-limb functional use (FAABOS). 6-axis IMU on both wrists at 50 Hz. "
            "Source: github.com/biorehab/upper-limb-use-assessment."
        ),
        "sampling_rate_hz": NATIVE_RATE,
        "channels": [
            {"name": f"acc_{a}", "description":
             f"accelerometer {a}-axis in g (gravity present), wrist-worn band",
             "sampling_rate_hz": NATIVE_RATE} for a in "xyz"
        ] + [
            {"name": f"gyro_{a}", "description":
             f"gyroscope {a}-axis in rad/s, wrist-worn band",
             "sampling_rate_hz": NATIVE_RATE} for a in "xyz"
        ],
        "subjects": 15,
        "activities": sorted(TASKS.values()),
        "placements": ["left wrist", "right wrist", "affected wrist", "unaffected wrist"],
        "device_profile": "device",
        "gravity_state": "present",
        "license": "see the source repository",
        "citation": ("David A et al. Quantification of the relative arm use in patients with "
                     "hemiparesis using inertial measurement units. Journal of Rehabilitation and "
                     "Assistive Technologies Engineering 8, 2021. doi:10.1177/20556683211019694"),
        "cohort_note": (
            "Control subject ids are prefixed c, patient ids p; the two cohorts number their "
            "subjects from 1 independently and would otherwise collide."
        ),
        "side_note": (
            "The released CSVs do not record which side is affected for each patient, so the "
            "patient streams are described by impairment (more-/less-affected arm) rather than by "
            "anatomical side."
        ),
    }


def recording_id(session_id: str) -> str:
    """The continuous capture a session was cut out of: one subject's visit, i.e. `control_c01`.

    LOAD-BEARING, and the reason this is not just tidy metadata. Session ids are
    `{cohort}_{subject}_{arm}_{activity}_{block}`, and the two wrist bands are worn
    SIMULTANEOUSLY — `control_c01_left_wrist_combing_hair_00` and
    `control_c01_right_wrist_combing_hair_00` are the same twenty seconds of the same person.
    Because the arm is part of the session id, an evaluator keying on the session would treat them
    as two independent executions and happily enrol the left band while querying the right one at
    the same instant, reading as change-of-configuration transfer. Every other simultaneous-placement
    source in the corpus keeps the placement OUT of the event id for exactly this reason. Grouping
    both arms and every task block onto the subject's visit restores that guarantee.
    See docs/data/DATASET_EXPANSION_AUDIT_2026-08-11.md section 2.2.
    """
    return "_".join(session_id.split("_")[:2])


def _runs(values: np.ndarray):
    """Yield (value, start, stop) for each maximal run."""
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(values)]):
        yield values[start], int(start), int(stop)


def main() -> None:
    sessions_dir = HERE / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, list[str]] = {}
    magnitudes: list[float] = []
    gyro_peaks: list[float] = []
    steps: list[float] = []
    short_blocks = 0
    n_rows = 0
    subjects_seen: set[str] = set()

    for relative, cohort, prefix, arm in SOURCES:
        path = RAW / relative
        if not path.exists():
            raise SystemExit(f"missing {path}")
        table = pd.read_csv(path)
        clock = pd.to_datetime(table["time"]).astype("int64").to_numpy() / 1e9
        acc = table[["ax", "ay", "az"]].to_numpy(dtype=np.float64)
        gyro = table[["gx", "gy", "gz"]].to_numpy(dtype=np.float64)
        if not (np.isfinite(acc).all() and np.isfinite(gyro).all()):
            raise ValueError(f"{relative}: non-finite inertial samples")
        magnitudes.append(float(np.median(np.linalg.norm(acc, axis=1))))
        gyro_peaks.append(float(np.percentile(np.abs(gyro), 99.9)))

        raw_task = table["task"].astype(str).str.strip().to_numpy()
        unknown = set(raw_task) - set(TASKS) - {""}
        if unknown:
            raise ValueError(f"{relative}: unknown task names {sorted(unknown)}")

        # Session = one contiguous (subject, task) run. `subject` changes between blocks of rows,
        # so it is part of the run key; a clock hole inside a run splits it further.
        key = np.array([f"{s}|{t}" for s, t in zip(table["subject"].to_numpy(), raw_task)])
        seen: dict[str, int] = {}
        for value, start, stop in _runs(key):
            subject_token, task_token = value.split("|", 1)
            if not task_token:
                continue                       # unlabelled gap between tasks
            subject = f"{prefix}{int(subject_token):02d}"
            subjects_seen.add(subject)
            activity = TASKS[task_token]
            piece_clock = clock[start:stop]
            gaps = np.flatnonzero(np.diff(piece_clock) > MAX_GAP_SEC) + 1
            steps.extend(np.diff(piece_clock)[np.diff(piece_clock) <= MAX_GAP_SEC].tolist())
            for cut_start, cut_stop in zip(np.r_[0, gaps], np.r_[gaps, stop - start]):
                if cut_stop - cut_start < 2:
                    continue
                block = seen.get(f"{subject}|{activity}", 0)
                seen[f"{subject}|{activity}"] = block + 1
                if (cut_stop - cut_start) < WINDOW_SECONDS * NATIVE_RATE:
                    short_blocks += 1
                session_id = f"{cohort}_{subject}_{arm}_{activity}_{block:02d}"
                lo, hi = start + cut_start, start + cut_stop
                frame = pd.DataFrame({
                    "timestamp_sec": np.arange(hi - lo, dtype=np.float64) / NATIVE_RATE,
                    **{f"acc_{a}": acc[lo:hi, i].astype(np.float32) for i, a in enumerate("xyz")},
                    **{f"gyro_{a}": gyro[lo:hi, i].astype(np.float32) for i, a in enumerate("xyz")},
                })
                frame["activity"] = activity
                frame["subject"] = subject
                target = sessions_dir / session_id
                target.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(target / "data.parquet", index=False)
                labels[session_id] = [activity]
                n_rows += hi - lo
        print(f"  {relative}: {len(seen)} (subject, task) pairs", flush=True)

    reference = float(np.median(magnitudes))
    if not ACC_G_RANGE[0] <= reference <= ACC_G_RANGE[1]:
        raise ValueError(f"median |acc| = {reference:.3f}, outside the g range {ACC_G_RANGE}; the "
                         "source units changed (m/s^2 would be ~9.8, milli-g ~1000).")
    peak = float(np.max(gyro_peaks))
    if peak > GYRO_RADS_MAX:
        raise ValueError(f"|gyro| p99.9 max = {peak:.1f} exceeds {GYRO_RADS_MAX} rad/s; the source "
                         "is probably deg/s now.")
    measured_rate = 1.0 / float(np.median(steps))
    if abs(measured_rate - NATIVE_RATE) > 1.0:
        raise ValueError(f"median clock rate {measured_rate:.3f} Hz != documented {NATIVE_RATE} Hz")

    (HERE / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True))
    (HERE / "recordings.json").write_text(json.dumps(
        {session: recording_id(session) for session in sorted(labels)}, indent=2, sort_keys=True))
    (HERE / "manifest.json").write_text(json.dumps(create_manifest(), indent=2))
    (HERE / "metadata.json").write_text(json.dumps({
        "dataset": "upper_limb_use", "sampling_rate_hz": NATIVE_RATE, "pre_windowed": False,
    }, indent=2))
    (HERE / "eval_labels.json").write_text(json.dumps(
        {"labels": sorted(TASKS.values())}, indent=2))

    hours = n_rows / NATIVE_RATE / 3600.0
    print(f"[upper_limb_use] units verified: median |acc| {reference:.3f} g (gravity present), "
          f"|gyro| p99.9 max {peak:.2f} rad/s, clock {measured_rate:.3f} Hz", flush=True)
    print(f"[upper_limb_use] {len(labels)} sessions, {len(subjects_seen)} subjects, {n_rows:,} "
          f"samples, {hours:.2f} h, {len(TASKS)} tasks; {short_blocks} runs shorter than "
          f"{WINDOW_SECONDS:g}s will yield no grid rows -> {sessions_dir}", flush=True)


if __name__ == "__main__":
    main()
