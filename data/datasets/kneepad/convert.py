"""Convert KneE-PAD (Zenodo 12112951) to the HALO session format.

Source: `dataset/Subject_NN/<label 0-8>/Trial_N/imu.npy`, 2,086 trials from 31 volunteers with a
diagnosed knee pathology (meniscus, osteoarthritis, cruciate-ligament and tendon injuries).

Layout, from the release's own `metadata.txt`: eight Delsys Trigno Avanti sensors on leg muscle
bellies, IMU at **148.148148 Hz**, accelerometer in **g**, gyroscope in **deg/s** (converted to rad/s
here). `imu.npy` is (48, T) = 8 sensors x 6 channels, sensors in the metadata's Table 3 order and
channels as acc xyz then gyro xyz — confirmed at convert time by the per-row magnitudes (accel rows
sit near 1 g, gyro rows two orders higher). The paired `emg.npy` is not read.

Two honest limitations, documented rather than engineered around
(docs/data/APPLICATION_DATASETS.md):

  * **Placement.** Muscle-belly electrodes on the thigh and calf are not a phone-pocket or
    watch-wrist deployment, so every stream here is `role="stress"` and is never mixed into the
    primary score. Its value is that the labels are real *pathological* execution errors.
  * **Duration.** Measured across all 2,086 trials: median 3.8 s, and only **4.7% reach 6 s**. Most
    trials therefore contribute no rows to a 6-second grid. Trials are converted anyway — the
    session store is lossless and the window length is not a property of the data — but nobody
    should expect a large grid from this source.

Labels are the release's own correct/incorrect execution descriptions (metadata.txt Table 3), which
is exactly the distinction a rehabilitation-monitoring claim needs and which almost nothing else in
the corpus carries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "downloads" / "x" / "dataset"
NATIVE_RATE = 148.14814814814815
DEG_TO_RAD = np.pi / 180.0

# metadata.txt Table 3 (sensor id -> muscle), in imu.npy row-block order.
SENSORS = (
    "right_rectus_femoris",
    "right_hamstrings",
    "right_tibialis_anterior",
    "right_gastrocnemius",
    "left_rectus_femoris",
    "left_hamstrings",
    "left_tibialis_anterior",
    "left_gastrocnemius",
)

# metadata.txt Table 3 (labels), verbatim wording snake_cased.
ACTIVITIES = {
    0: "squat",
    1: "squat_with_weight_transfer_onto_the_healthy_leg",
    2: "squat_with_the_injured_leg_placed_in_front",
    3: "seated_leg_extension",
    4: "seated_leg_extension_without_full_range_of_motion",
    5: "seated_leg_extension_with_the_limb_lifted_from_the_chair",
    6: "walking",
    7: "walking_with_the_injured_knee_not_fully_extended",
    8: "walking_with_the_injured_knee_fully_extended_and_the_hip_abducted",
}
CORRECT_EXECUTION = {0, 3, 6}

ACC_G_RANGE = (0.7, 1.4)         # dataset median |acc| per sensor
GYRO_DEGS_MIN = 20.0             # p99.9 |w| BEFORE conversion; rad/s would never reach this
WINDOW_SECONDS = 6.0


def create_manifest(usable: int, total: int) -> dict:
    return {
        "dataset_name": "KneE-PAD",
        "description": (
            "Knee Exercise Performance Assessment Dataset. 31 volunteers with a diagnosed knee "
            "pathology performed three rehabilitation exercises (squat, seated leg extension, "
            "walking), each in one correct and two clinically-defined incorrect variants, wearing "
            "8 Delsys Trigno Avanti sensors on thigh and calf muscle bellies. IMU at 148.15 Hz, "
            "accelerometer in g, gyroscope in deg/s. Source: Zenodo record 12112951."
        ),
        "sampling_rate_hz": NATIVE_RATE,
        "channels": [
            {"name": f"{sensor}_{kind}_{axis}",
             "description": (f"{'accelerometer' if kind == 'acc' else 'gyroscope'} {axis}-axis, "
                             f"Delsys Trigno Avanti over the {sensor.replace('_', ' ')}; "
                             f"{'g (gravity present)' if kind == 'acc' else 'rad/s'}"),
             "sampling_rate_hz": NATIVE_RATE}
            for sensor in SENSORS for kind in ("acc", "gyro") for axis in "xyz"
        ],
        "subjects": 31,
        "activities": sorted(ACTIVITIES.values()),
        "placements": list(SENSORS),
        "device_profile": "non_deployment",
        "gravity_state": "present",
        "license": "CC-BY (Zenodo record 12112951)",
        "citation": "KneE-PAD, Zenodo, doi:10.5281/zenodo.12112951",
        "unit_note": "Source gyroscope is deg/s and is converted to rad/s in this converter.",
        "duration_note": (
            f"{usable} of {total} trials reach the {WINDOW_SECONDS:g}s analysis window "
            f"({100.0 * usable / total:.1f}%); the median trial is ~3.8 s. This source is not "
            f"expected to produce a large grid."
        ),
        "placement_note": (
            "Muscle-belly sensors on the thigh and calf are outside the phone/watch deployment "
            "envelope, so every stream is role='stress' and never enters the primary score."
        ),
    }


def main() -> None:
    if not RAW.is_dir():
        raise SystemExit(f"raw dataset tree not found at {RAW}")
    sessions_dir = HERE / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, list[str]] = {}
    magnitudes: list[float] = []
    gyro_peaks_deg: list[float] = []
    usable = 0
    total = 0
    empty = 0
    n_rows = 0

    for subject_dir in sorted(RAW.glob("Subject_*"), key=lambda p: int(p.name.split("_")[1])):
        subject = f"s{int(subject_dir.name.split('_')[1]):02d}"
        for label_dir in sorted(subject_dir.glob("[0-8]")):
            code = int(label_dir.name)
            if code not in ACTIVITIES:
                raise ValueError(f"{label_dir}: unknown label directory")
            activity = ACTIVITIES[code]
            for trial_dir in sorted(label_dir.glob("Trial_*"),
                                    key=lambda p: int(p.name.split("_")[1])):
                path = trial_dir / "imu.npy"
                if not path.exists():
                    continue
                table = np.load(path)
                if table.shape[0] != 48:
                    raise ValueError(f"{path}: expected 48 rows (8 sensors x 6), got {table.shape}")
                if not np.isfinite(table).all():
                    raise ValueError(f"{path}: non-finite samples")
                total += 1
                if table.shape[1] < 2:
                    empty += 1        # the release contains a few zero-length trials
                    continue
                if table.shape[1] >= WINDOW_SECONDS * NATIVE_RATE:
                    usable += 1

                trial = int(trial_dir.name.split("_")[1])
                session_id = f"{subject}_{activity}_t{trial:02d}"
                frame = pd.DataFrame({
                    "timestamp_sec": np.arange(table.shape[1], dtype=np.float64) / NATIVE_RATE})
                for index, sensor in enumerate(SENSORS):
                    base = 6 * index
                    block = table[base:base + 6].T                  # (T, 6)
                    magnitudes.append(float(np.median(np.linalg.norm(block[:, :3], axis=1))))
                    gyro_peaks_deg.append(float(np.percentile(np.abs(block[:, 3:]), 99.9)))
                    for offset, name in enumerate(("acc_x", "acc_y", "acc_z")):
                        frame[f"{sensor}_{name}"] = block[:, offset].astype(np.float32)
                    for offset, name in enumerate(("gyro_x", "gyro_y", "gyro_z")):
                        frame[f"{sensor}_{name}"] = (block[:, 3 + offset] * DEG_TO_RAD
                                                     ).astype(np.float32)
                frame["activity"] = activity
                frame["subject"] = subject
                target = sessions_dir / session_id
                target.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(target / "data.parquet", index=False)
                labels[session_id] = [activity]
                n_rows += table.shape[1]
        print(f"  {subject_dir.name}: done", flush=True)

    reference = float(np.median(magnitudes))
    if not ACC_G_RANGE[0] <= reference <= ACC_G_RANGE[1]:
        raise ValueError(f"median |acc| = {reference:.3f}, outside the g range {ACC_G_RANGE}; "
                         "metadata.txt documents G units.")
    peak_deg = float(np.max(gyro_peaks_deg))
    if peak_deg < GYRO_DEGS_MIN:
        raise ValueError(
            f"raw |gyro| p99.9 max = {peak_deg:.1f} never exceeds {GYRO_DEGS_MIN}; the source looks "
            "like rad/s now, so the deg/s -> rad/s conversion here would shrink it 57x.")

    (HERE / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True))
    (HERE / "manifest.json").write_text(json.dumps(create_manifest(usable, total), indent=2))
    (HERE / "metadata.json").write_text(json.dumps({
        "dataset": "kneepad", "sampling_rate_hz": NATIVE_RATE, "pre_windowed": False,
    }, indent=2))
    (HERE / "eval_labels.json").write_text(json.dumps(
        {"labels": sorted(ACTIVITIES.values())}, indent=2))

    hours = n_rows / NATIVE_RATE / 3600.0
    print(f"[kneepad] units verified: median |acc| {reference:.3f} g (gravity present), raw |gyro| "
          f"p99.9 max {peak_deg:.1f} deg/s -> {peak_deg * DEG_TO_RAD:.2f} rad/s", flush=True)
    print(f"[kneepad] {len(labels)} trial sessions, {n_rows:,} samples, {hours:.2f} h; only "
          f"{usable}/{total} trials ({100.0 * usable / total:.1f}%) reach the "
          f"{WINDOW_SECONDS:g}s window; {empty} zero-length trials skipped -> {sessions_dir}",
          flush=True)


if __name__ == "__main__":
    main()
