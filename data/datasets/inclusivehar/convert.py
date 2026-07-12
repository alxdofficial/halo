"""
Convert InclusiveHAR dataset to standardized format.

Input:  data/datasets/inclusivehar/downloads/InclusiveHAR_dataset_v2*.csv
Output: data/datasets/inclusivehar/
  - manifest.json
  - labels.json
  - sessions/<id>/data.parquet   (RAW whole-recording sessions; build_grids does the 6 s windowing)

InclusiveHAR Dataset Info:
- 20 subjects: UserID 1-10 are able-bodied (disabled=0),
  UserID 11-20 have physical disabilities (disabled=1). This built-in
  ability split is the point of the dataset — it lets us report HAR
  performance stratified by physical ability (a real-world robustness axis).
- 6 activities: walking, jogging, sitting, standing, ramp ascent, ramp descent
- iPhone worn vertically in a WAIST POUCH (iOS CoreMotion via SensorLog)
- 50 Hz sampling rate (SensorLog was locked to 50 Hz)
- Single wide CSV; no timestamp column (fixed-rate stream), no trial column.

Sensor family is identical to MotionSense (iOS CoreMotion DeviceMotion), so we
map onto the exact same standardized channel schema, and `deployment_policy`'s
`phone_waist` stream reconstructs TOTAL acceleration = userAcceleration + gravity
(both in g), exactly like MotionSense:
  acc_{x,y,z}      <- motionUserAcceleration{X,Y,Z}  (g, gravity removed by iOS)
  gravity_{x,y,z}  <- motionGravity{X,Y,Z}            (unit vector, g)
  gyro_{x,y,z}     <- motionRotationRate{X,Y,Z}       (rad/s)
  attitude_{roll,pitch,yaw} <- motion{Roll,Pitch,Yaw} (rad, QA-only)

Sessions are cut by (UserID, contiguous run of the same activity): each
continuous same-activity recording for one user is saved whole as ONE session.

Used as a ZERO-SHOT TEST set (phone HAR + inclusivity robustness slice).

Reference:
InclusiveHAR: A Smartphone-Based Dataset for Human Activity Recognition Across
Diverse Physical Abilities. Mendeley Data. doi:10.17632/r78dn3f6nc
https://data.mendeley.com/datasets/r78dn3f6nc
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Run as `python -m data.datasets.inclusivehar.convert` from the repo root; repo root on path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Paths (new repo layout: this converter lives in data/datasets/inclusivehar/)
DS_DIR = Path(__file__).resolve().parent
RAW_DIR = DS_DIR / "downloads"
OUTPUT_DIR = DS_DIR

SAMPLE_RATE = 50.0  # Hz
MIN_SESSION_SAMPLES = int(SAMPLE_RATE * 1.0)  # drop runs shorter than 1 s

# Raw label string -> standardized activity name (underscore convention).
# Labels arrive mixed-case ("Ramp descent", "jogging", ...); we lower+strip first.
LABEL_MAP = {
    "walking": "walking",
    "jogging": "jogging",
    "sitting": "sitting",
    "standing": "standing",
    "ramp ascent": "ramp_ascent",
    "ramp descent": "ramp_descent",
}

# Source column -> standardized channel name (iOS CoreMotion, matches MotionSense).
CHANNEL_MAP = {
    "motionUserAccelerationX": "acc_x",
    "motionUserAccelerationY": "acc_y",
    "motionUserAccelerationZ": "acc_z",
    "motionRotationRateX": "gyro_x",
    "motionRotationRateY": "gyro_y",
    "motionRotationRateZ": "gyro_z",
    "motionGravityX": "gravity_x",
    "motionGravityY": "gravity_y",
    "motionGravityZ": "gravity_z",
    "motionRoll": "attitude_roll",
    "motionPitch": "attitude_pitch",
    "motionYaw": "attitude_yaw",
}

OUTPUT_COLUMNS = list(CHANNEL_MAP.values())


def find_raw_csv() -> Path | None:
    """Locate the single wide InclusiveHAR CSV inside downloads/ (odd filename with a space)."""
    if not RAW_DIR.exists():
        return None
    candidates = sorted(RAW_DIR.glob("*.csv"))
    return candidates[0] if candidates else None


def convert_dataset() -> bool:
    print("=" * 80)
    print("InclusiveHAR -> Standardized Format Converter")
    print("=" * 80)
    print("NOTE: ZERO-SHOT TEST set (phone HAR + physical-ability robustness)")

    raw_csv = find_raw_csv()
    if raw_csv is None:
        print(f"ERROR: no CSV found in {RAW_DIR}")
        print("Download InclusiveHAR_dataset_v2.csv from Mendeley r78dn3f6nc into downloads/.")
        return False

    sessions_dir = OUTPUT_DIR / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)  # drop stale sessions from any prior run
    sessions_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nReading {raw_csv} ...")
    usecols = list(CHANNEL_MAP) + ["label", "UserID", "disabled"]
    df = pd.read_csv(raw_csv, usecols=usecols)
    missing = [c for c in CHANNEL_MAP if c not in df.columns]
    if missing:
        print(f"ERROR: expected columns missing: {missing}")
        return False

    # Standardize channel names, keep label + subject + ability flag.
    df = df.rename(columns=CHANNEL_MAP)
    df["activity_std"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    unmapped = df["activity_std"].isna()
    if unmapped.any():
        print(f"  WARNING: {int(unmapped.sum())} rows had unmapped labels "
              f"({sorted(df.loc[unmapped, 'label'].unique())}); dropped.")
        df = df[~unmapped].reset_index(drop=True)

    all_labels = {}
    session_count = 0
    ability_by_subject = {}

    for uid in sorted(df["UserID"].unique()):
        sub = df[df["UserID"] == uid].reset_index(drop=True)
        ability_by_subject[int(uid)] = int(sub["disabled"].iloc[0])
        subject = f"user{int(uid)}"

        # Segment into contiguous runs of the same activity (rows are ordered as
        # consecutive per-activity recordings within each subject).
        acts = sub["activity_std"].values
        seg_starts = [0] + [i for i in range(1, len(acts)) if acts[i] != acts[i - 1]]
        seg_starts.append(len(acts))

        subj_sessions = 0
        for seg_idx in range(len(seg_starts) - 1):
            s, e = seg_starts[seg_idx], seg_starts[seg_idx + 1]
            activity = acts[s]
            if e - s < MIN_SESSION_SAMPLES:
                continue

            seg = sub.iloc[s:e][OUTPUT_COLUMNS].copy().reset_index(drop=True)
            seg.insert(0, "timestamp_sec", np.arange(len(seg)) / SAMPLE_RATE)
            # Subject for subject-disjoint splits; read by build_grids.iter_sessions.
            seg["subject"] = subject
            # Ability flag (0/1) for the ability-stratified slice; ignored by curate_frame.
            seg["disabled"] = ability_by_subject[int(uid)]

            session_id = f"inclusivehar_user{int(uid):02d}_{activity}_{seg_idx:03d}"
            sp = sessions_dir / session_id
            sp.mkdir(exist_ok=True)
            seg.to_parquet(sp / "data.parquet", index=False)
            all_labels[session_id] = [activity]
            session_count += 1
            subj_sessions += 1
        print(f"  UserID {int(uid):2d} (disabled={ability_by_subject[int(uid)]}): "
              f"{subj_sessions} sessions")

    if not all_labels:
        print("\nNo sessions created — check the raw CSV format.")
        return False

    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump(all_labels, f, indent=2)
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(create_manifest(ability_by_subject), f, indent=2)

    counts = {}
    for v in all_labels.values():
        counts[v[0]] = counts.get(v[0], 0) + 1
    print(f"\n{'=' * 80}\nConversion complete!\n{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {session_count} sessions")
    print(f"  - {len(counts)} activities: {sorted(counts)}")
    print(f"  - {SAMPLE_RATE} Hz")
    print("\nActivity distribution:")
    for a, c in sorted(counts.items()):
        print(f"  {a:16s} {c}")
    return True


def create_manifest(ability_by_subject: dict) -> dict:
    return {
        "dataset_name": "InclusiveHAR",
        "description": (
            "Smartphone (iOS CoreMotion) human activity recognition across diverse "
            "physical abilities. 20 subjects (10 able-bodied, 10 with physical "
            "disabilities) performing 6 activities (walking, jogging, sitting, "
            "standing, ramp ascent, ramp descent) at 50 Hz via the SensorLog app, "
            "with the phone worn vertically in a waist pouch."
        ),
        "source": "https://data.mendeley.com/datasets/r78dn3f6nc",
        "num_subjects": 20,
        "ability_by_subject": ability_by_subject,  # subject id -> disabled flag (0/1)
        "channels": [
            {"name": "acc_x", "description": "User acceleration X (g, gravity removed by iOS)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_y", "description": "User acceleration Y (g, gravity removed by iOS)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_z", "description": "User acceleration Z (g, gravity removed by iOS)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_x", "description": "Rotation rate X (rad/s)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_y", "description": "Rotation rate Y (rad/s)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_z", "description": "Rotation rate Z (rad/s)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gravity_x", "description": "Gravity unit vector X (g)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gravity_y", "description": "Gravity unit vector Y (g)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gravity_z", "description": "Gravity unit vector Z (g)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "attitude_roll", "description": "Device attitude roll (rad)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "attitude_pitch", "description": "Device attitude pitch (rad)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "attitude_yaw", "description": "Device attitude yaw (rad)", "sampling_rate_hz": SAMPLE_RATE},
        ],
    }


def main() -> int:
    return 0 if convert_dataset() else 1


if __name__ == "__main__":
    sys.exit(main())
