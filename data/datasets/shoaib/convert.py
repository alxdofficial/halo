"""
Convert Shoaib dataset to standardized format.

Input:  data/datasets/shoaib/downloads/DataSet/Participant_1..10.csv
Output: data/datasets/shoaib/
  - manifest.json
  - labels.json
  - sessions/<id>/data.parquet   (RAW whole-recording sessions; build_grids does the 6 s windowing)

Shoaib 2014 Fusion Dataset Info:
- 10 subjects (Participant_1..10)
- 7 activities: biking, sitting, standing, walking, upstairs, downstairs, jogging
  (the Readme notes the paper's "running" is actually jogging; we use jogging)
- 5 body positions: left_pocket, right_pocket, wrist, upper_arm, belt
- Per position 4 sensors: accelerometer(A), linear_acc(L), gyroscope(G), magnetometer(M)
- 50 Hz sampling rate
- CSV format: 2-row header, 70 columns (5 x 14 cols per position), activity in last col
- Used as a ZERO-SHOT TEST set (not for training)

Per-position column layout (14 cols each):
  time_stamp, Ax, Ay, Az, Lx, Ly, Lz, Gx, Gy, Gz, Mx, My, Mz, (blank separator)
Position order in file: left_pocket, right_pocket, wrist, upper_arm, belt
Last column (index 69) = activity label string.

deployment_policy needs, per kept position, the ACCELEROMETER (Ax) as `<pos>_acc_*`
and the GYROSCOPE (Gx) as `<pos>_gyro_*`:
  PRIMARY    stream `phone_right_pocket`: right_pocket_acc_/right_pocket_gyro_
  DIAGNOSTIC streams:                     left_pocket_acc_/gyro_, belt_acc_/gyro_, wrist_acc_/gyro_
We use the accelerometer Ax (m/s^2, gravity present) — NOT the linear acceleration Lx.
Magnetometer (M*) and the upper_arm position are dropped (not in the policy).

Each activity is one continuous time block per participant; each block is saved whole
as ONE session.

Reference:
Shoaib et al., "Fusion of Smartphone Motion Sensors for Physical Activity Recognition"
Sensors 2014, 14(6), 10146-10176. DOI: 10.3390/s140610146
https://www.utwente.nl/en/eemcs/ps/research/dataset/
"""

import sys
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Run as `python -m data.datasets.shoaib.convert` from the repo root; repo root on path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Paths (new repo layout: this converter lives in data/datasets/shoaib/)
DS_DIR = Path(__file__).resolve().parent
RAW_DIR = DS_DIR / "downloads"
OUTPUT_DIR = DS_DIR

SAMPLE_RATE = 50.0  # Hz
MIN_SESSION_SAMPLES = int(SAMPLE_RATE * 1.0)  # drop runs shorter than 1 s

# Activity mapping (raw label string -> standardized name). "upsatirs" is a source typo
# for "upstairs" present in Participant_8 — kept so that block is not silently dropped.
ACTIVITIES = {
    "biking": "cycling",
    "sitting": "sitting",
    "standing": "standing",
    "walking": "walking",
    "upstairs": "walking_upstairs",
    "upsatirs": "walking_upstairs",
    "downstairs": "walking_downstairs",
    "jogging": "jogging",
}

# Body positions in FILE order (5 positions, 14 columns each). block_start = idx * 14.
FILE_POSITIONS = ["left_pocket", "right_pocket", "wrist", "upper_arm", "belt"]
# Positions we keep (upper_arm dropped — not in deployment_policy).
KEEP_POSITIONS = ["left_pocket", "right_pocket", "wrist", "belt"]
COLS_PER_POSITION = 14
ACTIVITY_COL = 69  # last column (0-indexed)

# Within a position block, the accelerometer (Ax,Ay,Az) and gyroscope (Gx,Gy,Gz) offsets.
# We deliberately skip linear acceleration (Lx: 4,5,6) and magnetometer (Mx: 10,11,12).
SENSOR_OFFSETS = {"acc": [1, 2, 3], "gyro": [7, 8, 9]}
AXES = ["x", "y", "z"]


def get_column_names():
    """Standardized sensor column names for all kept positions (acc + gyro)."""
    cols = []
    for pos in KEEP_POSITIONS:
        for sensor in SENSOR_OFFSETS:
            for ax in AXES:
                cols.append(f"{pos}_{sensor}_{ax}")
    return cols


def load_participant_csv(filepath: Path) -> pd.DataFrame:
    """
    Load one participant CSV.

    Format: 2-row header (position groups, per-position sensor names), then data rows.
    70 columns: 5 positions x 14 columns each; activity string in the last column.

    Returns a DataFrame with timestamp_sec + kept sensor columns + 'activity'.
    """
    df = pd.read_csv(filepath, header=None, skiprows=2)
    if df.shape[1] < 70:
        print(f"    Warning: expected 70 columns, got {df.shape[1]}")
        return pd.DataFrame()

    activity_col = df.iloc[:, ACTIVITY_COL].astype(str).str.strip()

    extracted = {}
    for file_idx, pos in enumerate(FILE_POSITIONS):
        if pos not in KEEP_POSITIONS:
            continue
        block_start = file_idx * COLS_PER_POSITION
        for sensor, offsets in SENSOR_OFFSETS.items():
            for ax, off in zip(AXES, offsets):
                col_name = f"{pos}_{sensor}_{ax}"
                extracted[col_name] = pd.to_numeric(
                    df.iloc[:, block_start + off], errors="coerce"
                ).values

    result = pd.DataFrame(extracted)
    result.insert(0, "timestamp_sec", np.arange(len(result)) / SAMPLE_RATE)
    result["activity"] = activity_col.values

    # Repair occasional NaN gaps in sensor channels.
    for col in get_column_names():
        if result[col].isna().any():
            result[col] = result[col].interpolate(method="linear", limit_direction="both")
            result[col] = result[col].fillna(0)

    return result


def convert_participant(filepath: Path, participant_num: int):
    """Convert one participant: one session per contiguous same-activity run.

    The activity is the per-row string in the last CSV column; we cut sessions at
    every activity change (each activity is one continuous ~3-4 min block here).
    """
    print(f"  Processing Participant {participant_num}...")

    df = load_participant_csv(filepath)
    if df.empty:
        print("    No data found")
        return {}

    col_names = get_column_names()
    subject = f"participant{participant_num}"
    labels_dict = {}
    n_sessions = 0

    acts = df["activity"].astype(str).str.strip().str.lower().values
    seg_starts = [0] + [i for i in range(1, len(acts)) if acts[i] != acts[i - 1]]
    seg_starts.append(len(acts))

    for seg_idx in range(len(seg_starts) - 1):
        s, e = seg_starts[seg_idx], seg_starts[seg_idx + 1]
        std_activity = ACTIVITIES.get(acts[s])
        if std_activity is None:
            print(f"    Skipping unknown activity: {acts[s]!r}")
            continue
        if e - s < MIN_SESSION_SAMPLES:
            continue

        data = df.iloc[s:e][col_names].copy().reset_index(drop=True)
        data.insert(0, "timestamp_sec", np.arange(len(data)) / SAMPLE_RATE)
        # Subject for subject-disjoint splits; read by build_grids.iter_sessions.
        data["subject"] = subject

        session_id = f"shoaib_participant{participant_num:02d}_{std_activity}_{seg_idx:02d}"
        session_dir = OUTPUT_DIR / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        data.to_parquet(session_dir / "data.parquet", index=False)

        labels_dict[session_id] = [std_activity]
        n_sessions += 1

    print(f"    Created {n_sessions} sessions from {len(df)} samples")
    return labels_dict


def create_manifest():
    """Create manifest.json (acc + gyro for the 4 kept positions)."""
    channels = []
    pos_descriptions = {
        "left_pocket": "left trouser pocket",
        "right_pocket": "right trouser pocket",
        "wrist": "right wrist",
        "belt": "belt-mounted on right leg",
    }
    sensor_full = {"acc": "Accelerometer", "gyro": "Gyroscope"}
    for pos in KEEP_POSITIONS:
        pos_desc = pos_descriptions.get(pos, pos)
        for sensor in SENSOR_OFFSETS:
            for ax in AXES:
                channels.append({
                    "name": f"{pos}_{sensor}_{ax}",
                    "description": f"{sensor_full[sensor]} {ax.upper()}-axis from {pos_desc} smartphone",
                    "sampling_rate_hz": SAMPLE_RATE,
                })

    manifest = {
        "dataset_name": "Shoaib",
        "description": (
            "Activity recognition with multiple body-worn smartphones. "
            "10 subjects performing 7 physical activities. Accelerometer + gyroscope "
            f"at {len(KEEP_POSITIONS)} body positions ({', '.join(KEEP_POSITIONS)}) at 50 Hz. "
            "Accelerometer is total acceleration in m/s^2 (gravity present)."
        ),
        "source": "https://www.utwente.nl/en/eemcs/ps/research/dataset/",
        "num_subjects": 10,
        "channels": channels,
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Created manifest: {len(channels)} channels "
          f"({len(KEEP_POSITIONS)} positions x {len(SENSOR_OFFSETS) * len(AXES)} sensors)")


def main():
    print("=" * 80)
    print("Shoaib -> Standardized Format Converter")
    print("=" * 80)
    print("NOTE: ZERO-SHOT TEST set (not for training)")

    data_dir = RAW_DIR / "DataSet"
    if not data_dir.exists():
        data_dir = RAW_DIR  # maybe extracted without the subdirectory

    csv_files = sorted(data_dir.glob("Participant_*.csv"),
                       key=lambda p: int(p.stem.split("_")[1]))
    if not csv_files:
        print(f"ERROR: No Participant_*.csv files found in {data_dir}")
        return

    print(f"\nFound {len(csv_files)} participant files")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions_root = OUTPUT_DIR / "sessions"
    if sessions_root.exists():
        shutil.rmtree(sessions_root)  # drop stale sessions from any prior run

    all_labels = {}
    for csv_file in csv_files:
        participant_num = int(csv_file.stem.split("_")[1])
        all_labels.update(convert_participant(csv_file, participant_num))

    if not all_labels:
        print("\nNo sessions created. Check the raw data format.")
        return

    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump(all_labels, f, indent=2)
    print(f"\nCreated labels.json ({len(all_labels)} sessions)")

    create_manifest()

    activity_counts = {}
    for labels in all_labels.values():
        for label in labels:
            activity_counts[label] = activity_counts.get(label, 0) + 1

    print(f"\n{'=' * 80}\nConversion complete!\n{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {len(all_labels)} sessions")
    print(f"  - {SAMPLE_RATE} Hz sampling rate")
    print("\nActivity distribution:")
    for activity, count in sorted(activity_counts.items()):
        print(f"  {activity}: {count}")


if __name__ == "__main__":
    main()
