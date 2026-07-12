"""
Convert Shoaib dataset to standardized format.

Input: data/raw/shoaib/DataSet/
Output: data/shoaib/
  - manifest.json
  - labels.json
  - sessions/session_XXX/data.parquet

Shoaib 2014 Fusion Dataset Info:
- 10 subjects (Participant_1..10)
- 7 activities: biking, sitting, standing, walking, upstairs, downstairs, jogging
- 5 body positions: left_pocket, right_pocket, wrist, upper_arm, belt
- 4 sensors per position: accelerometer(A), linear_acc(L), gyroscope(G), magnetometer(M)
  (we use acc + gyro + mag = 9 channels per position, skip linear acceleration)
- 50 Hz sampling rate
- CSV format: 2-row header, 70 columns (5 x 14 cols per position), activity in last col
- Used as ZERO-SHOT TEST set (not for training)

Per-position column layout (14 cols each):
  timestamp, Ax, Ay, Az, Lx, Ly, Lz, Gx, Gy, Gz, Mx, My, Mz, (blank separator)
Position order in file: left_pocket, right_pocket, wrist, upper_arm, belt
Last column (69) = activity label string

Reference:
Shoaib et al., "Fusion of Smartphone Motion Sensors for Physical Activity Recognition"
Sensors 2014, 14(6), 10146-10176. DOI: 10.3390/s140610146
https://www.utwente.nl/en/eemcs/ps/research/dataset/
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add datascripts to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.scripts.assembly.windowing import create_variable_windows


# Activity mapping (raw label strings to standardized names)
ACTIVITIES = {
    "biking": "cycling",
    "sitting": "sitting",
    "standing": "standing",
    "walking": "walking",
    "upstairs": "walking_upstairs",
    "downstairs": "walking_downstairs",
    "jogging": "jogging",
}

# Body positions in file order (5 positions, each 14 columns)
BODY_POSITIONS = ["left_pocket", "right_pocket", "wrist", "upper_arm", "belt"]

# Sensors we extract per position (acc, gyro, mag — skip linear acceleration)
SENSORS = ["acc", "gyro", "mag"]
AXES = ["x", "y", "z"]
CHANNELS_PER_POSITION = len(SENSORS) * len(AXES)  # 9

# Column layout per position (14 columns each):
# 0: timestamp, 1-3: acc, 4-6: linear_acc (skip), 7-9: gyro, 10-12: mag, 13: blank
# We extract columns 1-3 (acc), 7-9 (gyro), 10-12 (mag) from each position block
EXTRACT_OFFSETS = [1, 2, 3, 7, 8, 9, 10, 11, 12]  # 9 channels per position
COLS_PER_POSITION = 14

# Paths
RAW_DIR = Path("data/raw/shoaib")
OUTPUT_DIR = Path("data/shoaib")

# Sampling rate
SAMPLE_RATE = 50.0  # Hz


def get_column_names():
    """Generate standardized column names for all positions."""
    columns = []
    for pos in BODY_POSITIONS:
        for sensor in SENSORS:
            for axis in AXES:
                columns.append(f"{pos}_{sensor}_{axis}")
    return columns


def load_participant_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a single participant CSV file.

    Format: 2-row header (position names, sensor names), then data rows.
    70 columns: 5 positions x 14 columns each.
    Last column (69) contains the activity label string.

    Returns:
        DataFrame with standardized sensor columns + 'activity' column.
    """
    # Read raw data, skip the 2-row header
    df = pd.read_csv(filepath, header=None, skiprows=2)

    if df.shape[1] < 70:
        print(f"    Warning: expected 70 columns, got {df.shape[1]}")
        return pd.DataFrame()

    # Extract activity labels from last column
    activity_col = df.iloc[:, 69].str.strip()

    # Extract sensor channels from each position block
    col_names = get_column_names()
    extracted = {}

    for pos_idx, pos_name in enumerate(BODY_POSITIONS):
        block_start = pos_idx * COLS_PER_POSITION
        for sensor_idx, offset in enumerate(EXTRACT_OFFSETS):
            col_idx = block_start + offset
            col_name = col_names[pos_idx * len(EXTRACT_OFFSETS) + sensor_idx]
            extracted[col_name] = df.iloc[:, col_idx].values

    result = pd.DataFrame(extracted)

    # Add timestamp and activity
    result.insert(0, 'timestamp_sec', np.arange(len(result)) / SAMPLE_RATE)
    result['activity'] = activity_col.values

    # Handle NaN values in sensor data
    sensor_cols = col_names
    for col in sensor_cols:
        if result[col].isna().any():
            result[col] = result[col].interpolate(method='linear', limit_direction='both')
            result[col] = result[col].fillna(0)

    return result


def convert_participant(filepath: Path, participant_num: int):
    """Convert one participant's data to sessions with variable-length windowing."""
    print(f"  Processing Participant {participant_num}...")

    df = load_participant_csv(filepath)
    if df.empty:
        print(f"    No data found")
        return [], {}

    col_names = get_column_names()
    labels_dict = {}
    session_data = []
    total_windows = 0

    # Segment by activity
    for raw_activity, group in df.groupby('activity'):
        std_activity = ACTIVITIES.get(raw_activity)
        if std_activity is None:
            print(f"    Skipping unknown activity: {raw_activity}")
            continue

        group = group.reset_index(drop=True)

        # Prepare data: keep timestamp + sensor columns only
        data = group[['timestamp_sec'] + col_names].copy()
        data['timestamp_sec'] = np.arange(len(data)) / SAMPLE_RATE

        # Create base session ID
        base_session_id = f"shoaib_P{participant_num:02d}_{std_activity}"

        # Apply variable-length windowing
        windows = create_variable_windows(
            df=data,
            session_prefix=base_session_id,
            activity=std_activity,
            sample_rate=SAMPLE_RATE,
            seed=42 + participant_num * 100,
        )

        # Save each window
        for window_id, window_df, window_activity in windows:
            session_dir = OUTPUT_DIR / "sessions" / window_id
            session_dir.mkdir(parents=True, exist_ok=True)

            window_df.to_parquet(session_dir / "data.parquet", index=False)

            labels_dict[window_id] = [window_activity]
            session_data.append(window_id)
            total_windows += 1

    print(f"    Created {total_windows} windows from {len(df)} samples")
    return session_data, labels_dict


def create_manifest():
    """Create manifest.json."""
    channels = []

    pos_descriptions = {
        "left_pocket": "left trouser pocket",
        "right_pocket": "right trouser pocket",
        "wrist": "right wrist",
        "upper_arm": "right upper arm",
        "belt": "belt-mounted on right leg",
    }

    for pos in BODY_POSITIONS:
        pos_desc = pos_descriptions.get(pos, pos)

        for sensor in SENSORS:
            sensor_full = {"acc": "Accelerometer", "gyro": "Gyroscope", "mag": "Magnetometer"}[sensor]
            for axis in AXES:
                channels.append({
                    "name": f"{pos}_{sensor}_{axis}",
                    "description": f"{sensor_full} {axis.upper()}-axis from {pos_desc} smartphone sensor",
                    "sampling_rate_hz": SAMPLE_RATE
                })

    manifest = {
        "dataset_name": "Shoaib",
        "description": (
            "Activity recognition with multiple body-worn smartphones. "
            "10 subjects performing 7 physical activities. "
            f"Sensors at {len(BODY_POSITIONS)} body positions ({', '.join(BODY_POSITIONS)}) "
            "with accelerometer, gyroscope, and magnetometer at 50 Hz."
        ),
        "source": "https://www.utwente.nl/en/eemcs/ps/research/dataset/",
        "num_subjects": 10,
        "channels": channels
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    n_ch = len(BODY_POSITIONS) * CHANNELS_PER_POSITION
    print(f"Created manifest: {manifest_path}")
    print(f"  {n_ch} channels ({len(BODY_POSITIONS)} positions x {CHANNELS_PER_POSITION} sensors)")


def main():
    """Convert Shoaib dataset to standardized format."""
    print("=" * 80)
    print("Shoaib -> Standardized Format Converter")
    print("=" * 80)
    print("NOTE: This dataset is used for ZERO-SHOT TESTING (not training)")

    # Find participant CSV files
    data_dir = RAW_DIR / "DataSet"
    if not data_dir.exists():
        data_dir = RAW_DIR  # Maybe extracted without subdirectory

    csv_files = sorted(data_dir.glob("Participant_*.csv"))
    if not csv_files:
        print(f"ERROR: No Participant_*.csv files found in {data_dir}")
        print("Download from: https://www.utwente.nl/en/eemcs/ps/research/dataset/")
        print("Extract: sensors-activity-recognition-dataset-shoaib.rar")
        return

    print(f"\nFound {len(csv_files)} participant files")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_labels = {}
    all_sessions = []

    for csv_file in csv_files:
        # Extract participant number from filename
        participant_num = int(csv_file.stem.split("_")[1])

        sessions, labels = convert_participant(csv_file, participant_num)
        all_sessions.extend(sessions)
        all_labels.update(labels)

    if not all_labels:
        print("\nNo sessions created. Check the raw data format.")
        return

    # Save labels.json
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(all_labels, f, indent=2)

    print(f"\nCreated labels: {labels_path}")
    print(f"  Total sessions: {len(all_labels)}")

    # Create manifest
    create_manifest()

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {len(all_labels)} sessions")
    print(f"  - {len(BODY_POSITIONS) * CHANNELS_PER_POSITION} channels")
    print(f"  - {SAMPLE_RATE} Hz sampling rate")

    # Activity distribution
    activity_counts = {}
    for labels in all_labels.values():
        for label in labels:
            activity_counts[label] = activity_counts.get(label, 0) + 1

    print("\nActivity distribution:")
    for activity, count in sorted(activity_counts.items()):
        print(f"  {activity}: {count}")

    # Generate debug visualizations
    try:
        from data.scripts.debug.visualization_utils import generate_debug_visualizations
        generate_debug_visualizations(OUTPUT_DIR)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
