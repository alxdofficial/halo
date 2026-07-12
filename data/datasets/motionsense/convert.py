"""
Process MotionSense dataset into standardized format.

MotionSense Dataset:
- 24 subjects performing 6 activities
- Activities: walking (wlk), jogging (jog), upstairs (ups), downstairs (dws), sitting (sit), standing (std)
- 15 trials total (2-3 trials per activity): dws[1,2,11], ups[3,4,12], wlk[7,8,15], jog[9,16], std[6,14], sit[5,13]
- 50Hz sampling rate (iPhone 6s in front trouser pocket)
- Sensors: DeviceMotion from Core Motion (attitude, gravity, rotation rate, user acceleration)
- Total sessions: 24 subjects x 15 trials = 360 sessions

Source: https://github.com/mmalekzadeh/motion-sense

Data format:
- Raw CSV files in A_DeviceMotion_data/{activity}_{trial}/sub_{subject}.csv
- Columns: attitude.roll/pitch/yaw, gravity.x/y/z, rotationRate.x/y/z, userAcceleration.x/y/z
- userAcceleration is in g units (gravity removed by Core Motion)
- gravity is a normalized unit vector
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from typing import Dict, List, Tuple

# Add datascripts to path for shared imports
sys.path.insert(0, str(Path(__file__).parent))
from data.scripts.assembly.windowing import create_variable_windows, get_window_range


# Activity code to label mapping (using underscores to match other datasets)
ACTIVITY_MAP = {
    'dws': 'walking_downstairs',
    'ups': 'walking_upstairs',
    'wlk': 'walking',
    'jog': 'jogging',
    'sit': 'sitting',
    'std': 'standing'
}

# Column renaming for consistency with other datasets
COLUMN_MAP = {
    'attitude.roll': 'attitude_roll',
    'attitude.pitch': 'attitude_pitch',
    'attitude.yaw': 'attitude_yaw',
    'gravity.x': 'gravity_x',
    'gravity.y': 'gravity_y',
    'gravity.z': 'gravity_z',
    'rotationRate.x': 'gyro_x',  # Rename to match standard naming
    'rotationRate.y': 'gyro_y',
    'rotationRate.z': 'gyro_z',
    'userAcceleration.x': 'acc_x',  # User acceleration (gravity removed)
    'userAcceleration.y': 'acc_y',
    'userAcceleration.z': 'acc_z'
}

SAMPLING_RATE = 50  # Hz


def create_manifest() -> Dict:
    """Create manifest.json content for MotionSense dataset."""
    return {
        "dataset_name": "MotionSense",
        "description": "Human activity recognition from iPhone 6s sensors. 24 subjects performing 6 activities (walking, jogging, stairs up, stairs down, sitting, standing) with smartphone in front trouser pocket. DeviceMotion data from Core Motion framework at 50Hz.",
        "channels": [
            {
                "name": "acc_x",
                "description": "User acceleration X-axis in g units (gravity removed by Core Motion)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "acc_y",
                "description": "User acceleration Y-axis in g units (gravity removed by Core Motion)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "acc_z",
                "description": "User acceleration Z-axis in g units (gravity removed by Core Motion)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "gyro_x",
                "description": "Rotation rate X-axis from gyroscope in rad/s",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "gyro_y",
                "description": "Rotation rate Y-axis from gyroscope in rad/s",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "gyro_z",
                "description": "Rotation rate Z-axis from gyroscope in rad/s",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "gravity_x",
                "description": "Gravity unit vector X component (normalized)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "gravity_y",
                "description": "Gravity unit vector Y component (normalized)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "gravity_z",
                "description": "Gravity unit vector Z component (normalized)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "attitude_roll",
                "description": "Device attitude roll angle in radians",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "attitude_pitch",
                "description": "Device attitude pitch angle in radians",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "attitude_yaw",
                "description": "Device attitude yaw angle in radians",
                "sampling_rate_hz": 50.0
            }
        ]
    }


def process_motionsense(
    raw_dir: Path,
    output_dir: Path,
    min_session_length: int = 200,  # Minimum samples per session
    use_windowing: bool = True  # Apply variable-length windowing
) -> Tuple[int, Dict[str, List[str]]]:
    """
    Process MotionSense raw data into standardized format.

    Args:
        raw_dir: Path to raw MotionSense data (A_DeviceMotion_data folder)
        output_dir: Path to output directory (data/motionsense)
        min_session_length: Minimum number of samples per session
        use_windowing: If True, split long sessions into variable-length windows

    Returns:
        Tuple of (num_sessions, labels_dict)
    """
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir = output_dir / "sessions"

    # Clear existing sessions if reprocessing
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    sessions_dir.mkdir(exist_ok=True)

    labels = {}
    session_count = 0
    window_count = 0
    skipped_short = 0
    skipped_error = 0

    # Get all activity folders
    activity_folders = [f for f in raw_dir.iterdir() if f.is_dir() and not f.name.startswith('.')]

    print(f"Found {len(activity_folders)} activity folders")
    print(f"Windowing enabled: {use_windowing}")

    for activity_folder in sorted(activity_folders):
        folder_name = activity_folder.name

        # Parse activity code and trial number (e.g., "wlk_7" -> "wlk", 7)
        parts = folder_name.rsplit('_', 1)
        if len(parts) != 2:
            print(f"  Skipping invalid folder: {folder_name}")
            continue

        activity_code, trial_str = parts

        if activity_code not in ACTIVITY_MAP:
            print(f"  Skipping unknown activity: {activity_code}")
            continue

        activity_label = ACTIVITY_MAP[activity_code]
        trial_num = int(trial_str)

        # Process each subject's file
        csv_files = list(activity_folder.glob("sub_*.csv"))

        for csv_file in csv_files:
            subject_num = int(csv_file.stem.replace('sub_', ''))

            try:
                # Load CSV
                df = pd.read_csv(csv_file)

                # Skip short sessions
                if len(df) < min_session_length:
                    skipped_short += 1
                    continue

                # Rename columns
                df = df.rename(columns=COLUMN_MAP)

                # Drop the unnamed index column if present
                if 'Unnamed: 0' in df.columns:
                    df = df.drop(columns=['Unnamed: 0'])
                elif df.columns[0] == '' or df.columns[0].isdigit() or 'Unnamed' in str(df.columns[0]):
                    df = df.iloc[:, 1:]  # Drop first column if it looks like an index

                # Add timestamp column
                df['timestamp_sec'] = np.arange(len(df)) / SAMPLING_RATE

                # Reorder columns: timestamp first, then sensor channels
                columns_order = ['timestamp_sec', 'acc_x', 'acc_y', 'acc_z',
                               'gyro_x', 'gyro_y', 'gyro_z',
                               'gravity_x', 'gravity_y', 'gravity_z',
                               'attitude_roll', 'attitude_pitch', 'attitude_yaw']

                # Only include columns that exist
                columns_order = [c for c in columns_order if c in df.columns]
                df = df[columns_order]

                # Create base session ID: sub{subject:02d}_{activity}_{trial:02d}
                base_session_id = f"sub{subject_num:02d}_{activity_code}_{trial_num:02d}"

                if use_windowing:
                    # Apply variable-length windowing
                    windows = create_variable_windows(
                        df=df,
                        session_prefix=base_session_id,
                        activity=activity_label,
                        sample_rate=SAMPLING_RATE,
                    )

                    # Save each window as a separate session
                    for window_id, window_df, window_activity in windows:
                        session_dir = sessions_dir / window_id
                        session_dir.mkdir(exist_ok=True)

                        # Reset timestamp to start from 0
                        window_df = window_df.copy()
                        window_df['timestamp_sec'] = window_df['timestamp_sec'] - window_df['timestamp_sec'].iloc[0]

                        output_path = session_dir / "data.parquet"
                        window_df.to_parquet(output_path, index=False)

                        labels[window_id] = [window_activity]
                        window_count += 1
                else:
                    # Save full session without windowing
                    session_dir = sessions_dir / base_session_id
                    session_dir.mkdir(exist_ok=True)

                    output_path = session_dir / "data.parquet"
                    df.to_parquet(output_path, index=False)

                    labels[base_session_id] = [activity_label]

                session_count += 1

            except Exception as e:
                print(f"  Error processing {csv_file}: {e}")
                skipped_error += 1
                continue

    print(f"\nProcessed {session_count} raw recordings")
    if use_windowing:
        print(f"Created {window_count} windowed sessions")
    print(f"Skipped {skipped_short} recordings (too short)")
    print(f"Skipped {skipped_error} recordings (errors)")

    return window_count if use_windowing else session_count, labels


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw" / "motionsense" / "A_DeviceMotion_data"
    output_dir = project_root / "data" / "motionsense"

    print("=" * 60)
    print("Processing MotionSense Dataset")
    print("=" * 60)
    print(f"Raw data: {raw_dir}")
    print(f"Output: {output_dir}")

    if not raw_dir.exists():
        print(f"\nError: Raw data not found at {raw_dir}")
        print("Please download from: https://github.com/mmalekzadeh/motion-sense")
        return

    # Process data
    num_sessions, labels = process_motionsense(raw_dir, output_dir)

    # Create and save manifest
    print("\nCreating manifest.json...")
    manifest = create_manifest()
    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    # Save labels
    print("Creating labels.json...")
    with open(output_dir / "labels.json", 'w') as f:
        json.dump(labels, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total sessions: {num_sessions}")

    # Count by activity
    activity_counts = {}
    for session_id, label_list in labels.items():
        label = label_list[0]
        activity_counts[label] = activity_counts.get(label, 0) + 1

    print("\nSessions per activity:")
    for activity, count in sorted(activity_counts.items()):
        print(f"  {activity:20s}: {count:4d}")

    # Verify a sample session
    print("\nVerifying sample session...")
    sample_session = list(labels.keys())[0]
    sample_path = output_dir / "sessions" / sample_session / "data.parquet"
    df = pd.read_parquet(sample_path)
    print(f"  Session: {sample_session}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Duration: {df['timestamp_sec'].max():.1f} sec")
    print(f"  Label: {labels[sample_session]}")

    print("\n" + "=" * 60)
    print("Done! MotionSense dataset processed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
