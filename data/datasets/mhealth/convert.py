"""
Convert MHEALTH dataset to standardized format.

Input: data/raw/mhealth/MHEALTHDATASET/
Output: data/mhealth/
  - manifest.json
  - labels.json
  - sessions/session_XXX/data.parquet
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add datascripts to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.scripts.assembly.windowing import create_variable_windows, get_window_range


# Activity mapping
ACTIVITIES = {
    0: "null",
    1: "standing",
    2: "sitting",
    3: "lying",
    4: "walking",
    5: "climbing_stairs",
    6: "waist_bends_forward",
    7: "frontal_elevation_arms",
    8: "knees_bending",
    9: "cycling",
    10: "jogging",
    11: "running",
    12: "jump_front_back"
}

# Column names (23 sensor columns + 1 label)
COLUMN_NAMES = [
    'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
    'ecg_lead1', 'ecg_lead2',
    'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
    'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
    'arm_acc_x', 'arm_acc_y', 'arm_acc_z',
    'arm_gyro_x', 'arm_gyro_y', 'arm_gyro_z',
    'arm_mag_x', 'arm_mag_y', 'arm_mag_z',
    'activity_id'
]

# Paths
RAW_DIR = Path("data/raw/mhealth/MHEALTHDATASET")
OUTPUT_DIR = Path("data/mhealth")


def segment_continuous_activity(df: pd.DataFrame, min_duration_sec: float = 3.0):
    """Segment continuous recording into sessions based on activity changes."""
    sessions = []

    # Find activity boundaries
    activity_changes = df['activity_id'].ne(df['activity_id'].shift())
    segment_ids = activity_changes.cumsum()

    # Sampling rate is 50 Hz
    sampling_rate = 50.0

    for seg_id in segment_ids.unique():
        segment = df[segment_ids == seg_id].copy()

        # Skip null activities and very short segments
        activity_id = segment['activity_id'].iloc[0]
        if activity_id == 0:
            continue

        duration_sec = len(segment) / sampling_rate
        if duration_sec < min_duration_sec:
            continue

        sessions.append({
            'data': segment,
            'activity_id': activity_id,
            'duration_sec': duration_sec
        })

    return sessions


def convert_subject(subject_file: Path):
    """Convert one subject file to sessions with variable-length windowing."""
    print(f"  Processing: {subject_file.name}")

    # Load data (tab/space-separated, 24 columns)
    try:
        df = pd.read_csv(subject_file, sep='\t', header=None, names=COLUMN_NAMES)
    except Exception as e:
        print(f"    ERROR loading {subject_file.name}: {e}")
        return [], {}

    # Segment into sessions
    sessions = segment_continuous_activity(df)

    print(f"    Found {len(sessions)} activity segments")

    subject_id = subject_file.stem.replace('mHealth_subject', '')
    session_data = []
    labels_dict = {}
    sample_rate = 50.0  # MHEALTH is 50Hz

    total_windows = 0

    for idx, session in enumerate(sessions):
        # Create base session ID
        base_session_id = f"subject{subject_id}_seg{idx:03d}"
        activity_name = ACTIVITIES.get(session['activity_id'], 'unknown')

        # Prepare DataFrame
        data = session['data'].copy()

        # Add timestamp column (50 Hz = 0.02 sec per sample)
        data.insert(0, 'timestamp_sec', np.arange(len(data)) * 0.02)

        # Drop activity_id (it's in labels.json)
        data = data.drop(columns=['activity_id'])

        # Apply variable-length windowing
        windows = create_variable_windows(
            df=data,
            session_prefix=base_session_id,
            activity=activity_name,
            sample_rate=sample_rate,
        )

        # Save each window
        for window_id, window_df, window_activity in windows:
            session_dir = OUTPUT_DIR / "sessions" / window_id
            session_dir.mkdir(parents=True, exist_ok=True)

            parquet_path = session_dir / "data.parquet"
            window_df.to_parquet(parquet_path, index=False)

            # Store label
            labels_dict[window_id] = [window_activity]
            session_data.append(window_id)
            total_windows += 1

    print(f"    Created {total_windows} windows from {len(sessions)} segments")
    return session_data, labels_dict


def create_manifest():
    """Create minimal manifest.json."""
    manifest = {
        "dataset_name": "MHEALTH",
        "description": "Mobile health monitoring with 3 wearable sensors (chest, left ankle, right wrist). 10 subjects performing 12 activities. Includes accelerometer, gyroscope, magnetometer, and 2-lead ECG.",
        "channels": [
            {
                "name": "chest_acc_x",
                "description": "Chest acceleration X-axis from wearable sensor",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "chest_acc_y",
                "description": "Chest acceleration Y-axis from wearable sensor",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "chest_acc_z",
                "description": "Chest acceleration Z-axis from wearable sensor",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "ecg_lead1",
                "description": "ECG lead 1 from chest sensor",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "ecg_lead2",
                "description": "ECG lead 2 from chest sensor",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "ankle_acc_x",
                "description": "Left ankle acceleration X-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "ankle_acc_y",
                "description": "Left ankle acceleration Y-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "ankle_acc_z",
                "description": "Left ankle acceleration Z-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "ankle_gyro_x",
                "description": "Left ankle angular velocity X-axis, low-reliability gyro (sample-and-hold ~14 Hz effective, near-constant magnitude)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "ankle_gyro_y",
                "description": "Left ankle angular velocity Y-axis, low-reliability gyro (sample-and-hold ~14 Hz effective, near-constant magnitude)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "ankle_gyro_z",
                "description": "Left ankle angular velocity Z-axis, low-reliability gyro (sample-and-hold ~14 Hz effective, near-constant magnitude)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "ankle_mag_x",
                "description": "Left ankle magnetic-field artifact X-axis (low reliability; excluded from HALO training input)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "ankle_mag_y",
                "description": "Left ankle magnetic-field artifact Y-axis (low reliability; excluded from HALO training input)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "ankle_mag_z",
                "description": "Left ankle magnetic-field artifact Z-axis (low reliability; excluded from HALO training input)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "arm_acc_x",
                "description": "Right wrist acceleration X-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "arm_acc_y",
                "description": "Right wrist acceleration Y-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "arm_acc_z",
                "description": "Right wrist acceleration Z-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "arm_gyro_x",
                "description": "Right wrist angular velocity X-axis, low-reliability gyro (sample-and-hold ~14 Hz effective, near-constant magnitude)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "arm_gyro_y",
                "description": "Right wrist angular velocity Y-axis, low-reliability gyro (sample-and-hold ~14 Hz effective, near-constant magnitude)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "arm_gyro_z",
                "description": "Right wrist angular velocity Z-axis, low-reliability gyro (sample-and-hold ~14 Hz effective, near-constant magnitude)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "arm_mag_x",
                "description": "Right wrist magnetic-field artifact X-axis (low reliability; excluded from HALO training input)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "arm_mag_y",
                "description": "Right wrist magnetic-field artifact Y-axis (low reliability; excluded from HALO training input)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "arm_mag_z",
                "description": "Right wrist magnetic-field artifact Z-axis (low reliability; excluded from HALO training input)",
                "sampling_rate_hz": 50.0
            }
        ]
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Created manifest: {manifest_path}")


def main():
    """Convert MHEALTH to standardized format."""
    print("=" * 80)
    print("MHEALTH → Standardized Format Converter")
    print("=" * 80)

    # Check input
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data not found at {RAW_DIR}")
        print("Run: python datascripts/download_all_datasets.py mhealth")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process all subject files
    subject_files = sorted(RAW_DIR.glob("mHealth_subject*.log"))
    print(f"\nFound {len(subject_files)} subject files")

    all_labels = {}
    all_sessions = []

    for subject_file in subject_files:
        sessions, labels = convert_subject(subject_file)
        all_sessions.extend(sessions)
        all_labels.update(labels)

    # Save labels.json
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(all_labels, f, indent=2)

    print(f"\n✓ Created labels: {labels_path}")
    print(f"  Total sessions: {len(all_labels)}")

    # Create manifest
    create_manifest()

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {len(all_labels)} sessions")
    print(f"  - 23 channels (3 IMUs + ECG)")
    print(f"  - 50 Hz sampling rate")

    # Generate debug visualizations
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))  # Add datascripts/ to path
        from data.scripts.debug.visualization_utils import generate_debug_visualizations

        generate_debug_visualizations(OUTPUT_DIR)
    except ImportError as e:
        print(f"\n⚠ Could not generate visualizations: {e}")
        print("Install matplotlib: pip install matplotlib")


if __name__ == "__main__":
    main()
