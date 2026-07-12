"""
Convert WISDM dataset to standardized format.

Input: data/raw/wisdm/wisdm-dataset/
Output: data/wisdm/
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

# Run as `python -m data.datasets.wisdm.convert` from repo root; repo root on path for shared imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# Activity mapping (A-S, excluding N)
ACTIVITIES = {
    'A': 'walking',
    'B': 'jogging',
    'C': 'stairs',
    'D': 'sitting',
    'E': 'standing',
    'F': 'typing',
    'G': 'brushing_teeth',
    'H': 'eating_soup',
    'I': 'eating_chips',
    'J': 'eating_pasta',
    'K': 'drinking',
    'L': 'eating_sandwich',
    'M': 'kicking',
    'O': 'playing_catch',
    'P': 'dribbling',
    'Q': 'writing',
    'R': 'clapping',
    'S': 'folding_clothes'
}

# Paths (new repo layout: this converter lives in data/datasets/wisdm/)
DS_DIR = Path(__file__).resolve().parent
RAW_DIR = DS_DIR / "downloads" / "wisdm-dataset" / "raw"
OUTPUT_DIR = DS_DIR


def load_sensor_data(device: str, sensor: str):
    """Load data for one device-sensor combination."""
    sensor_dir = RAW_DIR / device / sensor

    if not sensor_dir.exists():
        print(f"    WARNING: {sensor_dir} not found")
        return {}

    # Data organized by subject (1600-1650)
    subject_files = sorted(sensor_dir.glob("*.txt"))

    subject_data = {}
    for subject_file in subject_files:
        subject_id = subject_file.stem

        # Parse CSV format: subject_id,activity_code,timestamp,x,y,z;
        # Note: WISDM data has semicolons at the end of each line
        try:
            # Read without dtype specification to handle semicolons
            df = pd.read_csv(
                subject_file,
                names=['subject', 'activity', 'timestamp', 'x', 'y', 'z'],
                sep=',',
                on_bad_lines='skip'  # Skip malformed lines
            )
            # Remove trailing semicolons from the last column and convert to numeric
            df['z'] = df['z'].astype(str).str.rstrip(';').astype(float)
            df['x'] = df['x'].astype(float)
            df['y'] = df['y'].astype(float)

            subject_data[subject_id] = df
        except Exception as e:
            print(f"      ERROR loading {subject_file.name}: {e}")

    return subject_data


def segment_by_subject_activity(subject_data_dict: dict, device_label: str, sensor_label: str):
    """Segment data by subject and activity."""
    sessions = []

    for subject_id, df in subject_data_dict.items():
        # Group by activity
        for activity_code in df['activity'].unique():
            activity_data = df[df['activity'] == activity_code].copy()

            if len(activity_data) < 20:  # Skip very short segments (< 1 second at 20Hz)
                continue

            # Sort by timestamp
            activity_data = activity_data.sort_values('timestamp')

            # Create session ID
            activity_name = ACTIVITIES.get(activity_code, 'unknown')
            session_id = f"{device_label}_{sensor_label}_{subject_id}_{activity_code}"

            # Calculate duration
            duration_sec = len(activity_data) / 20.0  # 20 Hz

            sessions.append({
                'session_id': session_id,
                'data': activity_data,
                'activity_name': activity_name,
                'duration_sec': duration_sec,
                'device': device_label,
                'sensor': sensor_label
            })

    return sessions


def merge_device_sensors():
    """Merge data from phone and watch, accel and gyro."""
    print("\nLoading sensor data...")

    # Load all combinations
    phone_accel = load_sensor_data("phone", "accel")
    phone_gyro = load_sensor_data("phone", "gyro")
    watch_accel = load_sensor_data("watch", "accel")
    watch_gyro = load_sensor_data("watch", "gyro")

    print(f"  Phone accel: {len(phone_accel)} subjects")
    print(f"  Phone gyro: {len(phone_gyro)} subjects")
    print(f"  Watch accel: {len(watch_accel)} subjects")
    print(f"  Watch gyro: {len(watch_gyro)} subjects")

    # For simplicity, create separate sessions for each device-sensor combo
    # In a more advanced version, we could try to align and merge them

    all_sessions = []
    all_sessions.extend(segment_by_subject_activity(phone_accel, "phone", "accel"))
    all_sessions.extend(segment_by_subject_activity(phone_gyro, "phone", "gyro"))
    all_sessions.extend(segment_by_subject_activity(watch_accel, "watch", "accel"))
    all_sessions.extend(segment_by_subject_activity(watch_gyro, "watch", "gyro"))

    return all_sessions


def save_sessions(sessions):
    """Save all sessions to parquet with variable-length windowing."""
    labels_dict = {}
    sample_rate = 20.0  # WISDM is 20Hz

    total_windows = 0

    for session in sessions:
        base_session_id = session['session_id']
        data = session['data'].copy()
        activity = session['activity_name']

        # Reset timestamp to start from 0 (convert from nanoseconds to seconds)
        min_timestamp = data['timestamp'].min()
        data['timestamp_sec'] = (data['timestamp'] - min_timestamp) / 1e9  # Unix nanoseconds to sec
        data['timestamp_sec'] = data['timestamp_sec'].astype(float)  # Ensure float type

        # Keep only sensor columns
        device = session['device']
        sensor = session['sensor']

        # Subject id (from the raw first column) for subject-disjoint splits; the session id starts
        # with the device, so build_grids.iter_sessions reads the subject from this column.
        subject_id = str(data['subject'].iloc[0]) if 'subject' in data.columns else base_session_id.split('_')[2]

        # Rename x, y, z to be device/sensor-specific and ensure float
        data = data.rename(columns={
            'x': f'{device}_{sensor}_x',
            'y': f'{device}_{sensor}_y',
            'z': f'{device}_{sensor}_z'
        })

        # Select final columns; keep the whole continuous single-activity recording as ONE session
        # (build_grids does the fixed 6 s windowing — avoids double-windowing).
        cols = ['timestamp_sec', f'{device}_{sensor}_x', f'{device}_{sensor}_y', f'{device}_{sensor}_z']
        out = data[cols].astype(float)
        out['subject'] = subject_id

        session_dir = OUTPUT_DIR / "sessions" / base_session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(session_dir / "data.parquet", index=False)

        labels_dict[base_session_id] = [activity]
        total_windows += 1

    print(f"  Created {total_windows} sessions from {len(sessions)} original segments")
    return labels_dict


def create_manifest():
    """Create minimal manifest.json."""
    manifest = {
        "dataset_name": "WISDM",
        "description": "Smartphone and smartwatch activity recognition. 51 subjects performing 18 activities with accelerometer and gyroscope on phone (pocket) and watch (wrist) at 20Hz.",
        "channels": [
            {
                "name": "phone_accel_x",
                "description": "Phone accelerometer X-axis (pocket-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "phone_accel_y",
                "description": "Phone accelerometer Y-axis (pocket-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "phone_accel_z",
                "description": "Phone accelerometer Z-axis (pocket-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "phone_gyro_x",
                "description": "Phone gyroscope X-axis (pocket-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "phone_gyro_y",
                "description": "Phone gyroscope Y-axis (pocket-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "phone_gyro_z",
                "description": "Phone gyroscope Z-axis (pocket-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "watch_accel_x",
                "description": "Smartwatch accelerometer X-axis (wrist-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "watch_accel_y",
                "description": "Smartwatch accelerometer Y-axis (wrist-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "watch_accel_z",
                "description": "Smartwatch accelerometer Z-axis (wrist-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "watch_gyro_x",
                "description": "Smartwatch gyroscope X-axis (wrist-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "watch_gyro_y",
                "description": "Smartwatch gyroscope Y-axis (wrist-worn)",
                "sampling_rate_hz": 20.0
            },
            {
                "name": "watch_gyro_z",
                "description": "Smartwatch gyroscope Z-axis (wrist-worn)",
                "sampling_rate_hz": 20.0
            }
        ]
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Created manifest: {manifest_path}")


def main():
    """Convert WISDM to standardized format."""
    print("=" * 80)
    print("WISDM → Standardized Format Converter")
    print("=" * 80)

    # Check input
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data not found at {RAW_DIR}")
        print("Run: python datascripts/download_all_datasets.py wisdm")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Merge and segment all data
    sessions = merge_device_sensors()
    print(f"\nCreated {len(sessions)} sessions")

    # Save sessions and labels
    print("\nSaving sessions...")
    labels_dict = save_sessions(sessions)

    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(labels_dict, f, indent=2)

    print(f"\n✓ Created labels: {labels_path}")
    print(f"  Total sessions: {len(labels_dict)}")

    # Create manifest
    create_manifest()

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {len(labels_dict)} sessions")
    print(f"  - 12 channels (phone + watch, accel + gyro)")
    print(f"  - 20 Hz sampling rate")
    print(f"\nNOTE: Each session contains data from ONE device-sensor combination.")
    print(f"      To use multi-modal data, merge sessions by subject and activity.")

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
