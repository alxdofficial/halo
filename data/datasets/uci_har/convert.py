"""
Convert UCI HAR dataset to standardized format.

Input: data/raw/uci_har/UCI HAR Dataset/
Output: data/uci_har/
  - manifest.json (minimal, human-readable)
  - labels.json (activity labels per session)
  - sessions/session_XXX/data.parquet (all channels as DataFrame)
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path


# Activity mapping
ACTIVITIES = {
    1: "walking",
    2: "walking_upstairs",
    3: "walking_downstairs",
    4: "sitting",
    5: "standing",
    6: "laying"
}

# Paths
RAW_DIR = Path("data/raw/uci_har/UCI HAR Dataset")
OUTPUT_DIR = Path("data/uci_har")


def load_inertial_signals(base_path: Path, set_name: str):
    """Load raw inertial signals from UCI HAR format."""
    inertial_dir = base_path / set_name / "Inertial Signals"

    # Load all 9 inertial signal files
    signals = {}

    # Body acceleration (3 axes)
    for axis in ['x', 'y', 'z']:
        file_path = inertial_dir / f"body_acc_{axis}_{set_name}.txt"
        signals[f"body_acc_{axis}"] = np.loadtxt(file_path)

    # Gravity acceleration (3 axes)
    for axis in ['x', 'y', 'z']:
        file_path = inertial_dir / f"total_acc_{axis}_{set_name}.txt"
        total_acc = np.loadtxt(file_path)
        signals[f"total_acc_{axis}"] = total_acc

    # Body gyroscope (3 axes)
    for axis in ['x', 'y', 'z']:
        file_path = inertial_dir / f"body_gyro_{axis}_{set_name}.txt"
        signals[f"body_gyro_{axis}"] = np.loadtxt(file_path)

    return signals


def convert_set(set_name: str):
    """Convert train or test set to sessions."""
    print(f"\nProcessing {set_name} set...")

    # Load metadata
    subject_file = RAW_DIR / set_name / f"subject_{set_name}.txt"
    label_file = RAW_DIR / set_name / f"y_{set_name}.txt"

    subjects = np.loadtxt(subject_file, dtype=int)
    labels = np.loadtxt(label_file, dtype=int)

    # Load inertial signals
    signals = load_inertial_signals(RAW_DIR, set_name)

    # Each row in signals is one window (128 timesteps at 50Hz = 2.56 seconds)
    num_windows = signals['body_acc_x'].shape[0]
    print(f"  Found {num_windows} windows")

    sessions = []
    labels_dict = {}

    for idx in range(num_windows):
        # Create session ID
        subject_id = subjects[idx]
        activity_id = labels[idx]
        session_id = f"{set_name}_{subject_id:02d}_{idx:04d}"

        # Build DataFrame for this window (128 timesteps × 9 channels)
        data_dict = {}
        for channel_name, channel_data in signals.items():
            data_dict[channel_name] = channel_data[idx]  # 128 values

        df = pd.DataFrame(data_dict)

        # Add timestamp column (50 Hz = 0.02 sec per sample)
        df.insert(0, 'timestamp_sec', np.arange(128) * 0.02)

        # Save to parquet
        session_dir = OUTPUT_DIR / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = session_dir / "data.parquet"
        df.to_parquet(parquet_path, index=False)

        # Store label
        activity_name = ACTIVITIES[activity_id]
        labels_dict[session_id] = [activity_name]

        sessions.append(session_id)

    print(f"  ✓ Created {len(sessions)} sessions")
    return sessions, labels_dict


def create_manifest():
    """Create minimal manifest.json."""
    manifest = {
        "dataset_name": "UCI HAR",
        "description": "Human activity recognition from smartphone sensors. 30 subjects performing 6 activities with waist-mounted Samsung Galaxy S II. Triaxial accelerometer and gyroscope at 50Hz.",
        "channels": [
            {
                "name": "body_acc_x",
                "description": "Body acceleration X-axis from accelerometer (gravity removed)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "body_acc_y",
                "description": "Body acceleration Y-axis from accelerometer (gravity removed)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "body_acc_z",
                "description": "Body acceleration Z-axis from accelerometer (gravity removed)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "total_acc_x",
                "description": "Total acceleration X-axis from accelerometer (includes gravity)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "total_acc_y",
                "description": "Total acceleration Y-axis from accelerometer (includes gravity)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "total_acc_z",
                "description": "Total acceleration Z-axis from accelerometer (includes gravity)",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "body_gyro_x",
                "description": "Angular velocity X-axis from gyroscope",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "body_gyro_y",
                "description": "Angular velocity Y-axis from gyroscope",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "body_gyro_z",
                "description": "Angular velocity Z-axis from gyroscope",
                "sampling_rate_hz": 50.0
            }
        ]
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Created manifest: {manifest_path}")


def main():
    """Convert UCI HAR to standardized format."""
    print("=" * 80)
    print("UCI HAR → Standardized Format Converter")
    print("=" * 80)

    # Check input
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data not found at {RAW_DIR}")
        print("Run: python datascripts/download_all_datasets.py uci_har")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Convert train and test sets
    all_labels = {}

    train_sessions, train_labels = convert_set("train")
    all_labels.update(train_labels)

    test_sessions, test_labels = convert_set("test")
    all_labels.update(test_labels)

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
    print(f"  - 9 channels (accel + gyro)")
    print(f"  - 50 Hz sampling rate")

    # Generate debug visualizations
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))  # Add datascripts/ to path
        from data.scripts.visualization_utils import generate_debug_visualizations

        generate_debug_visualizations(OUTPUT_DIR)
    except ImportError as e:
        print(f"\n⚠ Could not generate visualizations: {e}")
        print("Install matplotlib: pip install matplotlib")


if __name__ == "__main__":
    main()
