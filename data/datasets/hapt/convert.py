"""
Convert HAPT dataset to standardized format.

Input: data/datasets/hapt/downloads/
Output: data/datasets/hapt/
  - manifest.json (channel metadata)
  - labels.json (activity labels per session)
  - sessions/session_XXX/data.parquet (sensor data)

HAPT format:
  - RawData folder contains:
    - acc_expXX_userYY.txt (accelerometer data)
    - gyro_expXX_userYY.txt (gyroscope data)
  - labels.txt: exp_id, user_id, activity_id, start_sample, end_sample
  - Activity IDs: 1-12 (6 basic + 6 transitions)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Run as `python -m data.datasets.hapt.convert` from the repo root; repo root on path for shared imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# Activity mapping - standardized names
ACTIVITIES = {
    1: "walking",
    2: "walking_upstairs",
    3: "walking_downstairs",
    4: "sitting",
    5: "standing",
    6: "lying",
    7: "stand_to_sit",
    8: "sit_to_stand",
    9: "sit_to_lie",
    10: "lie_to_sit",
    11: "stand_to_lie",
    12: "lie_to_stand",
}

# Paths (new repo layout: this converter lives in data/datasets/hapt/)
DS_DIR = Path(__file__).resolve().parent
RAW_DIR = DS_DIR / "downloads"
OUTPUT_DIR = DS_DIR

# Dataset parameters
SAMPLE_RATE = 50.0  # Hz


def find_raw_data_dir() -> Optional[Path]:
    """Find the RawData directory (may be nested)."""
    # Check direct path
    if (RAW_DIR / "RawData").exists():
        return RAW_DIR / "RawData"

    # Check nested paths
    for subdir in RAW_DIR.iterdir():
        if subdir.is_dir():
            if (subdir / "RawData").exists():
                return subdir / "RawData"
            # Check for data files directly
            if list(subdir.glob("acc_exp*.txt")):
                return subdir

    return None


def load_sensor_file(filepath: Path) -> Optional[np.ndarray]:
    """Load sensor data from space-separated text file."""
    try:
        data = np.loadtxt(filepath)
        return data
    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        return None


def load_labels(labels_path: Path) -> List[Tuple[int, int, int, int, int]]:
    """
    Load labels file.
    Format: exp_id user_id activity_id start_sample end_sample
    """
    labels = []
    try:
        with open(labels_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    exp_id = int(parts[0])
                    user_id = int(parts[1])
                    activity_id = int(parts[2])
                    start = int(parts[3])
                    end = int(parts[4])
                    labels.append((exp_id, user_id, activity_id, start, end))
    except Exception as e:
        print(f"  ERROR loading labels: {e}")

    return labels


def convert_hapt():
    """Convert HAPT dataset to standardized format."""
    print("=" * 80)
    print("HAPT → Standardized Format Converter")
    print("=" * 80)

    # Find raw data directory
    raw_data_dir = find_raw_data_dir()
    if raw_data_dir is None:
        print(f"ERROR: Raw data not found in {RAW_DIR}")
        print("Run: python -m data.scripts.download_datasets hapt")
        return False

    print(f"Found raw data: {raw_data_dir}")

    # Find labels file (may be in parent directory)
    labels_path = raw_data_dir / "labels.txt"
    if not labels_path.exists():
        labels_path = raw_data_dir.parent / "labels.txt"
    if not labels_path.exists():
        # Search for it
        for p in RAW_DIR.rglob("labels.txt"):
            labels_path = p
            break

    if not labels_path.exists():
        print(f"ERROR: labels.txt not found")
        return False

    print(f"Found labels: {labels_path}")

    # Load labels
    labels_list = load_labels(labels_path)
    print(f"Found {len(labels_list)} labeled segments")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions_dir = OUTPUT_DIR / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    # Group labels by experiment and user
    exp_user_labels = {}
    for exp_id, user_id, activity_id, start, end in labels_list:
        key = (exp_id, user_id)
        if key not in exp_user_labels:
            exp_user_labels[key] = []
        exp_user_labels[key].append((activity_id, start, end))

    all_labels = {}
    session_count = 0
    skipped_count = 0

    # Process each experiment/user combination
    for (exp_id, user_id), segments in exp_user_labels.items():
        # Load accelerometer data
        acc_file = raw_data_dir / f"acc_exp{exp_id:02d}_user{user_id:02d}.txt"
        gyro_file = raw_data_dir / f"gyro_exp{exp_id:02d}_user{user_id:02d}.txt"

        if not acc_file.exists():
            skipped_count += 1
            continue

        acc_data = load_sensor_file(acc_file)
        gyro_data = load_sensor_file(gyro_file) if gyro_file.exists() else None

        if acc_data is None:
            skipped_count += 1
            continue

        # Process each labeled segment. Each segment is one continuous single-activity chunk;
        # we save it whole as ONE session (build_grids does the fixed 6 s windowing).
        for seg_idx, (activity_id, start_sample, end_sample) in enumerate(segments):
            if activity_id not in ACTIVITIES:
                continue

            activity_name = ACTIVITIES[activity_id]

            # Extract segment (1-indexed in labels file)
            start_idx = max(0, start_sample - 1)
            end_idx = min(len(acc_data), end_sample)

            if end_idx - start_idx < 10:
                skipped_count += 1
                continue

            # Build DataFrame
            segment_acc = acc_data[start_idx:end_idx]
            num_samples = len(segment_acc)

            result = pd.DataFrame()
            result["timestamp_sec"] = np.linspace(0, (num_samples - 1) / SAMPLE_RATE, num_samples)
            result["acc_x"] = segment_acc[:, 0]
            result["acc_y"] = segment_acc[:, 1]
            result["acc_z"] = segment_acc[:, 2]

            if gyro_data is not None and len(gyro_data) >= end_idx:
                segment_gyro = gyro_data[start_idx:end_idx]
                result["gyro_x"] = segment_gyro[:, 0]
                result["gyro_y"] = segment_gyro[:, 1]
                result["gyro_z"] = segment_gyro[:, 2]
            else:
                result["gyro_x"] = np.nan
                result["gyro_y"] = np.nan
                result["gyro_z"] = np.nan

            # Subject id (user) for subject-disjoint splits; read by build_grids.iter_sessions.
            result["subject"] = f"user{user_id:02d}"

            session_id = f"exp{exp_id:02d}_user{user_id:02d}_{activity_name}_{seg_idx:02d}"
            session_dir = sessions_dir / session_id
            session_dir.mkdir(exist_ok=True)
            result.to_parquet(session_dir / "data.parquet", index=False)

            all_labels[session_id] = [activity_name]
            session_count += 1

    # Save labels.json
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(all_labels, f, indent=2)
    print(f"\n✓ Created labels.json ({len(all_labels)} sessions)")

    # Create manifest.json
    create_manifest()

    # Count unique activities
    activities = set()
    for labels in all_labels.values():
        activities.update(labels)

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {session_count} sessions converted")
    print(f"  - {skipped_count} segments skipped")
    print(f"  - {len(activities)} unique activities: {sorted(activities)}")
    print(f"  - {SAMPLE_RATE} Hz sampling rate")

    # Generate visualizations
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.scripts.debug.visualization_utils import generate_debug_visualizations
        generate_debug_visualizations(OUTPUT_DIR)
    except ImportError:
        pass

    return True


def create_manifest():
    """Create manifest.json with channel metadata."""
    manifest = {
        "dataset_name": "HAPT",
        "description": "Human Activities and Postural Transitions dataset from UCI. 30 subjects performing 6 basic activities (walking, stairs, sitting, standing, lying) and 6 postural transitions (stand_to_sit, sit_to_stand, etc.) with waist-mounted smartphone (Samsung Galaxy S II). Triaxial accelerometer and gyroscope at 50Hz.",
        "source": "https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions",
        "num_subjects": 30,
        "body_position": "waist",
        "channels": [
            {"name": "acc_x", "description": "Accelerometer X-axis (waist)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_y", "description": "Accelerometer Y-axis (waist)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_z", "description": "Accelerometer Z-axis (waist)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_x", "description": "Gyroscope X-axis (waist)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_y", "description": "Gyroscope Y-axis (waist)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_z", "description": "Gyroscope Z-axis (waist)", "sampling_rate_hz": SAMPLE_RATE},
        ],
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Created manifest.json")


def main():
    """Main entry point."""
    success = convert_hapt()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
