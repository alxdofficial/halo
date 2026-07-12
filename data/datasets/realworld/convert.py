"""
Convert RealWorld HAR dataset to standardized format.

Input: data/raw/realworld/
Output: data/realworld/
  - manifest.json (channel metadata)
  - labels.json (activity labels per session)
  - sessions/session_XXX/data.parquet (sensor data)

RealWorld format:
  - Organized by: subject/data/{sensor}_{activity}_csv.zip
  - Subject folders: proband1, proband2, ..., proband15
  - Activities: climbingdown, climbingup, jumping, lying, running, sitting, standing, walking
  - Positions (in ZIP): chest, forearm, head, shin, thigh, upperarm, waist
  - Sensor prefixes: acc (accelerometer), gyr (gyroscope), mag (magnetometer)
  - 15 subjects, 8 activities, 7 positions

We extract only the WAIST position for consistency with other datasets.
"""

import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent to path for shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.scripts.windowing import create_variable_windows


# Activity mapping - standardized names
ACTIVITIES = {
    "climbingdown": "stairs_down",
    "climbingup": "stairs_up",
    "jumping": "jumping",
    "lying": "lying",
    "running": "running",
    "sitting": "sitting",
    "standing": "standing",
    "walking": "walking",
}

# Sensor types mapping
SENSOR_TYPES = {
    "acc": "acc",
    "gyr": "gyro",
    "mag": "mag",
}

# Body position to use (for consistency with other waist-mounted datasets)
TARGET_POSITION = "waist"

# Paths - data may be in subdirectory or directly in raw folder
RAW_BASE_DIR = Path("data/raw/realworld")
RAW_SUBDIR = RAW_BASE_DIR / "realworld2016_dataset"
OUTPUT_DIR = Path("data/realworld")

# Target sampling rate
TARGET_SAMPLE_RATE = 50.0


def load_sensor_from_zip(zip_path: Path, sensor_prefix: str, activity: str) -> Optional[pd.DataFrame]:
    """
    Load sensor data from a ZIP file for the waist position.

    Args:
        zip_path: Path to the ZIP file (e.g., acc_walking_csv.zip)
        sensor_prefix: Sensor type prefix (acc, gyr, mag)
        activity: Activity name

    Returns:
        DataFrame with timestamp_sec and sensor columns, or None if failed.
    """
    try:
        target_filename = f"{sensor_prefix}_{activity}_{TARGET_POSITION}.csv"

        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Check if target file exists
            if target_filename not in zf.namelist():
                return None

            # Read CSV from ZIP
            with zf.open(target_filename) as f:
                df = pd.read_csv(f)

        if len(df) < 10:
            return None

        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]

        # Get timestamp
        time_col = "attr_time" if "attr_time" in df.columns else "time"
        if time_col not in df.columns:
            return None

        # Convert timestamp to seconds (RealWorld uses Unix milliseconds)
        time_values = df[time_col].values.astype(float)
        time_sec = (time_values - time_values[0]) / 1000.0  # Convert ms to sec

        # Map sensor prefix to output name
        output_sensor = SENSOR_TYPES.get(sensor_prefix, sensor_prefix)

        # Create result DataFrame
        result = pd.DataFrame()
        result["timestamp_sec"] = time_sec

        # Get x, y, z values
        for axis in ["x", "y", "z"]:
            src_col = f"attr_{axis}"
            if src_col in df.columns:
                result[f"{output_sensor}_{axis}"] = df[src_col].values.astype(float)

        return result

    except Exception as e:
        print(f"      ERROR loading {zip_path}: {e}")
        return None


def merge_sensor_data(sensor_dfs: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Merge accelerometer, gyroscope, and magnetometer data on timestamps.

    Uses linear interpolation to align sensors to common timeline.
    """
    if not sensor_dfs:
        return None

    # Find the sensor with most samples to use as reference timeline
    ref_sensor = max(sensor_dfs.keys(), key=lambda k: len(sensor_dfs[k]))
    ref_df = sensor_dfs[ref_sensor]

    result = pd.DataFrame()
    result["timestamp_sec"] = ref_df["timestamp_sec"].values

    # Add reference sensor columns
    for col in ref_df.columns:
        if col != "timestamp_sec":
            result[col] = ref_df[col].values

    # Interpolate other sensors to reference timeline
    for sensor_type, df in sensor_dfs.items():
        if sensor_type == ref_sensor:
            continue

        for col in df.columns:
            if col == "timestamp_sec":
                continue

            # Interpolate to reference timeline
            result[col] = np.interp(
                result["timestamp_sec"].values,
                df["timestamp_sec"].values,
                df[col].values,
            )

    return result


def resample_to_target_rate(df: pd.DataFrame, target_rate: float = 50.0) -> Optional[pd.DataFrame]:
    """Resample data to target sampling rate using linear interpolation."""
    if df is None or len(df) == 0:
        return None

    duration = df["timestamp_sec"].iloc[-1] - df["timestamp_sec"].iloc[0]
    if duration <= 0:
        return None

    num_samples = int(duration * target_rate) + 1
    new_timestamps = np.linspace(0, duration, num_samples)

    result = pd.DataFrame()
    result["timestamp_sec"] = new_timestamps

    for col in df.columns:
        if col != "timestamp_sec":
            result[col] = np.interp(new_timestamps, df["timestamp_sec"].values, df[col].values)

    return result


def convert_realworld():
    """Convert RealWorld dataset to standardized format."""
    print("=" * 80)
    print("RealWorld HAR → Standardized Format Converter")
    print("=" * 80)

    # Check input - data may be in subdirectory or directly in base folder
    if RAW_SUBDIR.exists():
        raw_dir = RAW_SUBDIR
    elif RAW_BASE_DIR.exists():
        raw_dir = RAW_BASE_DIR
    else:
        print(f"ERROR: Raw data not found at {RAW_BASE_DIR}")
        print("Run: python datascripts/realworld/download.py")
        return False

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions_dir = OUTPUT_DIR / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    # Find all subject folders
    subject_folders = sorted([d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("proband")])
    print(f"\nFound {len(subject_folders)} subjects")

    if len(subject_folders) == 0:
        print("ERROR: No proband folders found")
        return False

    all_labels = {}
    session_count = 0
    skipped_count = 0

    for subject_folder in subject_folders:
        subject_id = int(re.search(r"\d+", subject_folder.name).group())
        print(f"\n  Processing subject {subject_id} ({subject_folder.name})")

        # Find data folder
        data_dir = subject_folder / "data"
        if not data_dir.exists():
            print(f"    WARNING: No data folder found for {subject_folder.name}")
            continue

        subject_sessions = 0

        for activity_code, activity_name in ACTIVITIES.items():
            # Load each sensor type
            sensor_dfs = {}

            for sensor_prefix, sensor_name in SENSOR_TYPES.items():
                zip_filename = f"{sensor_prefix}_{activity_code}_csv.zip"
                zip_path = data_dir / zip_filename

                if not zip_path.exists():
                    continue

                df = load_sensor_from_zip(zip_path, sensor_prefix, activity_code)
                if df is not None and len(df) > 10:
                    sensor_dfs[sensor_name] = df

            if not sensor_dfs:
                skipped_count += 1
                continue

            # Merge sensor data
            merged_df = merge_sensor_data(sensor_dfs)
            if merged_df is None or len(merged_df) < 10:
                skipped_count += 1
                continue

            # Resample to target rate
            resampled_df = resample_to_target_rate(merged_df, TARGET_SAMPLE_RATE)
            if resampled_df is None or len(resampled_df) < 10:
                skipped_count += 1
                continue

            # Create session ID prefix
            session_prefix = f"s{subject_id:02d}_{activity_code}"

            # Split long sessions into variable-length windows
            windows = create_variable_windows(
                df=resampled_df,
                session_prefix=session_prefix,
                activity=activity_name,
                sample_rate=TARGET_SAMPLE_RATE,
                seed=42 + subject_id * 100,  # Reproducible but varied per subject
            )

            # Save each window as a separate session
            for window_id, window_df, window_activity in windows:
                window_path = sessions_dir / window_id
                window_path.mkdir(exist_ok=True)

                parquet_path = window_path / "data.parquet"
                window_df.to_parquet(parquet_path, index=False)

                # Store label
                all_labels[window_id] = [window_activity]
                session_count += 1
                subject_sessions += 1

        print(f"    Sessions: {subject_sessions}")

    # Save labels.json
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(all_labels, f, indent=2)
    print(f"\n✓ Created labels.json ({len(all_labels)} sessions)")

    # Create manifest.json
    create_manifest()

    # Summary
    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {session_count} sessions converted")
    print(f"  - {skipped_count} recordings skipped")
    print(f"  - Position: {TARGET_POSITION}")
    print(f"  - {TARGET_SAMPLE_RATE} Hz sampling rate")

    # Generate debug visualizations
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.scripts.visualization_utils import generate_debug_visualizations

        generate_debug_visualizations(OUTPUT_DIR)
    except ImportError as e:
        print(f"\n⚠ Could not generate visualizations: {e}")
        print("Install matplotlib: pip install matplotlib")

    return True


def create_manifest():
    """Create manifest.json with channel metadata."""
    manifest = {
        "dataset_name": "RealWorld HAR",
        "description": "RealWorld Human Activity Recognition dataset from University of Mannheim. 15 subjects performing 8 activities with sensors at waist position. Triaxial accelerometer, gyroscope, and magnetometer resampled to 50Hz.",
        "source": "https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/",
        "num_subjects": 15,
        "body_position": TARGET_POSITION,
        "channels": [
            {"name": "acc_x", "description": "Accelerometer X-axis (waist)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "acc_y", "description": "Accelerometer Y-axis (waist)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "acc_z", "description": "Accelerometer Z-axis (waist)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "gyro_x", "description": "Gyroscope X-axis (waist)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "gyro_y", "description": "Gyroscope Y-axis (waist)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "gyro_z", "description": "Gyroscope Z-axis (waist)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "mag_x", "description": "Magnetometer X-axis (waist)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "mag_y", "description": "Magnetometer Y-axis (waist)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "mag_z", "description": "Magnetometer Z-axis (waist)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
        ],
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Created manifest.json")


def main():
    """Main entry point."""
    success = convert_realworld()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
