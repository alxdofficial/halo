"""
Convert MobiAct/MobiFall dataset to standardized format.

Supports two formats:
1. MobiAct (Annotated Data): CSV files with combined sensor data
2. MobiFall (Kaggle): Separate TXT files for acc/gyro/ori per trial

Input: data/raw/mobiact/
Output: data/mobiact/
  - manifest.json (channel metadata)
  - labels.json (activity labels per session)
  - sessions/session_XXX/data.parquet (sensor data)

Activity codes:
  ADL: STD, WAL, JOG, JUM, STU, STN, SCH, SIT, CSI, CSO, LYI
  Falls: FOL, FKL, BSC, SDL
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent to path for shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.scripts.windowing import create_variable_windows


# Activity mapping - standardized names
ACTIVITIES = {
    # Activities of Daily Living
    "STD": "standing",
    "WAL": "walking",
    "JOG": "jogging",
    "JUM": "jumping",
    "STU": "stairs_up",
    "STN": "stairs_down",
    "SCH": "sitting_chair",
    "SIT": "sitting",
    "CSI": "car_step_in",
    "CSO": "car_step_out",
    "LYI": "lying",
    # Falls
    "FOL": "fall_forward",
    "FKL": "fall_forward_knees",  # FKL = Front-Knees-Lying: FORWARD fall onto knees (per paper Table 2)
    "BSC": "fall_backward_sitting",
    "SDL": "fall_sideways",
}

# Paths
RAW_DIR = Path("data/raw/mobiact")
OUTPUT_DIR = Path("data/mobiact")

# Target sampling rate
TARGET_SAMPLE_RATE = 50.0


def detect_format() -> str:
    """Detect which dataset format is present."""
    # Check for MobiAct format (Annotated Data folder)
    mobiact_dir = RAW_DIR / "Annotated Data"
    if mobiact_dir.exists():
        csv_files = list(mobiact_dir.glob("**/*.csv"))
        if csv_files:
            return "mobiact"

    # Check for MobiFall format
    mobifall_dir = RAW_DIR / "MobiFall_Dataset_v2.0"
    if mobifall_dir.exists():
        txt_files = list(mobifall_dir.glob("**/*.txt"))
        if txt_files:
            return "mobifall"

    # Check for MobiFall directly in RAW_DIR
    txt_files = list(RAW_DIR.glob("**/sub*/**/*.txt"))
    if txt_files:
        return "mobifall"

    return "unknown"


# ==================== MobiFall Format Functions ====================

def parse_mobifall_txt(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Parse MobiFall TXT file.

    Format:
    - Header lines start with #
    - Data: timestamp(ns), x, y, z
    """
    try:
        # Read file, skip comment lines
        lines = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    lines.append(line)

        if len(lines) < 10:
            return None

        # Parse data
        data = []
        for line in lines:
            parts = line.split(",")
            if len(parts) >= 4:
                try:
                    ts = float(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    data.append([ts, x, y, z])
                except ValueError:
                    continue

        if len(data) < 10:
            return None

        df = pd.DataFrame(data, columns=["timestamp_ns", "x", "y", "z"])

        # Convert timestamp from nanoseconds to seconds
        df["timestamp_sec"] = (df["timestamp_ns"] - df["timestamp_ns"].iloc[0]) / 1e9

        return df

    except Exception as e:
        return None


def find_mobifall_trials(base_dir: Path) -> List[Dict]:
    """
    Find all trials in MobiFall format.

    Returns list of dicts with: activity, subject, trial, acc_file, gyro_file
    """
    trials = []

    # Find all accelerometer files
    acc_files = list(base_dir.glob("**/*_acc_*.txt"))

    for acc_file in acc_files:
        # Parse filename: {ACTIVITY}_acc_{subject}_{trial}.txt
        match = re.match(r"([A-Z]+)_acc_(\d+)_(\d+)\.txt", acc_file.name)
        if not match:
            continue

        activity = match.group(1)
        subject = int(match.group(2))
        trial = int(match.group(3))

        # Find corresponding gyro file
        gyro_file = acc_file.parent / f"{activity}_gyro_{subject}_{trial}.txt"

        trials.append({
            "activity": activity,
            "subject": subject,
            "trial": trial,
            "acc_file": acc_file,
            "gyro_file": gyro_file if gyro_file.exists() else None,
        })

    return trials


def convert_mobifall_trial(trial_info: Dict) -> Optional[Tuple[pd.DataFrame, str]]:
    """Convert a single MobiFall trial to standardized format."""
    activity = trial_info["activity"]

    # Skip unknown activities
    if activity not in ACTIVITIES:
        return None

    activity_name = ACTIVITIES[activity]

    # Load accelerometer data
    acc_df = parse_mobifall_txt(trial_info["acc_file"])
    if acc_df is None or len(acc_df) < 10:
        return None

    # Create result DataFrame
    result = pd.DataFrame()
    result["timestamp_sec"] = acc_df["timestamp_sec"].values
    result["acc_x"] = acc_df["x"].values
    result["acc_y"] = acc_df["y"].values
    result["acc_z"] = acc_df["z"].values

    # Load gyroscope data if available
    if trial_info["gyro_file"] is not None:
        gyro_df = parse_mobifall_txt(trial_info["gyro_file"])
        if gyro_df is not None and len(gyro_df) > 10:
            # Interpolate gyro to acc timestamps
            result["gyro_x"] = np.interp(
                result["timestamp_sec"].values,
                gyro_df["timestamp_sec"].values,
                gyro_df["x"].values,
            )
            result["gyro_y"] = np.interp(
                result["timestamp_sec"].values,
                gyro_df["timestamp_sec"].values,
                gyro_df["y"].values,
            )
            result["gyro_z"] = np.interp(
                result["timestamp_sec"].values,
                gyro_df["timestamp_sec"].values,
                gyro_df["z"].values,
            )
    else:
        result["gyro_x"] = np.nan
        result["gyro_y"] = np.nan
        result["gyro_z"] = np.nan

    return result, activity_name


# ==================== MobiAct Format Functions ====================

def parse_mobiact_filename(filename: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """Parse MobiAct filename: {ACTIVITY}_{SUBJECT}_{TRIAL}_annotated.csv"""
    base = filename.replace("_annotated.csv", "").replace(".csv", "")
    parts = base.split("_")

    if len(parts) >= 3:
        activity = parts[0]
        try:
            subject = int(parts[1])
            trial = int(parts[2])
            return activity, subject, trial
        except ValueError:
            pass

    return None, None, None


def load_mobiact_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """Load MobiAct CSV file."""
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.lower().strip() for c in df.columns]

        # Find timestamp column
        time_col = None
        for candidate in ["rel_time", "timestamp", "time", "t"]:
            if candidate in df.columns:
                time_col = candidate
                break
        if time_col is None:
            time_col = df.columns[0]

        # Convert time to seconds
        time_values = df[time_col].values
        if time_values.max() > 1e9:
            time_sec = (time_values - time_values[0]) / 1e9
        elif time_values.max() > 1e6:
            time_sec = (time_values - time_values[0]) / 1e3
        else:
            time_sec = time_values - time_values[0]

        result = pd.DataFrame()
        result["timestamp_sec"] = time_sec

        # Map columns
        column_mapping = {
            "acc_x": ["acc_x", "accx"],
            "acc_y": ["acc_y", "accy"],
            "acc_z": ["acc_z", "accz"],
            "gyro_x": ["gyro_x", "gyrox"],
            "gyro_y": ["gyro_y", "gyroy"],
            "gyro_z": ["gyro_z", "gyroz"],
        }

        for target, candidates in column_mapping.items():
            for c in candidates:
                if c in df.columns:
                    result[target] = df[c].values
                    break
            if target not in result.columns:
                result[target] = np.nan

        return result

    except Exception as e:
        return None


# ==================== Common Functions ====================

def resample_to_target_rate(df: pd.DataFrame, target_rate: float = 50.0) -> Optional[pd.DataFrame]:
    """Resample to target sampling rate."""
    if df is None or len(df) == 0:
        return None

    duration = df["timestamp_sec"].iloc[-1] - df["timestamp_sec"].iloc[0]
    if duration <= 0:
        return None

    num_samples = int(duration * target_rate) + 1
    new_timestamps = np.linspace(0, duration, num_samples)

    result = pd.DataFrame()
    result["timestamp_sec"] = new_timestamps

    for col in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
        if col in df.columns and not df[col].isna().all():
            result[col] = np.interp(new_timestamps, df["timestamp_sec"].values, df[col].values)
        else:
            result[col] = np.nan

    return result


def convert_dataset():
    """Convert MobiAct/MobiFall dataset to standardized format."""
    print("=" * 80)
    print("MobiAct/MobiFall → Standardized Format Converter")
    print("=" * 80)

    # Check input
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data not found at {RAW_DIR}")
        print("Run: python datascripts/mobiact/download.py")
        return False

    # Detect format
    data_format = detect_format()
    print(f"\nDetected format: {data_format}")

    if data_format == "unknown":
        print("ERROR: Could not detect data format")
        print("Expected either:")
        print(f"  - {RAW_DIR}/Annotated Data/ (MobiAct)")
        print(f"  - {RAW_DIR}/MobiFall_Dataset_v2.0/ (MobiFall)")
        return False

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions_dir = OUTPUT_DIR / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    all_labels = {}
    session_count = 0
    skipped_count = 0

    if data_format == "mobifall":
        # Process MobiFall format
        mobifall_dir = RAW_DIR / "MobiFall_Dataset_v2.0"
        if not mobifall_dir.exists():
            mobifall_dir = RAW_DIR

        trials = find_mobifall_trials(mobifall_dir)
        print(f"Found {len(trials)} trials")

        for trial_info in trials:
            result = convert_mobifall_trial(trial_info)
            if result is None:
                skipped_count += 1
                continue

            df, activity_name = result

            # Resample
            df = resample_to_target_rate(df, TARGET_SAMPLE_RATE)
            if df is None or len(df) < 10:
                skipped_count += 1
                continue

            # Create session ID prefix
            activity = trial_info["activity"]
            subject = trial_info["subject"]
            trial = trial_info["trial"]
            session_prefix = f"{activity}_{subject:02d}_{trial:02d}"

            # Split long sessions into variable-length windows
            windows = create_variable_windows(
                df=df,
                session_prefix=session_prefix,
                activity=activity_name,
                sample_rate=TARGET_SAMPLE_RATE,
                seed=42 + subject * 100 + trial,
            )

            # Save each window
            for window_id, window_df, window_activity in windows:
                window_path = sessions_dir / window_id
                window_path.mkdir(exist_ok=True)
                window_df.to_parquet(window_path / "data.parquet", index=False)

                all_labels[window_id] = [window_activity]
                session_count += 1

            if session_count % 100 == 0:
                print(f"  Processed {session_count} sessions...")

    else:
        # Process MobiAct format
        mobiact_dir = RAW_DIR / "Annotated Data"
        activity_folders = [d for d in mobiact_dir.iterdir() if d.is_dir()]
        print(f"Found {len(activity_folders)} activity folders")

        for activity_folder in sorted(activity_folders):
            activity_code = activity_folder.name.upper()
            if activity_code not in ACTIVITIES:
                continue

            activity_name = ACTIVITIES[activity_code]
            csv_files = list(activity_folder.glob("*.csv"))

            for csv_file in csv_files:
                activity, subject, trial = parse_mobiact_filename(csv_file.name)
                if activity is None:
                    skipped_count += 1
                    continue

                df = load_mobiact_csv(csv_file)
                if df is None or len(df) < 10:
                    skipped_count += 1
                    continue

                df = resample_to_target_rate(df, TARGET_SAMPLE_RATE)
                if df is None or len(df) < 10:
                    skipped_count += 1
                    continue

                session_prefix = f"{activity}_{subject:02d}_{trial:02d}"

                # Split long sessions into variable-length windows
                windows = create_variable_windows(
                    df=df,
                    session_prefix=session_prefix,
                    activity=activity_name,
                    sample_rate=TARGET_SAMPLE_RATE,
                    seed=42 + subject * 100 + trial,
                )

                # Save each window
                for window_id, window_df, window_activity in windows:
                    window_path = sessions_dir / window_id
                    window_path.mkdir(exist_ok=True)
                    window_df.to_parquet(window_path / "data.parquet", index=False)

                    all_labels[window_id] = [window_activity]
                    session_count += 1

    # Save labels.json
    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump(all_labels, f, indent=2)
    print(f"\n✓ Created labels.json ({len(all_labels)} sessions)")

    # Create manifest
    create_manifest(data_format)

    # Count unique activities
    activities = set()
    for labels in all_labels.values():
        activities.update(labels)

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {session_count} sessions converted")
    print(f"  - {skipped_count} files skipped")
    print(f"  - {len(activities)} unique activities")
    print(f"  - {TARGET_SAMPLE_RATE} Hz sampling rate")

    # Generate visualizations
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.scripts.visualization_utils import generate_debug_visualizations
        generate_debug_visualizations(OUTPUT_DIR)
    except ImportError:
        pass

    return True


def create_manifest(data_format: str):
    """Create manifest.json."""
    if data_format == "mobifall":
        name = "MobiFall"
        desc = "MobiAct (v2.0): ADL + fall dataset. Samsung Galaxy S3 in a trouser pocket, freely chosen in any random orientation. 9 ADL activities + 4 fall types."
        num_subjects = 31
    else:
        name = "MobiAct"
        desc = "MobiAct: Recognition of Activities of Daily Living using Smartphones. 66 subjects performing 11 ADL activities and 4 fall types."
        num_subjects = 66

    manifest = {
        "dataset_name": name,
        "description": f"{desc} Triaxial accelerometer and gyroscope resampled to {TARGET_SAMPLE_RATE}Hz.",
        "source": "https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/",
        "num_subjects": num_subjects,
        "channels": [
            {"name": "acc_x", "description": "Accelerometer X-axis", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "acc_y", "description": "Accelerometer Y-axis", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "acc_z", "description": "Accelerometer Z-axis", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "gyro_x", "description": "Gyroscope X-axis", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "gyro_y", "description": "Gyroscope Y-axis", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "gyro_z", "description": "Gyroscope Z-axis", "sampling_rate_hz": TARGET_SAMPLE_RATE},
        ],
    }

    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("✓ Created manifest.json")


def main():
    success = convert_dataset()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
