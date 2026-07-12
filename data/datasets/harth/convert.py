"""
Convert HARTH dataset to standardized format.

Input: data/datasets/harth/downloads/
Output: data/datasets/harth/
  - manifest.json
  - labels.json
  - sessions/session_XXX/data.parquet

HARTH (Human Activity Recognition Trondheim) Dataset Info:
- 22 subjects
- 12 activities: walking, running, shuffling, stairs_up, stairs_down,
  standing, sitting, lying, cycling_sit, cycling_stand,
  transport_sit, transport_stand
- 2 sensors: 3-axis accelerometers at thigh and lower back
- 6 channels: back_acc_x/y/z + thigh_acc_x/y/z
- 50 Hz sampling rate
- CSV format with continuous recordings and activity annotations
- Used as ZERO-SHOT TEST set (not for training)

File structure:
  S{NNN}.csv  (NNN = subject ID, e.g., S006.csv)
  Columns: timestamp, back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label

Note: HARTH is accelerometer-only (no gyroscope). Gyro channels will be
zero-padded by the LIMU-BERT pipeline, similar to UniMiB SHAR and RealWorld.

Reference:
Logacjov et al., "HARTH - A Human Activity Recognition Dataset for Machine
Learning", Sensors 2021, 21(21), 7261. DOI: 10.3390/s21217261
https://archive.ics.uci.edu/dataset/779/harth
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Run as `python -m data.datasets.harth.convert` from the repo root; repo root on path for shared imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# Activity mapping (label codes to standardized names)
# HARTH uses integer labels 1-12 (some variants use string labels)
# Real released HARTH integer codes (Logacjov et al. 2021, Table A1). Codes 9-12
# do NOT exist in the data (previously fabricated); cycling uses 13/14 (active) and
# 130/140 (inactive, merged into the active class). No transport code exists.
ACTIVITIES_BY_INT = {
    1: "walking",
    2: "running",
    3: "shuffling",
    4: "stairs_up",     # HARTH "stairs (ascending)"
    5: "stairs_down",   # HARTH "stairs (descending)"
    6: "standing",
    7: "sitting",
    8: "lying",
    13: "cycling_sit",     # HARTH "cycling (sit)"
    14: "cycling_stand",   # HARTH "cycling (stand)"
    130: "cycling_sit",    # HARTH "cycling (sit, inactive)" -> merged
    140: "cycling_stand",  # HARTH "cycling (stand, inactive)" -> merged
}

# Some HARTH variants use string labels directly
ACTIVITIES_BY_STR = {
    "walking": "walking",
    "running": "running",
    "shuffling": "shuffling",
    "stairs (up)": "stairs_up",
    "stairs (down)": "stairs_down",
    "standing": "standing",
    "sitting": "sitting",
    "lying": "lying",
    "cycling (sit)": "cycling_sit",
    "cycling (stand)": "cycling_stand",
    "cycling (sit, inactive)": "cycling_sit",
    "cycling (stand, inactive)": "cycling_stand",
    # Alternative string formats (no transport code exists in released HARTH)
    "stairs_up": "stairs_up",
    "stairs_down": "stairs_down",
    "cycling_sit": "cycling_sit",
    "cycling_stand": "cycling_stand",
}

# Paths (new repo layout: this converter lives in data/datasets/harth/)
DS_DIR = Path(__file__).resolve().parent
RAW_DIR = DS_DIR / "downloads"
OUTPUT_DIR = DS_DIR

# Sampling rate
SAMPLE_RATE = 50.0

# Column names for standardized output
OUTPUT_COLUMNS = [
    "back_acc_x", "back_acc_y", "back_acc_z",
    "thigh_acc_x", "thigh_acc_y", "thigh_acc_z",
]


def _timestamp_seconds(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Return relative timestamp seconds from the raw HARTH timestamp column."""
    candidates = [c for c in df.columns if "time" in c]
    if not candidates:
        return None

    raw = df[candidates[0]]
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().mean() > 0.95:
        values = numeric.to_numpy(dtype=np.float64)
        values = values - values[0]
        return values if np.isfinite(values).all() else None

    timestamps = pd.to_datetime(raw, errors="coerce", utc=True)
    if timestamps.notna().mean() <= 0.95:
        return None
    values = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
    return values if np.isfinite(values).all() else None


def _infer_sample_rate(timestamp_sec: Optional[np.ndarray]) -> float:
    """Infer native rate from median raw timestamp spacing."""
    if timestamp_sec is None or len(timestamp_sec) < 3:
        return SAMPLE_RATE
    diffs = np.diff(timestamp_sec)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) < 3:
        return SAMPLE_RATE
    return float(1.0 / np.median(diffs))


def _resample_to_target_rate(df: pd.DataFrame, native_rate: float, subject_name: str) -> pd.DataFrame:
    """Resample a uniformly sampled subject stream to HARTH's canonical 50 Hz."""
    if native_rate <= 0:
        native_rate = SAMPLE_RATE

    source_t = np.arange(len(df), dtype=np.float64) / native_rate
    if abs(native_rate - SAMPLE_RATE) <= 0.5:
        out = df.copy()
        out["timestamp_sec"] = np.arange(len(out), dtype=np.float64) / SAMPLE_RATE
        return out

    target_t = np.arange(0.0, source_t[-1] + (0.5 / SAMPLE_RATE), 1.0 / SAMPLE_RATE)
    target_t = target_t[target_t <= source_t[-1] + 1e-9]

    out = pd.DataFrame({"timestamp_sec": target_t})
    for col in OUTPUT_COLUMNS:
        out[col] = np.interp(target_t, source_t, df[col].to_numpy(dtype=np.float64))

    label_idx = np.searchsorted(source_t, target_t, side="right") - 1
    label_idx = np.clip(label_idx, 0, len(df) - 1)
    out["activity"] = df["activity"].to_numpy()[label_idx]

    print(f"    Resampled {subject_name}: {native_rate:.2f} Hz -> {SAMPLE_RATE:.1f} Hz "
          f"({len(df)} -> {len(out)} rows)")
    return out


def parse_label(label_value) -> Optional[str]:
    """Convert a label value (int or string) to standardized activity name."""
    if isinstance(label_value, (int, float, np.integer, np.floating)):
        return ACTIVITIES_BY_INT.get(int(label_value))

    label_str = str(label_value).strip().lower()
    return ACTIVITIES_BY_STR.get(label_str)


def load_subject_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Load a subject CSV file with continuous recording and activity labels.

    Expected columns: timestamp, back_x, back_y, back_z,
                      thigh_x, thigh_y, thigh_z, label

    Returns:
        DataFrame with sensor data, activity column, and timestamp_sec.
    """
    try:
        # Try different delimiters
        df = None
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(filepath, sep=sep)
                if len(df.columns) >= 7:
                    break
                df = None
            except Exception:
                continue

        if df is None:
            print(f"    Could not parse {filepath.name}")
            return None

        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        raw_timestamp_sec = _timestamp_seconds(df)
        native_rate = _infer_sample_rate(raw_timestamp_sec)

        # Find sensor columns (back and thigh accelerometers)
        back_cols = []
        thigh_cols = []
        label_col = None

        for col in df.columns:
            if "label" in col:
                label_col = col
            elif "back" in col:
                back_cols.append(col)
            elif "thigh" in col:
                thigh_cols.append(col)

        # Fallback: if no explicit back/thigh, assume column order
        if not back_cols and not thigh_cols and len(df.columns) >= 7:
            # Try positional: skip timestamp, then 3 back, 3 thigh, label
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 6:
                back_cols = numeric_cols[:3]
                thigh_cols = numeric_cols[3:6]

        if not label_col:
            # Try last column as label
            label_col = df.columns[-1]

        if len(back_cols) < 3 or len(thigh_cols) < 3:
            print(f"    Warning: Could not find 6 accelerometer columns in {filepath.name}")
            print(f"    Columns: {list(df.columns)}")
            return None

        # Build output DataFrame
        result = pd.DataFrame()
        result["back_acc_x"] = df[back_cols[0]].values
        result["back_acc_y"] = df[back_cols[1]].values
        result["back_acc_z"] = df[back_cols[2]].values
        result["thigh_acc_x"] = df[thigh_cols[0]].values
        result["thigh_acc_y"] = df[thigh_cols[1]].values
        result["thigh_acc_z"] = df[thigh_cols[2]].values
        result["activity"] = df[label_col].values

        # Interpolate any NaN values in sensor data
        for col in OUTPUT_COLUMNS:
            if result[col].isna().any():
                result[col] = result[col].interpolate(method="linear", limit_direction="both")
                result[col] = result[col].fillna(0)

        return _resample_to_target_rate(result, native_rate, filepath.stem)

    except Exception as e:
        print(f"    Error loading {filepath}: {e}")
        return None


def convert_dataset():
    """Convert HARTH dataset to standardized format."""
    print("=" * 80)
    print("HARTH -> Standardized Format Converter")
    print("=" * 80)
    print("NOTE: This dataset is used for ZERO-SHOT TESTING (not training)")

    # Check input
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data not found at {RAW_DIR}")
        print("Run: python -m data.scripts.download_datasets harth")
        return False

    # Create output directory. Clear any prior conversion first so re-runs don't
    # leave orphan session dirs (e.g. when a subject's window count changes after
    # a resampling fix); the loader indexes labels.json, so orphans are inert but
    # they desync the on-disk count and bloat disk.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions_dir = OUTPUT_DIR / "sessions"
    shutil.rmtree(sessions_dir, ignore_errors=True)
    sessions_dir.mkdir(exist_ok=True)

    # Find subject CSV files
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        # Try subdirectories
        csv_files = sorted(RAW_DIR.glob("**/*.csv"))

    if not csv_files:
        print("ERROR: No CSV files found")
        return False

    print(f"\nFound {len(csv_files)} subject files")

    all_labels = {}
    session_count = 0
    skipped_count = 0

    for csv_file in csv_files:
        # Extract subject ID from filename
        subject_id = csv_file.stem  # e.g., "S006"
        print(f"\n  Processing {subject_id}...")

        # Load continuous recording
        df = load_subject_csv(csv_file)
        if df is None or len(df) < 10:
            print(f"    Skipping {subject_id}: no valid data")
            skipped_count += 1
            continue

        # Segment by activity (continuous recording with activity labels)
        # Find contiguous segments of same activity
        activities = df["activity"].values
        segment_starts = [0]
        for i in range(1, len(activities)):
            if activities[i] != activities[i - 1]:
                segment_starts.append(i)
        segment_starts.append(len(activities))

        subject_sessions = 0
        for seg_idx in range(len(segment_starts) - 1):
            start = segment_starts[seg_idx]
            end = segment_starts[seg_idx + 1]

            # Parse activity label
            raw_label = activities[start]
            activity_name = parse_label(raw_label)
            if activity_name is None:
                skipped_count += 1
                continue

            # Extract segment data (sensor columns + timestamp)
            segment_df = df.iloc[start:end][["timestamp_sec"] + OUTPUT_COLUMNS].copy()
            segment_df = segment_df.reset_index(drop=True)

            # Reset timestamp to start at 0
            segment_df["timestamp_sec"] = np.arange(len(segment_df)) / SAMPLE_RATE

            if len(segment_df) < int(SAMPLE_RATE * 1.0):
                # Skip segments shorter than 1 second
                skipped_count += 1
                continue

            # Subject id for subject-disjoint splits (read by build_grids.iter_sessions).
            segment_df["subject"] = subject_id

            # Save the whole continuous single-activity segment as ONE session; build_grids does the
            # fixed 6 s windowing (avoids double-windowing).
            session_id = f"harth_{subject_id}_{activity_name}_{seg_idx:04d}"
            session_dir = sessions_dir / session_id
            session_dir.mkdir(exist_ok=True)
            segment_df.to_parquet(session_dir / "data.parquet", index=False)

            all_labels[session_id] = [activity_name]
            session_count += 1
            subject_sessions += 1

        print(f"    Created {subject_sessions} sessions")

    if not all_labels:
        print("\nNo sessions created. Check the raw data format.")
        return False

    # Save labels.json
    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump(all_labels, f, indent=2)
    print(f"\nCreated labels.json ({len(all_labels)} sessions)")

    # Create manifest
    create_manifest()

    # Activity distribution
    activity_counts = {}
    for labels in all_labels.values():
        for label in labels:
            activity_counts[label] = activity_counts.get(label, 0) + 1

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {session_count} sessions converted")
    print(f"  - {skipped_count} segments skipped")
    print(f"  - {len(activity_counts)} unique activities")
    print(f"  - {SAMPLE_RATE} Hz sampling rate")

    print("\nActivity distribution:")
    for activity, count in sorted(activity_counts.items()):
        print(f"  {activity}: {count}")

    # Generate debug visualizations
    try:
        from data.scripts.debug.visualization_utils import generate_debug_visualizations
        generate_debug_visualizations(OUTPUT_DIR)
    except ImportError:
        pass

    return True


def create_manifest():
    """Create manifest.json."""
    manifest = {
        "dataset_name": "HARTH",
        "description": (
            "Human Activity Recognition Trondheim. 22 subjects performing 12 daily "
            "activities recorded in free-living conditions. Two 3-axis accelerometers "
            "(lower back and right thigh) at 50 Hz. Accelerometer only."
        ),
        "source": "https://archive.ics.uci.edu/dataset/779/harth",
        "num_subjects": 22,
        "channels": [
            {"name": "back_acc_x", "description": "Back accelerometer X-axis", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "back_acc_y", "description": "Back accelerometer Y-axis", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "back_acc_z", "description": "Back accelerometer Z-axis", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "thigh_acc_x", "description": "Thigh accelerometer X-axis", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "thigh_acc_y", "description": "Thigh accelerometer Y-axis", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "thigh_acc_z", "description": "Thigh accelerometer Z-axis", "sampling_rate_hz": SAMPLE_RATE},
        ],
    }

    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Created manifest.json")


def main():
    success = convert_dataset()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
