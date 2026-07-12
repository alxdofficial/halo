"""
Convert UniMiB SHAR dataset to standardized format.

Input: data/raw/unimib_shar/ (from Kaggle: wangboluo/unimib-shar-dataset)
Output: data/unimib_shar/
  - manifest.json
  - labels.json
  - sessions/session_XXX/data.parquet

UniMiB SHAR: 30 subjects, 17 activities (9 ADL + 8 falls), accelerometer only, 50Hz.
Note: This dataset does NOT have gyroscope data!

The Kaggle version provides CSV files with columns: ID, t, ax, ay, az, label, mag
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


# Activity mapping (1-indexed, 17 activities total)
ACTIVITIES = {
    1: 'standing_up_from_sitting',
    2: 'standing_up_from_laying',
    3: 'walking',
    4: 'running',
    5: 'going_up_stairs',
    6: 'jumping',
    7: 'going_down_stairs',
    8: 'lying_down_from_standing',
    9: 'sitting_down',
    10: 'falling_forward',
    11: 'falling_right',
    12: 'falling_backward',
    13: 'falling_hitting_obstacle',
    14: 'falling_with_protection',
    15: 'falling_backward_sitting',
    16: 'syncope',
    17: 'falling_left'
}

# Paths
RAW_DIR = Path("data/raw/unimib_shar")
OUTPUT_DIR = Path("data/unimib_shar")

# Sampling rate
SAMPLING_RATE_HZ = 50.0


def load_csv_data():
    """Load data from CSV files (Kaggle format)."""
    train_path = RAW_DIR / "unimib_train.csv"
    val_path = RAW_DIR / "unimib_val.csv"
    test_path = RAW_DIR / "unimib_test.csv"

    if not train_path.exists():
        print(f"ERROR: CSV files not found in {RAW_DIR}")
        print("Expected: unimib_train.csv, unimib_val.csv, unimib_test.csv")
        return None

    print("Loading CSV files...")
    dfs = []
    for path, split in [(train_path, 'train'), (val_path, 'val'), (test_path, 'test')]:
        if path.exists():
            df = pd.read_csv(path)
            df['split'] = split
            dfs.append(df)
            print(f"  {split}: {len(df)} rows, {df['ID'].nunique()} sessions")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined)} rows, {combined['ID'].nunique()} sessions")
    return combined


def process_csv_data(df):
    """Process CSV data into sessions."""
    sessions = []
    labels_dict = {}

    # Group by session ID
    unique_ids = df['ID'].unique()
    print(f"\nProcessing {len(unique_ids)} sessions...")

    for session_idx, session_id in enumerate(unique_ids):
        session_df = df[df['ID'] == session_id].sort_values('t')

        # Get activity label (should be constant for a session)
        label = session_df['label'].iloc[0]
        if label not in ACTIVITIES:
            continue

        activity_name = ACTIVITIES[label]

        # Extract sensor data
        session_data = pd.DataFrame({
            'timestamp_sec': np.arange(len(session_df)) / SAMPLING_RATE_HZ,
            'acc_x': session_df['ax'].values,
            'acc_y': session_df['ay'].values,
            'acc_z': session_df['az'].values
        })

        # Drop NaN/Inf values
        session_data = session_data.replace([np.inf, -np.inf], np.nan).dropna()

        if len(session_data) < 10:
            continue

        # Create unique session ID
        new_session_id = f"session_{session_idx:05d}"

        sessions.append({
            'session_id': new_session_id,
            'data': session_data,
            'activity_name': activity_name,
            'duration_sec': len(session_data) / SAMPLING_RATE_HZ
        })

        labels_dict[new_session_id] = [activity_name]

        if (session_idx + 1) % 2000 == 0:
            print(f"  Processed {session_idx + 1}/{len(unique_ids)} sessions...")

    print(f"Created {len(sessions)} valid sessions")
    return sessions, labels_dict


def save_sessions(sessions, labels_dict):
    """Save all sessions to parquet format."""
    sessions_dir = OUTPUT_DIR / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    for session in sessions:
        session_id = session['session_id']
        data = session['data']

        # Save to parquet
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = session_dir / "data.parquet"
        data.to_parquet(parquet_path, index=False)

    # Save labels
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(labels_dict, f, indent=2)

    print(f"\n✓ Created labels: {labels_path}")
    print(f"  Total sessions: {len(labels_dict)}")


def create_manifest():
    """Create manifest.json with channel definitions."""
    manifest = {
        "dataset_name": "UniMiB SHAR",
        "description": "Smartphone accelerometer activity recognition. 30 subjects performing 17 activities (9 ADL + 8 falls) at 50Hz. NOTE: Accelerometer only - no gyroscope data!",
        "channels": [
            {
                "name": "acc_x",
                "description": "Accelerometer X-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "acc_y",
                "description": "Accelerometer Y-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "acc_z",
                "description": "Accelerometer Z-axis",
                "sampling_rate_hz": 50.0
            }
        ]
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Created manifest: {manifest_path}")
    print("  NOTE: This dataset has ONLY accelerometer data (no gyroscope)")


def main():
    """Convert UniMiB SHAR to standardized format."""
    print("=" * 80)
    print("UniMiB SHAR → Standardized Format Converter")
    print("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load CSV data
    df = load_csv_data()
    if df is None:
        return

    # Process into sessions
    sessions, labels_dict = process_csv_data(df)

    if not sessions:
        print("ERROR: No sessions created. Check the data format.")
        return

    # Save sessions
    print("\nSaving sessions...")
    save_sessions(sessions, labels_dict)

    # Create manifest
    create_manifest()

    # Print summary
    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {len(labels_dict)} sessions")
    print(f"  - 3 channels (accelerometer only)")
    print(f"  - 50 Hz sampling rate")

    # Activity distribution
    activities = {}
    for label in labels_dict.values():
        act = label[0]
        activities[act] = activities.get(act, 0) + 1

    print("\nActivity distribution:")
    for act, count in sorted(activities.items()):
        print(f"  {act}: {count}")

    # Generate debug visualizations
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.scripts.visualization_utils import generate_debug_visualizations
        generate_debug_visualizations(OUTPUT_DIR)
    except ImportError as e:
        print(f"\n⚠ Could not generate visualizations: {e}")


if __name__ == "__main__":
    main()
