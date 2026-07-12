"""
Master pipeline script to download and convert all datasets.

Runs the complete pipeline:
1. Download raw data from sources
2. Convert to standardized format (parquet)
3. Generate minimal manifests
4. Validate output

Usage:
    python datascripts/setup_all_ts_datasets.py [dataset_name]

    Without arguments: process all datasets
    With dataset_name: process only that dataset

Available datasets:
    - uci_har (6 activities, 6 channels, 50Hz)
    - pamap2 (12 activities, 27 channels, 100Hz)
    - mhealth (12 activities, 6 channels, 50Hz)
    - wisdm (18 activities, 6 channels, 20Hz)
    - unimib_shar (9 activities, 3 channels ACC ONLY, 50Hz) - uses Kaggle API
    - hhar (6 activities, 6 channels, variable Hz)
    - mobiact (13 activities, 6 channels, 50Hz) - uses Kaggle API (MobiFall)
    - realworld (8 activities, 9 channels, 50Hz) - waist position only
    - dsads (19 activities, 9 channels, 25Hz) - torso position only
    - recgym (11 gym exercises, 6 channels, 20Hz) - uses Kaggle API, wrist position
    - capture24 (10 activities, 3 channels ACC ONLY, 100Hz) - TRAIN, 151-subject free-living wrist
    - hapt (12 activities incl. transitions, 6 channels, 50Hz)
    - kuhar (18 activities, 6 channels, 100Hz) - uses Kaggle API
    - shoaib (7 activities, 45 channels, 50Hz) - ZERO-SHOT TEST, manual download
    - opportunity (4 locomotion, 30 channels, 30Hz) - manual download
    - harth (12 activities, 6 channels, 50Hz) - ZERO-SHOT TEST, manual download
    - inclusivehar (6 activities incl. ramp_ascent/ramp_descent, 6 channels, 50Hz) - ZERO-SHOT TEST, iPhone iOS CoreMotion
"""

import sys
import subprocess
from pathlib import Path


DATASETS = {
    "uci_har": {
        "name": "UCI HAR",
        "download_script": "datascripts/shared/download_all_datasets.py",
        "convert_script": "datascripts/uci_har/convert.py",
        "requires_manual": False
    },
    "pamap2": {
        "name": "PAMAP2",
        "download_script": "datascripts/shared/download_all_datasets.py",
        "convert_script": "datascripts/pamap2/convert.py",
        "requires_manual": False
    },
    "mhealth": {
        "name": "MHEALTH",
        "download_script": "datascripts/shared/download_all_datasets.py",
        "convert_script": "datascripts/mhealth/convert.py",
        "requires_manual": False
    },
    "wisdm": {
        "name": "WISDM",
        "download_script": "datascripts/shared/download_all_datasets.py",
        "convert_script": "datascripts/wisdm/convert.py",
        "requires_manual": False
    },
    "unimib_shar": {
        "name": "UniMiB SHAR",
        "download_script": "datascripts/shared/download_all_datasets.py",
        "convert_script": "datascripts/unimib_shar/convert.py",
        "requires_manual": False,  # Uses Kaggle API
        "note": "Requires: pip install kaggle && kaggle API key in ~/.kaggle/kaggle.json"
    },
    "hhar": {
        "name": "HHAR",
        "download_script": "datascripts/shared/download_all_datasets.py",
        "convert_script": "datascripts/hhar/convert.py",
        "requires_manual": False
    },
    "mobiact": {
        "name": "MobiFall",
        "download_script": "datascripts/mobiact/download.py",
        "convert_script": "datascripts/mobiact/convert.py",
        "requires_manual": False,
        "note": "Uses Kaggle API (MobiFall v2.0). Requires: pip install kaggle && kaggle API key"
    },
    "realworld": {
        "name": "RealWorld HAR",
        "download_script": "datascripts/realworld/download.py",
        "convert_script": "datascripts/realworld/convert.py",
        "requires_manual": False
    },
    "dsads": {
        "name": "DSADS",
        "download_script": "datascripts/dsads/download.py",
        "convert_script": "datascripts/dsads/convert.py",
        "requires_manual": False
    },
    "recgym": {
        "name": "RecGym",
        "download_script": "datascripts/recgym/download.py",
        "convert_script": "datascripts/recgym/convert.py",
        "requires_manual": False,
        "note": "Uses Kaggle API. Requires: pip install kaggle && kaggle API key"
    },
    "hapt": {
        "name": "HAPT",
        "download_script": "datascripts/hapt/download.py",
        "convert_script": "datascripts/hapt/convert.py",
        "requires_manual": False,
        "note": "UCI HAR Postural Transitions - 30 subjects, 12 activities (6 basic + 6 transitions), 50Hz"
    },
    "kuhar": {
        "name": "KU-HAR",
        "download_script": "datascripts/kuhar/download.py",
        "convert_script": "datascripts/kuhar/convert.py",
        "requires_manual": False,
        "note": "Uses Kaggle API. 90 subjects, 18 activities, 100Hz"
    },
    # New datasets for baseline comparison
    "shoaib": {
        "name": "Shoaib",
        "download_script": None,  # Manual download from UTwente
        "convert_script": "datascripts/shoaib/convert.py",
        "requires_manual": True,
        "note": "ZERO-SHOT TEST SET. 10 subjects, 7 activities, 5 body positions, 50Hz. Download from UTwente."
    },
    "opportunity": {
        "name": "OPPORTUNITY",
        "download_script": None,  # Manual download from UCI
        "convert_script": "datascripts/opportunity/convert.py",
        "requires_manual": True,
        "note": "4 subjects, 4 locomotion activities, 5 body IMUs, 30Hz. Download from UCI ML Repository."
    },
    "harth": {
        "name": "HARTH",
        "download_script": None,  # Manual download from UCI
        "convert_script": "datascripts/harth/convert.py",
        "requires_manual": True,
        "note": "ZERO-SHOT TEST SET. 22 subjects, 12 activities, 2 accelerometers (back+thigh), 50Hz. Download from UCI."
    },
    # V2 adds (2026-07): broaden toward real-world phone/watch HAR
    "inclusivehar": {
        "name": "InclusiveHAR",
        "download_script": None,  # Manual: single CSV from Mendeley r78dn3f6nc (v4)
        "convert_script": "datascripts/inclusivehar/convert.py",
        "requires_manual": True,
        "note": "ZERO-SHOT TEST SET. 20 subjects (10 able-bodied, 10 with disabilities), 6 activities, iPhone iOS CoreMotion, 50Hz. Download InclusiveHAR_dataset_v2.csv from Mendeley to data/raw/inclusivehar/InclusiveHAR.csv"
    },
    "capture24": {
        "name": "CAPTURE-24",
        "download_script": None,  # Manual: capture24.zip (CC BY) from Oxford ORA
        "convert_script": "datascripts/capture24/convert.py",
        "requires_manual": True,
        "note": "TRAIN corpus. 151 subjects, ~2500h free-living wrist accel (no gyro), 100Hz, WillettsSpecific2018 10-class labels. Download capture24.zip from Oxford ORA, unzip under data/raw/capture24/"
    },
    "extrasensory": {
        "name": "ExtraSensory",
        "download_script": None,  # Manual: raw_acc.zip + features_labels.zip from extrasensory.ucsd.edu
        "convert_script": "datascripts/extrasensory/convert.py",
        "requires_manual": True,
        "note": "ZERO-SHOT TEST SET. 60 users, in-the-wild free-living phone accel (~40Hz, in g), multi-label projected to single movement primitive. Download raw_acc + features_labels from extrasensory.ucsd.edu into data/raw/extrasensory/{raw_acc,labels}/"
    },
}


def run_command(command: list, description: str):
    """Run a command and return success status."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Command failed with exit code {e.returncode}")
        return False


def process_dataset(dataset_key: str):
    """Process one dataset through the complete pipeline."""
    if dataset_key not in DATASETS:
        print(f"ERROR: Unknown dataset: {dataset_key}")
        return False

    dataset = DATASETS[dataset_key]

    print(f"\n{'=' * 80}")
    print(f"Processing: {dataset['name']}")
    print(f"{'=' * 80}")

    # Step 1: Download (if not manual)
    if dataset['requires_manual']:
        print(f"\n⚠ {dataset['name']} requires manual download")
        print("Please ensure data is available before conversion")
    else:
        download_success = run_command(
            [sys.executable, dataset['download_script'], dataset_key],
            f"Downloading {dataset['name']}"
        )
        if not download_success:
            print(f"✗ Failed to download {dataset['name']}")
            return False

    # Step 2: Convert
    convert_success = run_command(
        [sys.executable, dataset['convert_script']],
        f"Converting {dataset['name']} to standardized format"
    )

    if not convert_success:
        print(f"✗ Failed to convert {dataset['name']}")
        return False

    # Step 3: Validate output
    output_dir = Path(f"data/{dataset_key}")
    manifest_path = output_dir / "manifest.json"
    labels_path = output_dir / "labels.json"
    sessions_dir = output_dir / "sessions"

    if manifest_path.exists() and labels_path.exists() and sessions_dir.exists():
        # Count session directories (any subdirectory counts as a session)
        num_sessions = len([d for d in sessions_dir.iterdir() if d.is_dir()])
        print(f"\n✓ {dataset['name']} completed successfully")
        print(f"    Output: {output_dir}")
        print(f"    Sessions: {num_sessions}")
        return True
    else:
        print(f"\n✗ {dataset['name']} validation failed")
        missing = []
        if not manifest_path.exists():
            missing.append("manifest.json")
        if not labels_path.exists():
            missing.append("labels.json")
        if not sessions_dir.exists():
            missing.append("sessions/")
        print(f"    Missing in {output_dir}: {', '.join(missing)}")
        return False


def main():
    """Main pipeline execution."""
    print("=" * 80)
    print("Dataset Setup Pipeline")
    print("=" * 80)

    # Determine which datasets to process
    if len(sys.argv) > 1:
        datasets_to_process = sys.argv[1:]  # All arguments after script name
    else:
        datasets_to_process = list(DATASETS.keys())

    print(f"\nWill process {len(datasets_to_process)} dataset(s):")
    for key in datasets_to_process:
        if key in DATASETS:
            print(f"  - {DATASETS[key]['name']}")
        else:
            print(f"  - {key} (UNKNOWN)")

    # Process each dataset
    results = {}
    for dataset_key in datasets_to_process:
        success = process_dataset(dataset_key)
        results[dataset_key] = success

    # Summary
    print(f"\n{'=' * 80}")
    print("Pipeline Summary")
    print(f"{'=' * 80}")

    for dataset_key, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        dataset_name = DATASETS.get(dataset_key, {}).get('name', dataset_key)
        print(f"  {dataset_name}: {status}")

    total_success = sum(results.values())
    print(f"\nCompleted: {total_success}/{len(results)} datasets")

    # Overall success
    if total_success == len(results):
        print("\n✓ All datasets processed successfully!")
        return 0
    else:
        print(f"\n⚠ {len(results) - total_success} dataset(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
