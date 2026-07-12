"""
Convert CAPTURE-24 dataset to standardized format.

Input:  data/raw/capture24/**/P*.csv.gz  +  annotation-label-dictionary.csv
Output: data/capture24/
  - manifest.json
  - labels.json
  - sessions/session_XXX/data.parquet

CAPTURE-24 Dataset Info:
- 151 participants, ~24 h each of FREE-LIVING wear (2,562 annotated hours total).
- Axivity AX3 worn on the dominant WRIST; triaxial accelerometer only (no gyro).
- 100 Hz. Raw acceleration in g, INCLUDES gravity (unlike iOS userAcceleration).
- Labels: camera + sleep-diary free-text annotations, mapped through the
  official annotation-label-dictionary.csv to the WillettsSpecific2018 scheme
  (10 activity classes: bicycling, household-chores, manual-work, mixed-activity,
  sitting, sleep, sports, standing, vehicle, walking).
- This is a large, real-world, wrist-worn free-living corpus — a TRAIN dataset,
  broadening the model beyond scripted lab protocols toward the distribution a
  phone/watch actually sees in daily life.

Because the recordings are continuous multi-hour streams heavily dominated by
sleep/sitting, we (a) segment by contiguous label, (b) cap each segment to
MAX_SEG_SEC before windowing, and (c) cap windows per (participant, label) to
MAX_WIN_PER_PID_LABEL so no class/subject swamps the corpus. Downstream
subsampling (dataset_config.json) trims further.

License: CC BY 4.0. doi:10.5287/bodleian:NGx0JOMP5
Reference: Chan Chang et al., "CAPTURE-24: A large dataset of wrist-worn
activity tracker data collected in the wild for human activity recognition".
https://github.com/OxWearables/capture24
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.scripts.windowing import create_variable_windows

RAW_DIR = Path("data/raw/capture24")
OUTPUT_DIR = Path("data/capture24")
SAMPLE_RATE = 100.0

# Label scheme used from the annotation dictionary.
LABEL_SCHEME = "label:WillettsSpecific2018"

# WillettsSpecific2018 value -> standardized HALO label string (underscore conv).
LABEL_MAP = {
    "sleep": "sleeping",
    "sitting": "sitting",
    "standing": "standing",
    "walking": "walking",
    "bicycling": "bicycling",
    "vehicle": "vehicle",
    "household-chores": "household_chores",
    "manual-work": "manual_work",
    "mixed-activity": "mixed_activity",
    "sports": "sports",
}

OUTPUT_COLUMNS = ["acc_x", "acc_y", "acc_z"]

# Balancing / bounding knobs.
MAX_SEG_SEC = 120          # cap each contiguous label segment before windowing
MAX_WIN_PER_PID_LABEL = 50  # cap windows per (participant, label)
MIN_SEG_SEC = 3            # ignore contiguous segments shorter than this


def find_dictionary() -> Path:
    hits = list(RAW_DIR.glob("**/annotation-label-dictionary.csv"))
    if not hits:
        raise FileNotFoundError(
            f"annotation-label-dictionary.csv not found under {RAW_DIR}"
        )
    return hits[0]


def convert_dataset() -> bool:
    print("=" * 80)
    print("CAPTURE-24 -> Standardized Format Converter")
    print("=" * 80)
    print("NOTE: large free-living wrist-accel TRAIN corpus")

    if not RAW_DIR.exists():
        print(f"ERROR: raw data not found at {RAW_DIR}")
        print("Download capture24.zip (CC BY) from Oxford ORA:")
        print("  https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001")
        print(f"Unzip so that P*.csv.gz live under {RAW_DIR}/")
        return False

    pid_files = sorted(RAW_DIR.glob("**/P*.csv.gz"))
    if not pid_files:
        print(f"ERROR: no P*.csv.gz found under {RAW_DIR}")
        return False
    print(f"Found {len(pid_files)} participant files")

    dic = pd.read_csv(find_dictionary())
    ann2label = dict(zip(dic["annotation"], dic[LABEL_SCHEME].map(LABEL_MAP)))

    sessions_dir = OUTPUT_DIR / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    all_labels = {}
    session_count = 0
    label_counts = defaultdict(int)
    total_unmapped_frac = []

    for pi, pid_file in enumerate(pid_files):
        pid = pid_file.name.split(".")[0]  # 'P001'
        df = pd.read_csv(pid_file, dtype={"annotation": str}, low_memory=False)
        df["lab"] = df["annotation"].map(ann2label)
        total_unmapped_frac.append(float(df["lab"].isna().mean()))

        # Contiguous label segments (NaN/unmapped act as break points and are skipped).
        labs = df["lab"].values
        change = np.where(labs[1:] != labs[:-1])[0] + 1
        seg_bounds = np.concatenate([[0], change, [len(labs)]])
        segs_by_label = defaultdict(list)
        for k in range(len(seg_bounds) - 1):
            s, e = int(seg_bounds[k]), int(seg_bounds[k + 1])
            lab = labs[s]
            if lab is None or (isinstance(lab, float) and np.isnan(lab)):
                continue
            if (e - s) < MIN_SEG_SEC * SAMPLE_RATE:
                continue
            segs_by_label[lab].append((s, e))

        xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        rng = np.random.RandomState(1000 + pi)

        for lab, segs in segs_by_label.items():
            order = rng.permutation(len(segs))
            count = 0
            for gi in order:
                if count >= MAX_WIN_PER_PID_LABEL:
                    break
                s, e = segs[gi]
                e = min(e, s + int(MAX_SEG_SEC * SAMPLE_RATE))  # cap segment length
                seg = pd.DataFrame(xyz[s:e], columns=OUTPUT_COLUMNS)
                seg.insert(0, "timestamp_sec", np.arange(len(seg)) / SAMPLE_RATE)

                prefix = f"capture24_{pid}_{lab}_{int(gi):04d}"
                windows = create_variable_windows(
                    df=seg, session_prefix=prefix, activity=lab,
                    sample_rate=SAMPLE_RATE, seed=1000 + pi * 1000 + int(gi),
                )
                for window_id, window_df, window_activity in windows:
                    if count >= MAX_WIN_PER_PID_LABEL:
                        break
                    window_df = window_df.copy()
                    window_df["timestamp_sec"] = (
                        window_df["timestamp_sec"] - window_df["timestamp_sec"].iloc[0]
                    )
                    wp = sessions_dir / window_id
                    wp.mkdir(exist_ok=True)
                    window_df.to_parquet(wp / "data.parquet", index=False)
                    all_labels[window_id] = [window_activity]
                    label_counts[window_activity] += 1
                    session_count += 1
                    count += 1

        if (pi + 1) % 10 == 0 or pi == len(pid_files) - 1:
            print(f"  [{pi + 1:3d}/{len(pid_files)}] {pid}: "
                  f"{session_count} windows so far")

    if not all_labels:
        print("\nNo sessions created — check raw data / dictionary.")
        return False

    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump(all_labels, f)
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(create_manifest(len(pid_files)), f, indent=2)

    print(f"\n{'=' * 80}\nConversion complete!\n{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {session_count} sessions across {len(pid_files)} participants")
    print(f"  - mean unmapped-annotation fraction: "
          f"{np.mean(total_unmapped_frac):.2%} (skipped)")
    print("\nActivity distribution:")
    for a, c in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {a:18s} {c}")
    return True


def create_manifest(num_subjects: int) -> dict:
    return {
        "dataset_name": "CAPTURE-24",
        "description": (
            "Large free-living wrist-worn accelerometer corpus. 151 participants, "
            "~24 h each, Axivity AX3 on the dominant wrist at 100 Hz. Raw triaxial "
            "acceleration in g (INCLUDES gravity). Camera+diary annotations mapped "
            "to the WillettsSpecific2018 10-class activity scheme. Accelerometer only."
        ),
        "source": "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001",
        "num_subjects": num_subjects,
        "channels": [
            {"name": "acc_x", "description": "Wrist accelerometer X-axis in g (raw, includes gravity)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_y", "description": "Wrist accelerometer Y-axis in g (raw, includes gravity)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_z", "description": "Wrist accelerometer Z-axis in g (raw, includes gravity)", "sampling_rate_hz": SAMPLE_RATE},
        ],
    }


def main() -> int:
    return 0 if convert_dataset() else 1


if __name__ == "__main__":
    sys.exit(main())
