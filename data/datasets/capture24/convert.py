"""
Convert CAPTURE-24 to the new HALO raw-session layout.

Input  (this dataset dir): downloads/capture24/
  - P001.csv.gz ... P151.csv.gz   (columns: time, x, y, z, annotation)
  - annotation-label-dictionary.csv
  - metadata.csv                  (pid, age, sex)
Output (this dataset dir):
  - sessions/<session_id>/data.parquet : timestamp_sec + acc_x/y/z + subject
  - labels.json                        : {session_id: [activity_name]}
  - manifest.json / metadata.json      : informational + native rate for build_grids

CAPTURE-24 Dataset Info:
- 151 participants, ~24 h each of FREE-LIVING wear (~2,500 annotated hours total).
- Axivity AX3 worn on the dominant WRIST; triaxial accelerometer only (NO gyro).
- 100 Hz. Raw acceleration in g, INCLUDES gravity (unlike iOS userAcceleration).
- Labels: camera + sleep-diary free-text annotations, mapped through the official
  annotation-label-dictionary.csv WillettsSpecific2018 scheme to 10 activity classes:
  bicycling, household-chores, manual-work, mixed-activity, sitting, sleep, sports,
  standing, vehicle, walking.
- Large, real-world, wrist-worn free-living corpus — a TRAIN dataset that broadens the
  model beyond scripted lab protocols toward the distribution a phone/watch sees daily.

Segmentation (raw sessions, NO pre-windowing):
Each subject is one continuous multi-hour 100 Hz recording whose annotation changes over
time. We split it into RAW sessions by CONTIGUOUS RUNS of the same WillettsSpecific2018
label. Unlabelled / undictionaried spans (NaN after mapping) break runs and are dropped.
Each contiguous run is saved as one session parquet. build_grids does the fixed 6 s
windowing downstream — we do NOT window here and do NOT call create_variable_windows.
Runs shorter than one 6 s window can never yield a grid window, so they are skipped.

Accelerometer is already in g (capture24 is registered native-g in accel_units); it is
NOT rescaled here. The watch_wrist deployment stream is accelerometer-only; the harmonised
grid zero-pads + masks the absent gyroscope half.

SCALE: the full corpus is ~2,500 annotated hours -> on the order of a million 6 s windows.
Converting to sessions is cheap (they live on disk), but do NOT run
`build_grids --dataset capture24` on the full set: build_grids loads every session frame
into RAM at once and would OOM. Subsample sessions/windows before materialising a grid.

License: CC BY 4.0. doi:10.5287/bodleian:NGx0JOMP5
Reference: Chan Chang et al., "CAPTURE-24: A large dataset of wrist-worn activity tracker
data collected in the wild for human activity recognition".
https://github.com/OxWearables/capture24
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Repo root on path for shared imports (kept for parity with the other converters).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

DS_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = DS_DIR / "downloads" / "capture24"
OUTPUT_DIR = DS_DIR
SESSIONS_DIR = OUTPUT_DIR / "sessions"

SAMPLE_RATE = 100.0

# Label scheme column in annotation-label-dictionary.csv used as the activity label.
LABEL_SCHEME = "label:WillettsSpecific2018"

# WillettsSpecific2018 value -> standardized HALO label string (underscore convention).
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

# Drop contiguous runs shorter than one 6 s grid window: build_grids' fixed_windows yields
# zero windows for a session shorter than the window, so such runs are dead weight on disk.
MIN_SEG_SEC = 6.0

# Sentinel that stands in for unlabelled/undictionaried samples so a contiguous NaN span
# collapses to a single (skipped) run instead of one boundary per sample.
_UNLABELLED = "\x00unlabelled"


def find_dictionary() -> Path:
    hits = sorted(DOWNLOADS_DIR.glob("**/annotation-label-dictionary.csv"))
    if not hits:
        raise FileNotFoundError(
            f"annotation-label-dictionary.csv not found under {DOWNLOADS_DIR}")
    return hits[0]


def build_ann2label() -> dict:
    """annotation string -> standardized HALO label (or NaN if the WillettsSpecific2018
    value is unknown). Preserves the original WillettsSpecific2018 -> LABEL_MAP logic."""
    dic = pd.read_csv(find_dictionary())
    if LABEL_SCHEME not in dic.columns:
        raise KeyError(f"{LABEL_SCHEME!r} not in dictionary columns {list(dic.columns)}")
    return dict(zip(dic["annotation"], dic[LABEL_SCHEME].map(LABEL_MAP)))


def convert_dataset(limit: Optional[int] = None) -> bool:
    print("=" * 80)
    print("CAPTURE-24 -> HALO raw sessions (dominant wrist / watch_wrist)")
    print("=" * 80)
    print("NOTE: large free-living wrist-accel TRAIN corpus (accelerometer only, in g).")

    if not DOWNLOADS_DIR.exists():
        print(f"ERROR: raw data not found at {DOWNLOADS_DIR}")
        print("Download capture24.zip (CC BY) from Oxford ORA and unzip so that")
        print(f"  P*.csv.gz live under {DOWNLOADS_DIR}/")
        return False

    pid_files = sorted(DOWNLOADS_DIR.glob("**/P*.csv.gz"))
    if not pid_files:
        print(f"ERROR: no P*.csv.gz found under {DOWNLOADS_DIR}")
        return False
    if limit is not None:
        pid_files = pid_files[:limit]
    print(f"Found {len(pid_files)} participant files"
          + (f" (limited to first {limit})" if limit is not None else ""))

    ann2label = build_ann2label()

    if SESSIONS_DIR.exists():
        shutil.rmtree(SESSIONS_DIR)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    all_labels = {}
    session_count = 0
    label_session_counts = defaultdict(int)
    label_second_counts = defaultdict(float)
    unmapped_fracs = []
    min_seg_samples = int(round(MIN_SEG_SEC * SAMPLE_RATE))

    for pi, pid_file in enumerate(pid_files):
        pid = pid_file.name.split(".")[0]  # 'P001'
        df = pd.read_csv(pid_file, usecols=["x", "y", "z", "annotation"],
                         dtype={"annotation": str}, low_memory=False)

        lab_series = df["annotation"].map(ann2label)   # NaN where unmapped / unlabelled
        unmapped_fracs.append(float(lab_series.isna().mean()))

        # Contiguous runs of the same label. Unlabelled spans collapse to one sentinel run.
        labs = lab_series.fillna(_UNLABELLED).to_numpy(dtype=object)
        change = np.nonzero(labs[1:] != labs[:-1])[0] + 1
        bounds = np.concatenate(([0], change, [len(labs)])).astype(np.int64)

        xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)

        subject_sessions = 0
        for k in range(len(bounds) - 1):
            s, e = int(bounds[k]), int(bounds[k + 1])
            lab = labs[s]
            if lab == _UNLABELLED:
                continue
            if (e - s) < min_seg_samples:
                continue

            frame = pd.DataFrame(xyz[s:e], columns=OUTPUT_COLUMNS)
            frame.insert(0, "timestamp_sec", np.arange(e - s, dtype=np.float64) / SAMPLE_RATE)
            frame["subject"] = pid

            session_id = f"{pid}_{lab}_{k:06d}"
            sdir = SESSIONS_DIR / session_id
            sdir.mkdir(exist_ok=True)
            frame.to_parquet(sdir / "data.parquet", index=False)

            all_labels[session_id] = [lab]
            label_session_counts[lab] += 1
            label_second_counts[lab] += (e - s) / SAMPLE_RATE
            session_count += 1
            subject_sessions += 1

        print(f"  [{pi + 1:3d}/{len(pid_files)}] {pid}: {subject_sessions} sessions "
              f"(unmapped {unmapped_fracs[-1]:.1%}) -> {session_count} total")

    if not all_labels:
        print("\nNo sessions created — check raw data / dictionary.")
        return False

    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump(all_labels, f)

    total_seconds = float(sum(label_second_counts.values()))
    write_manifest(len(pid_files))
    update_metadata(num_sessions=session_count,
                    activities=sorted(label_session_counts.keys()))

    est_windows = int(total_seconds // 6)
    est_grid_gb = est_windows * 360 * 6 * 4 / 1e9  # 6 s @ 60 Hz, 6 harmonised channels, float32

    print(f"\n{'=' * 80}\nConversion complete!\n{'=' * 80}")
    print(f"Output dir            : {OUTPUT_DIR}")
    print(f"Sessions written      : {session_count} across {len(pid_files)} participants")
    print(f"Mean unmapped fraction: {np.mean(unmapped_fracs):.2%} (dropped)")
    print(f"Total labelled seconds: {total_seconds:,.0f} ({total_seconds / 3600:,.1f} h)")
    print("\nSessions per label (label: sessions, hours):")
    for lab in sorted(label_session_counts, key=lambda k: -label_session_counts[k]):
        print(f"  {lab:18s} {label_session_counts[lab]:7d}   {label_second_counts[lab] / 3600:8.1f} h")
    print(f"\nEstimated full-corpus 6 s windows : ~{est_windows:,}")
    print(f"Estimated harmonised grid size    : ~{est_grid_gb:,.1f} GB "
          f"(windows x 360 x 6 x 4 bytes)")
    print("Do NOT run `build_grids --dataset capture24` on the full set (in-RAM OOM).")
    return True


def write_manifest(num_subjects: int) -> None:
    manifest = {
        "dataset_name": "CAPTURE-24",
        "description": (
            "Large free-living wrist-worn accelerometer corpus. 151 participants, ~24 h "
            "each, Axivity AX3 on the dominant wrist at 100 Hz. Raw triaxial acceleration "
            "in g (INCLUDES gravity). Camera+diary annotations mapped to the "
            "WillettsSpecific2018 10-class activity scheme. Accelerometer only (no gyro). "
            "Sessions are raw contiguous-label runs; build_grids does the 6 s windowing."
        ),
        "source": "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001",
        "num_subjects": num_subjects,
        "channels": [
            {"name": "acc_x", "description": "Wrist accelerometer X-axis in g (raw, includes gravity)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_y", "description": "Wrist accelerometer Y-axis in g (raw, includes gravity)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_z", "description": "Wrist accelerometer Z-axis in g (raw, includes gravity)", "sampling_rate_hz": SAMPLE_RATE},
        ],
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def update_metadata(num_sessions: int, activities: list) -> None:
    """Refresh informational fields of metadata.json while preserving split/role fields that
    downstream tooling relies on. build_grids reads `sampling_rate_hz`; never set `pre_windowed`."""
    meta_path = OUTPUT_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    meta["display_name"] = meta.get("display_name", "CAPTURE-24")
    meta["num_sessions"] = num_sessions
    meta["sampling_rate_hz"] = int(SAMPLE_RATE)
    meta["channels"] = list(OUTPUT_COLUMNS)
    meta["core_channels"] = {c: c for c in OUTPUT_COLUMNS}
    meta["extra_channels"] = []
    meta["placement"] = meta.get("placement", "wrist")
    meta["num_subjects"] = 151
    meta["activities"] = activities
    meta["role"] = meta.get("role", "train")
    meta.setdefault("train_subjects", [])
    meta.setdefault("test_subjects", [])
    meta.pop("pre_windowed", None)

    meta_path.write_text(json.dumps(meta, indent=2))


def main() -> int:
    p = argparse.ArgumentParser(description="Convert CAPTURE-24 to HALO raw sessions.")
    p.add_argument("--limit", type=int, default=None,
                   help="Convert only the first N participant files (for quick verification).")
    args = p.parse_args()
    return 0 if convert_dataset(limit=args.limit) else 1


if __name__ == "__main__":
    sys.exit(main())
