"""Convert TNDA-HAR into HALO session format.

TNDA-HAR (Wu et al.), 23 subjects, 8 activities (3 static + 5 periodic), multi-IMU (wrist / back /
ankle etc.), 50 Hz, tri-axial accelerometer + gyroscope. We keep the WRIST stream only
(tnda_har/watch_wrist).

PROVENANCE / FALLBACK: the original raw dataset is behind an IEEE DataPort sign-in
(https://ieee-dataport.org/open-access/tnda-har-0, DOI 10.21227/4epb-pg26), so this converter uses
the UniMTS-released preprocessed bundle (HuggingFace xiyuanz/UniMTS -> UniMTS_data/TNDA-HAR/) as the
source. In that bundle each sample is a native-rate (50 Hz) 10 s segment; the 30 channels are 5 IMU
positions x (acc, gyro). Per UniMTS's skeleton assignment the RIGHT-WRIST IMU is columns 12:18
(acc 12:15, gyro 15:18). Accelerometer is total acceleration in m/s^2 (gravity present, |acc| ~ 9.8);
gyro is in rad/s. NOTE: gravity IS present (accel_units rescales m/s^2 -> g downstream).

The bundle ships no per-sample subject id, so `subject` records the UniMTS train/test partition
(subject-disjoint in UniMTS) rather than individual participants. build_grids does the 6 s windowing.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DS_DIR = Path(__file__).resolve().parent
BUNDLE = DS_DIR / "downloads" / "bundle" / "UniMTS_data" / "TNDA-HAR"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

SAMPLING_RATE = 50  # Hz (native rate; UniMTS resamples to 20 Hz only at load time, not in the bundle)
WRIST_COLS = slice(12, 18)  # right-wrist IMU: acc 12:15, gyro 15:18

# UniMTS TNDA-HAR.json label_dictionary index -> native label (cleaned/underscored).
LABELS = {
    0: "sitting",
    1: "standing",
    2: "lying_down",
    3: "ascending_stairs",
    4: "descending_stairs",
    5: "riding",
    6: "walking",
    7: "jogging",
}


def main():
    if not BUNDLE.exists():
        print(f"ERROR: UniMTS bundle not found at {BUNDLE}")
        print("Download UniMTS_data.zip from https://huggingface.co/xiyuanz/UniMTS and extract "
              "UniMTS_data/TNDA-HAR/ into downloads/bundle/.")
        return 1

    sessions_dir = DS_DIR / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    labels = {}
    n_sessions = 0
    for split in ("train", "test"):
        X = np.load(BUNDLE / f"X_{split}.npy")            # (N, 500, 30)
        y = np.load(BUNDLE / f"y_{split}.npy").astype(int)  # (N,)
        wrist = X[:, :, WRIST_COLS].astype(np.float64)     # (N, 500, 6): acc(0:3) + gyro(3:6)
        for i in range(len(X)):
            cls = int(y[i])
            if cls not in LABELS:
                continue
            seg = wrist[i]
            n = len(seg)
            out = pd.DataFrame({
                "timestamp_sec": np.arange(n) / SAMPLING_RATE,
                "acc_x": seg[:, 0], "acc_y": seg[:, 1], "acc_z": seg[:, 2],
                "gyro_x": seg[:, 3], "gyro_y": seg[:, 4], "gyro_z": seg[:, 5],
                # The UniMTS bundle has NO per-sample subject id (23 real participants lost). Recording
                # the train/test split tag as `subject` gave a fake 2-subject cohort, so the
                # subject-bootstrap CI printed a garbage near-degenerate interval as if legit. Use a
                # single sentinel instead → scoring's <2-subject guard flags the CI as degenerate
                # (point estimate is unaffected; it just isn't given a false subject-stratified CI).
                "subject": "unknown",
            })
            session_id = f"{split}_{i:05d}_{LABELS[cls]}"
            (sessions_dir / session_id).mkdir(parents=True, exist_ok=True)
            out.to_parquet(sessions_dir / session_id / "data.parquet", index=False)
            labels[session_id] = [LABELS[cls]]
            n_sessions += 1

    (DS_DIR / "labels.json").write_text(json.dumps(labels, indent=2))
    acts = {}
    for v in labels.values():
        acts[v[0]] = acts.get(v[0], 0) + 1
    print(f"TNDA-HAR: {n_sessions} sessions, {len(acts)} activities")
    for a, c in sorted(acts.items()):
        print(f"  {a:20s}: {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
