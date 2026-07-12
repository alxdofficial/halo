"""Convert USC-HAD into the standardized HALO session format.

USC-HAD (USC Human Activity Dataset), Zhang & Sawchuk 2012:
- 14 subjects, 12 activities, 5 trials each (~840 trials total).
- Single MotionNode IMU worn on the FRONT-RIGHT HIP.
- 100 Hz, tri-axial accelerometer (+-6g, unit: g) + gyroscope (+-500 dps, unit: dps).
- Accelerometer is TOTAL acceleration (gravity present): a still window reads |acc| ~ 1 g.

Source: https://sipi.usc.edu/had/  (USC-HAD.zip, .mat files under Subject{1..14}/a{m}t{n}.mat)

Each trial is one continuous single-activity recording -> ONE raw session here (build_grids does
the fixed 6 s windowing). The hip placement maps to deployment stream usc_had/phone_hip.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

DS_DIR = Path(__file__).resolve().parent
RAW_DIR = DS_DIR / "downloads" / "USC-HAD"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root for data.scripts imports

SAMPLING_RATE = 100  # Hz

# activity number (from the a{m}t{n}.mat filename) -> native label (Readme Section 3), underscored.
ACTIVITY_MAP = {
    1: "walking_forward",
    2: "walking_left",
    3: "walking_right",
    4: "walking_upstairs",
    5: "walking_downstairs",
    6: "running_forward",
    7: "jumping_up",
    8: "sitting",
    9: "standing",
    10: "sleeping",
    11: "elevator_up",
    12: "elevator_down",
}

_FNAME = re.compile(r"a(\d+)t(\d+)\.mat", re.IGNORECASE)


def main():
    if not RAW_DIR.exists():
        print(f"ERROR: raw data not found at {RAW_DIR}")
        print("Download USC-HAD.zip from https://sipi.usc.edu/had/ into downloads/ and unzip.")
        return 1

    sessions_dir = DS_DIR / "sessions"
    if sessions_dir.exists():
        import shutil
        shutil.rmtree(sessions_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    labels = {}
    n_sessions = 0
    subjects = set()

    for subj_dir in sorted(RAW_DIR.glob("Subject*"), key=lambda p: int(re.sub(r"\D", "", p.name))):
        subj = int(re.sub(r"\D", "", subj_dir.name))
        for mat_path in sorted(subj_dir.glob("a*t*.mat")):
            m = _FNAME.fullmatch(mat_path.name)
            if not m:
                continue
            act_num, trial = int(m.group(1)), int(m.group(2))
            if act_num not in ACTIVITY_MAP:
                continue
            activity = ACTIVITY_MAP[act_num]

            mat = sio.loadmat(mat_path)
            readings = np.asarray(mat["sensor_readings"], dtype=np.float64)  # (N, 6): acc(g) + gyro(dps)
            if readings.ndim != 2 or readings.shape[1] < 6 or len(readings) < 10:
                continue

            n = len(readings)
            df = pd.DataFrame({
                "timestamp_sec": np.arange(n) / SAMPLING_RATE,
                "acc_x": readings[:, 0], "acc_y": readings[:, 1], "acc_z": readings[:, 2],
                "gyro_x": readings[:, 3], "gyro_y": readings[:, 4], "gyro_z": readings[:, 5],
                "subject": f"s{subj:02d}",
            })

            session_id = f"s{subj:02d}_a{act_num:02d}_t{trial}"
            (sessions_dir / session_id).mkdir(parents=True, exist_ok=True)
            df.to_parquet(sessions_dir / session_id / "data.parquet", index=False)
            labels[session_id] = [activity]
            subjects.add(subj)
            n_sessions += 1

    (DS_DIR / "labels.json").write_text(json.dumps(labels, indent=2))

    acts = {}
    for v in labels.values():
        acts[v[0]] = acts.get(v[0], 0) + 1
    print(f"USC-HAD: {n_sessions} sessions, {len(subjects)} subjects, {len(acts)} activities")
    for a, c in sorted(acts.items()):
        print(f"  {a:22s}: {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
