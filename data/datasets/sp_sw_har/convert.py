"""Convert SP-SW-HAR (Matey-Sanz / GeoTec 2023) to the HALO session format.

Source: github.com/GeoTecINIT/sp-sw-har-dataset (CC-BY-4.0), raw CSVs under
downloads/DATA/s{NN}/s{NN}_{exec}_{sp|sw}.csv with columns
x_acc,y_acc,z_acc,x_gyro,y_gyro,z_gyro,timestamp,label.

  * Two co-located device streams per execution: `_sp` = smartphone (left front pocket,
    orientation-variable) and `_sw` = smartwatch (left wrist). Emitted as SEPARATE streams
    (phone_pocket / watch_wrist), routed by the `_sp`/`_sw` token in the session id.
  * Accelerometer is raw Android TYPE_ACCELEROMETER in **m/s² with gravity present**
    (verified: at-rest |acc| ≈ 9.8) -> divided by 9.81 to canonical **g**. Gyro is rad/s.
  * Each file is a continuous Timed-Up-and-Go sequence with PER-ROW labels. We split into
    contiguous single-activity runs, then slide a FIXED 1.0 s window (100 samples @100 Hz,
    stride 0.5 s) inside each run, emitting each window as its own single-activity session
    with ``pre_windowed: true``. TUG transitions are inherently short — `turning` (median
    1.15 s) and the sit/stand transitions (~1.3 s) — so a 1.0 s window is the natural
    granularity; a 6 s corpus window (or 1.5 s) would discard most `turning` windows. Fixed
    length keeps every emitted session identical so build_grids' forced_window is uniform.
  * ~102.5 Hz native (source `timestamp` in ms, with duplicate timestamps + small gaps) —
    resampled onto a real-time uniform 100 Hz grid so each fixed 100-sample window is honestly
    1.0 s (a synthetic clock would mis-time by ~2.5%). 23 subjects (s01–s23).

Labels -> canonical: SEATED->sitting, WALKING->walking, SITTING_DOWN->sitting_down,
STANDING_UP->standing_up_from_sitting, TURNING->turning (NEW canonical label).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "downloads" / "DATA"
G = 9.81
NATIVE_RATE = 100.0
LABEL_MAP = {
    "SEATED": "sitting",
    "WALKING": "walking",
    "SITTING_DOWN": "sitting_down",
    "STANDING_UP": "standing_up_from_sitting",
    "TURNING": "turning",                      # new canonical label (TUG turn-in-place)
}
CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
DEVICE = {"sp": "phone_pocket", "sw": "watch_wrist"}
WIN = 100      # 1.0 s @100 Hz — fixed window length so every emitted session is identical
STRIDE = 50    # 0.5 s hop (50% overlap) inside each single-activity run
GAP_SPLIT_S = 0.05   # split the source timeline at gaps > 50 ms (~5 native periods) BEFORE
                     # interpolation — never fabricate signal/labels across a real acquisition gap
                     # (the raw stream has small holes up to ~188 ms; a window must not span one)


def create_manifest() -> dict:
    place = {"phone_pocket": "smartphone in the left front trouser pocket (orientation-variable)",
             "watch_wrist": "smartwatch on the left wrist"}
    return {
        "dataset_name": "SP-SW-HAR",
        "description": ("Paired smartphone (front pocket) + smartwatch (wrist) IMU during "
                        "Timed-Up-and-Go, 23 subjects (ages 23-66), STMicro LSM6DSO @~100 Hz. "
                        "Accelerometer total acceleration in g (gravity present); gyroscope rad/s."),
        "channels": [
            {"name": c,
             "description": (f"{'accelerometer' if 'acc' in c else 'gyroscope'} {c[-1]}-axis "
                             f"in {'g units (gravity present)' if 'acc' in c else 'rad/s'}"),
             "sampling_rate_hz": NATIVE_RATE}
            for c in CHANNELS
        ],
    }


def main() -> None:
    sessions_dir = HERE / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    sessions_dir.mkdir(parents=True)

    labels: dict[str, list[str]] = {}
    n_seg = 0
    subj_dirs = sorted(d for d in RAW.iterdir() if d.is_dir() and d.name.startswith("s"))
    for sdir in subj_dirs:
        subject = sdir.name                                    # s01 ... s23
        for csv in sorted(sdir.glob(f"{subject}_*_s[pw].csv")):
            stem = csv.stem                                    # s01_01_sp
            device = stem.rsplit("_", 1)[1]                    # sp | sw
            df = pd.read_csv(csv)
            if df.empty:
                continue
            acc = df[["x_acc", "y_acc", "z_acc"]].to_numpy(dtype=np.float32) / G
            gyro = df[["x_gyro", "y_gyro", "z_gyro"]].to_numpy(dtype=np.float32)
            raw_lab = df["label"].to_numpy()
            # Resample onto a real-time uniform 100 Hz grid using the source `timestamp` (ms):
            # the true rate is ~102.5 Hz with duplicate timestamps + small gaps, so a synthetic
            # 100 Hz clock mis-times windows by ~2.5%. Sort, drop duplicate timestamps, interpolate.
            t = df["timestamp"].to_numpy(np.float64) / 1000.0      # ms -> s
            order = np.argsort(t, kind="stable")
            t, acc, gyro, raw_lab = t[order], acc[order], gyro[order], raw_lab[order]
            keep = np.concatenate([[True], np.diff(t) > 0])
            t, acc, gyro, raw_lab = t[keep], acc[keep], gyro[keep], raw_lab[keep]
            if len(t) < 2:
                continue
            exec_id = stem.split("_", 1)[1].rsplit("_", 1)[0]      # "01" from s01_01_sp
            # Split the (sorted, deduped) source timeline at acquisition gaps > GAP_SPLIT_S BEFORE
            # interpolation, so no interpolated sample/label ever spans a real hole. Each contiguous
            # segment is resampled onto its OWN uniform 100 Hz grid and windowed independently.
            gap = np.flatnonzero(np.diff(t) > GAP_SPLIT_S) + 1
            seg_bounds = np.concatenate([[0], gap, [len(t)]])
            for gi in range(len(seg_bounds) - 1):
                a0, a1 = int(seg_bounds[gi]), int(seg_bounds[gi + 1])
                ts = t[a0:a1]
                if a1 - a0 < 2 or (ts[-1] - ts[0]) < WIN / NATIVE_RATE:
                    continue                                        # too short to hold one window
                grid = np.arange(ts[0], ts[-1], 1.0 / NATIVE_RATE)
                acc_g = np.stack([np.interp(grid, ts, acc[a0:a1, i]) for i in range(3)],
                                 axis=1).astype(np.float32)
                gyro_g = np.stack([np.interp(grid, ts, gyro[a0:a1, i]) for i in range(3)],
                                  axis=1).astype(np.float32)
                nn = np.clip(np.searchsorted(ts, grid), 1, len(ts) - 1)   # nearest-source label
                nn = np.where(grid - ts[nn - 1] <= ts[nn] - grid, nn - 1, nn)
                lab_g = raw_lab[a0:a1][nn]
                # split into contiguous same-label runs, then slide fixed WIN windows inside each run
                change = np.flatnonzero(lab_g[1:] != lab_g[:-1]) + 1
                bounds = np.concatenate([[0], change, [len(lab_g)]])
                for k in range(len(bounds) - 1):
                    lo, hi = int(bounds[k]), int(bounds[k + 1])
                    canon = LABEL_MAP.get(str(lab_g[lo]))
                    if canon is None:
                        continue
                    for w, start in enumerate(range(lo, hi - WIN + 1, STRIDE)):
                        sid = f"{subject}_{exec_id}_{device}_g{gi}_seg{k}_w{w}"
                        # sid e.g. s01_01_sp_g0_seg0_w0 -> subject prefix "s01", stream token "_sp_"
                        out = pd.DataFrame({
                            "timestamp_sec": np.arange(WIN, dtype=np.float64) / NATIVE_RATE,
                            **{f"acc_{a}": acc_g[start:start + WIN, i] for i, a in enumerate("xyz")},
                            **{f"gyro_{a}": gyro_g[start:start + WIN, i] for i, a in enumerate("xyz")},
                            "subject": subject,
                        })
                        seg_dir = sessions_dir / sid
                        seg_dir.mkdir()
                        out.to_parquet(seg_dir / "data.parquet", index=False)
                        labels[sid] = [canon]
                        n_seg += 1

    (HERE / "labels.json").write_text(json.dumps(labels, indent=2))
    (HERE / "metadata.json").write_text(json.dumps(
        {"dataset": "sp_sw_har", "sampling_rate_hz": NATIVE_RATE, "pre_windowed": True}, indent=2))
    (HERE / "manifest.json").write_text(json.dumps(create_manifest(), indent=2))
    from collections import Counter
    dist = Counter(v[0] for v in labels.values())
    print(f"sp_sw_har: {n_seg} sessions ({sum('_sp_' in s for s in labels)} phone / "
          f"{sum('_sw_' in s for s in labels)} watch), labels {dict(dist)}")


if __name__ == "__main__":
    main()
