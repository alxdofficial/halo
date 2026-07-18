"""Convert XRF V2 "Plus" (16-volunteer, 853-sequence) to the HALO session format.

Source: kaggle.com/datasets/airslab2020/xrfv2-multimodal-tal-caption-qa-no-rgb (see fetch.py).
Paper: arXiv 2501.19034 (ACM IMWUT 2025). This is the video-aligned release; we use IMU + AirPods
only. Upgraded from the 3-subject WWADL_open subset to the full 16 volunteers.

SIX IMU streams. The 5-position body IMU order is read from the h5's own `device_order`
(authoritative — the Plus order differs from the old WWADL_open release):
    left_wrist | right_wrist | left_phone(pocket) | right_phone(pocket) | glasses(head)
Body IMU: 50 Hz, acc in g (gravity present, verified |acc|~1.0), gyro deg/s -> rad/s (p99 up to
~580 deg/s). AirPods Pro: 25 Hz, 6 channels = Acceleration (USER acceleration, gravity REMOVED,
verified |acc|~0.035 g) + Rotation (gyro, rad/s). AirPods is upsampled 25->50 Hz (build_grids
applies one dataset-wide rate) and carries gravity_state="removed" in the deployment policy.

Labels: `activitynet_tal_853.json` `database[sample]` -> {volunteer_id, scene, annotations:
[{segment:[start_s,end_s], label(Chinese), label_id}]}. Segments are in SECONDS. We map the
Chinese action name -> canonical (CN2CANON below; the 5 "Walk to X" collapse to `walking`, as the
authors merge them; the 3 posture transitions unify to the corpus-wide standing_up_from_* /
lying_down_from_standing scheme so they are not xrf-only singletons — see canonical_labels.py).

Windowing: pre_windowed=false, 6 s; sub-6 s action segments are dropped (reported at convert time).
Subject = volunteer_id (0..15) -> subject-disjoint split has ~14 train / ~2 val subjects.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "downloads"
DEG2RAD = np.pi / 180.0
IMU_RATE = 50.0
AIRPODS_RATE = 25.0

# Plus device_order token -> (HALO stream token, unused). left/right_phone are pocket phones.
DEVICE_TOKEN = {"left_wrist": "left_wrist", "right_wrist": "right_wrist",
                "left_phone": "left_pocket", "right_phone": "right_pocket", "glasses": "glasses"}

# Chinese action name -> canonical label (English translations from the XRFV2 repo action.py).
CN2CANON = {
    "伸懒腰": "stretching", "倒水": "pouring_water", "写字": "writing", "切水果": "cutting_fruit",
    "吃水果": "eating_fruit", "吃药": "taking_medicine", "喝水": "drinking", "坐下": "sitting_down",
    "开关护眼灯": "toggling_lamp", "开关窗帘": "opening_curtains", "开关窗户": "opening_windows",
    "打字": "typing", "打开信封": "opening_envelope", "扔垃圾": "throwing_garbage",
    "拿水果": "picking_fruit", "捡东西": "picking_up", "接电话": "answering_phone",
    "操作鼠标": "using_mouse", "擦桌子": "cleaning_table", "板书": "writing_on_blackboard",
    "洗手": "washing_hands", "玩手机": "using_phone", "看书": "reading", "给植物浇水": "watering_plants",
    "走向床": "walking", "走向椅子": "walking", "走向橱柜": "walking", "走向窗户": "walking",
    "走向黑板": "walking",
    # Posture transitions unified to the existing corpus scheme (canonical_labels.py:44-47) so they
    # share cross-config SupCon positives with kuhar/unimib/sp_sw_har instead of being xrf-only
    # singletons: 起立 = stand up from a chair, 起床 = get out of bed (from lying), 躺下 = lie down
    # from standing. (静止站立/静止躺着 remain the STATIC standing/lying postures.)
    "起床": "standing_up_from_lying", "起立": "standing_up_from_sitting",
    "躺下": "lying_down_from_standing",
    "静止站立": "standing", "静止躺着": "lying",
}


def _upsample_25_to_50(arr: np.ndarray) -> np.ndarray:
    """Linear-interpolate (n, C) from 25 Hz to 50 Hz (build_grids uses one dataset-wide rate)."""
    n = arr.shape[0]
    if n < 2:
        return arr
    x_old = np.arange(n)
    # EXACTLY 2N samples (25->50 Hz doubles the count): a 6 s / 25 Hz segment (N=150) must become
    # 300, not 2N-1=299, or it falls one sample short of a 300-sample 6 s window at 50 Hz and is
    # dropped by build_grids' non-overlapping windowing (was losing 22.8% of AirPods windows). The
    # trailing index n-0.5 > n-1 is clamped by np.interp to the last real sample (a held endpoint).
    x_new = np.arange(2 * n) * 0.5
    return np.stack([np.interp(x_new, x_old, arr[:, c]) for c in range(arr.shape[1])],
                    axis=1).astype(np.float32)


def _emit(sessions_dir, sid, acc, gyro, subject, labels, canon, dist):
    n = acc.shape[0]
    out = pd.DataFrame({
        "timestamp_sec": np.arange(n, dtype=np.float64) / IMU_RATE,
        **{f"acc_{a}": acc[:, i] for i, a in enumerate("xyz")},
        **{f"gyro_{a}": gyro[:, i] for i, a in enumerate("xyz")},
        "subject": subject,
    })
    seg = sessions_dir / sid
    seg.mkdir()
    out.to_parquet(seg / "data.parquet", index=False)
    labels[sid] = [canon]
    dist[canon] += 1


def main() -> None:
    sessions_dir = HERE / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    sessions_dir.mkdir(parents=True)

    tal = json.loads((RAW / "activitynet_tal_853.json").read_text())["database"]
    labels: dict[str, list[str]] = {}
    dist: Counter = Counter()
    dropped_short = 0

    with h5py.File(RAW / "imu_50hz_853_video_aligned.h5", "r") as fi, \
         h5py.File(RAW / "airpods_25hz_853_video_aligned.h5", "r") as fa:
        device_order = [d.decode() if isinstance(d, bytes) else d for d in fi["device_order"][:]]
        tokens = [DEVICE_TOKEN[d] for d in device_order]
        imu_samples, ap_samples = fi["samples"], fa["samples"]
        for name in fi["sample_names"][:]:
            name = name.decode() if isinstance(name, bytes) else name
            meta = tal.get(name)
            if meta is None:
                continue
            subject = f"v{meta['volunteer_id']}"
            imu = np.asarray(imu_samples[name]["imu"])                      # (T, 5, 6)
            ap = np.asarray(ap_samples[name][list(ap_samples[name].keys())[0]]) if name in ap_samples else None
            for ann in meta["annotations"]:
                canon = CN2CANON.get(ann["label"])
                if canon is None:
                    continue
                s, e = float(ann["segment"][0]), float(ann["segment"][1])
                # body IMU (50 Hz, gravity present)
                si, ei = int(round(s * IMU_RATE)), int(round(e * IMU_RATE))
                if ei - si >= 6.0 * IMU_RATE:
                    for dev, tok in enumerate(tokens):
                        acc = imu[si:ei, dev, :3].astype(np.float32)
                        gyro = imu[si:ei, dev, 3:].astype(np.float32) * DEG2RAD
                        _emit(sessions_dir, f"{subject}_{name}_{tok}_{canon}_a{si}",
                              acc, gyro, subject, labels, canon, dist)
                else:
                    dropped_short += 1
                # AirPods ear (25 Hz native -> upsample to 50 Hz; gravity removed)
                if ap is not None:
                    sa, ea = int(round(s * AIRPODS_RATE)), int(round(e * AIRPODS_RATE))
                    if ea - sa >= 6.0 * AIRPODS_RATE:
                        acc = _upsample_25_to_50(ap[sa:ea, :3].astype(np.float32))
                        gyro = _upsample_25_to_50(ap[sa:ea, 3:6].astype(np.float32))
                        _emit(sessions_dir, f"{subject}_{name}_airpods_ear_{canon}_a{sa}",
                              acc, gyro, subject, labels, canon, dist)

    (HERE / "labels.json").write_text(json.dumps(labels, indent=2))
    (HERE / "metadata.json").write_text(json.dumps(
        {"dataset": "xrf_v2", "sampling_rate_hz": IMU_RATE, "pre_windowed": False}, indent=2))
    (HERE / "manifest.json").write_text(json.dumps(create_manifest(), indent=2))
    subs = sorted({s.split("_")[0] for s in labels}, key=lambda x: int(x[1:]))
    print(f"xrf_v2 (Plus): {len(labels)} sessions; subjects={len(subs)} {subs}; "
          f"dropped_short(<6s)={dropped_short}; labels={len(dist)}")
    print("  per-label:", dict(sorted(dist.items(), key=lambda x: -x[1])))


def create_manifest() -> dict:
    chans = [{"name": c, "description": f"{'accelerometer' if 'acc' in c else 'gyroscope'} {c[-1]}-axis "
              f"in {'g' if 'acc' in c else 'rad/s'}"}
             for c in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")]
    return {
        "dataset_name": "XRF V2 (Plus; IMU + AirPods)",
        "description": ("XRF V2 indoor daily-activity dataset (arXiv 2501.19034), 16 volunteers, 853 "
                        "sequences: five-position body IMU (both wrists, both pocket phones, head "
                        "glasses) @50 Hz (acc g, gravity present; gyro rad/s) + AirPods Pro ear IMU "
                        "@25 Hz (user acceleration, gravity removed; gyro rad/s). WiFi/video not used."),
        "channels": chans,
    }


if __name__ == "__main__":
    main()
