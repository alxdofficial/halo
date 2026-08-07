"""Convert NFI-FARED (Netherlands Forensic Institute – Forensic Activity Recognition
Dataset), IMU subset, to the HALO session format.

Source: huggingface.co/datasets/NetherlandsForensicInstitute/NFI_FARED_IMU
Per-subject CSVs `pp<NN>/.../＿<DATE>_exp<E>_final_pp<NN>.csv` (the labelled protocol runs;
the companion `freeliving` files are unlabelled and skipped). 14 subjects (pp01–pp13,
pp15; pp14 absent). 17 columns, verified against a downloaded file:

    Unnamed: 0 | time | acc {x,y,z} rug | rot {x,y,z} rug | air pressure rug
              | acc {x,y,z} arm | rot {x,y,z} arm | air pressure arm | label activity

  * TWO body-strapped IMUs recorded simultaneously — `rug` = lower **back**, `arm` =
    dominant **forearm** (per the NFI/Hi-OSCAR paper; NOT the wrist). Emitted as SEPARATE
    streams (back / forearm), routed by the `_back_`/`_arm_` token in the session id
    (air-pressure channels dropped).
  * Accelerometer is in **g with gravity present** (verified at-rest |acc| ≈ 1.00 both
    placements) → kept as-is. Gyroscope is in **degrees/s** → converted to canonical
    **rad/s** (× π/180; verified: p99 ≈ 210 deg/s, impossible in rad/s).
  * 100 Hz native, uniform (median dt = 10.0 ms, no gaps). Each file is a continuous
    sequence of long single-activity runs (all ≥ 6 s); split into contiguous same-label
    runs, each emitted as one continuous session for build_grids' 6 s windowing
    (pre_windowed: false). `no activity` (transition/idle) is dropped.
  * 19 activities across 4 experiment groups (everyday / elevation / dynamic /
    transportation). Label → canonical below. The 4 transport modes collapse to the
    existing `vehicle` canonical (the wearer sits; the IMU sees vehicle motion). NEW
    canonical labels introduced: punching, throwing, dragging, escalator_up,
    escalator_down, elevator_up, elevator_down.
"""

from __future__ import annotations

import glob
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "downloads"
NATIVE_RATE = 100.0
DEG2RAD = np.pi / 180.0

LABEL_MAP = {
    "standing": "standing", "sitting": "sitting", "walking": "walking",
    "running": "running", "cycling": "cycling",
    "stair_up": "walking_upstairs", "stair_down": "walking_downstairs",
    # transportation -> existing generic `vehicle` canonical
    "train": "vehicle", "car": "vehicle", "tram": "vehicle", "bus": "vehicle",
    "kicking": "kicking",                          # existing canonical
    # --- new canonical labels ---
    "punching": "punching", "throwing": "throwing", "dragging": "dragging",
    "escalator_up": "escalator_up", "escalator_down": "escalator_down",
    "elevator_up": "elevator_up", "elevator_down": "elevator_down",
}
# `no activity` and anything else -> dropped (not in LABEL_MAP)

STREAMS = {   # session token -> (back|arm) source column suffix
    "back": "rug",
    "arm": "arm",
}


def create_manifest() -> dict:
    place = {"back": "an IMU strapped to the lower back",
             "arm": "an IMU strapped to the dominant forearm"}
    chans = []
    for tok in ("back", "arm"):
        for c in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"):
            chans.append({"name": c, "stream": tok,
                          "description": (f"{'accelerometer' if 'acc' in c else 'gyroscope'} "
                                          f"{c[-1]}-axis of {place[tok]} in "
                                          f"{'g units (gravity present)' if 'acc' in c else 'rad/s'}"),
                          "sampling_rate_hz": NATIVE_RATE})
    return {
        "dataset_name": "NFI-FARED (IMU)",
        "description": ("Netherlands Forensic Institute forensic activity dataset: two body-strapped "
                        "IMUs (lower back + dominant forearm), 14 subjects, 19 activities incl. "
                        "transportation and forensic dynamic movements, @100 Hz. Accelerometer in g "
                        "(gravity present); gyroscope converted deg/s -> rad/s."),
        "channels": chans,
    }


def merge_duplicate_labels(label_arrays: list[np.ndarray], source_names: list[str]) \
        -> tuple[np.ndarray, int]:
    """Merge annotations for signal-identical files without discarding mapped activity.

    NFI contains reruns whose IMU/time columns are identical but whose annotation columns differ.
    A mapped activity is more informative than transition/idle (``no activity``). Two different
    mapped canonical activities for the same row are ambiguous and therefore rejected.
    """
    if not label_arrays or len(label_arrays) != len(source_names):
        raise ValueError("duplicate-label merge needs aligned non-empty arrays and source names")
    lengths = {len(labels) for labels in label_arrays}
    if len(lengths) != 1:
        raise ValueError(f"signal-identical files have different label lengths: {source_names}")

    merged = np.asarray(label_arrays[0], dtype=str).copy()
    merged_canonical = np.asarray([LABEL_MAP.get(label) for label in merged], dtype=object)
    promoted = 0
    for labels, source in zip(label_arrays[1:], source_names[1:]):
        labels = np.asarray(labels, dtype=str)
        canonical = np.asarray([LABEL_MAP.get(label) for label in labels], dtype=object)
        conflict = ((merged_canonical != None) & (canonical != None)  # noqa: E711
                    & (merged_canonical != canonical))
        if bool(conflict.any()):
            rows = np.flatnonzero(conflict)[:5].tolist()
            raise ValueError(
                f"conflicting mapped labels in signal-identical NFI files {source_names}: "
                f"rows {rows}, incoming source {source}"
            )
        take = (merged_canonical == None) & (canonical != None)       # noqa: E711
        promoted += int(take.sum())
        merged[take] = labels[take]
        merged_canonical[take] = canonical[take]
    return merged, promoted


def main() -> None:
    sessions_dir = HERE / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    sessions_dir.mkdir(parents=True)

    labels: dict[str, list[str]] = {}
    events: dict[str, str] = {}
    n_sess = 0
    dist: Counter = Counter()
    files = sorted(glob.glob(str(RAW / "**" / "*final*.csv"), recursive=True))
    # Group signal-identical reruns before writing sessions. Their label columns can differ, so
    # keeping the first file would make lexicographic filename order decide the ground truth.
    signal_groups: dict[str, list[str]] = {}
    for csv in files:
        df = pd.read_csv(csv)
        signal = df.drop(columns=[c for c in df.columns
                                  if "label" in str(c).lower() or str(c).startswith("Unnamed")],
                         errors="ignore")
        sig_h = hashlib.sha256(
            pd.util.hash_pandas_object(signal, index=False).values.tobytes()).hexdigest()
        signal_groups.setdefault(sig_h, []).append(csv)

    n_dup = len(files) - len(signal_groups)
    n_promoted = 0
    for members in signal_groups.values():
        csv = members[0]
        name = Path(csv).stem                         # 20230608_exp1_final_pp01
        parts = name.split("_")
        date = parts[0]                                # disambiguates same-exp re-runs
        exp = next((p for p in parts if p.startswith("exp")), "exp")
        subject = next((p for p in parts if p.startswith("pp")), "pp00")
        member_subjects = {
            next((part for part in Path(path).stem.split("_") if part.startswith("pp")), "pp00")
            for path in members
        }
        if member_subjects != {subject}:
            raise ValueError(f"signal-identical files cross subjects: {members}")
        frames = [pd.read_csv(path) for path in members]
        df = frames[0]
        raw_lab, promoted = merge_duplicate_labels(
            [frame["label activity"].astype(str).to_numpy() for frame in frames],
            [Path(path).name for path in members],
        )
        n_promoted += promoted
        change = np.flatnonzero(raw_lab[1:] != raw_lab[:-1]) + 1
        bounds = np.concatenate([[0], change, [len(raw_lab)]])
        for k in range(len(bounds) - 1):
            lo, hi = int(bounds[k]), int(bounds[k + 1])
            canon = LABEL_MAP.get(raw_lab[lo])
            if canon is None:
                continue
            ts = np.arange(hi - lo, dtype=np.float64) / NATIVE_RATE
            for tok, suf in STREAMS.items():
                acc = df[[f"acc {a} {suf}" for a in "xyz"]].to_numpy(np.float32)[lo:hi]
                gyro = df[[f"rot {a} {suf}" for a in "xyz"]].to_numpy(np.float32)[lo:hi] * DEG2RAD
                sid = f"{subject}_{date}_{exp}_{tok}_run{k}"   # _back_/_arm_ token routes the stream
                events[sid] = f"{subject}_{date}_{exp}_run{k}"
                out = pd.DataFrame({
                    "timestamp_sec": ts,
                    **{f"acc_{a}": acc[:, i] for i, a in enumerate("xyz")},
                    **{f"gyro_{a}": gyro[:, i] for i, a in enumerate("xyz")},
                    "subject": subject,
                })
                seg_dir = sessions_dir / sid
                seg_dir.mkdir()
                out.to_parquet(seg_dir / "data.parquet", index=False)
                labels[sid] = [canon]
                n_sess += 1
            dist[canon] += 1

    (HERE / "labels.json").write_text(json.dumps(labels, indent=2))
    (HERE / "events.json").write_text(json.dumps(events, indent=2))
    (HERE / "metadata.json").write_text(json.dumps(
        {"dataset": "nfi_fared", "sampling_rate_hz": NATIVE_RATE, "pre_windowed": False}, indent=2))
    (HERE / "manifest.json").write_text(json.dumps(create_manifest(), indent=2))
    print(f"nfi_fared: {n_sess} sessions ({sum('_back_' in s for s in labels)} back / "
          f"{sum('_arm_' in s for s in labels)} forearm) from {len(files)} files; "
          f"{n_dup} signal-duplicate files merged, {n_promoted:,} activity rows recovered; "
          f"runs/label {dict(dist)}")


if __name__ == "__main__":
    main()
