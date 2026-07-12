"""
Convert RealWorld HAR (2016) dataset to the new HALO raw-session layout.

Deployment stream: `phone_waist` (see data/scripts/curate/deployment_policy.py) — we convert
ONLY the WAIST body position, emitting generic accelerometer columns acc_x/acc_y/acc_z and,
when a complete finite triad exists, gyroscope columns gyro_x/gyro_y/gyro_z.

RealWorld raw layout (data/datasets/realworld/downloads/probandN/data/):
  - Per (sensor, activity) outer zip: {acc|gyr}_{activity}_csv.zip
      * single recording  -> outer zip directly holds per-position CSVs
                             (acc_{activity}_waist.csv / Gyroscope_{activity}_waist.csv)
      * multi-part record  -> outer zip holds inner zips ({sensor}_{activity}_<n>_csv.zip),
                             each inner zip holds the per-position CSVs for that part.
  - CSV columns: id, attr_time (Unix ms), attr_x, attr_y, attr_z
  - Accelerometer is in m/s² (gravity present); the shared pipeline rescales m/s² -> g.

Output (new layout, this directory):
  - sessions/<session_id>/data.parquet : timestamp_sec + acc_x/y/z (+ gyro_x/y/z) + subject
  - labels.json                        : {session_id: [activity_name]}
  - manifest.json / metadata.json      : informational + native rate for build_grids

We emit RAW whole-recording sessions (one per continuous waist recording / recording part).
build_grids performs the fixed 6-second windowing — do NOT pre-window here.
"""

import io
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Repo root on path for shared imports (not strictly needed here, kept for parity).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

DS_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = DS_DIR / "downloads"
OUTPUT_DIR = DS_DIR
SESSIONS_DIR = OUTPUT_DIR / "sessions"

# Body position for the phone_waist deployment stream.
TARGET_POSITION = "waist"
# RealWorld records at ~50 Hz; we resample each recording onto a uniform 50 Hz grid so the
# acc/gyro streams share one timeline and build_grids' fixed native rate is exact.
TARGET_SAMPLE_RATE = 50.0

# RealWorld activities (raw folder/file codes == the label strings we store).
ACTIVITIES = [
    "climbingdown",
    "climbingup",
    "jumping",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking",
]

# Outer-zip sensor prefix -> output channel prefix. Magnetometer is intentionally dropped:
# the phone_waist stream only needs accelerometer (+ optional gyroscope).
SENSORS = {"acc": "acc", "gyr": "gyro"}


def _waist_member(namelist: List[str]) -> Optional[str]:
    """Return the per-position CSV member for the waist position (acc_*_waist.csv or
    Gyroscope_*_waist.csv), or None. Matching by the `_waist.csv` suffix is robust to the
    differing acc (`acc_`) vs gyro (`Gyroscope_`) filename prefixes."""
    for name in namelist:
        if name.lower().endswith("_waist.csv"):
            return name
    return None


def _parse_waist_csv(fileobj, sensor_out: str) -> Optional[pd.DataFrame]:
    """Parse one RealWorld per-position CSV into timestamp_sec + {sensor_out}_x/y/z."""
    df = pd.read_csv(fileobj)
    df.columns = [c.lower().strip() for c in df.columns]

    time_col = "attr_time" if "attr_time" in df.columns else ("time" if "time" in df.columns else None)
    if time_col is None:
        return None
    if not all(f"attr_{axis}" in df.columns for axis in "xyz"):
        return None
    if len(df) < 10:
        return None

    t_ms = df[time_col].values.astype(float)
    out = pd.DataFrame()
    out["timestamp_sec"] = (t_ms - t_ms[0]) / 1000.0  # Unix ms -> seconds, relative to start
    for axis in "xyz":
        out[f"{sensor_out}_{axis}"] = df[f"attr_{axis}"].values.astype(float)
    return out


def load_sensor_parts(outer_zip_path: Path, sensor_out: str) -> List[Optional[pd.DataFrame]]:
    """Return the ordered list of per-recording-part waist DataFrames for one sensor.

    Single-recording zips yield a 1-element list; multi-part (nested-zip) activities yield one
    entry per part, ordered by part index. Entries may be None if a part lacks a waist CSV.
    """
    if not outer_zip_path.exists():
        return []

    parts: List[Optional[pd.DataFrame]] = []
    with zipfile.ZipFile(outer_zip_path) as z:
        names = z.namelist()
        inner_zips = [n for n in names if n.lower().endswith(".zip")]

        if inner_zips:
            # Multi-part: order inner zips by their trailing part number. Part 1 for gyro has no
            # number (gyr_<act>_csv.zip); treat missing number as part 0 so it sorts first. acc/gyro
            # are then paired by list position (acc parts 1,2,3 ; gyro parts 0/1,2,3).
            def part_key(n: str) -> int:
                m = re.search(r"_(\d+)_csv\.zip$", n.lower())
                return int(m.group(1)) if m else 0

            for inner_name in sorted(inner_zips, key=part_key):
                inner_bytes = z.read(inner_name)
                with zipfile.ZipFile(io.BytesIO(inner_bytes)) as zi:
                    member = _waist_member(zi.namelist())
                    if member is None:
                        parts.append(None)
                        continue
                    with zi.open(member) as f:
                        parts.append(_parse_waist_csv(f, sensor_out))
        else:
            member = _waist_member(names)
            if member is not None:
                with z.open(member) as f:
                    parts.append(_parse_waist_csv(f, sensor_out))

    return parts


def resample_part(acc_df: pd.DataFrame, gyro_df: Optional[pd.DataFrame],
                  rate: float = TARGET_SAMPLE_RATE) -> Optional[pd.DataFrame]:
    """Resample one recording part onto a uniform `rate` Hz timeline.

    Accelerometer is required and defines the session duration; gyroscope (if present) is
    interpolated onto the same timeline. Both are treated relative to their own start (the
    acc/gyro clocks differ by a few ms; negligible at 50 Hz). Gyro is kept only if it forms a
    complete finite triad over the whole session.
    """
    src_t = acc_df["timestamp_sec"].values - acc_df["timestamp_sec"].values[0]
    duration = float(src_t[-1])
    if duration <= 0:
        return None

    n = int(duration * rate) + 1
    if n < 10:
        return None
    t = np.linspace(0.0, duration, n)

    out = pd.DataFrame()
    out["timestamp_sec"] = t
    for axis in "xyz":
        out[f"acc_{axis}"] = np.interp(t, src_t, acc_df[f"acc_{axis}"].values)

    if gyro_df is not None and len(gyro_df) >= 10:
        g_t = gyro_df["timestamp_sec"].values - gyro_df["timestamp_sec"].values[0]
        gyro_cols = {}
        for axis in "xyz":
            gyro_cols[f"gyro_{axis}"] = np.interp(t, g_t, gyro_df[f"gyro_{axis}"].values)
        # Retain gyro only when the converted triad is complete and finite.
        if all(np.all(np.isfinite(v)) for v in gyro_cols.values()):
            for col, vals in gyro_cols.items():
                out[col] = vals

    if not np.all(np.isfinite(out[["acc_x", "acc_y", "acc_z"]].values)):
        return None
    return out


def convert_realworld() -> bool:
    print("=" * 80)
    print("RealWorld HAR -> HALO raw sessions (waist / phone_waist)")
    print("=" * 80)

    if not DOWNLOADS_DIR.exists():
        print(f"ERROR: Raw data not found at {DOWNLOADS_DIR}")
        return False

    if SESSIONS_DIR.exists():
        shutil.rmtree(SESSIONS_DIR)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    subject_folders = sorted(
        [d for d in DOWNLOADS_DIR.iterdir() if d.is_dir() and d.name.startswith("proband")],
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )
    print(f"\nFound {len(subject_folders)} subjects")
    if not subject_folders:
        print("ERROR: No proband folders found")
        return False

    all_labels = {}
    subjects_seen = set()
    subjects_with_acc = set()
    subjects_with_gyro = set()
    activity_counts = {}
    session_count = 0
    gyro_session_count = 0
    acc_only_session_count = 0
    skipped_no_acc = 0

    for subject_folder in subject_folders:
        subject = subject_folder.name  # e.g. "proband1"
        data_dir = subject_folder / "data"
        if not data_dir.exists():
            print(f"  {subject}: WARNING no data folder, skipping")
            continue

        subject_sessions = 0

        for activity in ACTIVITIES:
            acc_parts = load_sensor_parts(data_dir / f"acc_{activity}_csv.zip", "acc")
            gyro_parts = load_sensor_parts(data_dir / f"gyr_{activity}_csv.zip", "gyro")

            if not acc_parts:
                # No waist accelerometer for this (subject, activity) — cannot emit.
                continue

            for idx, acc_df in enumerate(acc_parts):
                if acc_df is None:
                    skipped_no_acc += 1
                    continue
                gyro_df = gyro_parts[idx] if idx < len(gyro_parts) else None

                frame = resample_part(acc_df, gyro_df)
                if frame is None:
                    skipped_no_acc += 1
                    continue

                frame["subject"] = subject

                multi = len(acc_parts) > 1
                session_id = (f"{subject}_{activity}_part{idx + 1}" if multi
                              else f"{subject}_{activity}")

                session_dir = SESSIONS_DIR / session_id
                session_dir.mkdir(exist_ok=True)
                frame.to_parquet(session_dir / "data.parquet", index=False)

                all_labels[session_id] = [activity]
                activity_counts[activity] = activity_counts.get(activity, 0) + 1
                subjects_seen.add(subject)
                subjects_with_acc.add(subject)
                has_gyro = "gyro_x" in frame.columns
                if has_gyro:
                    gyro_session_count += 1
                    subjects_with_gyro.add(subject)
                else:
                    acc_only_session_count += 1
                session_count += 1
                subject_sessions += 1

        print(f"  {subject}: {subject_sessions} sessions")

    if session_count == 0:
        print("ERROR: no waist accelerometer sessions produced — STOP.")
        return False

    # labels.json
    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump(all_labels, f, indent=2)

    write_manifest(gyro_available=gyro_session_count > 0)
    update_metadata(num_sessions=session_count,
                    activities=sorted(activity_counts.keys()),
                    gyro_available=gyro_session_count > 0)

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output dir           : {OUTPUT_DIR}")
    print(f"Sessions written     : {session_count}")
    print(f"  with gyro (6-ch)   : {gyro_session_count}")
    print(f"  acc-only (3-ch)    : {acc_only_session_count}")
    print(f"Parts skipped (no acc): {skipped_no_acc}")
    print(f"Distinct subjects    : {len(subjects_seen)}")
    print(f"  subjects w/ gyro   : {len(subjects_with_gyro)}")
    print(f"Native rate          : {TARGET_SAMPLE_RATE} Hz")
    print("Sessions per activity:")
    for act in sorted(activity_counts):
        print(f"  {act:14s}: {activity_counts[act]}")
    return True


def write_manifest(gyro_available: bool) -> None:
    channels = [
        {"name": "acc_x", "description": "Accelerometer X-axis (waist), m/s^2", "sampling_rate_hz": TARGET_SAMPLE_RATE},
        {"name": "acc_y", "description": "Accelerometer Y-axis (waist), m/s^2", "sampling_rate_hz": TARGET_SAMPLE_RATE},
        {"name": "acc_z", "description": "Accelerometer Z-axis (waist), m/s^2", "sampling_rate_hz": TARGET_SAMPLE_RATE},
    ]
    if gyro_available:
        channels += [
            {"name": "gyro_x", "description": "Gyroscope X-axis (waist), rad/s", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "gyro_y", "description": "Gyroscope Y-axis (waist), rad/s", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "gyro_z", "description": "Gyroscope Z-axis (waist), rad/s", "sampling_rate_hz": TARGET_SAMPLE_RATE},
        ]
    manifest = {
        "dataset_name": "RealWorld HAR",
        "description": ("RealWorld HAR (University of Mannheim). 15 subjects, 8 activities. "
                        "Waist position, triaxial accelerometer (m/s^2, gravity present) plus "
                        "gyroscope where a complete finite triad exists, on a uniform 50 Hz grid."),
        "source": "https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/",
        "num_subjects": 15,
        "body_position": TARGET_POSITION,
        "channels": channels,
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def update_metadata(num_sessions: int, activities: List[str], gyro_available: bool) -> None:
    """Refresh informational fields of metadata.json while preserving the split/role fields that
    downstream tooling relies on. build_grids reads `sampling_rate_hz`; never set `pre_windowed`."""
    meta_path = OUTPUT_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    meta["display_name"] = meta.get("display_name", "RealWorld HAR")
    meta["num_sessions"] = num_sessions
    meta["sampling_rate_hz"] = int(TARGET_SAMPLE_RATE)
    channels = ["acc_x", "acc_y", "acc_z"]
    if gyro_available:
        channels += ["gyro_x", "gyro_y", "gyro_z"]
    meta["channels"] = channels
    meta["core_channels"] = {c: c for c in channels}
    meta["extra_channels"] = []
    meta.pop("note", None)
    meta["placement"] = meta.get("placement", TARGET_POSITION)
    meta["num_subjects"] = 15
    meta["activities"] = activities
    meta.pop("pre_windowed", None)

    meta_path.write_text(json.dumps(meta, indent=2))


def main() -> int:
    return 0 if convert_realworld() else 1


if __name__ == "__main__":
    sys.exit(main())
