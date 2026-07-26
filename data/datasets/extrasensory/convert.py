"""Convert ExtraSensory phone/watch accelerometry into HALO sessions.

Required archives under ``downloads/``:

* ``raw_acc.zip``: phone files, one 20-ish second recording per annotated minute.
  Every file has ``timestamp_unix, x, y, z``. Android users report m/s^2 while
  iPhone users report g, and effective clocks vary by device.
* ``watch_acc.zip``: Pebble watch accelerometry. Files are either ``x,y,z`` or
  ``milliseconds,x,y,z`` and use approximately milli-g units at 25 Hz.
* ``labels.zip``: one compressed feature/label CSV per participant.
* ``cv5Folds.zip``: the authors' participant split, used for the authoritative
  Android/iPhone device assignment.

The raw examples are multi-label contexts. HALO keeps examples with exactly one
observable movement primitive. Phone examples additionally require exactly one
deployment-plausible placement (pocket or hand); bag/table/unknown examples are
not given false placement text. The wrist stream does not depend on phone
placement.

Each example is unit-normalized from the subject's published phone platform,
resampled to 50 Hz, and truncated to an integer number of six-second windows
before examples are concatenated.
That makes subject/stream/activity aggregation safe: no grid window can cross an
original example boundary. The resulting files are accelerometer-only; the
common six-slot grid pads and masks gyroscope channels downstream.

Known source defect (NOT fixed here). The Pebble sometimes returns a stale buffer
instead of sampling, for hours at a time: subject ``0A986513`` has one group of 178
byte-identical recordings spanning 10,620 s, and ``3600D531`` is 87.7% duplicates.
The phone archives are clean, so this cannot be repaired by reading the watch files
differently — the bytes really are equal. It is screened downstream by
``data.scripts.scan_duplicates``, which also drops the whole group where members
disagree on the label (36.9% of duplicate pairs do, since the phone kept labelling
those minutes independently). Deduplicating here instead would silently change
window indices that the cached scans and built grids are keyed on.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


DS_DIR = Path(__file__).resolve().parent
DOWNLOADS = DS_DIR / "downloads"
OUTPUT_RATE_HZ = 50.0
WINDOW_SECONDS = 6.0
WINDOW_SAMPLES = int(OUTPUT_RATE_HZ * WINDOW_SECONDS)
GRAVITY_MS2 = 9.80665

ACTIVITY_COLUMNS = {
    "label:LYING_DOWN": "lying",
    "label:SITTING": "sitting",
    "label:OR_standing": "standing",
    "label:FIX_walking": "walking",
    "label:FIX_running": "running",
    "label:BICYCLING": "cycling",
    "label:STAIRS_-_GOING_UP": "walking_upstairs",
    "label:STAIRS_-_GOING_DOWN": "walking_downstairs",
}
PHONE_PLACEMENT_COLUMNS = {
    "label:PHONE_IN_POCKET": "phone_pocket",
    "label:PHONE_IN_HAND": "phone_hand",
    "label:PHONE_IN_BAG": None,
    "label:PHONE_ON_TABLE": None,
}
OUTPUT_COLUMNS = ("acc_x", "acc_y", "acc_z")


def _member_index(archive: ZipFile, suffix: str) -> dict[tuple[str, int], str]:
    result = {}
    for name in archive.namelist():
        if not name.endswith(suffix):
            continue
        parts = name.split("/")
        if len(parts) < 3:
            continue
        timestamp = int(parts[-1].split(".", 1)[0])
        result[(parts[-2], timestamp)] = name
    return result


def _load_text_array(archive: ZipFile, member: str) -> np.ndarray:
    array = np.loadtxt(io.BytesIO(archive.read(member)))
    array = np.atleast_2d(array).astype(np.float64, copy=False)
    if array.shape[1] not in (3, 4):
        raise ValueError(f"{member}: expected 3 or 4 columns, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{member}: non-finite values")
    return array


def _regularize(t: np.ndarray, xyz_g: np.ndarray, source_rate: float) -> np.ndarray:
    order = np.argsort(t, kind="stable")
    t = np.asarray(t[order], dtype=np.float64)
    xyz_g = np.asarray(xyz_g[order], dtype=np.float64)
    keep = np.r_[True, np.diff(t) > 1e-6]
    t, xyz_g = t[keep], xyz_g[keep]
    if len(t) < 3:
        raise ValueError("too few distinct timestamps")
    median_g = float(np.median(np.linalg.norm(xyz_g, axis=1)))
    if not 0.2 <= median_g <= 3.0:
        raise ValueError(f"implausible median acceleration {median_g:.3f} g")
    t = t - t[0]
    positive_dt = np.diff(t)
    positive_dt = positive_dt[positive_dt > 0]
    median_dt = float(np.median(positive_dt))
    duration = float(t[-1] + median_dt)
    if not 5.5 <= duration <= 60.0:
        raise ValueError(f"implausible example duration {duration:.3f}s")

    # A few Android recordings run above the 50 Hz storage clock. Filter before
    # interpolation so those files are not the only ones that alias high-frequency noise.
    if source_rate > OUTPUT_RATE_HZ * 1.1 and len(xyz_g) >= 32:
        cutoff_hz = 0.45 * OUTPUT_RATE_HZ
        sos = butter(6, cutoff_hz / (0.5 * source_rate), btype="low", output="sos")
        xyz_g = sosfiltfilt(sos, xyz_g, axis=0)

    n_out = max(1, int(round(duration * OUTPUT_RATE_HZ)))
    out_t = np.arange(n_out, dtype=np.float64) / OUTPUT_RATE_HZ
    out = np.column_stack([np.interp(out_t, t, xyz_g[:, axis]) for axis in range(3)])
    usable = (len(out) // WINDOW_SAMPLES) * WINDOW_SAMPLES
    if usable < WINDOW_SAMPLES:
        raise ValueError("example is shorter than one six-second window after resampling")
    return out[:usable].astype(np.float32)


def read_phone(archive: ZipFile, member: str, platform: str) -> np.ndarray:
    array = _load_text_array(archive, member)
    if array.shape[1] != 4:
        raise ValueError(f"{member}: phone recording has no timestamp column")
    t, xyz = array[:, 0], array[:, 1:4]
    if platform == "android":
        xyz = xyz / GRAVITY_MS2
    elif platform != "iphone":
        raise ValueError(f"{member}: unknown phone platform {platform!r}")
    dt = np.diff(t)
    dt = dt[dt > 0]
    if not len(dt):
        raise ValueError(f"{member}: non-increasing phone clock")
    return _regularize(t, xyz, source_rate=float(1.0 / np.median(dt)))


def read_watch(archive: ZipFile, member: str) -> np.ndarray:
    array = _load_text_array(archive, member)
    if array.shape[1] == 4:
        t = (array[:, 0] - array[0, 0]) / 1000.0
        xyz = array[:, 1:4]
        dt = np.diff(t)
        dt = dt[dt > 0]
        source_rate = float(1.0 / np.median(dt)) if len(dt) else 25.0
    else:
        xyz = array[:, :3]
        source_rate = 25.0
        t = np.arange(len(xyz), dtype=np.float64) / source_rate
    if float(np.median(np.linalg.norm(xyz, axis=1))) > 20.0:
        xyz = xyz / 1000.0
    return _regularize(t, xyz, source_rate=source_rate)


def _label_frame(labels_archive: ZipFile, member: str) -> pd.DataFrame:
    columns = ["timestamp", *ACTIVITY_COLUMNS, *PHONE_PLACEMENT_COLUMNS]
    payload = gzip.decompress(labels_archive.read(member))
    return pd.read_csv(io.BytesIO(payload), usecols=columns)


def _phone_platforms(split_archive: ZipFile) -> dict[str, str]:
    platforms: dict[str, str] = {}
    for platform in ("android", "iphone"):
        for split in ("train", "test"):
            member = f"cv_5_folds/fold_0_{split}_{platform}_uuids.txt"
            for subject in split_archive.read(member).decode("ascii").split():
                previous = platforms.setdefault(subject, platform)
                if previous != platform:
                    raise ValueError(f"{subject}: assigned to both phone platforms")
    if len(platforms) != 60:
        raise ValueError(f"expected 60 platform assignments, got {len(platforms)}")
    return platforms


def _activity(row: dict[str, object]) -> str | None:
    stair_up = row["label:STAIRS_-_GOING_UP"] == 1
    stair_down = row["label:STAIRS_-_GOING_DOWN"] == 1
    if stair_up != stair_down:
        return "walking_upstairs" if stair_up else "walking_downstairs"
    if stair_up and stair_down:
        return None
    active = [
        canonical
        for raw, canonical in ACTIVITY_COLUMNS.items()
        if "STAIRS" not in raw and row[raw] == 1
    ]
    return active[0] if len(active) == 1 else None


def convert(
    raw_dir: Path = DOWNLOADS,
    output_dir: Path = DS_DIR,
    limit_subjects: int | None = None,
    limit_examples_per_subject: int | None = None,
) -> bool:
    archive_paths = {
        "phone": raw_dir / "raw_acc.zip",
        "watch": raw_dir / "watch_acc.zip",
        "labels": raw_dir / "labels.zip",
        "splits": raw_dir / "cv5Folds.zip",
    }
    missing = [str(path) for path in archive_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing ExtraSensory archives; run `python -m data.datasets.extrasensory.fetch`: "
            + ", ".join(missing)
        )

    sessions_dir = output_dir / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    sessions_dir.mkdir(parents=True)

    labels_out: dict[str, list[str]] = {}
    counts: Counter = Counter()
    skips: Counter = Counter()

    with (
        ZipFile(archive_paths["phone"]) as phone_archive,
        ZipFile(archive_paths["watch"]) as watch_archive,
        ZipFile(archive_paths["labels"]) as labels_archive,
        ZipFile(archive_paths["splits"]) as split_archive,
    ):
        phone_platforms = _phone_platforms(split_archive)
        phone_members = _member_index(phone_archive, ".m_raw_acc.dat")
        watch_members = _member_index(watch_archive, ".m_watch_acc.dat")
        label_members = sorted(
            name for name in labels_archive.namelist() if name.endswith(".features_labels.csv.gz")
        )
        if limit_subjects is not None:
            label_members = label_members[:limit_subjects]

        for subject_i, label_member in enumerate(label_members, start=1):
            subject = label_member.split(".", 1)[0]
            platform = phone_platforms.get(subject)
            if platform is None:
                raise ValueError(f"{subject}: absent from the published platform split")
            labels = _label_frame(labels_archive, label_member)
            grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
            accepted_examples = 0

            for values in labels.itertuples(index=False, name=None):
                row = dict(zip(labels.columns, values))
                activity = _activity(row)
                if activity is None:
                    skips["ambiguous_movement"] += 1
                    continue
                timestamp = int(row["timestamp"])

                watch_member = watch_members.get((subject, timestamp))
                if watch_member is not None:
                    try:
                        grouped[("watch_wrist", activity)].append(
                            read_watch(watch_archive, watch_member)
                        )
                        accepted_examples += 1
                    except (ValueError, OSError):
                        skips["bad_watch"] += 1

                placements = [
                    PHONE_PLACEMENT_COLUMNS[name]
                    for name in PHONE_PLACEMENT_COLUMNS
                    if row[name] == 1
                ]
                deployable = [placement for placement in placements if placement is not None]
                phone_member = phone_members.get((subject, timestamp))
                if len(placements) == 1 and len(deployable) == 1 and phone_member is not None:
                    try:
                        grouped[(deployable[0], activity)].append(
                            read_phone(phone_archive, phone_member, platform)
                        )
                        accepted_examples += 1
                    except (ValueError, OSError):
                        skips["bad_phone"] += 1
                elif phone_member is not None:
                    skips["phone_placement_pruned"] += 1

                if (
                    limit_examples_per_subject is not None
                    and accepted_examples >= limit_examples_per_subject
                ):
                    break

            for (stream, activity), arrays in sorted(grouped.items()):
                if not arrays:
                    continue
                data = np.concatenate(arrays, axis=0)
                frame = pd.DataFrame(data, columns=OUTPUT_COLUMNS)
                frame.insert(
                    0,
                    "timestamp_sec",
                    np.arange(len(frame), dtype=np.float64) / OUTPUT_RATE_HZ,
                )
                frame["subject"] = subject
                session_id = f"extrasensory_{subject}_{stream}_{activity}"
                destination = sessions_dir / session_id
                destination.mkdir()
                frame.to_parquet(destination / "data.parquet", index=False)
                labels_out[session_id] = [activity]
                counts[(stream, activity)] += len(data) // WINDOW_SAMPLES

            print(
                f"[extrasensory] {subject_i:02d}/{len(label_members)} {subject[:8]}: "
                f"{sum(len(v) for v in grouped.values())} accepted examples"
            )

    if not labels_out:
        return False
    (output_dir / "labels.json").write_text(json.dumps(labels_out, indent=2) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "ExtraSensory",
                "source": "http://extrasensory.ucsd.edu/",
                "num_subjects": len(
                    {session_id.split("_")[1] for session_id in labels_out}
                ),
                "sampling_rate_hz": OUTPUT_RATE_HZ,
                "channels": list(OUTPUT_COLUMNS),
                "unit": "g",
                "gravity_state": "present",
                "note": (
                    "Phone/watch raw acceleration only. Android m/s^2 and iPhone g are "
                    "normalized using the authors' platform split; watch milli-g is converted "
                    "to g. Phone examples are restricted to explicitly labelled pocket/hand "
                    "placements."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print("[extrasensory] windows by stream/activity:")
    for key, count in sorted(counts.items()):
        print(f"  {key[0]:14s} {key[1]:20s} {count:8d}")
    print(f"[extrasensory] skips: {dict(skips)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DOWNLOADS)
    parser.add_argument("--output-dir", type=Path, default=DS_DIR)
    parser.add_argument("--limit-subjects", type=int)
    parser.add_argument("--limit-examples-per-subject", type=int)
    args = parser.parse_args()
    if not convert(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        limit_subjects=args.limit_subjects,
        limit_examples_per_subject=args.limit_examples_per_subject,
    ):
        raise SystemExit("no ExtraSensory sessions were produced")


if __name__ == "__main__":
    main()
