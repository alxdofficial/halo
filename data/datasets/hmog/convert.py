"""Convert H-MOG phone-in-hand IMU recordings into HALO sessions.

The official archive contains one nested zip per anonymized subject and 24
sessions per subject. Each session records a Samsung Galaxy S4 accelerometer
and gyroscope at approximately 100 Hz while the participant uses the phone
sitting or walking.

The source clocks are independent and contain occasional gaps. Conversion is
therefore activity-aware:

* duplicate ``Activity.csv`` rows are checked for consistent protocol fields
  and collapsed by ``ActivityID``;
* accelerometer and gyroscope are joined on their monotonic ``EventTime``
  clocks, not the millisecond wall clock;
* each sensor is linearly interpolated to a common 100 Hz grid, but a gap over
  250 ms in either stream is a hard boundary;
* each continuous block is truncated independently to complete six-second
  windows before blocks are concatenated.

That final rule guarantees that downstream non-overlapping grid windows never
cross a source gap. Raw acceleration remains in m/s^2 and is converted to g by
the shared ``accel_units`` stage. Android gyroscope values are already rad/s.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


DS_DIR = Path(__file__).resolve().parent
ARCHIVE = DS_DIR / "downloads" / "hmog_dataset.zip"
RATE_HZ = 100.0
WINDOW_SECONDS = 6.0
WINDOW_SAMPLES = int(RATE_HZ * WINDOW_SECONDS)
MAX_INTERPOLATION_GAP_SECONDS = 0.25

ACTIVITY_COLUMNS = (
    "activity_id",
    "subject_id",
    "session_number",
    "start_time_ms",
    "end_time_ms",
    "relative_start_time_ms",
    "relative_end_time_ms",
    "gesture_scenario",
    "task_id",
    "content_id",
)
SENSOR_COLUMNS = (
    "system_time_ms",
    "event_time_ns",
    "activity_id",
    "x",
    "y",
    "z",
    "phone_orientation",
)
OUTPUT_COLUMNS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
SCENARIO_LABELS = {1: "sitting", 2: "walking"}
SITTING_TASKS = frozenset(range(1, 25, 2))
WALKING_TASKS = frozenset(range(2, 25, 2))
SUBJECT_MEMBER = re.compile(r"public_dataset/(?P<subject>\d+)\.zip$")


def _read_numeric_csv(
    archive: ZipFile,
    member: str,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    frame = pd.read_csv(archive.open(member), header=None, names=columns)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.isna().any(axis=None):
        raise ValueError(f"{member}: non-numeric or missing values")
    return frame


def _activity_table(archive: ZipFile, member: str, subject: str) -> pd.DataFrame:
    frame = _read_numeric_csv(archive, member, ACTIVITY_COLUMNS)
    integer_columns = (
        "activity_id",
        "subject_id",
        "session_number",
        "gesture_scenario",
        "task_id",
        "content_id",
    )
    for column in integer_columns:
        frame[column] = frame[column].astype(np.int64)

    if set(frame["subject_id"].astype(str)) != {subject}:
        raise ValueError(
            f"{member}: subject field does not match archive subject {subject}"
        )
    if not frame["gesture_scenario"].isin(SCENARIO_LABELS).all():
        bad = sorted(set(frame["gesture_scenario"]) - set(SCENARIO_LABELS))
        raise ValueError(f"{member}: unknown Gesture_scenario values {bad}")
    task_consistent = (
        (frame["gesture_scenario"].eq(1) & frame["task_id"].isin(SITTING_TASKS))
        | (frame["gesture_scenario"].eq(2) & frame["task_id"].isin(WALKING_TASKS))
    )
    if not task_consistent.all():
        bad = frame.loc[
            ~task_consistent,
            ["activity_id", "gesture_scenario", "task_id"],
        ]
        raise ValueError(
            f"{member}: Gesture_scenario disagrees with TaskID: "
            f"{bad.to_dict('records')[:5]}"
        )

    # Some source files repeat an ActivityID with a slightly different end
    # timestamp. The protocol fields must agree before those rows are collapsed.
    protocol = (
        "subject_id",
        "session_number",
        "gesture_scenario",
        "task_id",
        "content_id",
    )
    disagreement = (
        frame.groupby("activity_id", sort=False)[list(protocol)]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if disagreement.any():
        ids = disagreement.index[disagreement].tolist()
        raise ValueError(f"{member}: conflicting duplicate ActivityIDs {ids[:5]}")
    return frame.drop_duplicates("activity_id", keep="first").reset_index(drop=True)


def _sensor_arrays(
    frame: pd.DataFrame,
    activity_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = frame.loc[
        frame["activity_id"].eq(activity_id),
        ["event_time_ns", "x", "y", "z"],
    ]
    if rows.empty:
        return np.empty(0, np.float64), np.empty((0, 3), np.float64)
    array = rows.to_numpy(np.float64)
    order = np.argsort(array[:, 0], kind="stable")
    array = array[order]
    keep = np.r_[True, np.diff(array[:, 0]) > 0]
    array = array[keep]
    return array[:, 0] / 1e9, array[:, 1:4]


def _interpolate_with_validity(
    source_time: np.ndarray,
    source_xyz: np.ndarray,
    grid: np.ndarray,
    max_gap_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    insertion = np.searchsorted(source_time, grid, side="left")
    right = np.clip(insertion, 0, len(source_time) - 1)
    left = np.clip(insertion - 1, 0, len(source_time) - 1)
    inside = (grid >= source_time[0]) & (grid <= source_time[-1])
    bracket = source_time[right] - source_time[left]
    exact = np.isclose(source_time[right], grid, rtol=0.0, atol=1e-7)
    valid = inside & ((bracket <= max_gap_seconds) | exact)
    values = np.column_stack(
        [np.interp(grid, source_time, source_xyz[:, axis]) for axis in range(3)]
    )
    return values, valid


def regularize_activity(
    acc_time: np.ndarray,
    acc_xyz: np.ndarray,
    gyro_time: np.ndarray,
    gyro_xyz: np.ndarray,
    *,
    rate_hz: float = RATE_HZ,
    max_gap_seconds: float = MAX_INTERPOLATION_GAP_SECONDS,
) -> tuple[list[np.ndarray], Counter]:
    """Synchronize one ActivityID and return complete, gap-safe IMU blocks."""
    stats: Counter = Counter()
    if min(len(acc_time), len(gyro_time)) < 3:
        stats["activity_missing_sensor"] += 1
        return [], stats

    start = max(float(acc_time[0]), float(gyro_time[0]))
    stop = min(float(acc_time[-1]), float(gyro_time[-1]))
    if stop <= start:
        stats["activity_no_clock_overlap"] += 1
        return [], stats
    n = int(np.floor((stop - start) * rate_hz)) + 1
    grid = start + np.arange(n, dtype=np.float64) / rate_hz
    acc, acc_valid = _interpolate_with_validity(
        acc_time, acc_xyz, grid, max_gap_seconds
    )
    gyro, gyro_valid = _interpolate_with_validity(
        gyro_time, gyro_xyz, grid, max_gap_seconds
    )
    valid = acc_valid & gyro_valid

    padded = np.r_[False, valid, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    blocks: list[np.ndarray] = []
    for block_start, block_end in zip(starts, ends):
        length = int(block_end - block_start)
        usable = (length // WINDOW_SAMPLES) * WINDOW_SAMPLES
        stats["samples_short_tail_dropped"] += length - usable
        if usable == 0:
            stats["short_continuous_blocks_dropped"] += 1
            continue
        block = np.column_stack((acc, gyro))[block_start : block_start + usable]
        if not np.isfinite(block).all():
            raise ValueError("regularization produced non-finite IMU values")
        blocks.append(block.astype(np.float32, copy=False))
        stats["retained_blocks"] += 1
        stats["retained_samples"] += usable
    stats["invalid_grid_samples"] += int((~valid).sum())
    return blocks, stats


def _session_members(archive: ZipFile) -> list[str]:
    return sorted(
        (
            name
            for name in archive.namelist()
            if name.endswith("/Activity.csv")
        ),
        key=lambda name: int(
            re.search(r"_session_(\d+)/Activity\.csv$", name).group(1)
        ),
    )


def convert(
    archive_path: Path = ARCHIVE,
    output_dir: Path = DS_DIR,
    *,
    limit_subjects: int | None = None,
    limit_sessions_per_subject: int | None = None,
) -> bool:
    if not archive_path.exists():
        raise FileNotFoundError(
            f"{archive_path} is missing; run "
            "`python -m data.datasets.hmog.fetch --accept-license`"
        )

    sessions_dir = output_dir / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    sessions_dir.mkdir(parents=True)

    labels: dict[str, list[str]] = {}
    stats: Counter = Counter()
    label_samples: Counter = Counter()
    subjects_seen: set[str] = set()
    subject_aliases: dict[str, str] = {}

    with ZipFile(archive_path) as outer:
        subject_members = [
            (match.group("subject"), info.filename)
            for info in outer.infolist()
            if (match := SUBJECT_MEMBER.fullmatch(info.filename))
        ]
        if limit_subjects is not None:
            subject_members = subject_members[:limit_subjects]
        if not subject_members:
            raise RuntimeError(f"{archive_path}: no subject archives found")

        for subject_index, (archive_subject, member) in enumerate(subject_members, start=1):
            with ZipFile(io.BytesIO(outer.read(member))) as subject_archive:
                session_members = _session_members(subject_archive)
                if limit_sessions_per_subject is not None:
                    session_members = session_members[:limit_sessions_per_subject]
                if not session_members:
                    stats["subjects_without_sessions"] += 1
                    continue
                first_activity = _read_numeric_csv(
                    subject_archive, session_members[0], ACTIVITY_COLUMNS
                )
                internal_subjects = {
                    str(int(value)) for value in first_activity["subject_id"].unique()
                }
                if len(internal_subjects) != 1:
                    raise ValueError(
                        f"{member}: expected one internal subject ID, found "
                        f"{sorted(internal_subjects)}"
                    )
                subject = internal_subjects.pop()
                if subject != archive_subject:
                    subject_aliases[archive_subject] = subject
                    stats["archive_subject_id_aliases"] += 1
                subjects_seen.add(subject)

                names = set(subject_archive.namelist())
                for activity_member in session_members:
                    base = activity_member.rsplit("/", 1)[0]
                    acc_member = f"{base}/Accelerometer.csv"
                    gyro_member = f"{base}/Gyroscope.csv"
                    if acc_member not in names or gyro_member not in names:
                        stats["sessions_missing_imu_file"] += 1
                        continue

                    activity = _activity_table(
                        subject_archive, activity_member, subject
                    )
                    acc_frame = _read_numeric_csv(
                        subject_archive, acc_member, SENSOR_COLUMNS
                    )
                    gyro_frame = _read_numeric_csv(
                        subject_archive, gyro_member, SENSOR_COLUMNS
                    )
                    known_ids = set(activity["activity_id"])
                    stats["orphan_acc_samples"] += int(
                        (~acc_frame["activity_id"].isin(known_ids)).sum()
                    )
                    stats["orphan_gyro_samples"] += int(
                        (~gyro_frame["activity_id"].isin(known_ids)).sum()
                    )

                    for row in activity.itertuples(index=False):
                        activity_id = int(row.activity_id)
                        acc_time, acc_xyz = _sensor_arrays(acc_frame, activity_id)
                        gyro_time, gyro_xyz = _sensor_arrays(
                            gyro_frame, activity_id
                        )
                        blocks, activity_stats = regularize_activity(
                            acc_time, acc_xyz, gyro_time, gyro_xyz
                        )
                        stats.update(activity_stats)
                        if not blocks:
                            stats["activities_without_complete_window"] += 1
                            continue

                        data = np.concatenate(blocks, axis=0)
                        label = SCENARIO_LABELS[int(row.gesture_scenario)]
                        session_number = int(row.session_number)
                        session_id = (
                            f"hmog_{subject}_session{session_number:02d}_"
                            f"activity{activity_id}_phone_hand"
                        )
                        frame = pd.DataFrame(data, columns=OUTPUT_COLUMNS)
                        frame.insert(
                            0,
                            "timestamp_sec",
                            np.arange(len(frame), dtype=np.float64) / RATE_HZ,
                        )
                        frame["subject"] = subject
                        destination = sessions_dir / session_id
                        destination.mkdir()
                        frame.to_parquet(
                            destination / "data.parquet",
                            index=False,
                            compression="zstd",
                        )
                        labels[session_id] = [label]
                        label_samples[label] += len(frame)
                        stats["output_sessions"] += 1

            print(
                f"[hmog] {subject_index:3d}/{len(subject_members)} "
                f"subject {subject}: {stats['output_sessions']:,} total sessions",
                flush=True,
            )

    if not labels:
        return False
    (output_dir / "labels.json").write_text(
        json.dumps(labels, indent=2, sort_keys=True) + "\n"
    )
    hours_by_label = {
        label: samples / RATE_HZ / 3600.0
        for label, samples in sorted(label_samples.items())
    }
    manifest = {
        "dataset_name": "H-MOG",
        "source": "https://hmog-dataset.github.io/hmog/",
        "num_subjects": len(subjects_seen),
        "subject_aliases": subject_aliases,
        "sampling_rate_hz": RATE_HZ,
        "channels": list(OUTPUT_COLUMNS),
        "accelerometer_unit": "m/s^2",
        "gyroscope_unit": "rad/s",
        "gravity_state": "present",
        "placement": "phone held in the hand",
        "output_sessions": int(stats["output_sessions"]),
        "retained_hours": sum(hours_by_label.values()),
        "hours_by_label": hours_by_label,
        "stats": {key: int(value) for key, value in sorted(stats.items())},
        "note": (
            "ActivityID-level streams synchronized on EventTime; gaps over "
            f"{MAX_INTERPOLATION_GAP_SECONDS:.2f}s split; complete 6s blocks only."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[hmog] wrote {len(labels):,} sessions from {len(subjects_seen)} subjects, "
        f"{manifest['retained_hours']:.2f} h: {hours_by_label}"
    )
    print(f"[hmog] conversion stats: {dict(stats)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DS_DIR)
    parser.add_argument("--limit-subjects", type=int)
    parser.add_argument("--limit-sessions-per-subject", type=int)
    args = parser.parse_args()
    if not convert(
        args.archive,
        args.output_dir,
        limit_subjects=args.limit_subjects,
        limit_sessions_per_subject=args.limit_sessions_per_subject,
    ):
        raise SystemExit("no H-MOG sessions were produced")


if __name__ == "__main__":
    main()
