"""Convert a bounded NHANES PAX80_G subset into Phase-A-only HALO sessions.

NHANES 2011-2012 uses an ActiGraph GT3X+ on the non-dominant wrist, sampled at
80 Hz. Values are calibrated triaxial acceleration in g and include gravity.
Each participant archive contains hourly CSV files and a QC interval CSV.

There are no activity annotations or diaries. Sessions therefore carry the
reserved ``__unlabeled__`` marker. Pipeline A may use them for self-supervised
objectives, but label vocabulary construction, validation probes, and the
Phase-B evidence bank must exclude that marker.

The released QC file flags sensor malfunction (~0.26% of the release), not
off-body time, and no non-wear detector is applied here. The result is
free-living as recorded: 43% of windows sit below 0.003 g of motion and 13.3%
are byte-identical repeats of a motionless posture. The exact repeats are
dropped by ``data.scripts.scan_duplicates``; the rest is genuine sedentary and
sleep time and is kept on purpose.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import tarfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


DS_DIR = Path(__file__).resolve().parent
DOWNLOADS = DS_DIR / "downloads"
RATE_HZ = 80.0
WINDOW_SECONDS = 6.0
WINDOW_SAMPLES = int(RATE_HZ * WINDOW_SECONDS)
UNLABELED = "__unlabeled__"
OUTPUT_COLUMNS = ("acc_x", "acc_y", "acc_z")
_STAMP = re.compile(r"\.(2000-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})-000-P0000\.sensor\.csv$")


def _member_start(name: str) -> datetime:
    match = _STAMP.search(name)
    if not match:
        raise ValueError(f"cannot parse NHANES timestamp from {name}")
    return datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S")


def _read_qc(archive: tarfile.TarFile, members: list[tarfile.TarInfo]) -> pd.DataFrame:
    log = next((member for member in members if member.name.endswith("_Logs.csv")), None)
    if log is None:
        return pd.DataFrame()
    handle = archive.extractfile(log)
    if handle is None:
        return pd.DataFrame()
    frame = pd.read_csv(handle)
    return frame.dropna(how="all")


def _qc_intervals(qc: pd.DataFrame, first_day: datetime) -> list[tuple[datetime, datetime]]:
    result = []
    if qc.empty:
        return result
    day0 = datetime.combine(first_day.date(), datetime.min.time())
    for row in qc.itertuples(index=False):
        try:
            day = int(row.DAY_OF_DATA)
            start_time = pd.to_timedelta(str(row.START_TIME))
            end_time = pd.to_timedelta(str(row.END_TIME))
        except (TypeError, ValueError):
            continue
        start = day0 + timedelta(days=day - 1) + start_time.to_pytimedelta()
        end = day0 + timedelta(days=day - 1) + end_time.to_pytimedelta()
        if end >= start:
            result.append((start, end))
    return result


def _evenly_spaced(members: list[tarfile.TarInfo], max_hours: int | None) -> list[tarfile.TarInfo]:
    if max_hours is None or len(members) <= max_hours:
        return members
    # The first and last files are usually partial hours. Avoid spending a small
    # bounded budget on those endpoints when enough full interior hours exist.
    start, stop = (1, len(members) - 2) if len(members) - 2 >= max_hours else (0, len(members) - 1)
    indices = np.unique(np.round(np.linspace(start, stop, max_hours)).astype(int))
    return [members[int(index)] for index in indices]


def _complete_blocks(
    frame: pd.DataFrame,
    intervals: list[tuple[datetime, datetime]],
) -> list[np.ndarray]:
    timestamps = pd.to_datetime(
        frame["HEADER_TIMESTAMP"],
        format="%Y-%m-%d %H:%M:%S.%f",
        errors="coerce",
    )
    xyz = frame[["X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    good = np.isfinite(xyz).all(axis=1) & timestamps.notna().to_numpy()
    for start, end in intervals:
        good &= ~((timestamps >= start) & (timestamps <= end)).to_numpy()

    # Sensor gaps are boundaries even when no published QC interval covers them.
    stamp_ns = timestamps.astype("int64", copy=False).to_numpy()
    dt = np.diff(stamp_ns) / 1e9
    boundaries = np.flatnonzero((~good[1:]) | (~good[:-1]) | (dt > 2.5 / RATE_HZ)) + 1
    bounds = np.r_[0, boundaries, len(frame)]
    blocks = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        if not good[start:end].all():
            continue
        usable = ((end - start) // WINDOW_SAMPLES) * WINDOW_SAMPLES
        if usable:
            blocks.append(xyz[start : start + usable])
    return blocks


def convert(
    raw_dir: Path = DOWNLOADS,
    output_dir: Path = DS_DIR,
    limit_subjects: int | None = None,
    max_hours_per_subject: int | None = 24,
) -> bool:
    archives = sorted(raw_dir.glob("*.tar.bz2"))
    if limit_subjects is not None:
        archives = archives[:limit_subjects]
    if not archives:
        raise FileNotFoundError(
            f"no participant archives under {raw_dir}; run "
            "`python -m data.datasets.nhanes.fetch --subjects <N>`"
        )

    sessions_dir = output_dir / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    sessions_dir.mkdir(parents=True)
    labels: dict[str, list[str]] = {}
    stats: Counter = Counter()

    for archive_i, archive_path in enumerate(archives, start=1):
        seqn = archive_path.name.split(".", 1)[0]
        with tarfile.open(archive_path, mode="r:bz2") as archive:
            members = archive.getmembers()
            sensor_members = sorted(
                (member for member in members if member.name.endswith(".sensor.csv")),
                key=lambda member: _member_start(member.name),
            )
            if not sensor_members:
                stats["subjects_without_sensor_files"] += 1
                continue
            qc = _read_qc(archive, members)
            intervals = _qc_intervals(qc, _member_start(sensor_members[0].name))
            chosen = _evenly_spaced(sensor_members, max_hours_per_subject)
            blocks: list[np.ndarray] = []
            for member in chosen:
                handle = archive.extractfile(member)
                if handle is None:
                    stats["unreadable_hour"] += 1
                    continue
                frame = pd.read_csv(
                    io.BytesIO(handle.read()),
                    usecols=["HEADER_TIMESTAMP", "X", "Y", "Z"],
                )
                blocks.extend(_complete_blocks(frame, intervals))

        if not blocks:
            stats["subjects_without_complete_blocks"] += 1
            continue
        data = np.concatenate(blocks, axis=0)
        out = pd.DataFrame(data, columns=OUTPUT_COLUMNS)
        out.insert(0, "timestamp_sec", np.arange(len(out), dtype=np.float64) / RATE_HZ)
        out["subject"] = seqn
        session_id = f"nhanes_{seqn}_watch_wrist"
        destination = sessions_dir / session_id
        destination.mkdir()
        out.to_parquet(destination / "data.parquet", index=False)
        labels[session_id] = [UNLABELED]
        stats["windows"] += len(out) // WINDOW_SAMPLES
        stats["subjects"] += 1
        print(
            f"[nhanes] {archive_i:03d}/{len(archives)} {seqn}: "
            f"{len(out) / RATE_HZ / 3600:.2f} h, {len(out) // WINDOW_SAMPLES} windows"
        )

    if not labels:
        return False
    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "NHANES PAX80_G (bounded subset)",
                "source": "https://ftp.cdc.gov/pub/pax_g/",
                "num_subjects": stats["subjects"],
                "sampling_rate_hz": RATE_HZ,
                "channels": list(OUTPUT_COLUMNS),
                "unit": "g",
                "gravity_state": "present",
                "phase_a_only": True,
                "max_hours_per_subject": max_hours_per_subject,
                "note": "No activity labels; reserved marker __unlabeled__.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[nhanes] complete: {dict(stats)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DOWNLOADS)
    parser.add_argument("--output-dir", type=Path, default=DS_DIR)
    parser.add_argument("--limit-subjects", type=int)
    parser.add_argument("--max-hours-per-subject", type=int, default=24)
    args = parser.parse_args()
    if not convert(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        limit_subjects=args.limit_subjects,
        max_hours_per_subject=args.max_hours_per_subject,
    ):
        raise SystemExit("no NHANES sessions were produced")


if __name__ == "__main__":
    main()
