"""Nested-ZIP streaming adapter for COPS bilateral hourly accelerometry."""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

import numpy as np
import pandas as pd

from applications.motion_monitoring.data.contracts import RawRecording, SensorStream


_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "sources" / "cops"
_RATE_HZ = 100.0
_CHANNELS = ("acc_x", "acc_y", "acc_z")
_ARCHIVE_RE = re.compile(r"COPS-(?P<subject>\d+)\.zip$")
_HOURLY_RE = re.compile(
    r"(?P<subject>COPS-\d+)_Day(?P<day>\d+)_(?P<hours>\d{2}h-\d{2}h)_(?P<side>left|right)Wrist\.zip$"
)


def _archive_paths(root: Path | None) -> list[Path]:
    source = _DEFAULT_ROOT if root is None else Path(root)
    for candidate in (source / "downloads", source):
        paths = [
            path
            for path in candidate.glob("COPS-*.zip")
            if _ARCHIVE_RE.fullmatch(path.name)
        ]
        if paths:
            return sorted(
                paths,
                key=lambda path: int(_ARCHIVE_RE.fullmatch(path.name).group("subject")),
            )
    raise FileNotFoundError(f"COPS participant archives not found beneath {source}")


def _time_seconds(values: pd.Series) -> np.ndarray:
    parts = values.astype(str).str.split(":", expand=True)
    if parts.shape[1] != 3:
        raise ValueError("COPS Time must use HH:MM:SS.sss")
    seconds = (
        parts[0].astype(np.float64).to_numpy() * 3600.0
        + parts[1].astype(np.float64).to_numpy() * 60.0
        + parts[2].astype(np.float64).to_numpy()
    )
    wraps = np.concatenate(([0], np.cumsum(np.diff(seconds) < -12 * 3600)))
    seconds = seconds + wraps * 86400.0
    seconds -= seconds[0]
    if np.any(np.diff(seconds) <= 0):
        raise ValueError("COPS hourly clock is not strictly increasing")
    return seconds.astype(np.float64, copy=False)


def _read_stream(outer: ZipFile, member: str, side: str) -> SensorStream:
    with ZipFile(BytesIO(outer.read(member))) as inner:
        csv_members = [
            name for name in inner.namelist() if name.lower().endswith(".csv")
        ]
        if len(csv_members) != 1:
            raise ValueError(f"{member} must contain exactly one CSV")
        frame = pd.read_csv(
            BytesIO(inner.read(csv_members[0])),
            sep=";",
            usecols=["Time", "X", "Y", "Z"],
        )
    values = frame[["X", "Y", "Z"]].to_numpy(dtype=np.float32, copy=True)
    valid = np.isfinite(values)
    return SensorStream(
        stream_id=f"geneactiv_{side}_wrist",
        placement=f"{side}_wrist",
        device="GENEActiv wrist accelerometer",
        timestamps_sec=_time_seconds(frame["Time"]),
        values=np.nan_to_num(values),
        channels=_CHANNELS,
        valid=valid,
        gravity_state="present",
        nominal_rate_hz=_RATE_HZ,
        metadata={
            "source_member": member,
            "source_acceleration_unit": "g",
            "side": side,
        },
    )


def _diary(outer: ZipFile, subject: str) -> pd.DataFrame:
    name = f"{subject}/{subject}_symptomdiary.csv"
    return pd.read_csv(BytesIO(outer.read(name)), sep=";")


def _target_metadata(row: pd.Series | None) -> dict[str, object]:
    if row is None:
        return {"diary_linked": False}
    result: dict[str, object] = {"diary_linked": True}
    for name in ("KinesiaScore", "TremorScore", "FreezingScore", "FallScore"):
        value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
        result[name] = None if pd.isna(value) else float(value)
    result["visit"] = str(row.get("Visit", ""))
    result["wearable_availability"] = str(row.get("WearableDataAvailability", ""))
    return result


def iter_recordings(
    root: Path | None = None,
    limit: int | None = None,
) -> Iterator[RawRecording]:
    """Yield one bilateral recording per observed participant-hour."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    yielded = 0
    for archive_path in _archive_paths(root):
        if limit is not None and yielded >= limit:
            break
        subject = archive_path.stem
        with ZipFile(archive_path) as outer:
            diary = _diary(outer, subject)
            by_zip: dict[str, pd.Series] = {}
            for _, row in diary.iterrows():
                for column in ("WearableDataLeftZIP", "WearableDataRightZIP"):
                    name = row.get(column)
                    if isinstance(name, str) and name.strip():
                        by_zip[name.strip()] = row
            groups: dict[tuple[int, str], dict[str, str]] = {}
            for member in outer.namelist():
                match = _HOURLY_RE.fullmatch(PurePosixPath(member).name)
                if match is None:
                    continue
                key = (int(match.group("day")), match.group("hours"))
                groups.setdefault(key, {})[match.group("side")] = member
            for (day, hours), sides in sorted(groups.items()):
                if limit is not None and yielded >= limit:
                    break
                streams = tuple(
                    _read_stream(outer, sides[side], side)
                    for side in ("left", "right")
                    if side in sides
                )
                diary_rows = {
                    id(by_zip.get(PurePosixPath(member).name)): by_zip.get(
                        PurePosixPath(member).name
                    )
                    for member in sides.values()
                    if by_zip.get(PurePosixPath(member).name) is not None
                }
                diary_row = next(iter(diary_rows.values()), None)
                if len(diary_rows) > 1:
                    raise ValueError(
                        f"COPS diary links disagree across wrists for {subject} day {day} {hours}"
                    )
                yield RawRecording(
                    dataset="cops",
                    recording_id=f"cops:{subject}:day{day}:{hours}",
                    subject_id=subject,
                    session_id=f"{subject}/day{day}",
                    streams=streams,
                    events=(),
                    split="evaluation",
                    metadata={
                        "day": day,
                        "hour_interval": hours,
                        "observation_kind": "free_living_hour",
                        "bounded_execution_annotations": False,
                        **_target_metadata(diary_row),
                    },
                )
                yielded += 1
