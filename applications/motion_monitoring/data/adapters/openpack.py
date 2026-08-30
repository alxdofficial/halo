"""Lazy adapter for the OpenPack non-RGB subject archives."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
    split_at_clock_gaps,
)


_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "sources" / "openpack"
_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
_GYRO_SLICE = slice(3, 6)
_GAP_THRESHOLD_SEC = 0.2
_MIN_PART_SAMPLES = 2
_SUBJECT_ALIASES = {
    "U0202": "U0105",
    "U0203": "U0108",
    "U0204": "U0110",
    "U0205": "U0107",
    "U0210": "U0103",
}


def _archive_paths(root: Path | None) -> tuple[Path, ...]:
    source = _DEFAULT_ROOT if root is None else Path(root)
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise ValueError(f"OpenPack source file must be a ZIP archive: {source}")
        return (source,)

    downloads = source / "downloads"
    archive_dir = downloads if downloads.is_dir() else source
    archives = tuple(sorted(archive_dir.glob("U*.zip")))
    if not archives:
        raise FileNotFoundError(f"no OpenPack subject archives found under {source}")
    return archives


def _read_wrist_stream(
    archive: ZipFile,
    member: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    columns = ("unixtime", *_CHANNELS)
    with archive.open(member) as handle:
        frame = pd.read_csv(
            handle,
            usecols=columns,
            dtype={"unixtime": np.float64, **{name: np.float32 for name in _CHANNELS}},
        )

    timestamp_ms = frame["unixtime"].to_numpy(dtype=np.float64, copy=False)
    if not len(timestamp_ms) or not np.isfinite(timestamp_ms).all():
        raise ValueError(f"invalid OpenPack timestamps in {member}")
    timestamps_sec = timestamp_ms / 1000.0

    values = frame.loc[:, list(_CHANNELS)].to_numpy(dtype=np.float32, copy=True)
    valid = np.isfinite(values)
    values[~valid] = 0.0
    values[:, _GYRO_SLICE] *= np.float32(np.pi / 180.0)

    positive_steps = np.diff(timestamps_sec)
    positive_steps = positive_steps[positive_steps > 0]
    measured_rate_hz = (
        float(1.0 / np.median(positive_steps)) if len(positive_steps) else 30.0
    )
    return timestamps_sec, values, valid, measured_rate_hz


def _timestamp_seconds(values: pd.Series, *, member: str, column: str) -> np.ndarray:
    parsed = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
    if parsed.isna().any():
        raise ValueError(f"invalid {column} timestamp in {member}")
    return parsed.astype("int64").to_numpy(dtype=np.float64) / 1e9


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _source_identifier(value: Any) -> str | int | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return str(value)


def _read_interval_table(archive: ZipFile, member: str) -> pd.DataFrame:
    try:
        with archive.open(member) as handle:
            frame = pd.read_csv(handle)
    except KeyError:
        return pd.DataFrame()
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["_start_sec"] = _timestamp_seconds(
        frame["start"], member=member, column="start"
    )
    frame["_end_sec"] = _timestamp_seconds(frame["end"], member=member, column="end")
    return frame


def _session_events(
    archive: ZipFile,
    session_id: str,
) -> tuple[tuple[EventInterval, ...], dict[str, int]]:
    action_member = f"annotation/openpack-actions/{session_id}.csv"
    operation_member = f"annotation/openpack-operations/{session_id}.csv"
    actions = _read_interval_table(archive, action_member)
    operations = _read_interval_table(archive, operation_member)

    events: list[EventInterval] = []
    excluded_zero_duration = 0
    excluded_other_invalid = 0

    for row in actions.to_dict("records"):
        start_sec = float(row["_start_sec"])
        end_sec = float(row["_end_sec"])
        if end_sec == start_sec:
            excluded_zero_duration += 1
            continue
        if end_sec < start_sec:
            excluded_other_invalid += 1
            continue
        events.append(
            EventInterval(
                start_sec=start_sec,
                end_sec=end_sec,
                label=str(row["action"]),
                annotation_kind="fine_action",
                metadata={
                    "source_uuid": str(row["uuid"]),
                    "source_label_id": _source_identifier(row.get("id")),
                    "operation": _optional_text(row.get("operation")),
                    "box": _source_identifier(row.get("box")),
                    "box_size": _optional_text(row.get("box_size")),
                    "work_position": _optional_text(row.get("work_position")),
                },
            )
        )

    valid_operations: list[dict[str, Any]] = []
    for row in operations.to_dict("records"):
        start_sec = float(row["_start_sec"])
        end_sec = float(row["_end_sec"])
        if end_sec == start_sec:
            excluded_zero_duration += 1
            continue
        if end_sec < start_sec:
            excluded_other_invalid += 1
            continue
        valid_operations.append(row)
        events.append(
            EventInterval(
                start_sec=start_sec,
                end_sec=end_sec,
                label=str(row["operation"]),
                annotation_kind="operation",
                metadata={
                    "source_uuid": str(row["uuid"]),
                    "source_label_id": _source_identifier(row.get("id")),
                    "box": _source_identifier(row.get("box")),
                },
            )
        )

    operations_by_box: dict[str | int, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_operations:
        box = _source_identifier(row.get("box"))
        if box is not None:
            operations_by_box[box].append(row)
    for box, rows in operations_by_box.items():
        events.append(
            EventInterval(
                start_sec=min(float(row["_start_sec"]) for row in rows),
                end_sec=max(float(row["_end_sec"]) for row in rows),
                label="packing_cycle",
                annotation_kind="box_cycle",
                metadata={
                    "box": box,
                    "operation_count": len(rows),
                    "boundary_source": "min/max source operation boundaries",
                },
            )
        )

    events.sort(
        key=lambda event: (event.start_sec, event.end_sec, event.annotation_kind)
    )
    counts = {
        "source_fine_action_rows": len(actions),
        "source_operation_rows": len(operations),
        "excluded_zero_duration_annotations": excluded_zero_duration,
        "excluded_other_invalid_annotations": excluded_other_invalid,
    }
    return tuple(events), counts


def _assign_events_to_parts(
    events: tuple[EventInterval, ...],
    timestamps_sec: np.ndarray,
    parts: tuple[slice, ...],
) -> tuple[tuple[tuple[EventInterval, ...], ...], int, int]:
    bounds = tuple(
        (float(timestamps_sec[part.start]), float(timestamps_sec[part.stop - 1]))
        for part in parts
    )
    assigned: list[list[EventInterval]] = [[] for _ in parts]
    without_sensor_overlap = 0
    clipped_to_sensor_span = 0
    for event in events:
        overlapping_parts = 0
        for part_index, (start, end) in enumerate(bounds):
            clipped_start = max(event.start_sec, start)
            clipped_end = min(event.end_sec, end)
            if clipped_end <= clipped_start:
                continue
            overlapping_parts += 1
            was_clipped = (
                clipped_start != event.start_sec or clipped_end != event.end_sec
            )
            if was_clipped:
                clipped_to_sensor_span += 1
            assigned[part_index].append(
                EventInterval(
                    start_sec=clipped_start,
                    end_sec=clipped_end,
                    label=event.label,
                    annotation_kind=event.annotation_kind,
                    metadata={
                        **event.metadata,
                        "source_start_sec": event.start_sec,
                        "source_end_sec": event.end_sec,
                        "clipped_to_observed_sensor_span": was_clipped,
                    },
                )
            )
        if not overlapping_parts:
            without_sensor_overlap += 1
    return (
        tuple(tuple(part_events) for part_events in assigned),
        without_sensor_overlap,
        clipped_to_sensor_span,
    )


def iter_recordings(
    root: Path | None = None,
    limit: int | None = None,
) -> Iterator[RawRecording]:
    """Yield native-clock right-wrist OpenPack sessions without materializing a corpus."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return

    yielded = 0
    for archive_path in _archive_paths(root):
        source_user_id = archive_path.stem
        subject_id = _SUBJECT_ALIASES.get(source_user_id, source_user_id)
        with ZipFile(archive_path) as archive:
            members = sorted(
                member
                for member in archive.namelist()
                if member.startswith("atr/atr01/") and member.endswith(".csv")
            )
            for member in members:
                source_session_id = PurePosixPath(member).stem
                timestamps_sec, values, valid, measured_rate_hz = _read_wrist_stream(
                    archive, member
                )
                source_parts = split_at_clock_gaps(
                    timestamps_sec,
                    max_gap_sec=_GAP_THRESHOLD_SEC,
                )
                parts = tuple(
                    part
                    for part in source_parts
                    if part.stop - part.start >= _MIN_PART_SAMPLES
                )
                dropped_short_parts = len(source_parts) - len(parts)
                dropped_short_samples = sum(
                    part.stop - part.start
                    for part in source_parts
                    if part.stop - part.start < _MIN_PART_SAMPLES
                )
                session_events, annotation_counts = _session_events(
                    archive, source_session_id
                )
                (
                    part_events,
                    intervals_without_sensor_overlap,
                    intervals_clipped_to_sensor_span,
                ) = _assign_events_to_parts(session_events, timestamps_sec, parts)

                for part_index, (part, events) in enumerate(zip(parts, part_events)):
                    part_timestamps = timestamps_sec[part].copy()
                    part_values = values[part].copy()
                    part_valid = valid[part].copy()
                    recording_id = f"openpack:{source_user_id}:{source_session_id}:atr01:part{part_index:02d}"
                    stream = SensorStream(
                        stream_id="atr01",
                        placement="right_wrist",
                        device="ATR TSND151",
                        timestamps_sec=part_timestamps,
                        values=part_values,
                        channels=_CHANNELS,
                        valid=part_valid,
                        gravity_state="present",
                        nominal_rate_hz=30.0,
                        metadata={
                            "source_archive": archive_path.name,
                            "source_member": member,
                            "measured_rate_hz": measured_rate_hz,
                            "acceleration_unit": "g",
                            "gyroscope_source_unit": "degree/s",
                            "gyroscope_output_unit": "rad/s",
                        },
                    )
                    yield RawRecording(
                        dataset="openpack",
                        recording_id=recording_id,
                        subject_id=subject_id,
                        session_id=f"{source_user_id}/{source_session_id}",
                        streams=(stream,),
                        events=events,
                        metadata={
                            "source_user_id": source_user_id,
                            "source_session_id": source_session_id,
                            "identity_alias_applied": subject_id != source_user_id,
                            "selected_device": "atr01",
                            "selected_placement": "right_wrist",
                            "quality_part_index": part_index,
                            "quality_part_count": len(parts),
                            "source_quality_part_count": len(source_parts),
                            "dropped_short_part_count": dropped_short_parts,
                            "dropped_short_sample_count": dropped_short_samples,
                            "clock_gap_threshold_sec": _GAP_THRESHOLD_SEC,
                            "source_intervals_without_sensor_overlap": (
                                intervals_without_sensor_overlap
                            ),
                            "interval_fragments_clipped_to_sensor_span": (
                                intervals_clipped_to_sensor_span
                            ),
                            **annotation_counts,
                        },
                    )
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return
