"""Lazy adapter for the Overhead Car Assembly (OCA) IMU dataset."""

from __future__ import annotations

import json
import math
import re
import zipfile
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
    split_at_clock_gaps,
)


_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "sources" / "oca"
_ACC_M_S2_TO_G = 1.0 / 9.80665
_GYRO_DEG_S_TO_RAD_S = math.pi / 180.0
_HARD_GAP_SEC = 1.0
_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")

# Arm streams are first because they are closest to the phone/watch deployment
# setting, while the two vest-mounted chest streams remain available.
_STREAM_ORDER = ("0", "2", "1", "3")
_ARM_SUPPORT = {
    "P0-R0": "none",
    "P0-R1": "until_17m24s",
    "P0-R2": "entire_session",
    "P1-R0": "none",
    "P1-R1": "until_16m50s",
    "P1-R2": "until_14m14s",
    "P2-R0": "none",
    "P2-R1": "none",
    "P2-R2": "none",
    "P2-R3": "none",
    "P3-R0": "none",
    "P4-R0": "entire_session",
}


def _resolve_archive(root: Path | None) -> Path:
    candidate = Path(root) if root is not None else _DEFAULT_ROOT
    paths = (
        candidate,
        candidate / "OCA.zip",
        candidate / "downloads" / "OCA.zip",
    )
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".zip":
            return path
    raise FileNotFoundError(f"OCA.zip not found beneath {candidate}")


def _split_lookup(metadata: Mapping[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    splits = metadata.get("benchmark_splits")
    if not isinstance(splits, Mapping):
        raise ValueError("OCA metadata has no benchmark_splits mapping")
    for split, files in splits.items():
        if not isinstance(split, str) or not isinstance(files, list):
            raise ValueError("invalid OCA benchmark_splits entry")
        for filename in files:
            if not isinstance(filename, str) or filename in result:
                raise ValueError(f"invalid or duplicate OCA split file: {filename!r}")
            result[filename] = split
    return result


def _event_runs(
    timestamps_sec: np.ndarray,
    labels: np.ndarray,
    label_map: Mapping[str, str],
    *,
    source_row_offset: int,
) -> tuple[EventInterval, ...]:
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], changes, [len(labels)]))
    events = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        label_id = int(labels[start])
        label = label_map.get(str(label_id))
        if label is None:
            raise ValueError(f"OCA contains unknown label id {label_id}")

        # Row labels describe sample support. Internal runs end at the first
        # timestamp of the following run; the terminal run ends at the last
        # source timestamp, avoiding an extrapolated annotation boundary.
        end_sec = (
            timestamps_sec[stop] if stop < len(labels) else timestamps_sec[stop - 1]
        )
        if end_sec <= timestamps_sec[start]:
            raise ValueError("OCA contains a zero-duration terminal label run")
        events.append(
            EventInterval(
                start_sec=float(timestamps_sec[start]),
                end_sec=float(end_sec),
                label=label,
                annotation_kind="sample_label_run",
                metadata={
                    "source_label_id": label_id,
                    "source_start_row": source_row_offset + int(start),
                    "source_stop_row": source_row_offset + int(stop),
                },
            )
        )
    return tuple(events)


def _stream(
    frame: pd.DataFrame,
    timestamps_sec: np.ndarray,
    *,
    imu_id: str,
    placement: str,
) -> SensorStream:
    columns = [
        f"imu{imu_id}_acc_x",
        f"imu{imu_id}_acc_y",
        f"imu{imu_id}_acc_z",
        f"imu{imu_id}_gyr_x",
        f"imu{imu_id}_gyr_y",
        f"imu{imu_id}_gyr_z",
    ]
    values = frame.loc[:, columns].to_numpy(dtype=np.float32, copy=True)
    valid = np.isfinite(values)
    values[:, :3] *= np.float32(_ACC_M_S2_TO_G)
    values[:, 3:] *= np.float32(_GYRO_DEG_S_TO_RAD_S)
    positive_delta = np.diff(timestamps_sec)
    nominal_rate_hz = (
        float(1.0 / np.median(positive_delta)) if len(positive_delta) else None
    )
    is_arm = placement.endswith("upper arm")
    return SensorStream(
        stream_id=f"imu{imu_id}",
        placement=placement,
        device="BNO055 IMU on sensor vest",
        timestamps_sec=timestamps_sec,
        values=values,
        channels=_CHANNELS,
        valid=valid,
        gravity_state="present",
        nominal_rate_hz=nominal_rate_hz,
        metadata={
            "source_imu_id": int(imu_id),
            "source_placement": placement,
            "application_compatible_placement": is_arm,
            "source_acceleration_unit": "m/s^2",
            "source_gyroscope_unit": "degree/s",
        },
    )


def iter_recordings(
    root: Path | None = None,
    limit: int | None = None,
) -> Iterator[RawRecording]:
    """Yield native-rate, quality-contiguous OCA recording parts.

    The source ZIP is opened lazily and one session CSV is materialized at a
    time. ``limit`` counts yielded quality-contiguous recordings, so the two
    sides of the known P0-R0 clock gap count separately.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    if limit == 0:
        return

    archive_path = _resolve_archive(root)
    yielded = 0
    with zipfile.ZipFile(archive_path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        label_map = metadata.get("label_map")
        placement_map = metadata.get("id_map")
        if not isinstance(label_map, Mapping) or not isinstance(placement_map, Mapping):
            raise ValueError("OCA metadata is missing label_map or id_map")
        split_by_file = _split_lookup(metadata)
        csv_names = sorted(name for name in archive.namelist() if name.endswith(".csv"))
        if set(csv_names) != set(split_by_file):
            raise ValueError(
                "OCA archive sessions do not match its official split metadata"
            )

        for filename in csv_names:
            session_id = Path(filename).stem
            match = re.fullmatch(r"(P\d+)-R\d+", session_id)
            if match is None:
                raise ValueError(f"unexpected OCA session name: {filename}")
            frame = pd.read_csv(archive.open(filename))
            timestamps_sec = frame["timestamp"].to_numpy(dtype=np.float64) / 1000.0
            labels = frame["label"].to_numpy(dtype=np.int64)
            parts = split_at_clock_gaps(timestamps_sec, max_gap_sec=_HARD_GAP_SEC)

            for part_index, row_slice in enumerate(parts):
                segment_frame = frame.iloc[row_slice]
                segment_timestamps = timestamps_sec[row_slice]
                segment_labels = labels[row_slice]
                streams = tuple(
                    _stream(
                        segment_frame,
                        segment_timestamps,
                        imu_id=imu_id,
                        placement=str(placement_map[imu_id]),
                    )
                    for imu_id in _STREAM_ORDER
                )
                recording_id = (
                    session_id
                    if len(parts) == 1
                    else f"{session_id}-part{part_index:02d}"
                )
                yield RawRecording(
                    dataset="oca",
                    recording_id=recording_id,
                    subject_id=match.group(1),
                    session_id=session_id,
                    streams=streams,
                    events=_event_runs(
                        segment_timestamps,
                        segment_labels,
                        label_map,
                        source_row_offset=int(row_slice.start or 0),
                    ),
                    split=split_by_file[filename],
                    metadata={
                        "source_file": filename,
                        "quality_part_index": part_index,
                        "quality_part_count": len(parts),
                        "source_timestamp_unit": "milliseconds",
                        "official_split": split_by_file[filename],
                        "arm_support": _ARM_SUPPORT[session_id],
                        "source_placements": {
                            f"imu{imu_id}": str(placement_map[imu_id])
                            for imu_id in sorted(placement_map)
                        },
                        "preferred_application_streams": ("imu0", "imu2"),
                    },
                )
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
