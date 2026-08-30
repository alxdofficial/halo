"""Lazy adapter for the RecoFit multi-activity MATLAB release."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader, miMATRIX

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
    split_at_clock_gaps,
)


_FILENAME = "exercise_data.50.0000_multionly.mat"
_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "sources" / "recofit"
_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
_DEG_TO_RAD = np.float32(np.pi / 180.0)


def _source_path(root: Path | None) -> Path:
    base = _DEFAULT_ROOT if root is None else Path(root)
    candidates = (
        (base,)
        if base.is_file()
        else (base / _FILENAME, base / "downloads" / _FILENAME)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"RecoFit source {_FILENAME!r} was not found under {base}. "
        "Run the application-data acquisition step first."
    )


def _iter_source_visits(path: Path) -> Iterator[tuple[int, Any]]:
    """Decode one subject cell at a time from the compressed MATLAB v5 file.

    The release stores all subjects in one compressed top-level cell. Public
    ``loadmat`` APIs inflate that entire 1.57 GB payload. Reading the nested
    matrix elements from SciPy's bounded zlib stream keeps only one subject's
    visits resident while retaining the source structure verbatim.
    """

    with path.open("rb") as handle:
        reader = MatFile5Reader(
            handle,
            struct_as_record=False,
            squeeze_me=True,
            chars_as_strings=True,
        )
        reader.read_file_header()
        reader.initialize_read()
        header, _ = reader.read_var_header()
        if header.name != b"subject_data" or tuple(header.dims) != (94, 1):
            raise ValueError(
                "unexpected RecoFit MATLAB layout: expected subject_data [94, 1]"
            )

        matrix_reader = reader._matrix_reader
        for subject_cell_index in range(int(np.prod(header.dims))):
            matrix_type, _ = matrix_reader.read_full_tag()
            if matrix_type != miMATRIX:
                raise ValueError(
                    f"invalid RecoFit subject cell {subject_cell_index}: "
                    f"MATLAB type {matrix_type}, expected {miMATRIX}"
                )
            subject_header = matrix_reader.read_header(False)
            subject_data = matrix_reader.array_from_header(subject_header, process=True)
            if np.size(subject_data) == 0:
                continue
            for visit in np.atleast_1d(subject_data).reshape(-1):
                yield subject_cell_index, visit


def _as_text(value: Any) -> str | None:
    array = np.asarray(value)
    if array.size == 0:
        return None
    if array.ndim == 0:
        return str(array.item())
    return "".join(str(item) for item in array.reshape(-1)).strip() or None


def _finite_scalar(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _activity_groups(path: Path) -> dict[str, tuple[str, ...]]:
    constants = loadmat(
        path,
        variable_names=["exerciseConstants"],
        simplify_cells=True,
    )["exerciseConstants"]
    groups: dict[str, list[str]] = {}
    for group_name, labels in np.asarray(
        constants["usefulActivityGroupings"], dtype=object
    ):
        for label in np.atleast_1d(labels):
            groups.setdefault(str(label), []).append(str(group_name))
    return {label: tuple(names) for label, names in groups.items()}


def _annotation_kind(label: str, activity_groups: tuple[str, ...]) -> str:
    if "Non-exercise" in activity_groups:
        return "background"
    if "Junk" in activity_groups:
        return "source_junk"
    return "set"


def _event_metadata(row: np.ndarray) -> dict[str, Any]:
    raw_count = _finite_scalar(row[4])
    metadata: dict[str, Any] = {
        "notes": _as_text(row[3]),
        "repetition_count": (
            int(raw_count) if raw_count is not None and raw_count >= 0 else None
        ),
        "source_repetition_count": int(raw_count) if raw_count is not None else None,
        "source_aligned_start_sec": _finite_scalar(row[5]),
    }
    annotation = row[6]
    for field_name in getattr(annotation, "_fieldnames", ()):
        metadata[f"source_{field_name}"] = _finite_scalar(
            getattr(annotation, field_name)
        )
    return metadata


def _events_for_clock_segment(
    activity_rows: np.ndarray,
    *,
    clock_start_sec: float,
    clock_end_sec: float,
    activity_groups: dict[str, tuple[str, ...]],
) -> tuple[EventInterval, ...]:
    events: list[EventInterval] = []
    rows = np.asarray(activity_rows, dtype=object)
    if rows.ndim == 1:
        rows = rows[None, :]

    for row_index, row in enumerate(rows):
        label = str(row[0])
        source_start = float(row[1])
        source_end = float(row[2])
        start = max(source_start, clock_start_sec)
        end = min(source_end, clock_end_sec)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue

        metadata = _event_metadata(row)
        metadata.update(
            {
                "source_activity_groups": activity_groups.get(label, ()),
                "source_row_index": row_index,
                "source_start_sec": source_start,
                "source_end_sec": source_end,
                "clipped_to_clock_segment": (
                    start != source_start or end != source_end
                ),
            }
        )
        events.append(
            EventInterval(
                start_sec=start,
                end_sec=end,
                label=label,
                annotation_kind=_annotation_kind(label, activity_groups.get(label, ())),
                metadata=metadata,
            )
        )
    return tuple(events)


def _visit_recordings(
    subject_cell_index: int,
    visit: Any,
    activity_groups: dict[str, tuple[str, ...]],
) -> Iterator[RawRecording]:
    accel = np.asarray(visit.data.accelDataMatrix)
    gyro = np.asarray(visit.data.gyroDataMatrix)
    if accel.ndim != 2 or accel.shape[1] != 4 or gyro.shape != accel.shape:
        raise ValueError(
            f"RecoFit visit {visit.fileIndex} has invalid IMU shapes "
            f"{accel.shape} and {gyro.shape}"
        )
    if not np.array_equal(accel[:, 0], gyro[:, 0]):
        raise ValueError(
            f"RecoFit visit {visit.fileIndex} has unsynchronized accelerometer "
            "and gyroscope clocks"
        )

    # Detach the clock from the four-column source matrix. Otherwise callers
    # retaining recordings also retain every full float64 accelerometer array.
    timestamps = np.array(accel[:, 0], dtype=np.float64, copy=True, order="C")
    values = np.empty((len(timestamps), 6), dtype=np.float32)
    values[:, :3] = accel[:, 1:4]
    np.multiply(gyro[:, 1:4], _DEG_TO_RAD, out=values[:, 3:], casting="unsafe")
    valid = np.isfinite(values)

    nominal_rate_hz = float(visit.sampleRate)
    max_gap_sec = 5.0 / nominal_rate_hz
    clock_slices = split_at_clock_gaps(timestamps, max_gap_sec=max_gap_sec)
    subject_id = str(int(visit.subjectID))
    visit_id = int(visit.fileIndex)
    incomplete = bool(int(visit.incompleteData))

    for clock_segment_index, clock_slice in enumerate(clock_slices):
        segment_timestamps = timestamps[clock_slice]
        suffix = "" if len(clock_slices) == 1 else f"_clock{clock_segment_index:02d}"
        session_id = f"visit_{visit_id:03d}{suffix}"
        stream = SensorStream(
            stream_id="right_forearm_imu",
            placement="right_forearm",
            device="SparkFun Razor IMU armband",
            timestamps_sec=segment_timestamps,
            values=values[clock_slice],
            channels=_CHANNELS,
            valid=valid[clock_slice],
            gravity_state="present",
            nominal_rate_hz=nominal_rate_hz,
            metadata={
                "source_acceleration_unit": "g",
                "source_gyroscope_unit": "degree/s",
                "gyroscope_conversion": "degree/s_to_rad/s",
                "source_fields": ("accelDataMatrix", "gyroDataMatrix"),
            },
        )
        yield RawRecording(
            dataset="recofit",
            recording_id=f"recofit_subject_{subject_id}_{session_id}",
            subject_id=subject_id,
            session_id=session_id,
            streams=(stream,),
            events=_events_for_clock_segment(
                visit.activityStartMatrix,
                clock_start_sec=float(segment_timestamps[0]),
                clock_end_sec=float(segment_timestamps[-1]),
                activity_groups=activity_groups,
            ),
            metadata={
                "source_subject_cell_index_matlab": subject_cell_index + 1,
                "source_subject_cell_index_zero_based": subject_cell_index,
                "source_file_index": visit_id,
                "source_recording_id_matlab_datenum": float(visit.recordingID),
                "source_incomplete": incomplete,
                "source_activity_name": str(visit.activityName),
                "source_master_file_token": str(visit.masterFileToken),
                "source_master_token": str(visit.masterToken),
                "clock_segment_index": clock_segment_index,
                "clock_segment_count": len(clock_slices),
                "documented_master_stream_only": True,
            },
        )


def iter_recordings(
    root: Path | None = None,
    limit: int | None = None,
) -> Iterator[RawRecording]:
    """Yield native-rate, quality-contiguous RecoFit lab visits.

    RecoFit annotations delimit exercise sets rather than individual
    repetitions. Their labels remain verbatim and repetition counts are stored
    as event metadata. Incomplete visits are yielded and explicitly marked.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    if limit == 0:
        return

    source_path = _source_path(root)
    activity_groups = _activity_groups(source_path)
    yielded = 0
    for subject_cell_index, visit in _iter_source_visits(source_path):
        for recording in _visit_recordings(subject_cell_index, visit, activity_groups):
            yield recording
            yielded += 1
            if limit is not None and yielded >= limit:
                return
