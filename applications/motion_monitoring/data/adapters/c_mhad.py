"""Lazy adapter for the C-MHAD continuous inertial timelines.

The source annotations use elapsed video time. C-MHAD reports that Bluetooth
latency removes samples from the beginning of most inertial streams, so this
adapter represents that unobserved interval on the timestamp axis instead of
padding it with fabricated zero measurements.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from applications.motion_monitoring.data.contracts import (
    CANONICAL_CHANNELS,
    EventInterval,
    RawRecording,
    SensorStream,
    split_at_clock_gaps,
)


_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "sources" / "c_mhad" / "raw"
_STANDARD_GRAVITY_M_S2 = 9.80665
_SOURCE_RATE_HZ = 50.0
# The authors' loader computes ``MissFrames = 6005 - len(pd.read_csv(csv))``.
# Pandas includes the two CAL/unit rows in that length, so the equivalent count
# after removing those rows is 6003 - N. Preserve that signed alignment offset:
# some released streams are longer, rather than pretending every discrepancy is
# missing data at the beginning.
_OFFICIAL_FRAME_OFFSET_BASE = 6003
_MAX_CLOCK_GAP_SEC = 0.1
_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

_CSV_COLUMNS = (
    "Time Stamp",
    "Low Noise Accelerometer X",
    "Low Noise Accelerometer Y",
    "Low Noise Accelerometer Z",
    "Gyroscope X",
    "Gyroscope Y",
    "Gyroscope Z",
)

_APPLICATIONS = {
    "TVGestureApplication": {
        "slug": "tv_gestures",
        "file_token": "tv",
        "placement": "right_wrist",
        "workbook_prefix": "ActionOfInterestTVSubject",
        "labels": {
            1: "swipe_left",
            2: "swipe_right",
            3: "wave",
            4: "circle_clockwise",
            5: "circle_counterclockwise",
        },
    },
    "TransitionMovementsApplication": {
        "slug": "transition_movements",
        "file_token": "tr",
        "placement": "middle_waist",
        "workbook_prefix": "ActionOfInterestTraSubject",
        "labels": {
            1: "stand_to_sit",
            2: "sit_to_stand",
            3: "sit_to_lie",
            4: "lie_to_sit",
            5: "lie_to_stand",
            6: "stand_to_lie",
            7: "stand_to_fall",
        },
    },
}

_FILE_PATTERN = re.compile(
    r"inertial_sub(?P<subject>\d+)_(?P<token>tv|tr)(?P<run>\d+)\.csv$"
)


def _worksheet_rows(path: Path) -> list[list[str | float | None]]:
    """Read the first XLSX worksheet without requiring an Excel engine."""

    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.itertext()) for node in root.findall(f"{_XLSX_NS}si")
            ]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: list[list[str | float | None]] = []
    for row in sheet.findall(f".//{_XLSX_NS}row"):
        cells: dict[int, str | float | None] = {}
        for cell in row.findall(f"{_XLSX_NS}c"):
            reference = cell.attrib["r"]
            letters = re.match(r"[A-Z]+", reference)
            if letters is None:
                raise ValueError(f"invalid XLSX cell reference {reference!r} in {path}")
            column = 0
            for letter in letters.group(0):
                column = column * 26 + ord(letter) - ord("A") + 1
            value_node = cell.find(f"{_XLSX_NS}v")
            if value_node is None:
                value: str | float | None = None
            elif cell.attrib.get("t") == "s":
                value = shared[int(value_node.text)]
            else:
                value = float(value_node.text)
            cells[column - 1] = value
        rows.append([cells.get(index) for index in range(max(cells, default=-1) + 1)])
    return rows


def _read_annotations(
    path: Path,
    labels: dict[int, str],
) -> dict[int, tuple[EventInterval, ...]]:
    rows = _worksheet_rows(path)
    expected_header = ["Video", "Action", "StartTime(Seconds)", "EndTime(Seconds)"]
    if not rows or rows[0] != expected_header:
        raise ValueError(f"unexpected C-MHAD annotation header in {path}")

    by_run: dict[int, list[EventInterval]] = {}
    for row_index, row in enumerate(rows[1:], start=2):
        if not row or row[0] is None:
            continue
        if len(row) < 4 or any(value is None for value in row[:4]):
            raise ValueError(f"incomplete C-MHAD annotation at {path}:{row_index}")
        run = int(row[0])
        label_id = int(row[1])
        if label_id not in labels:
            raise ValueError(f"unknown C-MHAD action {label_id} at {path}:{row_index}")
        by_run.setdefault(run, []).append(
            EventInterval(
                start_sec=float(row[2]),
                end_sec=float(row[3]),
                label=labels[label_id],
                metadata={
                    "source_label_id": label_id,
                    "source_annotation_row": row_index,
                    "usage": "scoring_only",
                },
            )
        )
    return {run: tuple(events) for run, events in by_run.items()}


def _source_files(root: Path) -> list[tuple[Path, str, int, int]]:
    files: list[tuple[Path, str, int, int]] = []
    for path in root.rglob("inertial_sub*_*.csv"):
        match = _FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        application = next((name for name in _APPLICATIONS if name in path.parts), None)
        if application is None:
            raise ValueError(f"cannot identify C-MHAD application for {path}")
        config = _APPLICATIONS[application]
        if match.group("token") != config["file_token"]:
            raise ValueError(f"C-MHAD filename/application mismatch: {path}")
        files.append(
            (path, application, int(match.group("subject")), int(match.group("run")))
        )
    return sorted(files, key=lambda item: (item[1], item[2], item[3]))


def _read_stream(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    frame = pd.read_csv(
        path,
        skiprows=[1, 2],
        dtype={column: np.float64 for column in _CSV_COLUMNS},
    )
    if tuple(frame.columns) != _CSV_COLUMNS:
        raise ValueError(
            f"unexpected C-MHAD inertial columns in {path}: {tuple(frame.columns)}"
        )
    numeric = frame.to_numpy(dtype=np.float64, copy=False)
    if not len(numeric):
        raise ValueError(f"empty C-MHAD inertial stream: {path}")

    raw_timestamps_sec = numeric[:, 0] / 1000.0
    if not np.isfinite(raw_timestamps_sec).all():
        raise ValueError(f"non-finite C-MHAD timestamp in {path}")

    # Reproduce the release loader's video-to-IMU index alignment, but retain
    # native timestamps and never insert the zero rows recommended by the paper.
    source_frame_offset_samples = _OFFICIAL_FRAME_OFFSET_BASE - len(numeric)
    source_frame_offset_sec = source_frame_offset_samples / _SOURCE_RATE_HZ
    video_clock_origin_sec = float(raw_timestamps_sec[0] - source_frame_offset_sec)

    values = np.empty((len(numeric), 6), dtype=np.float32)
    np.divide(
        numeric[:, 1:4],
        _STANDARD_GRAVITY_M_S2,
        out=values[:, :3],
        casting="unsafe",
    )
    np.multiply(
        numeric[:, 4:7],
        np.pi / 180.0,
        out=values[:, 3:],
        casting="unsafe",
    )
    valid = np.isfinite(values)
    values[~valid] = 0.0

    positive_steps = np.diff(raw_timestamps_sec)
    positive_steps = positive_steps[positive_steps > 0]
    measured_rate_hz = (
        float(1.0 / np.median(positive_steps)) if len(positive_steps) else np.nan
    )
    metadata: dict[str, float | int] = {
        "raw_timestamp_origin_sec": float(raw_timestamps_sec[0]),
        "video_clock_origin_sec": video_clock_origin_sec,
        "source_frame_offset_samples": source_frame_offset_samples,
        "source_frame_offset_sec": source_frame_offset_sec,
        "initial_missing_samples": max(0, source_frame_offset_samples),
        "extra_prefix_samples": max(0, -source_frame_offset_samples),
        "measured_rate_hz": measured_rate_hz,
        "source_sample_count": len(numeric),
    }
    return raw_timestamps_sec, values, valid, metadata


def _align_annotations(
    annotations: tuple[EventInterval, ...],
    *,
    video_clock_origin_sec: float,
) -> tuple[EventInterval, ...]:
    return tuple(
        EventInterval(
            start_sec=video_clock_origin_sec + event.start_sec,
            end_sec=video_clock_origin_sec + event.end_sec,
            label=event.label,
            annotation_kind=event.annotation_kind,
            metadata={
                **event.metadata,
                "source_start_elapsed_sec": event.start_sec,
                "source_end_elapsed_sec": event.end_sec,
            },
        )
        for event in annotations
    )


def _events_in_observed_span(
    annotations: tuple[EventInterval, ...],
    *,
    observed_start: float,
    observed_end: float,
) -> tuple[tuple[EventInterval, ...], int, int]:
    """Clip observable annotations and exclude intervals with no sensor evidence."""

    events: list[EventInterval] = []
    excluded = 0
    clipped = 0
    for event in annotations:
        start_sec = max(event.start_sec, observed_start)
        end_sec = min(event.end_sec, observed_end)
        if end_sec <= start_sec:
            excluded += 1
            continue
        was_clipped = start_sec != event.start_sec or end_sec != event.end_sec
        clipped += int(was_clipped)
        events.append(
            EventInterval(
                start_sec=start_sec,
                end_sec=end_sec,
                label=event.label,
                annotation_kind=event.annotation_kind,
                metadata={
                    **event.metadata,
                    "aligned_source_start_sec": event.start_sec,
                    "aligned_source_end_sec": event.end_sec,
                    "clipped_to_observed_sensor_span": was_clipped,
                },
            )
        )
    return tuple(events), excluded, clipped


def iter_recordings(
    root: Path | None = None,
    limit: int | None = None,
) -> Iterator[RawRecording]:
    """Yield quality-contiguous C-MHAD recordings in canonical physical units.

    ``root`` may be the dataset directory, its ``raw`` directory, or one of the
    two application directories. Labels are exposed only through ``events`` and
    are intended exclusively for sealed evaluation scoring.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    source_root = Path(root) if root is not None else _DEFAULT_ROOT
    if (source_root / "raw").is_dir():
        source_root = source_root / "raw"
    if not source_root.is_dir():
        raise FileNotFoundError(f"C-MHAD source directory not found: {source_root}")

    files = _source_files(source_root)
    if not files:
        raise FileNotFoundError(
            f"no C-MHAD inertial CSV files found under {source_root}"
        )

    annotation_cache: dict[tuple[str, int], dict[int, tuple[EventInterval, ...]]] = {}
    yielded = 0
    for path, application, subject, run in files:
        if limit is not None and yielded >= limit:
            return
        config = _APPLICATIONS[application]
        cache_key = (application, subject)
        if cache_key not in annotation_cache:
            workbook = path.parent / f"{config['workbook_prefix']}{subject}.xlsx"
            if not workbook.is_file():
                raise FileNotFoundError(
                    f"C-MHAD annotation workbook not found: {workbook}"
                )
            annotation_cache[cache_key] = _read_annotations(workbook, config["labels"])
        timestamps, values, valid, source_metadata = _read_stream(path)
        annotations = _align_annotations(
            annotation_cache[cache_key].get(run, ()),
            video_clock_origin_sec=float(source_metadata["video_clock_origin_sec"]),
        )
        slices = split_at_clock_gaps(timestamps, max_gap_sec=_MAX_CLOCK_GAP_SEC)
        base_id = f"{config['slug']}_subject_{subject:02d}_run_{run:02d}"
        for chunk_index, chunk in enumerate(slices):
            if limit is not None and yielded >= limit:
                return
            chunk_timestamps = timestamps[chunk]
            observed_start = float(chunk_timestamps[0])
            observed_end = float(chunk_timestamps[-1])
            overlapping_events, excluded_events, clipped_events = (
                _events_in_observed_span(
                    annotations,
                    observed_start=observed_start,
                    observed_end=observed_end,
                )
            )
            chunk_id = (
                base_id if len(slices) == 1 else f"{base_id}_chunk_{chunk_index:02d}"
            )
            stream = SensorStream(
                stream_id="imu",
                placement=str(config["placement"]),
                device="Shimmer3",
                timestamps_sec=chunk_timestamps,
                values=values[chunk],
                channels=CANONICAL_CHANNELS,
                valid=valid[chunk],
                gravity_state="present",
                nominal_rate_hz=_SOURCE_RATE_HZ,
                metadata={
                    "measured_rate_hz": source_metadata["measured_rate_hz"],
                    "source_units": (
                        "m/s^2",
                        "m/s^2",
                        "m/s^2",
                        "deg/s",
                        "deg/s",
                        "deg/s",
                    ),
                },
            )
            yield RawRecording(
                dataset="c_mhad",
                recording_id=chunk_id,
                subject_id=f"subject_{subject:02d}",
                session_id=base_id,
                streams=(stream,),
                events=overlapping_events,
                split=None,
                metadata={
                    "application": config["slug"],
                    "source_run": run,
                    "source_path": path.relative_to(source_root).as_posix(),
                    "annotation_usage": "scoring_only",
                    "application_role": "sealed_external_evaluation",
                    "annotation_clock": "native_sensor_seconds",
                    "source_annotation_clock": "elapsed_video_seconds",
                    "alignment_convention_uncertainty_samples": 2,
                    "alignment_convention_uncertainty_sec": 2 / _SOURCE_RATE_HZ,
                    "observed_start_sec": observed_start,
                    "observed_end_sec": observed_end,
                    "annotations_without_sensor_overlap": excluded_events,
                    "annotations_clipped_to_sensor_span": clipped_events,
                    "hard_gap_threshold_sec": _MAX_CLOCK_GAP_SEC,
                    **source_metadata,
                },
            )
            yielded += 1
