"""HARMES dominant-wrist ADL timelines with bounded execution annotations.

HARMES records 20 participants performing 15 fine-grained kitchen, cleaning and
hygiene activities of daily living while wearing a WearOS smartwatch on the
dominant wrist. Each participant contributes two to four recording sessions, 16
of the 20 on at least two distinct days, and repeats most activities several
times per session -- so it supplies same-person, same-task, cross-day bounded
executions on a consumer watch. That makes it the primary source of *accepted*
Task-2 positives (``docs/tasks/TASK2_CHANGE_QUANTIFICATION.md`` section 7.1).

Contract notes, all measured rather than assumed:

* the WearOS clock is written out of order and drops samples, so timestamps are
  sorted and de-duplicated. Measured over the release, dropouts are frequent
  (6,902 gaps over 50 ms in 24 sessions) but bounded: the largest is 997 ms, a
  buffer-flush ceiling rather than an acquisition failure. The timeline is
  therefore split only at a gap over 1 s, which keeps 845 of 848 sampled
  executions contiguous where a 250 ms rule kept 68. Nothing is interpolated
  across a hole; every retained hole is visible in the native timestamps and its
  size is recorded per execution as ``internal_gap_sec_max``.
* the start/end event log carries its own clock. For 4 of the 20 participants it
  is exactly one hour (DST/timezone) from the wrist clock, so the event-to-wrist
  offset is snapped to the nearest whole hour; anything else is a genuine
  mismatch and is left alone. This mirrors ``data/datasets/harmes/convert.py``.
* an activity whose interval spans a clock gap is dropped and counted, never
  clipped into a shorter execution.
* the left-wrist Puck.js stream is excluded upstream: its gyroscope saturates at
  the int16 rail on most hand motion and is unrecoverable.

Acceleration is released in m/s^2 with gravity present and is converted to g;
the gyroscope is already rad/s.
"""

from __future__ import annotations

from collections.abc import Iterator
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
    split_at_clock_gaps,
)


_DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1] / "sources" / "harmes" / "downloads" / "HARMES-RAW"
)
_RATE_HZ = 50.0
_GRAVITY_MS2 = 9.80665
_MAX_GAP_SEC = 1.0
_MIN_RUN_SEC = 6.0
_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
# Guard: a still wrist reads ~9.81 in m/s^2; g would land near 1.0.
_ACC_MS2_RANGE = (8.0, 11.5)

_LABELS = {
    "Brushing teeth": "brushing_teeth",
    "Cleaning out the dishwasher": "emptying_dishwasher",
    "Cleaning table": "cleaning_table",
    "Cream hands": "applying_hand_cream",
    "Cutting vegetables": "cutting_vegetables",
    "Disinfecting hands": "disinfecting_hands",
    "Drinking": "drinking",
    "Floor cleaning": "floor_cleaning",
    "Making tea": "making_tea",
    "Putting away the dishes": "putting_away_dishes",
    "Vacuum Cleaning": "vacuum_cleaning",
    "Washing dishes": "washing_dishes",
    "Washing hands": "washing_hands",
    "Watering plant": "watering_plants",
    "Window cleaning": "window_cleaning",
}


def _read_signal(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, usecols=["timestamp", *_CHANNELS], dtype=np.float64)
    if not len(frame):
        raise ValueError(f"{path}: no samples")
    timestamps = frame["timestamp"].to_numpy(dtype=np.float64) / 1000.0
    values = frame.loc[:, list(_CHANNELS)].to_numpy(dtype=np.float64)
    order = np.argsort(timestamps, kind="stable")
    timestamps, values = timestamps[order], values[order]
    keep = np.concatenate(([True], np.diff(timestamps) > 0))
    return timestamps[keep], values[keep]


def _read_events(path: Path, wrist_start_sec: float) -> list[tuple[str, float, float]]:
    """Parse the start/end log onto the wrist clock, snapping a whole-hour offset."""

    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            (row["Description"].strip(), row["Type"].strip(), float(row["Time"]))
            for row in csv.DictReader(stream)
            if row.get("Time")
        ]
    if not rows:
        return []
    offset = round((wrist_start_sec - min(time for _, _, time in rows)) / 3600.0) * 3600.0
    open_events: dict[str, float] = {}
    events: list[tuple[str, float, float]] = []
    for description, kind, time in rows:
        if description == "RECORD" or description.lower().startswith("deleted"):
            continue
        time += offset
        if kind == "start":
            open_events[description] = time
        elif kind == "end" and description in open_events:
            start = open_events.pop(description)
            label = _LABELS.get(description)
            if label is not None and time > start:
                events.append((label, start, time))
    events.sort(key=lambda item: item[1])
    return events


def iter_recordings(
    root: Path | None = None,
    limit: int | None = None,
) -> Iterator[RawRecording]:
    """Yield quality-contiguous wrist timelines with bounded ADL executions."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    source = _DEFAULT_ROOT if root is None else Path(root)
    if not source.is_dir():
        raise FileNotFoundError(f"HARMES release not found at {source}")
    yielded = 0
    for participant in sorted(path for path in source.iterdir() if path.is_dir()):
        subject_id = f"pp{participant.name}"
        for session in sorted(path for path in participant.iterdir() if path.is_dir()):
            signal_paths = sorted(session.glob("recording_*.csv"))
            event_paths = sorted(
                path for path in session.glob("*.csv") if not path.name.startswith("recording_")
            )
            if len(signal_paths) != 1 or len(event_paths) != 1:
                raise ValueError(
                    f"{session}: expected one wrist recording and one event log, "
                    f"found {len(signal_paths)} and {len(event_paths)}"
                )
            timestamps, values = _read_signal(signal_paths[0])
            magnitude = float(np.median(np.linalg.norm(values[:, :3], axis=1)))
            if not _ACC_MS2_RANGE[0] <= magnitude <= _ACC_MS2_RANGE[1]:
                raise ValueError(
                    f"{signal_paths[0]}: median |acc| {magnitude:.3f} is outside the "
                    f"documented m/s^2 range {_ACC_MS2_RANGE}"
                )
            values = np.column_stack((values[:, :3] / _GRAVITY_MS2, values[:, 3:]))
            events = _read_events(event_paths[0], float(timestamps[0]))
            slices = split_at_clock_gaps(timestamps, max_gap_sec=_MAX_GAP_SEC)
            for part_index, quality_slice in enumerate(slices):
                if limit is not None and yielded >= limit:
                    return
                part_timestamps = timestamps[quality_slice]
                if len(part_timestamps) < 2:
                    continue
                span = float(part_timestamps[-1] - part_timestamps[0])
                if span < _MIN_RUN_SEC:
                    continue
                start, end = float(part_timestamps[0]), float(part_timestamps[-1])
                origin = start
                inside = [
                    item for item in events if item[1] >= start - 1e-6 and item[2] <= end + 1e-6
                ]
                crossing = sum(
                    1
                    for label, begin, finish in events
                    if (begin < start <= finish) or (begin <= end < finish)
                )
                recording_id = f"harmes:{subject_id}:{session.name}"
                if len(slices) > 1:
                    recording_id += f":part:{part_index}"
                deltas = np.diff(part_timestamps)
                counts: dict[str, int] = {}
                intervals: list[EventInterval] = []
                for label, begin, finish in inside:
                    index = counts.get(label, 0)
                    counts[label] = index + 1
                    left = int(np.searchsorted(part_timestamps, begin, side="left"))
                    right = int(np.searchsorted(part_timestamps, finish, side="right"))
                    span = deltas[left : max(right - 1, left)]
                    intervals.append(
                        EventInterval(
                            start_sec=begin - origin,
                            end_sec=finish - origin,
                            label=label,
                            annotation_kind="bounded_execution",
                            metadata={
                                "source": "start_end_event_log",
                                "run_index": index,
                                "session": session.name,
                                "internal_gap_sec_max": float(span.max()) if len(span) else 0.0,
                                "samples": right - left,
                            },
                        )
                    )
                yield RawRecording(
                    dataset="harmes",
                    recording_id=recording_id,
                    subject_id=subject_id,
                    session_id=f"harmes:{subject_id}:{session.name}",
                    streams=(
                        SensorStream(
                            stream_id="watch_wrist",
                            placement="dominant_wrist",
                            device="WearOS smartwatch",
                            timestamps_sec=part_timestamps - origin,
                            values=values[quality_slice].astype(np.float32),
                            channels=_CHANNELS,
                            valid=np.ones(
                                (len(part_timestamps), len(_CHANNELS)), dtype=bool
                            ),
                            gravity_state="present",
                            nominal_rate_hz=_RATE_HZ,
                            metadata={
                                "source_acceleration_unit": "m/s^2",
                                "output_acceleration_unit": "g",
                                "source_epoch_start_sec": start,
                                "left_wrist_puck_excluded": "unrecoverable gyroscope scale",
                            },
                        ),
                    ),
                    events=tuple(intervals),
                    metadata={
                        "quality_part_index": part_index,
                        "quality_part_count": len(slices),
                        "max_internal_gap_sec": float(deltas.max()) if len(deltas) else 0.0,
                        "events_spanning_clock_gaps_dropped": crossing,
                        "bounded_execution_annotations": True,
                        "source_session": session.name,
                    },
                )
                yielded += 1
