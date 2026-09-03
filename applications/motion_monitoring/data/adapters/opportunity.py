"""Opportunity mid-level gesture timelines for Task-3 recurrence discovery.

Four participants ran a morning routine in a sensor-rich kitchen: five ADL runs
plus one scripted Drill run each. This adapter reads the release's **mid-level
gesture track** (`ML_Both_Arms`, column 250), which the Phase-A converter
deliberately skipped because its 2.8 s instances cannot survive a 6 s window
majority vote. Task 3 works on native event boundaries rather than windows, so
the objection does not apply here and the track is exactly what the task needs.

Why this source (measured 2026-09-03, doc section 10):

* 17 gesture identities, a median of 3 occurrences of each per run, and 61 % of
  identities recurring three or more times inside one recording -- a recurrence
  task needs recordings in which motions actually recur, and the sources it was
  previously evaluated on show a median of one occurrence per identity.
* instance durations of 1.7 to 5.7 s, 99 % of which are reachable by the frozen
  candidate grid, against 21 % for the AIDLAB markers it replaces.
* real background between gestures and real execution variability, which an
  assembled corpus cannot supply.

Only the five body-worn Xsens inertial units are converted: the loose
accelerometers, shoe, object and ambient sensors are out of scope. Column offsets
come from the release's own `column_names.txt` and match the Phase-A converter.
Acceleration is released in milli-g and the gyroscope in milli-rad/s, both scaled
by 1/1000 here; the units are asserted rather than assumed.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import re

import numpy as np
import pandas as pd

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
    split_at_clock_gaps,
)


_DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "sources"
    / "opportunity"
    / "downloads"
    / "OpportunityUCIDataset"
    / "dataset"
)
_RATE_HZ = 30.0
_MILLI = 1.0 / 1000.0
_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
# 0-based column of each unit's accelerometer x; gyroscope x is +3.
_SENSORS = {
    "back": 37,
    "right_upper_arm": 50,
    "right_lower_arm": 63,
    "left_upper_arm": 76,
    "left_lower_arm": 89,
}
_GESTURE_COLUMN = 249
# label_legend.txt, ML_Both_Arms track.
_GESTURES = {
    406516: "open_door_1", 406517: "open_door_2", 404516: "close_door_1",
    404517: "close_door_2", 406520: "open_fridge", 404520: "close_fridge",
    406505: "open_dishwasher", 404505: "close_dishwasher",
    406519: "open_drawer_1", 404519: "close_drawer_1",
    406511: "open_drawer_2", 404511: "close_drawer_2",
    406508: "open_drawer_3", 404508: "close_drawer_3",
    408512: "clean_table", 407521: "drink_from_cup", 405506: "toggle_switch",
}
_ACC_MILLI_G_RANGE = (700.0, 1400.0)
_MAX_GAP_SEC = 0.2
_MIN_RUN_SEC = 30.0
_RUN_RE = re.compile(r"^S(?P<subject>\d)-(?P<run>ADL\d|Drill)\.dat$", re.IGNORECASE)


def iter_recordings(
    root: Path | None = None,
    limit: int | None = None,
) -> Iterator[RawRecording]:
    """Yield one continuous run per recording, with its bounded gesture instances."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    source = _DEFAULT_ROOT if root is None else Path(root)
    if not source.is_dir():
        raise FileNotFoundError(f"Opportunity release not found at {source}")
    columns = [_GESTURE_COLUMN]
    for base in _SENSORS.values():
        columns.extend(range(base, base + 6))
    order = sorted(set(columns))
    position = {column: slot for slot, column in enumerate(order)}

    yielded = 0
    for path in sorted(source.glob("S*-*.dat")):
        match = _RUN_RE.match(path.name)
        if match is None:
            continue
        if limit is not None and yielded >= limit:
            return
        subject_id = f"S{int(match.group('subject'))}"
        run = match.group("run").upper()
        table = pd.read_csv(
            path, sep=r"\s+", header=None, usecols=order, dtype=np.float64
        ).to_numpy()
        codes = np.nan_to_num(table[:, position[_GESTURE_COLUMN]], nan=0.0).astype(int)
        unknown = set(np.unique(codes)) - set(_GESTURES) - {0}
        if unknown:
            raise ValueError(f"{path}: unknown ML_Both_Arms codes {sorted(unknown)}")
        timestamps = np.arange(len(table), dtype=np.float64) / _RATE_HZ

        streams = []
        for sensor, base in _SENSORS.items():
            block = table[:, [position[base + offset] for offset in range(6)]]
            valid = np.isfinite(block)
            magnitude = float(
                np.median(np.linalg.norm(np.nan_to_num(block[:, :3]), axis=1))
            )
            if not _ACC_MILLI_G_RANGE[0] <= magnitude <= _ACC_MILLI_G_RANGE[1]:
                raise ValueError(
                    f"{path}: {sensor} median |acc| {magnitude:.0f} is outside the "
                    f"documented milli-g range {_ACC_MILLI_G_RANGE}"
                )
            streams.append(
                SensorStream(
                    stream_id=sensor,
                    placement=sensor,
                    device="Xsens inertial measurement unit",
                    timestamps_sec=timestamps,
                    values=(np.nan_to_num(block) * _MILLI).astype(np.float32),
                    channels=_CHANNELS,
                    valid=valid,
                    gravity_state="present",
                    nominal_rate_hz=_RATE_HZ,
                    metadata={
                        "source_acceleration_unit": "milli-g",
                        "source_gyroscope_unit": "milli-rad/s",
                        "output_units": "g and rad/s",
                    },
                )
            )

        changes = np.flatnonzero(np.diff(codes)) + 1
        bounds = np.concatenate(([0], changes, [len(codes)]))
        period = 1.0 / _RATE_HZ
        events = []
        counts: dict[str, int] = {}
        for start, stop in zip(bounds[:-1], bounds[1:]):
            code = int(codes[start])
            if code == 0:
                continue
            label = _GESTURES[code]
            index = counts.get(label, 0)
            counts[label] = index + 1
            events.append(
                EventInterval(
                    start_sec=float(timestamps[start]),
                    end_sec=float(timestamps[stop - 1] + period),
                    label=label,
                    annotation_kind="gesture",
                    metadata={
                        "source": "ML_Both_Arms",
                        "released_code": code,
                        "occurrence_index": index,
                        "run": run,
                        # The Drill run is 20 scripted repetitions of each gesture;
                        # ADL runs are natural morning-routine sequences.
                        "scripted": run == "DRILL",
                    },
                )
            )
        span = float(timestamps[-1] - timestamps[0])
        if span < _MIN_RUN_SEC or not events:
            continue
        yield RawRecording(
            dataset="opportunity",
            recording_id=f"opportunity:{subject_id}:{run}",
            subject_id=subject_id,
            session_id=f"opportunity:{subject_id}:{run}",
            streams=tuple(streams),
            events=tuple(events),
            split="train",
            metadata={
                "run": run,
                "scripted_run": run == "DRILL",
                "bounded_execution_annotations": True,
                # The gesture track annotates only the gestures, so the complement
                # is unlabelled rather than known-empty.
                "exhaustive_annotation": False,
                "gesture_identities": len(counts),
            },
        )
        yielded += 1
