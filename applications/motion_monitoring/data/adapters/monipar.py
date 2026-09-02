"""Application adapter for weekly MoniPar smartwatch protocols."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)


_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ROOT = _REPO_ROOT / "data" / "datasets" / "monipar" / "sessions"
_RATE_HZ = 50.0
_GRAVITY_MS2 = 9.80665
_CHANNELS = ("acc_x", "acc_y", "acc_z")
TASK2_ACTIVE_LABELS = frozenset(
    {
        "arising_from_a_chair",
        "finger_tapping",
        "gait",
        "moving_hands_to_the_chest",
        "postural_hand_tremor",
        "pronation_supination",
        "rapid_hand_opening_and_closing",
    }
)
def _activity_events(
    labels: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[EventInterval, ...]:
    if len(labels) != len(timestamps) or not len(labels):
        raise ValueError("MoniPar labels and timestamps must be non-empty and aligned")
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], changes, [len(labels)]))
    sample_period = 1.0 / _RATE_HZ
    return tuple(
        EventInterval(
            start_sec=float(timestamps[start]),
            end_sec=float(timestamps[stop - 1] + sample_period),
            label=str(labels[start]),
            annotation_kind="protocol_state",
            metadata={
                "source": "sample_labels",
                "week_session": True,
                "independent_repetition": False,
            },
        )
        for start, stop in zip(bounds[:-1], bounds[1:])
    )


def iter_recordings(
    root: Path | None = None,
    limit: int | None = None,
) -> Iterator[RawRecording]:
    """Yield complete weekly protocols in canonical g units."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    source = _DEFAULT_ROOT if root is None else Path(root)
    if not source.is_dir():
        raise FileNotFoundError(f"MoniPar sessions not found at {source}")
    yielded = 0
    for path in sorted(source.glob("*/data.parquet")):
        if limit is not None and yielded >= limit:
            break
        frame = pd.read_parquet(
            path,
            columns=["timestamp_sec", *_CHANNELS, "activity", "subject"],
        )
        timestamps = frame["timestamp_sec"].to_numpy(dtype=np.float64, copy=True)
        values_ms2 = frame.loc[:, _CHANNELS].to_numpy(dtype=np.float32, copy=True)
        values = values_ms2 / _GRAVITY_MS2
        valid = np.isfinite(values)
        subjects = frame["subject"].astype(str).unique().tolist()
        if len(subjects) != 1:
            raise ValueError(f"{path} contains multiple MoniPar subjects: {subjects}")
        labels = frame["activity"].astype(str).to_numpy()
        session_id = path.parent.name
        subject_id = subjects[0]
        yield RawRecording(
            dataset="monipar",
            recording_id=f"monipar:{session_id}",
            subject_id=subject_id,
            session_id=session_id,
            streams=(
                SensorStream(
                    stream_id="watch_wrist",
                    placement="wrist",
                    device="TicWatch S2 smartwatch",
                    timestamps_sec=timestamps,
                    values=np.nan_to_num(values),
                    channels=_CHANNELS,
                    valid=valid,
                    gravity_state="present",
                    nominal_rate_hz=_RATE_HZ,
                    metadata={
                        "source_acceleration_unit": "m/s^2",
                        "output_acceleration_unit": "g",
                        "source_session_path": str(path.relative_to(_REPO_ROOT)),
                    },
                ),
            ),
            events=_activity_events(labels, timestamps),
            split="evaluation",
            metadata={
                "cohort": subject_id.rstrip("0123456789"),
                "visit_kind": "weekly_protocol",
                "bounded_event_annotations": True,
                "bounded_execution_annotations": False,
                "independent_repetition_annotations": False,
            },
        )
        yielded += 1
