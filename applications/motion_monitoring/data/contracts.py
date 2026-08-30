"""Shared raw-timeline contract for movement-monitoring dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


CANONICAL_CHANNELS = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)


@dataclass(frozen=True)
class EventInterval:
    """One source-provided interval on a recording's physical clock."""

    start_sec: float
    end_sec: float
    label: str
    annotation_kind: str = "event"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.start_sec) or not np.isfinite(self.end_sec):
            raise ValueError("event boundaries must be finite")
        if self.start_sec < 0 or self.end_sec <= self.start_sec:
            raise ValueError(
                f"invalid event interval [{self.start_sec}, {self.end_sec}]"
            )
        if not self.label:
            raise ValueError("event label must be non-empty")


@dataclass(frozen=True)
class SensorStream:
    """One native-rate sensor stream in canonical physical units.

    Acceleration is expressed in g and gyroscope values in rad/s. Missing values
    remain explicit through ``valid``; adapters must not synthesize measurements.
    """

    stream_id: str
    placement: str
    device: str
    timestamps_sec: np.ndarray
    values: np.ndarray
    channels: Sequence[str]
    valid: np.ndarray
    gravity_state: str
    nominal_rate_hz: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_sec)
        values = np.asarray(self.values)
        valid = np.asarray(self.valid)
        channels = tuple(self.channels)

        if timestamps.ndim != 1 or values.ndim != 2:
            raise ValueError("timestamps must be [T] and values must be [T, C]")
        if values.shape != valid.shape:
            raise ValueError("values and valid masks must have identical shapes")
        if len(timestamps) != len(values) or values.shape[1] != len(channels):
            raise ValueError("timestamps, values, and channel names disagree")
        if not len(timestamps):
            raise ValueError("sensor stream must contain at least one sample")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamps must be finite and strictly increasing")
        if valid.dtype != np.bool_:
            raise ValueError("valid mask must be boolean")
        if not np.isfinite(values[valid]).all():
            raise ValueError("valid sensor values must be finite")
        if len(set(channels)) != len(channels):
            raise ValueError("channel names must be unique within a stream")
        if any(channel not in CANONICAL_CHANNELS for channel in channels):
            raise ValueError(f"unsupported canonical channels: {channels}")
        if self.gravity_state not in {"present", "removed", "unknown"}:
            raise ValueError(f"invalid gravity state: {self.gravity_state}")
        if self.nominal_rate_hz is not None and self.nominal_rate_hz <= 0:
            raise ValueError("nominal rate must be positive")

        object.__setattr__(
            self, "timestamps_sec", timestamps.astype(np.float64, copy=False)
        )
        object.__setattr__(self, "values", values.astype(np.float32, copy=False))
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "channels", channels)


@dataclass(frozen=True)
class RawRecording:
    """A quality-contiguous source recording with one or more sensor streams."""

    dataset: str
    recording_id: str
    subject_id: str
    session_id: str
    streams: Sequence[SensorStream]
    events: Sequence[EventInterval] = ()
    split: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        streams = tuple(self.streams)
        events = tuple(self.events)
        if (
            not self.dataset
            or not self.recording_id
            or not self.subject_id
            or not self.session_id
        ):
            raise ValueError("dataset and recording identities must be non-empty")
        if not streams:
            raise ValueError("recording must contain at least one sensor stream")
        if len({stream.stream_id for stream in streams}) != len(streams):
            raise ValueError("stream ids must be unique within a recording")
        object.__setattr__(self, "streams", streams)
        object.__setattr__(self, "events", events)


def split_at_clock_gaps(
    timestamps_sec: np.ndarray,
    *,
    max_gap_sec: float,
) -> tuple[slice, ...]:
    """Return quality-contiguous slices without interpolating across clock gaps."""

    timestamps = np.asarray(timestamps_sec, dtype=np.float64)
    if timestamps.ndim != 1 or not len(timestamps):
        raise ValueError("timestamps must be a non-empty vector")
    if max_gap_sec <= 0:
        raise ValueError("max_gap_sec must be positive")
    breaks = (
        np.flatnonzero((np.diff(timestamps) <= 0) | (np.diff(timestamps) > max_gap_sec))
        + 1
    )
    bounds = np.concatenate(([0], breaks, [len(timestamps)]))
    return tuple(
        slice(int(start), int(end)) for start, end in zip(bounds[:-1], bounds[1:])
    )
