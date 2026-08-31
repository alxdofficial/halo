"""Small, deterministic helpers for constructing application-task examples.

These helpers preserve source identities and clocks. They are intended for smoke
tests and manifest construction, not for defining train/test splits implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from applications.motion_monitoring.data.cache import CachedRecordingDataset
from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)


DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parent / "sources"


@dataclass(frozen=True)
class EventExample:
    recording: RawRecording
    event: EventInterval
    execution_id: str


def open_cache(
    dataset: str,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    validate_provenance: bool = True,
) -> CachedRecordingDataset:
    return CachedRecordingDataset(
        Path(source_root) / dataset / "processed" / "canonical_v1",
        validate_provenance=validate_provenance,
    )


def iter_events(
    cache: CachedRecordingDataset,
    *,
    annotation_kind: str | None = None,
    include_labels: set[str] | None = None,
    exclude_labels: set[str] | None = None,
    max_recordings: int | None = None,
) -> Iterator[EventExample]:
    """Yield source events without changing their identity or temporal extent."""

    limit = len(cache) if max_recordings is None else min(len(cache), max_recordings)
    for recording_index in range(limit):
        recording = cache[recording_index]
        for event_index, event in enumerate(recording.events):
            if annotation_kind is not None and event.annotation_kind != annotation_kind:
                continue
            if include_labels is not None and event.label not in include_labels:
                continue
            if exclude_labels is not None and event.label in exclude_labels:
                continue
            yield EventExample(
                recording=recording,
                event=event,
                execution_id=(
                    f"{recording.dataset}/{recording.recording_id}/"
                    f"{event.annotation_kind}/{event_index}"
                ),
            )


def find_event_pair(
    cache: CachedRecordingDataset,
    *,
    annotation_kind: str,
    same_label: bool,
    different_recordings: bool = False,
    max_recordings: int | None = None,
) -> tuple[EventExample, EventExample]:
    """Find two independent events for a deterministic real-data smoke test."""

    first_by_label: dict[str, EventExample] = {}
    first: EventExample | None = None
    for example in iter_events(
        cache, annotation_kind=annotation_kind, max_recordings=max_recordings
    ):
        if first is None:
            first = example
        if same_label:
            prior = first_by_label.get(example.event.label)
            if (
                prior is not None
                and prior.execution_id != example.execution_id
                and (
                    not different_recordings
                    or prior.recording.recording_id != example.recording.recording_id
                )
            ):
                return prior, example
            if example.event.label not in first_by_label or different_recordings:
                first_by_label[example.event.label] = example
        elif (
            first is not None
            and example.event.label != first.event.label
            and (
                not different_recordings
                or first.recording.recording_id != example.recording.recording_id
            )
        ):
            return first, example
    relation = "same" if same_label else "different"
    raise LookupError(
        f"cache has no pair of independent {annotation_kind!r} events with {relation} labels"
    )


def crop_stream(stream: SensorStream, start_sec: float, end_sec: float) -> SensorStream:
    if not np.isfinite(start_sec) or not np.isfinite(end_sec) or end_sec <= start_sec:
        raise ValueError("crop boundaries must be finite and increasing")
    timestamps = np.asarray(stream.timestamps_sec, dtype=np.float64)
    step = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 1.0
    tolerance = max(abs(step) * 1e-6, np.finfo(np.float64).eps * max(abs(start_sec), abs(end_sec), 1.0))
    left = int(np.searchsorted(timestamps, start_sec - tolerance, side="left"))
    right = int(np.searchsorted(timestamps, end_sec - tolerance, side="left"))
    if right <= left:
        raise ValueError(
            f"crop [{start_sec}, {end_sec}] has no samples in stream {stream.stream_id!r}"
        )
    return SensorStream(
        stream_id=stream.stream_id,
        placement=stream.placement,
        device=stream.device,
        timestamps_sec=stream.timestamps_sec[left:right],
        values=stream.values[left:right],
        channels=stream.channels,
        valid=stream.valid[left:right],
        gravity_state=stream.gravity_state,
        nominal_rate_hz=stream.nominal_rate_hz,
        metadata=stream.metadata,
    )


def crop_recording(
    recording: RawRecording,
    start_sec: float,
    end_sec: float,
    *,
    recording_suffix: str = "crop",
    retain_overlapping_events: bool = True,
) -> RawRecording:
    """Crop every overlapping stream while keeping the original physical clock."""

    streams: list[SensorStream] = []
    for stream in recording.streams:
        stream_start = float(stream.timestamps_sec[0])
        stream_end = float(stream.timestamps_sec[-1])
        overlap_start = max(start_sec, stream_start)
        overlap_end = min(end_sec, stream_end + 1.0 / (stream.nominal_rate_hz or 1.0))
        if overlap_end > overlap_start:
            streams.append(crop_stream(stream, overlap_start, overlap_end))
    if not streams:
        raise ValueError("recording crop does not overlap any sensor stream")
    events: list[EventInterval] = []
    if retain_overlapping_events:
        for event in recording.events:
            event_start = max(start_sec, event.start_sec)
            event_end = min(end_sec, event.end_sec)
            if event_end > event_start:
                clipped = event_start > event.start_sec or event_end < event.end_sec
                events.append(
                    EventInterval(
                        start_sec=event_start,
                        end_sec=event_end,
                        label=event.label,
                        annotation_kind=event.annotation_kind,
                        metadata={
                            **event.metadata,
                            "source_event_start_sec": event.metadata.get(
                                "source_event_start_sec", event.start_sec
                            ),
                            "source_event_end_sec": event.metadata.get(
                                "source_event_end_sec", event.end_sec
                            ),
                            "clipped_by_recording_crop": bool(
                                event.metadata.get("clipped_by_recording_crop", False)
                                or clipped
                            ),
                        },
                    )
                )
    return RawRecording(
        dataset=recording.dataset,
        recording_id=f"{recording.recording_id}::{recording_suffix}",
        subject_id=recording.subject_id,
        session_id=recording.session_id,
        streams=tuple(streams),
        events=tuple(events),
        split=recording.split,
        metadata={
            **recording.metadata,
            "source_recording_id": recording.metadata.get(
                "source_recording_id", recording.recording_id
            ),
        },
    )


def crop_event(
    example: EventExample,
    *,
    margin_sec: float = 0.0,
) -> RawRecording:
    if margin_sec < 0:
        raise ValueError("event margin must be non-negative")
    stream_start = min(float(stream.timestamps_sec[0]) for stream in example.recording.streams)
    stream_end = max(float(stream.timestamps_sec[-1]) for stream in example.recording.streams)
    return crop_recording(
        example.recording,
        max(stream_start, example.event.start_sec - margin_sec),
        min(stream_end, example.event.end_sec + margin_sec),
        recording_suffix=f"event-{example.execution_id.rsplit('/', 1)[-1]}",
    )


def crop_query_around_event(
    example: EventExample,
    *,
    duration_sec: float = 120.0,
) -> RawRecording:
    """Take a fixed-duration natural query crop centered on a source event."""

    if duration_sec <= 0:
        raise ValueError("query duration must be positive")
    stream_start = max(float(stream.timestamps_sec[0]) for stream in example.recording.streams)
    stream_end = min(float(stream.timestamps_sec[-1]) for stream in example.recording.streams)
    available = stream_end - stream_start
    if available <= 0:
        raise ValueError("recording streams have no shared physical-time support")
    duration = min(duration_sec, available)
    center = 0.5 * (example.event.start_sec + example.event.end_sec)
    start = min(max(center - 0.5 * duration, stream_start), stream_end - duration)
    return crop_recording(
        example.recording,
        start,
        start + duration,
        recording_suffix="query",
    )
