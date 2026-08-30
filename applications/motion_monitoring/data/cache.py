"""Canonical on-disk cache for efficient application-task data loading.

The cache is deliberately lossless with respect to the raw adapter contract: it
does not resample, window, impute, or change annotations. Each recording remains
independently addressable so PyTorch workers can load disjoint examples without
re-decoding an entire source release.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)


CACHE_SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def _storage_name(recording_id: str) -> str:
    readable = "".join(
        character if character.isalnum() else "_" for character in recording_id
    )
    readable = readable.strip("_")[:64] or "recording"
    digest = sha1(recording_id.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"{readable}__{digest}"


def write_recording(recording: RawRecording, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    stream_rows: list[dict[str, Any]] = []
    for stream_index, stream in enumerate(recording.streams):
        prefix = f"stream_{stream_index:02d}"
        np.save(
            directory / f"{prefix}_timestamps.npy",
            stream.timestamps_sec,
            allow_pickle=False,
        )
        np.save(directory / f"{prefix}_values.npy", stream.values, allow_pickle=False)
        np.save(directory / f"{prefix}_valid.npy", stream.valid, allow_pickle=False)
        stream_rows.append(
            {
                "stream_id": stream.stream_id,
                "placement": stream.placement,
                "device": stream.device,
                "channels": list(stream.channels),
                "gravity_state": stream.gravity_state,
                "nominal_rate_hz": stream.nominal_rate_hz,
                "metadata": _jsonable(stream.metadata),
                "timestamps": f"{prefix}_timestamps.npy",
                "values": f"{prefix}_values.npy",
                "valid": f"{prefix}_valid.npy",
            }
        )

    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": recording.dataset,
        "recording_id": recording.recording_id,
        "subject_id": recording.subject_id,
        "session_id": recording.session_id,
        "split": recording.split,
        "metadata": _jsonable(recording.metadata),
        "streams": stream_rows,
        "events": [
            {
                "start_sec": event.start_sec,
                "end_sec": event.end_sec,
                "label": event.label,
                "annotation_kind": event.annotation_kind,
                "metadata": _jsonable(event.metadata),
            }
            for event in recording.events
        ],
    }
    (directory / "recording.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_recording(directory: Path, *, mmap: bool = True) -> RawRecording:
    payload = json.loads((directory / "recording.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported recording cache schema in {directory}")
    mmap_mode = "r" if mmap else None
    streams = tuple(
        SensorStream(
            stream_id=row["stream_id"],
            placement=row["placement"],
            device=row["device"],
            timestamps_sec=np.load(directory / row["timestamps"], mmap_mode=mmap_mode),
            values=np.load(directory / row["values"], mmap_mode=mmap_mode),
            channels=tuple(row["channels"]),
            valid=np.load(directory / row["valid"], mmap_mode=mmap_mode),
            gravity_state=row["gravity_state"],
            nominal_rate_hz=row["nominal_rate_hz"],
            metadata=row["metadata"],
        )
        for row in payload["streams"]
    )
    events = tuple(
        EventInterval(
            start_sec=row["start_sec"],
            end_sec=row["end_sec"],
            label=row["label"],
            annotation_kind=row["annotation_kind"],
            metadata=row["metadata"],
        )
        for row in payload["events"]
    )
    return RawRecording(
        dataset=payload["dataset"],
        recording_id=payload["recording_id"],
        subject_id=payload["subject_id"],
        session_id=payload["session_id"],
        streams=streams,
        events=events,
        split=payload["split"],
        metadata=payload["metadata"],
    )


class CachedRecordingDataset(Sequence[RawRecording]):
    """Map-style, worker-safe view over one completed canonical cache."""

    def __init__(self, root: Path, *, mmap: bool = True) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"recording cache manifest not found: {manifest_path}"
            )
        self.rows = tuple(
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        self.mmap = mmap

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int | slice) -> RawRecording | list[RawRecording]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        row = self.rows[index]
        return read_recording(self.root / row["directory"], mmap=self.mmap)

    def __iter__(self) -> Iterator[RawRecording]:
        for index in range(len(self)):
            yield self[index]
