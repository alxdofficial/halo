"""Pre-materialised Task-2 training variants (design doc section 6.3).

A physical modification changes the movement, so it has to be applied to the raw
signal and encoded afterwards. Baking a fixed set of variants into a derived
canonical cache keeps that reproducible and keeps training cheap: everything is
encoded once, exactly like the Task-1 synthetic corpus, and the frozen seeds make
the corpus a pure function of ``VARIANT_CONFIG`` plus its source caches.

For every bounded execution in the source caches this emits:

* ``modified`` variants -- a declared physical modification of known kind and
  severity. These are the ruler's negatives.
* one ``nuisance`` variant -- an acquisition transform that changes how the
  movement was recorded but not the movement. These ride on positives.

A nuisance is *also* applied to most modified variants, so "was transformed"
carries no information about the label. Clean executions are not copied here;
they are read from their own source caches, so nothing is stored twice.
"""

from __future__ import annotations

from collections.abc import Iterator
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from applications.motion_monitoring.data.cache import CachedRecordingDataset
from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)
from applications.motion_monitoring.task2.modifications import (
    MODIFICATIONS,
    NUISANCES,
    apply_modification,
    apply_nuisance,
)


DATASET = "task2_modified_v1"
_DATA_ROOT = Path(__file__).resolve().parents[1]

VARIANT_CONFIG: dict[str, Any] = {
    "version": 2,
    "seed": 20260902,
    # (source cache, annotation kind that is one bounded execution there)
    "sources": [["harmes", "bounded_execution"], ["crossfit", "repetition"]],
    "modified_variants_per_execution": 2,
    "nuisance_variants_per_execution": 1,
    "severity_range": [0.3, 1.0],
    "nuisance_on_modified_probability": 0.6,
    "minimum_samples": 16,
    # Channel subsets to materialise. ``null`` is the source's own channel set.
    # Adding ["acc_x","acc_y","acc_z"] builds the acceleration-only world MoniPar
    # is evaluated in; it must be generated here because a reduced signal has to
    # be ENCODED as such and cannot be projected out of six-channel embeddings.
    "channel_views": [None, ["acc_x", "acc_y", "acc_z"]],
}


def config_digest(config: dict[str, Any] = VARIANT_CONFIG) -> str:
    return sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _seed(*parts: object) -> int:
    text = ":".join(str(part) for part in parts)
    return int(sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _canonical_root(dataset: str, root: Path | None) -> Path:
    base = Path(root) if root is not None else _DATA_ROOT / "sources"
    return base / dataset / "processed" / "canonical_v1"


def _executions(recording: RawRecording, kind: str, *, minimum_samples: int):
    """Yield (event_index, event, stream, sample slice) for each bounded execution."""

    for stream in recording.streams:
        timestamps = np.asarray(stream.timestamps_sec)
        for index, event in enumerate(recording.events):
            if event.annotation_kind != kind:
                continue
            left = int(np.searchsorted(timestamps, event.start_sec, side="left"))
            right = int(np.searchsorted(timestamps, event.end_sec, side="left"))
            if right - left < minimum_samples:
                continue
            if not bool(np.asarray(stream.valid)[left:right].all()):
                # A modification applied across a dead-sensor region would be a
                # transform of missing data, not of a movement.
                continue
            yield index, event, stream, slice(left, right)


def _variant_recording(
    *,
    source: RawRecording,
    stream: SensorStream,
    window: slice,
    values: np.ndarray,
    event: EventInterval,
    event_index: int,
    variant: str,
    channels: tuple[str, ...],
    source_timestamps: np.ndarray,
    metadata: dict[str, Any],
) -> RawRecording:
    rate = float(stream.nominal_rate_hz or 1.0 / np.median(np.diff(stream.timestamps_sec)))
    if len(values) == len(source_timestamps):
        timestamps = np.asarray(source_timestamps, dtype=np.float64)
        timestamps = timestamps - timestamps[0]
    else:
        timestamps = np.arange(len(values), dtype=np.float64) / rate
    origin = f"{source.dataset}:{source.recording_id}:{event_index}"
    return RawRecording(
        dataset=DATASET,
        recording_id=f"{DATASET}:{origin}:{variant}",
        subject_id=source.subject_id,
        session_id=source.session_id,
        streams=(
            SensorStream(
                stream_id=stream.stream_id,
                placement=stream.placement,
                device=stream.device,
                timestamps_sec=timestamps,
                values=values.astype(np.float32),
                channels=channels,
                valid=np.ones((len(values), len(channels)), dtype=bool),
                gravity_state=stream.gravity_state,
                nominal_rate_hz=rate,
                metadata=dict(stream.metadata),
            ),
        ),
        events=(
            EventInterval(
                start_sec=0.0,
                end_sec=float(len(values) / rate),
                label=event.label,
                annotation_kind="bounded_execution",
                metadata=metadata,
            ),
        ),
        split="train",
        metadata={
            "origin_dataset": source.dataset,
            "origin_recording_id": source.recording_id,
            "origin_execution_id": origin,
            "origin_subject_id": source.subject_id,
            "variant": variant,
            **metadata,
        },
    )


def iter_recordings(
    root: Path | None = None,
    limit: int | None = None,
    config: dict[str, Any] = VARIANT_CONFIG,
) -> Iterator[RawRecording]:
    """Yield every declared variant of every source bounded execution."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    low, high = config["severity_range"]
    views = [None if view is None else tuple(view) for view in config.get("channel_views", [None])]
    kinds = sorted(MODIFICATIONS)
    nuisance_kinds = sorted(NUISANCES)
    yielded = 0
    for dataset, annotation_kind in config["sources"]:
        cache = CachedRecordingDataset(_canonical_root(dataset, root))
        for recording in cache:
            for event_index, event, stream, window in _executions(
                recording,
                annotation_kind,
                minimum_samples=int(config["minimum_samples"]),
            ):
                origin = f"{dataset}:{recording.recording_id}:{event_index}"
                for view in views:
                    channels = tuple(stream.channels) if view is None else tuple(view)
                    channel_indices = {name: i for i, name in enumerate(stream.channels)}
                    if any(name not in channel_indices for name in channels):
                        continue
                    indices = [channel_indices[name] for name in channels]
                    raw = np.asarray(stream.values[window], dtype=np.float64)[:, indices]
                    source_timestamps = np.asarray(stream.timestamps_sec[window], dtype=np.float64)
                    rate = float(
                        stream.nominal_rate_hz
                        or 1.0 / np.median(np.diff(stream.timestamps_sec))
                    )
                    view_name = "native" if view is None else "-".join(channels)
                    for index in range(int(config["modified_variants_per_execution"])):
                        if limit is not None and yielded >= limit:
                            return
                        seed = _seed(config["seed"], origin, view_name, "modified", index)
                        rng = np.random.default_rng(seed)
                        kind = kinds[int(rng.integers(len(kinds)))]
                        severity = float(low + (high - low) * rng.random())
                        values = apply_modification(
                            raw,
                            kind=kind,
                            severity=severity,
                            seed=seed % (2**32),
                            sampling_rate_hz=rate,
                            channels=channels,
                        )
                        nuisance_kind = None
                        if rng.random() < float(config["nuisance_on_modified_probability"]):
                            nuisance_kind = nuisance_kinds[int(rng.integers(len(nuisance_kinds)))]
                            values = apply_nuisance(
                                values,
                                kind=nuisance_kind,
                                seed=(seed + 1) % (2**32),
                                sampling_rate_hz=rate,
                                channels=channels,
                            )
                        yield _variant_recording(
                            source=recording,
                            stream=stream,
                            window=window,
                            values=values,
                            event=event,
                            event_index=event_index,
                            variant=f"{view_name}:modified_{index}",
                            channels=channels,
                            source_timestamps=source_timestamps,
                            metadata={
                                "variant": "modified",
                                "modification_kind": kind,
                                "severity": severity,
                                "nuisance_kind": nuisance_kind,
                                "seed": int(seed),
                                "origin_execution_id": origin,
                                "origin_dataset": dataset,
                                "channel_view": view_name,
                            },
                        )
                        yielded += 1
                    for index in range(int(config["nuisance_variants_per_execution"])):
                        if limit is not None and yielded >= limit:
                            return
                        seed = _seed(config["seed"], origin, view_name, "nuisance", index)
                        rng = np.random.default_rng(seed)
                        nuisance_kind = nuisance_kinds[int(rng.integers(len(nuisance_kinds)))]
                        values = apply_nuisance(
                            raw,
                            kind=nuisance_kind,
                            seed=seed % (2**32),
                            sampling_rate_hz=rate,
                            channels=channels,
                        )
                        yield _variant_recording(
                            source=recording,
                            stream=stream,
                            window=window,
                            values=values,
                            event=event,
                            event_index=event_index,
                            variant=f"{view_name}:nuisance_{index}",
                            channels=channels,
                            source_timestamps=source_timestamps,
                            metadata={
                                "variant": "nuisance",
                                "modification_kind": None,
                                "severity": 0.0,
                                "nuisance_kind": nuisance_kind,
                                "seed": int(seed),
                                "origin_execution_id": origin,
                                "origin_dataset": dataset,
                                "channel_view": view_name,
                            },
                        )
                        yielded += 1
