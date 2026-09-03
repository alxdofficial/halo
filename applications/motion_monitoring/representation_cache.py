"""Frozen, provenance-bound ``MotionSequence`` representation caches."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from hashlib import sha1, sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from applications.motion_monitoring.data.cache import CachedRecordingDataset
from applications.motion_monitoring.data.contracts import EventInterval, RawRecording, SensorStream
from applications.motion_monitoring.data.manifests import (
    CohortEntry,
    CohortManifest,
    validate_manifest_caches,
)
from applications.motion_monitoring.sequence import MotionEncoder, MotionSequence


REPRESENTATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, order=True)
class BoundedSegment:
    dataset: str
    cache_index: int
    recording_id: str
    stream_id: str
    event_index: int
    start_sec: float
    end_sec: float


def bounded_representation_id(recording_id: str, event_index: int) -> str:
    return f"{recording_id}::bounded_event_{event_index}"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def _storage_name(dataset: str, recording_id: str, stream_id: str) -> str:
    identity = f"{dataset}/{recording_id}/{stream_id}"
    readable = "_".join(
        part for part in (dataset, recording_id, stream_id) if part
    )
    readable = "".join(char if char.isalnum() else "_" for char in readable)
    digest = sha1(identity.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"{readable[:72].strip('_')}__{digest}"


def write_motion_sequence(sequence: MotionSequence, directory: Path) -> None:
    """Write one detached sequence without changing dtype or temporal geometry."""

    directory.mkdir(parents=True, exist_ok=False)
    arrays = {
        "embeddings": sequence.embeddings,
        "intervals_sec": sequence.intervals_sec,
        "valid": sequence.valid,
        "physical_features": sequence.physical_features,
        "physical_feature_mask": sequence.physical_feature_mask,
    }
    for name, value in arrays.items():
        tensor = torch.as_tensor(value)
        if tensor.requires_grad:
            raise ValueError("representation caches accept only frozen, detached tensors")
        np.save(directory / f"{name}.npy", tensor.detach().cpu().numpy(), allow_pickle=False)
    metadata = {
        "schema_version": REPRESENTATION_SCHEMA_VERSION,
        "dataset": sequence.dataset,
        "recording_id": sequence.recording_id,
        "subject_id": sequence.subject_id,
        "session_id": sequence.session_id,
        "stream_id": sequence.stream_id,
        "placement": sequence.placement,
        "device": sequence.device,
        "channels": list(sequence.channels),
        "gravity_state": sequence.gravity_state,
        "sampling_rate_hz": sequence.sampling_rate_hz,
        "physical_feature_names": list(sequence.physical_feature_names),
    }
    (directory / "sequence.json").write_text(
        json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_motion_sequence(directory: Path, *, mmap: bool = True) -> MotionSequence:
    metadata = json.loads((directory / "sequence.json").read_text(encoding="utf-8"))
    if metadata.pop("schema_version", None) != REPRESENTATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported representation schema in {directory}")
    # Copy-on-write mappings are writable from PyTorch's perspective but never alter
    # the frozen cache on disk.
    mmap_mode = "c" if mmap else None
    return MotionSequence(
        embeddings=torch.from_numpy(np.load(directory / "embeddings.npy", mmap_mode=mmap_mode)),
        intervals_sec=torch.from_numpy(
            np.load(directory / "intervals_sec.npy", mmap_mode=mmap_mode)
        ),
        valid=torch.from_numpy(np.load(directory / "valid.npy", mmap_mode=mmap_mode)),
        physical_features=torch.from_numpy(
            np.load(directory / "physical_features.npy", mmap_mode=mmap_mode)
        ),
        physical_feature_mask=torch.from_numpy(
            np.load(directory / "physical_feature_mask.npy", mmap_mode=mmap_mode)
        ),
        physical_feature_names=tuple(metadata.pop("physical_feature_names")),
        channels=tuple(metadata.pop("channels")),
        **metadata,
    )


class CachedMotionSequenceDataset(Sequence[MotionSequence]):
    """One representation cache, bound to the data it was encoded from.

    A cache is accepted for a cohort when either the cohort fingerprint matches
    (same cohort file) or, given the cohort's per-dataset canonical-cache
    fingerprints, every exposed dataset was encoded from the identical canonical
    cache. The second path lets a cache built under one cohort serve a task
    cohort with different split roles: a frozen encoder's output depends on the
    raw recordings and the encoder, never on which split a recording sits in.
    Datasets whose fingerprints disagree are hidden rather than served.
    """

    def __init__(
        self,
        root: Path,
        *,
        mmap: bool = True,
        manifest_fingerprint: str | None = None,
        cache_fingerprints: Mapping[str, str] | None = None,
    ) -> None:
        self.root = Path(root)
        metadata_path = self.root / "cache.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"representation cache metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != REPRESENTATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported representation cache schema in {metadata_path}")
        exposed: set[str] | None = None
        cohort_match = (
            manifest_fingerprint is not None
            and self.metadata.get("cohort_fingerprint") == manifest_fingerprint
        )
        if manifest_fingerprint is not None and not cohort_match:
            stored = self.metadata.get("cache_fingerprints")
            if cache_fingerprints is None or not isinstance(stored, Mapping):
                raise ValueError("representation cache belongs to a different cohort manifest")
            exposed = {
                dataset
                for dataset, fingerprint in stored.items()
                if cache_fingerprints.get(dataset) == fingerprint
            }
            if not exposed:
                raise ValueError(
                    "representation cache shares no canonical cache with the cohort"
                )
        self.exposed_datasets = exposed
        rows_path = self.root / "manifest.jsonl"
        self.rows = tuple(
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if len(self.rows) != self.metadata.get("sequence_count"):
            raise ValueError("representation manifest count disagrees with cache metadata")
        keys = [
            (row["dataset"], row["recording_id"], row["stream_id"])
            for row in self.rows
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("representation cache contains duplicate sequence identities")
        self.index_by_key = {
            key: index
            for index, key in enumerate(keys)
            if exposed is None or key[0] in exposed
        }
        self.mmap = mmap

    @property
    def datasets(self) -> tuple[str, ...]:
        return tuple(sorted({key[0] for key in self.index_by_key}))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int | slice) -> MotionSequence | list[MotionSequence]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        return read_motion_sequence(
            self.root / self.rows[index]["directory"], mmap=self.mmap
        )

    def __iter__(self) -> Iterator[MotionSequence]:
        for index in range(len(self)):
            yield self[index]

    def get(self, dataset: str, recording_id: str, stream_id: str) -> MotionSequence:
        try:
            index = self.index_by_key[(dataset, recording_id, stream_id)]
        except KeyError as error:
            raise KeyError(
                f"representation not found: {(dataset, recording_id, stream_id)!r}"
            ) from error
        return self[index]


class MotionSequenceUnion:
    """Read-through union of representation caches with disjoint sequence keys.

    A dataset may appear in more than one member so a complete-timeline cache and
    an independently bounded-event cache can serve one task. Exact
    ``(dataset, recording, stream)`` identities must remain unique.
    """

    def __init__(self, members: Sequence[CachedMotionSequenceDataset]) -> None:
        if not members:
            raise ValueError("representation union needs at least one cache")
        self.members = tuple(members)
        self.members_by_dataset: dict[str, list[CachedMotionSequenceDataset]] = {}
        for member in self.members:
            for dataset in member.datasets:
                self.members_by_dataset.setdefault(dataset, []).append(member)
        for dataset, dataset_members in self.members_by_dataset.items():
            keys = [
                key
                for member in dataset_members
                for key in member.index_by_key
                if key[0] == dataset
            ]
            if len(keys) != len(set(keys)):
                raise ValueError(
                    f"dataset {dataset!r} has duplicate sequence identities across caches"
                )
        encoders = {
            json.dumps(member.metadata.get("encoder_provenance"), sort_keys=True)
            for member in self.members
        }
        if len(encoders) != 1:
            raise ValueError("representation caches were produced by different encoders")
        self.metadata = dict(self.members[0].metadata)
        self.metadata["datasets"] = sorted(self.members_by_dataset)
        self.metadata["member_roots"] = [str(member.root) for member in self.members]
        self.metadata["stride_seconds_by_dataset"] = {}
        for dataset, dataset_members in sorted(self.members_by_dataset.items()):
            strides = {
                float(member.metadata.get("stride_seconds", 1.0))
                for member in dataset_members
            }
            if len(strides) != 1:
                raise ValueError(
                    f"representation caches disagree on stride for {dataset!r}"
                )
            self.metadata["stride_seconds_by_dataset"][dataset] = strides.pop()

    @property
    def datasets(self) -> tuple[str, ...]:
        return tuple(sorted(self.members_by_dataset))

    def get(self, dataset: str, recording_id: str, stream_id: str) -> MotionSequence:
        try:
            members = self.members_by_dataset[dataset]
        except KeyError as error:
            raise KeyError(f"no representation cache serves dataset {dataset!r}") from error
        for member in members:
            if (dataset, recording_id, stream_id) in member.index_by_key:
                return member.get(dataset, recording_id, stream_id)
        raise KeyError(f"representation not found: {(dataset, recording_id, stream_id)!r}")

    def stride_seconds(self, dataset: str) -> float:
        try:
            members = self.members_by_dataset[dataset]
        except KeyError as error:
            raise KeyError(f"no representation cache serves dataset {dataset!r}") from error
        strides = {float(member.metadata.get("stride_seconds", 1.0)) for member in members}
        if len(strides) != 1:
            raise ValueError(f"representation caches disagree on stride for {dataset!r}")
        return strides.pop()


def open_representations(
    roots: Sequence[Path],
    *,
    cohort: CohortManifest,
    mmap: bool = True,
) -> CachedMotionSequenceDataset | MotionSequenceUnion:
    """Open one or more caches validated against ``cohort`` (see the class docs)."""

    members = [
        CachedMotionSequenceDataset(
            root,
            mmap=mmap,
            manifest_fingerprint=cohort.fingerprint,
            cache_fingerprints=cohort.cache_fingerprints,
        )
        for root in roots
    ]
    if len(members) == 1:
        return members[0]
    return MotionSequenceUnion(members)


def require_complete_representations(
    representations: CachedMotionSequenceDataset | MotionSequenceUnion,
) -> None:
    """Reject pilot caches from a train/evaluation common-unit intersection.

    ``--limit`` is useful for a mechanical cache-build smoke, but it does not
    identify a defensible population of task units.  Keep that distinction at
    the cache boundary so a later common-unit file cannot silently turn a
    truncated cache into a reportable comparison.
    """

    members = (
        representations.members
        if isinstance(representations, MotionSequenceUnion)
        else (representations,)
    )
    limited = [str(member.root) for member in members if member.metadata.get("selection_limit")]
    if limited:
        raise ValueError(
            "official common-unit construction rejects --limit pilot representation caches: "
            + ", ".join(limited)
        )


def _selected_entries(
    manifest: CohortManifest,
    *,
    datasets: set[str] | None,
    splits: set[str] | None,
    limit: int | None,
) -> tuple[CohortEntry, ...]:
    entries = tuple(
        entry
        for entry in manifest.entries
        if (datasets is None or entry.dataset in datasets)
        and (splits is None or entry.split in splits)
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("representation cache limit must be positive")
        entries = entries[:limit]
    if not entries:
        raise ValueError("representation cache selection is empty")
    return entries


@torch.no_grad()
def build_representation_cache(
    output: Path,
    manifest: CohortManifest,
    recording_caches: Mapping[str, CachedRecordingDataset],
    encoder: MotionEncoder,
    *,
    encoder_provenance: Mapping[str, Any],
    datasets: set[str] | None = None,
    splits: set[str] | None = None,
    patch_seconds: float | None = None,
    stride_seconds: float = 1.0,
    limit: int | None = None,
    force: bool = False,
    resume: bool = False,
) -> None:
    """Encode a manifest subset atomically with a frozen encoder."""

    if (patch_seconds is not None and patch_seconds <= 0) or stride_seconds <= 0:
        raise ValueError("patch and stride durations must be positive")
    if isinstance(encoder, nn.Module) and any(
        parameter.requires_grad for parameter in encoder.parameters()
    ):
        raise ValueError("representation cache encoder must be frozen")
    if not encoder_provenance:
        raise ValueError("encoder provenance must be non-empty")
    validate_manifest_caches(manifest, recording_caches)
    entries = _selected_entries(
        manifest, datasets=datasets, splits=splits, limit=limit
    )
    output = Path(output)
    staging = output.with_name(f".{output.name}.staging")
    if output.exists():
        if not force:
            raise FileExistsError(f"representation cache already exists: {output}")
        shutil.rmtree(output)
    if staging.exists() and not resume:
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=resume)
    rows: list[dict[str, Any]] = []
    try:
        for entry in entries:
            recording = recording_caches[entry.dataset][entry.cache_index]
            if recording.recording_id != entry.recording_id:
                raise ValueError("cohort entry no longer matches canonical cache")
            for stream_id in entry.stream_ids:
                directory = _storage_name(entry.dataset, entry.recording_id, stream_id)
                destination = staging / directory
                if resume and destination.exists():
                    try:
                        cached = read_motion_sequence(destination, mmap=False)
                        identity = (
                            cached.dataset,
                            cached.recording_id,
                            cached.stream_id,
                        )
                        expected = (entry.dataset, entry.recording_id, stream_id)
                        if identity != expected:
                            raise ValueError(
                                f"resumed sequence identity {identity!r} != {expected!r}"
                            )
                    except Exception:
                        shutil.rmtree(destination, ignore_errors=True)
                    else:
                        rows.append(
                            {
                                "dataset": entry.dataset,
                                "recording_id": entry.recording_id,
                                "stream_id": stream_id,
                                "split": entry.split,
                                "directory": directory,
                            }
                        )
                        continue
                encode_kwargs = {
                    "stream_id": stream_id,
                    "stride_seconds": stride_seconds,
                }
                if patch_seconds is not None:
                    encode_kwargs["patch_seconds"] = patch_seconds
                sequence = encoder.encode_recording(recording, **encode_kwargs)
                write_motion_sequence(sequence, destination)
                rows.append(
                    {
                        "dataset": entry.dataset,
                        "recording_id": entry.recording_id,
                        "stream_id": stream_id,
                        "split": entry.split,
                        "directory": directory,
                    }
                )
        (staging / "manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        metadata = {
            "schema_version": REPRESENTATION_SCHEMA_VERSION,
            "cohort_name": manifest.name,
            "cohort_fingerprint": manifest.fingerprint,
            "cache_fingerprints": {
                dataset: manifest.cache_fingerprints[dataset]
                for dataset in sorted({entry.dataset for entry in entries})
            },
            "encoder_provenance": _jsonable(encoder_provenance),
            "patch_seconds": (
                patch_seconds
                if patch_seconds is not None
                else float(getattr(encoder, "window_seconds", 1.0))
            ),
            "stride_seconds": stride_seconds,
            "datasets": sorted({entry.dataset for entry in entries}),
            "splits": sorted({entry.split for entry in entries}),
            "recording_count": len(entries),
            "sequence_count": len(rows),
            "selection_limit": limit,
        }
        (staging / "cache.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    except Exception:
        if not resume:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _bounded_raw_recording(
    recording: RawRecording, segment: BoundedSegment
) -> RawRecording:
    matches = [stream for stream in recording.streams if stream.stream_id == segment.stream_id]
    if len(matches) != 1:
        raise ValueError(f"bounded segment has no unique stream {segment.stream_id!r}")
    stream = matches[0]
    timestamps = np.asarray(stream.timestamps_sec)
    left = int(np.searchsorted(timestamps, segment.start_sec, side="left"))
    right = int(np.searchsorted(timestamps, segment.end_sec, side="left"))
    if right <= left:
        raise ValueError("bounded segment contains no sensor samples")
    selected_timestamps = timestamps[left:right].astype(np.float64, copy=True)
    selected_timestamps -= selected_timestamps[0]
    duration = float(segment.end_sec - segment.start_sec)
    return RawRecording(
        dataset=recording.dataset,
        recording_id=bounded_representation_id(recording.recording_id, segment.event_index),
        subject_id=recording.subject_id,
        session_id=recording.session_id,
        streams=(
            SensorStream(
                stream_id=stream.stream_id,
                placement=stream.placement,
                device=stream.device,
                timestamps_sec=selected_timestamps,
                values=np.asarray(stream.values[left:right]).copy(),
                channels=stream.channels,
                valid=np.asarray(stream.valid[left:right]).copy(),
                gravity_state=stream.gravity_state,
                nominal_rate_hz=stream.nominal_rate_hz,
                metadata={**dict(stream.metadata), "bounded_source_interval_sec": [segment.start_sec, segment.end_sec]},
            ),
        ),
        events=(
            EventInterval(
                start_sec=0.0,
                end_sec=duration,
                label="bounded_execution",
                annotation_kind="bounded_representation",
            ),
        ),
        split=recording.split,
        metadata={
            **dict(recording.metadata),
            "source_recording_id": recording.recording_id,
            "source_event_index": segment.event_index,
            "source_interval_sec": [segment.start_sec, segment.end_sec],
        },
    )


@torch.no_grad()
def build_bounded_representation_cache(
    output: Path,
    manifest: CohortManifest,
    recording_caches: Mapping[str, CachedRecordingDataset],
    encoder: MotionEncoder,
    segments: Sequence[BoundedSegment],
    *,
    encoder_provenance: Mapping[str, Any],
    patch_seconds: float | None = None,
    stride_seconds: float = 1.0,
    force: bool = False,
    resume: bool = False,
) -> None:
    """Encode independently bounded events without surrounding-recording context."""

    if (patch_seconds is not None and patch_seconds <= 0) or stride_seconds <= 0:
        raise ValueError("patch and stride durations must be positive")
    if isinstance(encoder, nn.Module) and any(
        parameter.requires_grad for parameter in encoder.parameters()
    ):
        raise ValueError("representation cache encoder must be frozen")
    validate_manifest_caches(manifest, recording_caches)
    unique = tuple(sorted(set(segments)))
    if not unique:
        raise ValueError("bounded representation selection is empty")
    bounds_by_identity: dict[tuple[str, str, str, int], tuple[float, float]] = {}
    for segment in unique:
        identity = (
            segment.dataset,
            segment.recording_id,
            segment.stream_id,
            segment.event_index,
        )
        bounds = (segment.start_sec, segment.end_sec)
        previous = bounds_by_identity.setdefault(identity, bounds)
        if previous != bounds:
            raise ValueError(
                "one bounded representation identity was requested with conflicting "
                f"intervals: {identity!r}: {previous!r} versus {bounds!r}"
            )
    cohort_lookup = {
        (entry.dataset, entry.cache_index, entry.recording_id): entry
        for entry in manifest.entries
    }
    output = Path(output)
    staging = output.with_name(f".{output.name}.staging")
    if output.exists():
        if not force:
            raise FileExistsError(f"representation cache already exists: {output}")
        shutil.rmtree(output)
    if staging.exists() and not resume:
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=resume)
    rows: list[dict[str, Any]] = []
    try:
        for segment in unique:
            entry_key = (segment.dataset, segment.cache_index, segment.recording_id)
            try:
                entry = cohort_lookup[entry_key]
            except KeyError as error:
                raise ValueError(f"bounded segment lies outside the cohort: {entry_key!r}") from error
            recording = recording_caches[segment.dataset][segment.cache_index]
            if recording.recording_id != segment.recording_id:
                raise ValueError("bounded segment no longer matches its canonical cache")
            derived_id = bounded_representation_id(segment.recording_id, segment.event_index)
            directory = _storage_name(segment.dataset, derived_id, segment.stream_id)
            destination = staging / directory
            if resume and destination.exists():
                try:
                    cached = read_motion_sequence(destination, mmap=False)
                    if (cached.dataset, cached.recording_id, cached.stream_id) != (
                        segment.dataset,
                        derived_id,
                        segment.stream_id,
                    ):
                        raise ValueError("resumed bounded sequence identity is stale")
                except Exception:
                    shutil.rmtree(destination, ignore_errors=True)
                else:
                    rows.append(
                        {
                            "dataset": segment.dataset,
                            "recording_id": derived_id,
                            "stream_id": segment.stream_id,
                            "split": entry.split,
                            "directory": directory,
                        }
                    )
                    continue
            bounded = _bounded_raw_recording(recording, segment)
            encode_kwargs: dict[str, Any] = {
                "stream_id": segment.stream_id,
                "stride_seconds": stride_seconds,
            }
            if patch_seconds is not None:
                encode_kwargs["patch_seconds"] = patch_seconds
            sequence = encoder.encode_recording(bounded, **encode_kwargs)
            # Matching and manifest annotations use the original recording clock.
            sequence = replace(
                sequence,
                recording_id=derived_id,
                intervals_sec=sequence.intervals_sec + float(segment.start_sec),
            )
            write_motion_sequence(sequence, destination)
            rows.append(
                {
                    "dataset": segment.dataset,
                    "recording_id": derived_id,
                    "stream_id": segment.stream_id,
                    "split": entry.split,
                    "directory": directory,
                }
            )
        (staging / "manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        datasets = sorted({segment.dataset for segment in unique})
        metadata = {
            "schema_version": REPRESENTATION_SCHEMA_VERSION,
            "cohort_name": manifest.name,
            "cohort_fingerprint": manifest.fingerprint,
            "cache_fingerprints": {
                dataset: manifest.cache_fingerprints[dataset] for dataset in datasets
            },
            "encoder_provenance": _jsonable(encoder_provenance),
            "patch_seconds": (
                patch_seconds
                if patch_seconds is not None
                else float(getattr(encoder, "window_seconds", 1.0))
            ),
            "stride_seconds": stride_seconds,
            "datasets": datasets,
            "splits": sorted({row["split"] for row in rows}),
            "recording_count": len(unique),
            "sequence_count": len(rows),
            "selection_limit": None,
            "bounded_events": True,
        }
        (staging / "cache.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    except Exception:
        if not resume:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def backfill_cache_fingerprints(root: Path, cohort: CohortManifest) -> dict[str, str]:
    """Record per-dataset canonical-cache fingerprints on a cache built before
    they were persisted. Only the cohort the cache was built from can vouch for
    them, so the cohort fingerprint must match exactly."""

    root = Path(root)
    metadata_path = root / "cache.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("cohort_fingerprint") != cohort.fingerprint:
        raise ValueError(f"{root}: cache was not built from cohort {cohort.name!r}")
    fingerprints = {
        dataset: cohort.cache_fingerprints[dataset] for dataset in metadata["datasets"]
    }
    if metadata.get("cache_fingerprints") != fingerprints:
        metadata["cache_fingerprints"] = fingerprints
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return fingerprints
