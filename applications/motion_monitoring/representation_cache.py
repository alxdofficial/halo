"""Frozen, provenance-bound ``MotionSequence`` representation caches."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict
from hashlib import sha1, sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from applications.motion_monitoring.data.cache import CachedRecordingDataset
from applications.motion_monitoring.data.manifests import (
    CohortEntry,
    CohortManifest,
    validate_manifest_caches,
)
from applications.motion_monitoring.sequence import MotionEncoder, MotionSequence


REPRESENTATION_SCHEMA_VERSION = 1


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
    def __init__(
        self,
        root: Path,
        *,
        mmap: bool = True,
        manifest_fingerprint: str | None = None,
    ) -> None:
        self.root = Path(root)
        metadata_path = self.root / "cache.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"representation cache metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != REPRESENTATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported representation cache schema in {metadata_path}")
        if (
            manifest_fingerprint is not None
            and self.metadata.get("cohort_fingerprint") != manifest_fingerprint
        ):
            raise ValueError("representation cache belongs to a different cohort manifest")
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
        self.index_by_key = {key: index for index, key in enumerate(keys)}
        self.mmap = mmap

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
    patch_seconds: float = 1.0,
    stride_seconds: float = 1.0,
    limit: int | None = None,
    force: bool = False,
) -> None:
    """Encode a manifest subset atomically with a frozen encoder."""

    if patch_seconds <= 0 or stride_seconds <= 0:
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
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    try:
        for entry in entries:
            recording = recording_caches[entry.dataset][entry.cache_index]
            if recording.recording_id != entry.recording_id:
                raise ValueError("cohort entry no longer matches canonical cache")
            for stream_id in entry.stream_ids:
                sequence = encoder.encode_recording(
                    recording,
                    stream_id=stream_id,
                    patch_seconds=patch_seconds,
                    stride_seconds=stride_seconds,
                )
                directory = _storage_name(entry.dataset, entry.recording_id, stream_id)
                write_motion_sequence(sequence, staging / directory)
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
            "encoder_provenance": _jsonable(encoder_provenance),
            "patch_seconds": patch_seconds,
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
        shutil.rmtree(staging, ignore_errors=True)
        raise
