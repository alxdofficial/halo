"""Canonical on-disk cache for efficient application-task data loading.

The cache is deliberately lossless with respect to the raw adapter contract: it
does not resample, window, impute, or change annotations. Each recording remains
independently addressable so PyTorch workers can load disjoint examples without
re-decoding an entire source release.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from hashlib import sha1, sha256
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)


CACHE_SCHEMA_VERSION = 1
CACHE_PROVENANCE_VERSION = 1
_DATA_ROOT = Path(__file__).resolve().parent
_PAYLOAD_CHECKSUMS_PATH = _DATA_ROOT / "PAYLOAD_CHECKSUMS.json"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def cache_provenance(dataset: str) -> dict[str, Any]:
    """Return the source and converter identity required for a valid cache."""

    from applications.motion_monitoring.data.adapters.registry import ADAPTERS

    try:
        spec = ADAPTERS[dataset]
    except KeyError as error:
        raise KeyError(
            f"no registered adapter for cache dataset {dataset!r}"
        ) from error
    module_path = Path(import_module(spec.module).__file__).resolve()
    contracts_path = Path(__file__).with_name("contracts.py")
    if spec.cache_policy == "derived":
        # A synthesized dataset is exactly reproducible from its generator code
        # and the canonical caches it reads, so those replace the payload tree.
        return {
            "schema_version": CACHE_PROVENANCE_VERSION,
            "source_caches": {
                name: validate_cache_metadata(
                    _DATA_ROOT / "sources" / name / "processed" / "canonical_v1"
                )["provenance"]
                for name in spec.derived_from
            },
            "adapter_module": spec.module,
            "adapter_sha256": _sha256_file(module_path),
            "contracts_sha256": _sha256_file(contracts_path),
        }
    checksums = json.loads(_PAYLOAD_CHECKSUMS_PATH.read_text(encoding="utf-8"))
    try:
        payload = checksums["datasets"][dataset]
    except KeyError as error:
        raise KeyError(
            f"no frozen source payload for cache dataset {dataset!r}"
        ) from error
    return {
        "schema_version": CACHE_PROVENANCE_VERSION,
        "payload_tree_sha256": payload["tree_sha256"],
        "adapter_module": spec.module,
        "adapter_sha256": _sha256_file(module_path),
        "contracts_sha256": _sha256_file(contracts_path),
    }


def verify_source_payload(dataset: str) -> None:
    """Verify every source file before assigning frozen provenance to a cache."""

    from applications.motion_monitoring.data.adapters.registry import ADAPTERS

    spec = ADAPTERS.get(dataset)
    if spec is not None and spec.cache_policy == "derived":
        # Derived datasets read canonical caches; ``cache_provenance`` already
        # rejects a stale or missing source cache.
        cache_provenance(dataset)
        return
    checksums = json.loads(_PAYLOAD_CHECKSUMS_PATH.read_text(encoding="utf-8"))
    try:
        rows = checksums["datasets"][dataset]["files"]
    except KeyError as error:
        raise KeyError(
            f"no frozen source payload for cache dataset {dataset!r}"
        ) from error
    source_root = _DATA_ROOT / "sources"
    for row in rows:
        path = source_root / row["path"]
        if not path.is_file():
            raise FileNotFoundError(f"frozen source file is missing: {path}")
        if path.stat().st_size != row["bytes"] or _sha256_file(path) != row["sha256"]:
            raise ValueError(
                f"frozen source checksum mismatch: {path}; reacquire the dataset "
                "before building its cache"
            )


def validate_cache_metadata(
    root: Path, *, validate_provenance: bool = True
) -> dict[str, Any]:
    metadata_path = root / "cache.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"recording cache metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported cache schema in {metadata_path}")
    dataset = metadata.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError(f"cache dataset identity is missing in {metadata_path}")
    if validate_provenance:
        expected = cache_provenance(dataset)
        if metadata.get("provenance") != expected:
            raise ValueError(
                f"stale cache provenance in {metadata_path}; rebuild {dataset!r} "
                "with applications.motion_monitoring.data.build_cache --force"
            )
    return metadata


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

    def __init__(
        self,
        root: Path,
        *,
        mmap: bool = True,
        validate_provenance: bool = True,
    ) -> None:
        self.root = Path(root)
        self.metadata = validate_cache_metadata(
            self.root, validate_provenance=validate_provenance
        )
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
        if len(self.rows) != self.metadata.get("recording_count"):
            raise ValueError(
                f"cache manifest count disagrees with cache metadata in {self.root}"
            )
        if any(
            row.get("dataset", self.metadata["dataset"]) != self.metadata["dataset"]
            for row in self.rows
        ):
            raise ValueError(
                f"cache manifest contains the wrong dataset in {self.root}"
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
