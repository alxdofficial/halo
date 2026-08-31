"""Deterministic, leakage-safe application cohort manifests."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from applications.motion_monitoring.data.cache import CachedRecordingDataset
from applications.motion_monitoring.data.contracts import RawRecording
from applications.motion_monitoring.data.splits import (
    RecordingKey,
    recording_key,
    subject_leakage_group,
    validate_subject_disjoint_assignments,
)


MANIFEST_SCHEMA_VERSION = 1
SplitName = Literal["train", "development", "test"]


@dataclass(frozen=True)
class CohortEntry:
    dataset: str
    recording_id: str
    cache_index: int
    subject_id: str
    session_id: str
    leakage_group: str
    split: SplitName
    stream_ids: tuple[str, ...]
    event_counts: Mapping[str, int]


@dataclass(frozen=True)
class CohortManifest:
    schema_version: int
    name: str
    seed: int
    development_fraction: float
    cache_fingerprints: Mapping[str, str]
    entries: tuple[CohortEntry, ...]
    fingerprint: str

    def entries_for(
        self, *, dataset: str | None = None, split: SplitName | None = None
    ) -> tuple[CohortEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if (dataset is None or entry.dataset == dataset)
            and (split is None or entry.split == split)
        )


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def cache_fingerprint(cache: CachedRecordingDataset) -> str:
    """Bind a cohort to both cache provenance and ordered record membership."""

    payload = {
        "cache_json_sha256": _file_sha256(cache.root / "cache.json"),
        "manifest_jsonl_sha256": _file_sha256(cache.root / "manifest.jsonl"),
    }
    return sha256(_canonical_json(payload)).hexdigest()


def _eligible_recording(recording: RawRecording) -> bool:
    """Exclude only source-declared duplicate signal views from the cohort."""

    return not bool(recording.metadata.get("duplicates_parent_exercise_signal", False))


def _development_groups(
    groups: Sequence[str],
    group_cells: Mapping[str, set[tuple[str, str]]],
    *,
    dataset: str,
    seed: int,
    fraction: float,
) -> set[str]:
    if len(groups) < 2:
        raise ValueError(
            f"training/development dataset {dataset!r} needs at least two leakage groups"
        )
    count = max(1, min(len(groups) - 1, round(len(groups) * fraction)))
    support = Counter(
        cell for group in groups for cell in group_cells.get(group, set())
    )
    selected: set[str] = set()
    development_coverage: Counter[tuple[str, str]] = Counter()
    while len(selected) < count:
        candidates: list[tuple[float, str, str]] = []
        for group in groups:
            if group in selected:
                continue
            cells = group_cells.get(group, set())
            # Never strand an annotation cell in development. A cell observed in
            # only one leakage group remains available to fit, not to tune.
            if any(development_coverage[cell] + 1 >= support[cell] for cell in cells):
                continue
            gain = sum(
                1.0 / support[cell]
                for cell in cells
                if support[cell] >= 2 and development_coverage[cell] == 0
            )
            tie = sha256(f"{seed}:{dataset}:{group}".encode("utf-8")).hexdigest()
            candidates.append((-gain, tie, group))
        if not candidates:
            raise ValueError(
                f"cannot form a {count}-group development split for {dataset!r} "
                "without moving a source-scoped annotation entirely out of training"
            )
        _, _, chosen = min(candidates)
        selected.add(chosen)
        development_coverage.update(group_cells.get(chosen, set()))
    return selected


def build_cohort_manifest(
    caches: Mapping[str, CachedRecordingDataset],
    *,
    name: str,
    training_datasets: Sequence[str],
    evaluation_datasets: Sequence[str],
    seed: int = 20260831,
    development_fraction: float = 0.2,
) -> CohortManifest:
    """Create one deterministic train/development/test recording manifest."""

    if not name.strip():
        raise ValueError("manifest name must be non-empty")
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be in (0, 1)")
    training = tuple(training_datasets)
    evaluation = tuple(evaluation_datasets)
    if not training or not evaluation:
        raise ValueError("training and evaluation dataset lists must be non-empty")
    if len(set(training)) != len(training) or len(set(evaluation)) != len(evaluation):
        raise ValueError("dataset roles must not contain duplicates")
    overlap = set(training) & set(evaluation)
    if overlap:
        raise ValueError(f"datasets cannot have both development and test roles: {overlap}")
    expected = set(training) | set(evaluation)
    if set(caches) != expected:
        raise ValueError(
            f"cache datasets must exactly match declared roles: expected {sorted(expected)}"
        )

    entries: list[CohortEntry] = []
    fingerprints: dict[str, str] = {}
    for dataset in sorted(caches):
        cache = caches[dataset]
        fingerprints[dataset] = cache_fingerprint(cache)
        group_cells: dict[str, set[tuple[str, str]]] = {}
        for recording in cache:
            if not _eligible_recording(recording):
                continue
            group = subject_leakage_group(recording)
            group_cells.setdefault(group, set()).update(
                (event.annotation_kind, event.label) for event in recording.events
            )
        groups = sorted(group_cells)
        if not groups:
            raise ValueError(f"dataset {dataset!r} has no eligible recordings")
        development = (
            _development_groups(
                groups,
                group_cells,
                dataset=dataset,
                seed=seed,
                fraction=development_fraction,
            )
            if dataset in training
            else set()
        )
        assignments: dict[RecordingKey, str] = {}
        for cache_index, recording in enumerate(cache):
            if not _eligible_recording(recording):
                continue
            group = subject_leakage_group(recording)
            split: SplitName
            if dataset in evaluation:
                split = "test"
            else:
                split = "development" if group in development else "train"
            assignments[recording_key(recording)] = split
            entries.append(
                CohortEntry(
                    dataset=dataset,
                    recording_id=recording.recording_id,
                    cache_index=cache_index,
                    subject_id=recording.subject_id,
                    session_id=recording.session_id,
                    leakage_group=group,
                    split=split,
                    stream_ids=tuple(stream.stream_id for stream in recording.streams),
                    event_counts=dict(
                        sorted(Counter(event.annotation_kind for event in recording.events).items())
                    ),
                )
            )
        validate_subject_disjoint_assignments(
            (
                recording
                for recording in cache
                if _eligible_recording(recording)
            ),
            assignments,
        )

    entries.sort(key=lambda entry: (entry.dataset, entry.recording_id))
    unsigned = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": name,
        "seed": int(seed),
        "development_fraction": float(development_fraction),
        "cache_fingerprints": dict(sorted(fingerprints.items())),
        "entries": [asdict(entry) for entry in entries],
    }
    fingerprint = sha256(_canonical_json(unsigned)).hexdigest()
    return CohortManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name=name,
        seed=int(seed),
        development_fraction=float(development_fraction),
        cache_fingerprints=dict(sorted(fingerprints.items())),
        entries=tuple(entries),
        fingerprint=fingerprint,
    )


def write_cohort_manifest(manifest: CohortManifest, path: Path) -> None:
    payload = asdict(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_cohort_manifest(path: Path) -> CohortManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported cohort manifest schema in {path}")
    fingerprint = payload.pop("fingerprint", None)
    expected = sha256(_canonical_json(payload)).hexdigest()
    if fingerprint != expected:
        raise ValueError(f"cohort manifest fingerprint mismatch in {path}")
    entries = tuple(
        CohortEntry(
            **{
                **entry,
                "stream_ids": tuple(entry["stream_ids"]),
                "event_counts": dict(entry["event_counts"]),
            }
        )
        for entry in payload.pop("entries")
    )
    manifest = CohortManifest(**payload, entries=entries, fingerprint=fingerprint)
    _validate_manifest_structure(manifest)
    return manifest


def _validate_manifest_structure(manifest: CohortManifest) -> None:
    if not manifest.name.strip() or not 0 < manifest.development_fraction < 1:
        raise ValueError("cohort manifest metadata is invalid")
    keys: set[RecordingKey] = set()
    group_splits: dict[str, set[str]] = {}
    dataset_splits: dict[str, set[str]] = {}
    for entry in manifest.entries:
        key = (entry.dataset, entry.recording_id)
        if key in keys:
            raise ValueError(f"cohort manifest contains duplicate recording: {key!r}")
        keys.add(key)
        if entry.split not in {"train", "development", "test"}:
            raise ValueError(f"cohort manifest contains invalid split: {entry.split!r}")
        if entry.cache_index < 0 or not entry.stream_ids:
            raise ValueError(f"cohort manifest contains an invalid entry: {key!r}")
        group_splits.setdefault(entry.leakage_group, set()).add(entry.split)
        dataset_splits.setdefault(entry.dataset, set()).add(entry.split)
    leaking = {
        group: splits for group, splits in group_splits.items() if len(splits) > 1
    }
    if leaking:
        raise ValueError(f"cohort leakage groups cross splits: {leaking}")
    for dataset, splits in dataset_splits.items():
        if "test" in splits and len(splits) > 1:
            raise ValueError(f"dataset {dataset!r} mixes test with fitting splits")


def validate_manifest_caches(
    manifest: CohortManifest, caches: Mapping[str, CachedRecordingDataset]
) -> None:
    _validate_manifest_structure(manifest)
    if set(caches) != set(manifest.cache_fingerprints):
        raise ValueError("loaded caches do not match manifest datasets")
    for dataset, cache in caches.items():
        if cache_fingerprint(cache) != manifest.cache_fingerprints[dataset]:
            raise ValueError(f"cache changed after manifest freeze: {dataset}")
        entries = manifest.entries_for(dataset=dataset)
        entries_by_index = {entry.cache_index: entry for entry in entries}
        if len(entries_by_index) != len(entries):
            raise ValueError(f"cohort repeats a cache index for {dataset}")
        observed_indices: set[int] = set()
        for cache_index, recording in enumerate(cache):
            eligible = _eligible_recording(recording)
            entry = entries_by_index.get(cache_index)
            if eligible != (entry is not None):
                raise ValueError(
                    f"cohort membership changed for {dataset}[{cache_index}]"
                )
            if entry is None:
                continue
            observed_indices.add(cache_index)
            if recording.recording_id != entry.recording_id:
                raise ValueError(
                    f"cache order changed after manifest freeze: {dataset}[{entry.cache_index}]"
                )
            if (
                recording.subject_id != entry.subject_id
                or recording.session_id != entry.session_id
                or subject_leakage_group(recording) != entry.leakage_group
                or tuple(stream.stream_id for stream in recording.streams)
                != entry.stream_ids
                or dict(
                    sorted(
                        Counter(event.annotation_kind for event in recording.events).items()
                    )
                )
                != entry.event_counts
            ):
                raise ValueError(
                    f"cohort metadata changed after manifest freeze: {dataset}[{cache_index}]"
                )
        if observed_indices != set(entries_by_index):
            raise ValueError(f"cohort contains out-of-range cache indices for {dataset}")
