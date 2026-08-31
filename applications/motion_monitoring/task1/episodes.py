"""Episode contracts and cache adapters for arbitrary-task detection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from applications.motion_monitoring.data.cache import CachedRecordingDataset
from applications.motion_monitoring.data.contracts import EventInterval, RawRecording
from applications.motion_monitoring.sequence import MotionSequence


@dataclass(frozen=True)
class EmbeddingSequence:
    """Timestamped patch embeddings exported by any compatible encoder."""

    embeddings: torch.Tensor
    intervals_sec: torch.Tensor
    valid: torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        embeddings = torch.as_tensor(self.embeddings)
        intervals = torch.as_tensor(self.intervals_sec)
        valid = torch.as_tensor(self.valid, dtype=torch.bool)
        if embeddings.ndim != 2 or not embeddings.is_floating_point():
            raise ValueError("embeddings must be a floating [time, feature] tensor")
        if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
            raise ValueError("embedding sequences must be non-empty")
        if intervals.shape != (len(embeddings), 2) or valid.shape != (len(embeddings),):
            raise ValueError(
                "embedding times and validity must agree with sequence length"
            )
        if not torch.isfinite(embeddings).all() or not torch.isfinite(intervals).all():
            raise ValueError("embedding sequences and intervals must be finite")
        if torch.any(torch.linalg.vector_norm(embeddings[valid], dim=1) <= 1e-12):
            raise ValueError("valid embedding patches must have non-zero row norms")
        if torch.any(intervals[:, 1] <= intervals[:, 0]):
            raise ValueError("every embedding interval must have positive duration")
        if len(intervals) > 1 and (
            torch.any(intervals[1:, 0] < intervals[:-1, 0])
            or torch.any(intervals[1:, 1] < intervals[:-1, 1])
        ):
            raise ValueError("embedding intervals must be ordered in physical time")
        if not torch.any(valid):
            raise ValueError(
                "an embedding sequence must contain at least one valid patch"
            )
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "intervals_sec", intervals)
        object.__setattr__(self, "valid", valid)

    @property
    def feature_dim(self) -> int:
        return int(self.embeddings.shape[1])


def from_motion_sequence(sequence: MotionSequence) -> EmbeddingSequence:
    """Use the shared encoder export without copying or detaching its gradients."""

    return EmbeddingSequence(
        embeddings=sequence.embeddings,
        intervals_sec=sequence.intervals_sec,
        valid=sequence.valid,
        metadata={
            "dataset": sequence.dataset,
            "recording_id": sequence.recording_id,
            "subject_id": sequence.subject_id,
            "session_id": sequence.session_id,
            "stream_id": sequence.stream_id,
            "placement": sequence.placement,
            "gravity_state": sequence.gravity_state,
        },
    )


@dataclass(frozen=True)
class DetectionEpisode:
    """One enrollment reference and one complete query timeline."""

    reference: EmbeddingSequence
    query: EmbeddingSequence
    targets_sec: torch.Tensor
    loss_valid: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        targets = torch.as_tensor(self.targets_sec)
        if targets.ndim != 2 or targets.shape[1] != 2:
            raise ValueError("targets must have shape [events, 2]")
        if not torch.isfinite(targets).all() or torch.any(
            targets[:, 1] <= targets[:, 0]
        ):
            raise ValueError(
                "target intervals must be finite and have positive duration"
            )
        if self.reference.feature_dim != self.query.feature_dim:
            raise ValueError("reference and query embedding dimensions must match")
        loss_valid = (
            self.query.valid
            if self.loss_valid is None
            else torch.as_tensor(self.loss_valid, dtype=torch.bool)
        )
        if loss_valid.shape != self.query.valid.shape:
            raise ValueError("loss_valid must match the query sequence length")
        if torch.any(loss_valid & ~self.query.valid):
            raise ValueError("loss_valid cannot enable an invalid query patch")
        query_start = self.query.intervals_sec[0, 0]
        query_end = self.query.intervals_sec[-1, 1]
        if len(targets) and (
            torch.any(targets[:, 0] < query_start)
            or torch.any(targets[:, 1] > query_end)
        ):
            raise ValueError("targets must lie within the query timeline")
        unavailable = ~loss_valid
        for start, end in targets:
            overlaps_unavailable = (
                unavailable
                & (self.query.intervals_sec[:, 0] < end)
                & (self.query.intervals_sec[:, 1] > start)
            )
            if torch.any(overlaps_unavailable):
                raise ValueError(
                    "a target cannot cross an invalid or guarded query patch"
                )
        object.__setattr__(self, "targets_sec", targets)
        object.__setattr__(self, "loss_valid", loss_valid)


@dataclass(frozen=True)
class DetectionBatch:
    """Padded batch with independent masks for data, loss, and event targets."""

    reference: torch.Tensor
    query: torch.Tensor
    reference_valid: torch.Tensor
    query_valid: torch.Tensor
    loss_valid: torch.Tensor
    query_intervals_sec: torch.Tensor
    endpoint_targets: torch.Tensor
    targets_sec: torch.Tensor
    target_valid: torch.Tensor
    metadata: tuple[Mapping[str, Any], ...]

    def to(self, device: torch.device | str) -> "DetectionBatch":
        values = {
            name: getattr(self, name).to(device)
            for name in (
                "reference",
                "query",
                "reference_valid",
                "query_valid",
                "loss_valid",
                "query_intervals_sec",
                "endpoint_targets",
                "targets_sec",
                "target_valid",
            )
        }
        return DetectionBatch(**values, metadata=self.metadata)


def _endpoint_labels(episode: DetectionEpisode, tolerance_sec: float) -> torch.Tensor:
    labels = torch.zeros(
        len(episode.query.embeddings),
        dtype=torch.bool,
        device=episode.query.embeddings.device,
    )
    if not len(episode.targets_sec):
        return labels
    ends = episode.query.intervals_sec[:, 1]
    valid = episode.loss_valid
    for target_end in episode.targets_sec[:, 1]:
        distance = torch.abs(ends - target_end)
        selected = valid & (distance <= tolerance_sec)
        if not torch.any(selected):
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
            if len(valid_indices):
                nearest = valid_indices[torch.argmin(distance[valid_indices])]
                selected[nearest] = True
        labels |= selected
    return labels


def collate_detection_episodes(
    episodes: Sequence[DetectionEpisode], *, endpoint_tolerance_sec: float = 0.5
) -> DetectionBatch:
    """Pad independent episodes without turning padding or joins into negatives."""

    if not episodes:
        raise ValueError("cannot collate an empty episode batch")
    if endpoint_tolerance_sec < 0:
        raise ValueError("endpoint_tolerance_sec must be non-negative")
    feature_dims = {episode.reference.feature_dim for episode in episodes}
    if len(feature_dims) != 1:
        raise ValueError(
            "all episodes in one batch must use the same feature dimension"
        )

    reference = pad_sequence(
        [item.reference.embeddings for item in episodes], batch_first=True
    )
    query = pad_sequence([item.query.embeddings for item in episodes], batch_first=True)
    reference_valid = pad_sequence(
        [item.reference.valid for item in episodes],
        batch_first=True,
        padding_value=False,
    )
    query_valid = pad_sequence(
        [item.query.valid for item in episodes], batch_first=True, padding_value=False
    )
    loss_valid = pad_sequence(
        [item.loss_valid for item in episodes], batch_first=True, padding_value=False
    )
    query_intervals = pad_sequence(
        [item.query.intervals_sec for item in episodes], batch_first=True
    )
    endpoint_targets = pad_sequence(
        [_endpoint_labels(item, endpoint_tolerance_sec) for item in episodes],
        batch_first=True,
        padding_value=False,
    )
    targets = pad_sequence([item.targets_sec for item in episodes], batch_first=True)
    target_valid = pad_sequence(
        [
            torch.ones(
                len(item.targets_sec),
                dtype=torch.bool,
                device=item.targets_sec.device,
            )
            for item in episodes
        ],
        batch_first=True,
        padding_value=False,
    )
    return DetectionBatch(
        reference=reference,
        query=query,
        reference_valid=reference_valid,
        query_valid=query_valid,
        loss_valid=loss_valid,
        query_intervals_sec=query_intervals,
        endpoint_targets=endpoint_targets,
        targets_sec=targets,
        target_valid=target_valid,
        metadata=tuple(item.metadata for item in episodes),
    )


def _patches_in_interval(
    sequence: EmbeddingSequence, interval: EventInterval
) -> torch.Tensor:
    centers = sequence.intervals_sec.mean(dim=1)
    return (
        sequence.valid & (centers >= interval.start_sec) & (centers < interval.end_sec)
    )


def _trim_reference(
    sequence: EmbeddingSequence, event: EventInterval
) -> EmbeddingSequence:
    selected = torch.nonzero(
        _patches_in_interval(sequence, event), as_tuple=False
    ).flatten()
    if not len(selected):
        raise ValueError("the reference event contains no valid embedding patches")
    if len(selected) > 1 and torch.any(selected[1:] != selected[:-1] + 1):
        raise ValueError("the reference event crosses an invalid embedding gap")
    start, end = int(selected[0]), int(selected[-1]) + 1
    return EmbeddingSequence(
        sequence.embeddings[start:end],
        sequence.intervals_sec[start:end],
        sequence.valid[start:end],
        metadata=sequence.metadata,
    )


def episode_from_recordings(
    reference_recording: RawRecording,
    query_recording: RawRecording,
    reference_sequence: EmbeddingSequence,
    query_sequence: EmbeddingSequence,
    *,
    label: str,
    reference_event_index: int,
    guard_intervals_sec: Sequence[tuple[float, float]] = (),
    allow_same_recording: bool = False,
) -> DetectionEpisode:
    """Build an episode from source events and externally computed embeddings."""

    if not label:
        raise ValueError("episode label must be non-empty")
    if reference_event_index < 0:
        raise IndexError("reference_event_index must be non-negative")
    reference_source_id = str(
        reference_recording.metadata.get(
            "source_recording_id", reference_recording.recording_id
        )
    )
    query_source_id = str(
        query_recording.metadata.get("source_recording_id", query_recording.recording_id)
    )
    if (
        reference_recording.dataset == query_recording.dataset
        and reference_source_id == query_source_id
        and not allow_same_recording
    ):
        raise ValueError(
            "reference and query must be independent recordings from different sources"
        )
    try:
        reference_event = reference_recording.events[reference_event_index]
    except IndexError as error:
        raise IndexError("reference_event_index is outside the event list") from error
    if reference_event.label != label:
        raise ValueError(
            "the selected reference event does not match the requested label"
        )
    if bool(reference_event.metadata.get("clipped_by_recording_crop", False)):
        raise ValueError("the reference event is incomplete at a recording crop boundary")
    for query_event in query_recording.events:
        same_synchronized_event = (
            reference_recording.dataset == query_recording.dataset
            and reference_recording.session_id == query_recording.session_id
            and query_event.label == reference_event.label
            and np.isclose(query_event.start_sec, reference_event.start_sec)
            and np.isclose(query_event.end_sec, reference_event.end_sec)
        )
        reference_execution = reference_event.metadata.get("execution_id")
        query_execution = query_event.metadata.get("execution_id")
        same_explicit_execution = (
            reference_execution is not None
            and query_execution is not None
            and reference_execution == query_execution
        )
        if same_synchronized_event or same_explicit_execution:
            raise ValueError(
                "synchronized views of one execution cannot form reference/query evidence"
            )
    targets = [
        (event.start_sec, event.end_sec)
        for event in query_recording.events
        if event.label == label
        and not bool(event.metadata.get("clipped_by_recording_crop", False))
    ]
    query_start = float(query_sequence.intervals_sec[0, 0])
    query_end = float(query_sequence.intervals_sec[-1, 1])
    targets = [
        (max(start, query_start), min(end, query_end))
        for start, end in targets
        if min(end, query_end) > max(start, query_start)
    ]
    loss_valid = query_sequence.valid.clone()
    for start, end in guard_intervals_sec:
        if not np.isfinite((start, end)).all() or end <= start:
            raise ValueError(
                "guard intervals must be finite and have positive duration"
            )
        overlaps = (query_sequence.intervals_sec[:, 0] < end) & (
            query_sequence.intervals_sec[:, 1] > start
        )
        loss_valid &= ~overlaps
    return DetectionEpisode(
        reference=_trim_reference(reference_sequence, reference_event),
        query=query_sequence,
        targets_sec=torch.tensor(
            targets,
            dtype=query_sequence.intervals_sec.dtype,
            device=query_sequence.intervals_sec.device,
        ).reshape(-1, 2),
        loss_valid=loss_valid,
        metadata={
            "dataset": query_recording.dataset,
            "label": label,
            "reference_recording_id": reference_recording.recording_id,
            "query_recording_id": query_recording.recording_id,
            "reference_source_recording_id": reference_source_id,
            "query_source_recording_id": query_source_id,
            "reference_subject_id": reference_recording.subject_id,
            "query_subject_id": query_recording.subject_id,
        },
    )


@dataclass(frozen=True)
class CachedEventPair:
    reference_index: int
    query_index: int
    label: str
    reference_event_index: int
    reference_stream_id: str
    query_stream_id: str
    guard_intervals_sec: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.reference_index < 0 or self.query_index < 0:
            raise ValueError("cached recording indices must be non-negative")
        if self.reference_event_index < 0:
            raise ValueError("reference event index must be non-negative")
        if not self.label or not self.reference_stream_id or not self.query_stream_id:
            raise ValueError("label and stream identities must be non-empty")


@dataclass(frozen=True)
class RejectedCachedEventPair:
    pair_index: int
    pair: CachedEventPair
    reason: str


@dataclass(frozen=True)
class CachedPairAudit:
    eligible_pairs: tuple[CachedEventPair, ...]
    rejected_pairs: tuple[RejectedCachedEventPair, ...]


def audit_cached_event_pairs(
    cache_root: Path,
    pairs: Sequence[CachedEventPair],
    sequence_provider: Callable[[RawRecording, str], EmbeddingSequence],
    *,
    mmap: bool = True,
    validate_provenance: bool = True,
) -> CachedPairAudit:
    """Preflight a pair manifest and retain auditable data-quality rejections.

    Encoder/provider failures remain hard errors. Only an episode that violates the
    explicit independence, event-boundary, validity, or guard contracts is rejected.
    """

    cache = CachedRecordingDataset(
        cache_root, mmap=mmap, validate_provenance=validate_provenance
    )
    eligible: list[CachedEventPair] = []
    rejected: list[RejectedCachedEventPair] = []
    for pair_index, pair in enumerate(pairs):
        reference_recording = cache[pair.reference_index]
        query_recording = cache[pair.query_index]
        reference_sequence = sequence_provider(
            reference_recording, pair.reference_stream_id
        )
        query_sequence = sequence_provider(query_recording, pair.query_stream_id)
        try:
            episode_from_recordings(
                reference_recording,
                query_recording,
                reference_sequence,
                query_sequence,
                label=pair.label,
                reference_event_index=pair.reference_event_index,
                guard_intervals_sec=pair.guard_intervals_sec,
            )
        except ValueError as error:
            rejected.append(RejectedCachedEventPair(pair_index, pair, str(error)))
        else:
            eligible.append(pair)
    return CachedPairAudit(tuple(eligible), tuple(rejected))


class CachedEventPairDataset(Dataset[DetectionEpisode]):
    """Map canonical cache records into Task-1 episodes through any encoder."""

    def __init__(
        self,
        cache_root: Path,
        pairs: Sequence[CachedEventPair],
        sequence_provider: Callable[[RawRecording, str], EmbeddingSequence],
        *,
        mmap: bool = True,
        validate_provenance: bool = True,
    ) -> None:
        self.cache = CachedRecordingDataset(
            cache_root, mmap=mmap, validate_provenance=validate_provenance
        )
        self.pairs = tuple(pairs)
        self.sequence_provider = sequence_provider

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> DetectionEpisode:
        pair = self.pairs[index]
        reference_recording = self.cache[pair.reference_index]
        query_recording = self.cache[pair.query_index]
        return episode_from_recordings(
            reference_recording,
            query_recording,
            self.sequence_provider(reference_recording, pair.reference_stream_id),
            self.sequence_provider(query_recording, pair.query_stream_id),
            label=pair.label,
            reference_event_index=pair.reference_event_index,
            guard_intervals_sec=pair.guard_intervals_sec,
        )
