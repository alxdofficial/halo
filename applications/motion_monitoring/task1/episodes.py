"""Episode contracts and cache adapters for arbitrary-task detection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from applications.motion_monitoring.data.cache import CachedRecordingDataset
from applications.motion_monitoring.data.compatibility import (
    SensorCompatibilityKey,
    require_compatible_streams,
    sensor_compatibility_key,
)
from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)
from applications.motion_monitoring.sequence import MotionSequence, localization_intervals


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
        intervals_sec=localization_intervals(sequence),
        valid=sequence.valid,
        metadata={
            "dataset": sequence.dataset,
            "recording_id": sequence.recording_id,
            "subject_id": sequence.subject_id,
            "session_id": sequence.session_id,
            "stream_id": sequence.stream_id,
            "placement": sequence.placement,
            "device": sequence.device,
            "channels": sequence.channels,
            "gravity_state": sequence.gravity_state,
            "sampling_rate_hz": sequence.sampling_rate_hz,
        },
    )


def _sequence_compatibility_key(
    sequence: EmbeddingSequence,
) -> SensorCompatibilityKey | None:
    required = {"device", "placement", "channels", "gravity_state"}
    if not required.issubset(sequence.metadata):
        return None
    return sensor_compatibility_key(
        device=str(sequence.metadata["device"]),
        placement=str(sequence.metadata["placement"]),
        channels=tuple(sequence.metadata["channels"]),
        gravity_state=str(sequence.metadata["gravity_state"]),
    )


@dataclass(frozen=True)
class DetectionEpisode:
    """One enrollment reference and one complete query timeline."""

    reference: EmbeddingSequence
    query: EmbeddingSequence
    targets_sec: torch.Tensor
    alignment_valid: torch.Tensor | None = None
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
        alignment_valid = (
            self.query.valid
            if self.alignment_valid is None
            else torch.as_tensor(self.alignment_valid, dtype=torch.bool)
        )
        loss_valid = (
            alignment_valid
            if self.loss_valid is None
            else torch.as_tensor(self.loss_valid, dtype=torch.bool)
        )
        if alignment_valid.shape != self.query.valid.shape:
            raise ValueError("alignment_valid must match the query sequence length")
        if loss_valid.shape != self.query.valid.shape:
            raise ValueError("loss_valid must match the query sequence length")
        if torch.any(alignment_valid & ~self.query.valid):
            raise ValueError("alignment_valid cannot enable an invalid query patch")
        if torch.any(loss_valid & ~alignment_valid):
            raise ValueError("loss_valid cannot enable an alignment-invalid query patch")
        query_start = self.query.intervals_sec[0, 0]
        query_end = self.query.intervals_sec[-1, 1]
        if len(targets) and (
            torch.any(targets[:, 0] < query_start)
            or torch.any(targets[:, 1] > query_end)
        ):
            raise ValueError("targets must lie within the query timeline")
        unavailable = ~alignment_valid
        for start, end in targets:
            overlaps_unavailable = (
                unavailable
                & (self.query.intervals_sec[:, 0] < end)
                & (self.query.intervals_sec[:, 1] > start)
            )
            if torch.any(overlaps_unavailable):
                raise ValueError(
                    "a target cannot cross an invalid query patch"
                )
        object.__setattr__(self, "targets_sec", targets)
        object.__setattr__(self, "alignment_valid", alignment_valid)
        object.__setattr__(self, "loss_valid", loss_valid)


@dataclass(frozen=True)
class DetectionBatch:
    """Padded batch with independent masks for data, loss, and event targets."""

    reference: torch.Tensor
    query: torch.Tensor
    reference_valid: torch.Tensor
    query_valid: torch.Tensor
    alignment_valid: torch.Tensor
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
                "alignment_valid",
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
    alignment_valid = pad_sequence(
        [item.alignment_valid for item in episodes],
        batch_first=True,
        padding_value=False,
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
        alignment_valid=alignment_valid,
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


MINIMUM_REFERENCE_POSITIONS = 2


def crop_sequence(
    sequence: EmbeddingSequence, start_sec: float, end_sec: float
) -> EmbeddingSequence:
    """Select the positions whose centers fall inside [start_sec, end_sec)."""

    if not np.isfinite((start_sec, end_sec)).all() or end_sec <= start_sec:
        raise ValueError("crop bounds must be finite with positive duration")
    centers = sequence.intervals_sec.mean(dim=1)
    selected = torch.nonzero(
        (centers >= start_sec) & (centers < end_sec), as_tuple=False
    ).flatten()
    if not len(selected):
        raise ValueError("crop contains no representation positions")
    left, right = int(selected[0]), int(selected[-1]) + 1
    return EmbeddingSequence(
        sequence.embeddings[left:right],
        sequence.intervals_sec[left:right],
        sequence.valid[left:right],
        metadata=sequence.metadata,
    )


def _snap_reference_selection(
    sequence: EmbeddingSequence,
    start_sec: float,
    end_sec: float,
    *,
    minimum_positions: int = MINIMUM_REFERENCE_POSITIONS,
    context_bound_sec: float | None = None,
    rng: "random.Random | None" = None,
) -> tuple[int, int, dict[str, object]]:
    """Snap an enrolled interval to the localization grid with a position floor.

    Nearest snapping: a cell belongs to the reference when its center lies inside
    the enrolled interval (with midpoint cells this is the majority-overlap rule).
    If that yields nothing, the closest overlapping valid cell seeds the
    selection. Selections below ``minimum_positions`` are extended with real
    surrounding context inside the same contiguous valid run, choosing the side
    at random when ``rng`` is provided and deterministically (smaller added
    context first) otherwise. See TASK1_REFERENCE_RESOLUTION_SPEC.md.
    """

    if not np.isfinite((start_sec, end_sec)).all() or end_sec <= start_sec:
        raise ValueError("reference interval must be finite with positive duration")
    if minimum_positions < 1:
        raise ValueError("minimum_positions must be at least one")
    intervals = sequence.intervals_sec
    centers = intervals.mean(dim=1)
    widths = (intervals[:, 1] - intervals[:, 0]).abs()
    step = float(widths.median())
    if context_bound_sec is None:
        context_bound_sec = 0.5

    valid = sequence.valid
    inside = valid & (centers >= start_sec) & (centers < end_sec)
    selected = torch.nonzero(inside, as_tuple=False).flatten()
    seeded_by_overlap = False
    if not len(selected):
        overlap = (
            torch.minimum(intervals[:, 1], torch.as_tensor(end_sec, dtype=intervals.dtype))
            - torch.maximum(intervals[:, 0], torch.as_tensor(start_sec, dtype=intervals.dtype))
        ).clamp_min(0.0)
        candidates = torch.nonzero(valid & (overlap > 0), as_tuple=False).flatten()
        if not len(candidates):
            raise ValueError("the reference event contains no valid embedding patches")
        midpoint = 0.5 * (start_sec + end_sec)
        seed = candidates[torch.argmin((centers[candidates] - midpoint).abs())]
        selected = seed.reshape(1)
        seeded_by_overlap = True
    if len(selected) > 1 and torch.any(selected[1:] != selected[:-1] + 1):
        raise ValueError("the reference event crosses an invalid embedding gap")

    left, right = int(selected[0]), int(selected[-1]) + 1
    def context(left_index: int, right_index: int) -> tuple[float, float, float]:
        left_context = max(0.0, float(start_sec - intervals[left_index, 0]))
        right_context = max(0.0, float(intervals[right_index - 1, 1] - end_sec))
        return left_context, right_context, left_context + right_context

    while right - left < minimum_positions:
        options: list[tuple[float, str]] = []
        if left > 0 and bool(valid[left - 1]) and int(selected[0]) - (left - 1) <= 1:
            _, _, prospective = context(left - 1, right)
            if prospective <= context_bound_sec + 1e-9:
                options.append((prospective, "left"))
        if right < len(valid) and bool(valid[right]):
            _, _, prospective = context(left, right + 1)
            if prospective <= context_bound_sec + 1e-9:
                options.append((prospective, "right"))
        if not options:
            raise ValueError(
                "the reference event cannot reach the minimum position floor "
                "within its contiguous valid run and context bound"
            )
        if rng is not None and len(options) > 1:
            _, side = options[rng.randrange(len(options))]
        else:
            _, side = min(options)
        if side == "left":
            left -= 1
        else:
            right += 1
    left_context, right_context, added_context_sec = context(left, right)
    provenance = {
        "reference_positions": right - left,
        "reference_added_context_sec": round(added_context_sec, 6),
        "reference_left_context_sec": round(left_context, 6),
        "reference_right_context_sec": round(right_context, 6),
        "reference_seeded_by_overlap": seeded_by_overlap,
        "reference_grid_step_sec": round(step, 6),
    }
    return left, right, provenance


def _trim_reference(
    sequence: EmbeddingSequence,
    event: EventInterval,
    *,
    interval_sec: tuple[float, float] | None = None,
    rng: "random.Random | None" = None,
) -> tuple[EmbeddingSequence, dict[str, object]]:
    start_sec, end_sec = (
        (float(event.start_sec), float(event.end_sec))
        if interval_sec is None
        else (float(interval_sec[0]), float(interval_sec[1]))
    )
    left, right, provenance = _snap_reference_selection(
        sequence, start_sec, end_sec, rng=rng
    )
    return (
        EmbeddingSequence(
            sequence.embeddings[left:right],
            sequence.intervals_sec[left:right],
            sequence.valid[left:right],
            metadata=sequence.metadata,
        ),
        provenance,
    )


def episode_from_recordings(
    reference_recording: RawRecording,
    query_recording: RawRecording,
    reference_sequence: EmbeddingSequence,
    query_sequence: EmbeddingSequence,
    *,
    label: str,
    reference_event_index: int,
    target_intervals_sec: Sequence[tuple[float, float]] | None = None,
    reference_interval_sec: tuple[float, float] | None = None,
    reference_rng: random.Random | None = None,
    guard_intervals_sec: Sequence[tuple[float, float]] = (),
    allow_same_recording: bool = False,
) -> DetectionEpisode:
    """Build an episode from source events and externally computed embeddings.

    ``reference_interval_sec`` overrides the source event's extent with a derived
    single-execution enrollment interval (see TASK1_REFERENCE_RESOLUTION_SPEC.md
    section A); it must lie inside the source event. ``reference_rng`` randomizes
    the grid-snap context side during training; leave it ``None`` for the
    deterministic development/test draw.
    """

    if not label:
        raise ValueError("episode label must be non-empty")
    if reference_event_index < 0:
        raise IndexError("reference_event_index must be non-negative")
    reference_config = _sequence_compatibility_key(reference_sequence)
    query_config = _sequence_compatibility_key(query_sequence)
    if (reference_config is None) != (query_config is None):
        raise ValueError(
            "reference and query must either both declare sensor configurations or both omit them"
        )
    if (
        reference_config is not None
        and query_config is not None
        and reference_config != query_config
    ):
        raise ValueError(
            "reference and query use incompatible sensor configurations: "
            f"{reference_config} != {query_config}"
        )
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
    if reference_interval_sec is not None:
        interval_start, interval_end = map(float, reference_interval_sec)
        if not np.isfinite((interval_start, interval_end)).all() or interval_end <= interval_start:
            raise ValueError("reference_interval_sec must be finite with positive duration")
        if (
            interval_start < float(reference_event.start_sec) - 1e-6
            or interval_end > float(reference_event.end_sec) + 1e-6
        ):
            raise ValueError("reference_interval_sec must lie inside the source event")
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
    targets = (
        [tuple(map(float, interval)) for interval in target_intervals_sec]
        if target_intervals_sec is not None
        else [
            (event.start_sec, event.end_sec)
            for event in query_recording.events
            if event.label == label
            and not bool(event.metadata.get("clipped_by_recording_crop", False))
        ]
    )
    if any(not np.isfinite(interval).all() or interval[1] <= interval[0] for interval in targets):
        raise ValueError("target intervals must be finite with positive duration")
    query_start = float(query_sequence.intervals_sec[0, 0])
    query_end = float(query_sequence.intervals_sec[-1, 1])
    selected_targets = []
    for start, end in targets:
        if end <= query_start or start >= query_end:
            continue
        if start < query_start - 1e-6 or end > query_end + 1e-6:
            raise ValueError("a target is incomplete at the query crop boundary")
        selected_targets.append((start, end))
    targets = selected_targets
    alignment_valid = query_sequence.valid.clone()
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
    reference, reference_provenance = _trim_reference(
        reference_sequence,
        reference_event,
        interval_sec=reference_interval_sec,
        rng=reference_rng,
    )
    return DetectionEpisode(
        reference=reference,
        query=query_sequence,
        targets_sec=torch.tensor(
            targets,
            dtype=query_sequence.intervals_sec.dtype,
            device=query_sequence.intervals_sec.device,
        ).reshape(-1, 2),
        alignment_valid=alignment_valid,
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
            **reference_provenance,
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


def _selected_stream(recording: RawRecording, stream_id: str) -> SensorStream:
    matches = [stream for stream in recording.streams if stream.stream_id == stream_id]
    if len(matches) != 1:
        raise ValueError(
            f"recording {recording.recording_id!r} has no unique stream {stream_id!r}"
        )
    return matches[0]


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
            require_compatible_streams(
                _selected_stream(reference_recording, pair.reference_stream_id),
                _selected_stream(query_recording, pair.query_stream_id),
            )
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
        require_compatible_streams(
            _selected_stream(reference_recording, pair.reference_stream_id),
            _selected_stream(query_recording, pair.query_stream_id),
        )
        return episode_from_recordings(
            reference_recording,
            query_recording,
            self.sequence_provider(reference_recording, pair.reference_stream_id),
            self.sequence_provider(query_recording, pair.query_stream_id),
            label=pair.label,
            reference_event_index=pair.reference_event_index,
            guard_intervals_sec=pair.guard_intervals_sec,
        )
