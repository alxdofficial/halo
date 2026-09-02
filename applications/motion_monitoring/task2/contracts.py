"""Data contracts for set-conditioned Task-2 change quantification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from applications.motion_monitoring.data.compatibility import (
    SensorCompatibilityKey,
    sensor_compatibility_key,
)
from applications.motion_monitoring.sequence import MotionSequence


EpisodeKind = Literal["accepted_query", "changed_query", "unlabeled_query"]


@dataclass(frozen=True)
class ChangeTargetSpec:
    """One interpretable signed change target and its development-set scale."""

    name: str
    scale: float
    unit: str = "unitless"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("target name must be non-empty")
        if not torch.isfinite(torch.tensor(self.scale)) or self.scale <= 0:
            raise ValueError("target scale must be finite and positive")
        if not self.unit:
            raise ValueError("target unit must be non-empty")


@dataclass(frozen=True)
class BoundedExecution:
    """Timestamped encoder output for one independently bounded execution."""

    embeddings: Tensor
    patch_intervals_sec: Tensor
    patch_mask: Tensor
    dataset: str
    subject_id: str
    session_id: str
    execution_id: str
    task_id: str
    sensor_config: SensorCompatibilityKey | None = None

    def __post_init__(self) -> None:
        embeddings = torch.as_tensor(self.embeddings)
        intervals = torch.as_tensor(self.patch_intervals_sec)
        mask = torch.as_tensor(self.patch_mask)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("embeddings must be non-empty [patch, feature]")
        if intervals.shape != (embeddings.shape[0], 2):
            raise ValueError("patch intervals must have shape [patch, 2]")
        if mask.shape != (embeddings.shape[0],) or mask.dtype != torch.bool:
            raise ValueError("patch mask must be boolean [patch]")
        if not bool(mask.any()):
            raise ValueError("an execution must contain at least one valid patch")
        valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if len(valid_indices) > 1 and bool(
            (valid_indices[1:] != valid_indices[:-1] + 1).any()
        ):
            raise ValueError("a bounded execution cannot cross an internal invalid patch gap")
        if not bool(torch.isfinite(embeddings[mask]).all()):
            raise ValueError("valid embeddings must be finite")
        valid_intervals = intervals[mask]
        if not bool(torch.isfinite(valid_intervals).all()):
            raise ValueError("valid patch intervals must be finite")
        if bool((valid_intervals[:, 1] <= valid_intervals[:, 0]).any()):
            raise ValueError("valid patch intervals must have positive duration")
        if len(valid_intervals) > 1:
            centers = valid_intervals.mean(dim=1)
            if bool((centers[1:] <= centers[:-1]).any()):
                raise ValueError("valid patch intervals must be strictly time ordered")
        identities = (
            self.dataset,
            self.subject_id,
            self.session_id,
            self.execution_id,
            self.task_id,
        )
        if any(not value for value in identities):
            raise ValueError("execution provenance fields must be non-empty")
        if not embeddings.is_floating_point() or not intervals.is_floating_point():
            raise ValueError("embeddings and intervals must use floating-point tensors")
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "patch_intervals_sec", intervals)
        object.__setattr__(self, "patch_mask", mask)


def from_motion_sequence(
    sequence: MotionSequence,
    *,
    execution_id: str,
    task_id: str,
) -> BoundedExecution:
    """Adapt a bounded shared encoder export without severing encoder gradients."""

    return BoundedExecution(
        embeddings=sequence.embeddings,
        patch_intervals_sec=sequence.intervals_sec,
        patch_mask=sequence.valid,
        dataset=sequence.dataset,
        subject_id=sequence.subject_id,
        session_id=sequence.session_id,
        execution_id=execution_id,
        task_id=task_id,
        sensor_config=sensor_compatibility_key(
            device=sequence.device,
            placement=sequence.placement,
            channels=sequence.channels,
            gravity_state=sequence.gravity_state,
        ),
    )


def _same_config(first: BoundedExecution, second: BoundedExecution) -> bool:
    return first.sensor_config == second.sensor_config and (
        first.sensor_config is not None or second.sensor_config is None
    )


@dataclass(frozen=True)
class ExecutionEpisode:
    """A personal reference set, one query, and optional same-person context."""

    accepted_references: tuple[BoundedExecution, ...]
    query: BoundedExecution
    episode_kind: EpisodeKind
    change_targets: Tensor
    target_mask: Tensor
    target_specs: tuple[ChangeTargetSpec, ...]
    personal_context: tuple[BoundedExecution, ...] = ()
    sample_weight: float = 1.0

    def __post_init__(self) -> None:
        references = tuple(self.accepted_references)
        context = tuple(self.personal_context)
        if not references:
            raise ValueError("an episode requires at least one accepted reference")
        if self.episode_kind not in {
            "accepted_query",
            "changed_query",
            "unlabeled_query",
        }:
            raise ValueError(f"invalid episode kind: {self.episode_kind}")

        anchor = references[0]
        target_members = (*references, self.query)
        for execution in target_members:
            if execution.dataset != anchor.dataset:
                raise ValueError("target executions must share one identity namespace")
            if execution.subject_id != anchor.subject_id:
                raise ValueError("target executions must be within subject")
            if execution.task_id != anchor.task_id:
                raise ValueError("target executions must represent the same declared task")
            if execution.embeddings.shape[1] != anchor.embeddings.shape[1]:
                raise ValueError("episode executions must share embedding width")
            if not _same_config(anchor, execution):
                raise ValueError("target executions have incompatible sensor configurations")

        all_members = (*target_members, *context)
        execution_ids = [item.execution_id for item in all_members]
        if len(set(execution_ids)) != len(execution_ids):
            raise ValueError("query, references, and context must be independent executions")
        for execution in context:
            if execution.dataset != anchor.dataset or execution.subject_id != anchor.subject_id:
                raise ValueError("personal context must come from the same subject namespace")
            if execution.embeddings.shape[1] != anchor.embeddings.shape[1]:
                raise ValueError("personal context must share embedding width")
            if not _same_config(anchor, execution):
                raise ValueError("personal context has an incompatible sensor configuration")

        targets = torch.as_tensor(self.change_targets)
        mask = torch.as_tensor(self.target_mask)
        specs = tuple(self.target_specs)
        if targets.shape != (len(specs),):
            raise ValueError("change targets must match the target schema")
        if mask.shape != targets.shape or mask.dtype != torch.bool:
            raise ValueError("target mask must be boolean and match change targets")
        if not bool(torch.isfinite(targets[mask]).all()):
            raise ValueError("valid change targets must be finite")
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("change target names must be unique")
        if not torch.isfinite(torch.tensor(self.sample_weight)) or self.sample_weight <= 0:
            raise ValueError("sample weight must be finite and positive")
        object.__setattr__(self, "accepted_references", references)
        object.__setattr__(self, "personal_context", context)
        object.__setattr__(self, "change_targets", targets)
        object.__setattr__(self, "target_mask", mask)
        object.__setattr__(self, "target_specs", specs)

    @property
    def classification_target(self) -> float:
        return float(self.episode_kind == "changed_query")

    @property
    def classification_valid(self) -> bool:
        return self.episode_kind != "unlabeled_query"

    @property
    def reference_set_id(self) -> str:
        ids = ",".join(sorted(item.execution_id for item in self.accepted_references))
        anchor = self.accepted_references[0]
        return f"{anchor.dataset}/{anchor.subject_id}/{anchor.task_id}/{ids}"


@dataclass(frozen=True)
class EpisodeBatch:
    """Padded set-conditioned episodes ready for the Task-2 head."""

    reference_embeddings: Tensor
    reference_intervals_sec: Tensor
    reference_patch_mask: Tensor
    reference_execution_mask: Tensor
    context_embeddings: Tensor
    context_intervals_sec: Tensor
    context_patch_mask: Tensor
    context_execution_mask: Tensor
    query_embeddings: Tensor
    query_intervals_sec: Tensor
    query_mask: Tensor
    classification_targets: Tensor
    classification_mask: Tensor
    change_targets: Tensor
    target_mask: Tensor
    target_scales: Tensor
    sample_weights: Tensor
    task_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    reference_set_ids: tuple[str, ...]
    target_names: tuple[str, ...]

    def to(self, device: torch.device | str) -> "EpisodeBatch":
        values = {
            name: value.to(device)
            for name, value in self.__dict__.items()
            if isinstance(value, Tensor)
        }
        return EpisodeBatch(
            **values,
            task_ids=self.task_ids,
            subject_ids=self.subject_ids,
            reference_set_ids=self.reference_set_ids,
            target_names=self.target_names,
        )


class ExecutionEpisodeDataset(Dataset[ExecutionEpisode]):
    def __init__(self, episodes: Sequence[ExecutionEpisode]) -> None:
        if not episodes:
            raise ValueError("execution episode dataset must be non-empty")
        self._episodes = tuple(episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index: int) -> ExecutionEpisode:
        return self._episodes[index]


def _pad_executions(
    executions: Sequence[BoundedExecution],
    *,
    dtype: torch.dtype | None = None,
    interval_dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    max_patches = max(item.embeddings.shape[0] for item in executions)
    width = executions[0].embeddings.shape[1]
    device = device or executions[0].embeddings.device
    dtype = dtype or executions[0].embeddings.dtype
    interval_dtype = interval_dtype or executions[0].patch_intervals_sec.dtype
    embeddings = torch.zeros(len(executions), max_patches, width, dtype=dtype, device=device)
    intervals = torch.zeros(len(executions), max_patches, 2, dtype=interval_dtype, device=device)
    mask = torch.zeros(len(executions), max_patches, dtype=torch.bool, device=device)
    for row, execution in enumerate(executions):
        if execution.embeddings.shape[1] != width:
            raise ValueError("all executions in a batch must have the same embedding width")
        count = execution.embeddings.shape[0]
        valid = execution.patch_mask.to(device)
        embeddings[row, :count] = torch.where(
            valid.unsqueeze(-1), execution.embeddings.to(device=device, dtype=dtype), 0.0
        )
        intervals[row, :count] = torch.where(
            valid.unsqueeze(-1),
            execution.patch_intervals_sec.to(device=device, dtype=interval_dtype),
            0.0,
        )
        mask[row, :count] = valid
    return embeddings, intervals, mask


def _pad_execution_sets(
    sets: Sequence[tuple[BoundedExecution, ...]],
    *,
    width: int,
    dtype: torch.dtype,
    interval_dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    max_executions = max((len(items) for items in sets), default=0)
    max_patches = max(
        (item.embeddings.shape[0] for items in sets for item in items), default=0
    )
    embeddings = torch.zeros(
        len(sets), max_executions, max_patches, width, dtype=dtype, device=device
    )
    intervals = torch.zeros(
        len(sets), max_executions, max_patches, 2, dtype=interval_dtype, device=device
    )
    patch_mask = torch.zeros(
        len(sets), max_executions, max_patches, dtype=torch.bool, device=device
    )
    execution_mask = torch.zeros(len(sets), max_executions, dtype=torch.bool, device=device)
    for batch_index, items in enumerate(sets):
        for execution_index, item in enumerate(items):
            count = item.embeddings.shape[0]
            valid = item.patch_mask.to(device)
            embeddings[batch_index, execution_index, :count] = torch.where(
                valid.unsqueeze(-1), item.embeddings.to(device=device, dtype=dtype), 0.0
            )
            intervals[batch_index, execution_index, :count] = torch.where(
                valid.unsqueeze(-1),
                item.patch_intervals_sec.to(device=device, dtype=interval_dtype),
                0.0,
            )
            patch_mask[batch_index, execution_index, :count] = valid
            execution_mask[batch_index, execution_index] = True
    return embeddings, intervals, patch_mask, execution_mask


def collate_execution_episodes(episodes: Sequence[ExecutionEpisode]) -> EpisodeBatch:
    """Pad variable-size episodes without turning padding into evidence."""

    if not episodes:
        raise ValueError("cannot collate an empty episode batch")
    schema = episodes[0].target_specs
    if any(item.target_specs != schema for item in episodes[1:]):
        raise ValueError("all episodes in a batch must use the same target schema")
    first = episodes[0].query
    width = first.embeddings.shape[1]
    device = first.embeddings.device
    all_executions = [
        execution
        for episode in episodes
        for execution in (
            *episode.accepted_references,
            episode.query,
            *episode.personal_context,
        )
    ]
    if any(execution.embeddings.device != device for execution in all_executions):
        raise ValueError("all episode tensors must be on one device before collation")
    dtype = first.embeddings.dtype
    interval_dtype = first.patch_intervals_sec.dtype
    for execution in all_executions:
        dtype = torch.promote_types(dtype, execution.embeddings.dtype)
        interval_dtype = torch.promote_types(
            interval_dtype, execution.patch_intervals_sec.dtype
        )
    references = _pad_execution_sets(
        [item.accepted_references for item in episodes],
        width=width,
        dtype=dtype,
        interval_dtype=interval_dtype,
        device=device,
    )
    context = _pad_execution_sets(
        [item.personal_context for item in episodes],
        width=width,
        dtype=dtype,
        interval_dtype=interval_dtype,
        device=device,
    )
    query = _pad_executions(
        [item.query for item in episodes],
        dtype=dtype,
        interval_dtype=interval_dtype,
        device=device,
    )
    return EpisodeBatch(
        reference_embeddings=references[0],
        reference_intervals_sec=references[1],
        reference_patch_mask=references[2],
        reference_execution_mask=references[3],
        context_embeddings=context[0],
        context_intervals_sec=context[1],
        context_patch_mask=context[2],
        context_execution_mask=context[3],
        query_embeddings=query[0],
        query_intervals_sec=query[1],
        query_mask=query[2],
        classification_targets=torch.tensor(
            [item.classification_target for item in episodes], dtype=torch.float32
        ),
        classification_mask=torch.tensor(
            [item.classification_valid for item in episodes], dtype=torch.bool
        ),
        change_targets=torch.stack([item.change_targets for item in episodes]).float(),
        target_mask=torch.stack([item.target_mask for item in episodes]),
        target_scales=torch.tensor([spec.scale for spec in schema], dtype=torch.float32),
        sample_weights=torch.tensor([item.sample_weight for item in episodes], dtype=torch.float32),
        task_ids=tuple(item.query.task_id for item in episodes),
        subject_ids=tuple(item.query.subject_id for item in episodes),
        reference_set_ids=tuple(item.reference_set_id for item in episodes),
        target_names=tuple(spec.name for spec in schema),
    )
