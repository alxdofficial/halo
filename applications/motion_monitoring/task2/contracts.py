"""Data contracts for Task-2 change quantification (personal-normative ruler).

An episode is one personal reference set plus one query. The query's role says
what the ruler must do with it: an ``accepted_query`` is another execution of the
same person and task (pull together); a ``modified_query`` is a same-person
execution carrying a declared physical modification of known severity (push
apart); an ``other_subject_query`` is another person's execution of the same
task on the same configuration (push apart); an ``unlabeled_query`` is scored
but never supervises. Every member of an episode shares one dataset, task and
sensor-compatibility key (docs/tasks/TASK2_CHANGE_QUANTIFICATION.md section 3).
"""

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


EpisodeKind = Literal[
    "accepted_query", "modified_query", "other_subject_query", "unlabeled_query"
]
ROLE_INDEX: dict[str, int] = {
    "accepted_query": 0,
    "modified_query": 1,
    "other_subject_query": 2,
    "unlabeled_query": 3,
}
NEGATIVE_ROLES = frozenset({"modified_query", "other_subject_query"})


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
    physical_features: Tensor | None = None
    physical_feature_mask: Tensor | None = None
    physical_feature_names: tuple[str, ...] = ()

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
        physical = self.physical_features
        physical_mask = self.physical_feature_mask
        if (physical is None) != (physical_mask is None):
            raise ValueError("physical features and their mask must be supplied together")
        if physical is not None:
            physical = torch.as_tensor(physical)
            physical_mask = torch.as_tensor(physical_mask)
            if physical.shape[0] != embeddings.shape[0] or physical.shape != physical_mask.shape:
                raise ValueError("physical features must align with execution patches")
            if physical_mask.dtype != torch.bool:
                raise ValueError("physical feature mask must be boolean")
            if physical.shape[1] != len(self.physical_feature_names):
                raise ValueError("physical feature names must match their width")
            if not bool(torch.isfinite(physical[physical_mask]).all()):
                raise ValueError("valid physical features must be finite")
            object.__setattr__(self, "physical_features", physical)
            object.__setattr__(self, "physical_feature_mask", physical_mask)
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
        physical_features=sequence.physical_features,
        physical_feature_mask=sequence.physical_feature_mask,
        physical_feature_names=sequence.physical_feature_names,
    )


def _same_config(first: BoundedExecution, second: BoundedExecution) -> bool:
    return first.sensor_config == second.sensor_config and (
        first.sensor_config is not None or second.sensor_config is None
    )


@dataclass(frozen=True)
class ExecutionEpisode:
    """A personal reference set, one query with a declared role, optional context."""

    accepted_references: tuple[BoundedExecution, ...]
    query: BoundedExecution
    episode_kind: EpisodeKind
    severity: float = 0.0
    modification_kind: str | None = None
    nuisance_kind: str | None = None
    personal_context: tuple[BoundedExecution, ...] = ()
    sample_weight: float = 1.0

    def __post_init__(self) -> None:
        references = tuple(self.accepted_references)
        context = tuple(self.personal_context)
        if not references:
            raise ValueError("an episode requires at least one accepted reference")
        if self.episode_kind not in ROLE_INDEX:
            raise ValueError(f"invalid episode kind: {self.episode_kind}")
        severity = float(self.severity)
        if not torch.isfinite(torch.tensor(severity)) or severity < 0:
            raise ValueError("severity must be finite and non-negative")
        if self.episode_kind == "modified_query":
            if severity <= 0 or not self.modification_kind:
                raise ValueError("a modified query declares a positive severity and a kind")
        elif self.modification_kind is not None:
            raise ValueError("only modified queries carry a modification kind")
        elif self.episode_kind == "other_subject_query":
            severity = 1.0
        else:
            if severity != 0:
                raise ValueError("accepted and unlabeled queries have zero severity")
        if self.nuisance_kind is not None and self.episode_kind != "accepted_query":
            raise ValueError("nuisance transforms are declared only on accepted queries")

        anchor = references[0]
        for execution in references:
            if execution.subject_id != anchor.subject_id:
                raise ValueError("reference executions must be within subject")
        same_person = self.episode_kind != "other_subject_query"
        if same_person and self.query.subject_id != anchor.subject_id:
            raise ValueError("this query kind must be within subject")
        if not same_person and self.query.subject_id == anchor.subject_id:
            raise ValueError("an other-subject query must come from another person")
        for execution in (*references, self.query):
            if execution.dataset != anchor.dataset:
                raise ValueError("target executions must share one identity namespace")
            if execution.task_id != anchor.task_id:
                raise ValueError("target executions must represent the same declared task")
            if execution.embeddings.shape[1] != anchor.embeddings.shape[1]:
                raise ValueError("episode executions must share embedding width")
            if not _same_config(anchor, execution):
                raise ValueError("target executions have incompatible sensor configurations")

        all_members = (*references, self.query, *context)
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
        if not torch.isfinite(torch.tensor(self.sample_weight)) or self.sample_weight <= 0:
            raise ValueError("sample weight must be finite and positive")
        object.__setattr__(self, "accepted_references", references)
        object.__setattr__(self, "personal_context", context)
        object.__setattr__(self, "severity", severity)

    @property
    def role_index(self) -> int:
        return ROLE_INDEX[self.episode_kind]

    @property
    def is_positive(self) -> bool:
        return self.episode_kind == "accepted_query"

    @property
    def is_negative(self) -> bool:
        return self.episode_kind in NEGATIVE_ROLES

    @property
    def reference_set_id(self) -> str:
        ids = ",".join(sorted(item.execution_id for item in self.accepted_references))
        anchor = self.accepted_references[0]
        return f"{anchor.dataset}/{anchor.subject_id}/{anchor.task_id}/{ids}"


@dataclass(frozen=True)
class EpisodeBatch:
    """Padded set-conditioned episodes ready for the Task-2 ruler."""

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
    roles: Tensor
    severities: Tensor
    sample_weights: Tensor
    task_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    reference_set_ids: tuple[str, ...]
    modification_kinds: tuple[str | None, ...]
    nuisance_kinds: tuple[str | None, ...]

    @property
    def positive_mask(self) -> Tensor:
        return self.roles == ROLE_INDEX["accepted_query"]

    @property
    def negative_mask(self) -> Tensor:
        return (self.roles == ROLE_INDEX["modified_query"]) | (
            self.roles == ROLE_INDEX["other_subject_query"]
        )

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
            modification_kinds=self.modification_kinds,
            nuisance_kinds=self.nuisance_kinds,
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
        roles=torch.tensor([item.role_index for item in episodes], dtype=torch.long, device=device),
        severities=torch.tensor(
            [item.severity for item in episodes], dtype=torch.float32, device=device
        ),
        sample_weights=torch.tensor(
            [item.sample_weight for item in episodes], dtype=torch.float32, device=device
        ),
        task_ids=tuple(item.query.task_id for item in episodes),
        subject_ids=tuple(item.query.subject_id for item in episodes),
        reference_set_ids=tuple(item.reference_set_id for item in episodes),
        modification_kinds=tuple(item.modification_kind for item in episodes),
        nuisance_kinds=tuple(item.nuisance_kind for item in episodes),
    )
