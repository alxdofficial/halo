"""Data contracts for bounded-execution change-quantification training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from applications.motion_monitoring.sequence import MotionSequence


PairKind = Literal["accepted_variation", "known_change", "unlabeled"]


@dataclass(frozen=True)
class ChangeTargetSpec:
    """One interpretable signed change target and its training scale.

    ``scale`` is a development-set scale in the target's native units. Losses divide
    by it so duration, intensity, and smoothness targets do not compete by unit size.
    """

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
        if not torch.isfinite(embeddings[mask]).all():
            raise ValueError("valid embeddings must be finite")
        valid_intervals = intervals[mask]
        if not torch.isfinite(valid_intervals).all():
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
    )


@dataclass(frozen=True)
class ExecutionPair:
    """A reference/comparison pair with optional supervised change evidence."""

    reference: BoundedExecution
    comparison: BoundedExecution
    pair_kind: PairKind
    change_targets: Tensor
    target_mask: Tensor
    target_specs: tuple[ChangeTargetSpec, ...]
    sample_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.pair_kind not in {"accepted_variation", "known_change", "unlabeled"}:
            raise ValueError(f"invalid pair kind: {self.pair_kind}")
        if self.reference.task_id != self.comparison.task_id:
            raise ValueError("paired executions must represent the same declared task")
        if self.reference.dataset != self.comparison.dataset:
            raise ValueError("paired executions must come from the same identity namespace")
        if self.reference.subject_id != self.comparison.subject_id:
            raise ValueError("training pairs must be within subject")
        if self.reference.execution_id == self.comparison.execution_id:
            raise ValueError("a pair must contain independent executions")
        if self.reference.embeddings.shape[1] != self.comparison.embeddings.shape[1]:
            raise ValueError("paired executions must have the same embedding width")
        targets = torch.as_tensor(self.change_targets)
        mask = torch.as_tensor(self.target_mask)
        specs = tuple(self.target_specs)
        if targets.shape != (len(specs),):
            raise ValueError("change targets must match the target schema")
        if mask.shape != targets.shape or mask.dtype != torch.bool:
            raise ValueError("target mask must be boolean and match change targets")
        if not torch.isfinite(targets[mask]).all():
            raise ValueError("valid change targets must be finite")
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("change target names must be unique")
        if (
            not torch.isfinite(torch.tensor(self.sample_weight))
            or self.sample_weight <= 0
        ):
            raise ValueError("sample weight must be finite and positive")
        object.__setattr__(self, "change_targets", targets)
        object.__setattr__(self, "target_mask", mask)
        object.__setattr__(self, "target_specs", specs)

    @property
    def classification_target(self) -> float:
        if self.pair_kind == "known_change":
            return 1.0
        if self.pair_kind == "accepted_variation":
            return 0.0
        return 0.0

    @property
    def classification_valid(self) -> bool:
        return self.pair_kind != "unlabeled"


@dataclass(frozen=True)
class PairBatch:
    """Padded execution pairs ready for an encoder-agnostic Task-2 head."""

    reference_embeddings: Tensor
    reference_intervals_sec: Tensor
    reference_mask: Tensor
    comparison_embeddings: Tensor
    comparison_intervals_sec: Tensor
    comparison_mask: Tensor
    classification_targets: Tensor
    classification_mask: Tensor
    change_targets: Tensor
    target_mask: Tensor
    target_scales: Tensor
    sample_weights: Tensor
    task_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    target_names: tuple[str, ...]

    def to(self, device: torch.device | str) -> "PairBatch":
        values = {
            name: value.to(device)
            for name, value in self.__dict__.items()
            if isinstance(value, Tensor)
        }
        return PairBatch(
            **values,
            task_ids=self.task_ids,
            subject_ids=self.subject_ids,
            target_names=self.target_names,
        )


class ExecutionPairDataset(Dataset[ExecutionPair]):
    """Minimal dataset wrapper that keeps pair construction outside the trainer."""

    def __init__(self, pairs: Sequence[ExecutionPair]) -> None:
        if not pairs:
            raise ValueError("execution pair dataset must be non-empty")
        self._pairs = tuple(pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> ExecutionPair:
        return self._pairs[index]


def _pad_executions(
    executions: Sequence[BoundedExecution],
) -> tuple[Tensor, Tensor, Tensor]:
    batch_size = len(executions)
    max_patches = max(item.embeddings.shape[0] for item in executions)
    width = executions[0].embeddings.shape[1]
    dtype = executions[0].embeddings.dtype
    device = executions[0].embeddings.device
    embeddings = torch.zeros(
        batch_size, max_patches, width, dtype=dtype, device=device
    )
    interval_dtype = executions[0].patch_intervals_sec.dtype
    intervals = torch.zeros(
        batch_size, max_patches, 2, dtype=interval_dtype, device=device
    )
    mask = torch.zeros(batch_size, max_patches, dtype=torch.bool, device=device)
    for row, execution in enumerate(executions):
        if execution.embeddings.shape[1] != width:
            raise ValueError(
                "all executions in a batch must have the same embedding width"
            )
        count = execution.embeddings.shape[0]
        valid = execution.patch_mask
        embeddings[row, :count] = torch.where(
            valid.unsqueeze(-1),
            execution.embeddings.to(dtype=dtype),
            torch.zeros((), dtype=dtype, device=device),
        )
        intervals[row, :count] = torch.where(
            valid.unsqueeze(-1),
            execution.patch_intervals_sec.to(device=device, dtype=interval_dtype),
            torch.zeros((), dtype=interval_dtype, device=device),
        )
        mask[row, :count] = execution.patch_mask.to(device)
    return embeddings, intervals, mask


def collate_execution_pairs(pairs: Sequence[ExecutionPair]) -> PairBatch:
    """Pad variable-length pairs without turning padding into evidence."""

    if not pairs:
        raise ValueError("cannot collate an empty pair batch")
    schema = pairs[0].target_specs
    if any(pair.target_specs != schema for pair in pairs[1:]):
        raise ValueError("all pairs in a batch must use the same ordered target schema")
    reference = _pad_executions([pair.reference for pair in pairs])
    comparison = _pad_executions([pair.comparison for pair in pairs])
    return PairBatch(
        reference_embeddings=reference[0],
        reference_intervals_sec=reference[1],
        reference_mask=reference[2],
        comparison_embeddings=comparison[0],
        comparison_intervals_sec=comparison[1],
        comparison_mask=comparison[2],
        classification_targets=torch.tensor(
            [pair.classification_target for pair in pairs], dtype=torch.float32
        ),
        classification_mask=torch.tensor(
            [pair.classification_valid for pair in pairs], dtype=torch.bool
        ),
        change_targets=torch.stack([pair.change_targets for pair in pairs]).float(),
        target_mask=torch.stack([pair.target_mask for pair in pairs]),
        target_scales=torch.tensor(
            [spec.scale for spec in schema], dtype=torch.float32
        ),
        sample_weights=torch.tensor(
            [pair.sample_weight for pair in pairs], dtype=torch.float32
        ),
        task_ids=tuple(pair.reference.task_id for pair in pairs),
        subject_ids=tuple(pair.reference.subject_id for pair in pairs),
        target_names=tuple(spec.name for spec in schema),
    )
