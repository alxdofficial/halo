"""Tensor contracts for dense recurrence candidates and source supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _same_shape(name: str, tensor: torch.Tensor, shape: tuple[int, int]) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


@dataclass(frozen=True)
class CandidateBatch:
    """Padded dense candidates pooled from a batch of complete timelines."""

    embeddings: torch.Tensor
    candidate_mask: torch.Tensor
    start_sec: torch.Tensor
    end_sec: torch.Tensor
    scale_index: torch.Tensor
    start_patch: torch.Tensor
    end_patch: torch.Tensor
    recording_id: torch.Tensor

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, candidate, feature]")
        prefix = self.embeddings.shape[:2]
        for name in (
            "candidate_mask",
            "start_sec",
            "end_sec",
            "scale_index",
            "start_patch",
            "end_patch",
            "recording_id",
        ):
            _same_shape(name, getattr(self, name), prefix)
        if self.candidate_mask.dtype != torch.bool:
            raise TypeError("candidate_mask must be boolean")
        if not torch.isfinite(self.embeddings[self.candidate_mask]).all():
            raise ValueError("valid candidate embeddings must be finite")
        if not torch.all(
            self.end_sec[self.candidate_mask] > self.start_sec[self.candidate_mask]
        ):
            raise ValueError("valid candidates must have positive physical duration")
        if not torch.all(
            self.end_patch[self.candidate_mask] > self.start_patch[self.candidate_mask]
        ):
            raise ValueError("valid candidates must contain at least one patch")

    @property
    def valid_count(self) -> int:
        return int(self.candidate_mask.sum().item())


@dataclass(frozen=True)
class EventBatch:
    """Exact source events used only to derive training targets.

    ``scope_id`` identifies a dataset and annotation track. ``label_id`` has
    meaning only within that scope. ``instance_id`` identifies one bounded
    execution and prevents overlapping candidate copies from becoming a
    positive training pair.
    """

    start_sec: torch.Tensor
    end_sec: torch.Tensor
    label_id: torch.Tensor
    instance_id: torch.Tensor
    scope_id: torch.Tensor
    event_mask: torch.Tensor
    exhaustive: torch.Tensor

    def __post_init__(self) -> None:
        if self.start_sec.ndim != 2:
            raise ValueError("event tensors must have shape [batch, event]")
        prefix = self.start_sec.shape
        for name in ("end_sec", "label_id", "instance_id", "scope_id", "event_mask"):
            _same_shape(name, getattr(self, name), prefix)
        if self.event_mask.dtype != torch.bool:
            raise TypeError("event_mask must be boolean")
        if self.exhaustive.shape != (prefix[0],) or self.exhaustive.dtype != torch.bool:
            raise ValueError("exhaustive must be boolean with shape [batch]")
        if not torch.all(
            self.end_sec[self.event_mask] > self.start_sec[self.event_mask]
        ):
            raise ValueError("valid events must have positive duration")

    def to(self, device: torch.device | str) -> "EventBatch":
        return EventBatch(
            start_sec=self.start_sec.to(device),
            end_sec=self.end_sec.to(device),
            label_id=self.label_id.to(device),
            instance_id=self.instance_id.to(device),
            scope_id=self.scope_id.to(device),
            event_mask=self.event_mask.to(device),
            exhaustive=self.exhaustive.to(device),
        )


@dataclass(frozen=True)
class CandidateTargets:
    """Scoped equivalence targets assigned to dense temporal candidates."""

    label_id: torch.Tensor
    instance_id: torch.Tensor
    scope_id: torch.Tensor
    assigned_mask: torch.Tensor
    background_mask: torch.Tensor
    best_iou: torch.Tensor

    def __post_init__(self) -> None:
        if self.label_id.ndim != 2:
            raise ValueError("candidate targets must have shape [batch, candidate]")
        prefix = self.label_id.shape
        for name in (
            "instance_id",
            "scope_id",
            "assigned_mask",
            "background_mask",
            "best_iou",
        ):
            _same_shape(name, getattr(self, name), prefix)
        if (
            self.assigned_mask.dtype != torch.bool
            or self.background_mask.dtype != torch.bool
        ):
            raise TypeError("target masks must be boolean")
        if torch.any(self.assigned_mask & self.background_mask):
            raise ValueError("a candidate cannot be both assigned and background")
