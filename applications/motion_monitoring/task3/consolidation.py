"""Temporal duplicate consolidation and recurrence graph decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class RecurrenceCluster:
    member_indices: tuple[int, ...]
    medoid_index: int
    mean_affinity: float


def temporal_iou(starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    if starts.ndim != 1 or ends.shape != starts.shape:
        raise ValueError("starts and ends must be one-dimensional and aligned")
    if not torch.all(ends > starts):
        raise ValueError("all intervals must have positive duration")
    intersection = (
        torch.minimum(ends[:, None], ends[None, :])
        - torch.maximum(starts[:, None], starts[None, :])
    ).clamp_min(0)
    union = torch.maximum(ends[:, None], ends[None, :]) - torch.minimum(
        starts[:, None], starts[None, :]
    )
    return intersection / union.clamp_min(torch.finfo(starts.dtype).eps)


def temporal_nms(
    starts: torch.Tensor,
    ends: torch.Tensor,
    scores: torch.Tensor,
    *,
    iou_threshold: float = 0.5,
    max_candidates: int | None = None,
) -> torch.Tensor:
    """Keep high-scoring non-duplicate intervals; higher scores are better."""

    if scores.ndim != 1 or starts.shape != scores.shape or ends.shape != scores.shape:
        raise ValueError("starts, ends, and scores must be aligned vectors")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive when provided")
    if len(scores) == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    overlaps = temporal_iou(starts, ends)
    order = torch.argsort(scores, descending=True, stable=True)
    kept: list[int] = []
    for index in order.tolist():
        if any(float(overlaps[index, prior]) > iou_threshold for prior in kept):
            continue
        kept.append(index)
        if max_candidates is not None and len(kept) >= max_candidates:
            break
    return torch.tensor(kept, dtype=torch.long, device=scores.device)


def _components(edges: Iterable[tuple[int, int]], size: int) -> list[list[int]]:
    parent = list(range(size))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in edges:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    groups: dict[int, list[int]] = {}
    for index in range(size):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def recurrence_clusters(
    starts: torch.Tensor,
    ends: torch.Tensor,
    affinity: torch.Tensor,
    *,
    threshold: float,
    mutual_k: int = 1,
    min_occurrences: int = 2,
    overlap_iou: float = 0.1,
) -> list[RecurrenceCluster]:
    """Group non-overlapping occurrences with a mutual-neighbor graph."""

    count = len(starts)
    if affinity.shape != (count, count):
        raise ValueError("affinity must have shape [candidate, candidate]")
    if mutual_k <= 0 or min_occurrences <= 0:
        raise ValueError("mutual_k and min_occurrences must be positive")
    if not torch.allclose(affinity, affinity.T, atol=1e-6):
        raise ValueError("affinity must be symmetric")
    if count == 0:
        return []
    overlap = temporal_iou(starts, ends)
    eligible = (overlap <= overlap_iou) & ~torch.eye(
        count, dtype=torch.bool, device=affinity.device
    )
    masked = affinity.masked_fill(~eligible, -torch.inf)
    k = min(mutual_k, max(1, count - 1))
    nearest = torch.zeros_like(eligible)
    values, indices = torch.topk(masked, k=k, dim=1)
    nearest.scatter_(1, indices, torch.isfinite(values) & (values >= threshold))
    mutual = nearest & nearest.T
    edges = [
        (left, right)
        for left in range(count)
        for right in range(left + 1, count)
        if bool(mutual[left, right])
    ]

    clusters: list[RecurrenceCluster] = []
    for members in _components(edges, count):
        if len(members) < min_occurrences:
            continue
        member_tensor = torch.tensor(members, dtype=torch.long, device=affinity.device)
        submatrix = affinity[member_tensor[:, None], member_tensor[None, :]]
        without_diagonal = ~torch.eye(
            len(members), dtype=torch.bool, device=affinity.device
        )
        mean_by_member = (submatrix * without_diagonal).sum(dim=1) / max(
            1, len(members) - 1
        )
        medoid = members[int(mean_by_member.argmax().item())]
        mean_affinity = float(submatrix[without_diagonal].mean().item())
        clusters.append(RecurrenceCluster(tuple(members), medoid, mean_affinity))
    return sorted(
        clusters,
        key=lambda cluster: (-len(cluster.member_indices), cluster.member_indices),
    )
