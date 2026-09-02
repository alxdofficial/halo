"""Temporal duplicate consolidation and recurrence graph decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised by minimal installations
    njit = None


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


def _temporal_nms_reference(
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


if njit is not None:

    @njit(cache=True)
    def _temporal_nms_compiled(starts, ends, order, iou_threshold, max_candidates):
        kept = np.empty(len(order), dtype=np.int64)
        count = 0
        for index in order:
            overlaps = False
            for kept_offset in range(count):
                prior = kept[kept_offset]
                intersection = max(
                    0.0, min(ends[index], ends[prior]) - max(starts[index], starts[prior])
                )
                union = max(ends[index], ends[prior]) - min(starts[index], starts[prior])
                iou = intersection / union if union > 0 else 0.0
                if iou > iou_threshold:
                    overlaps = True
                    break
            if overlaps:
                continue
            kept[count] = index
            count += 1
            if max_candidates > 0 and count >= max_candidates:
                break
        return kept[:count]


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
    if njit is None:
        return _temporal_nms_reference(
            starts,
            ends,
            scores,
            iou_threshold=iou_threshold,
            max_candidates=max_candidates,
        )
    order = torch.argsort(scores, descending=True, stable=True)
    kept = _temporal_nms_compiled(
        starts.detach().cpu().double().numpy(),
        ends.detach().cpu().double().numpy(),
        order.detach().cpu().numpy(),
        iou_threshold,
        -1 if max_candidates is None else max_candidates,
    )
    return torch.as_tensor(kept, dtype=torch.long, device=scores.device)


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


def _component_mean_affinity(member_embeddings: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal cosine per member without a square similarity matrix."""

    if member_embeddings.ndim != 2 or len(member_embeddings) < 2:
        raise ValueError("a component requires at least two embedding rows")
    component_sum = member_embeddings.sum(dim=0)
    row_sum = member_embeddings @ component_sum
    diagonal = member_embeddings.square().sum(dim=1)
    return (row_sum - diagonal) / (len(member_embeddings) - 1)


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


def recurrence_clusters_blockwise(
    embeddings: torch.Tensor,
    starts_sec: torch.Tensor,
    ends_sec: torch.Tensor,
    *,
    threshold: float,
    min_occurrences: int = 2,
    mutual_k: int = 5,
    block_size: int = 1024,
    overlap_iou: float = 0.1,
) -> list[RecurrenceCluster]:
    """Exact mutual-k recurrence graph without materializing an NxN matrix."""

    if embeddings.ndim != 2 or not embeddings.is_floating_point():
        raise ValueError("embeddings must be floating [candidate, feature]")
    count = len(embeddings)
    if starts_sec.shape != (count,) or ends_sec.shape != (count,):
        raise ValueError("candidate intervals must align with embeddings")
    if count < 2 or mutual_k <= 0 or min_occurrences < 2 or block_size <= 0:
        if count < 2:
            return []
        raise ValueError("graph sizes and occurrence count must be positive")
    if not 0 <= overlap_iou <= 1:
        raise ValueError("overlap_iou must be in [0, 1]")
    if not torch.isfinite(embeddings).all() or not torch.isfinite(starts_sec).all() \
            or not torch.isfinite(ends_sec).all():
        raise ValueError("candidate graph inputs must be finite")
    if not torch.all(ends_sec > starts_sec):
        raise ValueError("candidate intervals must have positive duration")

    normalized = F.normalize(embeddings, dim=-1, eps=1e-8)
    k = min(mutual_k, count - 1)
    neighbor_rows: list[torch.Tensor] = []
    for start in range(0, count, block_size):
        stop = min(count, start + block_size)
        affinity = normalized[start:stop] @ normalized.T
        intersection = (
            torch.minimum(ends_sec[start:stop, None], ends_sec[None, :])
            - torch.maximum(starts_sec[start:stop, None], starts_sec[None, :])
        ).clamp_min(0)
        union = (
            torch.maximum(ends_sec[start:stop, None], ends_sec[None, :])
            - torch.minimum(starts_sec[start:stop, None], starts_sec[None, :])
        )
        overlap = intersection / union.clamp_min(torch.finfo(union.dtype).eps)
        affinity.masked_fill_(overlap > overlap_iou, -torch.inf)
        values, indices = torch.topk(affinity, k=k, dim=1)
        indices = indices.masked_fill(~torch.isfinite(values) | (values < threshold), -1)
        neighbor_rows.append(indices.cpu())
    neighbors = torch.cat(neighbor_rows, dim=0)
    directed = {
        (left, int(right))
        for left, row in enumerate(neighbors.tolist())
        for right in row
        if right >= 0
    }
    edges = [
        (left, right)
        for left, right in directed
        if left < right and (right, left) in directed
    ]

    clusters: list[RecurrenceCluster] = []
    for members in _components(edges, count):
        if len(members) < min_occurrences:
            continue
        member_tensor = torch.tensor(members, dtype=torch.long, device=embeddings.device)
        member_embeddings = normalized[member_tensor]
        mean_by_member = _component_mean_affinity(member_embeddings)
        medoid = members[int(mean_by_member.argmax().item())]
        clusters.append(
            RecurrenceCluster(
                tuple(members),
                medoid,
                float(mean_by_member.mean().item()),
            )
        )
    return sorted(
        clusters,
        key=lambda cluster: (-len(cluster.member_indices), cluster.member_indices),
    )
