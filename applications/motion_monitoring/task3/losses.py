"""Scoped same-motion pair construction and balanced metric loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .contracts import CandidateBatch, CandidateTargets
from .model import RecurrentMotionMetric


@dataclass(frozen=True)
class PairLossOutput:
    loss: torch.Tensor
    positive_loss: torch.Tensor
    negative_loss: torch.Tensor
    positive_logits: torch.Tensor
    negative_logits: torch.Tensor
    positive_pair_indices: torch.Tensor
    negative_pair_indices: torch.Tensor
    projected_embeddings: torch.Tensor


def scoped_pair_masks(
    candidates: CandidateBatch,
    targets: CandidateTargets,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct equivalence pairs without comparing identities across scopes."""

    if targets.label_id.shape != candidates.candidate_mask.shape:
        raise ValueError("targets must align with the candidate batch")
    flat_valid = (candidates.candidate_mask & targets.assigned_mask).reshape(-1)
    label = targets.label_id.reshape(-1)
    instance = targets.instance_id.reshape(-1)
    scope = targets.scope_id.reshape(-1)
    size = len(label)
    upper = torch.triu(
        torch.ones((size, size), dtype=torch.bool, device=label.device), diagonal=1
    )
    valid = flat_valid[:, None] & flat_valid[None, :] & upper
    same_scope = scope[:, None] == scope[None, :]
    same_label = label[:, None] == label[None, :]
    different_instance = instance[:, None] != instance[None, :]
    positive = valid & same_scope & same_label & different_instance
    negative = valid & same_scope & ~same_label
    return positive | negative, positive, negative


def _round_robin_pairs(
    sources: list[tuple[list[int], list[int]]], max_pairs: int
) -> list[tuple[int, int]]:
    """Take pairs fairly across identity combinations without materializing products."""

    pairs: list[tuple[int, int]] = []
    offsets = [0] * len(sources)
    active = list(range(len(sources)))
    while active and len(pairs) < max_pairs:
        next_active: list[int] = []
        for source_index in active:
            left, right = sources[source_index]
            offset = offsets[source_index]
            total = len(left) * len(right)
            if offset >= total:
                continue
            pairs.append((left[offset % len(left)], right[offset // len(left)]))
            offsets[source_index] += 1
            if offsets[source_index] < total:
                next_active.append(source_index)
            if len(pairs) >= max_pairs:
                break
        active = next_active
    return pairs


def scoped_pair_indices(
    candidates: CandidateBatch,
    targets: CandidateTargets,
    *,
    max_pairs_per_class: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build bounded positive and negative pair lists from scoped identities."""

    if max_pairs_per_class <= 0:
        raise ValueError("max_pairs_per_class must be positive")
    if targets.label_id.shape != candidates.candidate_mask.shape:
        raise ValueError("targets must align with the candidate batch")
    valid_indices = (
        (candidates.candidate_mask & targets.assigned_mask)
        .reshape(-1)
        .nonzero(as_tuple=False)
        .flatten()
    )
    label = targets.label_id.reshape(-1)
    instance = targets.instance_id.reshape(-1)
    scope = targets.scope_id.reshape(-1)

    groups: dict[tuple[int, int], dict[int, list[int]]] = {}
    for flat_index in valid_indices.tolist():
        key = (int(scope[flat_index]), int(label[flat_index]))
        groups.setdefault(key, {}).setdefault(int(instance[flat_index]), []).append(
            flat_index
        )

    positive_sources: list[tuple[list[int], list[int]]] = []
    labels_by_scope: dict[int, dict[int, list[int]]] = {}
    for (scope_id, label_id), instances in sorted(groups.items()):
        instance_groups = list(instances.values())
        for left_index in range(len(instance_groups)):
            for right_index in range(left_index + 1, len(instance_groups)):
                positive_sources.append(
                    (instance_groups[left_index], instance_groups[right_index])
                )
        labels_by_scope.setdefault(scope_id, {})[label_id] = [
            index for members in instance_groups for index in members
        ]

    negative_sources: list[tuple[list[int], list[int]]] = []
    for scoped_labels in labels_by_scope.values():
        label_groups = list(scoped_labels.values())
        for left_index in range(len(label_groups)):
            for right_index in range(left_index + 1, len(label_groups)):
                negative_sources.append(
                    (label_groups[left_index], label_groups[right_index])
                )

    device = candidates.embeddings.device
    positive = torch.tensor(
        _round_robin_pairs(positive_sources, max_pairs_per_class),
        dtype=torch.long,
        device=device,
    ).reshape(-1, 2)
    negative = torch.tensor(
        _round_robin_pairs(negative_sources, max_pairs_per_class),
        dtype=torch.long,
        device=device,
    ).reshape(-1, 2)
    return positive, negative


def scoped_pair_loss(
    model: RecurrentMotionMetric,
    candidates: CandidateBatch,
    targets: CandidateTargets,
    *,
    max_pairs_per_class: int = 4096,
) -> PairLossOutput:
    """Balanced BCE over valid within-scope positive and negative pairs."""

    positive, negative = scoped_pair_indices(
        candidates, targets, max_pairs_per_class=max_pairs_per_class
    )
    if not len(positive):
        raise ValueError("batch contains no cross-instance positive pairs")
    if not len(negative):
        raise ValueError("batch contains no explicit within-scope negative pairs")

    flat_embeddings = candidates.embeddings.reshape(-1, candidates.embeddings.shape[-1])
    all_pair_indices = torch.cat([positive, negative], dim=0)
    unique_indices, inverse = torch.unique(
        all_pair_indices, sorted=True, return_inverse=True
    )
    projected = model.embed(flat_embeddings[unique_indices])
    projected_pairs = projected[inverse].reshape(len(all_pair_indices), 2, -1)
    logits = model.logits_from_projected(projected_pairs[:, 0], projected_pairs[:, 1])
    positive_logits = logits[: len(positive)]
    negative_logits = logits[len(positive) :]
    positive_loss = F.binary_cross_entropy_with_logits(
        positive_logits, torch.ones_like(positive_logits)
    )
    negative_loss = F.binary_cross_entropy_with_logits(
        negative_logits, torch.zeros_like(negative_logits)
    )
    return PairLossOutput(
        loss=0.5 * (positive_loss + negative_loss),
        positive_loss=positive_loss,
        negative_loss=negative_loss,
        positive_logits=positive_logits,
        negative_logits=negative_logits,
        positive_pair_indices=positive,
        negative_pair_indices=negative,
        projected_embeddings=projected,
    )
