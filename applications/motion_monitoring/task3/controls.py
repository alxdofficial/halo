"""Transparent non-learned controls for recurrent-motion affinity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import CandidateBatch, CandidateTargets
from .losses import scoped_pair_indices
from .metrics import binary_auprc, binary_auroc


@dataclass(frozen=True)
class DirectAffinityOutput:
    positive_scores: Tensor
    negative_scores: Tensor
    auroc: float
    auprc: float


@torch.no_grad()
def direct_cosine_affinity(
    candidates: CandidateBatch,
    targets: CandidateTargets,
    *,
    max_pairs_per_class: int = 4096,
) -> DirectAffinityOutput:
    """Score the same scoped event pairs with raw normalized cosine similarity."""

    positive, negative = scoped_pair_indices(
        candidates, targets, max_pairs_per_class=max_pairs_per_class
    )
    if not len(positive) or not len(negative):
        raise ValueError("direct affinity requires positive and negative pairs")
    flat = F.normalize(
        candidates.embeddings.reshape(-1, candidates.embeddings.shape[-1]),
        dim=-1,
        eps=1e-8,
    )

    def score(indices: Tensor) -> Tensor:
        return (flat[indices[:, 0]] * flat[indices[:, 1]]).sum(dim=-1)

    positive_scores = score(positive)
    negative_scores = score(negative)
    scores = torch.cat((positive_scores, negative_scores))
    labels = torch.cat(
        (
            torch.ones_like(positive_scores, dtype=torch.bool),
            torch.zeros_like(negative_scores, dtype=torch.bool),
        )
    )
    return DirectAffinityOutput(
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        auroc=binary_auroc(scores, labels),
        auprc=binary_auprc(scores, labels),
    )
