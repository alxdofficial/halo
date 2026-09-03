"""Non-learned and adversarial controls for Task 3.

``direct_cosine_affinity`` is the untrained floor. ``shuffled_identity_targets``
is the mandatory leak control for the assembled corpus: relabel every candidate
with a permuted identity and refit. Splice seams, not motion, would still let a
model separate inserted regions from background, so if the shuffled arm scores
above chance the synthesis recipe is rejected (TASK3 doc section 10.3).
"""

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


def shuffled_identity_targets(
    targets: "CandidateTargets", *, seed: int
) -> "CandidateTargets":
    """Permute label ids within each scope, keeping every mask and instance intact.

    A model that still separates positives from negatives under this relabelling
    is reading something other than motion identity -- for an assembled corpus,
    almost certainly the splice seam.
    """

    import torch

    from .contracts import CandidateTargets

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    labels = targets.label_id.clone()
    assigned = targets.assigned_mask
    for scope in torch.unique(targets.scope_id[assigned]):
        selected = assigned & (targets.scope_id == scope)
        values = labels[selected]
        unique = torch.unique(values)
        if len(unique) < 2:
            continue
        permuted = unique[torch.randperm(len(unique), generator=generator)]
        mapping = {int(a): int(b) for a, b in zip(unique.tolist(), permuted.tolist())}
        labels[selected] = torch.tensor(
            [mapping[int(v)] for v in values.tolist()],
            dtype=labels.dtype,
            device=labels.device,
        )
    return CandidateTargets(
        label_id=labels,
        instance_id=targets.instance_id,
        scope_id=targets.scope_id,
        assigned_mask=targets.assigned_mask,
        background_mask=targets.background_mask,
        best_iou=targets.best_iou,
    )
