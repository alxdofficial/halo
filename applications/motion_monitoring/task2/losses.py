"""Contrastive ruler objective for Task 2 (design doc section 5).

Within each reference set of a batch, every accepted query must sit closer to
the personal prototype than every negative query by a margin. The margin scales
with the declared severity of a physical modification, so a barely modified
execution is not asked to sit as far away as a clearly changed one, and it is
the full margin for another person's execution. A small pull term keeps
accepted queries tight. There is no classifier, no regression and no threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import ROLE_INDEX, EpisodeBatch
from .model import RulerOutput


@dataclass(frozen=True)
class RulerLossConfig:
    margin: float = 0.2
    pull_weight: float = 0.1
    minimum_severity_multiplier: float = 0.25

    def __post_init__(self) -> None:
        if self.margin <= 0:
            raise ValueError("margin must be positive")
        if self.pull_weight < 0:
            raise ValueError("pull weight must be non-negative")
        if not 0 < self.minimum_severity_multiplier <= 1:
            raise ValueError("minimum severity multiplier must be in (0, 1]")


@dataclass(frozen=True)
class RulerLoss:
    total: Tensor
    ranking: Tensor
    pull: Tensor
    ranking_count: int
    positive_count: int
    negative_count: int


def ruler_loss(
    output: RulerOutput, batch: EpisodeBatch, config: RulerLossConfig = RulerLossConfig()
) -> RulerLoss:
    distances = output.distances
    roles = batch.roles
    modified = ROLE_INDEX["modified_query"]
    other = ROLE_INDEX["other_subject_query"]
    accepted = ROLE_INDEX["accepted_query"]
    terms: list[Tensor] = []
    weights: list[Tensor] = []
    for reference_set_id in dict.fromkeys(batch.reference_set_ids):
        indices = [i for i, value in enumerate(batch.reference_set_ids) if value == reference_set_id]
        positives = [i for i in indices if int(roles[i]) == accepted]
        negatives = [i for i in indices if int(roles[i]) in (modified, other)]
        for positive in positives:
            for negative in negatives:
                if int(roles[negative]) == modified:
                    multiplier = batch.severities[negative].clamp(
                        min=config.minimum_severity_multiplier, max=1.0
                    )
                else:
                    multiplier = distances.new_tensor(1.0)
                terms.append(
                    F.relu(distances[positive] - distances[negative] + config.margin * multiplier)
                )
                weights.append(
                    torch.sqrt(batch.sample_weights[positive] * batch.sample_weights[negative])
                )
    if terms:
        stacked = torch.stack(terms)
        weight = torch.stack(weights).to(stacked.dtype)
        ranking = (stacked * weight).sum() / weight.sum()
    else:
        ranking = distances.sum() * 0.0
    positive_mask = batch.positive_mask
    if bool(positive_mask.any()):
        weight = batch.sample_weights[positive_mask].to(distances.dtype)
        pull = (distances[positive_mask] * weight).sum() / weight.sum()
    else:
        pull = distances.sum() * 0.0
    total = ranking + config.pull_weight * pull
    return RulerLoss(
        total=total,
        ranking=ranking,
        pull=pull,
        ranking_count=len(terms),
        positive_count=int(positive_mask.sum()),
        negative_count=int(batch.negative_mask.sum()),
    )
