"""Masked, unit-controlled objectives for Task-2 metric learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import EpisodeBatch
from .model import ChangeHeadOutput


@dataclass(frozen=True)
class ChangeLossConfig:
    classification_weight: float = 1.0
    ranking_weight: float = 0.5
    regression_weight: float = 1.0
    ranking_margin: float = 0.2
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        if min(self.classification_weight, self.ranking_weight, self.regression_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        if self.classification_weight + self.ranking_weight + self.regression_weight <= 0:
            raise ValueError("at least one loss component must be enabled")
        if self.ranking_margin <= 0:
            raise ValueError("ranking margin must be positive")
        if self.huber_delta <= 0:
            raise ValueError("Huber delta must be positive")


@dataclass(frozen=True)
class ChangeLoss:
    total: Tensor
    classification: Tensor
    ranking: Tensor
    regression: Tensor
    classification_count: int
    ranking_count: int
    regression_count: int


def _weighted_masked_mean(
    values: Tensor, mask: Tensor, sample_weights: Tensor
) -> Tensor:
    weights = mask.to(values.dtype) * sample_weights
    denominator = weights.sum()
    if float(denominator.detach()) == 0.0:
        return values.sum() * 0.0
    return (values * weights).sum() / denominator


def change_quantification_loss(
    output: ChangeHeadOutput,
    batch: EpisodeBatch,
    config: ChangeLossConfig = ChangeLossConfig(),
) -> ChangeLoss:
    """Combine change discrimination and scaled interpretable-target regression."""

    classification_per_pair = F.binary_cross_entropy_with_logits(
        output.change_logits,
        batch.classification_targets,
        reduction="none",
    )
    classification = _weighted_masked_mean(
        classification_per_pair,
        batch.classification_mask,
        batch.sample_weights,
    )

    ranking_terms: list[Tensor] = []
    ranking_weights: list[Tensor] = []
    for reference_set_id in dict.fromkeys(batch.reference_set_ids):
        indices = [
            index
            for index, value in enumerate(batch.reference_set_ids)
            if value == reference_set_id and bool(batch.classification_mask[index])
        ]
        accepted = [index for index in indices if batch.classification_targets[index] < 0.5]
        changed = [index for index in indices if batch.classification_targets[index] >= 0.5]
        for accepted_index in accepted:
            for changed_index in changed:
                ranking_terms.append(
                    F.relu(
                        config.ranking_margin
                        - output.change_scores[changed_index]
                        + output.change_scores[accepted_index]
                    )
                )
                ranking_weights.append(
                    torch.sqrt(
                        batch.sample_weights[accepted_index]
                        * batch.sample_weights[changed_index]
                    )
                )
    if ranking_terms:
        stacked_terms = torch.stack(ranking_terms)
        stacked_weights = torch.stack(ranking_weights)
        ranking = (stacked_terms * stacked_weights).sum() / stacked_weights.sum()
    else:
        ranking = output.change_scores.sum() * 0.0

    scaled_error = (
        output.target_predictions - batch.change_targets
    ) / batch.target_scales.unsqueeze(0)
    regression_per_target = F.huber_loss(
        scaled_error,
        torch.zeros_like(scaled_error),
        delta=config.huber_delta,
        reduction="none",
    )
    valid_per_pair = batch.target_mask.sum(dim=1).clamp_min(1)
    regression_per_pair = (
        regression_per_target * batch.target_mask.to(regression_per_target.dtype)
    ).sum(dim=1) / valid_per_pair
    pair_has_target = batch.target_mask.any(dim=1)
    regression = _weighted_masked_mean(
        regression_per_pair,
        pair_has_target,
        batch.sample_weights,
    )

    total = (
        config.classification_weight * classification
        + config.ranking_weight * ranking
        + config.regression_weight * regression
    )
    return ChangeLoss(
        total=total,
        classification=classification,
        ranking=ranking,
        regression=regression,
        classification_count=int(batch.classification_mask.sum().item()),
        ranking_count=len(ranking_terms),
        regression_count=int(batch.target_mask.sum().item()),
    )
