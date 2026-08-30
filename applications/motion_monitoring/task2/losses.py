"""Masked, unit-controlled objectives for Task-2 metric learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import PairBatch
from .model import ChangeHeadOutput


@dataclass(frozen=True)
class ChangeLossConfig:
    classification_weight: float = 1.0
    regression_weight: float = 1.0
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        if self.classification_weight < 0 or self.regression_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.classification_weight + self.regression_weight <= 0:
            raise ValueError("at least one loss component must be enabled")
        if self.huber_delta <= 0:
            raise ValueError("Huber delta must be positive")


@dataclass(frozen=True)
class ChangeLoss:
    total: Tensor
    classification: Tensor
    regression: Tensor
    classification_count: int
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
    batch: PairBatch,
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
        + config.regression_weight * regression
    )
    return ChangeLoss(
        total=total,
        classification=classification,
        regression=regression,
        classification_count=int(batch.classification_mask.sum().item()),
        regression_count=int(batch.target_mask.sum().item()),
    )
