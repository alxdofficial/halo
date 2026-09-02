"""Evaluation metrics for Task-2 learned change heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def binary_auroc(scores: Tensor, targets: Tensor, mask: Tensor | None = None) -> float:
    """Exact rank AUROC with average ranks for ties."""

    scores = torch.as_tensor(scores, dtype=torch.float64).flatten()
    targets = torch.as_tensor(targets).bool().flatten()
    if mask is not None:
        valid = torch.as_tensor(mask).bool().flatten()
        scores, targets = scores[valid], targets[valid]
    if len(scores) == 0 or not torch.isfinite(scores).all():
        return float("nan")
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    ranks = torch.arange(
        1, len(scores) + 1, dtype=torch.float64, device=scores.device
    )
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = ranks[start:end].mean()
        start = end
    original_ranks = torch.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = original_ranks[targets].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    )


def binary_operating_metrics(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor | None = None,
    *,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Return decisions at a threshold fixed on development subjects."""

    logits = torch.as_tensor(logits).flatten()
    targets = torch.as_tensor(targets).bool().flatten()
    valid = (
        torch.ones_like(targets)
        if mask is None
        else torch.as_tensor(mask).bool().flatten()
    )
    predictions = logits[valid] >= threshold
    truth = targets[valid]
    positive = truth
    negative = ~truth
    if not bool(positive.any()) or not bool(negative.any()):
        return {
            name: float("nan")
            for name in (
                "sensitivity",
                "specificity",
                "balanced_accuracy",
                "precision",
                "f1",
                "false_positive_rate",
                "accepted_false_alarm_rate",
            )
        }
    sensitivity = (predictions[positive] == truth[positive]).float().mean()
    specificity = (predictions[negative] == truth[negative]).float().mean()
    true_positive = (predictions & truth).sum().float()
    predicted_positive = predictions.sum().float()
    precision = true_positive / predicted_positive.clamp_min(1.0)
    f1 = 2 * precision * sensitivity / (precision + sensitivity).clamp_min(1e-12)
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "precision": float(precision),
        "f1": float(f1),
        "false_positive_rate": float(1.0 - specificity),
        "accepted_false_alarm_rate": float(1.0 - specificity),
    }


def balanced_accuracy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor | None = None,
    *,
    threshold: float = 0.0,
) -> float:
    """Compatibility wrapper for the complete operating-point metrics."""

    return binary_operating_metrics(
        logits, targets, mask, threshold=threshold
    )["balanced_accuracy"]


@dataclass(frozen=True)
class RegressionMetrics:
    mae: dict[str, float]
    rmse: dict[str, float]
    counts: dict[str, int]


def masked_regression_metrics(
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
    target_names: tuple[str, ...],
) -> RegressionMetrics:
    predictions = torch.as_tensor(predictions)
    targets = torch.as_tensor(targets)
    mask = torch.as_tensor(mask).bool()
    if predictions.shape != targets.shape or mask.shape != targets.shape:
        raise ValueError("predictions, targets, and mask must have identical shapes")
    if predictions.ndim != 2 or predictions.shape[1] != len(target_names):
        raise ValueError("target names must match the regression width")
    mae: dict[str, float] = {}
    rmse: dict[str, float] = {}
    counts: dict[str, int] = {}
    for index, name in enumerate(target_names):
        valid = mask[:, index]
        count = int(valid.sum())
        counts[name] = count
        if count == 0:
            mae[name] = float("nan")
            rmse[name] = float("nan")
            continue
        error = predictions[valid, index] - targets[valid, index]
        mae[name] = float(error.abs().mean())
        rmse[name] = float(error.square().mean().sqrt())
    return RegressionMetrics(mae=mae, rmse=rmse, counts=counts)
