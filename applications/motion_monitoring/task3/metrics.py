"""Mask-aware pair, representation, and clustering diagnostics."""

from __future__ import annotations

import math

import torch


def _binary_inputs(
    scores: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = scores.reshape(-1)
    targets = targets.reshape(-1)
    if scores.shape != targets.shape:
        raise ValueError("scores and targets must have matching shapes")
    if mask is not None:
        mask = mask.reshape(-1)
        if mask.shape != scores.shape or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean and align with scores")
        scores, targets = scores[mask], targets[mask]
    targets = targets.to(torch.bool)
    if not bool(targets.any()) or not bool((~targets).any()):
        raise ValueError("binary metrics require positive and negative examples")
    return scores.detach().float(), targets


def binary_auroc(
    scores: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    """Tie-aware AUROC computed from pairwise score ordering."""

    scores, targets = _binary_inputs(scores, targets, mask)
    positive = scores[targets]
    negative = scores[~targets]
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean())


def binary_auprc(
    scores: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    """Average precision with all tied scores evaluated at one threshold."""

    scores, targets = _binary_inputs(scores, targets, mask)
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_targets = targets[order].float()
    true_positive = sorted_targets.cumsum(0)
    threshold_end = torch.ones(
        len(sorted_scores), dtype=torch.bool, device=scores.device
    )
    threshold_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    end_indices = threshold_end.nonzero(as_tuple=False).flatten()
    precision = true_positive[end_indices] / (end_indices + 1)
    recall = true_positive[end_indices] / sorted_targets.sum()
    recall_increment = torch.cat([recall[:1], recall[1:] - recall[:-1]])
    return float((precision * recall_increment).sum())


def interval_discovery_metrics(
    predicted_intervals: torch.Tensor,
    target_intervals: torch.Tensor,
    *,
    recording_duration_sec: float,
    match_iou: float = 0.5,
) -> dict[str, float]:
    """Greedy one-to-one occurrence coverage and false motifs per hour."""

    if predicted_intervals.ndim != 2 or predicted_intervals.shape[1:] != (2,):
        raise ValueError("predicted_intervals must have shape [prediction, 2]")
    if target_intervals.ndim != 2 or target_intervals.shape[1:] != (2,):
        raise ValueError("target_intervals must have shape [target, 2]")
    if recording_duration_sec <= 0 or not 0 < match_iou <= 1:
        raise ValueError("recording duration and match_iou must be positive")
    if len(predicted_intervals) and not torch.all(
        predicted_intervals[:, 1] > predicted_intervals[:, 0]
    ):
        raise ValueError("predicted intervals must have positive duration")
    if len(target_intervals) and not torch.all(
        target_intervals[:, 1] > target_intervals[:, 0]
    ):
        raise ValueError("target intervals must have positive duration")

    if len(predicted_intervals) and len(target_intervals):
        intersection = (
            torch.minimum(predicted_intervals[:, None, 1], target_intervals[None, :, 1])
            - torch.maximum(
                predicted_intervals[:, None, 0], target_intervals[None, :, 0]
            )
        ).clamp_min(0)
        union = torch.maximum(
            predicted_intervals[:, None, 1], target_intervals[None, :, 1]
        ) - torch.minimum(predicted_intervals[:, None, 0], target_intervals[None, :, 0])
        iou = intersection / union.clamp_min(torch.finfo(predicted_intervals.dtype).eps)
    else:
        iou = predicted_intervals.new_zeros(
            (len(predicted_intervals), len(target_intervals))
        )

    matched_prediction: set[int] = set()
    matched_target: set[int] = set()
    matched_ious: list[float] = []
    if iou.numel():
        for flat_index in torch.argsort(iou.reshape(-1), descending=True).tolist():
            prediction_index = flat_index // len(target_intervals)
            target_index = flat_index % len(target_intervals)
            value = float(iou[prediction_index, target_index])
            if value < match_iou:
                break
            if prediction_index in matched_prediction or target_index in matched_target:
                continue
            matched_prediction.add(prediction_index)
            matched_target.add(target_index)
            matched_ious.append(value)

    true_positive = len(matched_target)
    false_positive = len(predicted_intervals) - true_positive
    precision = (
        true_positive / len(predicted_intervals) if len(predicted_intervals) else 0.0
    )
    recall = true_positive / len(target_intervals) if len(target_intervals) else 0.0
    return {
        "discovery/occurrence_precision": precision,
        "discovery/occurrence_recall": recall,
        "discovery/count_absolute_error": float(
            abs(len(predicted_intervals) - len(target_intervals))
        ),
        "discovery/false_occurrences_per_hour": false_positive
        / (recording_duration_sec / 3600.0),
        "discovery/matched_mean_iou": (
            sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
        ),
    }


def effective_rank(embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Entropy effective rank after centering valid embeddings."""

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [item, feature]")
    if mask is not None:
        if mask.shape != (len(embeddings),) or mask.dtype != torch.bool:
            raise ValueError("mask must align with embedding rows")
        embeddings = embeddings[mask]
    if len(embeddings) < 2:
        return 0.0
    centered = embeddings.float() - embeddings.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    variance = singular.square()
    total = variance.sum()
    if float(total) <= torch.finfo(variance.dtype).eps:
        return 0.0
    probabilities = variance / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


def pair_score_metrics(
    positive_logits: torch.Tensor, negative_logits: torch.Tensor
) -> dict[str, float]:
    if positive_logits.numel() == 0 or negative_logits.numel() == 0:
        raise ValueError("pair score metrics require both pair classes")
    scores = torch.cat([positive_logits, negative_logits]).detach()
    targets = torch.cat(
        [
            torch.ones_like(positive_logits, dtype=torch.bool),
            torch.zeros_like(negative_logits, dtype=torch.bool),
        ]
    )
    positive_prob = positive_logits.detach().sigmoid()
    negative_prob = negative_logits.detach().sigmoid()
    return {
        "pair/auroc": binary_auroc(scores, targets),
        "pair/auprc": binary_auprc(scores, targets),
        "pair/positive_probability": float(positive_prob.mean()),
        "pair/negative_probability": float(negative_prob.mean()),
        "pair/probability_separation": float(
            positive_prob.mean() - negative_prob.mean()
        ),
    }


def clustering_metrics(
    predicted: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    """B-cubed and pairwise scores for assigned items.

    Cluster identifiers are treated as arbitrary partitions. Items with a
    negative predicted or target identifier are excluded.
    """

    predicted, target = predicted.reshape(-1), target.reshape(-1)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target cluster vectors must align")
    valid = (predicted >= 0) & (target >= 0)
    predicted, target = predicted[valid], target[valid]
    if len(predicted) == 0:
        raise ValueError("clustering metrics require assigned items")
    same_pred = predicted[:, None] == predicted[None, :]
    same_target = target[:, None] == target[None, :]
    intersection = same_pred & same_target
    precision_per_item = intersection.sum(1).float() / same_pred.sum(1).clamp_min(1)
    recall_per_item = intersection.sum(1).float() / same_target.sum(1).clamp_min(1)
    bc_precision = float(precision_per_item.mean())
    bc_recall = float(recall_per_item.mean())
    bc_f1 = 2 * bc_precision * bc_recall / max(1e-12, bc_precision + bc_recall)

    upper = torch.triu(torch.ones_like(same_pred, dtype=torch.bool), diagonal=1)
    true_positive = int((same_pred & same_target & upper).sum())
    predicted_positive = int((same_pred & upper).sum())
    target_positive = int((same_target & upper).sum())
    pair_precision = true_positive / predicted_positive if predicted_positive else 0.0
    pair_recall = true_positive / target_positive if target_positive else 0.0
    pair_f1 = (
        2 * pair_precision * pair_recall / (pair_precision + pair_recall)
        if pair_precision + pair_recall
        else 0.0
    )
    return {
        "cluster/bcubed_precision": bc_precision,
        "cluster/bcubed_recall": bc_recall,
        "cluster/bcubed_f1": bc_f1,
        "cluster/pair_precision": pair_precision,
        "cluster/pair_recall": pair_recall,
        "cluster/pair_f1": pair_f1,
    }


def finite_metrics(metrics: dict[str, float]) -> bool:
    return all(math.isfinite(value) for value in metrics.values())
