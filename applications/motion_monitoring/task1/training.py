"""Losses, metrics, and short-step training utilities for Task 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from applications.motion_monitoring.task1.episodes import DetectionBatch
from applications.motion_monitoring.task1.model import (
    AlignmentOutput,
    DifferentiableSubsequenceMatcher,
)
from applications.motion_monitoring.task1.matcher import TemporalMatch
from applications.motion_monitoring.training import optimizer_parameters, parameter_grad_norm


def balanced_endpoint_loss(
    output: AlignmentOutput, batch: DetectionBatch
) -> torch.Tensor:
    """Give present and absent endpoints equal influence without corpus-dependent weights."""

    valid = batch.loss_valid & batch.query_valid & output.endpoint_valid
    positive = valid & batch.endpoint_targets
    negative = valid & ~batch.endpoint_targets
    terms: list[torch.Tensor] = []
    if torch.any(positive):
        terms.append(F.softplus(-output.endpoint_logits[positive]).mean())
    if torch.any(negative):
        terms.append(F.softplus(output.endpoint_logits[negative]).mean())
    if not terms:
        raise ValueError("the batch contains no loss-valid query patches")
    return torch.stack(terms).mean()


def detection_metrics(
    output: AlignmentOutput,
    batch: DetectionBatch,
    *,
    logit_threshold: float = 0.0,
) -> dict[str, float]:
    valid = batch.loss_valid & batch.query_valid & output.endpoint_valid
    truth = batch.endpoint_targets & valid
    predicted = (output.endpoint_logits >= logit_threshold) & valid
    true_positive = int(torch.sum(predicted & truth))
    false_positive = int(torch.sum(predicted & ~truth & valid))
    false_negative = int(torch.sum(~predicted & truth))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    positive_logits = output.endpoint_logits[truth]
    negative_logits = output.endpoint_logits[valid & ~truth]
    absent_flags: list[float] = []
    endpoint_errors: list[float] = []
    for index in range(len(output.endpoint_logits)):
        episode_valid = valid[index]
        if not torch.any(episode_valid):
            continue
        valid_indices = torch.nonzero(episode_valid, as_tuple=False).flatten()
        if not torch.any(batch.target_valid[index]):
            absent_flags.append(
                float(torch.any(predicted[index, valid_indices]).detach().cpu())
            )
            continue
        predicted_index = valid_indices[
            torch.argmax(output.endpoint_logits[index, valid_indices])
        ]
        predicted_end = batch.query_intervals_sec[index, predicted_index, 1]
        target_ends = batch.targets_sec[index, batch.target_valid[index], 1]
        endpoint_errors.append(float(torch.min(torch.abs(target_ends - predicted_end))))

    return {
        "endpoint_precision": precision,
        "endpoint_recall": recall,
        "endpoint_f1": f1,
        "target_absent_false_alarm_rate": (
            sum(absent_flags) / len(absent_flags) if absent_flags else 0.0
        ),
        "mean_endpoint_error_sec": (
            sum(endpoint_errors) / len(endpoint_errors) if endpoint_errors else 0.0
        ),
        "positive_logit_mean": (
            float(positive_logits.mean().detach()) if len(positive_logits) else 0.0
        ),
        "negative_logit_mean": (
            float(negative_logits.mean().detach()) if len(negative_logits) else 0.0
        ),
        "positive_endpoint_fraction": float(truth.sum() / valid.sum()),
    }


def event_detection_metrics(
    matches: list[TemporalMatch],
    targets_sec: torch.Tensor,
    *,
    query_duration_sec: float,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Score localized deployment matches against source-provided event intervals."""

    targets = torch.as_tensor(targets_sec, dtype=torch.float64)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("targets_sec must have shape [events, 2]")
    if query_duration_sec <= 0 or not 0 < iou_threshold <= 1:
        raise ValueError("query duration and IoU threshold must be positive")
    unmatched = set(range(len(targets)))
    paired: list[tuple[TemporalMatch, int]] = []
    for match in sorted(matches, key=lambda item: item.score):
        best_target = None
        best_iou = 0.0
        for target_index in unmatched:
            target_start, target_end = targets[target_index].tolist()
            intersection = max(
                0.0,
                min(match.end_sec, target_end) - max(match.start_sec, target_start),
            )
            union = max(match.end_sec, target_end) - min(match.start_sec, target_start)
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_target = target_index
        if best_target is not None and best_iou >= iou_threshold:
            paired.append((match, best_target))
            unmatched.remove(best_target)

    true_positive = len(paired)
    false_positive = len(matches) - true_positive
    false_negative = len(targets) - true_positive
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    onset_errors = [
        abs(match.start_sec - float(targets[index, 0])) for match, index in paired
    ]
    offset_errors = [
        abs(match.end_sec - float(targets[index, 1])) for match, index in paired
    ]
    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "false_alarms_per_hour": false_positive / (query_duration_sec / 3600.0),
        "count_error": float(abs(len(matches) - len(targets))),
        "mean_onset_error_sec": (
            sum(onset_errors) / len(onset_errors) if onset_errors else 0.0
        ),
        "mean_offset_error_sec": (
            sum(offset_errors) / len(offset_errors) if offset_errors else 0.0
        ),
    }


def _gradient_norm(parameters) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().double().square().sum().cpu()
    return float(torch.sqrt(squared))


def gradient_telemetry(model: nn.Module) -> dict[str, float]:
    parameters = tuple(model.parameters())
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    return {
        "grad_norm_total": _gradient_norm(parameters),
        "grad_norm_projection": _gradient_norm(model.projection.parameters()),
        "grad_norm_score_bias": _gradient_norm((model.score_bias,)),
        "gradient_finite": float(finite),
        "parameters_with_gradient_fraction": len(gradients) / max(len(parameters), 1),
        "parameters_with_nonzero_gradient_fraction": (
            sum(bool(torch.any(gradient != 0)) for gradient in gradients)
            / max(len(parameters), 1)
        ),
    }


@dataclass(frozen=True)
class TrainStepResult:
    loss: float
    telemetry: dict[str, float]
    output: AlignmentOutput


def train_step(
    model: DifferentiableSubsequenceMatcher,
    batch: DetectionBatch,
    optimizer: torch.optim.Optimizer,
    *,
    grad_clip: float = 1.0,
) -> TrainStepResult:
    """Run one observable optimizer step; callers retain control of the outer loop."""

    if grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss = balanced_endpoint_loss(output, batch)
    if not torch.isfinite(loss):
        raise FloatingPointError("Task-1 loss is not finite")
    loss.backward()
    telemetry: dict[str, Any] = detection_metrics(output, batch)
    telemetry.update(gradient_telemetry(model))
    optimized_parameters = optimizer_parameters(optimizer)
    preclip = parameter_grad_norm(optimized_parameters)
    torch.nn.utils.clip_grad_norm_(
        optimized_parameters, grad_clip, error_if_nonfinite=True
    )
    telemetry["grad_norm_preclip"] = float(preclip)
    telemetry["grad_norm_optimizer_total"] = float(preclip)
    telemetry["gradient_clip_coefficient"] = min(
        1.0, grad_clip / max(float(preclip), 1e-12)
    )
    if not torch.isfinite(preclip):
        raise FloatingPointError("Task-1 gradients are not finite")
    optimizer.step()
    detached_output = AlignmentOutput(
        output.endpoint_logits.detach(),
        output.endpoint_valid.detach(),
        output.reference_embeddings.detach(),
        output.query_embeddings.detach(),
    )
    return TrainStepResult(float(loss.detach()), telemetry, detached_output)
