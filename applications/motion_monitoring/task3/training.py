"""Minimal training step and low-cost health telemetry for Task 3."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .contracts import CandidateBatch, CandidateTargets
from .losses import PairLossOutput, scoped_pair_indices, scoped_pair_loss
from .metrics import effective_rank, pair_score_metrics
from .model import RecurrentMotionMetric
from applications.motion_monitoring.training import optimizer_parameters, parameter_grad_norm


@dataclass(frozen=True)
class TrainStepOutput:
    loss: float
    telemetry: dict[str, float]


@torch.no_grad()
def initialize_affinity_threshold(
    model: RecurrentMotionMetric,
    candidates: CandidateBatch,
    targets: CandidateTargets,
) -> float:
    """Center the initial decision boundary between training positive/negative medians."""

    positive, negative = scoped_pair_indices(candidates, targets)
    if not len(positive) or not len(negative):
        raise ValueError("affinity initialization requires both positive and negative pairs")
    flat = candidates.embeddings.reshape(-1, candidates.embeddings.shape[-1])
    projected = model.embed(flat)
    positive_cosine = (
        projected[positive[:, 0]] * projected[positive[:, 1]]
    ).sum(dim=-1)
    negative_cosine = (
        projected[negative[:, 0]] * projected[negative[:, 1]]
    ).sum(dim=-1)
    threshold = 0.5 * (positive_cosine.median() + negative_cosine.median())
    scale = torch.nn.functional.softplus(model.logit_scale) + 1e-4
    model.logit_bias.copy_(-scale * threshold)
    return float(threshold)


def gradient_telemetry(module: nn.Module) -> dict[str, float]:
    squared_total = 0.0
    parameter_count = 0
    finite = True
    nonzero_tensors = 0
    for parameter in module.parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(gradient).all())
        norm = float(gradient.float().norm())
        squared_total += norm * norm
        parameter_count += gradient.numel()
        nonzero_tensors += int(norm > 0)
    return {
        "gradient/global_norm": squared_total**0.5,
        "gradient/parameter_count": float(parameter_count),
        "gradient/nonzero_tensor_count": float(nonzero_tensors),
        "gradient/all_finite": float(finite),
    }


def train_step(
    model: RecurrentMotionMetric,
    optimizer: torch.optim.Optimizer,
    candidates: CandidateBatch,
    targets: CandidateTargets,
    *,
    max_grad_norm: float | None = 5.0,
) -> TrainStepOutput:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output: PairLossOutput = scoped_pair_loss(model, candidates, targets)
    output.loss.backward()
    telemetry = gradient_telemetry(model)
    optimized_parameters = optimizer_parameters(optimizer)
    preclip_norm = float(parameter_grad_norm(optimized_parameters))
    if max_grad_norm is not None:
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when provided")
        torch.nn.utils.clip_grad_norm_(
            optimized_parameters, max_grad_norm, error_if_nonfinite=True
        )
    optimizer.step()

    with torch.no_grad():
        telemetry.update(
            pair_score_metrics(output.positive_logits, output.negative_logits)
        )
        telemetry.update(
            {
                "loss/total": float(output.loss),
                "loss/positive": float(output.positive_loss),
                "loss/negative": float(output.negative_loss),
                "pairs/positive": float(output.positive_logits.numel()),
                "pairs/negative": float(output.negative_logits.numel()),
                "representation/effective_rank": effective_rank(
                    output.projected_embeddings
                ),
                "gradient/preclip_norm": preclip_norm,
                "gradient/clip_coefficient": (
                    min(1.0, max_grad_norm / max(preclip_norm, 1e-12))
                    if max_grad_norm is not None
                    else 1.0
                ),
            }
        )
    return TrainStepOutput(float(output.loss.detach()), telemetry)
