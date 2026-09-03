"""Small reusable training step with Task-2 gradient telemetry."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from .contracts import EpisodeBatch
from .losses import RulerLossConfig, ruler_loss
from .model import ChangeRuler
from applications.motion_monitoring.training import optimizer_parameters, parameter_grad_norm


@dataclass(frozen=True)
class StepTelemetry:
    total_loss: float
    ranking_loss: float
    pull_loss: float
    total_grad_norm_preclip: float
    clip_coefficient: float
    parameter_grad_norms: dict[str, float]
    nonfinite_gradients: int
    ranking_count: int
    positive_count: int
    negative_count: int
    positive_distance_mean: float | None
    negative_distance_mean: float | None
    separation: float | None
    reference_attention_fraction: float
    attention_entropy: float


def _parameter_grad_norms(model: nn.Module) -> tuple[dict[str, float], int]:
    norms: dict[str, float] = {}
    nonfinite = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            norms[name] = 0.0
            continue
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            nonfinite += int((~torch.isfinite(gradient)).sum().item())
        norms[name] = float(torch.linalg.vector_norm(torch.nan_to_num(gradient)))
    return norms, nonfinite


def train_step(
    model: ChangeRuler,
    batch: EpisodeBatch,
    optimizer: torch.optim.Optimizer,
    *,
    loss_config: RulerLossConfig = RulerLossConfig(),
    grad_clip: float = 5.0,
) -> StepTelemetry:
    if grad_clip <= 0:
        raise ValueError("gradient clip must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss = ruler_loss(output, batch, loss_config)
    if not torch.isfinite(loss.total):
        raise FloatingPointError("Task-2 loss is non-finite")
    loss.total.backward()
    parameter_norms, nonfinite = _parameter_grad_norms(model)
    optimized_parameters = optimizer_parameters(optimizer)
    total_preclip = float(parameter_grad_norm(optimized_parameters))
    if nonfinite:
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Task-2 gradients are non-finite")
    torch.nn.utils.clip_grad_norm_(optimized_parameters, grad_clip, error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        positive = batch.positive_mask
        negative = batch.negative_mask
        positive_mean = output.distances[positive].mean() if bool(positive.any()) else None
        negative_mean = output.distances[negative].mean() if bool(negative.any()) else None
        reference_token_count = batch.reference_execution_mask.shape[1] * model.phase_bins
        reference_fraction = output.evidence_attention[..., :reference_token_count].sum(dim=-1).mean()
        probabilities = output.evidence_attention.clamp_min(1e-12)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean()
    return StepTelemetry(
        total_loss=float(loss.total.detach()),
        ranking_loss=float(loss.ranking.detach()),
        pull_loss=float(loss.pull.detach()),
        total_grad_norm_preclip=total_preclip,
        clip_coefficient=min(1.0, grad_clip / max(total_preclip, 1e-12)),
        parameter_grad_norms=parameter_norms,
        nonfinite_gradients=nonfinite,
        ranking_count=loss.ranking_count,
        positive_count=loss.positive_count,
        negative_count=loss.negative_count,
        positive_distance_mean=None if positive_mean is None else float(positive_mean),
        negative_distance_mean=None if negative_mean is None else float(negative_mean),
        separation=(
            None
            if positive_mean is None or negative_mean is None
            else float(negative_mean - positive_mean)
        ),
        reference_attention_fraction=float(reference_fraction),
        attention_entropy=float(entropy),
    )
