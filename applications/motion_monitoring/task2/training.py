"""Small reusable training step with Task-2 gradient telemetry."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from .contracts import EpisodeBatch
from .losses import ChangeLossConfig, change_quantification_loss
from .model import ChangeMetricHead
from applications.motion_monitoring.training import optimizer_parameters, parameter_grad_norm


@dataclass(frozen=True)
class StepTelemetry:
    total_loss: float
    classification_loss: float
    ranking_loss: float
    regression_loss: float
    total_grad_norm_preclip: float
    clip_coefficient: float
    parameter_grad_norms: dict[str, float]
    nonfinite_gradients: int
    classification_count: int
    ranking_count: int
    regression_count: int
    accepted_score_mean: float | None
    changed_score_mean: float | None
    score_separation: float | None
    reference_attention_fraction: float
    attention_entropy: float


@torch.no_grad()
def initialize_change_threshold(model: ChangeMetricHead, batch: EpisodeBatch) -> float:
    """Initialize the logistic boundary from labeled training-episode score medians.

    This calibrates only the scalar readout. It does not change the latent metric,
    and the bias remains trainable. Both pair classes must be present in the input.
    """

    model.eval()
    output = model(batch)
    valid = batch.classification_mask
    positive = valid & (batch.classification_targets >= 0.5)
    negative = valid & (batch.classification_targets < 0.5)
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("threshold initialization requires both labeled episode classes")
    threshold = (
        output.change_scores[positive].median()
        + output.change_scores[negative].median()
    ) / 2
    scale = torch.nn.functional.softplus(model.logit_scale_raw)
    model.change_bias.copy_(-scale * threshold)
    return float(threshold)


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
    model: ChangeMetricHead,
    batch: EpisodeBatch,
    optimizer: torch.optim.Optimizer,
    *,
    loss_config: ChangeLossConfig = ChangeLossConfig(),
    grad_clip: float = 5.0,
) -> StepTelemetry:
    if grad_clip <= 0:
        raise ValueError("gradient clip must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss = change_quantification_loss(output, batch, loss_config)
    if not torch.isfinite(loss.total):
        raise FloatingPointError("Task-2 loss is non-finite")
    loss.total.backward()
    parameter_norms, nonfinite = _parameter_grad_norms(model)
    optimized_parameters = optimizer_parameters(optimizer)
    total_preclip = float(parameter_grad_norm(optimized_parameters))
    if nonfinite:
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Task-2 gradients are non-finite")
    torch.nn.utils.clip_grad_norm_(
        optimized_parameters, grad_clip, error_if_nonfinite=True
    )
    optimizer.step()
    with torch.no_grad():
        accepted = batch.classification_mask & (batch.classification_targets < 0.5)
        changed = batch.classification_mask & (batch.classification_targets >= 0.5)
        accepted_mean = output.change_scores[accepted].mean() if bool(accepted.any()) else None
        changed_mean = output.change_scores[changed].mean() if bool(changed.any()) else None
        reference_token_count = (
            batch.reference_execution_mask.shape[1] * model.phase_bins
        )
        reference_fraction = output.evidence_attention[
            ..., :reference_token_count
        ].sum(dim=-1).mean()
        probabilities = output.evidence_attention.clamp_min(1e-12)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean()
    return StepTelemetry(
        total_loss=float(loss.total.detach()),
        classification_loss=float(loss.classification.detach()),
        ranking_loss=float(loss.ranking.detach()),
        regression_loss=float(loss.regression.detach()),
        total_grad_norm_preclip=total_preclip,
        clip_coefficient=min(1.0, grad_clip / max(total_preclip, 1e-12)),
        parameter_grad_norms=parameter_norms,
        nonfinite_gradients=nonfinite,
        classification_count=loss.classification_count,
        ranking_count=loss.ranking_count,
        regression_count=loss.regression_count,
        accepted_score_mean=None if accepted_mean is None else float(accepted_mean),
        changed_score_mean=None if changed_mean is None else float(changed_mean),
        score_separation=(
            None
            if accepted_mean is None or changed_mean is None
            else float(changed_mean - accepted_mean)
        ),
        reference_attention_fraction=float(reference_fraction),
        attention_entropy=float(entropy),
    )
