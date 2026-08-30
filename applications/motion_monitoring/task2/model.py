"""Lightweight nuisance-tolerant metric head for Task-2 execution pairs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import PairBatch


@dataclass(frozen=True)
class ChangeHeadOutput:
    change_logits: Tensor
    change_scores: Tensor
    target_predictions: Tensor
    phase_residuals: Tensor
    reference_phase: Tensor
    comparison_phase: Tensor


def _resample_one(values: Tensor, intervals: Tensor, mask: Tensor, bins: int) -> Tensor:
    valid_values = values[mask]
    valid_intervals = intervals[mask]
    if len(valid_values) == 1:
        return valid_values.expand(bins, -1)
    centers = valid_intervals.mean(dim=-1)
    phase = (centers - centers[0]) / (centers[-1] - centers[0]).clamp_min(1e-8)
    target = torch.linspace(0.0, 1.0, bins, device=values.device, dtype=phase.dtype)
    right = torch.searchsorted(phase.contiguous(), target).clamp(1, len(phase) - 1)
    left = right - 1
    denominator = (phase[right] - phase[left]).clamp_min(1e-8)
    weight = ((target - phase[left]) / denominator).to(values.dtype).unsqueeze(-1)
    return valid_values[left] + weight * (valid_values[right] - valid_values[left])


def resample_to_phase(
    values: Tensor, intervals: Tensor, mask: Tensor, *, bins: int
) -> Tensor:
    """Linearly resample valid timestamped patches onto normalized movement phase."""

    if values.ndim != 3 or intervals.shape != (*values.shape[:2], 2):
        raise ValueError(
            "values and intervals must be [batch, patch, feature] and [batch, patch, 2]"
        )
    if mask.shape != values.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean [batch, patch]")
    if bins < 2:
        raise ValueError("at least two phase bins are required")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every execution must have at least one valid patch")
    return torch.stack(
        [
            _resample_one(row, time, valid, bins)
            for row, time, valid in zip(values, intervals, mask)
        ]
    )


class ChangeMetricHead(nn.Module):
    """Learn a mild latent metric while preserving a phase-local residual curve.

    The default projection is a positive diagonal reweighting initialized to identity.
    A small linear projection can be selected explicitly for an ablation. Physical
    targets supervise the readout but are never supplied as model inputs.
    """

    def __init__(
        self,
        embedding_dim: int,
        target_dim: int,
        *,
        phase_bins: int = 8,
        projection_dim: int | None = None,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or target_dim <= 0:
            raise ValueError("embedding and target dimensions must be positive")
        if phase_bins < 2:
            raise ValueError("phase_bins must be at least two")
        self.embedding_dim = embedding_dim
        self.target_dim = target_dim
        self.phase_bins = phase_bins
        self.projection_dim = projection_dim
        if projection_dim is None:
            # softplus(raw) / softplus(0) is exactly one at initialization.
            self.raw_diagonal = nn.Parameter(torch.zeros(embedding_dim))
            self.projection = None
            projected_dim = embedding_dim
        else:
            if projection_dim <= 0:
                raise ValueError("projection_dim must be positive")
            self.raw_diagonal = None
            self.projection = nn.Linear(embedding_dim, projection_dim, bias=False)
            nn.init.orthogonal_(self.projection.weight)
            projected_dim = projection_dim

        self.logit_scale_raw = nn.Parameter(torch.tensor(1.0))
        self.change_bias = nn.Parameter(torch.tensor(0.0))
        self.target_readout = nn.Linear(projected_dim + phase_bins, target_dim)

    def _project(self, values: Tensor) -> Tensor:
        if self.projection is not None:
            return self.projection(values)
        assert self.raw_diagonal is not None
        weights = F.softplus(self.raw_diagonal) / F.softplus(
            torch.zeros((), device=values.device, dtype=values.dtype)
        )
        return values * weights

    def forward(self, batch: PairBatch) -> ChangeHeadOutput:
        if batch.reference_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError("batch embedding width does not match the metric head")
        reference = self._project(batch.reference_embeddings)
        comparison = self._project(batch.comparison_embeddings)
        reference_phase = resample_to_phase(
            reference,
            batch.reference_intervals_sec,
            batch.reference_mask,
            bins=self.phase_bins,
        )
        comparison_phase = resample_to_phase(
            comparison,
            batch.comparison_intervals_sec,
            batch.comparison_mask,
            bins=self.phase_bins,
        )
        normalization_eps = (
            1e-6 if reference_phase.dtype in {torch.float16, torch.bfloat16} else 1e-8
        )
        reference_normalized = F.normalize(
            reference_phase, dim=-1, eps=normalization_eps
        )
        comparison_normalized = F.normalize(
            comparison_phase, dim=-1, eps=normalization_eps
        )
        phase_residuals = 1.0 - (reference_normalized * comparison_normalized).sum(
            dim=-1
        )
        change_scores = phase_residuals.mean(dim=-1)
        logit_scale = F.softplus(self.logit_scale_raw)
        change_logits = logit_scale * change_scores + self.change_bias

        # Regress from normalized latent direction, not arbitrary encoder scale.
        signed_delta = (comparison_normalized - reference_normalized).mean(dim=1)
        readout_features = torch.cat((signed_delta, phase_residuals), dim=-1)
        target_predictions = self.target_readout(readout_features)
        return ChangeHeadOutput(
            change_logits=change_logits,
            change_scores=change_scores,
            target_predictions=target_predictions,
            phase_residuals=phase_residuals,
            reference_phase=reference_phase,
            comparison_phase=comparison_phase,
        )
