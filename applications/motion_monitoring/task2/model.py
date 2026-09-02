"""Set-conditioned change head for Task-2 execution episodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import EpisodeBatch


@dataclass(frozen=True)
class ChangeHeadOutput:
    change_logits: Tensor
    change_scores: Tensor
    target_predictions: Tensor
    phase_residuals: Tensor
    reference_phase: Tensor
    query_phase: Tensor
    evidence_attention: Tensor


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
        [_resample_one(row, time, valid, bins) for row, time, valid in zip(values, intervals, mask)]
    )


def _resample_execution_set(
    values: Tensor,
    intervals: Tensor,
    patch_mask: Tensor,
    execution_mask: Tensor,
    *,
    bins: int,
) -> Tensor:
    """Resample a padded [batch, execution, patch, feature] set."""

    batch_size, execution_count, _, width = values.shape
    rows: list[Tensor] = []
    zero = values.new_zeros(bins, width)
    for batch_index in range(batch_size):
        executions: list[Tensor] = []
        for execution_index in range(execution_count):
            if bool(execution_mask[batch_index, execution_index]):
                executions.append(
                    _resample_one(
                        values[batch_index, execution_index],
                        intervals[batch_index, execution_index],
                        patch_mask[batch_index, execution_index],
                        bins,
                    )
                )
            else:
                executions.append(zero)
        rows.append(torch.stack(executions))
    return torch.stack(rows)


class ChangeMetricHead(nn.Module):
    """Score a query relative to a personal accepted-reference set.

    Reference executions form the baseline without seeing the query. The query then
    cross-attends to those references and optional same-person context. Role and
    physical-phase embeddings tell the head which tokens serve which purpose.
    """

    REFERENCE_ROLE = 0
    CONTEXT_ROLE = 1
    QUERY_ROLE = 2

    def __init__(
        self,
        embedding_dim: int,
        target_dim: int,
        *,
        phase_bins: int = 8,
        projection_dim: int | None = None,
        context_dim: int = 64,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or target_dim <= 0:
            raise ValueError("embedding and target dimensions must be positive")
        if phase_bins < 2:
            raise ValueError("phase_bins must be at least two")
        if context_dim <= 0 or attention_heads <= 0 or context_dim % attention_heads:
            raise ValueError("context_dim must be positive and divisible by attention_heads")
        self.embedding_dim = embedding_dim
        self.target_dim = target_dim
        self.phase_bins = phase_bins
        self.projection_dim = projection_dim
        if projection_dim is None:
            self.raw_diagonal = nn.Parameter(torch.zeros(embedding_dim))
            self.projection = None
            metric_dim = embedding_dim
        else:
            if projection_dim <= 0:
                raise ValueError("projection_dim must be positive")
            self.raw_diagonal = None
            self.projection = nn.Linear(embedding_dim, projection_dim, bias=False)
            nn.init.orthogonal_(self.projection.weight)
            metric_dim = projection_dim
        self.metric_dim = metric_dim

        self.context_input = nn.Linear(metric_dim, context_dim)
        self.role_embedding = nn.Embedding(3, context_dim)
        self.phase_embedding = nn.Parameter(torch.empty(phase_bins, context_dim))
        nn.init.normal_(self.phase_embedding, std=0.02)
        reference_layer = nn.TransformerEncoderLayer(
            d_model=context_dim,
            nhead=attention_heads,
            dim_feedforward=2 * context_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.reference_encoder = nn.TransformerEncoder(
            reference_layer, num_layers=1, enable_nested_tensor=False
        )
        self.query_attention = nn.MultiheadAttention(
            context_dim, attention_heads, dropout=0.0, batch_first=True
        )
        self.query_norm = nn.LayerNorm(context_dim)
        self.residual_correction = nn.Sequential(
            nn.Linear(4 * context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, 1),
        )
        # Start close to the transparent cosine baseline while keeping gradient
        # paths into every contextualization component alive from step one.
        nn.init.normal_(self.residual_correction[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual_correction[-1].bias)

        self.logit_scale_raw = nn.Parameter(torch.tensor(1.0))
        self.change_bias = nn.Parameter(torch.tensor(0.0))
        self.target_readout = nn.Linear(metric_dim + phase_bins + context_dim, target_dim)

    def _project(self, values: Tensor) -> Tensor:
        if self.projection is not None:
            return self.projection(values)
        assert self.raw_diagonal is not None
        denominator = F.softplus(torch.zeros((), device=values.device, dtype=values.dtype))
        return values * (F.softplus(self.raw_diagonal) / denominator)

    def _role_phase_tokens(self, values: Tensor, role: int) -> Tensor:
        phase = self.phase_embedding.to(dtype=values.dtype)
        role_ids = torch.full(
            values.shape[:-1], role, dtype=torch.long, device=values.device
        )
        return self.context_input(values) + phase + self.role_embedding(role_ids)

    def forward(self, batch: EpisodeBatch) -> ChangeHeadOutput:
        if batch.query_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError("batch embedding width does not match the metric head")

        references = self._project(batch.reference_embeddings)
        query = self._project(batch.query_embeddings)
        reference_phase = _resample_execution_set(
            references,
            batch.reference_intervals_sec,
            batch.reference_patch_mask,
            batch.reference_execution_mask,
            bins=self.phase_bins,
        )
        query_phase = resample_to_phase(
            query, batch.query_intervals_sec, batch.query_mask, bins=self.phase_bins
        )
        batch_size, reference_count, _, _ = reference_phase.shape
        reference_tokens = self._role_phase_tokens(
            reference_phase, self.REFERENCE_ROLE
        ).reshape(batch_size, reference_count * self.phase_bins, -1)
        reference_token_mask = ~batch.reference_execution_mask.unsqueeze(-1).expand(
            -1, -1, self.phase_bins
        ).reshape(batch_size, -1)
        reference_tokens = self.reference_encoder(
            reference_tokens, src_key_padding_mask=reference_token_mask
        )

        memory = reference_tokens
        memory_mask = reference_token_mask
        if batch.context_embeddings.shape[1] > 0:
            context_phase = _resample_execution_set(
                self._project(batch.context_embeddings),
                batch.context_intervals_sec,
                batch.context_patch_mask,
                batch.context_execution_mask,
                bins=self.phase_bins,
            )
            context_count = context_phase.shape[1]
            context_tokens = self._role_phase_tokens(
                context_phase, self.CONTEXT_ROLE
            ).reshape(batch_size, context_count * self.phase_bins, -1)
            context_mask = ~batch.context_execution_mask.unsqueeze(-1).expand(
                -1, -1, self.phase_bins
            ).reshape(batch_size, -1)
            memory = torch.cat((memory, context_tokens), dim=1)
            memory_mask = torch.cat((memory_mask, context_mask), dim=1)

        query_tokens = self._role_phase_tokens(query_phase, self.QUERY_ROLE)
        attended, attention = self.query_attention(
            query_tokens,
            memory,
            memory,
            key_padding_mask=memory_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        contextual_query = self.query_norm(query_tokens + attended)

        reference_weights = batch.reference_execution_mask.to(reference_phase.dtype)
        reference_weights = reference_weights[:, :, None, None]
        reference_prototype = (reference_phase * reference_weights).sum(dim=1)
        reference_prototype = reference_prototype / reference_weights.sum(dim=1).clamp_min(1.0)
        eps = 1e-6 if query_phase.dtype in {torch.float16, torch.bfloat16} else 1e-8
        query_normalized = F.normalize(query_phase, dim=-1, eps=eps)
        reference_normalized = F.normalize(reference_prototype, dim=-1, eps=eps)
        cosine_residual = 1.0 - (query_normalized * reference_normalized).sum(dim=-1)

        reference_context = attended
        correction_input = torch.cat(
            (
                contextual_query,
                reference_context,
                contextual_query - reference_context,
                (contextual_query - reference_context).abs(),
            ),
            dim=-1,
        )
        correction = 0.25 * torch.tanh(self.residual_correction(correction_input).squeeze(-1))
        phase_residuals = cosine_residual + correction
        change_scores = phase_residuals.mean(dim=-1)
        change_logits = F.softplus(self.logit_scale_raw) * change_scores + self.change_bias

        signed_delta = (query_normalized - reference_normalized).mean(dim=1)
        readout_features = torch.cat(
            (signed_delta, phase_residuals, contextual_query.mean(dim=1)), dim=-1
        )
        target_predictions = self.target_readout(readout_features)
        return ChangeHeadOutput(
            change_logits=change_logits,
            change_scores=change_scores,
            target_predictions=target_predictions,
            phase_residuals=phase_residuals,
            reference_phase=reference_phase,
            query_phase=query_phase,
            evidence_attention=attention,
        )
