"""Personal-normative change ruler for Task-2 execution episodes.

The ruler does not classify change. It learns the coordinate system in which a
person's ordinary repetition scatter is tight and physical change is wide
(docs/tasks/TASK2_CHANGE_QUANTIFICATION.md section 5). Its output is a
phase-local residual between a query and the personal reference prototype; the
scalar distance is the mean cosine residual over movement phase. Scoring against
the personal envelope and the per-person threshold stay closed-form
(``personal.py``); nothing here produces a logit, a threshold or a regressed target.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import EpisodeBatch


@dataclass(frozen=True)
class RulerOutput:
    distances: Tensor
    phase_residuals: Tensor
    residual_vectors: Tensor
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


def _normalize(values: Tensor) -> Tensor:
    eps = 1e-6 if values.dtype in {torch.float16, torch.bfloat16} else 1e-8
    return F.normalize(values, dim=-1, eps=eps)


def _prototype(reference_phase: Tensor, execution_mask: Tensor) -> Tensor:
    weights = execution_mask.to(reference_phase.dtype)[:, :, None, None]
    total = (reference_phase * weights).sum(dim=1)
    return total / weights.sum(dim=1).clamp_min(1.0)


class ChangeRuler(nn.Module):
    """Map an aligned (reference set, query) pair into a comparable residual space.

    References are contextualised among themselves and never see the query, so
    an unusual query cannot redefine normality. The query attends to the
    reference (and optional same-person context) tokens and receives a
    zero-initialised refinement in the metric space, so the ruler starts exactly
    at the transparent cosine floor and only moves where the objective rewards it.
    """

    REFERENCE_ROLE = 0
    CONTEXT_ROLE = 1
    QUERY_ROLE = 2

    def __init__(
        self,
        embedding_dim: int,
        *,
        phase_bins: int = 8,
        projection_dim: int | None = None,
        context_dim: int = 64,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding dimension must be positive")
        if phase_bins < 2:
            raise ValueError("phase_bins must be at least two")
        if context_dim <= 0 or attention_heads <= 0 or context_dim % attention_heads:
            raise ValueError("context_dim must be positive and divisible by attention_heads")
        self.embedding_dim = embedding_dim
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
        self.refinement = nn.Linear(context_dim, metric_dim)
        # Zero-initialised: the ruler begins at the transparent cosine floor.
        nn.init.zeros_(self.refinement.weight)
        nn.init.zeros_(self.refinement.bias)

    def _project(self, values: Tensor) -> Tensor:
        if self.projection is not None:
            return self.projection(values)
        assert self.raw_diagonal is not None
        denominator = F.softplus(torch.zeros((), device=values.device, dtype=values.dtype))
        return values * (F.softplus(self.raw_diagonal) / denominator)

    def _role_phase_tokens(self, values: Tensor, role: int) -> Tensor:
        phase = self.phase_embedding.to(dtype=values.dtype)
        role_ids = torch.full(values.shape[:-1], role, dtype=torch.long, device=values.device)
        return self.context_input(values) + phase + self.role_embedding(role_ids)

    def align(self, batch: EpisodeBatch) -> tuple[Tensor, Tensor]:
        """Projected reference set and query resampled onto movement phase."""

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
        return reference_phase, query_phase

    def _reference_memory(
        self, reference_phase: Tensor, execution_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Contextualise an accepted-reference set without exposing the query."""

        batch_size, reference_count, _, _ = reference_phase.shape
        tokens = self._role_phase_tokens(
            reference_phase, self.REFERENCE_ROLE
        ).reshape(batch_size, reference_count * self.phase_bins, -1)
        token_mask = ~execution_mask.unsqueeze(-1).expand(
            -1, -1, self.phase_bins
        ).reshape(batch_size, -1)
        return (
            self.reference_encoder(tokens, src_key_padding_mask=token_mask),
            token_mask,
        )

    def _refine_query(
        self,
        query_phase: Tensor,
        reference_phase: Tensor,
        reference_mask: Tensor,
        *,
        context_phase: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        memory, memory_mask = self._reference_memory(reference_phase, reference_mask)
        if context_phase is not None:
            if context_mask is None:
                raise ValueError("context mask is required with context phase")
            batch_size, context_count, _, _ = context_phase.shape
            context_tokens = self._role_phase_tokens(
                context_phase, self.CONTEXT_ROLE
            ).reshape(batch_size, context_count * self.phase_bins, -1)
            context_token_mask = ~context_mask.unsqueeze(-1).expand(
                -1, -1, self.phase_bins
            ).reshape(batch_size, -1)
            memory = torch.cat((memory, context_tokens), dim=1)
            memory_mask = torch.cat((memory_mask, context_token_mask), dim=1)

        query_tokens = self._role_phase_tokens(query_phase, self.QUERY_ROLE)
        attended, attention = self.query_attention(
            query_tokens,
            memory,
            memory,
            key_padding_mask=memory_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        contextual = self.query_norm(query_tokens + attended)
        refined = query_phase + self.refinement(contextual).to(query_phase.dtype)
        return refined, attention

    def forward(self, batch: EpisodeBatch) -> RulerOutput:
        if batch.query_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError("batch embedding width does not match the ruler")
        reference_phase, query_phase = self.align(batch)
        context_phase = None
        context_mask = None
        if batch.context_embeddings.shape[1] > 0:
            context_phase = _resample_execution_set(
                self._project(batch.context_embeddings),
                batch.context_intervals_sec,
                batch.context_patch_mask,
                batch.context_execution_mask,
                bins=self.phase_bins,
            )
            context_mask = batch.context_execution_mask
        refined_query, attention = self._refine_query(
            query_phase,
            reference_phase,
            batch.reference_execution_mask,
            context_phase=context_phase,
            context_mask=context_mask,
        )

        prototype = _prototype(reference_phase, batch.reference_execution_mask)
        residual_vectors = _normalize(refined_query) - _normalize(prototype)
        phase_residuals = 1.0 - (_normalize(refined_query) * _normalize(prototype)).sum(dim=-1)
        return RulerOutput(
            distances=phase_residuals.mean(dim=-1),
            phase_residuals=phase_residuals,
            residual_vectors=residual_vectors,
            reference_phase=reference_phase,
            # Scoring must consume the contextualised metric-space query.  Returning
            # the pre-refinement tensor here silently bypassed the attention and
            # refinement path in personal_change_report().
            query_phase=refined_query,
            evidence_attention=attention,
        )

    @torch.no_grad()
    def reference_residuals(self, batch: EpisodeBatch) -> tuple[Tensor, Tensor]:
        """Leave-one-out phase residuals of every reference in the ruler's space.

        These feed the personal envelope at deployment: each accepted reference is
        compared with the prototype of the others, in the same projected space
        the query distance uses, so the person's ordinary scatter is measured with
        the same ruler. Returns ``[batch, reference, bins]`` residuals and the
        reference execution mask.
        """

        reference_phase, _ = self.align(batch)
        mask = batch.reference_execution_mask
        batch_size, count, bins, _ = reference_phase.shape
        residuals = reference_phase.new_zeros(batch_size, count, bins)
        held_out_rows: list[Tensor] = []
        remaining_rows: list[Tensor] = []
        remaining_masks: list[Tensor] = []
        destinations: list[tuple[int, int]] = []
        max_remaining = max(int(row.sum()) - 1 for row in mask)
        if max_remaining < 1:
            return residuals, mask
        for row in range(batch_size):
            valid = torch.nonzero(mask[row], as_tuple=False).flatten()
            for index in valid.tolist():
                others = [item for item in valid.tolist() if item != index]
                if not others:
                    continue
                remaining = reference_phase.new_zeros(
                    max_remaining, bins, reference_phase.shape[-1]
                )
                remaining[: len(others)] = reference_phase[row, others]
                remaining_mask = torch.zeros(
                    max_remaining, dtype=torch.bool, device=reference_phase.device
                )
                remaining_mask[: len(others)] = True
                held_out_rows.append(reference_phase[row, index])
                remaining_rows.append(remaining)
                remaining_masks.append(remaining_mask)
                destinations.append((row, index))
        refined, _ = self._refine_query(
            torch.stack(held_out_rows),
            torch.stack(remaining_rows),
            torch.stack(remaining_masks),
        )
        weights = torch.stack(remaining_masks).to(reference_phase.dtype)[..., None, None]
        prototype = (torch.stack(remaining_rows) * weights).sum(dim=1)
        prototype = prototype / weights.sum(dim=1).clamp_min(1.0)
        values = 1.0 - (_normalize(refined) * _normalize(prototype)).sum(dim=-1)
        for destination, value in zip(destinations, values):
            residuals[destination] = value
        return residuals, mask
