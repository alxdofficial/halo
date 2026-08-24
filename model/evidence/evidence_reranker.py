"""All-memory residual reranking for recording-level evidence.

Each query and memory entry is one pooled six-second recording.  Raw cosine similarity remains the
baseline.  This module adds a small bounded correction to every query-memory pair using both signal
representations and both acquisition-description embeddings.  It is fully vectorized and linear in
the number of pairs; memory rows never self-attend to one another.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import AttentionSpec


@dataclass(frozen=True)
class EvidenceRerankerConfig:
    text_dim: int = 384
    n_interaction_heads: int = 8
    interaction_dim: int = 16
    hidden_dim: int = 32
    correction_gain_init: float = 0.01
    max_correction: float = 0.50
    head_init_std: float = 1e-3

    def __post_init__(self) -> None:
        if self.n_interaction_heads < 1 or self.interaction_dim < 1 or self.hidden_dim < 1:
            raise ValueError("reranker dimensions must be positive")
        if not 0.0 < self.correction_gain_init < self.max_correction:
            raise ValueError("correction_gain_init must lie in (0, max_correction)")
        if self.head_init_std <= 0.0:
            raise ValueError("head_init_std must be positive")


class EvidenceReranker(nn.Module):
    """Return a bounded scalar correction for every recording pair."""

    def __init__(self, spec: AttentionSpec, cfg: EvidenceRerankerConfig | None = None):
        super().__init__()
        self.spec = spec
        self.cfg = cfg or EvidenceRerankerConfig()
        d = spec.d_model
        width = self.cfg.n_interaction_heads * self.cfg.interaction_dim
        self.descriptor_projection = nn.Linear(self.cfg.text_dim, d)
        self.query_projection = nn.Linear(2 * d, width)
        self.memory_projection = nn.Linear(2 * d, width)
        # Inputs are signal cosine, acquisition-description cosine, enrollment flag, and learned
        # joint signal/config interactions.  A single mechanism therefore learns statements such as
        # "this similarity is reliable across these two placements" rather than adding an unrelated
        # metadata score after retrieval.
        self.head_in = nn.Linear(self.cfg.n_interaction_heads + 3, self.cfg.hidden_dim)
        self.head_out = nn.Linear(self.cfg.hidden_dim, 1, bias=False)
        nn.init.normal_(self.head_out.weight, std=self.cfg.head_init_std)
        ratio = self.cfg.correction_gain_init / self.cfg.max_correction
        self.correction_gain_logit = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )

    def correction_gain(self) -> torch.Tensor:
        return self.cfg.max_correction * torch.sigmoid(self.correction_gain_logit)

    def _side(self, feature: torch.Tensor, descriptor: torch.Tensor, projection: nn.Linear):
        descriptor = self.descriptor_projection(descriptor)
        joint = torch.cat((feature, descriptor), dim=-1)
        return projection(joint).reshape(
            *feature.shape[:-1], self.cfg.n_interaction_heads, self.cfg.interaction_dim,
        )

    def forward_batched(
        self,
        query_feature: torch.Tensor,       # (E,Q,d)
        query_descriptor: torch.Tensor,    # (E,Q,text_dim)
        memory_feature: torch.Tensor,      # (E,M,d)
        memory_descriptor: torch.Tensor,   # (E,M,text_dim)
        memory_enrolled: torch.Tensor,     # (E,M) bool
    ) -> dict[str, torch.Tensor]:
        if query_feature.dim() != 3 or memory_feature.dim() != 3:
            raise ValueError("batched reranking expects (episode,row,feature) tensors")
        E, Q, d = query_feature.shape
        if memory_feature.shape[0] != E or memory_feature.shape[-1] != d:
            raise ValueError("query and memory feature shapes disagree")
        M = memory_feature.shape[1]
        if query_descriptor.shape[:2] != (E, Q) or memory_descriptor.shape[:2] != (E, M):
            raise ValueError("recording features and acquisition descriptors disagree")
        if memory_enrolled.shape != (E, M):
            raise ValueError("memory enrollment flag has the wrong shape")

        with torch.autocast(device_type=query_feature.device.type, enabled=False):
            q_feature = F.normalize(query_feature.float(), dim=-1)
            m_feature = F.normalize(memory_feature.float(), dim=-1)
            q_descriptor = F.normalize(query_descriptor.float(), dim=-1)
            m_descriptor = F.normalize(memory_descriptor.float(), dim=-1)
            signal_cosine = torch.bmm(q_feature, m_feature.transpose(1, 2))
            descriptor_cosine = torch.bmm(q_descriptor, m_descriptor.transpose(1, 2))

            q_heads = self._side(q_feature, q_descriptor, self.query_projection)
            m_heads = self._side(m_feature, m_descriptor, self.memory_projection)
            # One batched matmul for each episode/head, without materializing concatenated pair rows.
            qh = q_heads.permute(0, 2, 1, 3).reshape(
                E * self.cfg.n_interaction_heads, Q, self.cfg.interaction_dim,
            )
            mh = m_heads.permute(0, 2, 3, 1).reshape(
                E * self.cfg.n_interaction_heads, self.cfg.interaction_dim, M,
            )
            interaction = torch.bmm(qh, mh).reshape(
                E, self.cfg.n_interaction_heads, Q, M,
            ).permute(0, 2, 3, 1) / math.sqrt(self.cfg.interaction_dim)
            enrolled = memory_enrolled[:, None, :, None].expand(E, Q, M, 1).float()
            pair = torch.cat((
                signal_cosine.unsqueeze(-1), descriptor_cosine.unsqueeze(-1),
                enrolled, interaction,
            ), dim=-1)

        compute = (torch.bfloat16 if pair.is_cuda else pair.dtype)
        hidden = F.linear(
            pair.to(compute), self.head_in.weight.to(compute), self.head_in.bias.to(compute),
        )
        raw = F.linear(F.gelu(hidden), self.head_out.weight.to(compute)).squeeze(-1).float()
        correction = self.correction_gain().float() * torch.tanh(raw)
        return {
            "base_score": signal_cosine,
            "descriptor_cosine": descriptor_cosine,
            "raw_correction": raw,
            "score_correction": correction,
            "score": signal_cosine + correction,
        }

    def forward(
        self,
        query_feature: torch.Tensor,
        query_descriptor: torch.Tensor,
        memory_feature: torch.Tensor,
        memory_descriptor: torch.Tensor,
        memory_enrolled: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        result = self.forward_batched(
            query_feature.unsqueeze(0), query_descriptor.unsqueeze(0),
            memory_feature.unsqueeze(0), memory_descriptor.unsqueeze(0),
            memory_enrolled.unsqueeze(0),
        )
        return {name: value[0] for name, value in result.items()}

    @torch.no_grad()
    def telemetry(self) -> dict[str, float]:
        return {
            "reranker/correction_gain": float(self.correction_gain()),
            "reranker/head_norm": float(self.head_out.weight.norm()),
        }
