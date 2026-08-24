"""Contextual scalar reranking for recording-level evidence.

Each query and memory entry is one pooled six-second recording.  Retrieval first selects a cosine
shortlist.  Candidate labels, the query, and all shortlisted evidence rows then share one unordered
attention set.  The only learned output is one bounded scalar correction per evidence row: this
module cannot refine signal/text vectors, emit candidate logits, or vote for labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import AttentionSpec, ScaledSum, SetAttentionStack


ROLE_CANDIDATE, ROLE_QUERY, ROLE_EVIDENCE = range(3)
N_ROLES = 3


@dataclass(frozen=True)
class EvidenceRerankerConfig:
    text_dim: int = 384
    n_layers: int = 1
    identity_gain_init: float = 0.25
    correction_gain_init: float = 0.05
    max_correction: float = 0.50
    head_init_std: float = 1e-3

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("the contextual reranker needs at least one attention layer")
        if self.identity_gain_init <= 0.0:
            raise ValueError("identity_gain_init must be positive")
        if not 0.0 < self.correction_gain_init < self.max_correction:
            raise ValueError("correction_gain_init must lie in (0, max_correction)")
        if self.head_init_std <= 0.0:
            raise ValueError("head_init_std must be positive")


class EvidenceReranker(nn.Module):
    """Contextualize a cosine shortlist and return one scalar correction per selected row."""

    def __init__(self, spec: AttentionSpec, cfg: EvidenceRerankerConfig | None = None):
        super().__init__()
        self.spec = spec
        self.cfg = cfg or EvidenceRerankerConfig()
        d = spec.d_model
        self.proj_text = nn.Linear(self.cfg.text_dim, d)
        self.proj_signal = nn.Linear(d, d)
        self.proj_score = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.role_emb = nn.Embedding(N_ROLES, d)
        self.enrollment_emb = nn.Embedding(2, d)

        self.candidate_compose = ScaledSum(2, init=[1.0, self.cfg.identity_gain_init])
        self.query_compose = ScaledSum(3, init=[1.0, 1.0, self.cfg.identity_gain_init])
        self.evidence_compose = ScaledSum(
            6,
            init=[1.0, 1.0, 1.0, 1.0,
                  self.cfg.identity_gain_init, self.cfg.identity_gain_init],
        )
        self.stack = SetAttentionStack(spec, self.cfg.n_layers)
        self.row_head = nn.Linear(d, 1, bias=False)
        # A nonzero head is essential: a zero head would block the first-step gradient from every
        # projection and attention parameter below it.
        nn.init.normal_(self.row_head.weight, std=self.cfg.head_init_std)

        ratio = self.cfg.correction_gain_init / self.cfg.max_correction
        self.correction_gain_logit = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )

    def correction_gain(self) -> torch.Tensor:
        return self.cfg.max_correction * torch.sigmoid(self.correction_gain_logit)

    @staticmethod
    def _role(shape: tuple[int, ...], role: int, device: torch.device) -> torch.Tensor:
        return torch.full(shape, role, dtype=torch.long, device=device)

    @staticmethod
    def _gather_rows(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        """Gather (E,M,...) rows into (E,Q,K,...) without materializing E*Q*M."""
        E, M = values.shape[:2]
        if selected.shape[0] != E:
            raise ValueError("selected rows and episode tensors disagree")
        Q, K = selected.shape[1:]
        suffix = values.shape[2:]
        source = values[:, None].expand(E, Q, M, *suffix)
        index = selected.reshape(E, Q, K, *([1] * len(suffix))).expand(E, Q, K, *suffix)
        return torch.gather(source, 2, index)

    def forward_batched(
        self,
        *,
        query_feature: torch.Tensor,       # (E,Q,d)
        query_descriptor: torch.Tensor,    # (E,Q,text_dim)
        query_mask: torch.Tensor,          # (E,Q), True = real query
        memory_feature: torch.Tensor,      # (E,M,d)
        memory_descriptor: torch.Tensor,   # (E,M,text_dim)
        memory_label_text: torch.Tensor,   # (E,M,text_dim)
        memory_enrolled: torch.Tensor,     # (E,M) bool
        memory_mask: torch.Tensor,         # (E,M), True = real row
        candidate_text: torch.Tensor,      # (E,C,text_dim)
        top_k: int,
    ) -> dict[str, torch.Tensor]:
        if query_feature.dim() != 3 or memory_feature.dim() != 3:
            raise ValueError("batched reranking expects (episode,row,feature) tensors")
        E, Q, d = query_feature.shape
        if memory_feature.shape[0] != E or memory_feature.shape[-1] != d:
            raise ValueError("query and memory feature shapes disagree")
        M = memory_feature.shape[1]
        C = candidate_text.shape[1]
        if query_descriptor.shape[:2] != (E, Q) or memory_descriptor.shape[:2] != (E, M):
            raise ValueError("recording features and acquisition descriptors disagree")
        if query_mask.shape != (E, Q) or memory_mask.shape != (E, M):
            raise ValueError("recording masks have the wrong shape")
        if memory_label_text.shape[:2] != (E, M) or memory_enrolled.shape != (E, M):
            raise ValueError("memory labels or enrollment flags have the wrong shape")
        if candidate_text.shape[0] != E or top_k < 1:
            raise ValueError("candidate episodes disagree or top_k is not positive")

        with torch.autocast(device_type=query_feature.device.type, enabled=False):
            q_feature = F.normalize(query_feature.float(), dim=-1)
            m_feature = F.normalize(memory_feature.float(), dim=-1)
            q_descriptor = F.normalize(query_descriptor.float(), dim=-1)
            m_descriptor = F.normalize(memory_descriptor.float(), dim=-1)
            signal_cosine = torch.bmm(q_feature, m_feature.transpose(1, 2))
            descriptor_cosine = torch.bmm(q_descriptor, m_descriptor.transpose(1, 2))

        K = min(int(top_k), M)
        searchable = signal_cosine.masked_fill(~memory_mask[:, None, :], float("-inf"))
        selected = searchable.topk(K, dim=2).indices
        selected_mask = self._gather_rows(memory_mask, selected)
        selected_score = torch.gather(signal_cosine, 2, selected)
        selected_feature = self._gather_rows(m_feature, selected)
        selected_descriptor = self._gather_rows(m_descriptor, selected)
        selected_label_text = self._gather_rows(memory_label_text.float(), selected)
        selected_enrolled = self._gather_rows(memory_enrolled, selected)

        W = E * Q
        device = query_feature.device
        candidates = candidate_text[:, None].expand(E, Q, C, -1).reshape(W, C, -1)
        candidate = self.candidate_compose(
            self.proj_text(candidates),
            self.role_emb(self._role((W, C), ROLE_CANDIDATE, device)),
        )
        query = self.query_compose(
            self.proj_signal(q_feature.reshape(W, 1, d)),
            self.proj_text(q_descriptor.reshape(W, 1, -1)),
            self.role_emb(self._role((W, 1), ROLE_QUERY, device)),
        )

        flat_mask = selected_mask.reshape(W, K)
        scores = selected_score.reshape(W, K)
        count = flat_mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean = scores.masked_fill(~flat_mask, 0.0).sum(dim=1, keepdim=True) / count
        centered = (scores - mean).masked_fill(~flat_mask, 0.0)
        variance = centered.square().sum(dim=1, keepdim=True) / count
        standardized = centered / variance.sqrt().clamp_min(1e-6)
        evidence = self.evidence_compose(
            self.proj_signal(selected_feature.reshape(W, K, d)),
            self.proj_text(selected_descriptor.reshape(W, K, -1)),
            self.proj_text(selected_label_text.reshape(W, K, -1)),
            self.proj_score(standardized.unsqueeze(-1)),
            self.role_emb(self._role((W, K), ROLE_EVIDENCE, device)),
            self.enrollment_emb(selected_enrolled.reshape(W, K).long()),
        )

        tokens = torch.cat((candidate, query, evidence), dim=1)
        valid = torch.cat((
            torch.ones((W, C), dtype=torch.bool, device=device),
            query_mask.reshape(W, 1),
            flat_mask,
        ), dim=1)
        hidden = self.stack(tokens, key_padding_mask=valid)
        evidence_hidden = hidden[:, C + 1:]
        with torch.autocast(device_type=device.type, enabled=False):
            raw_selected = self.row_head(evidence_hidden.float()).squeeze(-1)
            correction_selected = self.correction_gain().float() * torch.tanh(raw_selected)
        correction_selected = correction_selected.masked_fill(~flat_mask, 0.0)

        correction = signal_cosine.new_zeros((E, Q, M))
        correction.scatter_(2, selected, correction_selected.reshape(E, Q, K))
        return {
            "base_score": signal_cosine,
            "descriptor_cosine": descriptor_cosine,
            "selected": selected,
            "selected_mask": selected_mask,
            "raw_correction": raw_selected.reshape(E, Q, K),
            "score_correction": correction,
            "score": signal_cosine + correction,
        }

    @torch.no_grad()
    def telemetry(self) -> dict[str, float]:
        return {
            "reranker/correction_gain": float(self.correction_gain()),
            "reranker/row_head_norm": float(self.row_head.weight.norm()),
            "reranker/candidate_content_gain": float(self.candidate_compose.log_gain[0].exp()),
            "reranker/query_content_gain_mean": float(
                self.query_compose.log_gain[:2].exp().mean()
            ),
            "reranker/evidence_content_gain_mean": float(
                self.evidence_compose.log_gain[:4].exp().mean()
            ),
        }
