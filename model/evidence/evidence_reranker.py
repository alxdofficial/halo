"""A deliberately small evidence-row reranker for Phase B.

Phase A owns representation learning.  This module receives the cosine shortlist for one query
recording and decides only how much to trust each retrieved row.  It cannot emit candidate logits:
the engine applies its scalar corrections to retrieval scores and then uses the ordinary evidence
labels to vote.  That bottleneck prevents the direct query-to-candidate classifier shortcut observed
in the earlier candidate-residual mixer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..blocks import AttentionSpec, ScaledSum, SetAttentionStack


ROLE_CANDIDATE, ROLE_QUERY, ROLE_EVIDENCE = range(3)
N_ROLES = 3
@dataclass(frozen=True)
class EvidenceRerankerConfig:
    text_dim: int = 384
    n_layers: int = 1
    n_groups: int = 96
    identity_gain_init: float = 0.25
    correction_gain_init: float = 0.05
    max_correction: float = 2.0
    head_init_std: float = 1e-3

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("the contextual reranker needs at least one attention layer")
        if self.n_groups < 3:
            raise ValueError("reranker needs at least 3 recording groups")
        if not 0.0 < self.identity_gain_init:
            raise ValueError("identity_gain_init must be positive")
        if not 0.0 < self.correction_gain_init < self.max_correction:
            raise ValueError("correction_gain_init must lie in (0, max_correction)")
        if self.max_correction <= 0.0 or self.head_init_std <= 0.0:
            raise ValueError("max_correction and head_init_std must be positive")


class EvidenceReranker(nn.Module):
    """Contextualize one recording's shortlist and return one scalar correction per row."""

    def __init__(self, spec: AttentionSpec, cfg: EvidenceRerankerConfig | None = None):
        super().__init__()
        self.spec = spec
        self.cfg = cfg or EvidenceRerankerConfig()
        d = spec.d_model
        self.proj_text = nn.Linear(self.cfg.text_dim, d)
        self.proj_signal = nn.Linear(d, d)
        # A nonlinear scalar embedding preserves score magnitude after normalization in ScaledSum.
        self.proj_score = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.role_emb = nn.Embedding(N_ROLES, d)
        self.group_emb = nn.Embedding(self.cfg.n_groups, d)

        self.candidate_compose = ScaledSum(
            2, init=[1.0, self.cfg.identity_gain_init],
        )
        self.query_compose = ScaledSum(
            3, init=[1.0, 1.0, self.cfg.identity_gain_init],
        )
        self.evidence_compose = ScaledSum(
            6, init=[1.0, 1.0, 1.0, 1.0,
                     self.cfg.identity_gain_init, self.cfg.identity_gain_init],
        )
        self.stack = SetAttentionStack(spec, self.cfg.n_layers)
        self.row_head = nn.Linear(d, 1, bias=False)
        nn.init.normal_(self.row_head.weight, std=self.cfg.head_init_std)

        ratio = self.cfg.correction_gain_init / self.cfg.max_correction
        self.correction_gain_logit = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))

    def correction_gain(self) -> torch.Tensor:
        return self.cfg.max_correction * torch.sigmoid(self.correction_gain_logit)

    @staticmethod
    def _role(shape: tuple[int, ...], role: int, device: torch.device) -> torch.Tensor:
        return torch.full(shape, role, dtype=torch.long, device=device)

    def forward(
        self,
        *,
        retrieval_score: torch.Tensor,        # (W, K), recording-level cosine score
        candidate_text: torch.Tensor,         # (C, text_dim) or (W, C, text_dim)
        query_feature: torch.Tensor,          # (W, Qmax, d)
        query_descriptor: torch.Tensor,       # (W, Qmax, text_dim)
        query_mask: torch.Tensor,             # (W, Qmax), True = real row
        evidence_feature: torch.Tensor,       # (W, K, d)
        evidence_descriptor: torch.Tensor,    # (W, K, text_dim)
        evidence_label_text: torch.Tensor,    # (W, K, text_dim)
        evidence_group: torch.Tensor,         # (W, K)
    ) -> dict[str, torch.Tensor]:
        W, Qmax, _ = query_feature.shape
        if candidate_text.dim() == 2:
            C = candidate_text.shape[0]
            candidate_text = candidate_text.unsqueeze(0).expand(W, -1, -1)
        elif candidate_text.dim() == 3 and candidate_text.shape[0] == W:
            C = candidate_text.shape[1]
        else:
            raise ValueError(
                "candidate_text must be (C, text_dim) or one (C, text_dim) roster per recording"
            )
        K = evidence_feature.shape[1]
        device = query_feature.device
        if retrieval_score.shape != (W, K):
            raise ValueError("retrieval_score and evidence rows disagree")

        candidate = self.candidate_compose(
            self.proj_text(candidate_text),
            self.role_emb(self._role((W, C), ROLE_CANDIDATE, device)),
        )
        query = self.query_compose(
            self.proj_signal(query_feature),
            self.proj_text(query_descriptor),
            self.role_emb(self._role((W, Qmax), ROLE_QUERY, device)),
        )

        centered = retrieval_score - retrieval_score.mean(dim=1, keepdim=True)
        centered = centered / centered.std(
            dim=1, keepdim=True, unbiased=False,
        ).clamp_min(1e-6)
        evidence = self.evidence_compose(
            self.proj_signal(evidence_feature),
            self.proj_text(evidence_descriptor),
            self.proj_text(evidence_label_text),
            self.proj_score(centered.unsqueeze(-1)),
            self.role_emb(self._role((W, K), ROLE_EVIDENCE, device)),
            self.group_emb(evidence_group),
        )
        tokens = torch.cat((candidate, query, evidence), dim=1)
        valid = torch.cat((
            torch.ones((W, C), dtype=torch.bool, device=device),
            query_mask,
            torch.ones((W, K), dtype=torch.bool, device=device),
        ), dim=1)
        hidden = self.stack(tokens, key_padding_mask=valid)
        evidence_hidden = hidden[:, C + Qmax:]
        with torch.autocast(device_type=device.type, enabled=False):
            raw = self.row_head(evidence_hidden.float()).squeeze(-1)
            correction = self.correction_gain().float() * torch.tanh(raw)
        return {"score_correction": correction, "raw_correction": raw}

    @torch.no_grad()
    def telemetry(self) -> dict[str, float]:
        return {
            "reranker/correction_gain": float(self.correction_gain()),
            "reranker/row_head_norm": float(self.row_head.weight.norm()),
            "reranker/candidate_content_gain": float(
                self.candidate_compose.log_gain[0].exp()
            ),
            "reranker/query_content_gain_mean": float(
                self.query_compose.log_gain[:2].exp().mean()
            ),
            "reranker/evidence_content_gain_mean": float(
                self.evidence_compose.log_gain[:4].exp().mean()
            ),
        }
