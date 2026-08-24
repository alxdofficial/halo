"""Recording-level evidence mixing for the compact HALO evidence engine.

One sequence represents one recording of interest:

    [candidate labels]
    [all query patch/sensor rows + their configuration descriptions]
    [globally retrieved evidence rows + descriptions + labels]

The mixer emits one shared, permutation-equivariant residual logit per candidate. The final linear
layer is zero-initialised, so the complete engine starts exactly at its closed-form retrieval vote.
Only that final layer is dormant on the first backward pass; after its first update, gradients reach
the attention stack. Training telemetry and tests explicitly check this two-step wake-up behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import AttentionSpec, ScaledSum, SetAttentionStack

(ROLE_CANDIDATE, ROLE_QUERY, ROLE_QUERY_DESC,
 ROLE_EVIDENCE, ROLE_EVIDENCE_DESC, ROLE_EVIDENCE_LABEL) = range(6)
N_ROLES = 6
UNBOUND_SLOT = 0
NO_RECORDING_GROUP = 0
QUERY_RECORDING_GROUP = 1


@dataclass(frozen=True)
class EvidenceMixerConfig:
    text_dim: int = 384
    n_layers: int = 2
    n_slots: int = 64
    n_groups: int = 96
    #: Content should lead at initialization. Role, coreference, and recording identity remain
    #: visible without jointly contributing three times as much vector magnitude as content.
    identity_gain_init: float = 0.25
    score_bias_init: float = 1.0


class EvidenceMixer(nn.Module):
    """Mix all query and retrieved rows for each recording, then correct candidate logits."""

    def __init__(self, spec: AttentionSpec, cfg: EvidenceMixerConfig | None = None):
        super().__init__()
        self.spec = spec
        self.cfg = cfg or EvidenceMixerConfig()
        d = spec.d_model
        self.proj_text = nn.Linear(self.cfg.text_dim, d)
        self.proj_signal = nn.Linear(d, d)
        self.role_emb = nn.Embedding(N_ROLES, d)
        self.slot_emb = nn.Embedding(self.cfg.n_slots, d)
        self.group_emb = nn.Embedding(self.cfg.n_groups, d)
        self.compose = ScaledSum(4, init=[1.0] + [self.cfg.identity_gain_init] * 3)
        self.stack = SetAttentionStack(spec, self.cfg.n_layers)
        self.score_bias_gain = nn.Parameter(torch.tensor(float(self.cfg.score_bias_init)))
        # A shared scalar head preserves candidate permutation equivariance and supports unseen
        # labels. Zero initialization makes the whole engine exactly its retrieval baseline at step
        # zero. The head itself receives gradient immediately; the stack wakes after that update.
        # No bias: one shared scalar bias would add the same constant to every candidate logit and
        # cancel exactly in softmax, making it a permanently unidentifiable parameter.
        self.residual_head = nn.Linear(d, 1, bias=False)
        nn.init.zeros_(self.residual_head.weight)

    def _token(self, content: torch.Tensor, role: int, slot: torch.Tensor,
               group: torch.Tensor) -> torch.Tensor:
        role_vec = self.role_emb(torch.full(
            content.shape[:-1], role, dtype=torch.long, device=content.device,
        ))
        return self.compose(content, role_vec, self.slot_emb(slot), self.group_emb(group))

    def forward(
        self,
        *,
        retrieval_score: torch.Tensor,        # (W, K), aggregated over query rows
        candidate_text: torch.Tensor,         # (C, text_dim)
        query_feature: torch.Tensor,          # (W, Qmax, d)
        query_descriptor: torch.Tensor,       # (W, Qmax, text_dim)
        query_mask: torch.Tensor,             # (W, Qmax), True = real row
        evidence_feature: torch.Tensor,       # (W, K, d)
        evidence_descriptor: torch.Tensor,    # (W, K, text_dim)
        evidence_label_text: torch.Tensor,    # (W, K, text_dim)
        candidate_slot: torch.Tensor,         # (C,), episode-randomized
        evidence_slot: torch.Tensor,          # (W, K)
        evidence_group: torch.Tensor,         # (W, K)
    ) -> dict[str, torch.Tensor]:
        W, Qmax, _ = query_feature.shape
        C, K = candidate_text.shape[0], evidence_feature.shape[1]
        device = query_feature.device
        candidate_slot = candidate_slot.unsqueeze(0).expand(W, C)
        neutral_candidate_group = torch.full(
            (W, C), NO_RECORDING_GROUP, dtype=torch.long, device=device,
        )
        query_slot = torch.full((W, Qmax), UNBOUND_SLOT, dtype=torch.long, device=device)
        query_group = torch.full(
            (W, Qmax), QUERY_RECORDING_GROUP, dtype=torch.long, device=device,
        )

        candidate = self._token(
            self.proj_text(candidate_text).unsqueeze(0).expand(W, C, -1),
            ROLE_CANDIDATE, candidate_slot, neutral_candidate_group,
        )
        query = self._token(
            self.proj_signal(query_feature), ROLE_QUERY, query_slot, query_group,
        )
        query_desc = self._token(
            self.proj_text(query_descriptor), ROLE_QUERY_DESC, query_slot, query_group,
        )
        evidence = self._token(
            self.proj_signal(evidence_feature), ROLE_EVIDENCE, evidence_slot, evidence_group,
        )
        evidence_desc = self._token(
            self.proj_text(evidence_descriptor), ROLE_EVIDENCE_DESC,
            evidence_slot, evidence_group,
        )
        evidence_label = self._token(
            self.proj_text(evidence_label_text), ROLE_EVIDENCE_LABEL,
            evidence_slot, evidence_group,
        )
        tokens = torch.cat(
            [candidate, query, query_desc, evidence, evidence_desc, evidence_label], dim=1,
        )
        valid = torch.cat([
            torch.ones((W, C), dtype=torch.bool, device=device),
            query_mask, query_mask,
            torch.ones((W, 3 * K), dtype=torch.bool, device=device),
        ], dim=1)

        # Retrieval scores bias all three tokens belonging to an evidence row. Standardization keeps
        # the bias meaningful when feature score scales drift during end-to-end training.
        centered = retrieval_score - retrieval_score.mean(dim=1, keepdim=True)
        centered = centered / centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        bias = tokens.new_zeros((W, tokens.shape[1]))
        evidence_offset = C + 2 * Qmax
        bias[:, evidence_offset:] = self.score_bias_gain * centered.repeat(1, 3)
        hidden = self.stack(tokens, key_padding_mask=valid, attn_bias=bias.unsqueeze(1))
        with torch.autocast(device_type=device.type, enabled=False):
            residual = self.residual_head(hidden[:, :C].float()).squeeze(-1)
        return {"residual_logits": residual}

    def telemetry(self) -> dict[str, float]:
        gains = self.compose.log_gain.detach().exp()
        return {
            "mixer/score_bias_gain": float(self.score_bias_gain.detach()),
            "mixer/residual_head_norm": float(self.residual_head.weight.detach().norm()),
            "mixer/content_gain": float(gains[0]),
            "mixer/identity_gain_mean": float(gains[1:].mean()),
        }
