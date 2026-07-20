"""M4a retrieval evidence head — a *learned* ConSE (docs/design/EVIDENCE_ENGINE.md §4.2).

Given a query vector `z` (from the frozen encoder) and a memory of labeled training
vectors, emit **per-candidate evidence** by (1) retrieving neighbors under a learned metric
`g`, (2) letting each neighbor vote across candidates through a shared text kernel `t`.

    s_i  = <g(z), g(z_i)>                      # learned-metric similarity to memory entry i
    w    = softmax(s / tau)                     # full-soft retrieval weights over allowed memory
    K    = relu(<t(label_i), t(c)>)             # a neighbor's label votes for candidate c by text sim
    e_c  = Σ_i w_i · K_{i,c}                     # per-candidate evidence

Learned + tiny: `g`, `t`, `tau`, an output scale. Memory vectors and base SBERT text are
FROZEN (passed in, stop-grad) — the encoder never runs here. This is M4a: evidence → argmax
is the prediction. The density gate + Dirichlet/abstention head are added in M5.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceHead(nn.Module):
    def __init__(self, d_model: int, text_dim: int = 384, proj: int = 256,
                 tau_init: float = 0.1, out_scale_init: float = 10.0):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(d_model, proj), nn.GELU(), nn.Linear(proj, proj))
        self.t = nn.Sequential(nn.Linear(text_dim, proj), nn.GELU(), nn.Linear(proj, proj))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        self.log_out_scale = nn.Parameter(torch.tensor(math.log(out_scale_init)))

    # -- projections (compute once per step over the whole memory, then reuse per batch) --
    def project_query(self, z: torch.Tensor) -> torch.Tensor:
        """(., d) -> (., proj) L2-normalized query/memory metric embedding."""
        return F.normalize(self.g(z), dim=-1)

    def project_text(self, txt: torch.Tensor) -> torch.Tensor:
        """(., 384) -> (., proj) L2-normalized label-text embedding (the voting kernel space)."""
        return F.normalize(self.t(txt), dim=-1)

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(min=1e-3, max=1.0)

    def evidence(
        self,
        gq: torch.Tensor,            # (B, proj)  projected queries
        g_mem: torch.Tensor,         # (N, proj)  projected memory
        mem_y: torch.Tensor,         # (N,)       memory label -> row of `t_labels`
        cand_proj: torch.Tensor,     # (C, proj)  projected candidate label texts
        t_labels: torch.Tensor,      # (L, proj)  projected vocab label texts (indexed by mem_y)
        retrieval_mask: torch.Tensor,  # (B, N) bool, True = neighbor allowed (subject-disjoint etc.)
        return_weights: bool = False,
    ):
        # Divide by tau BEFORE masking: routing -inf through the /tau node would give an
        # inf gradient w.r.t. tau at masked entries (0*inf = NaN). Mask the scaled scores.
        s = (gq @ g_mem.t()) / self.tau                           # (B, N)
        s = s.masked_fill(~retrieval_mask, float("-inf"))
        w = torch.softmax(s, dim=1)                              # (B, N) sums to 1 over allowed
        K = torch.relu(t_labels[mem_y] @ cand_proj.t())          # (N, C) neighbor->candidate votes
        e = w @ K                                                # (B, C) per-candidate evidence
        if return_weights:
            return e, w, s
        return e

    def logits(self, e: torch.Tensor) -> torch.Tensor:
        """Turn non-negative evidence into classification logits (M4a: scaled evidence)."""
        return self.log_out_scale.exp() * e
