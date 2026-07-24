"""Multi-subspace ("different ways of being similar") evidence re-weighting — Phase-B D2.

The frozen raw-cosine top-k retrieval measures similarity in ONE metric (the full-d encoder
space). But two windows can be similar in several distinct ways at once — same motion / different
placement, same placement / different activity, same rate / different limb, ... This head learns
``K`` low-dim projections ``P_m: d -> d_sub``, scores each retrieved evidence token by its cosine to
the query WITHIN each subspace, then mixes those ``K`` per-subspace similarities with a learned
convex weighting ``omega`` (softmax over ``K`` logits). The result ``psi[b,i]`` is an extra, learned
"how relevant is this evidence" signal that the decoder folds into its POOLING as an
identity-at-init gated residual: ``a_logit += gamma_ms * psi`` with ``gamma_ms`` zero-init.

Scope (deliberately conservative first cut): this re-weights the POOLING of the already-retrieved
evidence. It does NOT touch the frozen raw-cosine top-k retrieval itself — the retrieved set is
unchanged; only how much each member votes is re-weighted.

Do-no-harm: ``gamma_ms`` starts at exactly 0, so the head contributes nothing at init and the
decoder stays byte-identical to the untrained retrieval mechanism. Any gain is earned as gamma_ms
lifts off zero, and it can be regularized straight back toward 0 (like the Δ / pool residuals).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSubspaceHead(nn.Module):
    """K learned similarity subspaces + a learned convex mix of their per-evidence cosines.

    ``forward(zq (B,d), zev (B,k,d)) -> psi (B,k)`` where
        ``psi[b,i] = Σ_m omega_m · cos(P_m zq_b, P_m zev_{b,i})``,  ``omega = softmax(omega_logits)``.
    Cosines are computed by L2-normalizing WITHIN each subspace, so each ``P_m`` defines its own
    metric. ``omega`` is uniform at init (omega_logits all-zero), which is a benign starting mix.
    """

    def __init__(self, d_model: int, n_subspaces: int, subspace_dim: int):
        super().__init__()
        assert n_subspaces > 0 and subspace_dim > 0
        self.K = int(n_subspaces)
        self.d_sub = int(subspace_dim)
        # One bias-free projection per subspace, packed into a single matmul: d -> K*d_sub.
        self.proj = nn.Linear(d_model, self.K * self.d_sub, bias=False)
        # Learned convex weighting of the K subspaces (softmax); uniform (all-zeros) at init.
        self.omega_logits = nn.Parameter(torch.zeros(self.K))

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """(..., d) -> (..., K, d_sub), L2-normalized within each subspace (unit vectors)."""
        p = self.proj(x)                                             # (..., K*d_sub)
        p = p.reshape(*p.shape[:-1], self.K, self.d_sub)            # (..., K, d_sub)
        return F.normalize(p, dim=-1)

    def forward(self, zq: torch.Tensor, zev: torch.Tensor) -> torch.Tensor:
        """zq (B,d), zev (B,k,d) -> psi (B,k). Per-evidence, so it permutes with the evidence axis."""
        q_sub = self._project(zq)                                   # (B, K, d_sub)
        ev_sub = self._project(zev)                                 # (B, k, K, d_sub)
        # per-subspace cosine (unit vectors -> dot product): s[b,i,m]
        s = torch.einsum("bmd,bimd->bim", q_sub, ev_sub)           # (B, k, K)
        omega = torch.softmax(self.omega_logits, dim=0)            # (K,) convex weights
        psi = torch.einsum("bim,m->bi", s, omega)                  # (B, k)
        return psi
