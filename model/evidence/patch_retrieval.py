"""Learned, EMA-indexed subspace retrieval over frozen Phase-A patch vectors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PatchRetrieval:
    index: torch.Tensor       # (B, Q, H, K)
    score: torch.Tensor       # (B, Q, H, K)
    valid: torch.Tensor       # (B, Q, H, K)


class PatchSubspaceRetriever(nn.Module):
    """Project patches into several independently retrieved similarity spaces.

    The online projection receives gradients. A stop-gradient EMA projection builds the memory
    index and chooses neighbours, preventing the lookup target from changing at every optimizer
    microstep. The caller controls index rebuild cadence explicitly.
    """

    def __init__(
        self,
        d_model: int,
        n_subspaces: int = 4,
        subspace_dim: int = 64,
        ema_decay: float = 0.995,
    ):
        super().__init__()
        if n_subspaces < 1 or subspace_dim < 1:
            raise ValueError("n_subspaces and subspace_dim must be positive")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        self.d_model = int(d_model)
        self.n_subspaces = int(n_subspaces)
        self.subspace_dim = int(subspace_dim)
        self.ema_decay = float(ema_decay)
        self.proj = nn.Parameter(torch.empty(n_subspaces, d_model, subspace_dim))
        nn.init.orthogonal_(self.proj.view(n_subspaces * d_model, subspace_dim))
        self.register_buffer("ema_proj", self.proj.detach().clone())

    def project(self, z: torch.Tensor, *, ema: bool = False) -> torch.Tensor:
        """(..., D) -> (..., H, S), L2-normalized independently per subspace."""
        if z.shape[-1] != self.d_model:
            raise ValueError(f"expected final dim {self.d_model}, got {z.shape[-1]}")
        projection = self.ema_proj if ema else self.proj
        return F.normalize(torch.einsum("...d,hds->...hs", z.float(), projection), dim=-1)

    @torch.no_grad()
    def update_ema(self) -> None:
        self.ema_proj.lerp_(self.proj.detach(), 1.0 - self.ema_decay)

    @torch.no_grad()
    def build_index(self, memory: torch.Tensor) -> torch.Tensor:
        """Return the EMA-projected memory index (N,H,S), suitable for caching."""
        return self.project(memory, ema=True).detach()

    @torch.no_grad()
    def retrieve(
        self,
        query: torch.Tensor,
        memory_index: torch.Tensor,
        allowed_mask: torch.Tensor,
        k: int,
        query_mask: torch.Tensor | None = None,
    ) -> PatchRetrieval:
        """Independent top-k lookup for every query patch and subspace.

        ``query`` is (B,Q,D). ``allowed_mask`` may be (B,N) or (B,Q,N). Unlike a global minimum-k
        implementation, rows with fewer eligible items are padded and marked invalid rather than
        shrinking every other query's evidence set.
        """
        if query.dim() != 3:
            raise ValueError(f"query must have shape (B,Q,D), got {tuple(query.shape)}")
        B, Q, _ = query.shape
        N = memory_index.shape[0]
        if memory_index.shape[1:] != (self.n_subspaces, self.subspace_dim):
            raise ValueError("memory_index was built for a different retriever")
        if allowed_mask.shape == (B, N):
            allowed_mask = allowed_mask.unsqueeze(1).expand(B, Q, N)
        if allowed_mask.shape != (B, Q, N):
            raise ValueError(
                f"allowed_mask must have shape {(B, N)} or {(B, Q, N)}, "
                f"got {tuple(allowed_mask.shape)}"
            )
        if query_mask is None:
            query_mask = torch.ones(B, Q, dtype=torch.bool, device=query.device)
        if query_mask.shape != (B, Q):
            raise ValueError(f"query_mask must have shape {(B, Q)}, got {tuple(query_mask.shape)}")
        active_has_memory = allowed_mask.any(dim=-1) | ~query_mask
        if not bool(active_has_memory.all()):
            raise ValueError("at least one valid query patch has no eligible memory rows")
        k_eff = min(int(k), N)
        if k_eff < 1:
            raise ValueError("k must be positive and memory must be non-empty")

        q_index = self.project(query, ema=True)                                  # (B,Q,H,S)
        similarity = torch.einsum("bqhs,nhs->bqhn", q_index, memory_index)
        similarity = similarity.masked_fill(~allowed_mask.unsqueeze(2), float("-inf"))
        score, index = similarity.topk(k_eff, dim=-1)
        valid = torch.isfinite(score) & query_mask.unsqueeze(2).unsqueeze(3)
        # Padded indices must remain legal for later gather; validity keeps them out of attention.
        index = index.masked_fill(~valid, 0)
        score = score.masked_fill(~valid, 0.0)
        return PatchRetrieval(index=index, score=score, valid=valid)

    def score_selected(self, query: torch.Tensor, memory: torch.Tensor,
                       index: torch.Tensor) -> torch.Tensor:
        """Differentiable online-projector scores for EMA-selected rows."""
        B, Q, H, K = index.shape
        if H != self.n_subspaces:
            raise ValueError("index subspace count does not match retriever")
        q = self.project(query, ema=False)                                       # (B,Q,H,S)
        selected = memory[index]                                                 # (B,Q,H,K,D)
        selected = self.project(selected, ema=False)                             # (B,Q,H,K,H,S)
        selector = torch.arange(H, device=index.device).view(1, 1, H, 1, 1, 1)
        selected = torch.gather(
            selected, -2, selector.expand(B, Q, H, K, 1, self.subspace_dim)
        ).squeeze(-2)
        return torch.einsum("bqhs,bqhks->bqhk", q, selected)

    def score_pairs(
        self,
        query: torch.Tensor,
        evidence: torch.Tensor,
        head: torch.Tensor,
        query_patch: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiably score an assembled ``(query patch, head, evidence)`` roster."""
        if evidence.dim() != 3 or head.shape != evidence.shape[:2] \
                or query_patch.shape != evidence.shape[:2]:
            raise ValueError("assembled evidence/head/query_patch shapes do not align")
        B, E, _ = evidence.shape
        q = self.project(query, ema=False)                       # (B,Q,H,S)
        ev = self.project(evidence, ema=False)                   # (B,E,H,S)
        batch = torch.arange(B, device=query.device).view(B, 1).expand(B, E)
        q_selected = q[batch, query_patch, head]
        ev_selected = ev[batch, torch.arange(E, device=query.device), head]
        return torch.einsum("bes,bes->be", q_selected, ev_selected)
