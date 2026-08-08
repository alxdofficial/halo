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


@dataclass
class SoftRetrieval:
    logits: torch.Tensor
    entropy: torch.Tensor
    normalized_entropy: torch.Tensor
    effective_rows: torch.Tensor
    retained_mass: torch.Tensor | None


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

    def soft_candidate_logits(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        allowed_mask: torch.Tensor,
        candidate_weights: torch.Tensor,
        row_log_prior: torch.Tensor,
        *,
        tau: float,
        query_mask: torch.Tensor | None = None,
        selected_index: torch.Tensor | None = None,
        output_scale: torch.Tensor | float = 10.0,
    ) -> SoftRetrieval:
        """Differentiable all-row candidate vote used only as a backward surrogate.

        The hard decoder remains the numerical forward path. This soft route gives the query and
        online retrieval projection credit for eligible rows that hard top-k did not select.
        """
        if tau <= 0:
            raise ValueError("soft retrieval temperature must be positive")
        if query.dim() != 3 or memory.dim() != 2:
            raise ValueError("query and memory must have shapes (B,Q,D) and (N,D)")
        B, Q, _ = query.shape
        N = memory.shape[0]
        if allowed_mask.shape == (B, N):
            allowed_mask = allowed_mask.unsqueeze(1).expand(B, Q, N)
        if allowed_mask.shape != (B, Q, N):
            raise ValueError("allowed_mask does not align with query and memory")
        if candidate_weights.dim() == 2:
            if candidate_weights.shape[0] != N:
                raise ValueError("candidate_weights must have one row per memory item")
        elif candidate_weights.dim() == 3:
            if candidate_weights.shape[:2] != (B, N):
                raise ValueError("batched candidate_weights must have shape (B,N,C)")
        else:
            raise ValueError("candidate_weights must have shape (N,C) or (B,N,C)")
        if row_log_prior.shape != (N,):
            raise ValueError("row_log_prior must have shape (N,)")
        if query_mask is None:
            query_mask = torch.ones(B, Q, dtype=torch.bool, device=query.device)
        active = allowed_mask & query_mask.unsqueeze(-1)
        if not bool((active.any(-1) | ~query_mask).all()):
            raise ValueError("at least one valid query patch has no soft-retrieval rows")

        q = self.project(query, ema=False)
        m = self.project(memory, ema=False)
        score = torch.einsum("bqhs,nhs->bqhn", q, m)
        logit = score / float(tau) + row_log_prior.view(1, 1, 1, N)
        logit = logit.masked_fill(~active.unsqueeze(2), float("-inf"))
        # Padding queries are not reduced below; make their softmax finite for diagnostics.
        logit = logit.masked_fill(~query_mask.unsqueeze(2).unsqueeze(3), 0.0)
        probability = torch.softmax(logit, dim=-1)
        probability = probability * query_mask.unsqueeze(2).unsqueeze(3)
        if candidate_weights.dim() == 2:
            mass = torch.einsum("bqhn,nc->bqhc", probability, candidate_weights)
        else:
            mass = torch.einsum("bqhn,bnc->bqhc", probability, candidate_weights)
        denominator = query_mask.sum(1).clamp_min(1).to(mass.dtype) * self.n_subspaces
        candidate_mass = mass.sum(dim=(1, 2)) / denominator.unsqueeze(1)
        logits = candidate_mass * output_scale

        p = probability.clamp_min(1e-12)
        entropy_by_query = -(probability * p.log()).sum(-1)
        eligible = active.sum(-1).clamp_min(2).to(entropy_by_query.dtype)
        normalized = entropy_by_query / eligible.log().unsqueeze(2)
        valid_heads = query_mask.unsqueeze(2).expand_as(entropy_by_query)
        entropy = entropy_by_query[valid_heads].mean()
        normalized_entropy = normalized[valid_heads].mean()
        effective_rows = entropy_by_query[valid_heads].exp().mean()
        retained_mass = None
        if selected_index is not None:
            if selected_index.shape[:3] != (B, Q, self.n_subspaces):
                raise ValueError("selected_index does not align with soft retrieval")
            retained_mass = probability.gather(-1, selected_index).sum(-1)
            retained_mass = retained_mass[valid_heads].mean()
        return SoftRetrieval(
            logits=logits,
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            effective_rows=effective_rows,
            retained_mass=retained_mass,
        )
