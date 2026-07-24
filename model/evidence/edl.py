"""Density-gated evidential (Dirichlet) head for the evidence decoder — Phase-B M5/D1.

An OPT-IN alternative to the closed-vocab cross-entropy in ``train_decoder``. It consumes the
decoder's non-negative per-candidate evidence ``e`` (B,C) from ``aux["evidence"]`` and the raw
top-k retrieval cosines ``vals`` (B,k), and turns them into Dirichlet concentrations

    alpha = 1 + g * beta * e                                             (Sensoy et al. 2018, EDL)

with a **retrieval-density gate** ``g in (0,1]`` that shrinks the evidence (→ high uncertainty
``u = C/S``, i.e. abstain) when the query has no dense support in the memory (novel / unseen).

Prediction-preserving by construction:
  * ``g, beta > 0`` per row  ⇒  ``argmax_c alpha_c == argmax_c e_c == argmax_c logits_c``, so the
    argmax prediction (the transfer bAcc, the selection metric) is IDENTICAL to today's CE path.
  * The default density threshold keeps the sigmoid in its responsive region: dense retrieval gets
    high evidence while genuinely sparse retrieval can lower it. Prediction identity does not
    require ``g≈1``.

The decoder's forward math is NOT touched — this module is a loss/uncertainty head layered on top.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DensityGate(nn.Module):
    """Three learnable scalars (``log_gamma``, ``delta``, ``log_beta``) producing Dirichlet alphas.

    * ``gate(vals)``  → ``g`` (B,) in (0,1] from the top-m raw retrieval cosines.
    * ``alpha(e, vals)`` → ``(alpha, g)`` with ``alpha = 1 + g·beta·e`` (B,C).

    Inits (``gate_delta=0.3``, ``log_gamma=log 10``, ``log_beta=log 10``) put the gate near its
    responsive region for cosine retrieval and give ``beta=10``.
    """

    def __init__(self, gate_delta: float = 0.3, log_gamma: float = math.log(10.0),
                 log_beta: float = math.log(10.0)):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.tensor(float(log_gamma)))
        self.delta = nn.Parameter(torch.tensor(float(gate_delta)))
        self.log_beta = nn.Parameter(torch.tensor(float(log_beta)))

    def gate(self, vals: torch.Tensor, m: int = 8) -> torch.Tensor:
        """Retrieval-density gate ``g`` (B,) in (0,1] from raw top-k cosines ``vals`` (B,k).

        ``dens = mean of the top-min(m,k) raw cosines``; ``g = sigmoid(exp(log_gamma)·(dens-delta))``.
        ``exp(log_gamma) > 0`` keeps the gate monotone increasing in density.
        """
        if vals.ndim != 2 or vals.shape[1] == 0:
            raise ValueError("vals must have shape (batch, k) with k >= 1")
        mm = min(m, vals.shape[1])
        dens = vals[:, :mm].mean(dim=1)                                    # (B,)
        return torch.sigmoid(self.gamma() * (dens - self.delta))

    def gamma(self) -> torch.Tensor:
        return self.log_gamma.clamp(-10.0, 10.0).exp()

    def beta(self) -> torch.Tensor:
        return self.log_beta.clamp(-10.0, 10.0).exp()

    def alpha(self, e: torch.Tensor, vals: torch.Tensor):
        """Dirichlet concentrations ``alpha`` (B,C) and the gate ``g`` (B,).

        ``alpha = 1 + g·beta·e``. Since ``e ≥ 0`` and ``g, beta > 0`` per row, every alpha ≥ 1 (a
        valid Dirichlet) and the per-row argmax over ``c`` is unchanged from ``e`` — prediction
        identity with the CE path.
        """
        if e.ndim != 2 or e.shape[0] != vals.shape[0]:
            raise ValueError("e and vals must be rank-2 tensors with the same batch size")
        if torch.any(e < 0):
            raise ValueError("Dirichlet evidence must be non-negative")
        g = self.gate(vals)                                               # (B,)
        alpha = 1.0 + g.unsqueeze(1) * self.beta() * e                    # (B,C)
        return alpha, g


def kl_dirichlet_to_uniform(alpha_tilde: torch.Tensor, C: int) -> torch.Tensor:
    """Closed-form ``KL( Dir(alpha_tilde) || Dir(1) )`` per row (Sensoy et al. 2018 eq. 5).

    ``alpha_tilde`` (B,C) must have every entry ≥ 1 (a valid Dirichlet); then the divergence is
    ≥ 0 (== 0 iff alpha_tilde is all-ones). Returns (B,).
    """
    A0 = alpha_tilde.sum(1)                                               # (B,)
    return (torch.lgamma(A0) - torch.lgamma(alpha_tilde).sum(1) - math.lgamma(C)
            + ((alpha_tilde - 1.0)
               * (torch.digamma(alpha_tilde) - torch.digamma(A0).unsqueeze(1))).sum(1))


def edl_loss(alpha: torch.Tensor, target: torch.Tensor, lambda_kl: float):
    """Density-gated evidential loss (Sensoy et al. 2018), Bayes-risk form.

    ``alpha`` (B,C), ``target`` (B,) one-hot index. Returns ``(L_edl, L_data_mean, L_kl_mean)``.

      L_data = digamma(S) - digamma(alpha_target)            # expected CE under the Dirichlet
      L_kl   = KL( Dir(alpha_tilde) || Dir(1) )              # alpha_tilde = y + (1-y)·alpha
      L_edl  = mean( L_data + lambda_kl · L_kl )
    """
    if alpha.ndim != 2 or target.shape != (alpha.shape[0],):
        raise ValueError("alpha must be (B,C) and target must be (B,)")
    if torch.any(alpha < 1.0):
        raise ValueError("Dirichlet concentrations produced by EDL must be >= 1")
    if not 0.0 <= float(lambda_kl):
        raise ValueError("lambda_kl must be non-negative")
    B, C = alpha.shape
    if target.numel() and (
        int(target.min().item()) < 0 or int(target.max().item()) >= C
    ):
        raise ValueError("target contains a class index outside alpha")
    ar = torch.arange(B, device=alpha.device)
    S = alpha.sum(1)                                                      # (B,)
    L_data = torch.digamma(S) - torch.digamma(alpha[ar, target])         # (B,)
    alpha_tilde = alpha.clone()
    alpha_tilde[ar, target] = 1.0                                        # remove correct-class evidence
    L_kl = kl_dirichlet_to_uniform(alpha_tilde, C)                        # (B,)
    L_edl = (L_data + lambda_kl * L_kl).mean()
    return L_edl, L_data.mean(), L_kl.mean()


# ---------------------------------------------------------------------------- #
# Calibration / selective-risk metrics on the per-query uncertainty u = C / S. #
# ---------------------------------------------------------------------------- #
def risk_coverage(u, correct):
    """Sweep coverage from most-confident (low ``u``) to all. Returns (coverages, risks) as arrays.

    ``u`` (N,) uncertainty (higher ⇒ abstain first); ``correct`` (N,) bool. At coverage k/N we keep
    the k most-confident queries and ``risk = error rate among them``.
    """
    import numpy as np
    if len(u) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(u, kind="stable")                                 # ascending u = most confident first
    err = 1.0 - correct[order].astype(float)
    risks = np.cumsum(err) / np.arange(1, len(u) + 1)
    coverages = np.arange(1, len(u) + 1) / len(u)
    return coverages, risks


def aurc(u, correct) -> float:
    """Area under the risk–coverage curve (lower is better). NaN on empty input."""
    if len(u) == 0:
        return float("nan")
    _, risks = risk_coverage(u, correct)
    return float(risks.mean())


def acc_at_coverage(u, correct, cov: float = 0.8) -> float:
    """Accuracy among the most-confident ``cov`` fraction of queries (higher is better)."""
    import numpy as np
    if len(u) == 0:
        return float("nan")
    if not 0.0 < cov <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    order = np.argsort(u, kind="stable")
    k = max(1, int(round(cov * len(u))))
    return float(correct[order][:k].astype(float).mean())
