"""Candidate-set-size-aware confidence features for the Phase-B evidence engine."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def confidence_features(
    evidence: torch.Tensor,
    retrieval_scores: torch.Tensor,
    votes: torch.Tensor,
    pool_weights: torch.Tensor,
    ev_mask: torch.Tensor | None = None,
    ev_sensor_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build confidence features without adding an explicit UNKNOWN candidate.

    Features are absolute support, retrieval density, normalized concentration, top-two margin,
    evidence-example agreement, and sensor-group agreement. Entropy is divided by log(C), so merely
    changing the candidate budget does not mechanically change its range.
    """
    if evidence.dim() != 2 or votes.dim() != 3:
        raise ValueError("evidence must be (B,C) and votes must be (B,K,C)")
    B, C = evidence.shape
    if votes.shape[0] != B or votes.shape[2] != C:
        raise ValueError("votes shape does not match evidence")
    K = votes.shape[1]
    if ev_mask is None:
        ev_mask = torch.ones(B, K, dtype=torch.bool, device=evidence.device)
    if pool_weights.shape != (B, K) or ev_mask.shape != (B, K):
        raise ValueError("pool_weights/ev_mask must match votes' first two dimensions")

    total = evidence.sum(dim=1, keepdim=True)
    prob = evidence / total.clamp_min(1e-8)
    absolute = torch.log1p(evidence.max(dim=1).values)
    finite_retrieval = torch.isfinite(retrieval_scores) & ev_mask
    density = (
        retrieval_scores.masked_fill(~finite_retrieval, 0.0).sum(1)
        / finite_retrieval.sum(1).clamp_min(1)
    )
    if C > 1:
        entropy = -(prob * prob.clamp_min(1e-8).log()).sum(1) / math.log(C)
        top2 = prob.topk(2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
    else:
        entropy = torch.zeros(B, device=evidence.device, dtype=evidence.dtype)
        margin = torch.ones(B, device=evidence.device, dtype=evidence.dtype)

    winner = prob.argmax(dim=1)
    ev_winner = votes.argmax(dim=2)
    agrees = ev_winner.eq(winner.unsqueeze(1)).to(pool_weights.dtype)
    example_agreement = (
        (pool_weights * agrees).masked_fill(~ev_mask, 0.0).sum(1)
        / pool_weights.masked_fill(~ev_mask, 0.0).sum(1).clamp_min(1e-8)
    )

    sensor_agreement = example_agreement
    if ev_sensor_id is not None:
        if ev_sensor_id.shape != (B, K):
            raise ValueError("ev_sensor_id must have shape (B,K)")
        rows = []
        for b in range(B):
            valid_sensor = torch.unique(ev_sensor_id[b, ev_mask[b]])
            if not len(valid_sensor):
                rows.append(evidence.new_zeros(()))
                continue
            sensor_votes = []
            for sensor in valid_sensor:
                select = ev_mask[b] & ev_sensor_id[b].eq(sensor)
                score = (votes[b, select] * pool_weights[b, select, None]).sum(0)
                sensor_votes.append(score.argmax().eq(winner[b]).to(evidence.dtype))
            rows.append(torch.stack(sensor_votes).mean())
        sensor_agreement = torch.stack(rows)

    return torch.stack(
        [absolute, density, 1.0 - entropy, margin, example_agreement, sensor_agreement], dim=1
    )


class EvidenceConfidenceHead(nn.Module):
    """Small separate reject-confidence estimator; never changes candidate logits."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != 6:
            raise ValueError(f"expected six confidence features, got {features.shape[-1]}")
        return self.net(features).squeeze(-1)

    @staticmethod
    def loss(logit: torch.Tensor, correct_and_answerable: torch.Tensor) -> torch.Tensor:
        """Binary calibration loss for whether the emitted prediction is usable."""
        return F.binary_cross_entropy_with_logits(
            logit, correct_and_answerable.to(logit.dtype)
        )


def binary_auroc(score, target) -> float:
    """Dependency-free AUROC for binary confidence diagnostics."""
    import numpy as np

    score = np.asarray(score)
    target = np.asarray(target, dtype=bool)
    pos, neg = score[target], score[~target]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(
        (pos[:, None] > neg[None, :]).mean()
        + 0.5 * (pos[:, None] == neg[None, :]).mean()
    )


def aurc(uncertainty, correct) -> float:
    """Area under the empirical risk-coverage curve; lower is better."""
    import numpy as np

    uncertainty = np.asarray(uncertainty)
    correct = np.asarray(correct, dtype=bool)
    if not len(correct):
        return float("nan")
    order = np.argsort(uncertainty, kind="stable")
    risk = 1.0 - np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    return float(risk.mean())


def acc_at_coverage(uncertainty, correct, coverage: float) -> float:
    """Accuracy among the least-uncertain examples at a requested coverage."""
    import numpy as np

    uncertainty = np.asarray(uncertainty)
    correct = np.asarray(correct, dtype=bool)
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    if not len(correct):
        return float("nan")
    keep = max(1, int(np.ceil(coverage * len(correct))))
    return float(correct[np.argsort(uncertainty, kind="stable")[:keep]].mean())
