"""Contrastive objectives with explicit positive masks."""

from __future__ import annotations

import torch


def contrastive_positive_mask(labels: torch.Tensor, views: int, mode: str) -> torch.Tensor:
    sample_ids = torch.arange(len(labels), device=labels.device).repeat_interleave(views)
    if mode == "supervised":
        expanded_labels = labels.repeat_interleave(views)
        positives = expanded_labels[:, None].eq(expanded_labels[None, :])
    elif mode == "instance":
        positives = sample_ids[:, None].eq(sample_ids[None, :])
    else:
        raise ValueError("mode must be 'supervised' or 'instance'")
    positives.fill_diagonal_(False)
    return positives


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    mode: str = "supervised",
) -> torch.Tensor:
    """Multi-positive NT-Xent over `(batch, views, embedding)` normalized features."""
    if features.ndim != 3 or features.shape[0] != len(labels):
        raise ValueError("features must be (batch, views, embedding) and match labels")
    if features.shape[1] < 2:
        raise ValueError("At least two views are required")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    batch, views, _ = features.shape
    flattened = features.reshape(batch * views, -1)
    logits = flattened @ flattened.T / temperature
    self_mask = torch.eye(len(flattened), dtype=torch.bool, device=features.device)
    logits = logits.masked_fill(self_mask, float("-inf"))
    positives = contrastive_positive_mask(labels, views, mode)
    positive_count = positives.sum(dim=1)
    if torch.any(positive_count == 0):
        raise ValueError("Every anchor requires at least one positive")

    log_denominator = torch.logsumexp(logits, dim=1)
    positive_logits = logits.masked_fill(~positives, 0.0).sum(dim=1) / positive_count
    return (log_denominator - positive_logits).mean()
