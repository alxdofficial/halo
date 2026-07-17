"""Rotation, retrieval, and collapse diagnostics for learned representations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from .losses import supervised_contrastive_loss


def _cross_view_accuracy(
    first: torch.Tensor,
    second: torch.Tensor,
    labels: torch.Tensor,
    domains: list[str],
    identities: list[str],
    cross_domain: bool,
) -> float:
    similarities = first @ second.T
    different_recording = torch.tensor(
        [[left != right for right in identities] for left in identities],
        dtype=torch.bool,
        device=similarities.device,
    )
    allowed = different_recording
    if cross_domain:
        different_domain = torch.tensor(
            [[left != right for right in domains] for left in domains],
            dtype=torch.bool,
            device=similarities.device,
        )
        allowed = allowed & different_domain
    similarities = similarities.masked_fill(~allowed, float("-inf"))
    valid = allowed.any(dim=1)
    if not torch.any(valid):
        return float("nan")
    nearest = similarities[valid].argmax(dim=1)
    return float(labels[nearest].eq(labels[valid]).float().mean().item())


def embedding_effective_rank(embeddings: torch.Tensor) -> float:
    if len(embeddings) < 2:
        return 0.0
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    probabilities = singular_values.square()
    probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy.exp().item())


def _similarity_separation(
    first: torch.Tensor,
    second: torch.Tensor,
    labels: torch.Tensor,
    identities: list[str],
) -> dict[str, float]:
    similarities = first @ second.T
    different_recording = torch.tensor(
        [[left != right for right in identities] for left in identities],
        dtype=torch.bool,
    )
    same_activity = labels[:, None].eq(labels[None, :]) & different_recording
    different_activity = labels[:, None].ne(labels[None, :]) & different_recording
    same_mean = (
        similarities[same_activity].mean()
        if torch.any(same_activity)
        else torch.tensor(float("nan"))
    )
    different_mean = (
        similarities[different_activity].mean()
        if torch.any(different_activity)
        else torch.tensor(float("nan"))
    )
    return {
        "same_activity_different_recording_cosine_mean": float(same_mean.item()),
        "different_activity_cosine_mean": float(different_mean.item()),
        "activity_similarity_margin": float((same_mean - different_mean).item()),
    }


@torch.no_grad()
def evaluate_loader(
    model,
    loader: Iterable[dict],
    device: torch.device,
    temperature: float,
    loss_mode: str,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    first_representations = []
    second_representations = []
    first_projections = []
    second_projections = []
    labels = []
    domains: list[str] = []
    identities: list[str] = []
    losses = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        rates = batch["rate_hz"].to(device)
        mask = batch["channel_mask"].to(device)
        lengths = batch["window_length"].to(device)
        label = batch["label_id"].to(device)
        output_a = model(batch["view_a"].to(device), rates, mask, lengths)
        output_b = model(batch["view_b"].to(device), rates, mask, lengths)
        projections = torch.stack([output_a["projection"], output_b["projection"]], dim=1)
        losses.append(supervised_contrastive_loss(
            projections, label, temperature=temperature, mode=loss_mode
        ).item())
        first_representations.append(output_a["representation"].cpu())
        second_representations.append(output_b["representation"].cpu())
        first_projections.append(output_a["projection"].cpu())
        second_projections.append(output_b["projection"].cpu())
        labels.append(label.cpu())
        domains.extend(batch["domain"])
        identities.extend([
            f"{domain}:{int(window_index)}"
            for domain, window_index in zip(batch["domain"], batch["window_index"])
        ])

    if not losses:
        raise ValueError("Evaluation loader produced no batches")
    first = torch.cat(first_representations)
    second = torch.cat(second_representations)
    first_projection = torch.cat(first_projections)
    second_projection = torch.cat(second_projections)
    all_labels = torch.cat(labels)
    paired_cosine = F.cosine_similarity(first, second, dim=1)
    projection_paired_cosine = F.cosine_similarity(
        first_projection, second_projection, dim=1
    )
    combined = torch.cat([first, second])
    separation = _similarity_separation(
        first, second, all_labels, identities
    )
    return {
        "loss": float(np.mean(losses)),
        "paired_view_cosine_mean": float(paired_cosine.mean().item()),
        "paired_view_cosine_p10": float(torch.quantile(paired_cosine, 0.1).item()),
        "projection_paired_view_cosine_mean": float(
            projection_paired_cosine.mean().item()
        ),
        "projection_paired_view_cosine_p10": float(
            torch.quantile(projection_paired_cosine, 0.1).item()
        ),
        "activity_retrieval_accuracy": _cross_view_accuracy(
            first, second, all_labels, domains, identities, cross_domain=False
        ),
        "cross_domain_activity_retrieval_accuracy": _cross_view_accuracy(
            first, second, all_labels, domains, identities, cross_domain=True
        ),
        "embedding_dimension_std_mean": float(combined.std(dim=0).mean().item()),
        "embedding_effective_rank": embedding_effective_rank(combined),
        "evaluated_windows": int(len(first)),
        **separation,
        "paired_over_different_activity_margin": float(
            paired_cosine.mean().item() - separation["different_activity_cosine_mean"]
        ),
    }
