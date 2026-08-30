"""Deterministic short synthetic optimization of the complete Task 3 path."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from .candidates import assign_event_targets, pool_multiscale_candidates
from .contracts import EventBatch
from .model import RecurrentMotionMetric
from .training import initialize_affinity_threshold, train_step


def synthetic_batch(
    *, seed: int = 7
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, EventBatch]:
    """Create two scoped datasets with repeated, bounded latent motions."""

    generator = torch.Generator().manual_seed(seed)
    batch_size, patches, feature_dim = 4, 28, 8
    embeddings = 0.12 * torch.randn(
        batch_size, patches, feature_dim, generator=generator
    )
    intervals = (
        torch.stack(
            [
                torch.arange(patches, dtype=torch.float32),
                torch.arange(1, patches + 1, dtype=torch.float32),
            ],
            dim=-1,
        )
        .expand(batch_size, -1, -1)
        .clone()
    )
    patch_mask = torch.ones(batch_size, patches, dtype=torch.bool)

    # The two scopes intentionally reuse local labels 0 and 1. Their prototypes
    # differ, proving that numeric identities are not shared across datasets.
    prototypes = F.normalize(
        torch.randn(2, 2, feature_dim, generator=generator), dim=-1
    )
    event_starts = (
        torch.tensor([[2.0, 10.0, 18.0, 24.0]]).expand(batch_size, -1).clone()
    )
    event_ends = event_starts + 4.0
    event_labels = torch.tensor([[0, 1, 0, 1]]).expand(batch_size, -1).clone()
    scope = torch.tensor([0, 0, 1, 1])
    for recording in range(batch_size):
        for event_index, start in enumerate(event_starts[recording].long().tolist()):
            label = int(event_labels[recording, event_index])
            embeddings[recording, start : start + 4] += prototypes[
                scope[recording], label
            ]
    embeddings.requires_grad_(False)
    events = EventBatch(
        start_sec=event_starts,
        end_sec=event_ends,
        label_id=event_labels,
        instance_id=torch.arange(batch_size * 4).reshape(batch_size, 4),
        scope_id=scope[:, None].expand(-1, 4).clone(),
        event_mask=torch.ones(batch_size, 4, dtype=torch.bool),
        exhaustive=torch.ones(batch_size, dtype=torch.bool),
    )
    return embeddings, intervals, patch_mask, events


def run_smoke_training(*, steps: int = 30, seed: int = 7) -> dict[str, float]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    patch_embeddings, intervals, patch_mask, events = synthetic_batch(seed=seed)
    candidates = pool_multiscale_candidates(
        patch_embeddings,
        intervals,
        patch_mask,
        durations_sec=(2.0, 4.0, 6.0),
        candidate_stride_sec=1.0,
    )
    targets = assign_event_targets(candidates, events, positive_iou=0.7)
    model = RecurrentMotionMetric(patch_embeddings.shape[-1], projection_dim=6)
    initialize_affinity_threshold(model, candidates, targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=1e-3)
    first = None
    latest = None
    for _ in range(steps):
        latest = train_step(model, optimizer, candidates, targets)
        if first is None:
            first = latest
    assert first is not None and latest is not None
    return {
        "steps": float(steps),
        "candidate_count": float(candidates.valid_count),
        "initial_loss": first.loss,
        "final_loss": latest.loss,
        **latest.telemetry,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke_training(steps=args.steps, seed=args.seed),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
