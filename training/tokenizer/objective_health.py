"""Empirical health report for the consolidated JEPA + augmentation-VICReg objectives.

This is a CPU/data diagnostic: it draws real temperature-sampled batches, measures honest JEPA
supervision after validity masking, and verifies that both augmented VICReg views are populated.

Run: /home/alex/code/HALO/legacy_code/.venv/bin/python -m training.tokenizer.objective_health
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.tokenizer.losses_repr import make_per_resolution_mask_plan
from training.tokenizer.pretrain_data import (
    CorpusIndex,
    MultiResolutionCollate,
    PretrainDataset,
    TemperatureSampler,
    _seed_worker,
)

OUT = Path(__file__).resolve().parent / "outputs" / "objective_health"
GYRO = [3, 4, 5]
N_BATCHES = 20
BATCH_SIZE = 256


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "min": round(float(array.min()), 4),
        "p10": round(float(np.percentile(array, 10)), 4),
        "median": round(float(np.median(array)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
        "max": round(float(array.max()), 4),
        "mean": round(float(array.mean()), 4),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    index = CorpusIndex(seed=0)
    dataset = PretrainDataset(index, index.train, augment=True, two_view=True)
    sampler = TemperatureSampler(
        index.train,
        index.stream_datasets,
        num_samples=N_BATCHES * BATCH_SIZE,
        alpha=0.25,
        seed=0,
        batch_size=BATCH_SIZE,
        subject_ids=index.train_subject_ids,
        subject_alpha=0.5,
        max_dataset_share=0.25,
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=BATCH_SIZE,
        drop_last=True,
        collate_fn=MultiResolutionCollate(seed=0, two_view=True),
        num_workers=0,
        worker_init_fn=_seed_worker,
    )

    supervised_tokens: list[float] = []
    masked_fractions: list[float] = []
    dead_windows = 0
    source_counts: dict[str, int] = {}

    per_resolution: dict[int, list[float]] = {0: [], 1: []}
    temporal_by_source: dict[str, list[int]] = {}
    copyable_long, masked_long = 0, 0

    for batch in loader:
        _, _, _, channels = batch["patches"].shape
        valid = batch["patch_padding_mask"]
        channel_mask = batch["channel_mask"]
        rids = batch["resolution_ids"]
        plan = make_per_resolution_mask_plan(
            rids, channels, GYRO, valid_patches=valid, channel_mask=channel_mask,
        )
        real = valid.unsqueeze(2) & channel_mask.unsqueeze(1)
        supervised = plan.token_mask & real
        counts = supervised.flatten(1).sum(1)
        totals = real.flatten(1).sum(1).clamp(min=1)
        supervised_tokens.extend(counts.float().tolist())
        masked_fractions.extend((counts / totals).tolist())
        dead_windows += int(counts.eq(0).sum())

        # Realised fraction PER GRID. 'per_resolution' counts the block in tokens, so this should
        # track mask_ratio_time directly instead of the shared-interval scheme's (L+p)/W inflation.
        token_masked = supervised.any(dim=2)
        for group in (0, 1):
            sel = valid & rids.eq(group)
            n = sel.sum(dim=1)
            hit = (token_masked & sel).sum(dim=1)
            per_resolution[group].extend(
                (hit[n > 0].float() / n[n > 0].float()).tolist()
            )

        # The cost 'per_resolution' knowingly accepts: a masked LONG token whose overlapping short
        # tokens are ALL visible is close to copyable, because a coarse band summary of a
        # quasi-stationary second is nearly its own fine summary. Track it rather than assume.
        starts, ends = batch["patch_starts"], batch["patch_ends"]
        for b in range(rids.shape[0]):
            longs = torch.nonzero(valid[b] & rids[b].eq(1) & token_masked[b]).squeeze(1)
            shorts = torch.nonzero(valid[b] & rids[b].eq(0)).squeeze(1)
            for p in longs.tolist():
                masked_long += 1
                inside = shorts[(starts[b, shorts] >= starts[b, p] - 1e-6)
                                & (ends[b, shorts] <= ends[b, p] + 1e-6)]
                if inside.numel() and not bool(token_masked[b, inside].any()):
                    copyable_long += 1

        for source, row in zip(batch["sources"], token_masked):
            source_counts[source] = source_counts.get(source, 0) + 1
            temporal_by_source.setdefault(source, []).append(int(row.sum()))

        required_view_b = {
            "patches_b", "rates_b", "positions_b", "channel_mask_b", "patch_padding_mask_b"
        }
        missing = required_view_b - set(batch)
        if missing:
            raise RuntimeError(f"VICReg view is missing collate fields {sorted(missing)}")

    windows = N_BATCHES * BATCH_SIZE
    report = {
        "batches": N_BATCHES,
        "windows": windows,
        "jepa": {
            "mask_mode": "per_resolution",
            "supervised_tokens_per_window": distribution(supervised_tokens),
            "masked_fraction_of_real_tokens": distribution(masked_fractions),
            "zero_supervision_windows": dead_windows,
            "zero_supervision_fraction": round(dead_windows / windows, 4),
            "realised_masked_fraction_short": distribution(per_resolution[0]),
            "realised_masked_fraction_long": distribution(per_resolution[1]),
            # Accepted by design: cross-scale inference is the objective, not a leak. Reported so
            # the trade stays visible if it ever grows.
            "masked_long_tokens_with_all_short_visible": round(
                copyable_long / max(masked_long, 1), 4
            ),
            "mean_temporally_masked_tokens_by_source": {
                source: round(float(np.mean(values)), 2)
                for source, values in sorted(temporal_by_source.items())
            },
        },
        "vicreg": {
            "augmentation_pairs_per_batch": BATCH_SIZE,
        },
        "sampled_source_share": {
            source: round(count / windows, 4) for source, count in sorted(source_counts.items())
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
