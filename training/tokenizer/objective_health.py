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

from training.tokenizer.losses_repr import make_multiresolution_mask_plan
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

    for batch in loader:
        _, _, _, channels = batch["patches"].shape
        valid = batch["patch_padding_mask"]
        channel_mask = batch["channel_mask"]
        plan = make_multiresolution_mask_plan(
            batch["patch_starts"], batch["patch_ends"], batch["resolution_ids"],
            channels, GYRO, valid_patches=valid, channel_mask=channel_mask,
        )
        real = valid.unsqueeze(2) & channel_mask.unsqueeze(1)
        supervised = plan.token_mask & real
        counts = supervised.flatten(1).sum(1)
        totals = real.flatten(1).sum(1).clamp(min=1)
        supervised_tokens.extend(counts.float().tolist())
        masked_fractions.extend((counts / totals).tolist())
        dead_windows += int(counts.eq(0).sum())

        for source in batch["sources"]:
            source_counts[source] = source_counts.get(source, 0) + 1

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
            "supervised_tokens_per_window": distribution(supervised_tokens),
            "masked_fraction_of_real_tokens": distribution(masked_fractions),
            "zero_supervision_windows": dead_windows,
            "zero_supervision_fraction": round(dead_windows / windows, 4),
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
