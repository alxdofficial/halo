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

from training.tokenizer.losses_repr import make_sensor_mask_plan
from training.tokenizer.pretrain_data import (
    CorpusIndex,
    MultiScaleCollate,
    PretrainDataset,
    SEED,
    TemperatureSampler,
    _seed_worker,
    modalities_present,
)

OUT = Path(__file__).resolve().parent / "outputs" / "objective_health"
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
    index = CorpusIndex(seed=SEED)
    dataset = PretrainDataset(index, index.train, augment=True, two_view=True)
    sensor_batch_groups = [
        len(modalities_present(index.refs[key.stream_i].mask)) for key in index.train
    ]
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
        batch_group_ids=sensor_batch_groups,
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=BATCH_SIZE,
        drop_last=True,
        collate_fn=MultiScaleCollate(fixed_patch_seconds=1.0, seed=0, two_view=True),
        num_workers=0,
        worker_init_fn=_seed_worker,
    )

    supervised_tokens: list[float] = []
    masked_fractions: list[float] = []
    dead_windows = 0
    source_counts: dict[str, int] = {}

    temporal_by_source: dict[str, list[int]] = {}
    descriptor_events = 0
    config_pair_mismatches = 0
    unexpected_augmentations = 0
    ineligible_windows = 0
    planner_failures = 0

    for batch in loader:
        batch_size, _, _, _ = batch["patches"].shape
        valid = batch["patch_padding_mask"]
        n_sensors = max(len(texts) for texts in batch["sensor_texts"])
        present = torch.tensor([
            [sensor < len(texts) for sensor in range(n_sensors)]
            for texts in batch["sensor_texts"]
        ], dtype=torch.bool)
        plan = make_sensor_mask_plan(
            batch_size, valid.shape[1], n_sensors, valid_patches=valid,
            sensor_present=present, sensor_placement=batch["sensor_placement"],
            descriptor_event_p=0.0,
        )
        real = valid.unsqueeze(2) & present.unsqueeze(1)
        supervised = plan.token_mask & real
        counts = supervised.flatten(1).sum(1)
        totals = real.flatten(1).sum(1).clamp(min=1)
        eligible = real.flatten(1).sum(1) > 1
        supervised_tokens.extend(counts.float().tolist())
        masked_fractions.extend((counts / totals).tolist())
        dead_windows += int(counts.eq(0).sum())
        ineligible_windows += int((~eligible).sum())
        planner_failures += int((counts.eq(0) & eligible).sum())

        token_masked = plan.token_mask.any(dim=2)
        descriptor_events += int((plan.descriptor_mask & present).sum())

        for source, row in zip(batch["sources"], token_masked):
            source_counts[source] = source_counts.get(source, 0) + 1
            temporal_by_source.setdefault(source, []).append(int(row.sum()))

        required_view_b = {
            "patches_b", "rates_b", "positions_b", "channel_mask_b", "patch_padding_mask_b"
        }
        missing = required_view_b - set(batch)
        if missing:
            raise RuntimeError(f"VICReg view is missing collate fields {sorted(missing)}")
        config_pair_mismatches += int(not (
            torch.equal(batch["channel_mask"], batch["channel_mask_b"])
            and torch.equal(batch["sensor_placement"], batch["sensor_placement_b"])
            and torch.allclose(batch["rates"], batch["rates_b"])
            and torch.allclose(batch["source_rates"], batch["source_rates_b"])
        ))
        for traces in (batch["augmentations"], batch["augmentations_b"]):
            unexpected_augmentations += sum(bool(trace) for trace in traces)

    windows = N_BATCHES * BATCH_SIZE
    report = {
        "batches": N_BATCHES,
        "windows": windows,
        "jepa": {
            "mask_mode": "sensor_granularity",
            "supervised_tokens_per_window": distribution(supervised_tokens),
            "masked_fraction_of_real_tokens": distribution(masked_fractions),
            "zero_supervision_windows": dead_windows,
            "zero_supervision_fraction": round(dead_windows / windows, 4),
            "ineligible_one_token_windows": ineligible_windows,
            "eligible_windows_without_target": planner_failures,
            "descriptor_events": descriptor_events,
            "descriptor_events_per_window": round(descriptor_events / windows, 4),
            "mean_masked_time_positions_by_source": {
                source: round(float(np.mean(values)), 2)
                for source, values in sorted(temporal_by_source.items())
            },
        },
        "vicreg": {
            "augmentation_pairs_per_batch": BATCH_SIZE,
            "batches_with_config_mismatch_between_views": config_pair_mismatches,
            "views_with_unexpected_augmentation": unexpected_augmentations,
        },
        "sampled_source_share": {
            source: round(count / windows, 4) for source, count in sorted(source_counts.items())
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
