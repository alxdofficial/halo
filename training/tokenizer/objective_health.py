"""Per-objective 'is the model getting healthy data?' report (run before training).

For each elite-3 objective, report the distribution (min / p10 / median / p90 / max +
the fraction of degenerate cases) of the stats that determine whether the objective has
real supervision on each window/batch. The ones that matter, and why:

  A1 (masked latent prediction)
    - T (patches/window)              — the multi-scale axis; too-small T = weak temporal task
    - real patches/window             — phantom (rate-shortened) patches must be excluded
    - real channels/window            — 6 (full IMU) vs 3 (accel-only)
    - **A1 supervised tokens/window** — masked AND real-channel AND real-patch. THE stat:
                                        windows at 0 get NO A1 gradient (dead supervision)
    - A1 masked fraction of real tokens — should sit near the target ratio (~0.5)

  A2 (config-conditional SupCon)
    - **positives per anchor**        — same-label peers in the batch; 0 = anchor unused
    - valid anchors / batch           — anchors with >=1 positive (drives the loss)
    - distinct labels / batch

  A3 (physical-primitive grounding)
    - cadence-valid windows / batch   — cadence is structurally locomotion-only; low is OK
                                        but 0-heavy batches give A3 nothing to ground
    - eigen-valid windows / batch

Run:  /home/alex/code/HALO/legacy_code/.venv/bin/python -m training.tokenizer.objective_health
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from training.tokenizer.losses_repr import make_mask_plan
from training.tokenizer.pretrain_data import (
    BalancedBatchSampler, CorpusIndex, MultiScaleCollate, PretrainDataset, _seed_worker,
)
from torch.utils.data import DataLoader

OUT = Path(__file__).resolve().parent / "outputs" / "objective_health"
GYRO_IDX = [3, 4, 5]
N_BATCHES = 60
BATCH_CLASSES, BATCH_PER_CLASS = 32, 8


def dist(x, pct=(0, 10, 50, 90, 100)) -> dict:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size),
            **{f"p{p}": round(float(np.percentile(a, p)), 3) for p in pct},
            "mean": round(float(a.mean()), 3)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    index = CorpusIndex(seed=0)
    ds = PretrainDataset(index, index.train, augment=True)
    loader = DataLoader(
        ds,
        batch_sampler=BalancedBatchSampler(index.train, BATCH_CLASSES, BATCH_PER_CLASS,
                                           N_BATCHES, seed=0),
        collate_fn=MultiScaleCollate(seed=0), num_workers=4,
        worker_init_fn=_seed_worker, persistent_workers=True,
    )

    # per-window accumulators
    T_win, real_patch, real_chan, a1_tokens, a1_frac = [], [], [], [], []
    pos_per_anchor = []
    # per-batch accumulators
    ps_draws, valid_anchors, distinct_labels = [], [], []
    cad_valid, eig_valid = [], []
    a1_dead_windows = 0
    total_windows = 0

    for batch in loader:
        B, P, _, C = batch["patches"].shape
        pad = batch["patch_padding_mask"]                 # (B,P)
        cmask = batch["channel_mask"]                     # (B,C)
        labels = batch["labels"]
        plan = make_mask_plan(B, P, C, GYRO_IDX, generator=torch.Generator().manual_seed(B),
                              valid_patches=pad, channel_mask=cmask)
        real_tok = pad.unsqueeze(2) & cmask.unsqueeze(1)          # (B,P,C) real tokens
        a1m = plan.token_mask & real_tok                          # supervised A1 tokens

        ps_draws.append(batch["patch_seconds"])
        cad_valid.append(int(batch["cadence_valid"].sum()))
        eig_valid.append(int(batch["eigen_valid"].sum()))
        distinct_labels.append(int(labels.unique().numel()))

        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        same.fill_diagonal_(False)
        pos_counts = same.sum(dim=1)
        pos_per_anchor.extend(pos_counts.tolist())
        valid_anchors.append(int((pos_counts > 0).sum()))

        for b in range(B):
            total_windows += 1
            T_win.append(P)
            real_patch.append(int(pad[b].sum()))
            real_chan.append(int(cmask[b].sum()))
            nt = int(a1m[b].sum())
            a1_tokens.append(nt)
            denom = int(real_tok[b].sum())
            a1_frac.append(nt / denom if denom else 0.0)
            if nt == 0:
                a1_dead_windows += 1

    report = {
        "batches": N_BATCHES, "windows": total_windows,
        "batch_size": BATCH_CLASSES * BATCH_PER_CLASS,
        "patch_seconds_draws": sorted(set(ps_draws)),
        "A1_masked_latent": {
            "patches_per_window(T)": dist(T_win),
            "real_patches_per_window": dist(real_patch),
            "real_channels_per_window": dist(real_chan),
            "supervised_tokens_per_window": dist(a1_tokens),
            "masked_fraction_of_real_tokens": dist(a1_frac),
            "windows_with_ZERO_supervision": a1_dead_windows,
            "windows_with_ZERO_supervision_frac": round(a1_dead_windows / total_windows, 4),
        },
        "A2_supcon": {
            "positives_per_anchor": dist(pos_per_anchor),
            "valid_anchors_per_batch(of 256)": dist(valid_anchors),
            "distinct_labels_per_batch": dist(distinct_labels),
            "anchors_with_ZERO_positives_frac":
                round(float(np.mean(np.asarray(pos_per_anchor) == 0)), 4),
        },
        "A3_grounding": {
            "cadence_valid_per_batch(of 256)": dist(cad_valid),
            "eigen_valid_per_batch(of 256)": dist(eig_valid),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # health verdict
    a1 = report["A1_masked_latent"]; a2 = report["A2_supcon"]
    checks = {
        "A1: <1% windows with zero supervision":
            a1["windows_with_ZERO_supervision_frac"] < 0.01,
        "A1: median supervised tokens >= 5":
            a1["supervised_tokens_per_window"]["p50"] >= 5,
        "A1: masked fraction in [0.3, 0.7]":
            0.3 <= a1["masked_fraction_of_real_tokens"]["p50"] <= 0.7,
        "A2: <1% anchors with zero positives":
            a2["anchors_with_ZERO_positives_frac"] < 0.01,
        "A2: median positives per anchor >= samples_per_class-1":
            a2["positives_per_anchor"]["p50"] >= BATCH_PER_CLASS - 1,
        "A3: every batch has some cadence target":
            report["A3_grounding"]["cadence_valid_per_batch(of 256)"]["p0"] > 0,
        "A3: every batch has some eigen target":
            report["A3_grounding"]["eigen_valid_per_batch(of 256)"]["p0"] > 0,
    }
    print("\nHEALTH CHECKS:")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nOBJECTIVE HEALTH: {'PASS' if all(checks.values()) else 'ISSUES'}")


if __name__ == "__main__":
    main()
