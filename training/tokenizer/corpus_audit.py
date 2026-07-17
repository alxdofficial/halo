"""Pre-training corpus + task-distribution audit (run before the first Phase-1 run).

Answers, with numbers: how much data, does it load, unique examples, label/dataset
imbalances, the realized distribution of the pretraining tasks (augmentation firing,
mask ratios, patch-scale draws), and A3 target validity rates (including by label —
cadence is structurally locomotion-only).

Run:  /home/alex/code/HALO/legacy_code/.venv/bin/python -m training.tokenizer.corpus_audit
"""

from __future__ import annotations

import json
import random as stdlib_random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from training.tokenizer.losses_repr import make_mask_plan
from training.tokenizer.pretrain_data import (
    CHANNELS,
    PATCH_SECONDS_CHOICES,
    BalancedBatchSampler,
    CorpusIndex,
    MultiScaleCollate,
    PretrainDataset,
)

OUT = Path(__file__).resolve().parent / "outputs" / "corpus_audit"
SAMPLE_ITEMS = 800          # augmented items drawn for the task-distribution stats
SEED = 7


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    np.random.seed(SEED)
    stdlib_random.seed(SEED)
    torch.manual_seed(SEED)

    # ---------------------------------------------------------------- corpus volume
    index = CorpusIndex()      # the real training config (20k/stream cap)
    per_dataset = defaultdict(lambda: {"streams": 0, "windows_total": 0,
                                       "windows_used": 0, "subjects": set()})
    used_by_stream = Counter()
    for k in index.train + index.val:
        used_by_stream[k.stream_i] += 1
    for i, ref in enumerate(index.refs):
        d = per_dataset[ref.dataset]
        d["streams"] += 1
        d["windows_total"] += ref.n_windows
        d["windows_used"] += used_by_stream[i]
        d["subjects"].update(ref.subjects)
    report["per_dataset"] = {
        ds: {**{k: v for k, v in d.items() if k != "subjects"},
             "subjects": len(d["subjects"])}
        for ds, d in sorted(per_dataset.items())
    }
    report["totals"] = {
        "streams": len(index.refs),
        "windows_available": sum(r.n_windows for r in index.refs),
        "windows_used": len(index.train) + len(index.val),
        "train": len(index.train), "val": len(index.val),
        "labels": len(index.label_ids),
    }

    # ------------------------------------------------------------- label imbalance
    train_labels = Counter(k.label_id for k in index.train)
    id_to_label = {v: k for k, v in index.label_ids.items()}
    hist = {id_to_label[i]: c for i, c in train_labels.most_common()}
    counts = np.array(list(hist.values()))
    report["label_distribution"] = {
        "histogram": hist,
        "top5_share": round(float(counts[:5].sum() / counts.sum()), 3),
        "max_over_min": int(counts.max() / max(counts.min(), 1)),
        "labels_below_8_windows": [l for l, c in hist.items() if c < 8],
        "labels_below_64_windows": sum(1 for c in counts if c < 64),
    }
    # dataset share of the used corpus
    ds_share = Counter()
    for k in index.train:
        ds_share[index.refs[k.stream_i].dataset] += 1
    total_train = sum(ds_share.values())
    report["dataset_share_of_train"] = {
        d: round(c / total_train, 3) for d, c in ds_share.most_common()
    }

    # -------------------------------------------------- loading + integrity check
    load_stats = {"nan_windows": 0, "inf_windows": 0, "checked": 0,
                  "acc_median_mag_by_stream": {}}
    rng = np.random.default_rng(SEED)
    for i, ref in enumerate(index.refs):
        data = ref.load_data()
        sel = rng.choice(ref.n_windows, size=min(50, ref.n_windows), replace=False)
        block = np.asarray(data[np.sort(sel)], dtype=np.float32)
        load_stats["checked"] += len(sel)
        load_stats["nan_windows"] += int(np.isnan(block).any(axis=(1, 2)).sum())
        load_stats["inf_windows"] += int(np.isinf(block).any(axis=(1, 2)).sum())
        mag = np.linalg.norm(block[:, :, :3], axis=2)
        load_stats["acc_median_mag_by_stream"][f"{ref.dataset}/{ref.stream}"] = \
            round(float(np.median(mag)), 3)
    report["loading"] = load_stats

    # -------------------------------------------- realized pretraining-task stats
    ds_train = PretrainDataset(index, index.train, augment=True)
    picks = rng.choice(len(index.train), size=SAMPLE_ITEMS, replace=False)
    stats = {"rate_changed": 0, "channels_dropped": 0, "gravity_absent": 0,
             "cadence_valid": 0, "eigen_valid": 0, "coherence_valid": 0,
             "rates": [], "lengths": []}
    cadence_valid_by_label = Counter()
    label_seen = Counter()
    from model.tokenizer.primitives import compute_primitives
    for pi in picks:
        item = ds_train[int(pi)]
        label = id_to_label[item["label_id"]]
        label_seen[label] += 1
        stats["rates"].append(item["rate"])
        stats["lengths"].append(item["data"].shape[0])
        if abs(item["rate"] - 60.0) > 0.1:
            stats["rate_changed"] += 1
        if int(item["channel_mask"].sum()) < 6:
            stats["channels_dropped"] += 1
        prims = compute_primitives(item["data"].unsqueeze(0), CHANNELS, item["rate"])
        if not prims["dc_tilt"].valid[0]:
            stats["gravity_absent"] += 1
        if prims["cadence"].valid[0]:
            stats["cadence_valid"] += 1
            cadence_valid_by_label[label] += 1
        stats["eigen_valid"] += int(prims["eigen_ratios"].valid[0])
        stats["coherence_valid"] += int(prims["coherence"].valid[0])
    n = SAMPLE_ITEMS
    report["task_distribution"] = {
        "sampled_items": n,
        "rate_changed_frac": round(stats["rate_changed"] / n, 3),
        "rate_range_hz": [round(min(stats["rates"]), 1), round(max(stats["rates"]), 1)],
        "channels_dropped_frac": round(stats["channels_dropped"] / n, 3),
        "gravity_absent_frac": round(stats["gravity_absent"] / n, 3),
        "cadence_valid_frac": round(stats["cadence_valid"] / n, 3),
        "eigen_valid_frac": round(stats["eigen_valid"] / n, 3),
        "coherence_valid_frac": round(stats["coherence_valid"] / n, 3),
        "cadence_valid_by_label": {
            l: f"{cadence_valid_by_label[l]}/{label_seen[l]}"
            for l, _ in label_seen.most_common(15)
        },
    }

    # mask-plan realized ratios (the A1 task distribution)
    ratios, gyro_drops, causal = [], 0, 0
    for t_count in (3, 4, 6, 8, 12):
        plan = make_mask_plan(512, t_count, 6, [3, 4, 5],
                              generator=torch.Generator().manual_seed(1))
        ratios.append(round(float(plan.token_mask.float().mean()), 3))
        full = plan.token_mask.all(dim=1)
        gyro_drops += int((full[:, 3:].sum(dim=1) == 3).sum())
    report["mask_plan"] = {
        "realized_ratio_by_T[3,4,6,8,12]": ratios,
        "gyro_triad_drop_frac": round(gyro_drops / (512 * 5), 3),
        "patch_seconds_choices": list(PATCH_SECONDS_CHOICES),
    }

    # balanced-sampler coverage: how often each label anchors a batch
    sampler = BalancedBatchSampler(index.train, 32, 8, steps_per_epoch=200, seed=3)
    anchored = Counter()
    for batch in sampler:
        for lab in {index.train[i].label_id for i in batch}:
            anchored[id_to_label[lab]] += 1
    report["sampler"] = {
        "labels_never_anchored_in_200_steps":
            [l for l in hist if anchored[l] == 0],
        "min_anchor_count": min(anchored.values()) if anchored else 0,
        "oversampled_labels(<8 windows, drawn with replacement)":
            report["label_distribution"]["labels_below_8_windows"],
    }

    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"-> {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
