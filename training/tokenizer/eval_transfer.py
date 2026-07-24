"""Held-out-config transfer probe for a frozen Phase-1 encoder.

The internal pretraining val-kNN is subject-disjoint but WITHIN the training datasets.
This measures the thing we actually care about: does the frozen representation cluster
activities on **held-out eval datasets** (unseen placements/devices/subjects)? For each
eval dataset we encode its windows with the frozen encoder and run a subject-disjoint
kNN balanced accuracy over that dataset's OWN labels — pure representation transfer, no
ConSE/text head involved (that's Pipeline B).

NOT the baseline-table number (which is ConSE macro-F1) — it's a fast, honest transfer
signal to decide whether Pipeline A cleared the bar before building Pipeline B.

Run:  /home/alex/code/HALO/legacy_code/.venv/bin/python -m training.tokenizer.eval_transfer \
        --checkpoint training/tokenizer/outputs/pretrain_native/best.pt
      # NB: pretrain_native/best.pt is the real trained model (val_ba 0.659). The default
      # outputs/pretrain/ dir holds only smoke/debug runs — do NOT evaluate that one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.scripts.eda.grid_io import discover_grids
from model.tokenizer.encoder import SetTokenizerEncoder
from model.tokenizer.preprocess import gravity_align
from training.tokenizer.pretrain_data import (CHANNELS, DFT_SIZE, stream_channel_descriptions,
                                              stream_sensor_texts, _stream_gravity_state,
                                              MultiResolutionCollate, MultiScaleCollate,
                                              VAL_RESOLUTION_PAIR)

# Held-out eval datasets (never in TRAIN_DATASETS). tnda_har/ut_complex excluded
# (degenerate subject ids -> can't do subject-disjoint kNN).
EVAL_STREAMS = (
    ("motionsense", "phone_front_pocket"),
    ("realworld", "phone_waist"),
    ("shoaib", "phone_right_pocket"),
    ("inclusivehar", "phone_waist"),
)
PATCH_SECONDS = 1.0
KNN_K = 5
SEED = 20260718


def build_encoder(ckpt: dict, device) -> SetTokenizerEncoder:
    c = ckpt["config"]
    frontend = c.get("frontend", "fixed")
    kw = dict(
        d_model=c["d_model"], num_layers=c["num_layers"], num_heads=c["num_heads"],
        dim_feedforward=c["dim_feedforward"], dropout=0.0, dft_size=DFT_SIZE,
        frontend=frontend,                                  # reconstruct the ACTUAL arm (was: always filterbank)
        use_duration_embedding=c.get("multiresolution", False),
        duration_min_seconds=min(c.get("short_patch_choices", (0.4,))),
        duration_max_seconds=max(c.get("long_patch_choices", (1.5,))),
        duration_gate_init=c.get("duration_gate_init", 0.1),
        rope_min_period=0.4 if c.get("multiresolution", False) else 0.5,
        text_conditioning=c.get("text_conditioning", "per_channel"),  # reconstruct the ACTUAL arm
        gate_bias_init=c.get("gate_bias_init", -2.0),
    )
    kw.update(                                              # fixed / learnable filterbank hyperparams
        center_shift_fraction=c.get("center_shift_fraction", 0.45),
        bandwidth_factor_max=c.get("bandwidth_factor_max", 1.5),
        compression_gain_max=c.get("compression_gain_max", 2.0),
        filter_shape_min=c.get("filter_shape_min", 1.5),
        filter_shape_max=c.get("filter_shape_max", 2.5),
        adaptive_gate_init=c.get("adaptive_gate_init", 0.1),
    )
    enc = SetTokenizerEncoder(**kw)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval_resolution_pair = tuple(c.get("val_resolution_pair", VAL_RESOLUTION_PAIR))
    enc.min_resolution_ratio = float(c.get("min_resolution_ratio", 1.75))
    return enc.to(device).eval()


@torch.no_grad()
def encode_dataset(enc, data, texts, device, rate: float, gravity_state=None,
                   channel_mask=None, dataset=None, stream=None) -> torch.Tensor:
    """(N, T, 6) raw windows at the stream's NATIVE rate -> (N, d) pooled embeddings.

    ``dataset``/``stream`` are only needed for a FACTORED encoder (to build the role/sensor text);
    the default per_channel path uses the ``texts`` (per-channel descriptions) exactly as before.
    """
    collate = (
        MultiResolutionCollate(fixed_patch_seconds=enc.eval_resolution_pair,
                               min_resolution_ratio=enc.min_resolution_ratio,
                               compute_targets=False)
        if enc.use_duration_embedding else
        MultiScaleCollate(fixed_patch_seconds=PATCH_SECONDS, compute_targets=False)
    )
    enc_texts = texts
    cmask = (torch.ones(6, dtype=torch.bool) if channel_mask is None
             else torch.as_tensor(channel_mask, dtype=torch.bool))
    if cmask.shape != (6,):
        raise ValueError(f"channel_mask must have shape (6,), got {tuple(cmask.shape)}")
    factored = getattr(enc, "text_conditioning", "per_channel") == "factored"
    if factored:
        if dataset is None or stream is None:
            raise ValueError("a factored encoder needs dataset+stream to build the factored "
                             "(role/sensor) text conditioning; pass them to encode_dataset()")
        role_texts, sensor_texts, sensor_id_list = stream_sensor_texts(
            dataset,
            stream,
            gravity_removed=(None if gravity_state is None else gravity_state == "removed"),
            has_accel=bool(cmask[:3].any()),
            has_gyro=bool(cmask[3:].any()),
        )
        sensor_id_t = torch.tensor(sensor_id_list, dtype=torch.long)
    embs = []
    for start in range(0, len(data), 256):
        block = torch.tensor(np.asarray(data[start:start + 256]), dtype=torch.float32)
        items = []
        for window in block:
            item = {"data": window, "rate": rate, "texts": enc_texts, "label_id": 0,
                    "channel_mask": cmask, "gravity_state": gravity_state, "source": "eval"}
            if factored:
                item["role_texts"] = role_texts
                item["sensor_texts"] = sensor_texts
                item["sensor_id"] = sensor_id_t
            items.append(item)
        batch = collate(items)
        plen = batch["patch_len"]
        out = enc(
            batch["patches"].to(device), batch["rates"].to(device),
            plen.to(device),
            batch["role_texts"] if factored else batch["texts"],   # channel_texts = ROLE when factored
            batch["positions"].to(device),
            patch_durations=(batch["patch_durations"].to(device)
                             if "patch_durations" in batch else None),
            resolution_ids=(batch["resolution_ids"].to(device)
                            if "resolution_ids" in batch else None),
            channel_mask=batch["channel_mask"].to(device),
            patch_padding_mask=batch["patch_padding_mask"].to(device),
            sensor_texts=(batch["sensor_texts"] if factored else None),
            sensor_id=(batch["sensor_id"].to(device) if factored else None),
        )
        embs.append(out["pooled"].cpu())
    return torch.cat(embs)


def knn_balanced_acc(train_z, train_y, test_z, test_y, k=KNN_K) -> float:
    labels = sorted(set(train_y) & set(test_y))
    per_class = []
    for label in labels:
        idx = [i for i, y in enumerate(test_y) if y == label]
        hits = 0
        for i in idx:
            d = (train_z - test_z[i]).norm(dim=1)
            nn = [train_y[j] for j in d.argsort()[:k].tolist()]
            hits += max(set(nn), key=nn.count) == label
        per_class.append(hits / len(idx))
    return float(np.mean(per_class)) if per_class else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, device)
    print(f"loaded {args.checkpoint.name}: step {ckpt['step']}, "
          f"internal val_ba {ckpt['val_ba']:.3f}, git {ckpt['git']}", flush=True)

    refs = {(r.dataset, r.stream): r for r in discover_grids("native")}
    rng = np.random.default_rng(SEED)
    results = {}
    for dataset, stream in EVAL_STREAMS:
        ref = refs.get((dataset, stream))
        if ref is None:
            continue
        data = ref.load_data()
        labels = np.asarray(ref.labels)
        subjects = np.asarray(ref.subjects)
        texts = stream_channel_descriptions(dataset, stream)
        z = encode_dataset(enc, data, texts, device, ref.rate_hz,
                           _stream_gravity_state(dataset, stream),
                           channel_mask=ref.mask, dataset=dataset, stream=stream)

        # subject-disjoint 50/50 split
        subj = sorted(set(subjects.tolist()))
        rng.shuffle(subj)
        hold = set(subj[: max(1, len(subj) // 2)])
        tr = [i for i in range(len(data)) if subjects[i] not in hold]
        te = [i for i in range(len(data)) if subjects[i] in hold]
        ba = knn_balanced_acc(z[tr], labels[tr].tolist(), z[te], labels[te].tolist())
        n_lab = len(set(labels.tolist()))
        results[dataset] = {"knn_ba": round(ba, 4), "windows": len(data),
                            "labels": n_lab, "chance": round(1 / n_lab, 3)}
        print(f"  {dataset:14s} kNN-BA={ba:.3f}  ({n_lab} labels, chance {1/n_lab:.3f}, "
              f"{len(data)} windows)", flush=True)

    mean = float(np.mean([r["knn_ba"] for r in results.values()]))
    print(f"\nHELD-OUT-CONFIG TRANSFER: mean kNN-BA = {mean:.3f} across {len(results)} datasets")
    out = args.checkpoint.parent / "transfer_eval.json"
    out.write_text(json.dumps({"checkpoint": str(args.checkpoint), "step": ckpt["step"],
                               "internal_val_ba": ckpt["val_ba"], "per_dataset": results,
                               "mean_knn_ba": round(mean, 4)}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
