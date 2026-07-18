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
        --checkpoint training/tokenizer/outputs/pretrain/best.pt
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
from training.tokenizer.pretrain_data import CHANNELS, DFT_SIZE, stream_channel_descriptions

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
    enc = SetTokenizerEncoder(
        d_model=c["d_model"], num_layers=c["num_layers"], num_heads=c["num_heads"],
        dim_feedforward=c["dim_feedforward"], dropout=0.0, dft_size=DFT_SIZE,
    )
    enc.load_state_dict(ckpt["encoder"])
    return enc.to(device).eval()


@torch.no_grad()
def encode_dataset(enc, data, texts, device, rate: float) -> torch.Tensor:
    """(N, T, 6) raw windows at the stream's NATIVE rate -> (N, d) pooled embeddings."""
    n = int(round(rate * PATCH_SECONDS))
    P = max(1, data.shape[1] // n)
    embs = []
    for start in range(0, len(data), 256):
        block = torch.tensor(np.asarray(data[start:start + 256]), dtype=torch.float32)
        aligned, _, _ = gravity_align(block, list(CHANNELS), rate)
        B = aligned.shape[0]
        patches = torch.zeros(B, P, DFT_SIZE, 6)
        for p in range(P):
            patches[:, p, :n] = aligned[:, p * n:(p + 1) * n]
        positions = (torch.arange(P).float() * PATCH_SECONDS + PATCH_SECONDS / 2)
        positions = positions.unsqueeze(0).expand(B, P).contiguous()
        out = enc(patches.float().to(device), rate, torch.tensor([n] * B).to(device),
                  [texts] * B, positions.to(device))
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
        z = encode_dataset(enc, data, texts, device, ref.rate_hz)

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
