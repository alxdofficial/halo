"""Build the frozen archetypal memory bank for Phase-B (the evidence engine).

This is the ONE expensive pass in Phase B. We encode the training corpus a single
time with the frozen Pipeline-A encoder (the fixed+multiresolution default) and cache
the pooled window vectors + metadata to disk. Every downstream Phase-B training step
then operates purely on these cached vectors — the encoder never runs in the loop, so
episodic training is a batched matmul over a ~156 MB in-VRAM bank (see
docs/design/EVIDENCE_ENGINE.md §4.2).

Cached bank (``memory_bank.pt``):
    Z          (N, d)  float16   pooled frozen-encoder embeddings (L2-normalizable downstream)
    y          (N,)    int64     global-vocab label index (canonicalized; -1 dropped)
    subj       (N,)    int64     composite "dataset:subject" id (for subject-disjoint episodes)
    cfg        (N,)    int64     stream id "dataset/stream" (config bucket)
    vocab      list[str]         the 59-way global label vocabulary (row i == label index i)
    label_text (L, 384) float32  frozen-SBERT embedding of each vocab label (the ConSE text space)
    subj_names / cfg_names        int->string decoders for subj / cfg
    backbone   dict              provenance (ckpt path, step, val_ba, git, content fingerprint)

Memory is built from CLEAN (un-augmented) encodings — a retrieval bank of jittered
vectors would be matching against noise. Label/query augmentation is a *training-loop*
concern (the learned t-kernel), not a memory concern.

Run:
    PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
    HALO_CKPT=training/tokenizer/outputs/pretrain_fixed_mr/best.pt \
      $PY -m training.evidence.build_memory --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from data.scripts.eda.grid_io import discover_grids
from data.scripts.labels.canonical_labels import canonicalize
from eval.scoring import get_sbert_encoder
from training.tokenizer.eval_transfer import build_encoder, encode_dataset
from training.tokenizer.pretrain_data import (TRAIN_DATASETS, _stream_gravity_state,
                                              stream_channel_descriptions)

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_CKPT = _REPO / "training/tokenizer/outputs/pretrain_fixed_mr/best.pt"
_DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "memory_bank.pt"
_GLOBAL_LABELS = _REPO / "data/labels/global_labels.json"


def _load_vocab() -> list[str]:
    return list(json.loads(_GLOBAL_LABELS.read_text())["labels"])


def _backbone_fp(ckpt_path: Path) -> str:
    return hashlib.sha256(ckpt_path.read_bytes()).hexdigest() if ckpt_path.exists() else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.environ.get("HALO_CKPT", _DEFAULT_CKPT)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-per-stream", type=int, default=50000,
                    help="cap windows per stream at ENCODE time (tractable; -1 = all)")
    ap.add_argument("--max-per-label", type=int, default=8000,
                    help="cap windows per global label AFTER encoding (tames head-class hubness; "
                         "rare classes kept in full; -1 = no cap)")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cap = None if args.max_per_stream is not None and args.max_per_stream < 0 else args.max_per_stream
    label_cap = None if args.max_per_label is not None and args.max_per_label < 0 else args.max_per_label

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"encoder checkpoint missing at {args.checkpoint}. Point --checkpoint / HALO_CKPT "
            "at the frozen Phase-A run (default: the fixed+MR winner pretrain_fixed_mr/best.pt).")
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, device)
    for p in enc.parameters():
        p.requires_grad_(False)
    d_model = int(ckpt["config"]["d_model"])
    print(f"[memory] encoder {args.checkpoint.name}: step {ckpt['step']}, val_ba "
          f"{ckpt['val_ba']:.3f}, git {ckpt['git']}, frontend={ckpt['config'].get('frontend')}, "
          f"MR={ckpt['config'].get('multiresolution')}, d={d_model}", flush=True)

    vocab = _load_vocab()
    label_to_idx = {l: i for i, l in enumerate(vocab)}
    rng = np.random.RandomState(20260720)

    refs = sorted((r for r in discover_grids("native") if r.dataset in TRAIN_DATASETS),
                  key=lambda r: r.key)
    Z_parts, y_parts, subj_parts, cfg_parts = [], [], [], []
    subj_names: dict[str, int] = {}
    cfg_names: dict[str, int] = {}

    for ref in refs:
        gl = np.array([label_to_idx.get(canonicalize(l), -1) for l in ref.labels])
        keep = np.where(gl >= 0)[0]
        if keep.size == 0:
            print(f"[memory]   {ref.key}: 0 in-vocab windows, skipped", flush=True)
            continue
        if cap is not None and keep.size > cap:
            keep = np.sort(rng.choice(keep, cap, replace=False))
        data = np.asarray(ref.load_data()[keep])
        texts = stream_channel_descriptions(ref.dataset, ref.stream)
        gs = _stream_gravity_state(ref.dataset, ref.stream)
        z = encode_dataset(enc, data, texts, device, float(ref.rate_hz), gs)   # (n, d) cpu
        cfg_id = cfg_names.setdefault(ref.key, len(cfg_names))
        subj_arr = np.asarray(ref.subjects)[keep]
        s_ids = np.array([subj_names.setdefault(f"{ref.dataset}:{s}", len(subj_names))
                          for s in subj_arr], dtype=np.int64)
        Z_parts.append(z.to(torch.float16))
        y_parts.append(torch.from_numpy(gl[keep].astype(np.int64)))
        subj_parts.append(torch.from_numpy(s_ids))
        cfg_parts.append(torch.full((keep.size,), cfg_id, dtype=torch.int64))
        print(f"[memory]   {ref.key}: {keep.size} windows, {len(set(subj_arr))} subjects", flush=True)

    Z = torch.cat(Z_parts)
    y = torch.cat(y_parts)
    subj = torch.cat(subj_parts)
    cfg = torch.cat(cfg_parts)

    # Per-label balance: cap each global label at `label_cap` (rare classes kept in full).
    # A large but head-class-flooded bank hurts non-parametric retrieval via hubness, so we
    # trim the common activities rather than curate — the MVP keeps a big, roughly-balanced bank.
    if label_cap is not None:
        keep_mask = torch.zeros(len(y), dtype=torch.bool)
        for c in y.unique().tolist():
            idx = torch.nonzero(y == c, as_tuple=True)[0]
            if len(idx) > label_cap:
                idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(c))[:label_cap]]
            keep_mask[idx] = True
        before = len(y)
        Z, y, subj, cfg = Z[keep_mask], y[keep_mask], subj[keep_mask], cfg[keep_mask]
        print(f"[memory] per-label cap {label_cap}: {before} -> {len(y)} windows", flush=True)

    sbert = get_sbert_encoder()
    label_text = torch.from_numpy(sbert(vocab).astype(np.float32))   # (L, 384) L2-normalized

    n_per_label = np.bincount(y.numpy(), minlength=len(vocab))
    print(f"[memory] bank: {Z.shape[0]} windows · d={Z.shape[1]} · {len(subj_names)} subjects · "
          f"{len(cfg_names)} configs · {int((n_per_label > 0).sum())}/{len(vocab)} labels present",
          flush=True)
    print(f"[memory] size: Z={Z.numel() * 2 / 1e6:.0f} MB (fp16)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "Z": Z, "y": y, "subj": subj, "cfg": cfg,
        "vocab": vocab, "label_text": label_text,
        "subj_names": {v: k for k, v in subj_names.items()},
        "cfg_names": {v: k for k, v in cfg_names.items()},
        "d_model": d_model,
        "backbone": {"checkpoint": str(args.checkpoint), "step": int(ckpt["step"]),
                     "val_ba": float(ckpt["val_ba"]), "git": ckpt["git"],
                     "fingerprint": _backbone_fp(args.checkpoint),
                     "frontend": ckpt["config"].get("frontend"),
                     "multiresolution": ckpt["config"].get("multiresolution")},
        "max_per_stream": cap,
    }, str(args.out))
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
