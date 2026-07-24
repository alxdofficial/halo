"""Phase-A representation CEILING probe.

Decomposes the ZS-XD gap: for each held-out eval dataset, fit a SUPERVISED linear probe on
that dataset's OWN labels (subject-disjoint train/test on FROZEN HALO features) and compare its
macro-F1 to the zero-shot ConSE macro-F1 from the baseline table.

  * supervised probe HIGH, ConSE LOW  -> the representation CAN express the classes; the loss is
    in the zero-shot text bridge -> Phase B (better retrieval/grounding) has real headroom.
  * supervised probe ALSO LOW          -> the frozen representation cannot separate the classes
    in-distribution; no Phase-B decoder recovers it -> fix Phase A / grounding, not the head.

Same frozen encoder, same subject-disjoint discipline, same macro-F1 estimand as the table, so the
two numbers are directly comparable. This is an in-distribution UPPER BOUND on what a linear head on
these features can do — not a leaderboard number.

Run:  /home/alex/code/HALO/legacy_code/.venv/bin/python -m training.tokenizer.probe_ceiling \
        --checkpoint training/tokenizer/outputs/pretrain_native/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data.scripts.curate import deployment_policy as policy
from eval import data as eval_data
from eval import scoring
from training.tokenizer.eval_transfer import build_encoder, encode_dataset
from training.tokenizer.pretrain_data import _stream_gravity_state, stream_channel_descriptions

# Zero-shot ConSE macro-F1 from docs/baselines/RESULTS_V2.md (HALO row) for the side-by-side gap.
ZS_CONSE_F1 = {
    "motionsense": 54.0, "realworld": 43.0, "shoaib": 44.8, "inclusivehar": 26.7,
    "usc_had": 17.0, "tnda_har": 45.2, "ut_complex": 52.1,
}

FIT_EPOCHS = 300
FIT_BATCH = 512
FIT_LR = 1e-3
SEED = 20260720


def _encode(enc, windows, texts, rate, gravity_state, channel_mask, device,
            dataset, stream) -> np.ndarray:
    z = encode_dataset(enc, np.asarray(windows), texts, device, float(rate), gravity_state,
                       channel_mask=channel_mask, dataset=dataset, stream=stream)
    return z.numpy().astype(np.float32)


def _fit_probe(Xtr, ytr, Xte, n_classes, device, rng) -> np.ndarray:
    """Train a linear softmax probe to convergence on the train fold; return test-fold argmax preds."""
    head = nn.Linear(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=FIT_LR)
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xtr).float()
    yt = torch.from_numpy(ytr).long()
    n = len(Xt)
    for _ in range(FIT_EPOCHS):
        head.train()
        perm = rng.permutation(n)
        for s in range(0, n, FIT_BATCH):
            bi = perm[s:s + FIT_BATCH]
            opt.zero_grad()
            loss = crit(head(Xt[bi].to(device)), yt[bi].to(device))
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        preds = head(torch.from_numpy(Xte).float().to(device)).argmax(1).cpu().numpy()
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("training/tokenizer/outputs/pretrain_native/best.pt"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, device)
    print(f"loaded {args.checkpoint.name}: step {ckpt['step']}, val_ba {ckpt['val_ba']:.3f}, "
          f"git {ckpt['git']}", flush=True)

    rng = np.random.RandomState(SEED)
    rows, ceils, zss = [], [], []
    for ds in policy.PRIMARY_EVAL_DATASETS:
        specs = policy.stream_specs(ds, "primary")
        if not specs:
            continue
        stream = specs[0].stream_id
        s = eval_data.load_eval_stream(ds, stream, alignment="non_harmonised")
        gt, subjects, keep = scoring.filter_ground_truth(s.gt, s.subjects, s.eval_labels)
        if len(keep) == 0:
            continue
        windows = np.asarray(s.windows)[keep]
        texts = stream_channel_descriptions(ds, stream)
        gs = _stream_gravity_state(ds, stream)
        X = _encode(enc, windows, texts, s.rate_hz, gs, s.mask, device, ds, stream)

        subj = np.asarray(subjects)
        if len(set(subj.tolist())) < 3:
            print(f"  {ds:14s} SKIP (only {len(set(subj.tolist()))} subject(s) — "
                  f"can't subject-disjoint split; degenerate sentinel)", flush=True)
            continue

        labels = sorted(set(gt))
        l2i = {l: i for i, l in enumerate(labels)}
        y = np.array([l2i[g] for g in gt])

        ti, vi, _ = scoring.subject_disjoint_split(subj, seed=SEED)
        preds_idx = _fit_probe(X[ti], y[ti], X[vi], len(labels), device, rng)
        i2l = {i: l for l, i in l2i.items()}
        pred_names = [i2l[p] for p in preds_idx]
        gt_names = [i2l[t] for t in y[vi]]
        ceiling = scoring.classification_metrics(gt_names, pred_names)["f1_macro"]

        zs = ZS_CONSE_F1.get(ds, float("nan"))
        gap = ceiling - zs
        rows.append((ds, ceiling, zs, gap, len(labels), int(len(ti)), int(len(vi))))
        ceils.append(ceiling); zss.append(zs)
        print(f"  {ds:14s} ceiling(sup)={ceiling:5.1f}  zs(conse)={zs:5.1f}  gap={gap:+5.1f}  "
              f"({len(labels)} labels, {len(ti)}tr/{len(vi)}te)", flush=True)

    mc, mz = float(np.mean(ceils)), float(np.nanmean(zss))
    print(f"\nMEAN  ceiling={mc:.1f}  zs={mz:.1f}  gap={mc-mz:+.1f}", flush=True)
    out = args.checkpoint.parent / "ceiling_probe.json"
    out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "step": ckpt["step"],
        "per_dataset": [{"dataset": d, "ceiling_supervised_f1": round(c, 2),
                         "zeroshot_conse_f1": z, "gap": round(g, 2),
                         "n_labels": nl, "n_train": ntr, "n_test": nte}
                        for d, c, z, g, nl, ntr, nte in rows],
        "mean_ceiling": round(mc, 2), "mean_zeroshot": round(mz, 2), "mean_gap": round(mc - mz, 2),
    }, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
