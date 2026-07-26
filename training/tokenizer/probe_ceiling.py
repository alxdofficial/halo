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
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data.scripts.curate import deployment_policy as policy
from eval import data as eval_data
from eval import scoring
from eval.protocol import protocol_fingerprint, protocol_mismatch
from eval.run_baselines import RESULTS_DIR, result_path
from training.tokenizer.eval_transfer import build_encoder, encode_dataset
from training.tokenizer.pretrain_data import _stream_gravity_state, stream_channel_descriptions

FIT_EPOCHS = 300
FIT_BATCH = 512
FIT_LR = 1e-3
SEED = 20260720


def _encode(enc, windows, texts, rate, gravity_state, channel_mask, device,
            dataset, stream) -> np.ndarray:
    z = encode_dataset(enc, np.asarray(windows), texts, device, float(rate), gravity_state,
                       channel_mask=channel_mask, dataset=dataset, stream=stream)
    return z.numpy().astype(np.float32)


def _fit_probe(Xtr, ytr, Xval, yval, Xtest, n_classes, device, rng) -> np.ndarray:
    """Select a linear probe on subject-disjoint validation subjects, then score test subjects."""
    head = nn.Linear(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=FIT_LR)
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xtr).float()
    yt = torch.from_numpy(ytr).long()
    Xv = torch.from_numpy(Xval).float().to(device)
    n = len(Xt)
    best_ba = -float("inf")
    best_state = None
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
            val_pred = head(Xv).argmax(1).cpu().numpy()
        val_ba = scoring.classification_metrics(yval.tolist(), val_pred.tolist())[
            "balanced_accuracy"
        ]
        if val_ba > best_ba:
            best_ba = val_ba
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    if best_state is None:
        raise RuntimeError("linear ceiling probe never produced a validation checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        preds = head(torch.from_numpy(Xtest).float().to(device)).argmax(1).cpu().numpy()
    return preds


def _load_current_zeroshot(results_dir: Path, baseline: str, dataset: str, stream: str) -> float:
    path = result_path(results_dir, baseline, dataset, stream)
    if not path.exists():
        raise FileNotFoundError(f"missing current zero-shot cell: {path}")
    cell = json.loads(path.read_text())
    mismatch = protocol_mismatch(cell.get("_protocol"))
    if mismatch:
        raise RuntimeError(f"{path}: {mismatch}")
    if cell.get("_status") != "complete":
        raise RuntimeError(f"{path}: expected _status='complete', got {cell.get('_status')!r}")
    try:
        return float(cell["metrics"]["f1_macro"])
    except KeyError as exc:
        raise RuntimeError(f"{path}: complete cell has no metrics.f1_macro") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    started = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("training/tokenizer/outputs/pretrain_native/best.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--baseline", default="halo",
                    help="zero-shot result row to compare against (default: halo)")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--allow-missing-zeroshot", action="store_true",
                    help="write supervised ceilings with null zero-shot gaps when current result "
                         "cells are missing; default is fail-loud")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not args.checkpoint.exists():
        ap.error(f"checkpoint not found: {args.checkpoint}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, device)
    print(f"loaded {args.checkpoint.name}: step {ckpt['step']}, val_ba {ckpt['val_ba']:.3f}, "
          f"git {ckpt['git']}", flush=True)

    rng = np.random.RandomState(SEED)
    rows, ceils, paired_gaps = [], [], []
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

        ti, vi, te = scoring.subject_disjoint_split(subj, seed=SEED)
        preds_idx = _fit_probe(
            X[ti], y[ti], X[vi], y[vi], X[te], len(labels), device, rng
        )
        i2l = {i: l for l, i in l2i.items()}
        pred_names = [i2l[p] for p in preds_idx]
        gt_names = [i2l[t] for t in y[te]]
        ceiling = scoring.classification_metrics(gt_names, pred_names)["f1_macro"]

        try:
            zs = _load_current_zeroshot(args.results_dir, args.baseline, ds, stream)
        except (FileNotFoundError, RuntimeError) as exc:
            if not args.allow_missing_zeroshot:
                raise SystemExit(str(exc)) from exc
            print(f"  [warn] {exc}", flush=True)
            zs = None
        gap = ceiling - zs if zs is not None else None
        rows.append((ds, stream, ceiling, zs, gap, len(labels), int(len(ti)), int(len(vi)),
                     int(len(te))))
        ceils.append(ceiling)
        if zs is not None:
            paired_gaps.append(gap)
        zs_text = f"{zs:5.1f}" if zs is not None else "  n/a"
        gap_text = f"{gap:+5.1f}" if gap is not None else "  n/a"
        print(f"  {ds:14s} ceiling(sup)={ceiling:5.1f}  zs({args.baseline})={zs_text}  "
              f"gap={gap_text}  ({len(labels)} labels, {len(ti)}tr/{len(vi)}va/{len(te)}te)",
              flush=True)

    mc = float(np.mean(ceils))
    paired_zs = [row[3] for row in rows if row[3] is not None]
    mz = float(np.mean(paired_zs)) if paired_zs else None
    mg = float(np.mean(paired_gaps)) if paired_gaps else None
    print(f"\nMEAN  ceiling={mc:.1f}  zs={mz if mz is not None else 'n/a'}  "
          f"gap={mg if mg is not None else 'n/a'}", flush=True)
    out = args.out or args.checkpoint.parent / "ceiling_probe.json"
    payload = {
        "schema_version": 2,
        "protocol": protocol_fingerprint(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "step": ckpt["step"],
        "zero_shot_baseline": args.baseline,
        "results_dir": str(args.results_dir),
        "per_dataset": [{"dataset": d, "ceiling_supervised_f1": round(c, 2),
                         "stream": stream,
                         "zeroshot_f1": z,
                         "gap": round(g, 2) if g is not None else None,
                         "n_labels": nl, "n_train": ntr, "n_val": nva, "n_test": nte}
                        for d, stream, c, z, g, nl, ntr, nva, nte in rows],
        "mean_ceiling": round(mc, 2),
        "mean_zeroshot": round(mz, 2) if mz is not None else None,
        "mean_gap": round(mg, 2) if mg is not None else None,
        "elapsed_s": round(time.time() - started, 3),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2))
    partial.replace(out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
