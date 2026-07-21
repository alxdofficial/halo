"""Tier-1 sweep: squeeze the UNTRAINED retrieval mechanism (raw features + raw SBERT bridge),
which already beats ConSE (45.3 vs 42.7), with no learning — so no overfitting risk.

Knobs (all applied to the raw-feature / raw-text mechanism; g,t are NOT used):
  - top_k        : hard neighbor cutoff before softmax (fixes the effective-k~2200 diffuseness)
  - tau          : retrieval softmax temperature
  - csls         : hubness correction — subtract each memory entry's centrality r_mem(j) (Gini 0.81)
  - text_ens     : average E paraphrase/template variants per label (robust text anchor), mem + cand

Each eval dataset is encoded ONCE; the similarity matrix S = z·Zᵀ is computed once per dataset and
every config is a cheap re-weighting of it. Sweep on a query subsample for speed, then re-score the
single best config on the FULL queries. No training.

Run:
    PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
    HALO_CKPT=training/tokenizer/outputs/pretrain_fixed_mr/best.pt $PY -m training.evidence.tier1_sweep --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.scripts.curate import deployment_policy as policy
from eval.data import load_eval_stream
from eval.scoring import classification_metrics, filter_ground_truth, get_sbert_encoder
# F1 fix: use the train_only-gated implementation, NOT train_head's, which merges the
# eval datasets' hand-authored synonym tables (motionsense/realworld/shoaib).
from training.evidence.labeltext import global_label_paraphrases
from training.tokenizer.eval_transfer import build_encoder, encode_dataset
from training.tokenizer.pretrain_data import _stream_gravity_state, stream_channel_descriptions

_REPO = Path(__file__).resolve().parents[2]
_DIR = Path(__file__).resolve().parent / "outputs"

TOP_KS = [0, 200, 50]          # 0 = full-soft (no cutoff)
TAUS = [0.03, 0.05, 0.08]
CSLS = [False, True]
TEXT_ENS = [False, True]
ENS_E = 8                      # paraphrase variants per label when text_ens=True


def ensemble_text(labels, sbert, E, seed=0):
    """(L, 384) L2-normalized: mean SBERT over E template/synonym variants per label."""
    import random as _random
    syn, templates = global_label_paraphrases(train_only=True)
    rng = _random.Random(seed)
    acc = np.zeros((len(labels), 384), dtype=np.float32)
    for e in range(E):
        if e == 0:
            row = [l.replace("_", " ") for l in labels]
        else:
            row = [rng.choice(templates).format(rng.choice(syn.get(l, [l.replace("_", " ")])))
                   for l in labels]
        acc += sbert(row).astype(np.float32)
    v = torch.from_numpy(acc)
    return F.normalize(v, dim=-1)


@torch.no_grad()
def eval_config(S, r_mem, K, eval_labels, gt, subjects, top_k, tau, csls):
    """Predict + macro-F1 for one config, given precomputed S (Nq,N), K (N,C)."""
    s = S - 0.5 * r_mem.unsqueeze(0) if csls else S
    if top_k:
        thr = s.topk(top_k, dim=1).values[:, -1:]
        s = s.masked_fill(s < thr, float("-inf"))
    w = torch.softmax(s / tau, dim=1)
    e = w @ K
    preds = np.array([eval_labels[i] for i in e.argmax(1).cpu().numpy()], dtype=object)
    kept_gt, _, keep_idx = filter_ground_truth(gt, subjects, eval_labels)
    if not len(keep_idx):
        return None
    return float(classification_metrics(kept_gt, list(preds[keep_idx]))["f1_macro"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.environ.get("HALO_CKPT",
                                 _REPO / "training/tokenizer/outputs/pretrain_fixed_mr/best.pt")))
    ap.add_argument("--bank", type=Path, default=_DIR / "memory_bank.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sweep-queries", type=int, default=4000, help="query cap during the sweep")
    ap.add_argument("--datasets", nargs="*", default=list(policy.PRIMARY_EVAL_DATASETS))
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    bank = torch.load(str(args.bank), map_location="cpu", weights_only=True)
    from training.evidence.bank_guard import assert_bank_current
    assert_bank_current(bank, context="tier1_sweep")
    fp = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if bank["backbone"].get("fingerprint") and fp != bank["backbone"]["fingerprint"]:
        raise SystemExit("[tier1] checkpoint != bank backbone")
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, device)
    for p in enc.parameters():
        p.requires_grad_(False)
    Z = F.normalize(bank["Z"].float().to(device), dim=-1)   # raw-feature retrieval space
    mem_y = bank["y"].to(device)
    label_text = F.normalize(bank["label_text"].float().to(device), dim=-1)   # raw SBERT vocab
    sbert = get_sbert_encoder()

    # CSLS hubness: r_mem(j) = mean top-10 cosine of memory entry j to a random reference subset.
    g = torch.Generator().manual_seed(0)
    ref = Z[torch.randperm(Z.shape[0], generator=g)[:8192].to(device)]
    r_mem = torch.empty(Z.shape[0], device=device)
    for s in range(0, Z.shape[0], 8192):
        r_mem[s:s + 8192] = (Z[s:s + 8192] @ ref.t()).topk(10, dim=1).values.mean(1)
    print(f"[tier1] bank {Z.shape[0]} · r_mem(hubness) mean {float(r_mem.mean()):.3f} "
          f"[{float(r_mem.min()):.2f},{float(r_mem.max()):.2f}]", flush=True)

    # ensemble vocab (memory-label) text once
    vocab = bank["vocab"]
    ml_ens = ensemble_text(vocab, sbert, ENS_E).to(device)

    # ---- encode each eval dataset once; cache S + both kernels ----
    cache = {}
    for ds in args.datasets:
        for spec in policy.stream_specs(ds, "primary"):
            stream = spec.stream_id
            try:
                es = load_eval_stream(ds, stream, alignment="non_harmonised")
            except FileNotFoundError:
                continue
            texts = stream_channel_descriptions(ds, stream)
            gs = _stream_gravity_state(ds, stream)
            z = F.normalize(encode_dataset(enc, np.asarray(es.windows), texts, device,
                                           float(es.rate_hz), gs).to(device), dim=-1)
            cand_single = torch.from_numpy(sbert(es.eval_labels).astype(np.float32)).to(device)
            cand_single = F.normalize(cand_single, dim=-1)
            cand_ens = ensemble_text(es.eval_labels, sbert, ENS_E).to(device)
            K_single = torch.relu(label_text[mem_y] @ cand_single.t())
            K_ens = torch.relu(ml_ens[mem_y] @ cand_ens.t())
            cache[f"{ds}/{stream}"] = {
                "z": z, "gt": es.gt, "subjects": es.subjects, "labels": es.eval_labels,
                "K": {False: K_single, True: K_ens},
            }
            print(f"[tier1] encoded {ds}/{stream}: {len(z)} windows", flush=True)

    # ---- sweep (query subsample) ----
    configs = list(itertools.product(TOP_KS, TAUS, CSLS, TEXT_ENS))
    results = {}
    for cell, c in cache.items():
        gsub = torch.Generator().manual_seed(1)
        qi = torch.randperm(len(c["z"]), generator=gsub)[:args.sweep_queries]
        z = c["z"][qi.to(device)]
        gt = [c["gt"][i] for i in qi.tolist()]
        subjects = np.asarray(c["subjects"])[qi.numpy()]
        S = z @ Z.t()                                          # (nq, N)
        for cfg in configs:
            top_k, tau, csls, tens = cfg
            f1 = eval_config(S, r_mem, c["K"][tens][:, :], c["labels"], gt, subjects, top_k, tau, csls)
            results.setdefault(cfg, []).append(f1)
        del S
        torch.cuda.empty_cache()

    means = {cfg: round(float(np.mean(v)), 2) for cfg, v in results.items()}
    ranked = sorted(means.items(), key=lambda kv: -kv[1])
    print("\n=== TIER-1 SWEEP (mean macro-F1 over gate, query-subsampled) ===", flush=True)
    print("  (top_k, tau, csls, text_ens) -> meanF1   [ConSE 42.7 · untrained-baseline 45.3 · harnet 47.3]",
          flush=True)
    for cfg, m in ranked[:12]:
        print(f"  {cfg} -> {m}", flush=True)
    baseline = means.get((0, 0.03, False, False))  # closest to the both=identity baseline knobs

    # ---- confirm the best config on FULL queries ----
    best_cfg = ranked[0][0]
    top_k, tau, csls, tens = best_cfg
    print(f"\n=== BEST {best_cfg} on FULL queries ===", flush=True)
    full = {}
    for cell, c in cache.items():
        S = c["z"] @ Z.t()
        f1 = eval_config(S, r_mem, c["K"][tens], c["labels"], c["gt"], c["subjects"], top_k, tau, csls)
        full[cell] = round(f1, 1)
        print(f"  {cell:26} F1={f1:.1f}", flush=True)
        del S; torch.cuda.empty_cache()
    full_mean = round(float(np.mean(list(full.values()))), 1)
    print(f"  MEAN = {full_mean}  (ConSE 42.7 · harnet 47.3)", flush=True)

    (_DIR / "tier1_sweep.json").write_text(json.dumps(
        {"best_config": {"top_k": top_k, "tau": tau, "csls": csls, "text_ens": tens},
         "best_full_per_cell": full, "best_full_mean": full_mean,
         "sweep_ranked": [{"cfg": list(cfg), "mean_f1": m} for cfg, m in ranked]}, indent=2))
    print(f"-> {_DIR / 'tier1_sweep.json'}", flush=True)


if __name__ == "__main__":
    main()
