"""Does a SMALLER, BALANCED memory bank help? (Tier-2 diagnostic.)

The MVP bank is "big and roughly balanced" (164,516 windows, per-label cap 8,000). Two competing
intuitions: (a) more exemplars = better coverage of acquisition configs; (b) a huge head-class-heavy
bank causes hubness and diffuse retrieval (measured effective-k ~2200, hubness Gini 0.81), so a small
balanced bank might force sharper discrimination.

This settles it empirically on the UNTRAINED mechanism (no learning, so no confound): sweep the
per-label cap, holding the tier-1 winning config fixed (full-soft, tau=0.03, text-ensemble E=8).
Each eval cell is encoded ONCE and reused across every bank size, so the sweep is cheap.

Run:
    PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
    HALO_CKPT=training/tokenizer/outputs/pretrain_fixed_mr/best.pt \
      $PY -m training.evidence.bank_size_sweep --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.scripts.curate import deployment_policy as policy
from eval.data import load_eval_stream
from eval.scoring import classification_metrics, filter_ground_truth, get_sbert_encoder
from training.evidence.labeltext import ensemble_text
from training.tokenizer.eval_transfer import build_encoder, encode_dataset
from training.tokenizer.pretrain_data import _stream_gravity_state, stream_channel_descriptions

_REPO = Path(__file__).resolve().parents[2]
_DIR = Path(__file__).resolve().parent / "outputs"

CAPS = [8000, 2000, 500, 200, 50, 20, 5]   # per-label cap (8000 = current bank, effectively uncapped)
TAU = 0.03
ENS = 8


def subsample(y, cap, seed=0):
    """Indices of a per-label-capped, balanced subsample of the bank."""
    keep = []
    for c in torch.unique(y).tolist():
        idx = torch.nonzero(y == c, as_tuple=True)[0]
        if len(idx) > cap:
            g = torch.Generator().manual_seed(seed + c)
            idx = idx[torch.randperm(len(idx), generator=g)[:cap]]
        keep.append(idx)
    return torch.cat(keep)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.environ.get("HALO_CKPT",
                                 _REPO / "training/tokenizer/outputs/pretrain_fixed_mr/best.pt")))
    ap.add_argument("--bank", type=Path, default=_DIR / "memory_bank.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", nargs="*", default=list(policy.PRIMARY_EVAL_DATASETS))
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    bank = torch.load(str(args.bank), map_location="cpu", weights_only=True)
    from training.evidence.bank_guard import assert_bank_current
    assert_bank_current(bank, context="bank_size_sweep")
    Z_all = F.normalize(bank["Z"].float(), dim=-1).to(device)
    y_all = bank["y"].to(device)
    vocab = list(bank["vocab"])
    sbert = get_sbert_encoder()
    ml_ens = ensemble_text(vocab, sbert, ENS).to(device)

    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, device)
    for p in enc.parameters():
        p.requires_grad_(False)

    # encode every eval cell ONCE
    cells = {}
    for ds in args.datasets:
        for spec in policy.stream_specs(ds, "primary"):
            try:
                es = load_eval_stream(ds, spec.stream_id, alignment="non_harmonised")
            except FileNotFoundError:
                continue
            z = encode_dataset(enc, np.asarray(es.windows),
                               stream_channel_descriptions(ds, spec.stream_id), device,
                               float(es.rate_hz), _stream_gravity_state(ds, spec.stream_id),
                               channel_mask=es.mask, dataset=ds,
                               stream=spec.stream_id).to(device)
            cells[f"{ds}/{spec.stream_id}"] = {
                "z": F.normalize(z, dim=-1),
                "cand": ensemble_text(es.eval_labels, sbert, ENS).to(device),
                "labels": list(es.eval_labels), "gt": es.gt, "subjects": es.subjects}
            print(f"[bank] encoded {ds}/{spec.stream_id}: {len(z)}", flush=True)

    results = {}
    for cap in CAPS:
        idx = subsample(y_all.cpu(), cap).to(device)
        Z, y = Z_all[idx], y_all[idx]
        per = {}
        for name, c in cells.items():
            K = torch.relu(ml_ens[y] @ c["cand"].t())              # (N, C)
            preds = np.empty(len(c["z"]), dtype=object)
            with torch.no_grad():
                for s in range(0, len(c["z"]), 256):
                    sim = c["z"][s:s + 256] @ Z.t()
                    w = torch.softmax(sim / TAU, dim=1)
                    e = w @ K
                    preds[s:s + 256] = [c["labels"][i] for i in e.argmax(1).cpu().numpy()]
            kept, _, keep = filter_ground_truth(c["gt"], c["subjects"], c["labels"])
            if len(keep):
                per[name] = round(float(classification_metrics(kept, list(preds[keep]))["f1_macro"]), 1)
        mean = round(float(np.mean(list(per.values()))), 1)
        results[cap] = {"n_windows": int(len(idx)), "per_cell": per, "mean": mean}
        print(f"[bank] cap={cap:<5} N={len(idx):>7}  MEAN={mean}   {per}", flush=True)

    print("\n=== BANK SIZE vs ZS-XD (untrained mechanism, full-soft tau=0.03, ens=8) ===")
    print(f"{'per-label cap':>14} {'bank N':>9} {'mean F1':>9}")
    for cap, r in results.items():
        print(f"{cap:>14} {r['n_windows']:>9} {r['mean']:>9}")
    (_DIR / "bank_size_sweep.json").write_text(json.dumps(results, indent=2))
    print(f"-> {_DIR / 'bank_size_sweep.json'}", flush=True)


if __name__ == "__main__":
    main()
