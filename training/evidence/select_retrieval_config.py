"""F2 FIX — select the retrieval config on HELD-OUT TRAINING CONFIGS, never on the eval cells.

`tier1_sweep.py` ranked 36 retrieval configs by mean macro-F1 **on the 7 primary eval datasets** and
froze the winner into the adapter (FINDINGS §6 F2). That is test-set model selection: the reported
number inherits the max over 36 configs. This script does the selection honestly.

Protocol — each held-out training CONFIG is treated as a pseudo-eval-dataset, which is a close
structural analogue of ZS-XD:
  * query set   = all bank windows from a held-out config (an unseen device/placement/rate)
  * memory      = bank windows from the REMAINING configs only (the held-out config is absent,
                  exactly as a real eval dataset is absent from memory)
  * candidates  = the label set present in that held-out config (like a dataset's eval_labels)
  * score       = macro-F1, averaged over held-out configs
The config split matches `train_decoder.py` (same seed / val_frac_cfg) so selection and training
agree on what "held out" means.

Everything runs on cached bank vectors — the encoder never runs, so the whole sweep is seconds.

Report BOTH the honestly-selected config and the eval-selected one; the gap is the optimism bias.

Run:
    PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
    $PY -m training.evidence.select_retrieval_config --device cuda
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.scoring import classification_metrics, get_sbert_encoder
from training.evidence.labeltext import ensemble_text

_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 20260720

TOP_KS = [0, 200, 50]          # 0 = full-soft (no cutoff)
TAUS = [0.03, 0.05, 0.08]
CSLS = [False, True]
TEXT_ENS = [False, True]
ENS_E = 8


@torch.no_grad()
def score_config(Zq, yq, Zm, ym, cand, K, r_mem, top_k, tau, csls, batch=512):
    """macro-F1 of the untrained mechanism for one held-out config."""
    preds, trues = [], []
    for s in range(0, len(Zq), batch):
        sim = Zq[s:s + batch] @ Zm.t()
        if csls:
            sim = sim - 0.5 * r_mem.unsqueeze(0)
        if top_k:
            thr = sim.topk(min(top_k, sim.shape[1]), dim=1).values[:, -1:]
            sim = sim.masked_fill(sim < thr, float("-inf"))
        w = torch.softmax(sim / tau, dim=1)
        e = w @ K                                            # (b, C)
        preds.append(cand[e.argmax(1)].cpu().numpy())
        trues.append(yq[s:s + batch].cpu().numpy())
    p, t = np.concatenate(preds), np.concatenate(trues)
    return float(classification_metrics([str(x) for x in t], [str(x) for x in p])["f1_macro"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, default=_DIR / "memory_bank.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val-frac-cfg", type=float, default=0.2)
    ap.add_argument("--max-queries", type=int, default=4000, help="cap per held-out config")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    bank = torch.load(str(args.bank), map_location="cpu", weights_only=True)
    Z = F.normalize(bank["Z"].float(), dim=-1).to(device)
    y = bank["y"].to(device)
    cfg = bank["cfg"].to(device)
    vocab = list(bank["vocab"])
    cfg_names = bank["cfg_names"]
    sbert = get_sbert_encoder()
    t_plain = ensemble_text(vocab, sbert, 1, use_descriptions=False).to(device)   # bare label string
    t_ens = ensemble_text(vocab, sbert, ENS_E).to(device)                          # train-only ensemble

    # SAME config split as train_decoder.py
    rng = np.random.default_rng(SEED)
    cfg_ids = np.arange(int(cfg.max()) + 1)
    rng.shuffle(cfg_ids)
    n_val = max(1, int(len(cfg_ids) * args.val_frac_cfg))
    held = sorted(cfg_ids[:n_val].tolist())
    print(f"[sel] held-out configs for SELECTION: {[cfg_names[c] for c in held]}", flush=True)

    gen = torch.Generator().manual_seed(SEED)
    cells = []
    for c in held:
        qmask = cfg == c
        qi = torch.nonzero(qmask, as_tuple=True)[0]
        if len(qi) > args.max_queries:
            qi = qi[torch.randperm(len(qi), generator=gen)[:args.max_queries].to(device)]
        mi = torch.nonzero(~qmask, as_tuple=True)[0]          # memory excludes this config
        cand = torch.unique(y[qi])
        if len(cand) < 2:
            continue
        cells.append({"name": cfg_names[c], "qi": qi, "mi": mi, "cand": cand})
        print(f"[sel]   {cfg_names[c]:28} {len(qi)} queries · {len(cand)} labels · "
              f"{len(mi)} memory", flush=True)

    results = {}
    for cell in cells:
        Zq, yq = Z[cell["qi"]], y[cell["qi"]]
        Zm, ym = Z[cell["mi"]], y[cell["mi"]]
        cand = cell["cand"]
        # CSLS hubness reference over this cell's memory
        ref = Zm[torch.randperm(len(Zm), generator=gen)[:8192].to(device)]
        r_mem = torch.empty(len(Zm), device=device)
        for s in range(0, len(Zm), 8192):
            r_mem[s:s + 8192] = (Zm[s:s + 8192] @ ref.t()).topk(10, dim=1).values.mean(1)
        Ks = {False: torch.relu(t_plain[ym] @ t_plain[cand].t()),
              True: torch.relu(t_ens[ym] @ t_ens[cand].t())}
        for top_k, tau, csls, tens in itertools.product(TOP_KS, TAUS, CSLS, TEXT_ENS):
            f1 = score_config(Zq, yq, Zm, ym, cand, Ks[tens], r_mem, top_k, tau, csls)
            results.setdefault((top_k, tau, csls, tens), []).append(f1)
        del Ks, r_mem
        torch.cuda.empty_cache()

    means = {k: round(float(np.mean(v)), 2) for k, v in results.items()}
    ranked = sorted(means.items(), key=lambda kv: -kv[1])
    print("\n=== SELECTION on HELD-OUT TRAINING CONFIGS (mean macro-F1) ===", flush=True)
    print("  (top_k, tau, csls, text_ens) -> meanF1", flush=True)
    for cfgk, m in ranked[:10]:
        print(f"  {cfgk} -> {m}", flush=True)
    best = ranked[0][0]
    eval_selected = (0, 0.03, False, True)     # what tier1_sweep picked ON THE EVAL CELLS
    print(f"\n  honestly-selected : {best} -> {means[best]}", flush=True)
    print(f"  eval-selected     : {eval_selected} -> {means.get(eval_selected)}  "
          f"(rank {[c for c, _ in ranked].index(eval_selected) + 1}/{len(ranked)})", flush=True)
    print(f"  AGREE" if best == eval_selected else "  DIFFER -> apply the honest one to eval ONCE",
          flush=True)

    out = _DIR / "select_retrieval_config.json"
    out.write_text(json.dumps({
        "held_out_configs": [c["name"] for c in cells],
        "best_heldout_selected": {"top_k": best[0], "tau": best[1], "csls": best[2],
                                  "text_ens": best[3], "f1": means[best]},
        "eval_selected_config": {"top_k": eval_selected[0], "tau": eval_selected[1],
                                 "csls": eval_selected[2], "text_ens": eval_selected[3],
                                 "f1_on_heldout": means.get(eval_selected)},
        "ranked": [{"cfg": list(c), "mean_f1": m} for c, m in ranked]}, indent=2))
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
