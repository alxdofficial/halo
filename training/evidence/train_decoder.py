"""T2.2 + T2.3 — episodic class-holdout trainer for the evidence decoder.

The M4a diagnostic proved the *loss* was the bug: closed-vocab CE over the fixed 59-way vocab
overfits the seen-label geometry and the untrained mechanism wins (40.9 vs 47.5). This trainer
replaces that with a **transfer-aligned episodic loss** and trains the §2.2 decoder as a residual
on the untrained mechanism (docs/design/EVIDENCE_ENGINE_TIER2.md §3):

  * **Class-disjoint episodes.** Each step samples a held-out label set H. Queries have labels in
    H; the retrievable memory EXCLUDES all of H (+ always subject-disjoint); candidates = H. So a
    neighbour (label ∉ H) is never itself a valid answer — the decoder must retrieve semantically
    related evidence and TEXT-TRANSFER its (refined) label text to the correct held-out candidate.
    This is genuine zero-shot per episode (Matching/Prototypical-net style), the faithful analog
    of the ZS-XD eval, and it removes M4a's "retrieve the same label" crutch (M4a was label-present).
  * **Reg-to-identity.** The refinement Δ and the pooling residual (KL to the retrieval prior) are
    penalized, so the decoder can only improve on the untrained 47.5 mechanism, never destroy it.
  * **Checkpoint selection = held-out-CONFIG × class-disjoint transfer** on FIXED val episodes
    (never closed-vocab val — the M4a trap).

Everything runs on the cached bank — the encoder never runs here. Retrieval is raw cosine over the
frozen memory (top-k); the decoder does all the learning. Smoke-testable core of the pivotal
experiment; the ZS-XD gate adapter is a separate step.

Run (smoke):
    PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
    $PY -m training.evidence.train_decoder --device cuda --steps 40 --val-every 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.scoring import get_sbert_encoder
from model.evidence.decoder import DecoderConfig, EvidenceDecoder
from training.evidence.labeltext import ensemble_text

_DIR = Path(__file__).resolve().parent / "outputs"
_DEFAULT_BANK = _DIR / "memory_bank.pt"
_DEFAULT_OUT = _DIR / "evidence_decoder.pt"
SEED = 20260720


def balanced_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    accs = [float((pred[true == c] == c).mean()) for c in np.unique(true)]
    return float(np.mean(accs)) if accs else float("nan")


@torch.no_grad()
def retrieve(zq, Z, allowed_mask, k, tau):
    """Top-k subject/label-disjoint retrieval. Returns (idx (B,k), w_retr (B,k) normalized)."""
    sim = zq @ Z.t()                                    # (B, N) cosine (Z, zq pre-normalized)
    sim = sim.masked_fill(~allowed_mask, float("-inf"))
    vals, idx = sim.topk(k, dim=1)
    w = torch.softmax(vals / tau, dim=1)                # renormalized over the k neighbours
    return idx, w


def label_index(H: torch.Tensor, n_vocab: int, device) -> torch.Tensor:
    """Map global label id -> its column in candidate set H (or -1)."""
    pos = torch.full((n_vocab,), -1, device=device, dtype=torch.long)
    pos[H] = torch.arange(len(H), device=device)
    return pos


def run_episode(dec, Z, y, subj, qi, H, t_ens, k, tau, return_aux=False):
    """One class-disjoint episode forward. Memory excludes H (+ subject-disjoint); candidates = H."""
    zq = Z[qi]
    not_in_H = ~torch.isin(y, H)                                  # (N,) labels allowed in memory
    allowed = not_in_H.unsqueeze(0) & (subj.unsqueeze(0) != subj[qi].unsqueeze(1))
    idx, w = retrieve(zq, Z, allowed, k, tau)
    out = dec(zq=zq, zev=Z[idx], ev_label_text=t_ens[y[idx]], w_retr=w,
              cand_text=t_ens[H], return_aux=return_aux)
    return (out, w) if return_aux else out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, default=_DEFAULT_BANK)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--topk", type=int, default=48, help="evidence set size per query")
    ap.add_argument("--tau-retr", type=float, default=0.05, help="retrieval softmax temperature")
    ap.add_argument("--episode-labels", type=int, nargs=2, default=(12, 24),
                    help="[min,max] held-out label-set size H per episode")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lambda-delta", type=float, default=0.1, help="reg-to-identity on Δ")
    ap.add_argument("--lambda-pool", type=float, default=0.1, help="reg-to-identity on pooling (KL to prior)")
    ap.add_argument("--ensemble", type=int, default=8)
    ap.add_argument("--val-frac-cfg", type=float, default=0.2, help="fraction of CONFIGS held out")
    ap.add_argument("--val-episodes", type=int, default=6)
    ap.add_argument("--val-queries", type=int, default=800)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    bank = torch.load(str(args.bank), map_location="cpu", weights_only=True)
    Z = F.normalize(bank["Z"].float(), dim=-1).to(device)
    y = bank["y"].to(device)
    subj = bank["subj"].to(device)
    cfg = bank["cfg"].to(device)
    vocab = list(bank["vocab"])
    n_vocab, d = len(vocab), Z.shape[1]
    sbert = get_sbert_encoder()
    t_ens = ensemble_text(vocab, sbert, args.ensemble).to(device)
    print(f"[dec] bank {Z.shape[0]} windows d={d} · {n_vocab} vocab · {int(cfg.max()) + 1} configs "
          f"· backbone {bank['backbone']['git']} (val_ba {bank['backbone']['val_ba']:.3f})", flush=True)

    # held-out-CONFIG split for selection
    cfg_ids = np.arange(int(cfg.max()) + 1); rng.shuffle(cfg_ids)
    n_val = max(1, int(len(cfg_ids) * args.val_frac_cfg))
    is_val = torch.isin(cfg, torch.tensor(cfg_ids[:n_val], device=device))
    train_q = torch.nonzero(~is_val, as_tuple=True)[0]
    val_q = torch.nonzero(is_val, as_tuple=True)[0]
    train_present = torch.unique(y[train_q])
    val_present = torch.unique(y[val_q])
    print(f"[dec] queries: {len(train_q)} train / {len(val_q)} val · labels present "
          f"{len(train_present)} train / {len(val_present)} val · {n_val}/{len(cfg_ids)} configs held out",
          flush=True)

    lo, hi = args.episode_labels

    def sample_H(present):
        size = min(int(rng.integers(lo, hi + 1)), len(present))
        pick = rng.choice(present.cpu().numpy(), size=size, replace=False)
        return torch.tensor(sorted(pick.tolist()), device=device)

    def sample_queries(pool, H, n):
        """n query indices from `pool` whose label ∈ H, sqrt-balanced across H."""
        in_H = torch.isin(y[pool], H)
        cand = pool[in_H]
        counts = torch.bincount(y[cand], minlength=n_vocab).float().clamp(min=1)
        wts = (1.0 / counts.sqrt())[y[cand]]
        return cand[torch.multinomial(wts, min(n, len(cand)), replacement=len(cand) < n)]

    # FIXED val episodes (same H + queries every eval → a stable selection metric)
    val_eps = []
    for _ in range(args.val_episodes):
        H = sample_H(val_present)
        qi = sample_queries(val_q, H, args.val_queries)
        val_eps.append((H, qi))

    @torch.no_grad()
    def evaluate():
        dec.eval()
        accs = []
        for H, qi in val_eps:
            logits = run_episode(dec, Z, y, subj, qi, H, t_ens, args.topk, args.tau_retr)
            pred = H[logits.argmax(1)].cpu().numpy()
            accs.append(balanced_accuracy(pred, y[qi].cpu().numpy()))
        return float(np.mean(accs))

    dec = EvidenceDecoder(DecoderConfig(d_model=d, n_layers=args.layers, n_heads=args.heads)).to(device)
    opt = torch.optim.AdamW(dec.param_groups(weight_decay=0.01), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) *
        (0.5 * (1 + np.cos(np.pi * max(0, s - args.warmup) / max(1, args.steps - args.warmup)))))

    init_val = evaluate()
    print(f"[dec] init val transfer bAcc (decoder≡untrained) = {init_val:.4f} "
          f"(chance≈{1/len(val_eps[0][0]):.3f})", flush=True)

    best_val, best_sd, t0 = init_val, {k_: v.detach().cpu().clone() for k_, v in dec.state_dict().items()}, time.time()
    for step in range(1, args.steps + 1):
        dec.train()
        H = sample_H(train_present)
        qi = sample_queries(train_q, H, args.batch)
        pos = label_index(H, n_vocab, device)
        target = pos[y[qi]]
        (logits, aux), w = run_episode(dec, Z, y, subj, qi, H, t_ens, args.topk, args.tau_retr,
                                       return_aux=True)
        ce = F.cross_entropy(logits, target)
        a = aux["pool_weights"]
        kl_pool = (w * (torch.log(w.clamp_min(1e-12)) - torch.log(a.clamp_min(1e-12)))).sum(1).mean()
        reg = args.lambda_delta * aux["delta"].norm(dim=-1).mean() + args.lambda_pool * kl_pool
        loss = ce + reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        do_log = step % args.val_every == 0 or step == 1
        gtel = {}
        if do_log:
            gn = lambda ps: float(sum(float(p.grad.pow(2).sum()) for p in ps if p.grad is not None) ** 0.5)
            gtel = {"grad/refiner": round(gn(dec.refiner.parameters()), 4),
                    "grad/pool": round(gn(dec.pool_phi.parameters()), 4),
                    "grad/blocks": round(gn(dec.blocks.parameters()), 4)}
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step(); sched.step()

        if do_log:
            va = evaluate()
            if va > best_val:
                best_val = va
                best_sd = {k_: v.detach().cpu().clone() for k_, v in dec.state_dict().items()}
            eff_k = float((1.0 / a.detach().pow(2).sum(1).clamp(min=1e-12)).mean())
            print(json.dumps({"step": step, "loss": round(float(loss), 4), "ce": round(float(ce), 4),
                              "reg": round(float(reg), 5), "kl_pool": round(float(kl_pool), 5),
                              "val_transfer_ba": round(va, 4), "best": round(best_val, 4),
                              "n_cand": len(H), "delta_norm": round(aux["delta_norm"], 4),
                              "eff_k": round(eff_k, 1), **gtel,
                              "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    dec.load_state_dict(best_sd)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"decoder": {k_: v.cpu() for k_, v in dec.state_dict().items()},
                "cfg": {"d_model": d, "n_layers": args.layers, "n_heads": args.heads},
                "topk": args.topk, "tau_retr": args.tau_retr, "ensemble": args.ensemble,
                "vocab": vocab, "best_val_transfer_ba": best_val, "init_val_transfer_ba": init_val,
                "bank": str(args.bank), "backbone": bank["backbone"]}, str(args.out))
    print(f"[dec] done: init {init_val:.4f} -> best held-out transfer bAcc {best_val:.4f} -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
