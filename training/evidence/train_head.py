"""M4a episodic trainer for the retrieval evidence head (Phase B, the learned ConSE).

Everything is cached: the frozen memory bank (built by ``build_memory``) lives entirely in
VRAM, and every step is a batched matmul over it — the encoder never runs here. A step:

  1. project the whole memory once: g(Z) and t(vocab)           (the only per-step recompute)
  2. sample a class-balanced query batch from TRAIN subjects
  3. subject-disjoint retrieval mask (drop the query's own subject → no self-retrieval leak)
  4. evidence -> scaled-evidence logits -> cross-entropy on the true label

Internal validation uses HELD-OUT subjects (never sampled as training queries) → an honest
generalization proxy for checkpoint selection. The real gate (held-out eval *datasets* vs
ConSE) is a separate step. This is M4a: retrieval accuracy only; abstention/Dirichlet = M5.

Run:
    PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
    $PY -m training.evidence.train_head --device cuda --steps 3000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from eval.scoring import get_sbert_encoder
from model.evidence.head import EvidenceHead

_DEFAULT_BANK = Path(__file__).resolve().parent / "outputs" / "memory_bank.pt"
_DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "evidence_head.pt"
SEED = 20260720


def balanced_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    accs = [float((pred[true == c] == c).mean()) for c in np.unique(true)]
    return float(np.mean(accs)) if accs else float("nan")


def _global_label_paraphrases():
    """Merge every per-dataset synonym table + template list into ONE dataset-agnostic pool.

    `augment_label(label, dataset_name="")` falls into a generic fallback that ignores the
    synonym tables entirely (only ~4 template wraps), so the vocab labels here — which no
    single dataset owns — would collapse to near-duplicates. Merging the tables recovers real
    lexical diversity (walking -> strolling/ambulating, cycling -> biking, ...).
    """
    from data.scripts.labels.label_augmentation import DATASET_CONFIGS
    synonyms: dict[str, set] = {}
    templates: set = set()
    for cfg in DATASET_CONFIGS.values():
        for lab, forms in cfg.get("synonyms", {}).items():
            synonyms.setdefault(lab, set()).update(forms)
        templates.update(cfg.get("templates", []))
    return ({k: sorted(v) for k, v in synonyms.items()}, sorted(templates) or ["{}"])


def build_label_variants(vocab, K: int, seed: int) -> torch.Tensor:
    """(L, K, 384) SBERT embeddings of K paraphrase variants per label (variant 0 = canonical).

    Trains the t-adapter for label-phrasing invariance — newly useful in Phase B because
    `t(label)` is the trainable, load-bearing text path (it was computed-then-discarded in
    Phase A, where A2 keyed on label IDs). Precomputed ONCE; the loop just samples an index,
    so there is no live SBERT in training. Eval always uses the canonical text (variant 0).
    """
    import random as _random
    sbert = get_sbert_encoder()
    synonyms, templates = _global_label_paraphrases()
    rng = _random.Random(seed)
    rows = []
    for k in range(K):
        if k == 0:
            rows.append([lab.replace("_", " ") for lab in vocab])   # canonical
            continue
        variant = []
        for lab in vocab:
            base = rng.choice(synonyms.get(lab, [lab.replace("_", " ")]))
            variant.append(rng.choice(templates).format(base))
        rows.append(variant)
    embs = np.stack([sbert(r).astype(np.float32) for r in rows], axis=1)   # (L, K, 384)
    return torch.from_numpy(embs)


@torch.no_grad()
def evaluate(head, y, subj, q_idx, g_mem, t_lab, batch=1024):
    """Balanced accuracy of argmax-evidence for the given query indices (subject-disjoint)."""
    preds, trues = [], []
    for s in range(0, len(q_idx), batch):
        qi = q_idx[s:s + batch]
        gq = g_mem[qi]
        mask = subj.unsqueeze(0) != subj[qi].unsqueeze(1)          # (b, N)
        e = head.evidence(gq, g_mem, y, cand_proj=t_lab, t_labels=t_lab, retrieval_mask=mask)
        preds.append(e.argmax(1).cpu().numpy())
        trues.append(y[qi].cpu().numpy())
    return balanced_accuracy(np.concatenate(preds), np.concatenate(trues))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, default=_DEFAULT_BANK)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--proj", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of subjects held out")
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--label-aug", type=int, default=8,
                    help="paraphrase variants per label for the t-kernel (0 = canonical only)")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    bank = torch.load(str(args.bank), map_location="cpu", weights_only=True)
    Z = bank["Z"].float().to(device)                    # (N, d)
    y = bank["y"].to(device)                            # (N,)
    subj = bank["subj"].to(device)                      # (N,)
    label_text = bank["label_text"].float().to(device)  # (L, 384)
    vocab = bank["vocab"]
    N, d = Z.shape
    print(f"[m4a] bank: {N} windows · d={d} · {len(vocab)} vocab · "
          f"{len(bank['subj_names'])} subjects · backbone {bank['backbone']['git']} "
          f"(val_ba {bank['backbone']['val_ba']:.3f})", flush=True)

    # subject-disjoint query split (memory = ALL entries; only the query pool is split)
    rng = np.random.default_rng(SEED)
    subj_ids = np.arange(len(bank["subj_names"]))
    rng.shuffle(subj_ids)
    n_val = max(1, int(len(subj_ids) * args.val_frac))
    val_subj = torch.tensor(subj_ids[:n_val], device=device)
    is_val = torch.isin(subj, val_subj)
    train_q = torch.nonzero(~is_val, as_tuple=True)[0]
    val_q = torch.nonzero(is_val, as_tuple=True)[0]
    print(f"[m4a] query pools: {len(train_q)} train / {len(val_q)} val "
          f"({n_val}/{len(subj_ids)} subjects held out)", flush=True)

    # sqrt-balanced query sampling weights (tame head-class flooding — capture24/wisdm)
    counts = torch.bincount(y[train_q], minlength=len(vocab)).float()
    inv = 1.0 / counts.clamp(min=1).sqrt()
    q_weights = inv[y[train_q]]

    variants = None
    if args.label_aug > 0:
        variants = build_label_variants(vocab, args.label_aug, SEED).to(device)   # (L, K, 384)
        print(f"[m4a] label augmentation: {args.label_aug} paraphrase variants/label", flush=True)
    L_idx = torch.arange(len(vocab), device=device)

    head = EvidenceHead(d_model=d, proj=args.proj).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_val, best_sd, t0 = -1.0, None, time.time()
    for step in range(1, args.steps + 1):
        head.train()
        g_mem = head.project_query(Z)                   # (N, proj) — recomputed (g learns)
        # sample a paraphrase variant per label this step (phrasing-invariance for t)
        lt = label_text if variants is None else variants[
            L_idx, torch.randint(0, variants.shape[1], (len(vocab),), device=device)]
        t_lab = head.project_text(lt)                   # (L, proj) — recomputed (t learns)
        sel = torch.multinomial(q_weights, args.batch, replacement=True)
        qi = train_q[sel]
        gq = g_mem[qi]
        mask = subj.unsqueeze(0) != subj[qi].unsqueeze(1)   # (B, N) subject-disjoint
        e = head.evidence(gq, g_mem, y, cand_proj=t_lab, t_labels=t_lab, retrieval_mask=mask)
        loss = crit(head.logits(e), y[qi])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.val_every == 0 or step == 1:
            head.eval()
            with torch.no_grad():
                g_mem = head.project_query(Z)
                t_lab = head.project_text(label_text)
                va = evaluate(head, y, subj, val_q, g_mem, t_lab)
            if va > best_val:
                best_val = va
                best_sd = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            print(json.dumps({"step": step, "loss": round(float(loss.detach()), 4),
                              "val_bal_acc": round(va, 4), "best": round(best_val, 4),
                              "tau": round(float(head.tau.detach()), 4),
                              "out_scale": round(float(head.log_out_scale.detach().exp()), 2),
                              "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    if best_sd is not None:
        head.load_state_dict(best_sd)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": {k: v.cpu() for k, v in head.state_dict().items()},
                "proj": args.proj, "d_model": d, "vocab": vocab,
                "best_val_bal_acc": best_val,
                "bank": str(args.bank), "backbone": bank["backbone"]}, str(args.out))
    print(f"[m4a] done: best held-out-subject balanced acc {best_val:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
