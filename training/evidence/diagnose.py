"""Diagnose the trained evidence engine: what is it doing, and why ~= ConSE?

Runs on the EXISTING trained head + bank + encoder — NO training. Three parts:

  A. RETRIEVAL TELEMETRY (internal, held-out subjects where true labels are vocab):
     - purity@k: fraction of the top-k subject-disjoint neighbors that share the query's label
       (is retrieval finding the right label BEFORE the text bridge maps it to a candidate?)
     - effective-k: 1/Σw²  (winner-take-all vs diffuse)
     - hubness (Gini of memory-entry retrieval frequency): are a few entries dominating votes?
     - retrieval-vote balanced accuracy (top-1 neighbor label).

  B. COMPONENT ABLATION on the gate (mean macro-F1 over the 7 primary cells) — localizes the
     bottleneck by swapping ONE learned piece for its untrained identity, at inference:
       trained        g,t,tau all learned                          (the shipped head)
       g=identity     retrieve in RAW frozen space (t,tau kept)    -> does the metric g earn its keep?
       t=identity     kernel in RAW SBERT-384 space (g,tau kept)   -> does the text adapter t help or HURT?
       uniform-w      retrieval OFF (w uniform), t kept            -> does retrieval weighting matter at all?
       both=identity  raw features + raw SBERT bridge              -> value of the WHOLE learned head
     If g=identity ~= trained, the metric adds nothing; if t=identity ~= trained, the bridge is
     frozen-SBERT-limited (the ConSE ceiling); if both=identity ~= trained, the head is inert.

  C. EVIDENCE QUALITY (trained head): argmax margin + the worst per-class F1 cells.

Run:
    PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
    HALO_CKPT=training/tokenizer/outputs/pretrain_fixed_mr/best.pt $PY -m training.evidence.diagnose --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.scripts.curate import deployment_policy as policy
from eval.data import load_eval_stream
from eval.scoring import classification_metrics, filter_ground_truth, get_sbert_encoder, per_class_f1
from model.evidence.head import EvidenceHead
from training.tokenizer.eval_transfer import build_encoder, encode_dataset
from training.tokenizer.pretrain_data import _stream_gravity_state, stream_channel_descriptions

_REPO = Path(__file__).resolve().parents[2]
_DIR = Path(__file__).resolve().parent / "outputs"


def _softmax_weights(s, tau):
    return torch.softmax(s / tau, dim=1)


@torch.no_grad()
def retrieval_telemetry(head, Z, y, subj, device, n_q=3000, seed=0):
    """Part A: purity@k, effective-k, hubness, retrieval-vote bAcc on held-out-subject queries."""
    g = torch.Generator().manual_seed(seed)
    # held-out subjects: take the last 15% of subject ids as query-only
    uniq = torch.unique(subj)
    perm = uniq[torch.randperm(len(uniq), generator=g)]
    val_subj = perm[: max(1, int(0.15 * len(uniq)))]
    val_mask = torch.isin(subj, val_subj)
    q_all = torch.nonzero(val_mask, as_tuple=True)[0]
    q_all = q_all[torch.randperm(len(q_all), generator=g)[:n_q]].to(device)

    g_mem = head.project_query(Z)                          # (N, proj)
    tau = head.tau
    ks = (1, 5, 20)
    hit = {k: [] for k in ks}
    votes, trues, eff_ks, hub_counts = [], [], [], torch.zeros(Z.shape[0], device=device)
    for s in range(0, len(q_all), 256):
        qi = q_all[s:s + 256]
        sim = (head.project_query(Z[qi]) @ g_mem.t()) / tau
        sim = sim.masked_fill(subj.unsqueeze(0) == subj[qi].unsqueeze(1), float("-inf"))
        w = torch.softmax(sim, dim=1)
        eff_ks.append((1.0 / w.pow(2).sum(1).clamp(min=1e-12)))
        topk = sim.topk(max(ks), dim=1).indices                    # (b, 20)
        hub_counts.scatter_add_(0, topk.reshape(-1), torch.ones(topk.numel(), device=device))
        nn_lab = y[topk]                                           # (b, 20)
        for k in ks:
            hit[k].append((nn_lab[:, :k] == y[qi].unsqueeze(1)).float().mean(1))
        votes.append(y[topk[:, 0]]); trues.append(y[qi])
    votes = torch.cat(votes).cpu().numpy(); trues = torch.cat(trues).cpu().numpy()
    vote_bacc = float(np.mean([np.mean(votes[trues == c] == c) for c in np.unique(trues)]))
    # hubness: Gini of how often each memory entry is in someone's top-20
    hc = torch.sort(hub_counts).values.cpu().numpy()
    n = len(hc); gini = float((np.sum((2 * np.arange(1, n + 1) - n - 1) * hc)) / (n * hc.sum() + 1e-9))
    return {
        **{f"purity@{k}": round(float(torch.cat(hit[k]).mean()), 3) for k in ks},
        "effective_k": round(float(torch.cat(eff_ks).mean()), 1),
        "hubness_gini": round(gini, 3),
        "retrieval_vote_bAcc": round(vote_bacc, 3),
        "n_queries": len(q_all),
    }


@torch.no_grad()
def gate_ablation(head, Z, mem_y, label_text_raw, enc, sbert, device, datasets):
    """Part B+C: per-variant mean macro-F1 over the gate + margin/per-class for the trained head."""
    g_mem_trained = head.project_query(Z)                 # (N, proj)
    Z_raw = F.normalize(Z, dim=-1)                        # raw-feature retrieval space
    t_lab_trained = head.project_text(label_text_raw)     # (L, proj)
    label_text_norm = F.normalize(label_text_raw, dim=-1)  # (L, 384) raw SBERT bridge
    tau = head.tau
    variants = ["trained", "g=identity", "t=identity", "uniform-w", "both=identity"]
    f1s = {v: [] for v in variants}
    margins, worst = {}, {}

    for ds in datasets:
        for spec in policy.stream_specs(ds, "primary"):
            stream = spec.stream_id
            try:
                es = load_eval_stream(ds, stream, alignment="non_harmonised")
            except FileNotFoundError:
                continue
            texts = stream_channel_descriptions(ds, stream)
            gs = _stream_gravity_state(ds, stream)
            z = encode_dataset(enc, np.asarray(es.windows), texts, device,
                               float(es.rate_hz), gs,
                               channel_mask=es.mask, dataset=ds, stream=stream).to(device)
            cand_raw = torch.from_numpy(sbert(es.eval_labels).astype(np.float32)).to(device)  # (C,384)
            cand_proj = head.project_text(cand_raw)                                            # (C,proj)
            cand_norm = F.normalize(cand_raw, dim=-1)
            K_t = torch.relu(t_lab_trained[mem_y] @ cand_proj.t())        # trained bridge (N,C)
            K_id = torch.relu(label_text_norm[mem_y] @ cand_norm.t())     # raw-SBERT bridge (N,C)

            preds = {v: np.empty(len(z), dtype=object) for v in variants}
            for s in range(0, len(z), 256):
                zb = z[s:s + 256]
                gq_t, gq_r = head.project_query(zb), F.normalize(zb, dim=-1)
                s_t = (gq_t @ g_mem_trained.t()) / tau
                s_r = (gq_r @ Z_raw.t()) / tau
                w_t, w_r = torch.softmax(s_t, 1), torch.softmax(s_r, 1)
                w_u = torch.full_like(w_t, 1.0 / Z.shape[0])
                ev = {
                    "trained": w_t @ K_t,
                    "g=identity": w_r @ K_t,
                    "t=identity": w_t @ K_id,
                    "uniform-w": w_u @ K_t,
                    "both=identity": w_r @ K_id,
                }
                for v in variants:
                    idx = ev[v].argmax(1).cpu().numpy()
                    preds[v][s:s + 256] = [es.eval_labels[i] for i in idx]
                    if v == "trained" and s == 0:
                        top2 = ev[v].topk(2, dim=1).values
                        margins[f"{ds}/{stream}"] = round(float((top2[:, 0] - top2[:, 1]).mean()), 3)

            kept_gt, _, keep_idx = filter_ground_truth(es.gt, es.subjects, es.eval_labels)
            if not len(keep_idx):
                continue
            for v in variants:
                m = classification_metrics(kept_gt, list(preds[v][keep_idx]))
                f1s[v].append(float(m["f1_macro"]))
            worst[f"{ds}/{stream}"] = per_class_f1(kept_gt, list(preds["trained"][keep_idx]))

    return ({v: round(float(np.mean(f1s[v])), 1) for v in variants}, margins, worst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.environ.get("HALO_CKPT",
                                 _REPO / "training/tokenizer/outputs/pretrain_fixed_mr/best.pt")))
    ap.add_argument("--bank", type=Path, default=_DIR / "memory_bank.pt")
    ap.add_argument("--head", type=Path, default=_DIR / "evidence_head.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", nargs="*", default=list(policy.PRIMARY_EVAL_DATASETS))
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    bank = torch.load(str(args.bank), map_location="cpu", weights_only=True)
    from training.evidence.bank_guard import assert_bank_current
    assert_bank_current(bank, context="diagnose")
    head_blob = torch.load(str(args.head), map_location="cpu", weights_only=True)
    fp = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if bank["backbone"].get("fingerprint") and fp != bank["backbone"]["fingerprint"]:
        raise SystemExit("[diag] checkpoint != bank backbone; rebuild bank or fix --checkpoint")

    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, device)
    for p in enc.parameters():
        p.requires_grad_(False)
    head = EvidenceHead(d_model=int(head_blob["d_model"]), proj=head_blob["proj"]).to(device)
    head.load_state_dict(head_blob["head"]); head.eval()
    Z = bank["Z"].float().to(device)
    mem_y = bank["y"].to(device)
    subj = bank["subj"].to(device)
    label_text = bank["label_text"].float().to(device)
    sbert = get_sbert_encoder()

    print(f"[diag] head best-val {head_blob.get('best_val_bal_acc')}, tau {float(head.tau):.4f}, "
          f"out_scale {float(head.log_out_scale.exp()):.1f} · bank {Z.shape[0]} windows", flush=True)

    print("\n=== A. RETRIEVAL TELEMETRY (internal held-out subjects) ===", flush=True)
    tel = retrieval_telemetry(head, Z, mem_y, subj, device)
    for k, v in tel.items():
        print(f"  {k:22} {v}", flush=True)

    print("\n=== B. COMPONENT ABLATION (gate mean macro-F1) ===", flush=True)
    abl, margins, worst = gate_ablation(head, Z, mem_y, label_text, enc, sbert, device, args.datasets)
    for v, f1 in abl.items():
        print(f"  {v:16} {f1}", flush=True)

    print("\n=== C. EVIDENCE QUALITY (trained head) ===", flush=True)
    print(f"  argmax margin (top1-top2) per cell: {margins}", flush=True)
    for cell, pcf in worst.items():
        lo = sorted(pcf.items(), key=lambda kv: kv[1])[:3]
        print(f"  {cell:26} worst classes: {[(c, round(f,2)) for c,f in lo]}", flush=True)

    out = _DIR / "diagnose.json"
    out.write_text(json.dumps({"retrieval": tel, "ablation": abl, "margins": margins},
                              indent=2, default=float))
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
