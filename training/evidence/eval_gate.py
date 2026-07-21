"""M4a gate: cross-dataset ZS-XD macro-F1 of the retrieval evidence head vs ConSE.

For each primary eval dataset/stream we encode the eval windows with the SAME frozen
fixed+MR encoder the memory was built from, retrieve against the frozen bank, and let the
evidence head score the dataset's own candidate labels. Scoring reuses the harness path
(`eval.data.load_eval_stream` + `eval.scoring.filter_ground_truth` + `classification_metrics`)
so the macro-F1 is directly comparable to the ConSE baseline table.

This is the make-or-break question: does a learned retrieval bridge beat the heuristic ConSE
convex blend on the *same* encoder? Reference ConSE-on-fixed+MR:
    HALO_CKPT=training/tokenizer/outputs/pretrain_fixed_mr/best.pt \
      PY -m eval.run_baselines --baselines halo --device cuda   # refits its head on fixed+MR

Run:
    PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
    HALO_CKPT=training/tokenizer/outputs/pretrain_fixed_mr/best.pt $PY -m training.evidence.eval_gate --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from data.scripts.curate import deployment_policy as policy
from eval.data import load_eval_stream
from eval.scoring import classification_metrics, filter_ground_truth, get_sbert_encoder
from model.evidence.head import EvidenceHead
from training.tokenizer.eval_transfer import build_encoder, encode_dataset
from training.tokenizer.pretrain_data import _stream_gravity_state, stream_channel_descriptions

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_BANK = Path(__file__).resolve().parent / "outputs" / "memory_bank.pt"
_DEFAULT_HEAD = Path(__file__).resolve().parent / "outputs" / "evidence_head.pt"


def load_head(head_blob: dict, d_model: int, device) -> EvidenceHead:
    head = EvidenceHead(d_model=d_model, proj=head_blob["proj"]).to(device)
    head.load_state_dict(head_blob["head"])
    head.eval()
    return head


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.environ.get("HALO_CKPT",
                                 _REPO / "training/tokenizer/outputs/pretrain_fixed_mr/best.pt")))
    ap.add_argument("--bank", type=Path, default=_DEFAULT_BANK)
    ap.add_argument("--head", type=Path, default=_DEFAULT_HEAD)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", nargs="*", default=list(policy.PRIMARY_EVAL_DATASETS))
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    bank = torch.load(str(args.bank), map_location="cpu", weights_only=True)
    from training.evidence.bank_guard import assert_bank_current
    assert_bank_current(bank, context="eval_gate")
    head_blob = torch.load(str(args.head), map_location="cpu", weights_only=True)

    # PROVENANCE GUARD: the eval encoder MUST be the exact checkpoint the bank was built from,
    # else queries land in an embedding space g/t never saw and the F1 is silently meaningless.
    fp = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    bank_fp = bank["backbone"].get("fingerprint")
    if bank_fp and fp != bank_fp:
        raise SystemExit(
            f"[gate] encoder MISMATCH: --checkpoint {args.checkpoint} (fp {fp[:12]}) != memory-bank "
            f"backbone {bank['backbone'].get('checkpoint')} (fp {bank_fp[:12]}). Rebuild the bank "
            f"with this checkpoint, or point --checkpoint at the bank's backbone.")
    head_fp = head_blob.get("backbone", {}).get("fingerprint")
    if head_fp and head_fp != bank_fp:
        raise SystemExit(f"[gate] head was trained on a different bank backbone (fp {head_fp[:12]}).")

    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, device)
    for p in enc.parameters():
        p.requires_grad_(False)
    d_live = int(ckpt["config"]["d_model"])
    d_head = int(head_blob["d_model"])           # construct the head with the d it was TRAINED at
    if d_head != d_live:
        raise SystemExit(f"[gate] d_model mismatch: head {d_head} != encoder {d_live}")

    Z = bank["Z"].float().to(device)
    mem_y = bank["y"].to(device)
    label_text = bank["label_text"].float().to(device)
    head = load_head(head_blob, d_head, device)
    sbert = get_sbert_encoder()

    with torch.no_grad():
        g_mem = head.project_query(Z)                     # (N, proj)
        t_lab = head.project_text(label_text)             # (L, proj) vocab, indexed by mem_y

    print(f"[gate] encoder {args.checkpoint.name} (val_ba {ckpt['val_ba']:.3f}, "
          f"frontend={ckpt['config'].get('frontend')}, MR={ckpt['config'].get('multiresolution')}) "
          f"· bank {Z.shape[0]} windows", flush=True)

    results, f1s = {}, []
    for ds in args.datasets:
        for spec in policy.stream_specs(ds, "primary"):
            stream = spec.stream_id
            try:
                es = load_eval_stream(ds, stream, alignment="non_harmonised")
            except FileNotFoundError as e:
                print(f"  {ds}/{stream}: SKIP ({e})", flush=True)
                continue
            texts = stream_channel_descriptions(ds, stream)
            gs = _stream_gravity_state(ds, stream)
            z = encode_dataset(enc, np.asarray(es.windows), texts, device,
                               float(es.rate_hz), gs).to(device)     # (N, d)
            cand_proj = head.project_text(
                torch.from_numpy(sbert(es.eval_labels).astype(np.float32)).to(device))  # (C, proj)

            preds = np.empty(len(z), dtype=object)
            with torch.no_grad():
                for s in range(0, len(z), 256):
                    gq = head.project_query(z[s:s + 256])
                    mask = torch.ones(gq.shape[0], Z.shape[0], dtype=torch.bool, device=device)
                    e = head.evidence(gq, g_mem, mem_y, cand_proj=cand_proj,
                                      t_labels=t_lab, retrieval_mask=mask)
                    idx = e.argmax(1).cpu().numpy()
                    preds[s:s + 256] = [es.eval_labels[i] for i in idx]

            kept_gt, _, keep_idx = filter_ground_truth(es.gt, es.subjects, es.eval_labels)
            if not len(keep_idx):
                print(f"  {ds}/{stream}: no in-vocab windows", flush=True)
                continue
            m = classification_metrics(kept_gt, list(preds[keep_idx]))
            f1 = float(m["f1_macro"])
            results[f"{ds}/{stream}"] = round(f1, 1)
            f1s.append(f1)
            print(f"  {ds:14} {stream:20} F1={f1:5.1f}  ({len(keep_idx)} windows, "
                  f"{len(es.eval_labels)} candidates)", flush=True)

    mean = float(np.mean(f1s)) if f1s else float("nan")
    print(f"\n[gate] evidence-head ZS-XD mean macro-F1 = {mean:.1f} across {len(f1s)} cells", flush=True)
    out = args.head.parent / "eval_gate.json"
    out.write_text(json.dumps({"mean_f1_macro": round(mean, 2), "per_cell": results,
                               "checkpoint": str(args.checkpoint)}, indent=2))
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
