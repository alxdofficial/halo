"""ZS-XD gate for the trained evidence decoder (T2.3) — score it against the 47.5 floor.

Harness-identical to the T2.0 adapter, but each eval window's top-k retrieved evidence is run
through the trained decoder before voting (docs/design/EVIDENCE_ENGINE_TIER2.md milestone T2.3):

  encode window -> top-k cosine retrieval over the FULL frozen bank (no holdout at deploy) ->
  decoder refines evidence + pools -> argmax over the stream's candidate labels.

Provenance-guarded: the encoder checkpoint hash must match the bank + the decoder's recorded
backbone. Prints per-cell macro-F1 and the mean, next to ConSE 42.7 / untrained 47.5 / harnet 47.3.
The gate: BEAT 47.5 on the primary cells (else the decoder is dropped and we ship untrained).

Run:
    PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
    HALO_CKPT=training/tokenizer/outputs/pretrain_fixed_mr/best.pt \
      $PY -m training.evidence.eval_decoder --device cuda
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
from eval.scoring import classification_metrics, filter_ground_truth, get_sbert_encoder
from model.evidence.decoder import DecoderConfig, EvidenceDecoder
from training.evidence.bank_guard import assert_bank_current
from training.evidence.labeltext import ensemble_text
from training.tokenizer.eval_transfer import build_encoder, encode_dataset
from training.tokenizer.pretrain_data import _stream_gravity_state, stream_channel_descriptions

_REPO = Path(__file__).resolve().parents[2]
_DIR = Path(__file__).resolve().parent / "outputs"


@torch.no_grad()
def score_cell(dec, enc, es, Z, mem_y, t_ens_mem, sbert, ens, topk, tau, device, batch=128,
               raw_labels=False):
    z = encode_dataset(enc, np.asarray(es.windows), stream_channel_descriptions(es.dataset, es.stream),
                       device, float(es.rate_hz), _stream_gravity_state(es.dataset, es.stream)).to(device)
    z = F.normalize(z, dim=-1)
    # raw_labels => bare string, identical to what eval/scoring.py gives every ConSE baseline
    cand_text = ensemble_text(es.eval_labels, sbert, 1 if raw_labels else ens,
                              use_descriptions=not raw_labels).to(device)     # (C, 384) frozen target
    preds = np.empty(len(z), dtype=object)
    for s in range(0, len(z), batch):
        zq = z[s:s + batch]
        sim = zq @ Z.t()                                                      # (b, N) — full bank allowed
        vals, idx = sim.topk(topk, dim=1)
        w = torch.softmax(vals / tau, dim=1)
        logits = dec(zq=zq, zev=Z[idx], ev_label_text=t_ens_mem[mem_y[idx]], w_retr=w,
                     cand_text=cand_text)
        preds[s:s + batch] = [es.eval_labels[i] for i in logits.argmax(1).cpu().numpy()]
    kept_gt, _, keep = filter_ground_truth(es.gt, es.subjects, es.eval_labels)
    if not len(keep):
        return None
    return float(classification_metrics(kept_gt, list(preds[keep]))["f1_macro"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.environ.get("HALO_CKPT",
                                 _REPO / "training/tokenizer/outputs/pretrain_fixed_mr/best.pt")))
    ap.add_argument("--bank", type=Path, default=_DIR / "memory_bank.pt")
    ap.add_argument("--decoder", type=Path, default=_DIR / "evidence_decoder.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", nargs="*", default=list(policy.PRIMARY_EVAL_DATASETS))
    ap.add_argument("--raw-labels", action="store_true",
                    help="BASELINE PARITY: embed the bare eval label string (what eval/scoring.py "
                         "gives every ConSE baseline) instead of the paraphrase ensemble. Any "
                         "HALO-vs-baseline number must use this, or ensemble every model.")
    ap.add_argument("--untrained", action="store_true",
                    help="CONTROL: skip loading the trained weights. Identity-at-init means this is "
                         "EXACTLY the untrained mechanism at the SAME top-k/tau, isolating the "
                         "decoder's contribution from the retrieval-config change.")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    bank = torch.load(str(args.bank), map_location="cpu", weights_only=True)
    assert_bank_current(bank, context="eval_decoder")
    blob = torch.load(str(args.decoder), map_location="cpu", weights_only=True)
    fp = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if bank["backbone"].get("fingerprint") and fp != bank["backbone"]["fingerprint"]:
        raise SystemExit("[eval_dec] checkpoint != bank backbone")
    if blob["backbone"].get("fingerprint") and fp != blob["backbone"]["fingerprint"]:
        raise SystemExit("[eval_dec] checkpoint != decoder backbone")

    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    enc = build_encoder(ckpt, device)
    for p in enc.parameters():
        p.requires_grad_(False)
    dc = blob["cfg"]
    dec = EvidenceDecoder(DecoderConfig(d_model=dc["d_model"], n_layers=dc["n_layers"],
                                        n_heads=dc["n_heads"])).to(device)
    if not args.untrained:
        dec.load_state_dict(blob["decoder"])
    dec.eval()

    Z = F.normalize(bank["Z"].float().to(device), dim=-1)
    mem_y = bank["y"].to(device)
    sbert = get_sbert_encoder()
    t_ens_mem = ensemble_text(list(bank["vocab"]), sbert, blob["ensemble"]).to(device)
    print(f"[eval_dec] decoder init {blob.get('init_val_transfer_ba'):.4f} -> best "
          f"{blob.get('best_val_transfer_ba'):.4f} · topk {blob['topk']} tau {blob['tau_retr']}", flush=True)

    per_cell = {}
    for ds in args.datasets:
        for spec in policy.stream_specs(ds, "primary"):
            try:
                es = load_eval_stream(ds, spec.stream_id, alignment="non_harmonised")
            except FileNotFoundError:
                continue
            f1 = score_cell(dec, enc, es, Z, mem_y, t_ens_mem, sbert, blob["ensemble"],
                            blob["topk"], blob["tau_retr"], device,
                            raw_labels=args.raw_labels)
            if f1 is not None:
                per_cell[f"{ds}/{spec.stream_id}"] = round(f1, 1)
                print(f"  {ds}/{spec.stream_id:22} F1={f1:.1f}", flush=True)
    mean = round(float(np.mean(list(per_cell.values()))), 1)
    print(f"  MEAN = {mean}  (ConSE 42.7 · untrained 47.5 · harnet 47.3)", flush=True)
    # separate artifacts so the identity CONTROL never clobbers the trained result
    tag = ("_untrained" if args.untrained else "") + ("_rawlabels" if args.raw_labels else "")
    out = _DIR / f"eval_decoder{tag}.json"
    out.write_text(json.dumps({"per_cell": per_cell, "mean": mean,
                               "untrained_control": bool(args.untrained),
                               "raw_labels_parity": bool(args.raw_labels),
                               "topk": blob["topk"], "tau_retr": blob["tau_retr"]}, indent=2))
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
