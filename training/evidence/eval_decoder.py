"""ZS-XD gate for the trained evidence decoder (T2.3) — score it against the 47.5 floor.

Harness-identical to the T2.0 adapter, but each eval window's top-k retrieved evidence is run
through the trained decoder before voting (docs/design/EVIDENCE_ENGINE_TIER2.md milestone T2.3):

  encode window -> top-k cosine retrieval over the FULL frozen bank (no holdout at deploy) ->
  decoder refines evidence + pools -> argmax over the stream's candidate labels.

Provenance-guarded: the encoder checkpoint hash must match the bank + the decoder's recorded
backbone. Prints per-cell macro-F1 and the mean.

**The gate is `--untrained` at the SAME top-k/tau, not a remembered constant.** The historical
"beat 47.5" framing was unsound: 47.5 was measured at top_k=0/tau=0.03 with ensembled candidate
text, while this script runs top_k=48/tau=0.05, so the two were never comparable. Output defaults
to baseline parity (bare eval label strings).

Run:
    PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
    HALO_CKPT=training/tokenizer/outputs/pretrain_fixed_mr/best.pt \
      $PY -m training.evidence.eval_decoder --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.scripts.curate import deployment_policy as policy
from eval.data import load_eval_stream
from eval.scoring import classification_metrics, filter_ground_truth, get_sbert_encoder
from model.evidence.decoder import DecoderConfig, EvidenceDecoder
from model.evidence.edl import DensityGate, acc_at_coverage, aurc
from training.evidence.bank_guard import (
    assert_bank_current, assert_bank_matches_backbone, assert_embedding_path_current,
    vocab_fingerprint)
from training.evidence.labeltext import ensemble_text
from training.tokenizer.eval_transfer import build_encoder, encode_dataset
from training.tokenizer.pretrain_data import _stream_gravity_state, stream_channel_descriptions

_REPO = Path(__file__).resolve().parents[2]
_DIR = Path(__file__).resolve().parent / "outputs"


@torch.no_grad()
def score_cell(dec, enc, es, Z, mem_y, t_ens_mem, sbert, ens, topk, tau, device, batch=128,
               raw_labels=False, gate=None):
    z = encode_dataset(enc, np.asarray(es.windows), stream_channel_descriptions(es.dataset, es.stream),
                       device, float(es.rate_hz), _stream_gravity_state(es.dataset, es.stream),
                       channel_mask=es.mask, dataset=es.dataset, stream=es.stream).to(device)
    z = F.normalize(z, dim=-1)
    # raw_labels => bare string, identical to what eval/scoring.py gives every ConSE baseline
    cand_text = ensemble_text(es.eval_labels, sbert, 1 if raw_labels else ens,
                              use_descriptions=not raw_labels).to(device)     # (C, 384) frozen target
    preds = np.empty(len(z), dtype=object)
    uncertainties = np.empty(len(z), dtype=np.float32) if gate is not None else None
    for s in range(0, len(z), batch):
        zq = z[s:s + batch]
        sim = zq @ Z.t()                                                      # (b, N) — full bank allowed
        vals, idx = sim.topk(min(int(topk), Z.shape[0]), dim=1)
        w = torch.softmax(vals / tau, dim=1)
        out = dec(zq=zq, zev=Z[idx], ev_label_text=t_ens_mem[mem_y[idx]], w_retr=w,
                  cand_text=cand_text, return_aux=gate is not None)
        if gate is not None:
            logits, aux = out
            alpha, _ = gate.alpha(aux["evidence"], vals)
            uncertainties[s:s + len(zq)] = (
                alpha.shape[1] / alpha.sum(1)
            ).cpu().numpy()
        else:
            logits = out
        preds[s:s + batch] = [es.eval_labels[i] for i in logits.argmax(1).cpu().numpy()]
    kept_gt, _, keep = filter_ground_truth(es.gt, es.subjects, es.eval_labels)
    if not len(keep):
        return None
    f1 = float(classification_metrics(kept_gt, list(preds[keep]))["f1_macro"])
    calibration = {}
    if gate is not None:
        correct = np.asarray(preds[keep]) == np.asarray(kept_gt)
        u = uncertainties[keep]
        calibration = {
            "mean_u_correct": float(u[correct].mean()) if correct.any() else float("nan"),
            "mean_u_incorrect": float(u[~correct].mean()) if (~correct).any() else float("nan"),
            "aurc": aurc(u, correct),
            "acc@0.8cov": acc_at_coverage(u, correct, 0.8),
        }
    return f1, calibration


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.environ.get("HALO_CKPT",
                                 _REPO / "training/tokenizer/outputs/pretrain_fixed_mr/best.pt")))
    ap.add_argument("--bank", type=Path, default=_DIR / "memory_bank.pt")
    ap.add_argument("--decoder", type=Path, default=_DIR / "evidence_decoder.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", nargs="*", default=list(policy.PRIMARY_EVAL_DATASETS))
    # Parity is the DEFAULT. It used to be opt-in, so the default run gave HALO an 8-way
    # training-derived paraphrase ensemble over the eval candidate labels while every ConSE
    # baseline got a bare `label.replace("_", " ")` -- worth ~2.6 F1 (46.7 vs 44.1). A non-parity
    # number must now be asked for explicitly, and is labelled as such in the output.
    ap.add_argument("--ensemble-candidates", dest="raw_labels", action="store_false", default=True,
                    help="NON-PARITY DIAGNOSTIC: embed the paraphrase ensemble over the eval "
                         "candidate labels. No ConSE baseline gets this, so the result is NOT "
                         "comparable to the baseline table. Default is bare eval label strings, "
                         "exactly what eval/scoring.py hands every baseline.")
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
    assert_bank_matches_backbone(bank, ckpt, context="eval_decoder")   # F3: same Phase-A corpus
    enc = build_encoder(ckpt, device)
    # ...and the same encode CODE path (pooling/tail/text), which no other fingerprint covers.
    assert_embedding_path_current(bank, enc, device, context="eval_decoder")
    for p in enc.parameters():
        p.requires_grad_(False)
    dc = blob["cfg"]
    dec = EvidenceDecoder(DecoderConfig(d_model=dc["d_model"], n_layers=dc["n_layers"],
                                        n_heads=dc["n_heads"],
                                        n_subspaces=dc.get("n_subspaces", 0),
                                        subspace_dim=dc.get("subspace_dim", 64))).to(device)
    if not args.untrained:
        dec.load_state_dict(blob["decoder"])
    dec.eval()
    gate = None
    if blob.get("loss") == "edl":
        gc = blob.get("gate_cfg", {})
        gate = DensityGate(
            gate_delta=float(gc.get("density_threshold_init", 0.3)),
            log_gamma=math.log(float(gc.get("density_slope_init", 10.0))),
            log_beta=math.log(float(gc.get("evidence_scale_init", 10.0))),
        ).to(device)
        if not args.untrained:
            if "gate" not in blob:
                raise SystemExit("[eval_dec] EDL checkpoint is missing its density-gate state")
            gate.load_state_dict(blob["gate"])
        gate.eval()

    Z = F.normalize(bank["Z"].float().to(device), dim=-1)
    mem_y = bank["y"].to(device)
    sbert = get_sbert_encoder()
    t_ens_mem = ensemble_text(list(bank["vocab"]), sbert, blob["ensemble"]).to(device)
    print(f"[eval_dec] decoder init {blob.get('init_val_transfer_ba'):.4f} -> best "
          f"{blob.get('best_val_transfer_ba'):.4f} · topk {blob['topk']} tau {blob['tau_retr']}", flush=True)

    per_cell, calibration = {}, {}
    for ds in args.datasets:
        for spec in policy.stream_specs(ds, "primary"):
            try:
                es = load_eval_stream(ds, spec.stream_id, alignment="non_harmonised")
            except FileNotFoundError:
                continue
            result = score_cell(dec, enc, es, Z, mem_y, t_ens_mem, sbert, blob["ensemble"],
                                blob["topk"], blob["tau_retr"], device,
                                raw_labels=args.raw_labels, gate=gate)
            if result is not None:
                f1, cal = result
                per_cell[f"{ds}/{spec.stream_id}"] = round(f1, 1)
                if cal:
                    calibration[f"{ds}/{spec.stream_id}"] = cal
                suffix = f" AURC={cal['aurc']:.3f}" if cal else ""
                print(f"  {ds}/{spec.stream_id:22} F1={f1:.1f}{suffix}", flush=True)
    mean = round(float(np.mean(list(per_cell.values()))), 1)
    # Do NOT print a hardcoded comparison here. The old line quoted "untrained 47.5", which was
    # measured at a DIFFERENT retrieval config (top_k=0, tau=0.03) and a different label-text
    # protocol from anything this script produces -- so a run could print 47.6 and read as
    # "beat the floor" when its like-for-like control was 46.7. Run --untrained to get the control.
    parity = "baseline-parity (bare eval labels)" if args.raw_labels else "NON-PARITY (ensembled candidates)"
    print(f"  MEAN = {mean}   [{parity}, top_k={blob['topk']}, tau={blob['tau_retr']}]", flush=True)
    print(f"  compare ONLY against `--untrained` at these same settings.", flush=True)
    if calibration:
        mean_aurc = float(np.nanmean([c["aurc"] for c in calibration.values()]))
        mean_cov_acc = float(np.nanmean([c["acc@0.8cov"] for c in calibration.values()]))
        print(f"  EDL calibration: mean AURC={mean_aurc:.3f}, "
              f"mean acc@0.8cov={mean_cov_acc:.3f}", flush=True)
    # Separate artifacts so no run can clobber another's PER-CELL breakdown. The arm tags alone
    # were not enough: rebuilding the bank at a new vocabulary reuses the same filenames, and the
    # Phase-2 93-label rerun destroyed the 59-label per-cell numbers that way (only the mean
    # survived, in prose). The bank vocabulary is therefore part of the filename.
    vocab_fp = bank.get("vocab_fp") or vocab_fingerprint(list(bank["vocab"]))
    # Parity is the default, so it is the UNmarked case; a non-parity diagnostic is marked.
    tag = (
        ("_untrained" if args.untrained else "")
        + ("_edl" if blob.get("loss") == "edl" else "")
        + ("" if args.raw_labels else "_enscand")
    )
    out = _DIR / f"eval_decoder{tag}__v{len(bank['vocab'])}_{vocab_fp[:8]}.json"
    out.write_text(json.dumps({"per_cell": per_cell, "mean": mean,
                               "untrained_control": bool(args.untrained),
                               "raw_labels_parity": bool(args.raw_labels),
                               "loss": blob.get("loss", "ce"),
                               "calibration": calibration,
                               "topk": blob["topk"], "tau_retr": blob["tau_retr"],
                               "bank": {"n_windows": int(bank["Z"].shape[0]),
                                        "n_labels": len(bank["vocab"]),
                                        "vocab_fp": vocab_fp,
                                        "backbone_fp": bank["backbone"].get("fingerprint")}},
                              indent=2))
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
