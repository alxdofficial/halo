"""Evaluate the trained schema-v2 patch evidence engine on the ZS-XD protocol."""

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
from model.evidence.confidence import EvidenceConfidenceHead, acc_at_coverage, aurc
from model.evidence.decoder import DecoderConfig, EvidenceDecoder
from model.evidence.patch_retrieval import PatchSubspaceRetriever
from training.evidence.bank_guard import (
    assert_artifact_matches_bank,
    assert_bank_current,
    assert_bank_matches_backbone,
    assert_embedding_path_current,
    assert_patch_bank,
    assert_patch_embedding_path_current,
    vocab_fingerprint,
)
from training.evidence.labeltext import ensemble_text
from training.evidence.patch_episodes import PatchTable, queries_from_encoded
from training.evidence.train_patch_decoder import (
    decode_patch_queries,
    encode_bank_config_text,
    load_activity_families,
)
from training.tokenizer.eval_transfer import build_encoder, encode_dataset_detailed
from training.tokenizer.pretrain_data import _stream_gravity_state, stream_channel_descriptions

_REPO = Path(__file__).resolve().parents[2]
_DIR = Path(__file__).resolve().parent / "outputs"


@torch.no_grad()
def score_cell(
    dec,
    retriever,
    confidence,
    enc,
    es,
    bank,
    index_rows,
    memory_index,
    t_memory,
    sbert,
    retrieval_cfg,
    device,
    *,
    memory_config_text=None,
    raw_labels=True,
    candidate_ensemble=8,
    batch=16,
):
    encoded = encode_dataset_detailed(
        enc, np.asarray(es.windows),
        stream_channel_descriptions(es.dataset, es.stream),
        device, float(es.rate_hz), _stream_gravity_state(es.dataset, es.stream),
        channel_mask=es.mask, dataset=es.dataset, stream=es.stream,
    )
    candidate_text = ensemble_text(
        es.eval_labels, sbert, 1 if raw_labels else candidate_ensemble,
        use_descriptions=not raw_labels,
    ).to(device)
    candidate_ids = torch.arange(len(es.eval_labels), device=device)
    query_config_vector = None
    if memory_config_text is not None:
        prompt = "sensor configuration: " + "; ".join(
            stream_channel_descriptions(es.dataset, es.stream)
        ) + f"; sampling rate {float(es.rate_hz):g} hertz"
        query_config_vector = torch.from_numpy(
            sbert([prompt]).astype(np.float32)
        ).to(device)[0]
    prediction = np.empty(len(es.windows), dtype=object)
    identity_prediction = np.empty(len(es.windows), dtype=object)
    confidence_score = np.empty(len(es.windows), dtype=np.float32) \
        if confidence is not None else None
    for start in range(0, len(es.windows), batch):
        stop = min(start + batch, len(es.windows))
        rows = torch.arange(start, stop)
        query = queries_from_encoded(encoded, rows, device, sensor_id=-1)
        allowed = query.mask.unsqueeze(-1).expand(
            len(rows), query.mask.shape[1], len(index_rows)
        )
        query_config_text = (
            query_config_vector.view(1, 1, -1).expand(
                len(rows), query.mask.shape[1], -1
            )
            if query_config_vector is not None else None
        )
        logits, aux = decode_patch_queries(
            dec, retriever, bank, index_rows, memory_index,
            query, allowed, candidate_ids, t_memory, candidate_text,
            topk_per_head=retrieval_cfg["topk_per_head"],
            max_evidence=retrieval_cfg["max_evidence"],
            max_per_window=retrieval_cfg["max_per_window"],
            max_per_label=retrieval_cfg["max_per_label"],
            tau=retrieval_cfg["tau_retr"],
            memory_config_text=memory_config_text,
            query_config_text=query_config_text,
        )
        prediction[start:stop] = [
            es.eval_labels[index] for index in logits.argmax(1).cpu().tolist()
        ]
        identity_prediction[start:stop] = [
            es.eval_labels[index] for index in aux["identity_logits"].argmax(1).cpu().tolist()
        ]
        if confidence_score is not None:
            confidence_score[start:stop] = torch.sigmoid(
                confidence(aux["confidence_features"])
            ).cpu().numpy()

    kept_gt, _, keep = filter_ground_truth(es.gt, es.subjects, es.eval_labels)
    if not len(keep):
        return None
    kept_prediction = list(prediction[keep])
    f1 = float(classification_metrics(kept_gt, kept_prediction)["f1_macro"])
    identity_f1 = float(classification_metrics(
        kept_gt, list(identity_prediction[keep])
    )["f1_macro"])
    if confidence_score is None:
        return f1, identity_f1, None
    correct = np.asarray(kept_prediction) == np.asarray(kept_gt)
    uncertainty = 1.0 - confidence_score[keep]
    calibration = {
        "mean_conf_correct": (
            float(confidence_score[keep][correct].mean()) if correct.any() else float("nan")
        ),
        "mean_conf_incorrect": (
            float(confidence_score[keep][~correct].mean()) if (~correct).any() else float("nan")
        ),
        "aurc": aurc(uncertainty, correct),
        "acc@0.8cov": acc_at_coverage(uncertainty, correct, 0.8),
    }
    return f1, identity_f1, calibration


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path(os.environ.get(
        "HALO_CKPT", _REPO / "training/tokenizer/outputs/phase_a_headline/best.pt"
    )))
    ap.add_argument("--bank", type=Path, default=_DIR / "memory_bank.pt")
    ap.add_argument("--predictor", type=Path, default=_DIR / "patch_evidence_predictor.pt")
    ap.add_argument("--confidence", type=Path, default=None,
                    help="optional separately calibrated confidence artifact")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", nargs="*", default=list(policy.PRIMARY_EVAL_DATASETS))
    ap.add_argument("--ensemble-candidates", dest="raw_labels", action="store_false", default=True,
                    help="NON-PARITY diagnostic; default uses bare eval label strings")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    assert_bank_current(bank, context="eval_patch_decoder")
    assert_patch_bank(bank, context="eval_patch_decoder")
    blob = torch.load(args.predictor, map_location="cpu", weights_only=True)
    assert_artifact_matches_bank(
        blob, bank, context="eval_patch_decoder", artifact_name="patch evidence predictor"
    )
    if blob.get("objective") != "candidate_cross_entropy":
        raise SystemExit("evaluation requires the consolidated candidate-CE predictor")
    if int(blob.get("memory_schema", 0)) != int(bank["schema_version"]):
        raise SystemExit("patch decoder / memory schema mismatch")
    _, _, current_family_fp = load_activity_families(list(bank["vocab"]))
    if blob.get("activity_family_fp") != current_family_fp:
        raise SystemExit("patch decoder / activity-family taxonomy mismatch")
    checkpoint_fp = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checkpoint_fp != bank["backbone"].get("fingerprint"):
        raise SystemExit("encoder checkpoint != patch-bank backbone")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert_bank_matches_backbone(bank, checkpoint, context="eval_patch_decoder")
    enc = build_encoder(checkpoint, device)
    assert_embedding_path_current(bank, enc, device, context="eval_patch_decoder")
    assert_patch_embedding_path_current(bank, enc, device, context="eval_patch_decoder")
    for parameter in enc.parameters():
        parameter.requires_grad_(False)

    cfg = blob["cfg"]
    dec = EvidenceDecoder(DecoderConfig(
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        candidate_tokens=cfg["candidate_tokens"],
        structural_metadata=cfg["structural_metadata"],
    )).to(device)
    dec.load_state_dict(blob["decoder"])
    retriever = PatchSubspaceRetriever(
        cfg["d_model"], cfg["n_subspaces"], cfg["subspace_dim"], cfg["subspace_ema"]
    ).to(device)
    retriever.load_state_dict(blob["retriever"])
    confidence = EvidenceConfidenceHead().to(device)
    if args.confidence is not None:
        confidence_blob = torch.load(args.confidence, map_location="cpu", weights_only=True)
        predictor_fp = hashlib.sha256(args.predictor.read_bytes()).hexdigest()
        if confidence_blob.get("predictor_fp") != predictor_fp:
            raise SystemExit("confidence artifact was calibrated for a different predictor")
        if confidence_blob.get("bank_fp") != bank.get("bank_fp"):
            raise SystemExit("confidence artifact was calibrated for a different memory bank")
        confidence.load_state_dict(confidence_blob["confidence"])
        confidence.eval()
    else:
        confidence = None
    dec.eval(); retriever.eval()

    retrieval_cfg = blob["retrieval_cfg"]
    table = PatchTable(bank)
    all_windows = torch.ones(len(bank["Z"]), dtype=torch.bool)
    index_rows = table.sample_index_rows(
        all_windows, retrieval_cfg["index_per_label"],
        np.random.default_rng(retrieval_cfg["index_seed"]),
    )
    memory = F.normalize(bank["patch"]["Z"][index_rows].float().to(device), dim=-1)
    memory_index = retriever.build_index(memory)
    sbert = get_sbert_encoder()
    t_memory = ensemble_text(list(bank["vocab"]), sbert, 8, train_only=True).to(device)
    memory_config_text = (
        encode_bank_config_text(bank, sbert, device)
        if cfg.get("explicit_config_text", False) else None
    )

    per_cell, identity_control, calibration = {}, {}, {}
    for dataset in args.datasets:
        for spec in policy.stream_specs(dataset, "primary"):
            try:
                stream = load_eval_stream(dataset, spec.stream_id, alignment="non_harmonised")
            except FileNotFoundError:
                continue
            result = score_cell(
                dec, retriever, confidence, enc, stream, bank, index_rows, memory_index,
                t_memory, sbert, retrieval_cfg, device, raw_labels=args.raw_labels,
                batch=args.batch, memory_config_text=memory_config_text,
            )
            if result is None:
                continue
            f1, identity_f1, cell_calibration = result
            key = f"{dataset}/{spec.stream_id}"
            per_cell[key] = round(f1, 1)
            identity_control[key] = round(identity_f1, 1)
            if cell_calibration is not None:
                calibration[key] = cell_calibration
                print(
                    f"  {key:38s} F1={f1:.1f} identity={identity_f1:.1f} "
                    f"AURC={cell_calibration['aurc']:.3f}", flush=True,
                )
            else:
                print(f"  {key:38s} F1={f1:.1f} identity={identity_f1:.1f}", flush=True)

    mean = round(float(np.mean(list(per_cell.values()))), 1)
    parity = "baseline-parity" if args.raw_labels else "NON-PARITY ensembled candidates"
    print(f"  MEAN={mean} [{parity}, patch evidence]", flush=True)
    vocab_fp = bank.get("vocab_fp") or vocab_fingerprint(bank["vocab"])
    tag = "" if args.raw_labels else "_enscand"
    out = _DIR / f"eval_patch_decoder{tag}__v{len(bank['vocab'])}_{vocab_fp[:8]}.json"
    out.write_text(json.dumps({
        "per_cell": per_cell, "mean": mean,
        "identity_control": identity_control, "calibration": calibration,
        "raw_labels_parity": bool(args.raw_labels),
        "memory_schema": bank["schema_version"],
        "retrieval_cfg": retrieval_cfg,
        "bank_fp": bank.get("bank_fp"),
        "predictor": str(args.predictor),
        "confidence": str(args.confidence) if args.confidence is not None else None,
    }, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
