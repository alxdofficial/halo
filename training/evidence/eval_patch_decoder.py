"""Evaluate the trained source-aware patch evidence engine on the ZS-XD protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.scripts.curate import deployment_policy
from data.scripts.labels.canonical_labels import canonicalize
from eval.data import load_eval_stream
from eval.scoring import (
    align_ground_truth_labels,
    classification_metrics,
    filter_ground_truth,
    get_sbert_encoder,
    paired_subject_bootstrap_difference,
    subject_bootstrap_ci,
)
from model.evidence.confidence import (
    EvidenceConfidenceHead,
    acc_at_coverage,
    aurc,
    binary_auprc,
    binary_auroc,
    expected_calibration_error,
)
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
from training.evidence.device import resolve_device
from training.evidence.live_encoder import SourcePatchEncoder
from training.evidence.policy import (
    ACTIVE_WINDOWS_PER_LABEL,
    PHASE_B_DEV_DATASETS,
    PHASE_B_TEST_DATASETS,
    PHASE_B_TRAINING_REGIME,
    PhaseBPolicy,
)
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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
    phaseb_policy,
    device,
    *,
    memory_config_text=None,
    raw_labels=True,
    candidate_ensemble=8,
    batch=16,
    live_source=None,
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
            policy=phaseb_policy,
            memory_config_text=memory_config_text,
            query_config_text=query_config_text,
            live_source=live_source,
            live_encoder=enc if live_source is not None else None,
            live_requires_grad=False,
            query_already_live=live_source is not None,
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

    aligned_gt = align_ground_truth_labels(es.gt, es.eval_labels)
    kept_gt, _, keep = filter_ground_truth(es.gt, es.subjects, es.eval_labels)
    if not len(keep):
        return None
    kept_prediction = list(prediction[keep])
    f1 = float(classification_metrics(kept_gt, kept_prediction)["f1_macro"])
    identity_f1 = float(classification_metrics(
        kept_gt, list(identity_prediction[keep])
    )["f1_macro"])
    bank_vocab = set(bank["vocab"])
    seen_concept = np.asarray([
        canonicalize(label) in bank_vocab for label in kept_gt
    ])

    def subset_f1(values, mask):
        return float(classification_metrics(
            np.asarray(kept_gt, dtype=object)[mask].tolist(),
            np.asarray(values, dtype=object)[mask].tolist(),
        )["f1_macro"]) if bool(mask.any()) else float("nan")

    concept_breakdown = {
        "seen_concept_f1_macro": subset_f1(kept_prediction, seen_concept),
        "unseen_concept_f1_macro": subset_f1(kept_prediction, ~seen_concept),
        "seen_concept_queries": int(seen_concept.sum()),
        "unseen_concept_queries": int((~seen_concept).sum()),
    }
    kept_subjects = np.asarray(es.subjects)[keep]
    sampling_uncertainty = {
        "learned": subject_bootstrap_ci(
            kept_gt, kept_prediction, kept_subjects, metric="f1_macro"
        ),
        "identity": subject_bootstrap_ci(
            kept_gt, list(identity_prediction[keep]), kept_subjects, metric="f1_macro"
        ),
        "adaptation_gain": paired_subject_bootstrap_difference(
            kept_gt, kept_prediction, list(identity_prediction[keep]), kept_subjects,
            metric="f1_macro",
        ),
    }
    if confidence_score is None:
        return f1, identity_f1, None, concept_breakdown, sampling_uncertainty
    correct = np.asarray(kept_prediction) == np.asarray(kept_gt)
    present_score = confidence_score[keep]
    uncertainty = 1.0 - present_score

    # Truth-absent canary: remove one ground-truth label from both the candidate roster and any exact
    # canonical memory rows, then ask the separately calibrated head whether the forced answer is safe.
    absent_scores = []
    vocab_position = {label: index for index, label in enumerate(bank["vocab"])}
    memory_y = torch.as_tensor(bank["patch"]["y"])[index_rows.cpu()].to(device)
    gt_array = np.asarray(aligned_gt, dtype=object)
    for omitted_position, omitted_label in enumerate(es.eval_labels):
        omitted_rows = np.flatnonzero(gt_array == omitted_label)
        if not len(omitted_rows) or len(es.eval_labels) < 2:
            continue
        candidate_keep = torch.tensor([
            index for index in range(len(es.eval_labels)) if index != omitted_position
        ], device=device)
        canonical = canonicalize(omitted_label)
        for start in range(0, len(omitted_rows), batch):
            rows_np = omitted_rows[start:start + batch]
            rows = torch.from_numpy(rows_np)
            query = queries_from_encoded(encoded, rows, device, sensor_id=-1)
            allowed = query.mask.unsqueeze(-1).expand(
                len(rows), query.mask.shape[1], len(index_rows)
            ).clone()
            if canonical in vocab_position:
                allowed &= memory_y.ne(vocab_position[canonical]).view(1, 1, -1)
            query_config_text = (
                query_config_vector.view(1, 1, -1).expand(
                    len(rows), query.mask.shape[1], -1
                ) if query_config_vector is not None else None
            )
            _, absent_aux = decode_patch_queries(
                dec, retriever, bank, index_rows, memory_index,
                query, allowed, candidate_keep, t_memory, candidate_text,
                policy=phaseb_policy,
                memory_config_text=memory_config_text,
                query_config_text=query_config_text,
                live_source=live_source,
                live_encoder=enc if live_source is not None else None,
                live_requires_grad=False,
                query_already_live=live_source is not None,
            )
            absent_scores.extend(torch.sigmoid(
                confidence(absent_aux["confidence_features"])
            ).cpu().tolist())
    absent_score = np.asarray(absent_scores, dtype=np.float32)
    combined_score = np.concatenate([present_score, absent_score])
    combined_target = np.concatenate([
        correct.astype(np.float32), np.zeros(len(absent_score), dtype=np.float32)
    ])
    calibration = {
        "mean_conf_correct": (
            float(present_score[correct].mean()) if correct.any() else float("nan")
        ),
        "mean_conf_incorrect": (
            float(present_score[~correct].mean()) if (~correct).any() else float("nan")
        ),
        "mean_conf_truth_absent": (
            float(absent_score.mean()) if len(absent_score) else float("nan")
        ),
        "truth_absent_p95_confidence": (
            float(np.quantile(absent_score, 0.95)) if len(absent_score) else float("nan")
        ),
        "correctness_auroc": binary_auroc(present_score, correct),
        "correctness_auprc": binary_auprc(present_score, correct),
        "correctness_ece": expected_calibration_error(present_score, correct),
        "correctness_brier": float(np.mean((present_score - correct.astype(float)) ** 2)),
        "answerability_auroc": binary_auroc(combined_score, combined_target),
        "answerability_auprc": binary_auprc(combined_score, combined_target),
        "answerability_ece": expected_calibration_error(combined_score, combined_target),
        "answerability_brier": float(np.mean((combined_score - combined_target) ** 2)),
        "truth_absent_queries": int(len(absent_score)),
        "aurc": aurc(uncertainty, correct),
        "acc@0.5cov": acc_at_coverage(uncertainty, correct, 0.5),
        "acc@0.8cov": acc_at_coverage(uncertainty, correct, 0.8),
        "acc@0.9cov": acc_at_coverage(uncertainty, correct, 0.9),
        "truth_absent_rejection_rate@0.5": (
            float(np.mean(absent_score < 0.5)) if len(absent_score) else float("nan")
        ),
    }
    return f1, identity_f1, calibration, concept_breakdown, sampling_uncertainty


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
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="explicit override; otherwise --protocol-role selects the sealed roster")
    ap.add_argument("--protocol-role", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--ensemble-candidates", dest="raw_labels", action="store_false", default=True,
                    help="NON-PARITY diagnostic; default uses bare eval label strings")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    explicit_datasets = args.datasets is not None
    if args.datasets is None:
        args.datasets = {
            "dev": list(PHASE_B_DEV_DATASETS),
            "test": list(PHASE_B_TEST_DATASETS),
            "all": list(deployment_policy.PRIMARY_EVAL_DATASETS),
        }[args.protocol_role]
    device = resolve_device(args.device)

    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    assert_bank_current(bank, context="eval_patch_decoder")
    assert_patch_bank(bank, context="eval_patch_decoder")
    blob = torch.load(args.predictor, map_location="cpu", weights_only=True)
    predictor_fp = hashlib.sha256(args.predictor.read_bytes()).hexdigest()
    assert_artifact_matches_bank(
        blob, bank, context="eval_patch_decoder", artifact_name="patch evidence predictor"
    )
    if blob.get("objective") != "candidate_cross_entropy":
        raise SystemExit("evaluation requires the consolidated candidate-CE predictor")
    if blob.get("training_regime") != PHASE_B_TRAINING_REGIME:
        raise SystemExit("evaluation requires a predictor trained with the current adaptation regime")
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
        support_role=cfg.get("support_role", False),
        n_retrieval_heads=cfg.get("n_retrieval_heads", cfg["n_subspaces"]),
    )).to(device)
    dec.load_state_dict(blob["decoder"])
    retriever = PatchSubspaceRetriever(
        cfg["d_model"], cfg["n_subspaces"], cfg["subspace_dim"], cfg["subspace_ema"]
    ).to(device)
    retriever.load_state_dict(blob["retriever"])
    retrieval_cfg = blob["retrieval_cfg"]
    phaseb_policy = PhaseBPolicy(
        int(retrieval_cfg["evidence_budget"]), cfg.get("tokenizer_mode", "frozen")
    )
    live_source = None
    if phaseb_policy.tokenizer_mode == "ema_finetune":
        if "tokenizer_ema" not in blob:
            raise SystemExit("fine-tuned predictor is missing its EMA tokenizer state")
        enc.load_state_dict(blob["tokenizer_ema"])
        enc.eval()
        live_source = SourcePatchEncoder(bank, device)
    confidence_fp = None
    if args.confidence is not None:
        confidence = EvidenceConfidenceHead().to(device)
        confidence_blob = torch.load(args.confidence, map_location="cpu", weights_only=True)
        confidence_fp = hashlib.sha256(args.confidence.read_bytes()).hexdigest()
        if confidence_blob.get("predictor_fp") != predictor_fp:
            raise SystemExit("confidence artifact was calibrated for a different predictor")
        if confidence_blob.get("bank_fp") != bank.get("bank_fp"):
            raise SystemExit("confidence artifact was calibrated for a different memory bank")
        confidence.load_state_dict(confidence_blob["confidence"])
        confidence.eval()
    else:
        confidence = None
    dec.eval(); retriever.eval()

    table = PatchTable(bank)
    all_windows = torch.ones(len(bank["Z"]), dtype=torch.bool)
    index_rows = table.sample_index_rows(
        all_windows, ACTIVE_WINDOWS_PER_LABEL,
        np.random.default_rng(retrieval_cfg["index_seed"]),
    )
    memory = (
        bank["patch"]["Z"][index_rows].float().to(device)
        if live_source is None else
        live_source.encode_patch_rows(index_rows, enc, requires_grad=False).to(device)
    )
    memory = F.normalize(memory, dim=-1)
    memory_index = retriever.build_index(memory)
    sbert = get_sbert_encoder()
    t_memory = ensemble_text(list(bank["vocab"]), sbert, 8, train_only=True).to(device)
    memory_config_text = (
        encode_bank_config_text(bank, sbert, device)
        if cfg.get("explicit_config_text", False) else None
    )

    per_cell, identity_control, calibration, concept_breakdown = {}, {}, {}, {}
    sampling_uncertainty = {}
    for dataset in args.datasets:
        for spec in deployment_policy.stream_specs(dataset, "primary"):
            try:
                stream = load_eval_stream(dataset, spec.stream_id, alignment="native")
            except FileNotFoundError:
                continue
            result = score_cell(
                dec, retriever, confidence, enc, stream, bank, index_rows, memory_index,
                t_memory, sbert, phaseb_policy, device, raw_labels=args.raw_labels,
                batch=args.batch, memory_config_text=memory_config_text,
                live_source=live_source,
            )
            if result is None:
                continue
            f1, identity_f1, cell_calibration, cell_concepts, cell_uncertainty = result
            key = f"{dataset}/{spec.stream_id}"
            per_cell[key] = float(f1)
            identity_control[key] = float(identity_f1)
            concept_breakdown[key] = cell_concepts
            sampling_uncertainty[key] = cell_uncertainty
            if cell_calibration is not None:
                calibration[key] = cell_calibration
                print(
                    f"  {key:38s} F1={f1:.1f} identity={identity_f1:.1f} "
                    f"AURC={cell_calibration['aurc']:.3f}", flush=True,
                )
            else:
                print(f"  {key:38s} F1={f1:.1f} identity={identity_f1:.1f}", flush=True)

    if not per_cell:
        raise SystemExit("no evaluable deployment streams were found for the selected protocol")
    mean = float(np.mean(list(per_cell.values())))
    identity_mean = float(np.mean(list(identity_control.values())))
    mean_adaptation_gain = float(np.mean([
        per_cell[key] - identity_control[key] for key in per_cell
    ]))
    parity = "baseline-parity" if args.raw_labels else "NON-PARITY ensembled candidates"
    print(f"  MEAN={mean:.1f} [{parity}, patch evidence]", flush=True)
    vocab_fp = bank.get("vocab_fp") or vocab_fingerprint(bank["vocab"])
    tag = "" if args.raw_labels else "_enscand"
    protocol_tag = "custom" if explicit_datasets else args.protocol_role
    out = _DIR / (
        f"eval_patch_decoder_{protocol_tag}{tag}__v{len(bank['vocab'])}_{vocab_fp[:8]}.json"
    )
    _write_json(out, {
        "per_cell": per_cell, "mean": mean,
        "identity_control": identity_control, "calibration": calibration,
        "identity_mean": identity_mean,
        "mean_adaptation_gain": mean_adaptation_gain,
        "concept_breakdown": concept_breakdown,
        "sampling_uncertainty": sampling_uncertainty,
        "protocol_role": args.protocol_role,
        "dataset_selection": "explicit" if explicit_datasets else "protocol_roster",
        "datasets": args.datasets,
        "batch_size": args.batch,
        "raw_labels_parity": bool(args.raw_labels),
        "memory_schema": bank["schema_version"],
        "retrieval_cfg": retrieval_cfg,
        "bank": str(args.bank),
        "bank_fp": bank.get("bank_fp"),
        "predictor": str(args.predictor),
        "predictor_fp": predictor_fp,
        "checkpoint": str(args.checkpoint),
        "checkpoint_fp": checkpoint_fp,
        "confidence": str(args.confidence) if args.confidence is not None else None,
        "confidence_fp": confidence_fp,
    })
    print(f"-> {out}")


if __name__ == "__main__":
    main()
