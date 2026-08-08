"""Evaluate no-gradient Phase-B enrollment on held-out deployment streams."""

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
from eval.data import load_eval_stream
from eval.scoring import classification_metrics, get_sbert_encoder
from model.evidence.decoder import DecoderConfig, EvidenceDecoder
from model.evidence.patch_retrieval import PatchSubspaceRetriever
from training.evidence.bank_guard import (
    assert_artifact_matches_bank,
    assert_bank_current,
    assert_bank_matches_backbone,
    assert_patch_bank,
    assert_embedding_path_current,
    assert_patch_embedding_path_current,
)
from training.evidence.episode_labels import encode_neutral_aliases, episode_label_set
from training.evidence.device import resolve_device
from training.evidence.labeltext import ensemble_text
from training.evidence.live_encoder import SourcePatchEncoder
from training.evidence.patch_episodes import (
    PatchTable,
    balanced_memory_log_prior,
    queries_from_encoded,
)
from training.evidence.policy import (
    ACTIVE_WINDOWS_PER_LABEL,
    PHASE_B_TRAINING_REGIME,
    SOFT_RETRIEVAL_TEMPERATURE_END,
    PhaseBPolicy,
)
from training.evidence.runtime_memory import build_enrollment_memory
from training.evidence.train_patch_decoder import decode_adaptation_episode
from training.tokenizer.eval_transfer import build_encoder, encode_dataset_detailed
from training.tokenizer.pretrain_data import _stream_gravity_state, stream_channel_descriptions

_REPO = Path(__file__).resolve().parents[2]
_OUT = Path(__file__).resolve().parent / "outputs"


def _support_and_query_rows(
    labels: np.ndarray,
    subjects: np.ndarray,
    execution_ids: np.ndarray,
    subject,
    candidate_names: list[str],
    *,
    support_count: int,
    mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support, support_position, query = [], [], []
    for position, label in enumerate(candidate_names):
        same = np.flatnonzero((labels == label) & (subjects == subject))
        other = np.flatnonzero((labels == label) & (subjects != subject))
        source = same if mode == "same_subject" else other
        source_executions = np.unique(execution_ids[source])
        if len(source_executions) < support_count:
            raise ValueError(f"{label}: insufficient {mode} enrollment support")
        chosen_execution = np.asarray(
            rng.choice(source_executions, size=support_count, replace=False), dtype=object
        )
        chosen = np.asarray([
            rng.choice(source[execution_ids[source] == execution])
            for execution in chosen_execution
        ], dtype=np.int64)
        support.extend(chosen.tolist())
        support_position.extend([position] * support_count)
        remaining = (
            same[~np.isin(execution_ids[same], chosen_execution)]
            if mode == "same_subject" else same
        )
        query.extend(remaining.tolist())
    return (
        np.asarray(support, dtype=np.int64),
        np.asarray(support_position, dtype=np.int64),
        np.asarray(query, dtype=np.int64),
    )


@torch.no_grad()
def score_enrollment_cell(
    encoded,
    labels,
    subjects,
    execution_ids,
    eval_labels,
    base_bank,
    base_rows,
    base_selector_z,
    decoder,
    retriever,
    canonical_text,
    sbert,
    alias_embeddings,
    policy,
    device,
    *,
    support_count: int,
    mode: str,
    random_aliases: bool,
    batch_size: int,
    seed: int,
):
    all_true, all_pred, all_identity_pred = [], [], []
    evaluated_subjects = 0
    for subject_index, subject in enumerate(np.unique(subjects)):
        same_subject_labels = set()
        for label in eval_labels:
            same = np.flatnonzero((labels == label) & (subjects == subject))
            needed = support_count + 1 if mode == "same_subject" else 1
            if len(np.unique(execution_ids[same])) >= needed:
                same_subject_labels.add(label)
        if mode == "cross_subject":
            same_subject_labels = {
                label for label in same_subject_labels
                if len(np.unique(execution_ids[
                    (labels == label) & (subjects != subject)
                ])) >= support_count
            }
        candidate_names = [label for label in eval_labels if label in same_subject_labels]
        if len(candidate_names) < 2:
            continue
        rng = np.random.default_rng(seed + subject_index)
        support, support_position, query_rows = _support_and_query_rows(
            labels, subjects, execution_ids, subject, candidate_names,
            support_count=support_count, mode=mode, rng=rng,
        )
        if not len(query_rows):
            continue
        coherent = ensemble_text(candidate_names, sbert, 1).to(device)
        label_mode = "random_alias" if random_aliases else "coherent"
        label_set = episode_label_set(
            torch.arange(len(candidate_names), device=device),
            coherent,
            mode=label_mode,
            rng=rng,
            alias_embeddings=alias_embeddings,
            canonical_names=candidate_names,
        )
        enrollment = build_enrollment_memory(
            base_bank,
            base_rows,
            base_selector_z,
            encoded,
            torch.from_numpy(support),
            torch.from_numpy(support_position),
            canonical_text,
            label_set.embeddings,
        )
        memory_index = retriever.build_index(enrollment.selector_z)
        row_prior = balanced_memory_log_prior(
            enrollment.bank["patch"], enrollment.index_rows, device
        )
        position_by_name = {name: i for i, name in enumerate(candidate_names)}
        for start in range(0, len(query_rows), batch_size):
            rows = torch.from_numpy(query_rows[start:start + batch_size])
            query = queries_from_encoded(
                encoded, rows, device, sensor_id=enrollment.runtime_sensor_id
            )
            target_position = torch.tensor([
                position_by_name[str(labels[int(row)])] for row in rows.tolist()
            ], device=device)
            view = enrollment.episode_view(
                query, target_position, label_mode=label_mode
            )
            logits, aux = decode_adaptation_episode(
                decoder, retriever, enrollment.bank, enrollment.index_rows,
                enrollment.selector_z, memory_index, query, view,
                enrollment.canonical_text, enrollment.candidate_text,
                row_prior, policy=policy,
                soft_tau=SOFT_RETRIEVAL_TEMPERATURE_END,
                rng=np.random.default_rng(seed + subject_index),
            )
            prediction = logits.argmax(1).cpu().tolist()
            identity_prediction = aux["identity_logits"].argmax(1).cpu().tolist()
            all_pred.extend(candidate_names[index] for index in prediction)
            all_identity_pred.extend(candidate_names[index] for index in identity_prediction)
            all_true.extend(str(labels[int(row)]) for row in rows.tolist())
        evaluated_subjects += 1
    if not all_true:
        return {
            "status": "insufficient_independent_executions",
            "queries": 0,
            "subjects": 0,
        }
    metrics = classification_metrics(all_true, all_pred)
    identity_metrics = classification_metrics(all_true, all_identity_pred)
    return {
        "f1_macro": float(metrics["f1_macro"]),
        "identity_f1_macro": float(identity_metrics["f1_macro"]),
        "adaptation_f1_gain": float(metrics["f1_macro"] - identity_metrics["f1_macro"]),
        "accuracy": float(np.mean(np.asarray(all_true) == np.asarray(all_pred)) * 100.0),
        "identity_accuracy": float(
            np.mean(np.asarray(all_true) == np.asarray(all_identity_pred)) * 100.0
        ),
        "queries": len(all_true),
        "subjects": evaluated_subjects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get(
        "HALO_CKPT", _REPO / "training/tokenizer/outputs/phase_a_headline/best.pt"
    )))
    parser.add_argument("--bank", type=Path, default=_OUT / "memory_bank.pt")
    parser.add_argument("--predictor", type=Path, default=_OUT / "patch_evidence_predictor.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="*", default=list(deployment_policy.PRIMARY_EVAL_DATASETS))
    parser.add_argument("--support", nargs="*", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--modes", nargs="*", choices=("same_subject", "cross_subject"),
                        default=["same_subject", "cross_subject"])
    parser.add_argument("--random-aliases", action="store_true")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    if any(value < 0 for value in args.support):
        parser.error("support counts must be nonnegative")
    device = resolve_device(args.device)

    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    assert_bank_current(bank, context="eval_enrollment")
    assert_patch_bank(bank, context="eval_enrollment")
    predictor = torch.load(args.predictor, map_location="cpu", weights_only=True)
    assert_artifact_matches_bank(
        predictor, bank, context="eval_enrollment", artifact_name="patch evidence predictor"
    )
    if predictor.get("training_regime") != PHASE_B_TRAINING_REGIME:
        raise SystemExit("enrollment evaluation requires the current adaptation-trained predictor")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert_bank_matches_backbone(bank, checkpoint, context="eval_enrollment")
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() \
            != bank["backbone"].get("fingerprint"):
        raise SystemExit("encoder checkpoint != patch-bank backbone")
    encoder = build_encoder(checkpoint, device)
    assert_embedding_path_current(bank, encoder, device, context="eval_enrollment")
    assert_patch_embedding_path_current(bank, encoder, device, context="eval_enrollment")
    cfg = predictor["cfg"]
    if cfg.get("tokenizer_mode") == "ema_finetune":
        encoder.load_state_dict(predictor["tokenizer_ema"])
    decoder = EvidenceDecoder(DecoderConfig(
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        candidate_tokens=True, structural_metadata=True,
        support_role=cfg.get("support_role", False),
        n_retrieval_heads=cfg.get("n_retrieval_heads", cfg["n_subspaces"]),
    )).to(device)
    decoder.load_state_dict(predictor["decoder"]); decoder.eval()
    retriever = PatchSubspaceRetriever(
        cfg["d_model"], cfg["n_subspaces"], cfg["subspace_dim"], cfg["subspace_ema"]
    ).to(device)
    retriever.load_state_dict(predictor["retriever"]); retriever.eval()
    policy = PhaseBPolicy(
        int(predictor["retrieval_cfg"]["evidence_budget"]),
        cfg.get("tokenizer_mode", "frozen"),
    )

    table = PatchTable(bank)
    base_rows = table.sample_index_rows(
        torch.ones(len(bank["Z"]), dtype=torch.bool), ACTIVE_WINDOWS_PER_LABEL,
        np.random.default_rng(predictor["retrieval_cfg"]["index_seed"]),
    )
    if policy.tokenizer_mode == "ema_finetune":
        source = SourcePatchEncoder(bank, device)
        base_selector_z = source.encode_patch_rows(
            base_rows, encoder, requires_grad=False
        ).to(device)
    else:
        base_selector_z = bank["patch"]["Z"][base_rows].float().to(device)
    base_selector_z = F.normalize(base_selector_z, dim=-1)
    sbert = get_sbert_encoder()
    canonical_text = ensemble_text(list(bank["vocab"]), sbert, 8, train_only=True).to(device)
    aliases = encode_neutral_aliases(sbert, device)

    results = {}
    for dataset in args.datasets:
        for stream_spec in deployment_policy.stream_specs(dataset, "primary"):
            try:
                stream = load_eval_stream(dataset, stream_spec.stream_id, alignment="native")
            except FileNotFoundError:
                continue
            encoded = encode_dataset_detailed(
                encoder, np.asarray(stream.windows),
                stream_channel_descriptions(stream.dataset, stream.stream),
                device, float(stream.rate_hz),
                _stream_gravity_state(stream.dataset, stream.stream),
                channel_mask=stream.mask, dataset=stream.dataset, stream=stream.stream,
            )
            labels = np.asarray(stream.gt, dtype=object)
            subjects = np.asarray(stream.subjects, dtype=object)
            if stream.execution_ids is None:
                raise RuntimeError(f"{dataset}/{stream.stream}: native grid has no execution ids")
            execution_ids = np.asarray(stream.execution_ids, dtype=object)
            for mode in args.modes:
                for support_count in args.support:
                    if args.random_aliases and support_count == 0:
                        continue
                    result = score_enrollment_cell(
                        encoded, labels, subjects, execution_ids,
                        list(stream.eval_labels), bank,
                        base_rows, base_selector_z, decoder, retriever,
                        canonical_text, sbert, aliases, policy, device,
                        support_count=support_count, mode=mode,
                        random_aliases=args.random_aliases, batch_size=args.batch,
                        seed=args.seed,
                    )
                    key = f"{dataset}/{stream_spec.stream_id}/{mode}/k{support_count}"
                    results[key] = result
                    if result.get("status"):
                        print(f"{key}: skipped ({result['status']})", flush=True)
                    else:
                        print(
                            f"{key}: F1={result['f1_macro']:.1f} "
                            f"identity={result['identity_f1_macro']:.1f} n={result['queries']}",
                            flush=True,
                        )
    suffix = "_aliases" if args.random_aliases else ""
    out = _OUT / f"eval_enrollment{suffix}.json"
    out.write_text(json.dumps({
        "results": results,
        "random_aliases": bool(args.random_aliases),
        "support": args.support,
        "modes": args.modes,
        "predictor": str(args.predictor),
        "bank_fp": bank.get("bank_fp"),
    }, indent=2, sort_keys=True) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
