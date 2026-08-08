"""Evaluate no-gradient Phase-B enrollment on held-out deployment streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.scripts.curate import deployment_policy
from data.scripts.labels.canonical_labels import canonicalize
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
    PHASE_B_DEV_DATASETS,
    PHASE_B_TEST_DATASETS,
    SOFT_RETRIEVAL_TEMPERATURE_END,
    PhaseBPolicy,
)
from training.evidence.runtime_memory import build_enrollment_memory
from training.evidence.train_patch_decoder import decode_adaptation_episode
from training.tokenizer.eval_transfer import build_encoder, encode_dataset_detailed
from training.tokenizer.pretrain_data import _stream_gravity_state, stream_channel_descriptions

_REPO = Path(__file__).resolve().parents[2]
_OUT = Path(__file__).resolve().parent / "outputs"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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


@dataclass(frozen=True)
class EnrollmentSubjectPlan:
    subject: object
    candidate_names: tuple[str, ...]
    support_rows: tuple[tuple[int, ...], ...]
    query_rows: np.ndarray


def paired_subject_summary(subject_results: dict, *, seed: int, samples: int = 2_000) -> dict:
    """Summarize controls with subjects as the independent unit and paired bootstrap CIs."""
    fields = {
        "learned": "f1_macro",
        "identity": "identity_f1_macro",
        "support_removed": "support_removed_f1_macro",
        "support_label_shuffled": "support_label_shuffled_f1_macro",
        "prototype": "prototype_f1_macro",
        "ridge_head": "ridge_head_f1_macro",
    }
    result = {"n_subjects": len(subject_results)}
    values = {
        name: np.asarray([
            record[field] for record in subject_results.values()
            if record.get(field) is not None
        ], dtype=np.float64)
        for name, field in fields.items()
    }
    for name, scores in values.items():
        result[f"{name}_f1_macro"] = float(scores.mean()) if len(scores) else None

    comparisons = {
        "adaptation_gain_over_identity": "identity_f1_macro",
        "gain_over_support_removed": "support_removed_f1_macro",
        "gain_over_support_label_shuffled": "support_label_shuffled_f1_macro",
        "gain_over_prototype": "prototype_f1_macro",
        "gain_over_ridge_head": "ridge_head_f1_macro",
    }
    rng = np.random.default_rng(seed)
    for name, control_field in comparisons.items():
        paired = np.asarray([
            record["f1_macro"] - record[control_field]
            for record in subject_results.values()
            if record.get(control_field) is not None
        ], dtype=np.float64)
        if not len(paired):
            result[name] = None
            result[f"{name}_ci95"] = None
            continue
        result[name] = float(paired.mean())
        if len(paired) < 2:
            result[f"{name}_ci95"] = None
            continue
        draws = rng.choice(paired, size=(samples, len(paired)), replace=True).mean(1)
        result[f"{name}_ci95"] = [
            float(value) for value in np.quantile(draws, [0.025, 0.975])
        ]
    result["bootstrap_samples"] = samples if len(subject_results) >= 2 else 0
    result["independent_unit"] = "subject"
    return result


def _execution_quality(execution_ids: np.ndarray) -> dict[str, float | bool]:
    _, counts = np.unique(execution_ids, return_counts=True)
    singleton_share = float(np.mean(counts == 1)) if len(counts) else 1.0
    return {
        "unique_executions": int(len(counts)),
        "singleton_execution_share": singleton_share,
        "window_level_ids": bool(singleton_share > 0.95),
    }


def build_paired_enrollment_plans(
    labels: np.ndarray,
    subjects: np.ndarray,
    execution_ids: np.ndarray,
    eval_labels: list[str],
    *,
    requested_support: list[int],
    mode: str,
    seed: int,
) -> tuple[list[EnrollmentSubjectPlan], dict]:
    """Freeze one candidate/query cohort and nested support prefixes for a complete curve."""
    positive = sorted({int(value) for value in requested_support if value > 0})
    quality = _execution_quality(execution_ids)
    if mode == "same_subject" and positive and quality["window_level_ids"]:
        return [], {
            **quality,
            "status": "unverified_window_level_execution_ids",
            "support_ceiling": 0,
        }

    unique_subjects = np.unique(subjects)

    def candidates_for(subject, support_count: int) -> list[str]:
        result = []
        for label in eval_labels:
            same = np.flatnonzero((labels == label) & (subjects == subject))
            if mode == "same_subject":
                feasible = len(np.unique(execution_ids[same])) >= support_count + 1
            else:
                other = np.flatnonzero((labels == label) & (subjects != subject))
                feasible = len(same) > 0 and len(np.unique(execution_ids[other])) >= support_count
            if feasible:
                result.append(label)
        return result

    support_ceiling = 0
    eligible_subjects: list[object] = []
    for support_count in sorted(positive, reverse=True):
        feasible = [
            subject for subject in unique_subjects
            if len(candidates_for(subject, support_count)) >= 2
        ]
        if feasible:
            support_ceiling = support_count
            eligible_subjects = feasible
            break
    if support_ceiling == 0:
        eligible_subjects = [
            subject for subject in unique_subjects
            if len(candidates_for(subject, 0)) >= 2
        ]

    plans: list[EnrollmentSubjectPlan] = []
    for subject_index, subject in enumerate(eligible_subjects):
        candidate_names = candidates_for(subject, support_ceiling)
        rng = np.random.default_rng(seed + 10_007 * subject_index)
        support_by_candidate: list[tuple[int, ...]] = []
        query_parts = []
        for label in candidate_names:
            same = np.flatnonzero((labels == label) & (subjects == subject))
            source = same if mode == "same_subject" else np.flatnonzero(
                (labels == label) & (subjects != subject)
            )
            source_executions = np.unique(execution_ids[source])
            source_executions = source_executions[rng.permutation(len(source_executions))]
            selected_executions = source_executions[:support_ceiling]
            selected_rows = tuple(
                int(rng.choice(source[execution_ids[source] == execution]))
                for execution in selected_executions
            )
            support_by_candidate.append(selected_rows)
            query = (
                same[~np.isin(execution_ids[same], selected_executions)]
                if mode == "same_subject" else same
            )
            query_parts.append(query)
        query_rows = np.concatenate(query_parts).astype(np.int64, copy=False)
        if len(query_rows):
            plans.append(EnrollmentSubjectPlan(
                subject=subject,
                candidate_names=tuple(candidate_names),
                support_rows=tuple(support_by_candidate),
                query_rows=np.sort(query_rows),
            ))
    return plans, {
        **quality,
        "status": "ok" if plans else "insufficient_independent_executions",
        "support_ceiling": support_ceiling,
        "subjects": len(plans),
        "candidate_count_min": min((len(plan.candidate_names) for plan in plans), default=0),
        "candidate_count_median": float(np.median([
            len(plan.candidate_names) for plan in plans
        ])) if plans else 0.0,
        "candidate_count_max": max((len(plan.candidate_names) for plan in plans), default=0),
    }


def _few_shot_baselines(encoded, support, support_position, query_rows, n_candidates, device):
    if len(support) == 0:
        return None, None
    pooled = F.normalize(torch.as_tensor(encoded["pooled"]).float().to(device), dim=-1)
    x = pooled[torch.as_tensor(support, device=device)]
    position = torch.as_tensor(support_position, device=device, dtype=torch.long)
    q = pooled[torch.as_tensor(query_rows, device=device)]
    centroids = torch.stack([
        F.normalize(x[position.eq(candidate)].mean(0), dim=0)
        for candidate in range(n_candidates)
    ])
    prototype = (q @ centroids.T).argmax(1).cpu().numpy()
    target = F.one_hot(position, n_candidates).float()
    # Closed-form L2-regularized linear head in the sample-space dual. It is a genuine fitted
    # few-shot comparator but is deterministic and much cheaper than an optimizer loop per subject.
    gram = x @ x.T
    alpha = torch.linalg.solve(
        gram + 1.0 * torch.eye(len(x), device=device, dtype=x.dtype), target
    )
    weight = x.T @ alpha
    ridge = (q @ weight).argmax(1).cpu().numpy()
    return prototype, ridge


@torch.no_grad()
def score_enrollment_cell(
    encoded,
    labels,
    plans,
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
    all_removed_pred, all_shuffled_pred = [], []
    all_prototype_pred, all_ridge_pred = [], []
    subject_results = {}
    for subject_index, plan in enumerate(plans):
        candidate_names = list(plan.candidate_names)
        query_rows = plan.query_rows
        support = np.asarray([
            row for rows in plan.support_rows for row in rows[:support_count]
        ], dtype=np.int64)
        support_position = np.asarray([
            position for position, rows in enumerate(plan.support_rows)
            for _ in rows[:support_count]
        ], dtype=np.int64)
        rng = np.random.default_rng(seed + 50_021 * subject_index)
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
        def make_memory(position):
            memory = build_enrollment_memory(
                base_bank, base_rows, base_selector_z, encoded,
                torch.from_numpy(support), torch.from_numpy(position),
                canonical_text, label_set.embeddings,
            )
            return memory, retriever.build_index(memory.selector_z), balanced_memory_log_prior(
                memory.bank["patch"], memory.index_rows, device
            )

        enrollment, memory_index, row_prior = make_memory(support_position)
        removed, removed_index, removed_prior = build_enrollment_memory(
            base_bank, base_rows, base_selector_z, encoded,
            torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
            canonical_text, label_set.embeddings,
        ), None, None
        removed_index = retriever.build_index(removed.selector_z)
        removed_prior = balanced_memory_log_prior(
            removed.bank["patch"], removed.index_rows, device
        )
        shuffled = None
        if support_count > 0:
            shuffled_position = (support_position + 1) % len(candidate_names)
            shuffled = make_memory(shuffled_position)
        position_by_name = {name: i for i, name in enumerate(candidate_names)}
        subject_true, subject_pred, subject_identity = [], [], []
        subject_removed, subject_shuffled = [], []
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
            removed_view = removed.episode_view(
                query, target_position, label_mode=label_mode
            )
            removed_logits, _ = decode_adaptation_episode(
                decoder, retriever, removed.bank, removed.index_rows,
                removed.selector_z, removed_index, query, removed_view,
                removed.canonical_text, removed.candidate_text,
                removed_prior, policy=policy,
                soft_tau=SOFT_RETRIEVAL_TEMPERATURE_END,
                rng=np.random.default_rng(seed + subject_index),
            )
            shuffled_logits = None
            if shuffled is not None:
                shuffled_memory, shuffled_index, shuffled_prior = shuffled
                shuffled_view = shuffled_memory.episode_view(
                    query, target_position, label_mode=label_mode
                )
                shuffled_logits, _ = decode_adaptation_episode(
                    decoder, retriever, shuffled_memory.bank, shuffled_memory.index_rows,
                    shuffled_memory.selector_z, shuffled_index, query, shuffled_view,
                    shuffled_memory.canonical_text, shuffled_memory.candidate_text,
                    shuffled_prior, policy=policy,
                    soft_tau=SOFT_RETRIEVAL_TEMPERATURE_END,
                    rng=np.random.default_rng(seed + subject_index),
                )
            prediction = logits.argmax(1).cpu().tolist()
            identity_prediction = aux["identity_logits"].argmax(1).cpu().tolist()
            removed_prediction = removed_logits.argmax(1).cpu().tolist()
            shuffled_prediction = (
                shuffled_logits.argmax(1).cpu().tolist()
                if shuffled_logits is not None else removed_prediction
            )
            truth = [str(labels[int(row)]) for row in rows.tolist()]
            predicted = [candidate_names[index] for index in prediction]
            identity = [candidate_names[index] for index in identity_prediction]
            removed_names = [candidate_names[index] for index in removed_prediction]
            shuffled_names = [candidate_names[index] for index in shuffled_prediction]
            all_true.extend(truth); subject_true.extend(truth)
            all_pred.extend(predicted); subject_pred.extend(predicted)
            all_identity_pred.extend(identity); subject_identity.extend(identity)
            all_removed_pred.extend(removed_names); subject_removed.extend(removed_names)
            all_shuffled_pred.extend(shuffled_names); subject_shuffled.extend(shuffled_names)
        prototype, ridge = _few_shot_baselines(
            encoded, support, support_position, query_rows, len(candidate_names), device
        )
        if prototype is not None:
            prototype_names = [candidate_names[index] for index in prototype]
            ridge_names = [candidate_names[index] for index in ridge]
            all_prototype_pred.extend(prototype_names)
            all_ridge_pred.extend(ridge_names)
        else:
            prototype_names = ridge_names = None
        subject_results[str(plan.subject)] = {
            "f1_macro": float(classification_metrics(subject_true, subject_pred)["f1_macro"]),
            "identity_f1_macro": float(
                classification_metrics(subject_true, subject_identity)["f1_macro"]
            ),
            "support_removed_f1_macro": float(
                classification_metrics(subject_true, subject_removed)["f1_macro"]
            ),
            "support_label_shuffled_f1_macro": float(
                classification_metrics(subject_true, subject_shuffled)["f1_macro"]
            ),
            "prototype_f1_macro": (
                float(classification_metrics(subject_true, prototype_names)["f1_macro"])
                if prototype_names is not None else None
            ),
            "ridge_head_f1_macro": (
                float(classification_metrics(subject_true, ridge_names)["f1_macro"])
                if ridge_names is not None else None
            ),
            "queries": len(subject_true),
            "candidate_count": len(candidate_names),
        }
    if not all_true:
        return {
            "status": "insufficient_independent_executions",
            "queries": 0,
            "subjects": 0,
        }
    metrics = classification_metrics(all_true, all_pred)
    identity_metrics = classification_metrics(all_true, all_identity_pred)
    removed_metrics = classification_metrics(all_true, all_removed_pred)
    shuffled_metrics = classification_metrics(all_true, all_shuffled_pred)
    bank_vocab = set(base_bank["vocab"])
    seen_mask = np.asarray([canonicalize(label) in bank_vocab for label in all_true])

    def subset_f1(prediction, mask):
        return float(classification_metrics(
            np.asarray(all_true, dtype=object)[mask].tolist(),
            np.asarray(prediction, dtype=object)[mask].tolist(),
        )["f1_macro"]) if bool(mask.any()) else float("nan")

    return {
        "f1_macro": float(metrics["f1_macro"]),
        "identity_f1_macro": float(identity_metrics["f1_macro"]),
        "adaptation_f1_gain": float(metrics["f1_macro"] - identity_metrics["f1_macro"]),
        "support_removed_f1_macro": float(removed_metrics["f1_macro"]),
        "support_removal_f1_gain": float(
            metrics["f1_macro"] - removed_metrics["f1_macro"]
        ),
        "support_label_shuffled_f1_macro": float(shuffled_metrics["f1_macro"]),
        "correct_support_label_f1_gain": float(
            metrics["f1_macro"] - shuffled_metrics["f1_macro"]
        ),
        "prototype_f1_macro": (
            float(classification_metrics(all_true, all_prototype_pred)["f1_macro"])
            if all_prototype_pred else None
        ),
        "ridge_head_f1_macro": (
            float(classification_metrics(all_true, all_ridge_pred)["f1_macro"])
            if all_ridge_pred else None
        ),
        "seen_concept_f1_macro": subset_f1(all_pred, seen_mask),
        "unseen_concept_f1_macro": subset_f1(all_pred, ~seen_mask),
        "seen_concept_queries": int(seen_mask.sum()),
        "unseen_concept_queries": int((~seen_mask).sum()),
        "accuracy": float(np.mean(np.asarray(all_true) == np.asarray(all_pred)) * 100.0),
        "identity_accuracy": float(
            np.mean(np.asarray(all_true) == np.asarray(all_identity_pred)) * 100.0
        ),
        "queries": len(all_true),
        "subjects": len(plans),
        "subject_results": subject_results,
        "subject_macro": paired_subject_summary(
            subject_results, seed=seed + 700_001 + support_count
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get(
        "HALO_CKPT", _REPO / "training/tokenizer/outputs/phase_a_headline/best.pt"
    )))
    parser.add_argument("--bank", type=Path, default=_OUT / "memory_bank.pt")
    parser.add_argument("--predictor", type=Path, default=_OUT / "patch_evidence_predictor.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="explicit override; otherwise --protocol-role selects the sealed roster")
    parser.add_argument("--protocol-role", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--support", nargs="*", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--modes", nargs="*", choices=("same_subject", "cross_subject"),
                        default=["same_subject", "cross_subject"])
    parser.add_argument("--random-aliases", action="store_true")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    explicit_datasets = args.datasets is not None
    if args.datasets is None:
        args.datasets = {
            "dev": list(PHASE_B_DEV_DATASETS),
            "test": list(PHASE_B_TEST_DATASETS),
            "all": list(deployment_policy.PRIMARY_EVAL_DATASETS),
        }[args.protocol_role]
    if any(value < 0 for value in args.support):
        parser.error("support counts must be nonnegative")
    device = resolve_device(args.device)

    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    assert_bank_current(bank, context="eval_enrollment")
    assert_patch_bank(bank, context="eval_enrollment")
    predictor = torch.load(args.predictor, map_location="cpu", weights_only=True)
    predictor_fp = hashlib.sha256(args.predictor.read_bytes()).hexdigest()
    assert_artifact_matches_bank(
        predictor, bank, context="eval_enrollment", artifact_name="patch evidence predictor"
    )
    if predictor.get("training_regime") != PHASE_B_TRAINING_REGIME:
        raise SystemExit("enrollment evaluation requires the current adaptation-trained predictor")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert_bank_matches_backbone(bank, checkpoint, context="eval_enrollment")
    checkpoint_fp = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checkpoint_fp != bank["backbone"].get("fingerprint"):
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
    protocol = {}
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
                plans, coverage = build_paired_enrollment_plans(
                    labels, subjects, execution_ids, list(stream.eval_labels),
                    requested_support=args.support, mode=mode, seed=args.seed,
                )
                protocol_key = f"{dataset}/{stream_spec.stream_id}/{mode}"
                protocol[protocol_key] = coverage
                for support_count in args.support:
                    if args.random_aliases and support_count == 0:
                        continue
                    key = f"{dataset}/{stream_spec.stream_id}/{mode}/k{support_count}"
                    if not plans:
                        result = {**coverage, "queries": 0}
                    elif support_count > int(coverage["support_ceiling"]):
                        result = {
                            "status": "above_paired_support_ceiling",
                            "support_ceiling": int(coverage["support_ceiling"]),
                            "subjects": len(plans),
                            "queries": 0,
                        }
                    else:
                        result = score_enrollment_cell(
                            encoded, labels, plans, bank,
                            base_rows, base_selector_z, decoder, retriever,
                            canonical_text, sbert, aliases, policy, device,
                            support_count=support_count, mode=mode,
                            random_aliases=args.random_aliases, batch_size=args.batch,
                            seed=args.seed,
                        )
                        result["paired_protocol"] = coverage
                    results[key] = result
                    if result.get("status"):
                        print(f"{key}: skipped ({result['status']})", flush=True)
                    else:
                        print(
                            f"{key}: F1={result['f1_macro']:.1f} "
                            f"identity={result['identity_f1_macro']:.1f} "
                            f"removed={result['support_removed_f1_macro']:.1f} "
                            f"shuffled={result['support_label_shuffled_f1_macro']:.1f} "
                            f"n={result['queries']}",
                            flush=True,
                        )
    suffix = "_aliases" if args.random_aliases else ""
    protocol_tag = "custom" if explicit_datasets else args.protocol_role
    out = _OUT / f"eval_enrollment_{protocol_tag}{suffix}.json"
    _write_json(out, {
        "results": results,
        "random_aliases": bool(args.random_aliases),
        "support": args.support,
        "modes": args.modes,
        "protocol_role": args.protocol_role,
        "dataset_selection": "explicit" if explicit_datasets else "protocol_roster",
        "datasets": args.datasets,
        "seed": args.seed,
        "batch_size": args.batch,
        "protocol": protocol,
        "curve_policy": "fixed_subject_candidate_query_cohort_with_nested_execution_support",
        "controls": [
            "identity_decoder", "support_removed", "support_labels_cyclically_shifted",
            "prototype", "l2_ridge_head",
        ],
        "bank": str(args.bank),
        "predictor": str(args.predictor),
        "predictor_fp": predictor_fp,
        "checkpoint": str(args.checkpoint),
        "checkpoint_fp": checkpoint_fp,
        "bank_fp": bank.get("bank_fp"),
    })
    print(f"-> {out}")


if __name__ == "__main__":
    main()
