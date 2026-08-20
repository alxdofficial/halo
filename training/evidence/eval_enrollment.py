"""Evaluate no-gradient Phase-B enrollment on held-out deployment streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
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
    get_sbert_encoder,
    macro_f1_fast,
)
from model.evidence.patch_retrieval import PatchSubspaceRetriever
from training.evidence.bank_guard import (
    assert_artifact_matches_bank,
    assert_bank_current,
    assert_bank_matches_backbone,
    assert_patch_bank,
    assert_embedding_path_current,
    assert_patch_embedding_path_current,
    assert_sensor_bank,
    assert_sensor_embedding_path_current,
    assert_text_embedding_path_current,
)
from training.evidence.gate_predictor import (
    ADMISSIBILITY_REFINEMENT_REGIME,
    ADMISSIBILITY_TRAINING_REGIME,
    BankRows,
    concatenate_bank_rows,
    load_evidence_engine,
    load_gate,
    predict_bank_grouped,
    sensor_rows_from_encoded,
    subset_bank_rows,
)
from training.evidence.episode_labels import (
    EpisodeLabelSet,
    encode_neutral_aliases,
    episode_label_set,
)
from training.evidence.device import resolve_device
from training.evidence.labeltext import ensemble_text
from training.evidence.live_encoder import SourcePatchEncoder
from training.evidence.patch_episodes import (
    PatchTable,
    queries_from_encoded,
)
from training.evidence.policy import (
    ACTIVE_WINDOWS_PER_LABEL,
    PHASE_B_EVALUATION_REGIME,
    PHASE_B_TRAINING_REGIME,
    PHASE_B_DEV_DATASETS,
    PHASE_B_TEST_DATASETS,
    PhaseBPolicy,
)
from training.evidence.runtime_memory import build_enrollment_memory
from training.evidence.train_patch_decoder import (
    build_decoder,
    decode_adaptation_episode,
    phase_b_source_fingerprint,
)
from training.tokenizer.eval_transfer import build_encoder, encode_dataset_detailed
from training.tokenizer.pretrain_data import _stream_gravity_state, stream_channel_descriptions

_REPO = Path(__file__).resolve().parents[2]
_OUT = Path(__file__).resolve().parent / "outputs"
_EVALUATION_BEHAVIOR_PATHS = (
    "training/evidence/eval_enrollment.py",
    "training/evidence/admissible_retrieval.py",
    "training/evidence/admissibility_gate.py",
    "training/evidence/gate_predictor.py",
    "training/evidence/bank_guard.py",
    "training/evidence/runtime_memory.py",
    "training/evidence/patch_episodes.py",
    "training/evidence/policy.py",
    "training/evidence/train_patch_decoder.py",
    "training/tokenizer/eval_transfer.py",
    "model/evidence/relational_decoder.py",
    "model/evidence/patch_retrieval.py",
    "eval/enrollment_protocol.py",
    "eval/data.py",
    "eval/scoring.py",
    "data/scripts/curate/deployment_policy.py",
    "data/scripts/labels/canonical_labels.py",
)


def accepted_training_regimes(predictor_mode: str) -> set[str]:
    """Current regimes that can be compared without an archival override."""
    if predictor_mode == "admissibility_gate":
        return {ADMISSIBILITY_TRAINING_REGIME, ADMISSIBILITY_REFINEMENT_REGIME}
    return {PHASE_B_TRAINING_REGIME}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _git_provenance() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_REPO, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty}


def _predictor_step(predictor: dict) -> int | None:
    step = predictor.get("checkpoint_step")
    if step is None:
        step = predictor.get("provenance", {}).get("stage2", {}).get("step")
    return int(step) if step is not None else None


def phase_b_evaluation_source_fingerprint(paths=None) -> str:
    """Fingerprint every source file that can change enrollment-evaluation behavior."""
    return phase_b_source_fingerprint(paths or _EVALUATION_BEHAVIOR_PATHS)


def _json_fingerprint(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summarize_protocol_capabilities(protocol: dict) -> dict:
    """State which adaptation claims the realized external protocol can actually test."""
    usable = [
        value for value in protocol.values()
        if value.get("status") == "ok" and int(value.get("support_ceiling", 0)) > 0
    ]

    def supports(*, subject_relation=None, configuration_relation=None):
        return any(
            (subject_relation is None or value.get("subject_relation") == subject_relation)
            and (
                configuration_relation is None
                or value.get("configuration_relation") == configuration_relation
            )
            for value in usable
        )

    limitations = []
    if not supports(subject_relation="same_subject"):
        limitations.append("no genuine same-subject enrollment cohort")
    if not supports(subject_relation="cross_subject"):
        limitations.append("no genuine cross-subject enrollment cohort")
    if not supports(configuration_relation="cross_configuration"):
        limitations.append("no genuine cross-configuration enrollment cohort")
    return {
        "same_subject_enrollment": supports(subject_relation="same_subject"),
        "cross_subject_enrollment": supports(subject_relation="cross_subject"),
        "cross_configuration_enrollment": supports(
            configuration_relation="cross_configuration"
        ),
        "usable_protocol_relations": len(usable),
        "limitations": limitations,
    }


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
    support_execution_rows: tuple[tuple[tuple[int, ...], ...], ...] | None = None


def _manifest_plans(payload: dict) -> list[EnrollmentSubjectPlan]:
    return [
        EnrollmentSubjectPlan(
            subject=plan["subject"],
            candidate_names=tuple(plan["candidate_names"]),
            support_rows=tuple(() for _ in plan["candidate_names"]),
            query_rows=np.asarray(plan["query_rows"], dtype=np.int64),
            support_execution_rows=tuple(
                tuple(tuple(int(row) for row in execution) for execution in candidate)
                for candidate in plan["support_execution_rows"]
            ),
        )
        for plan in payload.get("plans", ())
    ]


def _selected_support_rows(
    plan: EnrollmentSubjectPlan, position: int, support_count: int
) -> tuple[int, ...]:
    if plan.support_execution_rows is None:
        return tuple(plan.support_rows[position][:support_count])
    return tuple(
        row
        for execution in plan.support_execution_rows[position][:support_count]
        for row in execution
    )


def _episode_labels(
    candidate_names,
    coherent,
    sbert,
    device,
    *,
    random_aliases,
    rng,
    alias_embeddings,
    fixed_aliases=None,
):
    if random_aliases and fixed_aliases is not None:
        phrases = tuple(fixed_aliases[name] for name in candidate_names)
        embeddings = torch.from_numpy(sbert(list(phrases)).astype(np.float32)).to(device)
        return EpisodeLabelSet("random_alias", phrases, embeddings)
    return episode_label_set(
        torch.arange(len(candidate_names), device=device), coherent,
        mode="random_alias" if random_aliases else "coherent", rng=rng,
        alias_embeddings=alias_embeddings, canonical_names=candidate_names,
    )


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
    support_labels: np.ndarray | None = None,
    support_subjects: np.ndarray | None = None,
    support_execution_ids: np.ndarray | None = None,
) -> tuple[list[EnrollmentSubjectPlan], dict]:
    """Freeze one query cohort and nested support prefixes for a complete enrollment curve.

    The support arrays may describe another acquisition stream. This makes cross-configuration
    evaluation explicit while preserving subject identity and excluding the selected execution from
    same-subject queries when execution ids are shared across streams.
    """
    support_labels = labels if support_labels is None else support_labels
    support_subjects = subjects if support_subjects is None else support_subjects
    support_execution_ids = (
        execution_ids if support_execution_ids is None else support_execution_ids
    )
    if not (len(support_labels) == len(support_subjects) == len(support_execution_ids)):
        raise ValueError("support labels, subjects, and execution ids must align")
    positive = sorted({int(value) for value in requested_support if value > 0})
    quality = _execution_quality(support_execution_ids)
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
                source = np.flatnonzero(
                    (support_labels == label) & (support_subjects == subject)
                )
                support_executions = np.unique(support_execution_ids[source])
                query_executions = np.unique(execution_ids[same])
                feasible = len(same) > 0 and len(support_executions) >= support_count
                if feasible and support_count > 0:
                    # There must be an anchor query execution that can be excluded from every nested
                    # support prefix. Across streams, an equal execution id denotes the paired bout.
                    feasible = any(
                        len(support_executions[support_executions != query_execution])
                        >= support_count
                        for query_execution in query_executions
                    )
            else:
                other = np.flatnonzero(
                    (support_labels == label) & (support_subjects != subject)
                )
                feasible = len(same) > 0 and len(
                    np.unique(support_execution_ids[other])
                ) >= support_count
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
            source = np.flatnonzero(
                (support_labels == label)
                & ((support_subjects == subject) if mode == "same_subject"
                   else (support_subjects != subject))
            )
            source_executions = np.unique(support_execution_ids[source])
            if mode == "same_subject" and support_ceiling > 0:
                query_executions = np.unique(execution_ids[same])
                anchors = [
                    execution for execution in query_executions
                    if len(source_executions[source_executions != execution]) >= support_ceiling
                ]
                if not anchors:
                    raise RuntimeError("paired enrollment feasibility changed during plan build")
                anchor = anchors[int(rng.integers(len(anchors)))]
                source_executions = source_executions[source_executions != anchor]
            source_executions = source_executions[rng.permutation(len(source_executions))]
            selected_executions = source_executions[:support_ceiling]
            selected_rows = tuple(
                int(rng.choice(source[support_execution_ids[source] == execution]))
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


def _few_shot_baselines(
    query_encoded,
    support_encoded,
    support,
    support_position,
    query_rows,
    n_candidates,
    device,
):
    if len(support) == 0:
        return None, None
    support_pooled = F.normalize(
        torch.as_tensor(support_encoded["pooled"]).float().to(device), dim=-1
    )
    query_pooled = F.normalize(
        torch.as_tensor(query_encoded["pooled"]).float().to(device), dim=-1
    )
    x = support_pooled[torch.as_tensor(support, device=device)]
    position = torch.as_tensor(support_position, device=device, dtype=torch.long)
    q = query_pooled[torch.as_tensor(query_rows, device=device)]
    if any(not bool(position.eq(candidate).any()) for candidate in range(n_candidates)):
        # Prototype/ridge controls require at least one fitted example for every output class.
        return None, None
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


PREDICTOR_MODES = ("relational_decoder", "closed_form_retrieval_vote", "admissibility_gate")


def active_index_seed(predictor: dict, evaluation_seed: int) -> int:
    """One protocol-owned archive draw shared by every predictor family being compared."""
    return int(evaluation_seed) + 2


def predictor_logits(logits: torch.Tensor, aux: dict, predictor_mode: str) -> torch.Tensor:
    """Select an archived pooled episode path.

    The sensor-row admissibility predictor has a separate evaluator path and never passes through the
    parked relational decoder. Keeping that boundary explicit prevents a missing gate output from
    silently falling back to an unrelated pooled prediction.
    """
    if predictor_mode == "relational_decoder":
        return logits
    if predictor_mode == "closed_form_retrieval_vote":
        return aux["identity_logits"]
    raise ValueError(f"unknown predictor mode {predictor_mode!r}")


def _summarize_enrollment_predictions(
    *,
    all_true,
    all_pred,
    all_identity_pred,
    all_removed_pred,
    all_shuffled_pred,
    all_prototype_pred,
    all_ridge_pred,
    all_enrolled_query,
    subject_results,
    base_bank,
    phase_b_seen_labels,
    plans,
    seed,
    support_count,
):
    """One result contract for the legacy and admissibility prediction paths."""
    if not all_true:
        return {
            "status": "insufficient_independent_executions",
            "queries": 0,
            "subjects": 0,
        }
    metrics = classification_metrics(all_true, all_pred)
    identity_metrics = classification_metrics(all_true, all_identity_pred)
    removed_metrics = (
        classification_metrics(all_true, all_removed_pred) if all_removed_pred else None
    )
    shuffled_metrics = (
        classification_metrics(all_true, all_shuffled_pred) if all_shuffled_pred else None
    )
    bank_vocab = set(base_bank["vocab"])
    exact_phase_a_mask = np.asarray([label in bank_vocab for label in all_true])
    canonical_phase_a_mask = np.asarray([
        canonicalize(label) in bank_vocab for label in all_true
    ])
    phase_b_mask = np.asarray([
        canonicalize(label) in phase_b_seen_labels for label in all_true
    ])
    enrolled_mask = np.asarray(all_enrolled_query, dtype=bool)

    def subset_f1(prediction, mask):
        return float(classification_metrics(
            np.asarray(all_true, dtype=object)[mask].tolist(),
            np.asarray(prediction, dtype=object)[mask].tolist(),
        )["f1_macro"]) if bool(mask.any()) else float("nan")

    return {
        "f1_macro": float(metrics["f1_macro"]),
        "identity_f1_macro": float(identity_metrics["f1_macro"]),
        "adaptation_f1_gain": float(metrics["f1_macro"] - identity_metrics["f1_macro"]),
        "support_removed_f1_macro": (
            float(removed_metrics["f1_macro"]) if removed_metrics else None
        ),
        "support_removal_f1_gain": (
            float(metrics["f1_macro"] - removed_metrics["f1_macro"])
            if removed_metrics else None
        ),
        "support_label_shuffled_f1_macro": (
            float(shuffled_metrics["f1_macro"]) if shuffled_metrics else None
        ),
        "correct_support_label_f1_gain": (
            float(metrics["f1_macro"] - shuffled_metrics["f1_macro"])
            if shuffled_metrics else None
        ),
        "prototype_f1_macro": (
            float(classification_metrics(all_true, all_prototype_pred)["f1_macro"])
            if all_prototype_pred else None
        ),
        "ridge_head_f1_macro": (
            float(classification_metrics(all_true, all_ridge_pred)["f1_macro"])
            if all_ridge_pred else None
        ),
        "phase_a_exact_label_seen_f1_macro": subset_f1(all_pred, exact_phase_a_mask),
        "phase_a_exact_label_unseen_f1_macro": subset_f1(all_pred, ~exact_phase_a_mask),
        "phase_a_canonical_concept_seen_f1_macro": subset_f1(all_pred, canonical_phase_a_mask),
        "phase_a_canonical_concept_unseen_f1_macro": subset_f1(all_pred, ~canonical_phase_a_mask),
        "phase_b_candidate_seen_f1_macro": subset_f1(all_pred, phase_b_mask),
        "phase_b_candidate_unseen_f1_macro": subset_f1(all_pred, ~phase_b_mask),
        "phase_b_candidate_seen_queries": int(phase_b_mask.sum()),
        "phase_b_candidate_unseen_queries": int((~phase_b_mask).sum()),
        "enrolled_candidate_f1_macro": subset_f1(all_pred, enrolled_mask),
        "unenrolled_candidate_f1_macro": subset_f1(all_pred, ~enrolled_mask),
        "enrolled_candidate_queries": int(enrolled_mask.sum()),
        "unenrolled_candidate_queries": int((~enrolled_mask).sum()),
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


@torch.no_grad()
def score_enrollment_cell(
    encoded,
    labels,
    subjects,
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
    support_encoded=None,
    support_subjects: np.ndarray | None = None,
    enrolled_candidate_count: int | None = None,
    phase_b_seen_labels: set[str] | None = None,
    same_configuration: bool = True,
    predictor_mode: str = "relational_decoder",
    fixed_aliases: dict[str, str] | None = None,
):
    all_true, all_pred, all_identity_pred = [], [], []
    all_removed_pred, all_shuffled_pred = [], []
    all_prototype_pred, all_ridge_pred = [], []
    all_enrolled_query = []
    subject_results = {}
    support_encoded = encoded if support_encoded is None else support_encoded
    support_subjects = subjects if support_subjects is None else support_subjects
    phase_b_seen_labels = phase_b_seen_labels or set()
    base_subject_max = int(torch.as_tensor(base_bank["patch"]["subj"]).max())
    for subject_index, plan in enumerate(plans):
        candidate_names = list(plan.candidate_names)
        vocab_position = {label: index for index, label in enumerate(base_bank["vocab"])}
        excluded_base_labels = torch.tensor(sorted({
            vocab_position[canonicalize(label)]
            for label in candidate_names
            if canonicalize(label) in vocab_position
        }), dtype=torch.long)
        query_rows = plan.query_rows
        rng = np.random.default_rng(seed + 50_021 * subject_index)
        if support_count == 0:
            enrolled_positions = np.empty(0, dtype=np.int64)
        elif enrolled_candidate_count is None:
            enrolled_positions = np.arange(len(candidate_names), dtype=np.int64)
        else:
            requested = (
                max(1, len(candidate_names) // 2)
                if int(enrolled_candidate_count) == 0 else int(enrolled_candidate_count)
            )
            count = max(1, min(requested, len(candidate_names) - 1))
            enrolled_positions = np.sort(
                rng.choice(len(candidate_names), size=count, replace=False)
            ).astype(np.int64, copy=False)
        enrolled_set = set(enrolled_positions.tolist())
        support = np.asarray([
            row for position in range(len(candidate_names))
            if position in enrolled_set
            for row in _selected_support_rows(plan, position, support_count)
        ], dtype=np.int64)
        support_position = np.asarray([
            position for position in range(len(candidate_names))
            if position in enrolled_set
            for _ in _selected_support_rows(plan, position, support_count)
        ], dtype=np.int64)
        external_subjects = [plan.subject] + [support_subjects[int(row)] for row in support]
        runtime_subject = {
            value: base_subject_max + 1 + offset
            for offset, value in enumerate(dict.fromkeys(external_subjects))
        }
        support_runtime_subject = torch.tensor([
            runtime_subject[support_subjects[int(row)]] for row in support
        ], dtype=torch.long)
        coherent = ensemble_text(candidate_names, sbert, 1).to(device)
        label_mode = "random_alias" if random_aliases else "coherent"
        label_set = _episode_labels(
            candidate_names, coherent, sbert, device,
            random_aliases=random_aliases, rng=rng,
            alias_embeddings=alias_embeddings, fixed_aliases=fixed_aliases,
        )
        def make_memory(position):
            memory = build_enrollment_memory(
                base_bank, base_rows, base_selector_z, support_encoded,
                torch.from_numpy(support), torch.from_numpy(position),
                canonical_text, label_set.embeddings,
                excluded_base_labels=excluded_base_labels,
                support_subject_ids=support_runtime_subject,
                query_matches_support_config=same_configuration,
            )
            return memory, retriever.build_index(memory.selector_z)

        enrollment, memory_index = make_memory(support_position)
        removed, removed_index = build_enrollment_memory(
            base_bank, base_rows, base_selector_z, support_encoded,
            torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
            canonical_text, label_set.embeddings,
            excluded_base_labels=excluded_base_labels,
            support_subject_ids=torch.empty(0, dtype=torch.long),
            query_matches_support_config=same_configuration,
        ), None
        removed_index = retriever.build_index(removed.selector_z)
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
                encoded, rows, device, sensor_id=enrollment.runtime_sensor_id,
                subject_ids=torch.full(
                    (len(rows),), runtime_subject[plan.subject], dtype=torch.long
                ),
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
                policy=policy,
                rng=np.random.default_rng(seed + subject_index),
            )
            logits = predictor_logits(logits, aux, predictor_mode)
            removed_view = removed.episode_view(
                query, target_position, label_mode=label_mode
            )
            removed_logits, removed_aux = decode_adaptation_episode(
                decoder, retriever, removed.bank, removed.index_rows,
                removed.selector_z, removed_index, query, removed_view,
                removed.canonical_text, removed.candidate_text,
                policy=policy,
                rng=np.random.default_rng(seed + subject_index),
            )
            removed_logits = predictor_logits(removed_logits, removed_aux, predictor_mode)
            shuffled_logits = None
            if shuffled is not None:
                shuffled_memory, shuffled_index = shuffled
                shuffled_view = shuffled_memory.episode_view(
                    query, target_position, label_mode=label_mode
                )
                shuffled_logits, shuffled_aux = decode_adaptation_episode(
                    decoder, retriever, shuffled_memory.bank, shuffled_memory.index_rows,
                    shuffled_memory.selector_z, shuffled_index, query, shuffled_view,
                    shuffled_memory.canonical_text, shuffled_memory.candidate_text,
                    policy=policy,
                    rng=np.random.default_rng(seed + subject_index),
                )
                shuffled_logits = predictor_logits(
                    shuffled_logits, shuffled_aux, predictor_mode
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
            all_enrolled_query.extend([
                position_by_name[name] in enrolled_set for name in truth
            ])
        prototype, ridge = _few_shot_baselines(
            encoded, support_encoded, support, support_position, query_rows,
            len(candidate_names), device
        )
        if prototype is not None:
            prototype_names = [candidate_names[index] for index in prototype]
            ridge_names = [candidate_names[index] for index in ridge]
            all_prototype_pred.extend(prototype_names)
            all_ridge_pred.extend(ridge_names)
        else:
            prototype_names = ridge_names = None
        subject_results[str(plan.subject)] = {
            "f1_macro": macro_f1_fast(subject_true, subject_pred),
            "identity_f1_macro": macro_f1_fast(subject_true, subject_identity),
            "support_removed_f1_macro": macro_f1_fast(subject_true, subject_removed),
            "support_label_shuffled_f1_macro": macro_f1_fast(
                subject_true, subject_shuffled
            ),
            "prototype_f1_macro": (
                macro_f1_fast(subject_true, prototype_names)
                if prototype_names is not None else None
            ),
            "ridge_head_f1_macro": (
                macro_f1_fast(subject_true, ridge_names)
                if ridge_names is not None else None
            ),
            "queries": len(subject_true),
            "candidate_count": len(candidate_names),
            "enrolled_candidate_count": len(enrolled_positions),
        }
    return _summarize_enrollment_predictions(
        all_true=all_true,
        all_pred=all_pred,
        all_identity_pred=all_identity_pred,
        all_removed_pred=all_removed_pred,
        all_shuffled_pred=all_shuffled_pred,
        all_prototype_pred=all_prototype_pred,
        all_ridge_pred=all_ridge_pred,
        all_enrolled_query=all_enrolled_query,
        subject_results=subject_results,
        base_bank=base_bank,
        phase_b_seen_labels=phase_b_seen_labels,
        plans=plans,
        seed=seed,
        support_count=support_count,
    )


@torch.no_grad()
def score_admissibility_cell(
    encoded,
    labels,
    subjects,
    plans,
    base_bank,
    base_sensor_rows: BankRows,
    gate,
    canonical_text,
    sbert,
    alias_embeddings,
    device,
    *,
    query_dataset: str,
    query_stream: str,
    query_mask,
    support_count: int,
    random_aliases: bool,
    seed: int,
    support_encoded=None,
    support_subjects: np.ndarray | None = None,
    support_dataset: str | None = None,
    support_stream: str | None = None,
    support_mask=None,
    enrolled_candidate_count: int | None = None,
    phase_b_seen_labels: set[str] | None = None,
    top_k: int = 64,
    batch_size: int = 64,
    fixed_aliases: dict[str, str] | None = None,
    counterfactual_controls: bool = True,
    engine=None,
):
    """Evaluate the sensor-row path. The parked relational decoder is never invoked.

    With ``engine=None`` this is the closed-form rule. With a trained evidence engine it is the
    deployed form of the end-to-end model: the learned weight replaces ``exp(cos/tau)*admissibility``
    on the pooled rows and nothing else moves. The identity/prototype/ridge comparators stay
    closed-form in both cases, because they are what the learned path is measured against.
    """
    all_true, all_pred, all_identity_pred = [], [], []
    all_removed_pred, all_shuffled_pred = [], []
    all_prototype_pred, all_ridge_pred = [], []
    all_enrolled_query, subject_results = [], {}
    gate_telemetry: list[dict[str, float]] = []
    support_encoded = encoded if support_encoded is None else support_encoded
    support_subjects = subjects if support_subjects is None else support_subjects
    support_dataset = query_dataset if support_dataset is None else support_dataset
    support_stream = query_stream if support_stream is None else support_stream
    support_mask = query_mask if support_mask is None else support_mask
    phase_b_seen_labels = phase_b_seen_labels or set()
    vocab_position = {label: index for index, label in enumerate(base_bank["vocab"])}
    source_sensor_window = torch.as_tensor(encoded["sensor_window"]).long().cpu()
    sensor_rows_by_window: dict[int, list[int]] = {}
    for sensor_row, source_window in enumerate(source_sensor_window.tolist()):
        sensor_rows_by_window.setdefault(int(source_window), []).append(sensor_row)

    for subject_index, plan in enumerate(plans):
        candidate_names = list(plan.candidate_names)
        query_rows = plan.query_rows
        rng = np.random.default_rng(seed + 50_021 * subject_index)
        if support_count == 0:
            enrolled_positions = np.empty(0, dtype=np.int64)
        elif enrolled_candidate_count is None:
            enrolled_positions = np.arange(len(candidate_names), dtype=np.int64)
        else:
            requested = (
                max(1, len(candidate_names) // 2)
                if int(enrolled_candidate_count) == 0 else int(enrolled_candidate_count)
            )
            count = max(1, min(requested, len(candidate_names) - 1))
            enrolled_positions = np.sort(
                rng.choice(len(candidate_names), size=count, replace=False)
            ).astype(np.int64, copy=False)
        enrolled_set = set(enrolled_positions.tolist())
        support = np.asarray([
            row for position in range(len(candidate_names))
            if position in enrolled_set
            for row in _selected_support_rows(plan, position, support_count)
        ], dtype=np.int64)
        support_position = np.asarray([
            position for position in range(len(candidate_names))
            if position in enrolled_set
            for _ in _selected_support_rows(plan, position, support_count)
        ], dtype=np.int64)

        excluded = torch.tensor(sorted({
            vocab_position[canonicalize(label)]
            for label in candidate_names if canonicalize(label) in vocab_position
        }), device=device, dtype=torch.long)
        keep = ~torch.isin(base_sensor_rows.rows.label, excluded)
        corpus = subset_bank_rows(base_sensor_rows, keep)
        coherent = ensemble_text(candidate_names, sbert, 1).to(device)
        label_mode = "random_alias" if random_aliases else "coherent"
        label_set = _episode_labels(
            candidate_names, coherent, sbert, device,
            random_aliases=random_aliases, rng=rng,
            alias_embeddings=alias_embeddings, fixed_aliases=fixed_aliases,
        )
        semantic_labels = not random_aliases

        def memory(position: np.ndarray) -> BankRows:
            if not len(support):
                return corpus
            enrolled = sensor_rows_from_encoded(
                support_encoded, torch.from_numpy(support), support_dataset, support_stream,
                channel_mask=support_mask,
                candidate_positions=torch.from_numpy(position),
            ).to(device)
            return concatenate_bank_rows(corpus, enrolled)

        enrollment = memory(support_position)
        removed = corpus
        shuffled = (
            memory((support_position + 1) % len(candidate_names))
            if counterfactual_controls and len(support_position) else None
        )
        position_by_name = {name: i for i, name in enumerate(candidate_names)}
        subject_true, subject_pred, subject_identity = [], [], []
        subject_removed, subject_shuffled = [], []

        for start in range(0, len(query_rows), batch_size):
            batch_rows = np.asarray(query_rows[start:start + batch_size], dtype=np.int64)
            batch_tensor = torch.from_numpy(batch_rows)
            selected_sensor_rows = torch.tensor([
                sensor_row
                for window in batch_rows
                for sensor_row in sensor_rows_by_window.get(int(window), ())
            ], dtype=torch.long)
            selected_sensor_window = source_sensor_window.index_select(
                0, selected_sensor_rows
            )
            owner_of = {int(window): position for position, window in enumerate(batch_rows)}
            query_group = torch.tensor([
                owner_of[int(window)] for window in selected_sensor_window.tolist()
            ], dtype=torch.long, device=device)
            query = sensor_rows_from_encoded(
                encoded, batch_tensor, query_dataset, query_stream,
                channel_mask=query_mask,
                sensor_row_indices=selected_sensor_rows,
            ).to(device).rows
            if len(query_group) != len(query.feature):
                raise RuntimeError("query sensor ownership does not align with encoded sensor rows")

            def run(memory_rows: BankRows):
                return predict_bank_grouped(
                    gate, memory_rows,
                    query_feature=query.feature,
                    query_bias=query.bias,
                    query_descriptor=query.descriptor,
                    query_modality=query.modality,
                    query_gravity=query.gravity,
                    candidate_text=label_set.embeddings,
                    label_text=canonical_text,
                    top_k=top_k,
                    semantic_labels=semantic_labels,
                    query_group=query_group,
                    group_count=len(batch_rows),
                    engine=engine,
                )

            logits, aux = run(enrollment)
            removed_logits = run(removed)[0] if counterfactual_controls else None
            shuffled_logits = run(shuffled)[0] if shuffled is not None else removed_logits
            gate_telemetry.append({
                key: float(value) for key, value in aux.items()
                if key.startswith(("gate/", "evidence/"))
            })
            predictions = logits.argmax(1).tolist()
            identities = aux["identity_logits"].argmax(1).tolist()
            removed_predictions = (
                removed_logits.argmax(1).tolist() if removed_logits is not None else None
            )
            shuffled_predictions = (
                shuffled_logits.argmax(1).tolist() if shuffled_logits is not None else None
            )
            for offset, row in enumerate(batch_rows):
                truth = str(labels[row])
                prediction = candidate_names[int(predictions[offset])]
                identity = candidate_names[int(identities[offset])]
                all_true.append(truth); subject_true.append(truth)
                all_pred.append(prediction); subject_pred.append(prediction)
                all_identity_pred.append(identity); subject_identity.append(identity)
                if removed_predictions is not None:
                    removed_name = candidate_names[int(removed_predictions[offset])]
                    all_removed_pred.append(removed_name); subject_removed.append(removed_name)
                if shuffled_predictions is not None:
                    shuffled_name = candidate_names[int(shuffled_predictions[offset])]
                    all_shuffled_pred.append(shuffled_name); subject_shuffled.append(shuffled_name)
                all_enrolled_query.append(position_by_name[truth] in enrolled_set)

        prototype, ridge = _few_shot_baselines(
            encoded, support_encoded, support, support_position, query_rows,
            len(candidate_names), device,
        )
        if prototype is not None:
            prototype_names = [candidate_names[index] for index in prototype]
            ridge_names = [candidate_names[index] for index in ridge]
            all_prototype_pred.extend(prototype_names)
            all_ridge_pred.extend(ridge_names)
        else:
            prototype_names = ridge_names = None
        subject_results[str(plan.subject)] = {
            "f1_macro": macro_f1_fast(subject_true, subject_pred),
            "identity_f1_macro": macro_f1_fast(subject_true, subject_identity),
            "support_removed_f1_macro": (
                macro_f1_fast(subject_true, subject_removed) if subject_removed else None
            ),
            "support_label_shuffled_f1_macro": (
                macro_f1_fast(subject_true, subject_shuffled) if subject_shuffled else None
            ),
            "prototype_f1_macro": (
                macro_f1_fast(subject_true, prototype_names)
                if prototype_names is not None else None
            ),
            "ridge_head_f1_macro": (
                macro_f1_fast(subject_true, ridge_names)
                if ridge_names is not None else None
            ),
            "queries": len(subject_true),
            "candidate_count": len(candidate_names),
            "enrolled_candidate_count": len(enrolled_positions),
        }

    result = _summarize_enrollment_predictions(
        all_true=all_true, all_pred=all_pred, all_identity_pred=all_identity_pred,
        all_removed_pred=all_removed_pred, all_shuffled_pred=all_shuffled_pred,
        all_prototype_pred=all_prototype_pred, all_ridge_pred=all_ridge_pred,
        all_enrolled_query=all_enrolled_query, subject_results=subject_results,
        base_bank=base_bank, phase_b_seen_labels=phase_b_seen_labels, plans=plans,
        seed=seed, support_count=support_count,
    )
    if gate_telemetry and "status" not in result:
        result["admissibility"] = {
            key: float(np.mean([row[key] for row in gate_telemetry if key in row]))
            for key in gate_telemetry[0]
        }
    return result


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get(
        "HALO_CKPT",
        _REPO / "training/tokenizer/outputs/phase_a_fixed_1s_rotation_20260817/best.pt",
    )))
    parser.add_argument("--bank", type=Path, default=_OUT / "memory_bank.pt")
    parser.add_argument(
        "--predictor", type=Path, required=True,
        help="explicit Stage-1 or selected Stage-2 predictor artifact",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="explicit override; otherwise --protocol-role selects the sealed roster")
    parser.add_argument("--protocol-role", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--support", nargs="*", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--modes", nargs="*", choices=("same_subject", "cross_subject"),
                        default=["same_subject", "cross_subject"])
    parser.add_argument(
        "--configuration-modes", nargs="*", choices=("same", "cross"),
        default=["same", "cross"],
        help="evaluate enrollment from the query stream and, where available, another placement",
    )
    parser.add_argument(
        "--enrollment-shapes", nargs="*", choices=("full", "partial"),
        default=["full", "partial"],
        help="positive-support protocols; partial enrolls half of each candidate set",
    )
    parser.add_argument("--random-aliases", action="store_true")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--skip-zero-shot", action="store_true",
        help="skip k=0 when it was already scored by another manifest seed",
    )
    parser.add_argument(
        "--counterfactual-controls", choices=("full", "none"), default="full",
        help="support-removal and support-label-shuffle controls; use full for at least one seed",
    )
    parser.add_argument(
        "--encoded-cache-dir", type=Path,
        default=_OUT / "encoded_eval_cache",
        help="persistent CPU cache for frozen Phase-A eval embeddings; keyed by checkpoint, grid, "
             "predictor layout, and evaluation source",
    )
    parser.add_argument("--no-encoded-cache", action="store_true")
    parser.add_argument("--no-engine", dest="use_engine", action="store_false",
                        help="score the closed-form rule even when the checkpoint carries a "
                             "trained evidence engine (the closed-form comparison)")
    parser.add_argument("--engine-checkpoint", type=Path, default=None,
                        help="Phase-B episodic checkpoint carrying the trained evidence engine. The "
                             "memory bank must be built from this same checkpoint's encoder, since "
                             "the engine was trained against rows that encoder produced")
    parser.add_argument("--gate-top-k", type=int, default=64,
                        help="evidence rows retrieved per query patch and candidate")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="versioned shared support/query manifest. When set, dataset rosters, candidate sets, "
             "query rows, support prefixes, and seeds come exclusively from this artifact.",
    )
    parser.add_argument(
        "--manifest-seed", type=int, default=None,
        help="one serialized manifest seed to score; defaults to the manifest's first seed",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="explicit result path; otherwise derived from the protocol role. "
                             "Use this to score several arms without clobbering each other.")
    parser.add_argument(
        "--accept-training-regime", default=None,
        help="score a predictor recorded under an older regime. Required to re-score an archived "
             "checkpoint after a "
             "training-only recipe change; the accepted regime is written into the result so the "
             "comparison stays auditable. Never use it to mix EVALUATION protocols.",
    )
    parser.add_argument(
        "--accept-training-source-fingerprint", action="store_true",
        help="explicitly score an archived predictor after its behavioral source changed",
    )
    args = parser.parse_args()
    explicit_datasets = args.datasets is not None
    shared_manifest = None
    manifest_seed = None
    if args.manifest is not None:
        from eval.enrollment_protocol import load_manifest

        shared_manifest = load_manifest(args.manifest, validate_grids=True)
        if shared_manifest["alignment"] != "native":
            parser.error("HALO manifest evaluation currently requires alignment='native'")
        args.datasets = list(shared_manifest["datasets"])
        args.support = list(shared_manifest["support_counts"])
        available_seeds = [int(value) for value in shared_manifest["seeds"]]
        manifest_seed = args.manifest_seed if args.manifest_seed is not None else available_seeds[0]
        if manifest_seed not in available_seeds:
            parser.error(
                f"--manifest-seed {manifest_seed} is absent from manifest seeds {available_seeds}"
            )
        args.seed = manifest_seed
        args.enrollment_shapes = ["full"]
        explicit_datasets = False
    if args.datasets is None:
        args.datasets = {
            "dev": list(PHASE_B_DEV_DATASETS),
            "test": list(PHASE_B_TEST_DATASETS),
            "all": list(deployment_policy.PRIMARY_EVAL_DATASETS),
        }[args.protocol_role]
    if any(value < 0 for value in args.support):
        parser.error("support counts must be nonnegative")
    if args.gate_top_k < 1:
        parser.error("--gate-top-k must be positive")
    if args.batch < 1:
        parser.error("--batch must be positive")
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    assert_bank_current(bank, context="eval_enrollment")
    assert_patch_bank(bank, context="eval_enrollment")
    # Runtime candidate text is compared directly with stored label text in the active sensor-bank
    # design. Schema-3 banks predate this probe and exist only for historical reproduction.
    if int(bank.get("schema_version", 1)) >= 4:
        assert_text_embedding_path_current(bank, context="eval_enrollment")
    predictor = torch.load(args.predictor, map_location="cpu", weights_only=True)
    predictor_fp = hashlib.sha256(args.predictor.read_bytes()).hexdigest()
    assert_artifact_matches_bank(
        predictor, bank, context="eval_enrollment", artifact_name="patch evidence predictor"
    )
    predictor_mode = predictor.get("predictor_mode", "relational_decoder")
    if predictor_mode not in set(PREDICTOR_MODES):
        raise SystemExit(f"unsupported predictor mode {predictor_mode!r}")
    if predictor_mode == "admissibility_gate" and bank.get("schema_version", 0) < 5:
        # Fail before scoring rather than after: older banks do not bind sensor text semantics, and
        # "which sensor produced this row" — the gate's entire input — is undefined for them.
        raise SystemExit(
            "predictor_mode='admissibility_gate' requires a schema-5 bank with per-sensor rows "
            f"(this bank is schema {bank.get('schema_version', 0)}). Rebuild it with "
            "`build_memory --sensor-rows` against a sensor-granularity checkpoint."
        )
    recorded_regime = predictor.get("training_regime")
    native_regimes = accepted_training_regimes(predictor_mode)
    accepted = set(native_regimes)
    if args.accept_training_regime:
        accepted.add(args.accept_training_regime)
    if recorded_regime not in accepted:
        raise SystemExit(
            f"enrollment evaluation requires one of {sorted(native_regimes)!r}; the predictor "
            f"records {recorded_regime!r}. Pass --accept-training-regime to score it anyway."
        )
    current_source_fp = (
        None if predictor_mode == "admissibility_gate" else phase_b_source_fingerprint()
    )
    recorded_source_fp = predictor.get("training_source_fp")
    if predictor_mode != "admissibility_gate" and recorded_source_fp != current_source_fp \
            and not args.accept_training_source_fingerprint:
        raise SystemExit(
            "Phase-B behavioral source differs from the predictor artifact. Re-train under the "
            "current regime, or pass --accept-training-source-fingerprint for an explicitly "
            "archived comparison."
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert_bank_matches_backbone(bank, checkpoint, context="eval_enrollment")
    checkpoint_fp = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checkpoint_fp != bank["backbone"].get("fingerprint"):
        raise SystemExit("encoder checkpoint != patch-bank backbone")
    encoder = build_encoder(checkpoint, device)
    assert_embedding_path_current(bank, encoder, device, context="eval_enrollment")
    assert_patch_embedding_path_current(bank, encoder, device, context="eval_enrollment")
    gate = base_sensor_rows = None
    engine = None
    decoder = retriever = policy = None
    if predictor_mode == "admissibility_gate":
        assert_sensor_bank(bank, context="eval_enrollment")
        assert_sensor_embedding_path_current(bank, encoder, device, context="eval_enrollment")
        gate = load_gate(args.predictor).to(device)
        if args.use_engine and args.engine_checkpoint is not None:
            engine = load_evidence_engine(args.engine_checkpoint, device=device)
            if engine is None:
                parser.error(
                    f"{args.engine_checkpoint} carries no evidence engine; it predates the module. "
                    "Pass a Phase-B episodic checkpoint or drop --engine-checkpoint."
                )
            # The engine was trained against rows from ITS OWN encoder. The bank guard above already
            # refuses a mismatched encoder, but say why plainly rather than leaving the operator to
            # decode an embedding-path assertion.
            print(f"[eval_enrollment] evidence engine loaded from {args.engine_checkpoint} "
                  f"(top-k {engine.cfg.top_k}, readout "
                  f"{engine.cfg.mixer.readout}); the bank must have been built from the "
                  f"SAME checkpoint's encoder")
    else:
        cfg = predictor["cfg"]
        if cfg.get("tokenizer_mode") == "ema_finetune":
            encoder.load_state_dict(predictor["tokenizer_ema"])
        decoder = build_decoder(cfg).to(device)
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
        np.random.default_rng(active_index_seed(predictor, args.seed)),
    )
    base_selector_z = None
    if predictor_mode != "admissibility_gate":
        if policy is not None and policy.tokenizer_mode == "ema_finetune":
            source = SourcePatchEncoder(bank, device)
            base_selector_z = source.encode_patch_rows(
                base_rows, encoder, requires_grad=False
            ).to(device)
        else:
            base_selector_z = bank["patch"]["Z"][base_rows].float().to(device)
        base_selector_z = F.normalize(base_selector_z, dim=-1)
    if predictor_mode == "admissibility_gate":
        # Use the same label-balanced active window archive as the legacy path, then take every
        # per-sensor patch row belonging to those windows. This bounds retrieval cost without changing
        # the archive population or introducing a second sampling policy.
        active_windows = torch.unique(bank["patch"]["window"][base_rows].long())
        sensor_row_index = torch.isin(
            bank["sensor"]["window"].long(), active_windows
        ).nonzero(as_tuple=False).flatten()
        from training.evidence.gate_predictor import sensor_rows_from_bank

        base_sensor_rows = sensor_rows_from_bank(
            bank, row_indices=sensor_row_index
        ).to(device)
    sbert = get_sbert_encoder()
    # Predictor comparisons must differ in predictor mechanics, not in the text representation they
    # receive. The bank stores the canonical MiniLM vectors used by the active Phase-B protocol.
    canonical_text = F.normalize(bank["label_text"].float(), dim=-1).to(device)
    aliases = encode_neutral_aliases(sbert, device)
    phase_b_seen_labels = set(
        predictor.get("phase_b_train_labels", ())
        or (predictor.get("provenance") or {}).get("training_concepts", ())
    )
    if not phase_b_seen_labels:
        held = set(int(value) for value in predictor.get("heldout_labels", ()))
        phase_b_seen_labels = {
            label for index, label in enumerate(bank["vocab"]) if index not in held
        }
    bank_config_names = bank.get("cfg_names", {})
    phase_a_datasets = {
        str(name).split("/", 1)[0] for name in bank_config_names.values()
    }
    phase_b_datasets = set(
        predictor.get("phase_b_train_datasets", ())
        or (predictor.get("provenance") or {}).get("training_datasets", ())
    )
    phase_b_subject_names = set(predictor.get("phase_b_train_subject_names", ()))

    encoded_streams = {}
    encoded_cache_stats = {"hits": 0, "misses": 0}
    evaluation_source_fp = phase_b_evaluation_source_fingerprint()

    def load_and_encode(stream_spec):
        cache_key = (stream_spec.dataset, stream_spec.stream_id)
        if cache_key in encoded_streams:
            return encoded_streams[cache_key]
        stream = load_eval_stream(
            stream_spec.dataset, stream_spec.stream_id, alignment="native"
        )
        if stream.execution_ids is None:
            raise RuntimeError(
                f"{stream_spec.dataset}/{stream.stream}: native grid has no execution ids"
            )
        encoded = None
        cache_path = None
        if shared_manifest is not None and not args.no_encoded_cache:
            stream_key = f"{stream.dataset}/{stream.stream}"
            cache_identity = _json_fingerprint({
                "checkpoint": checkpoint_fp,
                "stream": stream_key,
                "stream_fingerprint": shared_manifest["stream_fingerprints"][stream_key],
                "predictor_mode": predictor_mode,
                "evaluation_source": evaluation_source_fp,
            })
            cache_path = args.encoded_cache_dir / f"{cache_identity}.pt"
            if cache_path.exists():
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
                if payload.get("identity") != cache_identity:
                    raise RuntimeError(f"encoded eval cache identity mismatch: {cache_path}")
                encoded = payload["encoded"]
                encoded_cache_stats["hits"] += 1
        if encoded is None:
            encoded = encode_dataset_detailed(
                encoder, np.asarray(stream.windows),
                stream_channel_descriptions(stream.dataset, stream.stream),
                device, float(stream.rate_hz),
                _stream_gravity_state(stream.dataset, stream.stream),
                channel_mask=stream.mask, dataset=stream.dataset, stream=stream.stream,
                export_sensor_rows=predictor_mode == "admissibility_gate",
            )
            # Phase-B evaluation is frozen. Keeping detailed patch/sensor exports on CUDA only
            # consumes VRAM between cells; move once to CPU and transfer selected rows on demand.
            encoded = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in encoded.items()
            }
            encoded_cache_stats["misses"] += 1
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
                torch.save({"identity": cache_identity, "encoded": encoded}, temporary)
                os.replace(temporary, cache_path)
        value = {
            "stream": stream,
            "encoded": encoded,
            "labels": np.asarray(
                align_ground_truth_labels(stream.gt, stream.eval_labels), dtype=object
            ),
            "subjects": np.asarray(stream.subjects, dtype=object),
            "executions": np.asarray(stream.execution_ids, dtype=object),
            # "recording" means the converter told us which continuous capture each label block came
            # from, so an execution is a real capture; "block" means one contiguous label run is the
            # finest unit available. A k-curve read off "block" ids on a source that splits one bout
            # into many blocks is adjacency, not enrollment, so the artifact has to say which.
            "execution_granularity": stream.execution_granularity,
        }
        encoded_streams[cache_key] = value
        return value

    results = {}
    protocol = {}
    for dataset in args.datasets:
        primary_specs = deployment_policy.stream_specs(dataset, "primary")
        all_deployment_specs = tuple(
            spec for spec in deployment_policy.stream_specs(dataset, None)
            if spec.device_profile in {"phone", "watch", "device"}
        )
        for query_spec in primary_specs:
            try:
                query_source = load_and_encode(query_spec)
            except FileNotFoundError:
                continue
            support_specs = []
            if "same" in args.configuration_modes:
                support_specs.append(("same_configuration", query_spec))
            if "cross" in args.configuration_modes:
                support_specs.extend(
                    ("cross_configuration", spec)
                    for spec in all_deployment_specs if spec.stream_id != query_spec.stream_id
                )
            for configuration_relation, support_spec in support_specs:
                try:
                    support_source = (
                        query_source if support_spec == query_spec else load_and_encode(support_spec)
                    )
                except FileNotFoundError:
                    continue
                common_labels = [
                    label for label in query_source["stream"].eval_labels
                    if label in set(support_source["labels"].tolist())
                ]
                if len(common_labels) < 2:
                    continue
                relation_id = (
                    f"{dataset}/{query_spec.stream_id}/from_{support_spec.stream_id}/"
                    f"{configuration_relation}"
                )
                relation_seed = args.seed + int(
                    hashlib.sha256(relation_id.encode()).hexdigest()[:8], 16
                )
                for mode in args.modes:
                    protocol_key = f"{relation_id}/{mode}"
                    fixed_aliases = None
                    if shared_manifest is not None:
                        manifest_cell = shared_manifest["cells"].get(protocol_key)
                        if manifest_cell is None:
                            continue
                        manifest_seed_payload = manifest_cell["seeds"][str(manifest_seed)]
                        plans = _manifest_plans(manifest_seed_payload)
                        fixed_aliases = manifest_seed_payload.get("aliases")
                        coverage = {
                            "status": manifest_cell["status"],
                            "support_ceiling": int(manifest_cell["support_ceiling"]),
                            "subjects": len(plans),
                            "candidate_count_min": len(manifest_cell["candidate_names"]),
                            "candidate_count_median": float(len(manifest_cell["candidate_names"])),
                            "candidate_count_max": len(manifest_cell["candidate_names"]),
                            "manifest_fingerprint": shared_manifest["manifest_fingerprint"],
                        }
                    else:
                        plans, coverage = build_paired_enrollment_plans(
                            query_source["labels"], query_source["subjects"],
                            query_source["executions"], common_labels,
                            requested_support=args.support, mode=mode, seed=relation_seed,
                            support_labels=support_source["labels"],
                            support_subjects=support_source["subjects"],
                            support_execution_ids=support_source["executions"],
                        )
                    coverage.update({
                        "query_stream": query_spec.stream_id,
                        "support_stream": support_spec.stream_id,
                        "configuration_relation": configuration_relation,
                        "subject_relation": mode,
                        # Whether an "execution" here is a verified continuous capture or merely one
                        # contiguous label block. A curve read off block ids on a source that cuts a
                        # single bout into many blocks measures adjacency, not enrollment.
                        "query_execution_granularity": query_source["execution_granularity"],
                        "support_execution_granularity":
                            support_source["execution_granularity"],
                    })
                    protocol[protocol_key] = coverage
                    cells = []
                    if 0 in args.support and not args.skip_zero_shot and not args.random_aliases \
                            and configuration_relation == "same_configuration" \
                            and (shared_manifest is None or mode == "cross_subject"):
                        cells.append(("zero", 0))
                    for support_count in sorted({value for value in args.support if value > 0}):
                        shapes = ("full",) if args.random_aliases else args.enrollment_shapes
                        cells.extend((shape, support_count) for shape in shapes)
                    for enrollment_shape, support_count in cells:
                        key = f"{relation_id}/{mode}/{enrollment_shape}/k{support_count}"
                        cell_plans = plans
                        cell_coverage = coverage
                        cell_fixed_aliases = fixed_aliases
                        cohort = "main"
                        if shared_manifest is not None and support_count == 0:
                            zero_id = f"{dataset}/{query_spec.stream_id}/zero_shot"
                            zero_cell = shared_manifest["cells"].get(zero_id)
                            if zero_cell is None:
                                continue
                            cell_plans = _manifest_plans(zero_cell["seeds"]["0"])
                            cell_coverage = {
                                **coverage,
                                "status": zero_cell["status"],
                                "subjects": len(cell_plans),
                                "support_ceiling": 0,
                            }
                        elif shared_manifest is not None and support_count > int(
                            coverage["support_ceiling"]
                        ):
                            secondary = manifest_cell.get("secondary_high_support", {})
                            if secondary.get("status") == "ok" and support_count <= int(
                                secondary.get("support_ceiling", 0)
                            ):
                                secondary_payload = secondary["seeds"][str(manifest_seed)]
                                cell_plans = _manifest_plans(secondary_payload)
                                cell_fixed_aliases = secondary_payload.get("aliases")
                                cell_coverage = {
                                    **coverage,
                                    "status": "ok",
                                    "subjects": len(cell_plans),
                                    "support_ceiling": int(secondary["support_ceiling"]),
                                    "cohort": "secondary_high_support",
                                }
                                cohort = "secondary_high_support"
                        if not cell_plans:
                            result = {**coverage, "queries": 0}
                        elif support_count > int(cell_coverage["support_ceiling"]):
                            result = {
                                **cell_coverage,
                                "status": "above_paired_support_ceiling",
                                "subjects": len(cell_plans),
                                "queries": 0,
                            }
                        else:
                            common = dict(
                                support_count=support_count,
                                random_aliases=args.random_aliases,
                                seed=relation_seed,
                                support_encoded=support_source["encoded"],
                                support_subjects=support_source["subjects"],
                                enrolled_candidate_count=(
                                    0 if enrollment_shape == "partial" else None
                                ),
                                phase_b_seen_labels=phase_b_seen_labels,
                                fixed_aliases=cell_fixed_aliases,
                            )
                            if predictor_mode == "admissibility_gate":
                                result = score_admissibility_cell(
                                    query_source["encoded"], query_source["labels"],
                                    query_source["subjects"], cell_plans, bank,
                                    base_sensor_rows, gate, canonical_text, sbert, aliases, device,
                                    query_dataset=query_source["stream"].dataset,
                                    query_stream=query_source["stream"].stream,
                                    query_mask=query_source["stream"].mask,
                                    support_dataset=support_source["stream"].dataset,
                                    support_stream=support_source["stream"].stream,
                                    support_mask=support_source["stream"].mask,
                                    top_k=args.gate_top_k,
                                    engine=engine,
                                    batch_size=args.batch,
                                    counterfactual_controls=(
                                        args.counterfactual_controls == "full"
                                    ),
                                    **common,
                                )
                            else:
                                result = score_enrollment_cell(
                                    query_source["encoded"], query_source["labels"],
                                    query_source["subjects"], cell_plans, bank,
                                    base_rows, base_selector_z, decoder, retriever,
                                    canonical_text, sbert, aliases, policy, device,
                                    mode=mode, batch_size=args.batch,
                                    same_configuration=(
                                        configuration_relation == "same_configuration"
                                    ),
                                    predictor_mode=predictor_mode,
                                    **common,
                                )
                            result["paired_protocol"] = cell_coverage
                        result.update({
                            "query_stream": query_spec.stream_id,
                            "support_stream": support_spec.stream_id,
                            "configuration_relation": configuration_relation,
                            "subject_relation": mode,
                            "enrollment_shape": enrollment_shape,
                            "support_count": support_count,
                            "cohort": cohort,
                            "phase_a_dataset_seen": dataset in phase_a_datasets,
                            "phase_b_dataset_seen": dataset in phase_b_datasets,
                            "phase_b_any_subject_seen": any(
                                str(plan.subject) in phase_b_subject_names
                                or f"{dataset}:{plan.subject}" in phase_b_subject_names
                                for plan in cell_plans
                            ),
                            "phase_b_all_subjects_seen": bool(cell_plans) and all(
                                str(plan.subject) in phase_b_subject_names
                                or f"{dataset}:{plan.subject}" in phase_b_subject_names
                                for plan in cell_plans
                            ),
                        })
                        results[key] = result
                        if result.get("status"):
                            print(f"{key}: skipped ({result['status']})", flush=True)
                        else:
                            removed = result.get("support_removed_f1_macro")
                            shuffled = result.get("support_label_shuffled_f1_macro")
                            removed_text = "n/a" if removed is None else f"{removed:.1f}"
                            shuffled_text = "n/a" if shuffled is None else f"{shuffled:.1f}"
                            print(
                                f"{key}: F1={result['f1_macro']:.1f} "
                                f"identity={result['identity_f1_macro']:.1f} "
                                f"removed={removed_text} "
                                f"shuffled={shuffled_text} "
                                f"n={result['queries']}",
                                flush=True,
                            )
    suffix = "_aliases" if args.random_aliases else ""
    protocol_tag = "custom" if explicit_datasets else args.protocol_role
    out = args.out or _OUT / (
        f"eval_enrollment_{protocol_tag}_{predictor_mode}{suffix}.json"
    )
    evaluation_protocol = {
        "protocol_role": args.protocol_role,
        "dataset_selection": "explicit" if explicit_datasets else "protocol_roster",
        "datasets": args.datasets,
        "support": args.support,
        "modes": args.modes,
        "configuration_modes": args.configuration_modes,
        "enrollment_shapes": args.enrollment_shapes,
        "seed": args.seed,
        "skip_zero_shot": bool(args.skip_zero_shot),
        "counterfactual_controls": args.counterfactual_controls,
        "shared_manifest": str(args.manifest.resolve()) if args.manifest else None,
        "shared_manifest_fingerprint": (
            shared_manifest["manifest_fingerprint"] if shared_manifest else None
        ),
        "shared_manifest_seed": manifest_seed,
        "encoded_cache": {
            "enabled": not args.no_encoded_cache,
            "directory": str(args.encoded_cache_dir),
            **encoded_cache_stats,
        },
        "gate_top_k": args.gate_top_k,
        "active_windows_per_label": ACTIVE_WINDOWS_PER_LABEL,
        "active_archive_patch_rows": int(len(base_rows)),
        "active_archive_windows": int(
            torch.unique(bank["patch"]["window"][base_rows].long()).numel()
        ),
        "archive_windows": int(len(bank["Z"])),
        "archive_window_fraction_active": float(
            torch.unique(bank["patch"]["window"][base_rows].long()).numel()
            / max(len(bank["Z"]), 1)
        ),
        "active_sensor_rows": (
            int(len(base_sensor_rows.rows.feature)) if base_sensor_rows is not None else None
        ),
        "curve_policy": (
            "fixed_subject_candidate_query_cohort_with_nested_execution_support_"
            "and_fixed_half_candidate_partial_enrollment"
        ),
        "protocol": protocol,
    }
    capabilities = summarize_protocol_capabilities(protocol)
    if capabilities["limitations"]:
        print("[coverage] " + "; ".join(capabilities["limitations"]), flush=True)
    _write_json(out, {
        "schema_version": 1,
        **_git_provenance(),
        "results": results,
        "random_aliases": bool(args.random_aliases),
        "support": args.support,
        "modes": args.modes,
        "configuration_modes": args.configuration_modes,
        "enrollment_shapes": args.enrollment_shapes,
        "protocol_role": args.protocol_role,
        "dataset_selection": "explicit" if explicit_datasets else "protocol_roster",
        "datasets": args.datasets,
        "seed": args.seed,
        "manifest": str(args.manifest.resolve()) if args.manifest else None,
        "manifest_fingerprint": (
            shared_manifest["manifest_fingerprint"] if shared_manifest else None
        ),
        "manifest_seed": manifest_seed,
        "encoded_cache": {
            "enabled": not args.no_encoded_cache,
            "directory": str(args.encoded_cache_dir),
            **encoded_cache_stats,
        },
        "batch_size": args.batch,
        "skip_zero_shot": bool(args.skip_zero_shot),
        "counterfactual_controls": args.counterfactual_controls,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "gate_top_k": args.gate_top_k,
        "active_windows_per_label": ACTIVE_WINDOWS_PER_LABEL,
        "active_archive_patch_rows": int(len(base_rows)),
        "active_archive_windows": int(
            torch.unique(bank["patch"]["window"][base_rows].long()).numel()
        ),
        "archive_windows": int(len(bank["Z"])),
        "archive_window_fraction_active": float(
            torch.unique(bank["patch"]["window"][base_rows].long()).numel()
            / max(len(bank["Z"]), 1)
        ),
        "active_sensor_rows": (
            int(len(base_sensor_rows.rows.feature)) if base_sensor_rows is not None else None
        ),
        "predictor_mode": predictor_mode,
        "protocol": protocol,
        "evaluation_regime": PHASE_B_EVALUATION_REGIME,
        "evaluation_source_fp": evaluation_source_fp,
        "evaluation_protocol_fp": _json_fingerprint(evaluation_protocol),
        "evaluation_capabilities": capabilities,
        "curve_policy": (
            "fixed_subject_candidate_query_cohort_with_nested_execution_support_"
            "and_fixed_half_candidate_partial_enrollment"
        ),
        "controls": [
            ("admissibility_disabled" if predictor_mode == "admissibility_gate"
             else "identity_decoder"),
            "prototype", "l2_ridge_head",
        ] + (
            ["support_removed", "support_labels_cyclically_shifted"]
            if args.counterfactual_controls == "full" else []
        ),
        "bank": str(args.bank),
        "predictor": str(args.predictor),
        "predictor_fp": predictor_fp,
        # Which arm produced this file. Comparisons across checkpoints are only meaningful when
        # these agree on everything except the step, so they are recorded, not inferred.
        "predictor_step": _predictor_step(predictor),
        "predictor_selection": predictor.get("checkpoint_selection"),
        "untrained_control": bool(predictor.get("untrained_control", False)),
        "training_regime": recorded_regime,
        "training_source_fp": recorded_source_fp,
        "current_training_source_fp": current_source_fp,
        "accepted_foreign_source_fingerprint": bool(
            recorded_source_fp != current_source_fp
            and args.accept_training_source_fingerprint
        ),
        "accepted_foreign_regime": (
            recorded_regime if recorded_regime not in native_regimes else None
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_fp": checkpoint_fp,
        "bank_fp": bank.get("bank_fp"),
    })
    print(f"-> {out}")


if __name__ == "__main__":
    main()
