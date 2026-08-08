"""Train the Phase-B patch evidence predictor on answerable candidate episodes.

The sole predictor objective is cross-entropy over the runtime candidate set. True-label support,
acquisition configuration, candidate-set size, and distractor difficulty vary as episode inputs.
Reject confidence is calibrated later by ``train_patch_confidence`` with this predictor frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.scoring import get_sbert_encoder
from model.evidence.confidence import confidence_features
from model.evidence.decoder import DecoderConfig, EvidenceDecoder
from model.evidence.patch_retrieval import PatchSubspaceRetriever
from training.evidence.bank_guard import (
    assert_embedding_path_current,
    assert_bank_matches_backbone,
    assert_bank_current,
    assert_patch_bank,
    assert_patch_embedding_path_current,
    bank_fingerprint,
)
from training.evidence.labeltext import build_label_variants, ensemble_text
from training.evidence.episode_labels import encode_neutral_aliases, episode_label_set
from training.evidence.patch_episodes import (
    EpisodeMemoryView,
    PatchTable,
    assemble_evidence,
    balanced_memory_log_prior,
    build_allowed_mask,
    build_episode_memory_view,
    reweight_evidence,
    realized_support_examples,
    support_capacity_by_label,
)
from training.evidence.live_encoder import PatchViewSpec, SourcePatchEncoder
from training.evidence.policy import (
    ACTIVE_REFRESH_STEPS,
    ACTIVE_WINDOWS_PER_LABEL,
    CANDIDATE_COUNTS,
    DISTRACTOR_MODES,
    EPISODE_TYPES,
    LABEL_TEXT_MODES,
    RETRIEVAL_PROJECTION_EMA,
    RETRIEVAL_SUBSPACE_DIM,
    RETRIEVAL_SUBSPACES,
    RETRIEVAL_TEMPERATURE,
    SOFT_RETRIEVAL_ANNEAL_STEPS,
    SOFT_RETRIEVAL_TEMPERATURE_END,
    SOFT_RETRIEVAL_TEMPERATURE_START,
    SUPPORT_COUNTS,
    TOKENIZER_EMA_DECAY,
    TOKENIZER_FINETUNE_WARMUP_STEPS,
    TOKENIZER_KEY_REFRESH_SHARDS,
    TOKENIZER_KEY_REFRESH_STEPS,
    TOKENIZER_LR_SCALE,
    PhaseBPolicy,
)
from training.evidence.subject_style import sample_subject_style
from training.evidence.telemetry import PhaseBTelemetry
from training.tokenizer.pretrain_data import stream_channel_descriptions
from training.tokenizer.eval_transfer import build_encoder

_DIR = Path(__file__).resolve().parent / "outputs"
_DEFAULT_BANK = _DIR / "memory_bank.pt"
_DEFAULT_OUT = _DIR / "patch_evidence_predictor.pt"
_FAMILY_PATH = Path(__file__).resolve().parents[2] / "data/labels/activity_families.json"
SEED = 20260725


@dataclass(frozen=True)
class AdaptationEpisodeSpec:
    episode_type: str
    support_count: int
    candidate_count: int
    label_mode: str


class EpisodeCurriculum:
    """Exact balanced cycle over the four agreed adaptation regimes."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._queue: list[str] = []

    def sample(self) -> AdaptationEpisodeSpec:
        if not self._queue:
            self._queue = list(self.rng.permutation(EPISODE_TYPES))
        episode_type = self._queue.pop()
        if episode_type == "semantic_zero_support":
            support_count = 0
            label_mode = "coherent"
        else:
            support_count = int(self.rng.choice(SUPPORT_COUNTS))
            label_mode = str(self.rng.choice(LABEL_TEXT_MODES))
        return AdaptationEpisodeSpec(
            episode_type=episode_type,
            support_count=support_count,
            candidate_count=int(self.rng.choice(CANDIDATE_COUNTS)),
            label_mode=label_mode,
        )


def soft_retrieval_temperature(step: int) -> float:
    fraction = min(1.0, max(0.0, float(step) / SOFT_RETRIEVAL_ANNEAL_STEPS))
    return (
        SOFT_RETRIEVAL_TEMPERATURE_START
        + fraction * (
            SOFT_RETRIEVAL_TEMPERATURE_END - SOFT_RETRIEVAL_TEMPERATURE_START
        )
    )


def balanced_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    per_class = [float((pred[true == label] == label).mean()) for label in np.unique(true)]
    return float(np.mean(per_class)) if per_class else float("nan")


def label_index(candidates: torch.Tensor, n_vocab: int, device) -> torch.Tensor:
    position = torch.full((n_vocab,), -1, device=device, dtype=torch.long)
    position[candidates] = torch.arange(len(candidates), device=device)
    return position


def sample_text_tables(variants: torch.Tensor, generator: torch.Generator):
    """Independently sample evidence and candidate phrasings for every vocabulary label."""
    n_labels, n_variants, _ = variants.shape
    ev_pick = torch.randint(n_variants, (n_labels,), generator=generator, device=variants.device)
    cand_pick = torch.randint(n_variants, (n_labels,), generator=generator, device=variants.device)
    rows = torch.arange(n_labels, device=variants.device)
    return variants[rows, ev_pick], variants[rows, cand_pick]


def parameter_gradient_norm(parameters, device) -> torch.Tensor:
    """L2 norm of present gradients, returning zero before a module enters the graph."""
    total = torch.zeros((), dtype=torch.float32, device=device)
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.detach().float().square().sum()
    return total.sqrt()


@torch.no_grad()
def update_tokenizer_ema(ema_encoder, online_encoder, decay: float = TOKENIZER_EMA_DECAY) -> None:
    """Momentum-update trainable encoder state; fixed buffers are copied exactly."""
    for ema_parameter, online_parameter in zip(
        ema_encoder.parameters(), online_encoder.parameters(), strict=True
    ):
        ema_parameter.lerp_(online_parameter.detach(), 1.0 - decay)
    for ema_buffer, online_buffer in zip(
        ema_encoder.buffers(), online_encoder.buffers(), strict=True
    ):
        ema_buffer.copy_(online_buffer.detach())


def synthetic_smoke_bank() -> dict:
    """Small source-shaped bank spanning every current label, config, and subject split."""
    from eval.data import load_global_labels
    from training.evidence.bank_guard import vocab_fingerprint

    generator = torch.Generator().manual_seed(SEED)
    vocab = list(load_global_labels())
    n_labels, d, n_cfg, n_subj = len(vocab), 32, 4, 4
    centers = F.normalize(torch.randn(n_labels, d, generator=generator), dim=-1)
    ys, subjs, cfgs = [], [], []
    for label in range(n_labels):
        for config in range(n_cfg):
            for subject in range(n_subj):
                ys.append(label); cfgs.append(config); subjs.append(subject)
    y = torch.tensor(ys, dtype=torch.long)
    subj = torch.tensor(subjs, dtype=torch.long)
    cfg = torch.tensor(cfgs, dtype=torch.long)
    Z = F.normalize(centers[y] + 0.05 * torch.randn(len(y), d, generator=generator), dim=-1)
    event = torch.arange(len(y), dtype=torch.long)
    event_verified = torch.zeros(len(y), dtype=torch.bool)
    window = torch.arange(len(y)).repeat_interleave(2)
    patch_Z = F.normalize(
        Z[window] + 0.08 * torch.randn(len(window), d, generator=generator), dim=-1
    )
    patch = {
        "Z": patch_Z.half(), "y": y[window], "subj": subj[window], "cfg": cfg[window],
        "sensor": cfg[window],
        "window": window, "event": event[window],
        "event_verified": event_verified[window],
        "time": torch.tensor([0.5, 1.5] * len(y), dtype=torch.float32),
        "duration": torch.ones(len(window)),
        "resolution": torch.tensor([0, 1] * len(y), dtype=torch.long),
        "ordinal": torch.tensor([0, 1] * len(y), dtype=torch.long),
    }
    bank = {
        "schema_version": 3, "Z": Z.half(), "y": y, "subj": subj, "cfg": cfg,
        "event": event, "event_verified": event_verified, "patch": patch,
        "source_row": torch.arange(len(y), dtype=torch.long),
        "patch_embed_probe": torch.zeros(1), "embed_probe": torch.zeros(1),
        "vocab": vocab, "vocab_fp": vocab_fingerprint(vocab),
        "d_model": d, "subj_names": {i: f"smoke-subject-{i}" for i in range(n_subj)},
        "cfg_names": {i: f"smoke/config-{i}" for i in range(n_cfg)},
        "cfg_rate_hz": {i: 50.0 + 10.0 * i for i in range(n_cfg)},
        "event_names": {i: f"smoke-event-{i}" for i in range(len(y))},
        "backbone": {
            "checkpoint": "synthetic-smoke", "step": 0, "val_ba": 0.0,
            "git": "synthetic-smoke", "fingerprint": "synthetic-smoke",
        },
        "corpus": {
            "datasets": ["synthetic-smoke"], "n_streams": n_cfg,
            "streams": {f"synthetic/config-{i}": len(y) // n_cfg for i in range(n_cfg)},
            "n_encoded_windows": len(y), "phase_a_corpus_fp": "synthetic-smoke",
        },
        "archive_budget_windows": len(y), "source_scan_cap_per_stream": len(y),
        "source_alignment": "native",
    }
    bank["bank_fp"] = bank_fingerprint(bank)
    return bank


def load_activity_families(vocab: list[str]) -> tuple[torch.Tensor, dict[str, list[str]], str]:
    payload = json.loads(_FAMILY_PATH.read_text())
    families = {str(name): list(labels) for name, labels in payload["families"].items()}
    flattened = [label for labels in families.values() for label in labels]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(vocab):
        raise ValueError("activity_families.json must map every current vocabulary label exactly once")
    family_index = {}
    for index, labels in enumerate(families.values()):
        family_index.update({label: index for label in labels})
    ids = torch.tensor([family_index[label] for label in vocab], dtype=torch.long)
    fingerprint = hashlib.sha256(_FAMILY_PATH.read_bytes()).hexdigest()[:16]
    return ids, families, fingerprint


def encode_bank_config_text(bank: dict, sbert, device) -> torch.Tensor:
    """Frozen text embeddings for the optional explicit acquisition-text ablation."""
    names = bank["cfg_names"]
    rates = bank["cfg_rate_hz"]
    prompts = []
    for config_id in range(len(names)):
        key = names[config_id] if config_id in names else names[str(config_id)]
        rate = rates[config_id] if config_id in rates else rates[str(config_id)]
        try:
            dataset, stream = key.split("/", 1)
            descriptions = stream_channel_descriptions(dataset, stream)
            prompt = (
                "sensor configuration: " + "; ".join(descriptions)
                + f"; sampling rate {float(rate):g} hertz"
            )
        except (ValueError, KeyError):
            prompt = (
                "sensor configuration: " + str(key).replace("/", ", ")
                + f"; sampling rate {float(rate):g} hertz"
            )
        prompts.append(prompt)
    return torch.from_numpy(sbert(prompts).astype(np.float32)).to(device)


def choose_candidates(
    query_labels: torch.Tensor,
    n_candidates: int,
    n_vocab: int,
    text_table: torch.Tensor,
    physical_centroids: torch.Tensor,
    *,
    truth_present: bool,
    mode: str,
    rng: np.random.Generator,
    allowed_vocab: torch.Tensor | None = None,
    family_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Candidate set with random, language-near, physical-near, or mixed distractors."""
    device = text_table.device
    truth = torch.unique(query_labels).long()
    allowed = (
        torch.arange(n_vocab, device=device)
        if allowed_vocab is None else allowed_vocab.to(device=device, dtype=torch.long)
    )
    required = truth if truth_present else truth.new_empty(0)
    if len(required) > n_candidates:
        raise ValueError(
            f"candidate budget {n_candidates} cannot contain {len(required)} query labels"
        )
    pool = allowed[~torch.isin(allowed, truth)]
    n_distractors = n_candidates - len(required)
    if n_distractors < 1 and not truth_present:
        raise ValueError("truth-absent candidate set needs at least one distractor")
    if n_distractors > len(pool):
        n_distractors = len(pool)

    if mode == "random":
        chosen = rng.choice(pool.cpu().numpy(), size=n_distractors, replace=False)
    elif mode == "motion_family":
        if family_ids is None:
            raise ValueError("motion_family distractors require the canonical family mapping")
        family_ids = family_ids.to(device)
        nearby = pool[torch.isin(family_ids[pool], torch.unique(family_ids[truth]))]
        near_count = min(n_distractors, len(nearby))
        near = (
            rng.choice(nearby.cpu().numpy(), size=near_count, replace=False)
            if near_count else np.empty(0, dtype=np.int64)
        )
        remaining = pool[~torch.isin(pool, torch.as_tensor(near, device=device))]
        random_count = n_distractors - near_count
        random_part = (
            rng.choice(remaining.cpu().numpy(), size=random_count, replace=False)
            if random_count else np.empty(0, dtype=np.int64)
        )
        chosen = np.concatenate([near, random_part])
    else:
        if mode not in {"language", "physical", "mixed"}:
            raise ValueError(f"unknown distractor mode {mode!r}")
        language_score = (
            F.normalize(text_table[pool], dim=-1)
            @ F.normalize(text_table[truth], dim=-1).t()
        ).max(dim=1).values
        physical_score = (
            F.normalize(physical_centroids[pool], dim=-1)
            @ F.normalize(physical_centroids[truth], dim=-1).t()
        ).max(dim=1).values
        score = language_score if mode == "language" else physical_score
        if mode == "mixed":
            score = 0.5 * language_score + 0.5 * physical_score
        near_count = n_distractors if mode != "mixed" else n_distractors // 2
        near = pool[score.topk(min(near_count, len(pool))).indices]
        remainder_pool = pool[~torch.isin(pool, near)]
        random_count = n_distractors - len(near)
        if random_count:
            random_part = rng.choice(
                remainder_pool.cpu().numpy(), size=random_count, replace=False
            )
            chosen = np.concatenate([near.cpu().numpy(), random_part])
        else:
            chosen = near.cpu().numpy()
    result = torch.cat([
        required.to(device),
        torch.as_tensor(chosen, device=device, dtype=torch.long),
    ])
    return result[torch.argsort(result)]


def sample_queries(
    pool: torch.Tensor,
    labels: torch.Tensor,
    y: torch.Tensor,
    n: int,
    rng: np.random.Generator,
    *,
    config_ids: torch.Tensor | None = None,
    subject_ids: torch.Tensor | None = None,
    label_alpha: float = 0.5,
    subject_alpha: float = 0.5,
) -> torch.Tensor:
    """Draw queries hierarchically across label, configuration, subject, then window."""
    candidates = pool[torch.isin(y[pool], labels)]
    if not len(candidates):
        raise ValueError("no query windows have the requested episode labels")
    if not 0.0 <= label_alpha <= 1.0 or not 0.0 <= subject_alpha <= 1.0:
        raise ValueError("label_alpha and subject_alpha must be in [0, 1]")

    candidate_labels = y[candidates].long()
    label_counts = torch.bincount(
        candidate_labels, minlength=int(y.max()) + 1
    ).float().clamp_min(1)
    if config_ids is None:
        # Compatibility path: the default alpha=0.5 reproduces the historical inverse-sqrt
        # per-row query weights.
        weights = label_counts[candidate_labels].pow(label_alpha - 1.0)
    else:
        if len(config_ids) != len(y):
            raise ValueError("config_ids must align 1:1 with y")
        candidate_configs = config_ids[candidates].long()
        if bool((candidate_configs < 0).any()):
            raise ValueError("config_ids must be nonnegative")
        config_base = int(candidate_configs.max()) + 1
        label_config_code = candidate_labels * config_base + candidate_configs
        unique_lc, lc_inverse, lc_counts = torch.unique(
            label_config_code, sorted=True, return_inverse=True, return_counts=True
        )
        lc_labels = torch.div(unique_lc, config_base, rounding_mode="floor")
        configs_per_label = torch.bincount(
            lc_labels, minlength=len(label_counts)
        ).float().clamp_min(1)
        label_mass = label_counts[candidate_labels].pow(label_alpha)

        if subject_ids is None:
            within_config = lc_counts[lc_inverse].float().reciprocal()
        else:
            if len(subject_ids) != len(y):
                raise ValueError("subject_ids must align 1:1 with y")
            candidate_subjects = subject_ids[candidates].long()
            if bool((candidate_subjects < 0).any()):
                raise ValueError("subject_ids must be nonnegative")
            subject_base = int(candidate_subjects.max()) + 1
            lcs_code = label_config_code * subject_base + candidate_subjects
            unique_lcs, lcs_inverse, lcs_counts = torch.unique(
                lcs_code, sorted=True, return_inverse=True, return_counts=True
            )
            lcs_lc_code = torch.div(unique_lcs, subject_base, rounding_mode="floor")
            lcs_lc_index = torch.searchsorted(unique_lc, lcs_lc_code)
            subject_score_sum = torch.zeros(
                len(unique_lc), dtype=torch.float32, device=pool.device
            )
            subject_score_sum.index_add_(
                0, lcs_lc_index, lcs_counts.float().pow(subject_alpha)
            )
            within_config = (
                lcs_counts[lcs_inverse].float().pow(subject_alpha - 1.0)
                / subject_score_sum[lc_inverse]
            )
        weights = (
            label_mass
            / configs_per_label[candidate_labels]
            * within_config
        )
    generator = torch.Generator(device=pool.device).manual_seed(
        int(rng.integers(0, 2**31 - 1))
    )
    draw = min(n, len(candidates))
    return candidates[torch.multinomial(
        weights, draw, replacement=len(candidates) < n, generator=generator
    )]


def sample_queries_covering_labels(
    pool: torch.Tensor,
    labels: torch.Tensor,
    y: torch.Tensor,
    n: int,
    rng: np.random.Generator,
    *,
    config_ids: torch.Tensor,
    subject_ids: torch.Tensor,
) -> torch.Tensor:
    """Guarantee candidate coverage, then fill the episode with balanced query draws."""
    generator = torch.Generator(device=labels.device).manual_seed(
        int(rng.integers(0, 2**31 - 1))
    )
    if n < len(labels):
        chosen_labels = labels[torch.randperm(
            len(labels), generator=generator, device=labels.device,
        )[:n]]
    else:
        chosen_labels = labels
    rows = [
        sample_queries(
            pool, label.view(1), y, 1, rng,
            config_ids=config_ids, subject_ids=subject_ids,
            label_alpha=0.0, subject_alpha=0.5,
        )
        for label in chosen_labels
    ]
    remaining = n - len(rows)
    if remaining > 0:
        rows.append(sample_queries(
            pool, labels, y, remaining, rng,
            config_ids=config_ids, subject_ids=subject_ids,
            label_alpha=0.0, subject_alpha=0.5,
        ))
    result = torch.cat(rows)
    generator = torch.Generator(device=result.device).manual_seed(
        int(rng.integers(0, 2**31 - 1))
    )
    order = torch.randperm(len(result), generator=generator, device=result.device)
    return result[order]


def physical_label_centroids(Z: torch.Tensor, y: torch.Tensor, n_vocab: int) -> torch.Tensor:
    centroids = Z.new_zeros(n_vocab, Z.shape[1])
    centroids.index_add_(0, y, Z)
    counts = torch.bincount(y, minlength=n_vocab).to(Z.dtype).unsqueeze(1)
    return F.normalize(centroids / counts.clamp_min(1), dim=-1)


def family_holdout_labels(
    family_ids: torch.Tensor,
    labels_present_in_val: torch.Tensor,
    n_families: int,
) -> torch.Tensor:
    """Reserve complete canonical motion families that are actually represented in validation."""
    candidates = []
    present = set(labels_present_in_val.detach().cpu().tolist())
    for family in torch.unique(family_ids).tolist():
        members = torch.nonzero(family_ids.eq(family), as_tuple=True)[0]
        represented = [label for label in members.tolist() if label in present]
        if len(represented) >= 2:
            candidates.append((family, represented))
    if len(candidates) < n_families:
        raise ValueError("not enough complete activity families have two validation labels")
    # Deterministic: prefer the most represented families, then stable family id.
    candidates.sort(key=lambda item: (-len(item[1]), item[0]))
    selected_ids = {family for family, _ in candidates[:n_families]}
    return torch.nonzero(
        torch.isin(family_ids, torch.tensor(sorted(selected_ids))), as_tuple=True
    )[0]


def decode_patch_queries(
    dec: EvidenceDecoder,
    retriever: PatchSubspaceRetriever,
    bank: dict,
    index_rows: torch.Tensor,
    memory_index: torch.Tensor,
    query,
    allowed: torch.Tensor,
    candidates: torch.Tensor,
    t_ev: torch.Tensor,
    t_cand: torch.Tensor,
    *,
    policy: PhaseBPolicy,
    memory_config_text: torch.Tensor | None = None,
    query_config_text: torch.Tensor | None = None,
    live_source: SourcePatchEncoder | None = None,
    live_encoder=None,
    live_requires_grad: bool = False,
    query_already_live: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Retrieve and decode an already-packed query patch set."""
    device = next(dec.parameters()).device
    patch = bank["patch"]
    if live_source is not None and not query_already_live:
        if live_encoder is None:
            raise ValueError("live_source requires a live tokenizer encoder")
        query = live_source.reencode_query(
            query, live_encoder, requires_grad=live_requires_grad
        )
    query.Z = F.normalize(query.Z, dim=-1)
    query_patches = int(query.mask.sum(1).max())
    topk = policy.topk_per_subspace(query_patches)
    retrieval = retriever.retrieve(
        query.Z, memory_index, allowed, topk, query_mask=query.mask
    )
    if live_source is None:
        memory_Z = F.normalize(
            torch.as_tensor(patch["Z"])[index_rows.cpu()].float().to(device), dim=-1
        )
        online_score = retriever.score_selected(query.Z, memory_Z, retrieval.index)
        evidence = assemble_evidence(
            retrieval, online_score, index_rows, patch,
            max_evidence=policy.evidence_budget,
            max_per_window=policy.max_per_window,
            max_per_label=policy.max_per_label,
            tau=RETRIEVAL_TEMPERATURE,
        )
        ev_Z = None
    else:
        # Global selection stays detached. Only the bounded final roster is re-encoded, then its
        # query/evidence similarities are recomputed with the online projection and tokenizer.
        evidence = assemble_evidence(
            retrieval, retrieval.score, index_rows, patch,
            max_evidence=policy.evidence_budget,
            max_per_window=policy.max_per_window,
            max_per_label=policy.max_per_label,
            tau=RETRIEVAL_TEMPERATURE,
        )
        ev_Z = F.normalize(
            live_source.reencode_evidence(
                evidence, live_encoder, requires_grad=live_requires_grad
            ),
            dim=-1,
        )
        online_score = retriever.score_pairs(
            query.Z, ev_Z, evidence.head, evidence.query_patch
        )
        evidence = reweight_evidence(
            evidence, online_score, patch, tau=RETRIEVAL_TEMPERATURE
        )
    ev_idx = evidence.index

    def ev_field(name, dtype=None):
        value = torch.as_tensor(patch[name])[ev_idx.detach().cpu()].to(device)
        return value.to(dtype=dtype) if dtype is not None else value

    ev_y = ev_field("y", torch.long)
    ev_window = ev_field("window", torch.long)
    ev_cfg = ev_field("cfg", torch.long)
    ev_sensor = ev_field("sensor", torch.long)
    if ev_Z is None:
        ev_Z = F.normalize(ev_field("Z", torch.float32), dim=-1)
    window_ids = torch.cat([query.window, ev_window], dim=1)
    if memory_config_text is not None:
        ev_config_text = memory_config_text[ev_cfg]
        if query_config_text is None:
            if bool((query.cfg[query.mask] < 0).any()):
                raise ValueError(
                    "external/unknown query configs need explicit query_config_text"
                )
            query_config_text = memory_config_text[query.cfg.clamp_min(0)]
    else:
        ev_config_text = None
    logits, aux = dec(
        zq=query.Z,
        zev=ev_Z,
        ev_label_text=t_ev[ev_y],
        w_retr=evidence.weights,
        cand_text=t_cand[candidates],
        ev_mask=evidence.mask,
        q_mask=query.mask,
        q_config_text=query_config_text,
        ev_config_text=ev_config_text,
        q_time=query.time,
        ev_time=ev_field("time", torch.float32),
        q_duration=query.duration,
        ev_duration=ev_field("duration", torch.float32),
        q_resolution=query.resolution,
        ev_resolution=ev_field("resolution", torch.long),
        q_sensor_id=query.sensor,
        ev_sensor_id=ev_sensor,
        window_id=window_ids,
        return_aux=True,
    )
    features = confidence_features(
        aux["evidence"], evidence.scores, aux["votes"], aux["pool_weights"],
        ev_mask=evidence.mask, ev_sensor_id=ev_sensor,
    )
    raw_candidate = F.normalize(t_cand[candidates], dim=-1)
    raw_evidence = F.normalize(t_ev[ev_y], dim=-1)
    identity_votes = torch.relu(torch.einsum("bkt,ct->bkc", raw_evidence, raw_candidate))
    identity_logits = dec.cfg.out_scale_init * torch.einsum(
        "bk,bkc->bc", evidence.weights, identity_votes
    )
    aux.update({
        "retrieval_prior": evidence.weights,
        "retrieval_scores": evidence.scores,
        "evidence_mask": evidence.mask,
        "evidence_head": evidence.head,
        "evidence_index": evidence.index,
        "evidence_window": ev_window,
        "evidence_sensor": ev_sensor,
        "evidence_resolution": ev_field("resolution", torch.long),
        "evidence_query_patch": evidence.query_patch,
        "retrieval_topk": topk,
        "confidence_features": features,
        "candidate_ids": candidates,
        "query_repr": query.Z[query.mask],
        "evidence_label": ev_y,
        "identity_logits": identity_logits,
    })
    return logits, aux


def run_patch_episode(
    dec: EvidenceDecoder,
    retriever: PatchSubspaceRetriever,
    table: PatchTable,
    bank: dict,
    index_rows: torch.Tensor,
    memory_index: torch.Tensor,
    qi: torch.Tensor,
    candidates: torch.Tensor,
    t_ev: torch.Tensor,
    t_cand: torch.Tensor,
    *,
    truth_present: bool,
    true_support: int | None,
    config_mode: str,
    rng: np.random.Generator,
    policy: PhaseBPolicy,
    query_window_mask: torch.Tensor | None = None,
    config_text: torch.Tensor | None = None,
    live_source: SourcePatchEncoder | None = None,
    live_encoder=None,
    live_requires_grad: bool = False,
) -> tuple[torch.Tensor, dict]:
    """One patch-level training episode with all leakage exclusions."""
    device = next(dec.parameters()).device
    query = table.gather_queries(
        qi, device, expand_verified_events=True, allowed_window_mask=query_window_mask
    )
    y_window = torch.as_tensor(bank["y"], device=device, dtype=torch.long)
    query_label = y_window[qi]
    allowed = build_allowed_mask(
        bank["patch"], index_rows, query, query_label,
        truth_present=truth_present, true_support=true_support,
        config_mode=config_mode, rng=rng,
    )
    support = realized_support_examples(
        bank["patch"], index_rows, allowed, query_label
    )
    if truth_present and true_support is not None and bool((support < true_support).any()):
        raise ValueError(
            f"requested true support {true_support} but only {int(support.min())} is eligible"
        )
    if truth_present and true_support is None and bool((support < 1).any()):
        raise ValueError("requested true support all but at least one query has no eligible support")
    logits, aux = decode_patch_queries(
        dec, retriever, bank, index_rows, memory_index, query, allowed,
        candidates, t_ev, t_cand, policy=policy, memory_config_text=config_text,
        live_source=live_source, live_encoder=live_encoder,
        live_requires_grad=live_requires_grad,
    )
    aux["query_label"] = query_label
    aux["realized_true_support"] = support
    return logits, aux


def _episode_candidate_weights(
    view: EpisodeMemoryView,
    index_rows: torch.Tensor,
    patch: dict,
    canonical_text: torch.Tensor,
    candidate_text: torch.Tensor,
) -> torch.Tensor:
    """Map active rows to candidate votes for the soft backward route."""
    device = candidate_text.device
    n_rows, n_candidates = len(index_rows), len(view.candidate_ids)
    if view.label_mode == "coherent":
        row_y = torch.as_tensor(patch["y"])[index_rows.detach().cpu()].long().to(device)
        weights = torch.relu(
            F.normalize(canonical_text[row_y], dim=-1)
            @ F.normalize(candidate_text, dim=-1).t()
        )
    else:
        weights = torch.zeros(n_rows, n_candidates, device=device)
    if bool(view.support_mask.any()):
        weights[view.support_mask] = F.one_hot(
            view.support_candidate[view.support_mask], num_classes=n_candidates
        ).to(weights.dtype)
    return weights


def _episode_view_specs(
    query,
    view: EpisodeMemoryView,
    index_rows: torch.Tensor,
    patch: dict,
    rng: np.random.Generator,
) -> tuple[list[PatchViewSpec], list[PatchViewSpec]]:
    """Assign persistent subject character and independent acquisition variation."""
    support_style = query_style = None
    if view.episode_type == "same_subject_enrollment":
        support_style = query_style = sample_subject_style(rng)
    elif view.episode_type == "cross_subject_few_support":
        support_style = sample_subject_style(rng)
        query_style = sample_subject_style(rng)
    query_specs = [
        PatchViewSpec(query_style, int(rng.integers(1, 2**31 - 1)))
        for _ in range(query.Z.shape[0])
    ]
    support_global = index_rows.detach().cpu()[view.support_rows.detach().cpu()]
    parent = torch.as_tensor(patch["window"])[support_global].long()
    seed_by_window = {
        int(window): int(rng.integers(1, 2**31 - 1))
        for window in torch.unique(parent).tolist()
    }
    support_specs = [
        PatchViewSpec(support_style, seed_by_window[int(window)])
        for window in parent.tolist()
    ]
    return query_specs, support_specs


def prepare_adaptation_views(
    query,
    view: EpisodeMemoryView,
    bank: dict,
    index_rows: torch.Tensor,
    selector_z: torch.Tensor,
    memory_index: torch.Tensor,
    retriever: PatchSubspaceRetriever,
    *,
    rng: np.random.Generator,
    live_source: SourcePatchEncoder | None,
    selector_encoder=None,
    online_encoder=None,
    online_requires_grad: bool = False,
):
    """Create episode-specific query/support vectors and selector keys."""
    if live_source is None:
        return query, query, selector_z, memory_index
    query_specs, support_specs = _episode_view_specs(
        query, view, index_rows, bank["patch"], rng
    )
    selector_query = live_source.reencode_query_views(
        query, selector_encoder, query_specs, requires_grad=False
    )
    online_model = online_encoder if online_encoder is not None else selector_encoder
    online_query = live_source.reencode_query_views(
        query, online_model, query_specs, requires_grad=online_requires_grad
    ) if online_requires_grad else selector_query

    memory_online = selector_z
    selector_episode_index = memory_index
    support_local = view.support_rows
    if len(support_local):
        support_global = index_rows.detach().cpu()[support_local.detach().cpu()]
        support_selector = live_source.encode_patch_rows_with_views(
            support_global, support_specs, selector_encoder, requires_grad=False
        )
        support_online = (
            live_source.encode_patch_rows_with_views(
                support_global, support_specs, online_model, requires_grad=online_requires_grad
            )
            if online_requires_grad else support_selector
        )
        memory_online = selector_z.index_copy(
            0, support_local.to(selector_z.device), support_online.to(selector_z)
        )
        replacement_index = retriever.project(
            F.normalize(support_selector.to(selector_z), dim=-1), ema=True
        ).detach()
        selector_episode_index = memory_index.index_copy(
            0, support_local.to(memory_index.device), replacement_index
        )
    return selector_query, online_query, memory_online, selector_episode_index


def decode_adaptation_episode(
    dec: EvidenceDecoder,
    retriever: PatchSubspaceRetriever,
    bank: dict,
    index_rows: torch.Tensor,
    selector_z: torch.Tensor,
    memory_index: torch.Tensor,
    query,
    view: EpisodeMemoryView,
    canonical_text: torch.Tensor,
    candidate_text: torch.Tensor,
    row_log_prior: torch.Tensor,
    *,
    policy: PhaseBPolicy,
    soft_tau: float,
    rng: np.random.Generator,
    config_text: torch.Tensor | None = None,
    live_source: SourcePatchEncoder | None = None,
    selector_encoder=None,
    online_encoder=None,
    online_requires_grad: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Hard-forward decoder plus an all-memory soft backward estimator."""
    device = next(dec.parameters()).device
    patch = bank["patch"]
    selector_query, online_query, memory_online, selector_episode_index = (
        prepare_adaptation_views(
            query, view, bank, index_rows, selector_z, memory_index, retriever,
            rng=rng, live_source=live_source, selector_encoder=selector_encoder,
            online_encoder=online_encoder, online_requires_grad=online_requires_grad,
        )
    )
    selector_query.Z = F.normalize(selector_query.Z, dim=-1)
    online_query.Z = F.normalize(online_query.Z, dim=-1)
    memory_online = F.normalize(memory_online, dim=-1)
    topk = policy.topk_per_subspace(int(query.mask.sum(1).max()))
    retrieval = retriever.retrieve(
        selector_query.Z, selector_episode_index, view.allowed, topk,
        query_mask=query.mask,
    )
    online_score = retriever.score_selected(
        online_query.Z, memory_online, retrieval.index
    )
    evidence = assemble_evidence(
        retrieval, online_score, index_rows, patch,
        max_evidence=policy.evidence_budget,
        max_per_window=policy.max_per_window,
        max_per_label=policy.max_per_label,
        tau=RETRIEVAL_TEMPERATURE,
    )
    if evidence.local_index is None:
        raise RuntimeError("assembled evidence omitted active-index row identities")
    ev_Z = memory_online[evidence.local_index]
    ev_idx = evidence.index
    ev_support = view.support_mask[evidence.local_index]
    if online_requires_grad and live_source is not None and online_encoder is not None:
        live_selected = F.normalize(
            live_source.reencode_evidence(
                evidence, online_encoder, requires_grad=True
            ),
            dim=-1,
        )
        # Provided support keeps its episode-specific augmented view; selected background rows
        # use a fresh live forward so future tokenizer fine-tuning receives the hard-path gradient.
        ev_Z = torch.where(ev_support.unsqueeze(-1), ev_Z, live_selected)

    def ev_field(name, dtype=None):
        value = torch.as_tensor(patch[name])[ev_idx.detach().cpu()].to(device)
        return value.to(dtype=dtype) if dtype is not None else value

    ev_y = ev_field("y", torch.long)
    ev_window = ev_field("window", torch.long)
    ev_cfg = ev_field("cfg", torch.long)
    ev_sensor = ev_field("sensor", torch.long)
    ev_support_position = view.support_candidate[evidence.local_index].clamp_min(0)
    ev_label_text = canonical_text[ev_y].clone()
    if bool(ev_support.any()):
        ev_label_text = torch.where(
            ev_support.unsqueeze(-1),
            candidate_text[ev_support_position],
            ev_label_text,
        )
    if config_text is not None:
        ev_config_text = config_text[ev_cfg]
        query_config_text = config_text[query.cfg.clamp_min(0)]
    else:
        ev_config_text = query_config_text = None
    hard_logits, aux = dec(
        zq=online_query.Z,
        zev=ev_Z,
        ev_label_text=ev_label_text,
        w_retr=evidence.weights,
        cand_text=candidate_text,
        ev_mask=evidence.mask,
        ev_support_mask=ev_support,
        q_mask=query.mask,
        q_config_text=query_config_text,
        ev_config_text=ev_config_text,
        q_time=query.time,
        ev_time=ev_field("time", torch.float32),
        q_duration=query.duration,
        ev_duration=ev_field("duration", torch.float32),
        q_resolution=query.resolution,
        ev_resolution=ev_field("resolution", torch.long),
        q_sensor_id=query.sensor,
        ev_sensor_id=ev_sensor,
        window_id=torch.cat([query.window, ev_window], dim=1),
        return_aux=True,
    )
    aux["confidence_features"] = confidence_features(
        aux["evidence"], evidence.scores, aux["votes"], aux["pool_weights"],
        ev_mask=evidence.mask, ev_sensor_id=ev_sensor,
    )
    candidate_weights = _episode_candidate_weights(
        view, index_rows, patch, canonical_text, candidate_text
    )
    soft = retriever.soft_candidate_logits(
        online_query.Z,
        memory_online,
        view.allowed,
        candidate_weights,
        row_log_prior,
        tau=soft_tau,
        query_mask=query.mask,
        selected_index=retrieval.index,
        output_scale=dec.log_out_scale.exp().detach(),
    )
    # Forward is exactly the hard decoder. Backward includes the all-row soft retrieval route.
    logits = hard_logits + (soft.logits - soft.logits.detach())
    aux.update({
        "hard_logits": hard_logits,
        "soft_logits": soft.logits,
        "soft_entropy": soft.entropy,
        "soft_normalized_entropy": soft.normalized_entropy,
        "soft_effective_rows": soft.effective_rows,
        "soft_topk_mass": soft.retained_mass,
        "retrieval_scores": evidence.scores,
        "retrieval_prior": evidence.weights,
        "evidence_mask": evidence.mask,
        "evidence_index": evidence.index,
        "evidence_local_index": evidence.local_index,
        "evidence_window": ev_window,
        "evidence_label": ev_y,
        "evidence_support": ev_support,
        "evidence_support_candidate": ev_support_position,
        "evidence_head": evidence.head,
        "candidate_ids": view.candidate_ids,
        "query_label": view.query_label,
        "realized_true_support": view.support_units_per_candidate,
        "retrieval_topk": topk,
    })
    return logits, aux


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, default=_DEFAULT_BANK)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Phase-A checkpoint; required by --tokenizer-mode ema_finetune")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=None,
                    help="episode batch (default: 8 frozen, 4 ema_finetune)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--evidence-budget", type=int, default=64,
                    help="sole retrieval-capacity knob; K and contribution caps are derived")
    ap.add_argument("--tokenizer-mode", choices=("frozen", "ema_finetune"), default="frozen")
    ap.add_argument("--explicit-config-text", action="store_true",
                    help="ablation: re-inject acquisition/config text already conditioned in Phase A")
    ap.add_argument("--val-families", type=int, default=1,
                    help="complete canonical activity families excluded from every training role")
    ap.add_argument("--val-frac-cfg", type=float, default=0.2)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-episodes", type=int, default=16)
    ap.add_argument("--val-queries", type=int, default=32)
    ap.add_argument("--label-variants", type=int, default=16)
    ap.add_argument("--telemetry-seconds", type=float, default=60.0)
    ap.add_argument("--telemetry-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true",
                    help="run a two-step frozen CPU integration test on a synthetic bank")
    args = ap.parse_args()
    if args.smoke:
        if args.tokenizer_mode != "frozen":
            ap.error("synthetic --smoke supports frozen mode; live fine-tuning is tested on grids")
        args.device = "cpu"
        args.steps = 2
        args.batch = 4
        args.evidence_budget = 8
        args.val_every = 1
        args.val_episodes = 2
        args.val_queries = 4
        args.val_frac_cfg = 0.5
        args.warmup_steps = 1
        args.telemetry_seconds = 0.01
        args.out = Path("/tmp/halo_phase_b_predictor_smoke.pt") if args.out == _DEFAULT_OUT else args.out
    if args.batch is None:
        args.batch = 4 if args.tokenizer_mode == "ema_finetune" else 8
    if args.steps < 1 or args.batch < 1 or args.val_every < 1:
        ap.error("steps, batch, and val-every must be positive")
    if args.warmup_steps < 0 or args.grad_clip <= 0:
        ap.error("warmup-steps must be nonnegative and grad-clip must be positive")
    if args.telemetry_seconds <= 0:
        ap.error("telemetry-seconds must be positive")
    if args.tokenizer_mode == "ema_finetune" \
            and args.steps <= TOKENIZER_FINETUNE_WARMUP_STEPS:
        ap.error(
            "ema_finetune needs more than "
            f"{TOKENIZER_FINETUNE_WARMUP_STEPS} steps so the tokenizer is actually updated"
        )
    policy = PhaseBPolicy(args.evidence_budget, args.tokenizer_mode)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    bank = synthetic_smoke_bank() if args.smoke else torch.load(
        args.bank, map_location="cpu", weights_only=True
    )
    if not args.smoke:
        assert_bank_current(bank, context="train_patch_decoder")
    assert_patch_bank(bank, context="train_patch_decoder")
    table = PatchTable(bank)
    Z = F.normalize(bank["Z"].float(), dim=-1).to(device)
    y = bank["y"].long().to(device)
    subj = bank["subj"].long().to(device)
    cfg = bank["cfg"].long().to(device)
    vocab = list(bank["vocab"])
    n_vocab, d = len(vocab), Z.shape[1]
    family_ids, activity_families, family_fp = load_activity_families(vocab)
    family_ids = family_ids.to(device)

    sbert = get_sbert_encoder()
    text = ensemble_text(vocab, sbert, 8, train_only=True).to(device)
    config_text = encode_bank_config_text(bank, sbert, device) \
        if args.explicit_config_text else None
    variants = (
        build_label_variants(vocab, sbert, args.label_variants, train_only=True).to(device)
        if args.label_variants > 0 else None
    )
    alias_embeddings = encode_neutral_aliases(sbert, device)
    text_gen = torch.Generator(device=device).manual_seed(args.seed)

    # Subject/config-disjoint validation. Held-out families never occur in a training query,
    # candidate, or memory role. Their validation support comes only from the disjoint base fold.
    cfg_ids = np.arange(int(cfg.max()) + 1)
    rng.shuffle(cfg_ids)
    n_val_cfg = max(1, int(len(cfg_ids) * args.val_frac_cfg))
    val_cfg = torch.tensor(cfg_ids[:n_val_cfg], device=device)
    subjects = torch.unique(subj).cpu().numpy()
    rng.shuffle(subjects)
    n_val_subject = max(1, int(len(subjects) * args.val_frac_cfg))
    val_subject = torch.tensor(subjects[:n_val_subject], device=device)
    is_val_cfg = torch.isin(cfg, val_cfg)
    is_val_subject = torch.isin(subj, val_subject)
    raw_val_pool = torch.nonzero(is_val_cfg & is_val_subject, as_tuple=True)[0]
    base_train_pool = torch.nonzero(~is_val_cfg & ~is_val_subject, as_tuple=True)[0]
    if not len(raw_val_pool) or not len(base_train_pool):
        raise SystemExit("empty train/validation fold after config x subject split")
    represented_for_validation = torch.unique(y[raw_val_pool])
    represented_for_validation = represented_for_validation[
        torch.isin(represented_for_validation, torch.unique(y[base_train_pool]))
    ]
    heldout_labels = family_holdout_labels(
        family_ids.cpu(), represented_for_validation.cpu(), args.val_families
    )
    heldout_labels = heldout_labels.to(device)
    train_pool = base_train_pool[~torch.isin(y[base_train_pool], heldout_labels)]
    val_pool = raw_val_pool[torch.isin(y[raw_val_pool], heldout_labels)]
    if len(torch.unique(y[val_pool])) < 2 or len(torch.unique(y[train_pool])) < 3:
        raise SystemExit("activity-family holdout left too few train or validation labels")
    # Hard-distractor centroids may use only the non-validation fold. In particular, never compute
    # a held-out-family centroid from the query windows later used to select a checkpoint.
    physical = physical_label_centroids(
        Z[base_train_pool], y[base_train_pool], n_vocab
    )
    memory_window_mask = torch.zeros(len(Z), dtype=torch.bool, device=device)
    memory_window_mask[train_pool] = True
    val_memory_window_mask = torch.zeros(len(Z), dtype=torch.bool, device=device)
    val_memory_window_mask[base_train_pool] = True
    val_window_mask = torch.zeros(len(Z), dtype=torch.bool, device=device)
    val_window_mask[val_pool] = True
    retriever = PatchSubspaceRetriever(
        d, RETRIEVAL_SUBSPACES, RETRIEVAL_SUBSPACE_DIM, RETRIEVAL_PROJECTION_EMA
    ).to(device)
    dec = EvidenceDecoder(DecoderConfig(
        d_model=d, n_layers=args.layers, n_heads=args.heads,
        candidate_tokens=True, structural_metadata=True, support_role=True,
    )).to(device)
    params = dec.param_groups(args.weight_decay) + [
        {"params": retriever.parameters(), "weight_decay": args.weight_decay},
    ]
    live_source = online_encoder = ema_encoder = None
    checkpoint = None
    if not args.smoke:
        checkpoint_path = args.checkpoint or Path(bank["backbone"]["checkpoint"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"fine-tuning checkpoint not found at {checkpoint_path}; pass --checkpoint"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert_bank_matches_backbone(bank, checkpoint, context="train_patch_decoder fine-tune")
        checkpoint_fp = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if checkpoint_fp != bank["backbone"].get("fingerprint"):
            raise SystemExit("fine-tuning checkpoint is not the encoder that built the memory bank")
        ema_encoder = build_encoder(checkpoint, device, training=False)
        assert_embedding_path_current(
            bank, ema_encoder, device, context="train_patch_decoder fine-tune"
        )
        assert_patch_embedding_path_current(
            bank, ema_encoder, device, context="train_patch_decoder fine-tune"
        )
        live_source = SourcePatchEncoder(bank, device)
        for parameter in ema_encoder.parameters():
            parameter.requires_grad_(False)
        if policy.tokenizer_mode == "ema_finetune":
            online_encoder = build_encoder(checkpoint, device, training=True)
            # The frozen LM is lazy/non-stateful. Reuse the instance/cache warmed by the probes.
            online_encoder.text_encoder = ema_encoder.text_encoder
            params.append({
                "params": [p for p in online_encoder.parameters() if p.requires_grad],
                "weight_decay": args.weight_decay,
                "lr": args.lr * TOKENIZER_LR_SCALE,
            })
    opt = torch.optim.AdamW(params, lr=args.lr)

    def lr_factor(step_index: int) -> float:
        if args.warmup_steps and step_index < args.warmup_steps:
            return float(step_index + 1) / args.warmup_steps
        progress = (step_index - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    index_rows = table.sample_index_rows(
        memory_window_mask, ACTIVE_WINDOWS_PER_LABEL, np.random.default_rng(args.seed + 2)
    )
    val_index_rows = table.sample_index_rows(
        val_memory_window_mask, ACTIVE_WINDOWS_PER_LABEL, np.random.default_rng(args.seed + 3)
    )
    selector_z = F.normalize(
        torch.as_tensor(bank["patch"]["Z"])[index_rows].float().to(device), dim=-1
    )
    memory_index = retriever.build_index(selector_z)
    row_log_prior = balanced_memory_log_prior(bank["patch"], index_rows, device)
    curriculum = EpisodeCurriculum(rng)

    def build_selector_index(rows, *, encoder=None):
        if encoder is None:
            values = torch.as_tensor(bank["patch"]["Z"])[rows].float().to(device)
        else:
            values = live_source.encode_patch_rows(
                rows, encoder, requires_grad=False
            ).detach().to(device)
        return retriever.build_index(F.normalize(values, dim=-1))

    def make_adaptation_episode(
        pool,
        rows,
        spec: AdaptationEpisodeSpec,
        *,
        count,
        local_rng,
        validation=False,
    ):
        present = torch.unique(y[pool])
        capacity = support_capacity_by_label(bank["patch"], rows, n_vocab).to(device)
        required = spec.support_count + (0 if validation else 1)
        present = present[capacity[present] >= required]
        allowed_vocab = present if validation else present[
            ~torch.isin(present, heldout_labels)
        ]
        candidate_count = min(spec.candidate_count, len(allowed_vocab))
        if candidate_count < 2:
            raise ValueError("adaptation episode needs at least two represented candidate labels")
        seed_count = min(2, candidate_count - 1)
        seed = local_rng.choice(
            allowed_vocab.detach().cpu().numpy(), size=seed_count, replace=False
        )
        seed_labels = torch.as_tensor(seed, device=device, dtype=torch.long)
        mode = str(DISTRACTOR_MODES[int(local_rng.integers(len(DISTRACTOR_MODES)))])
        candidates = choose_candidates(
            seed_labels,
            candidate_count,
            n_vocab,
            text,
            physical,
            truth_present=True,
            mode=mode,
            rng=local_rng,
            allowed_vocab=allowed_vocab,
            family_ids=family_ids,
        )
        # choose_candidates may have fewer rows when the represented pool is tiny.
        candidates = candidates[torch.isin(candidates, allowed_vocab)]
        query_pool = pool
        if spec.episode_type == "cross_subject_few_support":
            patch_y = torch.as_tensor(bank["patch"]["y"])[rows.detach().cpu()].long().to(device)
            patch_cfg = torch.as_tensor(bank["patch"]["cfg"])[rows.detach().cpu()].long().to(device)
            patch_subj = torch.as_tensor(bank["patch"]["subj"])[rows.detach().cpu()].long().to(device)
            selected_query_rows = []
            for label in candidates.tolist():
                label_pool = pool[y[pool].eq(label)]
                viable_pairs = []
                for query_cfg in torch.unique(cfg[label_pool]).tolist():
                    config_pool = label_pool[cfg[label_pool].eq(query_cfg)]
                    for query_subj in torch.unique(subj[config_pool]).tolist():
                        support_exists = (
                            patch_y.eq(label)
                            & patch_cfg.ne(query_cfg)
                            & patch_subj.ne(query_subj)
                        ).any()
                        if bool(support_exists):
                            viable_pairs.append((int(query_cfg), int(query_subj)))
                if not viable_pairs:
                    raise ValueError(
                        f"candidate label {label} has no cross-subject/configuration support"
                    )
                chosen_cfg, chosen_subj = viable_pairs[
                    int(local_rng.integers(len(viable_pairs)))
                ]
                selected_query_rows.append(
                    label_pool[
                        cfg[label_pool].eq(chosen_cfg)
                        & subj[label_pool].eq(chosen_subj)
                    ]
                )
            query_pool = torch.cat(selected_query_rows)
        qi = sample_queries_covering_labels(
            query_pool, candidates, y, count, local_rng,
            config_ids=cfg, subject_ids=subj,
        )
        query = table.gather_queries(
            qi,
            device,
            expand_verified_events=True,
            allowed_window_mask=(val_window_mask if validation else memory_window_mask),
        )
        view = build_episode_memory_view(
            bank["patch"], rows, query, y[qi], candidates,
            support_count=spec.support_count,
            episode_type=spec.episode_type,
            label_mode=spec.label_mode,
            rng=local_rng,
        )
        return qi, query, view, mode

    # Fixed held-out-family canaries cover zero-shot semantics and memory-based adaptation.
    val_specs = []
    val_rng = np.random.default_rng(args.seed + 1)
    val_selector_z = F.normalize(
        torch.as_tensor(bank["patch"]["Z"])[val_index_rows].float().to(device), dim=-1
    )
    val_row_log_prior = balanced_memory_log_prior(bank["patch"], val_index_rows, device)
    for i in range(args.val_episodes):
        support_index = i % 4
        requested_support = (0, 1, 4, 8)[support_index]
        episode_type = EPISODE_TYPES[support_index]
        label_mode = "coherent" if requested_support == 0 or i % 2 == 0 else "random_alias"
        episode_seed = args.seed + 1000 + i
        support_attempts = [requested_support]
        if requested_support:
            support_attempts += [value for value in (4, 2, 1) if value < requested_support]
        built = None
        for support in support_attempts:
            val_spec = AdaptationEpisodeSpec(
                episode_type=episode_type,
                support_count=support,
                candidate_count=CANDIDATE_COUNTS[i % len(CANDIDATE_COUNTS)],
                label_mode=label_mode,
            )
            for _attempt in range(50):
                try:
                    qi, query, view, distractor_mode = make_adaptation_episode(
                        val_pool, val_index_rows, val_spec,
                        count=args.val_queries, local_rng=val_rng, validation=True,
                    )
                    label_set = episode_label_set(
                        view.candidate_ids, text, mode=label_mode,
                        rng=val_rng, alias_embeddings=alias_embeddings,
                        canonical_names=vocab,
                    )
                    built = (val_spec, qi, query, view, distractor_mode, label_set)
                    break
                except ValueError:
                    continue
            if built is not None:
                break
        if built is None:
            raise RuntimeError(
                f"could not construct held-out adaptation canary requested_support="
                f"{requested_support}"
            )
        val_spec, qi, query, view, distractor_mode, label_set = built
        val_specs.append({
            "spec": val_spec,
            "qi": qi,
            "query": query,
            "view": view,
            "candidate_text": label_set.embeddings,
            "candidate_phrases": label_set.phrases,
            "distractor_mode": distractor_mode,
            "seed": episode_seed,
            "requested_support": requested_support,
        })

    @torch.no_grad()
    def evaluate():
        dec.eval(); retriever.eval()
        if ema_encoder is not None:
            ema_encoder.eval()
        all_pred, all_true = [], []
        per_cell, true_mass, support_recall = [], [], []
        random_scores = []
        support_removal_drop = []
        alias_permutation_agreement = []
        ran_adaptation_canary = False
        eval_selector_z = val_selector_z
        if policy.tokenizer_mode == "ema_finetune":
            eval_selector_z = F.normalize(
                live_source.encode_patch_rows(
                    val_index_rows, ema_encoder, requires_grad=False
                ).detach().to(device),
                dim=-1,
            )
        val_memory_index = retriever.build_index(eval_selector_z)
        for canary in val_specs:
            spec = canary["spec"]
            qi, view = canary["qi"], canary["view"]
            logits, aux = decode_adaptation_episode(
                dec, retriever, bank, val_index_rows, eval_selector_z, val_memory_index,
                canary["query"], view, text, canary["candidate_text"],
                val_row_log_prior, policy=policy,
                soft_tau=SOFT_RETRIEVAL_TEMPERATURE_END,
                rng=np.random.default_rng(canary["seed"]),
                config_text=config_text,
                live_source=live_source, selector_encoder=ema_encoder,
                online_requires_grad=False,
            )
            true = y[qi]
            pred = view.candidate_ids[logits.argmax(1)]
            cell_ba = balanced_accuracy(pred.cpu().numpy(), true.cpu().numpy())
            per_cell.append((spec.support_count, cell_ba))
            if spec.label_mode == "random_alias":
                random_scores.append(cell_ba)
            all_pred.extend(pred.cpu().tolist())
            all_true.extend(true.cpu().tolist())
            target_position = label_index(view.candidate_ids, n_vocab, device)[true]
            selected_true_support = (
                aux["evidence_support"]
                & aux["evidence_support_candidate"].eq(target_position.unsqueeze(1))
                & aux["evidence_mask"]
            )
            mass = (aux["pool_weights"] * selected_true_support).sum(1)
            true_mass.extend(mass.cpu().tolist())
            support_recall.extend(selected_true_support.any(1).float().cpu().tolist())
            if spec.label_mode == "random_alias" and not ran_adaptation_canary:
                removed_view = replace(
                    view,
                    allowed=view.allowed & ~view.support_mask.view(1, 1, -1),
                    support_mask=torch.zeros_like(view.support_mask),
                    support_candidate=torch.full_like(view.support_candidate, -1),
                    support_units_per_candidate=torch.zeros_like(
                        view.support_units_per_candidate
                    ),
                )
                removed_logits, _ = decode_adaptation_episode(
                    dec, retriever, bank, val_index_rows, eval_selector_z,
                    val_memory_index, canary["query"], removed_view, text,
                    canary["candidate_text"], val_row_log_prior, policy=policy,
                    soft_tau=SOFT_RETRIEVAL_TEMPERATURE_END,
                    rng=np.random.default_rng(canary["seed"]),
                    config_text=config_text, live_source=live_source,
                    selector_encoder=ema_encoder, online_requires_grad=False,
                )
                normal_probability = torch.softmax(aux["hard_logits"], dim=1)
                removed_probability = torch.softmax(removed_logits, dim=1)
                row = torch.arange(len(target_position), device=device)
                support_removal_drop.extend((
                    normal_probability[row, target_position]
                    - removed_probability[row, target_position]
                ).cpu().tolist())
                permutation = torch.roll(
                    torch.arange(len(view.candidate_ids), device=device), shifts=1
                )
                permuted_logits, _ = decode_adaptation_episode(
                    dec, retriever, bank, val_index_rows, eval_selector_z,
                    val_memory_index, canary["query"], view, text,
                    canary["candidate_text"][permutation], val_row_log_prior,
                    policy=policy, soft_tau=SOFT_RETRIEVAL_TEMPERATURE_END,
                    rng=np.random.default_rng(canary["seed"]),
                    config_text=config_text, live_source=live_source,
                    selector_encoder=ema_encoder, online_requires_grad=False,
                )
                alias_permutation_agreement.extend(
                    permuted_logits.argmax(1).eq(logits.argmax(1)).float().cpu().tolist()
                )
                ran_adaptation_canary = True
        zero = [score for support, score in per_cell if support == 0]
        low = [score for support, score in per_cell if support != 0]
        return {
            "macro_cell_ba": float(np.mean([score for _, score in per_cell])),
            "ba": balanced_accuracy(np.asarray(all_pred), np.asarray(all_true)),
            "zero_support_ba": float(np.mean(zero)) if zero else float("nan"),
            "positive_support_ba": float(np.mean(low)) if low else float("nan"),
            "random_alias_ba": float(np.mean(random_scores)) if random_scores else float("nan"),
            "true_support_recall_at_k": float(np.mean(support_recall)),
            "mean_retrieved_true_support_mass": float(np.mean(true_mass)),
            "support_removal_true_probability_drop": (
                float(np.mean(support_removal_drop)) if support_removal_drop else float("nan")
            ),
            "alias_permutation_prediction_agreement": (
                float(np.mean(alias_permutation_agreement))
                if alias_permutation_agreement else float("nan")
            ),
        }

    best = {"macro_cell_ba": -float("inf")}
    best_step = 0
    best_state = None
    t0 = time.time()
    active_refreshes = 0
    tokenizer_key_refreshes = 0
    telemetry = PhaseBTelemetry(
        args.telemetry_dir or (args.out.parent / "telemetry"),
        interval_seconds=args.telemetry_seconds,
    )
    for step in range(1, args.steps + 1):
        dec.train(); retriever.train()
        if online_encoder is not None:
            online_encoder.train()
        tokenizer_active = online_encoder is not None and step > TOKENIZER_FINETUNE_WARMUP_STEPS
        active_refreshed = False
        if step > 1 and (step - 1) % ACTIVE_REFRESH_STEPS == 0:
            index_rows = table.sample_index_rows(
                memory_window_mask,
                ACTIVE_WINDOWS_PER_LABEL,
                np.random.default_rng(args.seed + 10_000 + step),
            )
            selector_values = (
                live_source.encode_patch_rows(
                    index_rows, ema_encoder, requires_grad=False
                ).detach().to(device)
                if tokenizer_active else
                torch.as_tensor(bank["patch"]["Z"])[index_rows].float().to(device)
            )
            selector_z = F.normalize(selector_values, dim=-1)
            memory_index = retriever.build_index(selector_z)
            row_log_prior = balanced_memory_log_prior(bank["patch"], index_rows, device)
            active_refreshes += 1
            active_refreshed = True

        episode_error = None
        spec = curriculum.sample()
        if args.smoke and spec.support_count > 1:
            spec = AdaptationEpisodeSpec(
                spec.episode_type, 1, spec.candidate_count, spec.label_mode
            )
        for _attempt in range(20):
            t_ev, t_cand = (
                (text, text) if variants is None else sample_text_tables(variants, text_gen)
            )
            try:
                qi, query, view, distractor_mode = make_adaptation_episode(
                    train_pool, index_rows, spec, count=args.batch,
                    local_rng=rng, validation=False,
                )
                episode_labels = episode_label_set(
                    view.candidate_ids,
                    t_cand,
                    mode=spec.label_mode,
                    rng=rng,
                    alias_embeddings=alias_embeddings,
                    canonical_names=vocab,
                )
                logits, aux = decode_adaptation_episode(
                    dec, retriever, bank, index_rows, selector_z, memory_index,
                    query, view, t_ev, episode_labels.embeddings,
                    row_log_prior,
                    policy=policy,
                    soft_tau=soft_retrieval_temperature(step),
                    rng=rng,
                    config_text=config_text,
                    live_source=live_source,
                    selector_encoder=ema_encoder,
                    online_encoder=online_encoder,
                    online_requires_grad=tokenizer_active,
                )
                break
            except ValueError as exc:
                if not any(token in str(exc) for token in (
                    "no eligible", "no evidence", "support units", "no episodic memory",
                )):
                    raise
                episode_error = exc
        else:
            raise RuntimeError(
                "could not draw a feasible adaptation episode in 20 attempts"
            ) from episode_error
        candidates = view.candidate_ids
        target = label_index(candidates, n_vocab, device)[y[qi]]
        if bool((target < 0).any()):
            raise RuntimeError("answerable episode omitted a query label from candidates")
        loss = F.cross_entropy(logits, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite Phase-B loss at step {step}")
        opt.zero_grad(set_to_none=True)
        hard_retriever_grad = soft_retriever_grad = float("nan")
        if telemetry.due():
            hard_probe = F.cross_entropy(aux["hard_logits"], target)
            soft_proxy = aux["hard_logits"].detach() + (
                aux["soft_logits"] - aux["soft_logits"].detach()
            )
            soft_probe = F.cross_entropy(soft_proxy, target)
            hard_grad = torch.autograd.grad(
                hard_probe, retriever.proj, retain_graph=True, allow_unused=True
            )[0]
            soft_grad = torch.autograd.grad(
                soft_probe, retriever.proj, retain_graph=True, allow_unused=True
            )[0]
            hard_retriever_grad = 0.0 if hard_grad is None else float(hard_grad.norm())
            soft_retriever_grad = 0.0 if soft_grad is None else float(soft_grad.norm())
        loss.backward()
        dec_grad = parameter_gradient_norm(dec.parameters(), device)
        retrieval_grad = parameter_gradient_norm(retriever.parameters(), device)
        tokenizer_grad = torch.tensor(0.0, device=device)
        if online_encoder is not None:
            tokenizer_grad = parameter_gradient_norm(online_encoder.parameters(), device)
        trainable_params = list(dec.parameters()) + list(retriever.parameters())
        if online_encoder is not None:
            trainable_params += [p for p in online_encoder.parameters() if p.requires_grad]
        preclip_grad = torch.nn.utils.clip_grad_norm_(
            trainable_params, args.grad_clip,
        )
        opt.step(); sched.step(); retriever.update_ema()
        if tokenizer_active:
            update_tokenizer_ema(ema_encoder, online_encoder)
        if step % TOKENIZER_KEY_REFRESH_STEPS == 0:
            if tokenizer_active and not active_refreshed:
                ema_encoder.eval()
                selector_z = live_source.refresh_shard(
                    index_rows, selector_z, ema_encoder,
                    shard=tokenizer_key_refreshes,
                    n_shards=TOKENIZER_KEY_REFRESH_SHARDS,
                )
                tokenizer_key_refreshes += 1
            memory_index = retriever.build_index(selector_z)

        with torch.no_grad():
            hard_probability = torch.softmax(aux["hard_logits"], dim=1)
            hard_entropy = -(
                hard_probability * hard_probability.clamp_min(1e-12).log()
            ).sum(1)
            hard_entropy = hard_entropy / max(np.log(len(candidates)), 1e-8)
            top2 = hard_probability.topk(min(2, len(candidates)), dim=1).values
            candidate_margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0)
            selected = aux["evidence_local_index"][aux["evidence_mask"]]
            counts = torch.bincount(selected, minlength=len(index_rows)).float()
            selected_concentration = float(counts.max() / counts.sum().clamp_min(1))
            true_position = target
            selected_true_support = (
                aux["evidence_support"]
                & aux["evidence_support_candidate"].eq(true_position.unsqueeze(1))
                & aux["evidence_mask"]
            )
            support_recall = float(selected_true_support.any(1).float().mean())
            target_hist = torch.bincount(target, minlength=len(candidates)).float()
            target_position_max_share = float(target_hist.max() / target_hist.sum())
            head_mass = torch.zeros(RETRIEVAL_SUBSPACES, device=device)
            head_mass.scatter_add_(
                0,
                aux["evidence_head"][aux["evidence_mask"]],
                aux["pool_weights"][aux["evidence_mask"]],
            )
            head_mass = head_mass / head_mass.sum().clamp_min(1e-8)
            unique_windows = float(np.mean([
                torch.unique(aux["evidence_window"][b, row]).numel()
                for b, row in enumerate(aux["evidence_mask"])
            ]))
            unique_labels = float(np.mean([
                torch.unique(aux["evidence_label"][b, row]).numel()
                for b, row in enumerate(aux["evidence_mask"])
            ]))
        telemetry_metrics = {
            "loss": float(loss.detach()),
            "decoder_grad_norm": float(dec_grad),
            "retriever_grad_norm": float(retrieval_grad),
            "tokenizer_grad_norm": float(tokenizer_grad),
            "preclip_grad_norm": float(preclip_grad),
            "hard_retriever_probe_grad_norm": hard_retriever_grad,
            "soft_retriever_probe_grad_norm": soft_retriever_grad,
            "retrieval_normalized_entropy": float(aux["soft_normalized_entropy"].detach()),
            "retrieval_effective_rows": float(aux["soft_effective_rows"].detach()),
            "topk_retained_soft_mass": float(aux["soft_topk_mass"].detach()),
            "selected_row_max_share": selected_concentration,
            "true_support_recall_at_k": support_recall,
            "candidate_normalized_entropy": float(hard_entropy.mean()),
            "candidate_top1_margin": float(candidate_margin.mean()),
            "target_position_max_share": target_position_max_share,
            "unique_evidence_windows": unique_windows,
            "unique_evidence_labels": unique_labels,
        }
        telemetry_metrics.update({
            f"subspace_{head}_mass": float(value)
            for head, value in enumerate(head_mass)
        })
        telemetry.update(telemetry_metrics, categories={
            "episode_type": spec.episode_type,
            "label_mode": spec.label_mode,
            "support_count": str(spec.support_count),
            "candidate_count": str(len(candidates)),
            "target_position": target.detach().cpu().tolist(),
        })

        if step == 1 or step % args.val_every == 0:
            metrics = evaluate()
            telemetry.set_validation(metrics)
            better = metrics["macro_cell_ba"] > best["macro_cell_ba"]
            if better:
                best, best_step = dict(metrics), step
                best_state = {
                    "decoder": {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()},
                    "retriever": {
                        k: v.detach().cpu().clone() for k, v in retriever.state_dict().items()
                    },
                }
                if ema_encoder is not None:
                    best_state["tokenizer_online"] = {
                        k: v.detach().cpu().clone() for k, v in online_encoder.state_dict().items()
                    }
                    best_state["tokenizer_ema"] = {
                        k: v.detach().cpu().clone() for k, v in ema_encoder.state_dict().items()
                    }
            mask = aux["evidence_mask"]
            usage = torch.zeros(RETRIEVAL_SUBSPACES, device=device)
            usage.scatter_add_(
                0, aux["evidence_head"][mask], aux["pool_weights"][mask]
            )
            usage = usage / usage.sum().clamp_min(1e-8)
            true_mass = (
                aux["pool_weights"]
                * aux["evidence_label"].eq(y[qi].unsqueeze(1)).to(aux["pool_weights"].dtype)
            ).sum(1).mean()
            tokenizer_ema_drift = 0.0
            if ema_encoder is not None:
                delta_sq = sum(
                    (online.detach().float() - ema.detach().float()).square().sum()
                    for online, ema in zip(
                        online_encoder.parameters(), ema_encoder.parameters(), strict=True
                    )
                )
                base_sq = sum(
                    ema.detach().float().square().sum() for ema in ema_encoder.parameters()
                )
                tokenizer_ema_drift = float(torch.sqrt(delta_sq / base_sq.clamp_min(1e-12)))
            print(json.dumps({
                "step": step, "loss": round(float(loss.detach()), 5),
                "candidate_count": len(candidates),
                "episode_type": spec.episode_type,
                "label_mode": spec.label_mode,
                "true_support": spec.support_count,
                "realized_true_support_mean": round(
                    float(aux["realized_true_support"].float().mean()), 3
                ),
                "retrieved_true_mass": round(float(true_mass.detach()), 4),
                "head_usage": [round(float(value.detach()), 4) for value in usage],
                "decoder_grad_norm": round(float(dec_grad), 4),
                "retriever_grad_norm": round(float(retrieval_grad), 4),
                "tokenizer_grad_norm": round(float(tokenizer_grad), 4),
                "tokenizer_active": tokenizer_active,
                "tokenizer_ema_relative_drift": round(tokenizer_ema_drift, 6),
                "active_index_refreshes": active_refreshes,
                "tokenizer_key_refreshes": tokenizer_key_refreshes,
                "preclip_grad_norm": round(float(preclip_grad), 4),
                "delta_norm": round(float(aux["delta_norm"]), 4),
                "candidate_delta_norm": round(float(aux["candidate_delta_norm"]), 4),
                "lr": opt.param_groups[0]["lr"],
                "retrieval_topk": int(aux["retrieval_topk"]),
                "soft_tau": round(soft_retrieval_temperature(step), 5),
                "soft_retrieval_entropy": round(float(aux["soft_entropy"].detach()), 4),
                "soft_retrieval_normalized_entropy": round(
                    float(aux["soft_normalized_entropy"].detach()), 4
                ),
                "soft_effective_rows": round(float(aux["soft_effective_rows"].detach()), 2),
                "soft_topk_retained_mass": round(
                    float(aux["soft_topk_mass"].detach()), 4
                ),
                "evidence_count_mean": round(float(mask.sum(1).float().mean()), 2),
                "unique_evidence_windows_mean": round(float(np.mean([
                    torch.unique(aux["evidence_window"][b, row]).numel()
                    for b, row in enumerate(mask)
                ])), 2),
                "distractor_mode": distractor_mode,
                **{key: round(value, 4) for key, value in metrics.items()},
                "best_macro_cell_ba": round(best["macro_cell_ba"], 4),
                "elapsed_s": round(time.time() - t0, 1),
            }), flush=True)
        emitted = telemetry.emit(step=step, elapsed_seconds=time.time() - t0)
        if emitted is not None:
            print(json.dumps({
                "telemetry": str(telemetry.latest),
                "step": step,
                "window_seconds": round(emitted["window_seconds"], 2),
            }), flush=True)

    if best_state is None:
        raise RuntimeError("training completed without a valid checkpoint")
    payload = {
        **best_state,
        "cfg": {
            "d_model": d, "n_layers": args.layers, "n_heads": args.heads,
            "candidate_tokens": True, "structural_metadata": True,
            "support_role": True,
            "explicit_config_text": bool(args.explicit_config_text),
            "n_subspaces": RETRIEVAL_SUBSPACES,
            "subspace_dim": RETRIEVAL_SUBSPACE_DIM,
            "subspace_ema": RETRIEVAL_PROJECTION_EMA,
            "tokenizer_mode": policy.tokenizer_mode,
        },
        "episode_cfg": {
            "episode_types": list(EPISODE_TYPES),
            "candidate_counts": list(CANDIDATE_COUNTS),
            "support_counts": list(SUPPORT_COUNTS),
            "label_text_modes": list(LABEL_TEXT_MODES),
            "mixture": "balanced_cycle",
            "distractor_modes": list(DISTRACTOR_MODES),
            "query_balance": "hierarchical",
            "query_subject_alpha": 0.5,
            "candidate_support_policy": "equal_count_remove_canonical_background",
            "physical_views": "subject_style_then_independent_phase_b_generic",
        },
        "retrieval_cfg": {**policy.as_dict(), "index_seed": args.seed + 2},
        "phase_b_policy": policy.as_dict(),
        "memory_schema": int(bank["schema_version"]),
        "bank_fp": bank.get("bank_fp") or bank_fingerprint(bank),
        "backbone": bank["backbone"], "vocab": vocab,
        "heldout_labels": heldout_labels.cpu().tolist(),
        "heldout_activity_families": sorted({
            name for name, labels in activity_families.items()
            if set(labels) & {vocab[index] for index in heldout_labels.cpu().tolist()}
        }),
        "activity_family_fp": family_fp,
        "fold": {
            "validation_config_ids": val_cfg.cpu().tolist(),
            "validation_subject_ids": val_subject.cpu().tolist(),
        },
        "optimizer_cfg": {
            "lr": args.lr, "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps, "grad_clip": args.grad_clip,
            "tokenizer_lr_scale": TOKENIZER_LR_SCALE,
            "tokenizer_ema_decay": TOKENIZER_EMA_DECAY,
            "tokenizer_finetune_warmup_steps": TOKENIZER_FINETUNE_WARMUP_STEPS,
            "tokenizer_key_refresh_steps": TOKENIZER_KEY_REFRESH_STEPS,
            "tokenizer_key_refresh_shards": TOKENIZER_KEY_REFRESH_SHARDS,
        },
        "objective": "candidate_cross_entropy",
        "phase_b_schema_version": 2,
        "training_regime": "episodic_memory_adaptation_hard_forward_soft_backward_v1",
        "best_step": best_step, "best_metrics": best,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    telemetry.emit(step=args.steps, elapsed_seconds=time.time() - t0, force=True)
    print(f"[patch-dec] best step {best_step}: {best} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
