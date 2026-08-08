"""Train the Phase-B patch evidence predictor on answerable candidate episodes.

The sole predictor objective is cross-entropy over the runtime candidate set. True-label support,
acquisition configuration, candidate-set size, and distractor difficulty vary as episode inputs.
Reject confidence is calibrated later by ``train_patch_confidence`` with this predictor frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from training.evidence.device import resolve_device
from training.evidence.folds import VALIDATION_QUERY_POLICY, phase_b_fold_masks
from training.evidence.patch_episodes import (
    EpisodeMemoryView,
    PatchTable,
    assemble_evidence,
    balanced_memory_log_prior,
    build_allowed_mask,
    build_episode_memory_view,
    describe_episode_composition,
    reweight_evidence,
    realized_support_examples,
    simultaneous_stream_pairs,
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
    PHASE_B_TRAINING_REGIME,
    PHYSICAL_VIEW_MODES,
    RETRIEVAL_PROJECTION_EMA,
    RETRIEVAL_SUBSPACE_DIM,
    RETRIEVAL_SUBSPACES,
    RETRIEVAL_TEMPERATURE,
    SOFT_BACKWARD_SCALE,
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
    physical_view_mode: str = "augmented"


class EpisodeCurriculum:
    """Exact balanced cycle over the four agreed adaptation regimes."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._queue: list[str] = []
        self._view_queue: list[str] = []

    def sample(self) -> AdaptationEpisodeSpec:
        if not self._queue:
            self._queue = list(self.rng.permutation(EPISODE_TYPES))
        if not self._view_queue:
            self._view_queue = list(self.rng.permutation(PHYSICAL_VIEW_MODES))
        episode_type = self._queue.pop()
        physical_view_mode = self._view_queue.pop()
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
            physical_view_mode=physical_view_mode,
        )

    def state_dict(self) -> dict:
        return {
            "queue": list(self._queue),
            "view_queue": list(self._view_queue),
        }

    def load_state_dict(self, state: dict) -> None:
        queue = list(state.get("queue", []))
        if any(value not in EPISODE_TYPES for value in queue):
            raise ValueError("resume checkpoint contains an invalid episode curriculum queue")
        view_queue = list(state.get("view_queue", []))
        if any(value not in PHYSICAL_VIEW_MODES for value in view_queue):
            raise ValueError("resume checkpoint contains an invalid physical-view queue")
        self._queue = queue
        self._view_queue = view_queue


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Replace a checkpoint atomically so interruption cannot leave a partial artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


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
        "schema_version": 3, "population_fp": "synthetic-smoke",
        "Z": Z.half(), "y": y, "subj": subj, "cfg": cfg,
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


def prepare_support_feasible_query_pool(
    pool: torch.Tensor,
    index_rows: torch.Tensor,
    bank: dict,
    labels: torch.Tensor,
    *,
    support_count: int,
    episode_type: str,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reserve support identities before drawing a shared episode query batch.

    A Phase-B support overlay is shared by every query in an episode. Sampling many query
    executions first can therefore exclude more than the single unit assumed by a ``k + 1``
    capacity check. For ordinary/enrollment episodes, reserve ``k`` active support units and remove
    those identities from the query pool. For cross-subject episodes, choose one query subject per
    label only when at least ``k`` units remain on other subjects.
    """
    if support_count == 0:
        return labels, pool
    device = pool.device
    rows = index_rows.detach().cpu().long()
    patch = bank["patch"]
    active_y = torch.as_tensor(patch["y"])[rows].long().to(device)
    active_subj = torch.as_tensor(patch["subj"])[rows].long().to(device)
    active_window = torch.as_tensor(patch["window"])[rows].long().to(device)
    active_event = torch.as_tensor(patch["event"])[rows].long().to(device)
    active_verified = torch.as_tensor(patch["event_verified"])[rows].bool().to(device)
    unit_offset = int(torch.as_tensor(patch["window"]).max()) + 1
    active_unit = torch.where(
        active_verified, active_event + unit_offset, active_window
    )

    window_y = torch.as_tensor(bank["y"], device=device, dtype=torch.long)
    window_subj = torch.as_tensor(bank["subj"], device=device, dtype=torch.long)
    window_event = torch.as_tensor(bank["event"], device=device, dtype=torch.long)
    window_verified = torch.as_tensor(
        bank["event_verified"], device=device, dtype=torch.bool
    )
    pool_unit = torch.where(
        window_verified[pool], window_event[pool] + unit_offset, pool
    )

    feasible_labels = []
    query_parts = []
    for label in labels.tolist():
        label_pool_mask = window_y[pool].eq(label)
        label_pool = pool[label_pool_mask]
        if not len(label_pool):
            continue
        active_label = active_y.eq(label)
        if episode_type == "cross_subject_few_support":
            viable_subjects = []
            for subject in torch.unique(window_subj[label_pool]).tolist():
                available = active_label & active_subj.ne(subject)
                if torch.unique(active_unit[available]).numel() >= support_count:
                    viable_subjects.append(int(subject))
            if not viable_subjects:
                continue
            query_subject = int(rng.choice(np.asarray(viable_subjects)))
            query_part = label_pool[window_subj[label_pool].eq(query_subject)]
        else:
            units = torch.unique(active_unit[active_label])
            if len(units) < support_count:
                continue
            reserved = torch.as_tensor(
                rng.choice(
                    units.detach().cpu().numpy(), size=support_count, replace=False
                ),
                device=device,
                dtype=torch.long,
            )
            query_part = label_pool[~torch.isin(pool_unit[label_pool_mask], reserved)]
        if len(query_part):
            feasible_labels.append(label)
            query_parts.append(query_part)

    if not feasible_labels:
        return labels[:0], pool[:0]
    return (
        torch.tensor(feasible_labels, device=device, dtype=torch.long),
        torch.cat(query_parts),
    )


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
        ev_retrieval_head=evidence.head,
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
    *,
    physical_view_mode: str,
) -> tuple[list[PatchViewSpec], list[PatchViewSpec]]:
    """Build either exact source views or the full subject/acquisition simulation."""
    if physical_view_mode not in PHYSICAL_VIEW_MODES:
        raise ValueError(
            f"physical_view_mode must be one of {PHYSICAL_VIEW_MODES}, "
            f"got {physical_view_mode!r}"
        )
    support_global = index_rows.detach().cpu()[view.support_rows.detach().cpu()]
    if physical_view_mode == "clean":
        return (
            [PatchViewSpec() for _ in range(query.Z.shape[0])],
            [PatchViewSpec() for _ in range(len(support_global))],
        )

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
    physical_view_mode: str = "augmented",
    reuse_stored_clean: bool = False,
):
    """Create episode-specific query/support vectors and selector keys."""
    if live_source is None:
        return query, query, selector_z, memory_index
    if physical_view_mode == "clean" and reuse_stored_clean and not online_requires_grad:
        # Stored fp16 vectors are the clean frozen-encoder path. Re-encoding them only reproduces
        # the same representation and needlessly serializes CPU augmentation/loading with the GPU.
        return query, query, selector_z, memory_index
    query_specs, support_specs = _episode_view_specs(
        query, view, index_rows, bank["patch"], rng,
        physical_view_mode=physical_view_mode,
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
    physical_view_mode: str = "augmented",
) -> tuple[torch.Tensor, dict]:
    """Hard-forward decoder plus an all-memory soft backward estimator."""
    device = next(dec.parameters()).device
    patch = bank["patch"]
    selector_query, online_query, memory_online, selector_episode_index = (
        prepare_adaptation_views(
            query, view, bank, index_rows, selector_z, memory_index, retriever,
            rng=rng, live_source=live_source, selector_encoder=selector_encoder,
            online_encoder=online_encoder, online_requires_grad=online_requires_grad,
            physical_view_mode=physical_view_mode,
            reuse_stored_clean=policy.tokenizer_mode == "frozen",
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
        ev_retrieval_head=evidence.head,
        window_id=torch.cat([query.window, ev_window], dim=1),
        return_aux=True,
    )
    raw_candidate = F.normalize(candidate_text, dim=-1)
    raw_evidence = F.normalize(ev_label_text, dim=-1)
    identity_votes = torch.relu(torch.einsum("bkt,ct->bkc", raw_evidence, raw_candidate))
    identity_logits = dec.cfg.out_scale_init * torch.einsum(
        "bk,bkc->bc", evidence.weights, identity_votes
    )
    aux["confidence_features"] = confidence_features(
        aux["evidence"], evidence.scores, aux["votes"], aux["pool_weights"],
        ev_mask=evidence.mask, ev_sensor_id=ev_sensor,
    )
    # The soft all-row route exists only to estimate gradients for rows outside hard top-k. It has
    # no effect under no_grad evaluation, so do not spend validation/confidence time computing it.
    soft = None
    logits = hard_logits
    if torch.is_grad_enabled() and SOFT_BACKWARD_SCALE > 0:
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
        # Forward remains exactly hard top-k; only the backward estimator is scaled.
        logits = hard_logits + SOFT_BACKWARD_SCALE * (
            soft.logits - soft.logits.detach()
        )
    aux.update({
        "hard_logits": hard_logits,
        "soft_backward_scale": SOFT_BACKWARD_SCALE,
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
        "identity_logits": identity_logits,
    })
    if soft is not None:
        aux.update({
            "soft_logits": soft.logits,
            "soft_entropy": soft.entropy,
            "soft_normalized_entropy": soft.normalized_entropy,
            "soft_effective_rows": soft.effective_rows,
            "soft_topk_mass": soft.retained_mass,
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
    ap.add_argument("--grad-clip", type=float, default=20.0)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--evidence-budget", type=int, default=64,
                    help="sole retrieval-capacity knob; K and contribution caps are derived")
    ap.add_argument("--tokenizer-mode", choices=("frozen", "ema_finetune"), default="frozen")
    ap.add_argument("--explicit-config-text", action="store_true",
                    help="ablation: re-inject acquisition/config text already conditioned in Phase A")
    ap.add_argument("--val-families", type=int, default=2,
                    help="complete canonical activity families excluded from every training role")
    ap.add_argument("--val-frac-cfg", type=float, default=0.2)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-episodes", type=int, default=32)
    ap.add_argument("--val-queries", type=int, default=32)
    ap.add_argument("--label-variants", type=int, default=16)
    ap.add_argument("--telemetry-seconds", type=float, default=60.0)
    ap.add_argument("--telemetry-dir", type=Path, default=None)
    ap.add_argument("--save-every", type=int, default=200,
                    help="write a resumable trainer state every N optimizer steps")
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume from a .last.pt trainer state produced by this command")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true",
                    help="run a two-step frozen CPU integration test on a synthetic bank")
    ap.add_argument("--real-smoke", action="store_true",
                    help="run three frozen steps through the real bank, grids, and encoder")
    args = ap.parse_args()
    if args.smoke and args.real_smoke:
        ap.error("--smoke and --real-smoke are mutually exclusive")
    if args.smoke:
        if args.tokenizer_mode != "frozen":
            ap.error("synthetic --smoke supports frozen mode; live fine-tuning is tested on grids")
        args.device = "cpu"
        args.steps = 2
        args.batch = 4
        args.evidence_budget = 8
        args.val_every = 1
        args.val_episodes = 3
        args.val_queries = 4
        args.val_frac_cfg = 0.5
        args.val_families = 1
        args.warmup_steps = 1
        args.telemetry_seconds = 0.01
        args.save_every = 1
        args.out = Path("/tmp/halo_phase_b_predictor_smoke.pt") if args.out == _DEFAULT_OUT else args.out
    if args.real_smoke:
        if args.tokenizer_mode != "frozen":
            ap.error("--real-smoke checks the default frozen launch path")
        args.steps = 3
        args.batch = 2
        args.evidence_budget = 16
        args.val_every = 1
        args.val_episodes = 4
        args.val_queries = 8
        args.warmup_steps = 1
        args.save_every = 1
        args.telemetry_seconds = 0.01
        args.out = (
            Path("/tmp/halo_phase_b_predictor_real_smoke.pt")
            if args.out == _DEFAULT_OUT else args.out
        )
    if args.batch is None:
        args.batch = 4 if args.tokenizer_mode == "ema_finetune" else 8
    if args.steps < 1 or args.batch < 1 or args.val_every < 1 or args.save_every < 1:
        ap.error("steps, batch, val-every, and save-every must be positive")
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

    device = resolve_device(args.device)
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

    # Training occupies the non-held-subject/non-held-configuration quadrant. Held-family
    # validation uses every other quadrant, separately exposing subject, configuration, and joint
    # transfer instead of silently discarding the two off-diagonal cells.
    cfg_ids = torch.unique(cfg).cpu().numpy()
    rng.shuffle(cfg_ids)
    n_val_cfg = max(1, int(len(cfg_ids) * args.val_frac_cfg))
    val_cfg = torch.tensor(cfg_ids[:n_val_cfg], device=device)
    subjects = torch.unique(subj).cpu().numpy()
    rng.shuffle(subjects)
    n_val_subject = max(1, int(len(subjects) * args.val_frac_cfg))
    val_subject = torch.tensor(subjects[:n_val_subject], device=device)
    fold_masks = phase_b_fold_masks(cfg, subj, val_cfg, val_subject)
    raw_val_pool = torch.nonzero(fold_masks.validation, as_tuple=True)[0]
    base_train_pool = torch.nonzero(fold_masks.train_base, as_tuple=True)[0]
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
        n_retrieval_heads=RETRIEVAL_SUBSPACES,
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
    # Which streams were captured at the same instant. Used only to describe what an episode drew;
    # it never gates candidate selection.
    simultaneous_pairs = simultaneous_stream_pairs(bank["cfg"], bank["event"])
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
            raise ValueError(
                "no eligible support-feasible labels for an adaptation episode"
            )
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
        feasible_candidates, episode_query_pool = prepare_support_feasible_query_pool(
            pool,
            rows,
            bank,
            candidates,
            support_count=spec.support_count,
            episode_type=spec.episode_type,
            rng=local_rng,
        )
        if len(feasible_candidates) != len(candidates):
            raise ValueError(
                "no eligible support-feasible labels for an adaptation episode"
            )
        qi = sample_queries_covering_labels(
            episode_query_pool, candidates, y, count, local_rng,
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

    # Fixed canaries cover the full support curriculum. The held-out set is the checkpoint-selection
    # target; a smaller matched training set provides an interpretable generalization gap.
    validation_recipes = [("semantic_zero_support", 0)] + [
        (episode_type, support)
        for episode_type in EPISODE_TYPES if episode_type != "semantic_zero_support"
        for support in SUPPORT_COUNTS
    ]
    val_selector_z = F.normalize(
        torch.as_tensor(bank["patch"]["Z"])[val_index_rows].float().to(device), dim=-1
    )
    val_row_log_prior = balanced_memory_log_prior(bank["patch"], val_index_rows, device)
    train_canary_index_rows = index_rows.clone()
    train_canary_selector_z = selector_z.detach().clone()
    train_canary_row_log_prior = balanced_memory_log_prior(
        bank["patch"], train_canary_index_rows, device
    )
    validation_query_pools = []
    for fold_name in ("subject_only", "configuration_only", "joint"):
        relation_pool = val_pool[getattr(fold_masks, fold_name)[val_pool]]
        if len(torch.unique(y[relation_pool])) < 2:
            raise RuntimeError(
                f"validation relation {fold_name!r} has fewer than two held-family labels"
            )
        validation_query_pools.append((fold_name, relation_pool))

    def build_fixed_canaries(pool, rows, *, count, validation, seed_offset):
        canaries = []
        local_rng = np.random.default_rng(args.seed + seed_offset)
        for i in range(count):
            fold_relation = None
            episode_pool = pool
            if validation:
                fold_relation, episode_pool = validation_query_pools[
                    i % len(validation_query_pools)
                ]
            episode_type, requested_support = validation_recipes[i % len(validation_recipes)]
            cycle = i // len(validation_recipes)
            label_mode = (
                "coherent" if requested_support == 0
                else LABEL_TEXT_MODES[(cycle + i) % len(LABEL_TEXT_MODES)]
            )
            episode_seed = args.seed + seed_offset * 1000 + i
            support_attempts = [requested_support]
            if requested_support:
                support_attempts += [
                    value for value in reversed(SUPPORT_COUNTS) if value < requested_support
                ]
            built = None
            for support in support_attempts:
                canary_spec = AdaptationEpisodeSpec(
                    episode_type=episode_type,
                    support_count=support,
                    candidate_count=CANDIDATE_COUNTS[i % len(CANDIDATE_COUNTS)],
                    label_mode=label_mode,
                )
                for _attempt in range(50):
                    try:
                        qi, query, view, distractor_mode = make_adaptation_episode(
                            episode_pool, rows, canary_spec,
                            count=args.val_queries, local_rng=local_rng,
                            validation=validation,
                        )
                        label_set = episode_label_set(
                            view.candidate_ids, text, mode=label_mode,
                            rng=local_rng, alias_embeddings=alias_embeddings,
                            canonical_names=vocab,
                        )
                        built = (
                            canary_spec, qi, query, view, distractor_mode, label_set
                        )
                        break
                    except ValueError:
                        continue
                if built is not None:
                    break
            if built is None:
                split = "validation" if validation else "training"
                raise RuntimeError(
                    f"could not construct {split} adaptation canary requested_support="
                    f"{requested_support}"
                )
            canary_spec, qi, query, view, distractor_mode, label_set = built
            for physical_view_mode in PHYSICAL_VIEW_MODES:
                canaries.append({
                    "spec": replace(canary_spec, physical_view_mode=physical_view_mode),
                    "qi": qi,
                    "query": query,
                    "view": view,
                    "candidate_text": label_set.embeddings,
                    "candidate_phrases": label_set.phrases,
                    "distractor_mode": distractor_mode,
                    "seed": episode_seed,
                    "requested_support": requested_support,
                    "fold_relation": fold_relation,
                })
        return canaries

    val_specs = build_fixed_canaries(
        val_pool, val_index_rows, count=args.val_episodes,
        validation=True, seed_offset=1,
    )
    train_canary_specs = build_fixed_canaries(
        train_pool, train_canary_index_rows,
        count=min(args.val_episodes, len(validation_recipes)),
        validation=False, seed_offset=2,
    )

    @torch.no_grad()
    def evaluate():
        dec.eval(); retriever.eval()
        if ema_encoder is not None:
            ema_encoder.eval()
        all_pred, all_identity_pred, all_true = [], [], []
        per_cell, identity_per_cell, true_mass, positive_support_recall = [], [], [], []
        cell_records = []
        random_scores = []
        support_removal_drop = []
        support_label_shuffle_drop = []
        label_renaming_agreement = []
        support_removal_by_view = {mode: [] for mode in PHYSICAL_VIEW_MODES}
        support_label_shuffle_by_view = {mode: [] for mode in PHYSICAL_VIEW_MODES}
        label_renaming_by_view = {mode: [] for mode in PHYSICAL_VIEW_MODES}
        fold_predictions = {
            name: {"pred": [], "true": []}
            for name in ("subject_only", "configuration_only", "joint")
        }
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
                physical_view_mode=spec.physical_view_mode,
            )
            true = y[qi]
            pred = view.candidate_ids[logits.argmax(1)]
            identity_pred = view.candidate_ids[aux["identity_logits"].argmax(1)]
            cell_ba = balanced_accuracy(pred.cpu().numpy(), true.cpu().numpy())
            identity_cell_ba = balanced_accuracy(
                identity_pred.cpu().numpy(), true.cpu().numpy()
            )
            target_position = label_index(view.candidate_ids, n_vocab, device)[true]
            normalized_ce = float(
                F.cross_entropy(logits, target_position)
                / max(np.log(len(view.candidate_ids)), 1e-8)
            )
            per_cell.append((spec.support_count, spec.physical_view_mode, cell_ba))
            identity_per_cell.append(
                (spec.support_count, spec.physical_view_mode, identity_cell_ba)
            )
            cell_records.append({
                "support": spec.support_count,
                "requested_support": canary["requested_support"],
                "physical_view_mode": spec.physical_view_mode,
                "episode_type": spec.episode_type,
                "label_mode": spec.label_mode,
                "ba": cell_ba,
                "identity_ba": identity_cell_ba,
                "loss_over_random": normalized_ce,
            })
            if spec.label_mode == "random_alias":
                random_scores.append(cell_ba)
            all_pred.extend(pred.cpu().tolist())
            all_identity_pred.extend(identity_pred.cpu().tolist())
            all_true.extend(true.cpu().tolist())
            for fold_name in fold_predictions:
                member = getattr(fold_masks, fold_name)[qi]
                fold_predictions[fold_name]["pred"].extend(pred[member].cpu().tolist())
                fold_predictions[fold_name]["true"].extend(true[member].cpu().tolist())
            selected_true_support = (
                aux["evidence_support"]
                & aux["evidence_support_candidate"].eq(target_position.unsqueeze(1))
                & aux["evidence_mask"]
            )
            mass = (aux["pool_weights"] * selected_true_support).sum(1)
            true_mass.extend(mass.cpu().tolist())
            recall_values = selected_true_support.any(1).float().cpu().tolist()
            if spec.support_count > 0:
                positive_support_recall.extend(recall_values)
                cell_records[-1]["true_support_recall_at_k"] = float(np.mean(recall_values))
            if spec.label_mode == "random_alias" and spec.support_count > 0:
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
                    physical_view_mode=spec.physical_view_mode,
                )
                normal_probability = torch.softmax(aux["hard_logits"], dim=1)
                removed_probability = torch.softmax(removed_logits, dim=1)
                row = torch.arange(len(target_position), device=device)
                removal_values = (
                    normal_probability[row, target_position]
                    - removed_probability[row, target_position]
                ).cpu().tolist()
                support_removal_drop.extend(removal_values)
                support_removal_by_view[spec.physical_view_mode].extend(removal_values)

                # Keep the candidate text fixed while assigning every enrolled execution the next
                # candidate's label. A model that genuinely uses enrollment labels must react.
                shuffled_support_candidate = view.support_candidate.clone()
                shuffled_support_candidate[view.support_mask] = (
                    shuffled_support_candidate[view.support_mask] + 1
                ) % len(view.candidate_ids)
                shuffled_view = replace(
                    view, support_candidate=shuffled_support_candidate,
                )
                shuffled_logits, _ = decode_adaptation_episode(
                    dec, retriever, bank, val_index_rows, eval_selector_z,
                    val_memory_index, canary["query"], shuffled_view, text,
                    canary["candidate_text"], val_row_log_prior, policy=policy,
                    soft_tau=SOFT_RETRIEVAL_TEMPERATURE_END,
                    rng=np.random.default_rng(canary["seed"]),
                    config_text=config_text, live_source=live_source,
                    selector_encoder=ema_encoder, online_requires_grad=False,
                    physical_view_mode=spec.physical_view_mode,
                )
                shuffled_probability = torch.softmax(shuffled_logits, dim=1)
                shuffle_values = (
                    normal_probability[row, target_position]
                    - shuffled_probability[row, target_position]
                ).cpu().tolist()
                support_label_shuffle_drop.extend(shuffle_values)
                support_label_shuffle_by_view[spec.physical_view_mode].extend(shuffle_values)

                # Rename every candidate and its enrolled support consistently. This measures label
                # naming stability; it is intentionally not treated as evidence that support is used.
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
                    physical_view_mode=spec.physical_view_mode,
                )
                agreement_values = (
                    permuted_logits.argmax(1).eq(logits.argmax(1)).float().cpu().tolist()
                )
                label_renaming_agreement.extend(agreement_values)
                label_renaming_by_view[spec.physical_view_mode].extend(agreement_values)
        zero = [score for support, _, score in per_cell if support == 0]
        low = [score for support, _, score in per_cell if support != 0]
        macro_cell_ba = float(np.mean([score for _, _, score in per_cell]))
        identity_macro_cell_ba = float(
            np.mean([score for _, _, score in identity_per_cell])
        )
        metrics = {
            "macro_cell_ba": macro_cell_ba,
            "identity_macro_cell_ba": identity_macro_cell_ba,
            "adaptation_macro_cell_ba_gain": macro_cell_ba - identity_macro_cell_ba,
            "ba": balanced_accuracy(np.asarray(all_pred), np.asarray(all_true)),
            "identity_ba": balanced_accuracy(
                np.asarray(all_identity_pred), np.asarray(all_true)
            ),
            "loss_over_random": float(np.mean([
                record["loss_over_random"] for record in cell_records
            ])),
            "zero_support_ba": float(np.mean(zero)) if zero else float("nan"),
            "positive_support_ba": float(np.mean(low)) if low else float("nan"),
            "random_alias_ba": float(np.mean(random_scores)) if random_scores else float("nan"),
            "positive_support_recall_at_k": (
                float(np.mean(positive_support_recall))
                if positive_support_recall else float("nan")
            ),
            "mean_retrieved_true_support_mass": float(np.mean(true_mass)),
            "support_removal_true_probability_drop": (
                float(np.mean(support_removal_drop)) if support_removal_drop else float("nan")
            ),
            "support_label_shuffle_true_probability_drop": (
                float(np.mean(support_label_shuffle_drop))
                if support_label_shuffle_drop else float("nan")
            ),
            "label_renaming_prediction_agreement": (
                float(np.mean(label_renaming_agreement))
                if label_renaming_agreement else float("nan")
            ),
            "support_fallback_fraction": float(np.mean([
                record["support"] != record["requested_support"] for record in cell_records
            ])),
        }
        for support in (0, *SUPPORT_COUNTS):
            selected_records = [record for record in cell_records if record["support"] == support]
            if selected_records:
                metrics[f"support_k{support}_macro_cell_ba"] = float(np.mean([
                    record["ba"] for record in selected_records
                ]))
                metrics[f"support_k{support}_identity_macro_cell_ba"] = float(np.mean([
                    record["identity_ba"] for record in selected_records
                ]))
                metrics[f"support_k{support}_loss_over_random"] = float(np.mean([
                    record["loss_over_random"] for record in selected_records
                ]))
                recalls = [
                    record["true_support_recall_at_k"] for record in selected_records
                    if "true_support_recall_at_k" in record
                ]
                if recalls:
                    metrics[f"support_k{support}_true_support_recall_at_k"] = float(
                        np.mean(recalls)
                    )
        for fold_name, values in fold_predictions.items():
            if values["true"]:
                metrics[f"validation_fold/{fold_name}_ba"] = balanced_accuracy(
                    np.asarray(values["pred"]), np.asarray(values["true"])
                )
                metrics[f"validation_fold/{fold_name}_queries"] = len(values["true"])
        for episode_type in EPISODE_TYPES:
            selected_records = [
                record for record in cell_records if record["episode_type"] == episode_type
            ]
            if selected_records:
                metrics[f"episode/{episode_type}_macro_cell_ba"] = float(np.mean([
                    record["ba"] for record in selected_records
                ]))
        for label_mode in LABEL_TEXT_MODES:
            selected_records = [
                record for record in cell_records if record["label_mode"] == label_mode
            ]
            if selected_records:
                metrics[f"label_mode/{label_mode}_macro_cell_ba"] = float(np.mean([
                    record["ba"] for record in selected_records
                ]))
        for physical_view_mode in PHYSICAL_VIEW_MODES:
            view_scores = [
                score for _, mode, score in per_cell if mode == physical_view_mode
            ]
            identity_view_scores = [
                score for _, mode, score in identity_per_cell
                if mode == physical_view_mode
            ]
            metrics[f"{physical_view_mode}_macro_cell_ba"] = float(np.mean(view_scores))
            metrics[f"{physical_view_mode}_identity_macro_cell_ba"] = float(
                np.mean(identity_view_scores)
            )
            metrics[f"{physical_view_mode}_support_removal_true_probability_drop"] = (
                float(np.mean(support_removal_by_view[physical_view_mode]))
                if support_removal_by_view[physical_view_mode] else float("nan")
            )
            metrics[f"{physical_view_mode}_support_label_shuffle_true_probability_drop"] = (
                float(np.mean(support_label_shuffle_by_view[physical_view_mode]))
                if support_label_shuffle_by_view[physical_view_mode] else float("nan")
            )
            metrics[f"{physical_view_mode}_label_renaming_prediction_agreement"] = (
                float(np.mean(label_renaming_by_view[physical_view_mode]))
                if label_renaming_by_view[physical_view_mode] else float("nan")
            )

        # Matched fixed training canaries use the same episode recipe and metrics. Their purpose is
        # diagnosis, not checkpoint selection: the held-out-family score above remains authoritative.
        train_selector = train_canary_selector_z
        if policy.tokenizer_mode == "ema_finetune":
            train_selector = F.normalize(
                live_source.encode_patch_rows(
                    train_canary_index_rows, ema_encoder, requires_grad=False
                ).detach().to(device),
                dim=-1,
            )
        train_index = retriever.build_index(train_selector)
        train_records = []
        for canary in train_canary_specs:
            spec = canary["spec"]
            logits, aux = decode_adaptation_episode(
                dec, retriever, bank, train_canary_index_rows, train_selector, train_index,
                canary["query"], canary["view"], text, canary["candidate_text"],
                train_canary_row_log_prior, policy=policy,
                soft_tau=SOFT_RETRIEVAL_TEMPERATURE_END,
                rng=np.random.default_rng(canary["seed"]),
                config_text=config_text,
                live_source=live_source, selector_encoder=ema_encoder,
                online_requires_grad=False,
                physical_view_mode=spec.physical_view_mode,
            )
            true = y[canary["qi"]]
            target_position = label_index(
                canary["view"].candidate_ids, n_vocab, device
            )[true]
            pred = canary["view"].candidate_ids[logits.argmax(1)]
            identity_pred = canary["view"].candidate_ids[aux["identity_logits"].argmax(1)]
            train_records.append({
                "support": spec.support_count,
                "ba": balanced_accuracy(pred.cpu().numpy(), true.cpu().numpy()),
                "identity_ba": balanced_accuracy(
                    identity_pred.cpu().numpy(), true.cpu().numpy()
                ),
                "loss_over_random": float(
                    F.cross_entropy(logits, target_position)
                    / max(np.log(len(canary["view"].candidate_ids)), 1e-8)
                ),
            })
        metrics["train_macro_cell_ba"] = float(np.mean([
            record["ba"] for record in train_records
        ]))
        metrics["train_identity_macro_cell_ba"] = float(np.mean([
            record["identity_ba"] for record in train_records
        ]))
        metrics["train_loss_over_random"] = float(np.mean([
            record["loss_over_random"] for record in train_records
        ]))
        metrics["train_validation_macro_cell_ba_gap"] = (
            metrics["train_macro_cell_ba"] - metrics["macro_cell_ba"]
        )
        return metrics

    best = {"macro_cell_ba": -float("inf")}
    best_step = 0
    best_state = None
    t0 = time.time()
    active_refreshes = 0
    tokenizer_key_refreshes = 0
    state_path = args.out.with_name(f"{args.out.stem}.last{args.out.suffix}")
    run_config = {
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "grad_clip": args.grad_clip,
        "layers": args.layers,
        "heads": args.heads,
        "evidence_budget": args.evidence_budget,
        "tokenizer_mode": args.tokenizer_mode,
        "explicit_config_text": bool(args.explicit_config_text),
        "val_families": args.val_families,
        "val_frac_cfg": args.val_frac_cfg,
        "val_episodes": args.val_episodes,
        "val_queries": args.val_queries,
        "label_variants": args.label_variants,
        "seed": args.seed,
    }
    current_bank_fp = bank.get("bank_fp") or bank_fingerprint(bank)

    def save_trainer_state(step: int) -> None:
        state = {
            "kind": "phase_b_patch_decoder_trainer_state_v1",
            "step": step,
            "elapsed_seconds": time.time() - t0,
            "run_config": run_config,
            "bank_fp": current_bank_fp,
            "decoder": dec.state_dict(),
            "retriever": retriever.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "tokenizer_online": (
                online_encoder.state_dict() if online_encoder is not None else None
            ),
            "tokenizer_ema": (
                ema_encoder.state_dict() if online_encoder is not None else None
            ),
            "best": best,
            "best_step": best_step,
            "best_state": best_state,
            "index_rows": index_rows.detach().cpu(),
            "selector_z": selector_z.detach().cpu(),
            "active_refreshes": active_refreshes,
            "tokenizer_key_refreshes": tokenizer_key_refreshes,
            "rng": {
                "numpy_generator": rng.bit_generator.state,
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "text_generator": text_gen.get_state(),
                "curriculum": curriculum.state_dict(),
            },
        }
        atomic_torch_save(state, state_path)

    start_step = 0
    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("kind") != "phase_b_patch_decoder_trainer_state_v1":
            raise SystemExit("--resume is not a Phase-B patch-decoder trainer state")
        if resume.get("bank_fp") != current_bank_fp:
            raise SystemExit("resume checkpoint was built against a different memory bank")
        if resume.get("run_config") != run_config:
            differing = sorted(
                key for key in set(run_config) | set(resume.get("run_config", {}))
                if run_config.get(key) != resume.get("run_config", {}).get(key)
            )
            raise SystemExit(f"resume configuration differs in: {differing}")
        start_step = int(resume["step"])
        if start_step >= args.steps:
            raise SystemExit(
                f"resume state is already at step {start_step}, not below --steps {args.steps}"
            )
        dec.load_state_dict(resume["decoder"])
        retriever.load_state_dict(resume["retriever"])
        opt.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["scheduler"])
        if online_encoder is not None:
            if resume.get("tokenizer_online") is None or resume.get("tokenizer_ema") is None:
                raise SystemExit("ema_finetune resume state lacks tokenizer weights")
            online_encoder.load_state_dict(resume["tokenizer_online"])
            ema_encoder.load_state_dict(resume["tokenizer_ema"])
        best = dict(resume["best"])
        best_step = int(resume["best_step"])
        best_state = resume.get("best_state")
        index_rows = resume["index_rows"].long().cpu()
        selector_z = resume["selector_z"].float().to(device)
        memory_index = retriever.build_index(selector_z)
        row_log_prior = balanced_memory_log_prior(bank["patch"], index_rows, device)
        active_refreshes = int(resume.get("active_refreshes", 0))
        tokenizer_key_refreshes = int(resume.get("tokenizer_key_refreshes", 0))
        saved_rng = resume["rng"]
        rng.bit_generator.state = saved_rng["numpy_generator"]
        torch.set_rng_state(saved_rng["torch"].cpu())
        if device.type == "cuda" and saved_rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved_rng["cuda"]])
        text_gen.set_state(saved_rng["text_generator"].cpu())
        curriculum.load_state_dict(saved_rng["curriculum"])
        t0 = time.time() - float(resume.get("elapsed_seconds", 0.0))
        print(f"[patch-dec] resumed {args.resume} at step {start_step}", flush=True)

    telemetry = PhaseBTelemetry(
        args.telemetry_dir or (args.out.parent / "telemetry" / args.out.stem),
        interval_seconds=args.telemetry_seconds,
        stage="predictor",
    )
    telemetry.start(
        step=start_step,
        elapsed_seconds=time.time() - t0,
        metadata={
            "planned_steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "grad_clip": args.grad_clip,
            "soft_anneal_steps": SOFT_RETRIEVAL_ANNEAL_STEPS,
            "soft_backward_scale": SOFT_BACKWARD_SCALE,
            "n_retrieval_heads": RETRIEVAL_SUBSPACES,
            "output": str(args.out),
            "bank_fp": current_bank_fp,
            "tokenizer_mode": args.tokenizer_mode,
            "resume": str(args.resume) if args.resume is not None else None,
            "run_config": run_config,
        },
    )

    for step in range(start_step + 1, args.steps + 1):
        step_started = time.perf_counter()
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
                spec.episode_type, 1, spec.candidate_count, spec.label_mode,
                spec.physical_view_mode,
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
                    physical_view_mode=spec.physical_view_mode,
                )
                break
            except ValueError as exc:
                if not any(token in str(exc) for token in (
                    "no eligible", "no evidence", "support units", "no episodic memory",
                    "no cross-subject/configuration support",
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
        hard_soft_grad_cosine = hard_soft_grad_ratio = float("nan")
        telemetry_due = telemetry.due()
        if telemetry_due:
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
            if hard_grad is not None and soft_grad is not None:
                hard_flat = hard_grad.detach().float().flatten()
                soft_flat = soft_grad.detach().float().flatten()
                denominator = hard_flat.norm() * soft_flat.norm()
                if float(denominator) > 0:
                    hard_soft_grad_cosine = float(
                        torch.dot(hard_flat, soft_flat) / denominator
                    )
                hard_soft_grad_ratio = float(
                    soft_flat.norm() / hard_flat.norm().clamp_min(1e-12)
                )
        loss.backward()
        dec_grad = parameter_gradient_norm(dec.parameters(), device)
        retrieval_grad = parameter_gradient_norm(retriever.parameters(), device)
        tokenizer_grad = torch.tensor(0.0, device=device)
        if online_encoder is not None:
            tokenizer_grad = parameter_gradient_norm(online_encoder.parameters(), device)
        component_gradients = {}
        if telemetry_due:
            components = {
                "evidence_attention": dec.blocks,
                "evidence_text_refiner": dec.refiner,
                "evidence_pool": dec.pool_phi,
                "candidate_attention": dec.candidate_blocks,
                "candidate_refiner": dec.candidate_refiner,
                "role_embeddings": dec.role_emb,
                "retrieval_head_embeddings": dec.retrieval_head_emb,
            }
            for name, module in components.items():
                if module is not None:
                    component_gradients[f"component_grad_norm/{name}"] = float(
                        parameter_gradient_norm(module.parameters(), device)
                    )
            for name in ("same_window_bias", "same_sensor_bias", "log_out_scale"):
                parameter = getattr(dec, name, None)
                if parameter is not None:
                    component_gradients[f"component_grad_norm/{name}"] = (
                        0.0 if parameter.grad is None else float(parameter.grad.detach().abs())
                    )
        trainable_params = list(dec.parameters()) + list(retriever.parameters())
        if online_encoder is not None:
            trainable_params += [p for p in online_encoder.parameters() if p.requires_grad]
        preclip_grad = torch.nn.utils.clip_grad_norm_(
            trainable_params, args.grad_clip, error_if_nonfinite=True,
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
            train_accuracy = float(aux["hard_logits"].argmax(1).eq(target).float().mean())
            chance_accuracy = 1.0 / len(candidates)
            chance_normalized_accuracy = (
                (train_accuracy - chance_accuracy) / (1.0 - chance_accuracy)
            )
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
            pool_weights = aux["pool_weights"].masked_fill(~aux["evidence_mask"], 0.0)
            pool_entropy = -(
                pool_weights * pool_weights.clamp_min(1e-12).log()
            ).sum(1)
            pool_valid = aux["evidence_mask"].sum(1).clamp_min(2).float()
            pool_normalized_entropy = pool_entropy / pool_valid.log()
            pool_max_share = pool_weights.max(1).values
            support_pool_mass = (
                pool_weights * aux["evidence_support"].to(pool_weights.dtype)
            ).sum(1)
            true_label_pool_mass = (
                pool_weights
                * aux["evidence_label"].eq(y[qi].unsqueeze(1)).to(pool_weights.dtype)
            ).sum(1)
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
            subspace_topk_overlap = float("nan")
            if telemetry_due:
                overlap_values = []
                evidence_index_cpu = aux["evidence_index"].detach().cpu()
                evidence_head_cpu = aux["evidence_head"].detach().cpu()
                evidence_mask_cpu = aux["evidence_mask"].detach().cpu()
                for batch_index in range(len(evidence_index_cpu)):
                    for left in range(RETRIEVAL_SUBSPACES):
                        left_rows = torch.unique(evidence_index_cpu[batch_index][
                            evidence_mask_cpu[batch_index]
                            & evidence_head_cpu[batch_index].eq(left)
                        ])
                        for right in range(left + 1, RETRIEVAL_SUBSPACES):
                            right_rows = torch.unique(evidence_index_cpu[batch_index][
                                evidence_mask_cpu[batch_index]
                                & evidence_head_cpu[batch_index].eq(right)
                            ])
                            union = torch.unique(torch.cat([left_rows, right_rows])).numel()
                            if union:
                                intersection = torch.isin(left_rows, right_rows).sum().item()
                                overlap_values.append(intersection / union)
                if overlap_values:
                    subspace_topk_overlap = float(np.mean(overlap_values))
        telemetry_metrics = {
            "loss": float(loss.detach()),
            "loss_over_random": float(loss.detach()) / max(np.log(len(candidates)), 1e-8),
            "loss_improvement_over_random": float(np.log(len(candidates)) - float(loss.detach())),
            "train_accuracy": train_accuracy,
            "chance_normalized_train_accuracy": chance_normalized_accuracy,
            "hard_forward_max_abs_error": float(
                (logits.detach() - aux["hard_logits"].detach()).abs().max()
            ),
            "decoder_grad_norm": float(dec_grad),
            "retriever_grad_norm": float(retrieval_grad),
            "tokenizer_grad_norm": float(tokenizer_grad),
            "preclip_grad_norm": float(preclip_grad),
            "gradient_clipped_fraction": float(float(preclip_grad) > args.grad_clip),
            "retrieval_normalized_entropy": float(aux["soft_normalized_entropy"].detach()),
            "retrieval_effective_rows": float(aux["soft_effective_rows"].detach()),
            "topk_retained_soft_mass": float(aux["soft_topk_mass"].detach()),
            "selected_row_max_share": selected_concentration,
            "provided_support_recall_at_k": (
                support_recall if spec.support_count > 0 else None
            ),
            "provided_support_pool_mass": float(support_pool_mass.mean()),
            "true_label_pool_mass": float(true_label_pool_mass.mean()),
            "pool_normalized_entropy": float(pool_normalized_entropy.mean()),
            "pool_effective_evidence": float(pool_entropy.exp().mean()),
            "pool_weight_max_share": float(pool_max_share.mean()),
            "candidate_normalized_entropy": float(hard_entropy.mean()),
            "candidate_top1_margin": float(candidate_margin.mean()),
            "target_position_max_share": target_position_max_share,
            "unique_evidence_windows": unique_windows,
            "unique_evidence_labels": unique_labels,
            "evidence_refinement_norm": float(aux["delta_norm"]),
            "candidate_refinement_norm": float(aux["candidate_delta_norm"]),
            "output_scale": float(dec.log_out_scale.detach().exp()),
            "same_window_attention_bias": float(dec.same_window_bias.detach()),
            "same_sensor_attention_bias": float(dec.same_sensor_bias.detach()),
            "learning_rate": float(opt.param_groups[0]["lr"]),
            "episode_draw_attempts": float(_attempt + 1),
            "realized_true_support": float(aux["realized_true_support"].float().mean()),
            "step_seconds": float(time.perf_counter() - step_started),
        }
        if device.type == "cuda":
            telemetry_metrics.update({
                "gpu_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                "gpu_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            })
        if telemetry_due:
            probe_metrics = {
                "hard_retriever_probe_grad_norm": hard_retriever_grad,
                "soft_retriever_probe_grad_norm": soft_retriever_grad,
                "hard_soft_logit_mean_abs_difference": float(
                    (aux["hard_logits"].detach() - aux["soft_logits"].detach()).abs().mean()
                ),
                **component_gradients,
            }
            if np.isfinite(hard_soft_grad_cosine):
                probe_metrics["hard_soft_retriever_grad_cosine"] = hard_soft_grad_cosine
            if np.isfinite(hard_soft_grad_ratio):
                probe_metrics["soft_to_hard_retriever_grad_ratio"] = hard_soft_grad_ratio
                probe_metrics["effective_soft_to_hard_retriever_grad_ratio"] = (
                    SOFT_BACKWARD_SCALE * hard_soft_grad_ratio
                )
            if np.isfinite(subspace_topk_overlap):
                probe_metrics["subspace_topk_jaccard"] = subspace_topk_overlap
            telemetry_metrics.update(probe_metrics)
        telemetry_metrics.update({
            f"subspace_{head}_mass": float(value)
            for head, value in enumerate(head_mass)
        })
        # What this batch actually enrolled, in recorded executions. Cross-stream support is allowed
        # but never required, so these counts are the only record of how much of it a run really saw.
        composition = describe_episode_composition(
            bank["patch"], index_rows, query, view, simultaneous_pairs
        )
        telemetry_metrics.update({
            f"enrolled_{name}": float(value)
            for name, value in composition.items() if isinstance(value, int)
        })
        category_values = {
            "episode_type": spec.episode_type,
            "label_mode": spec.label_mode,
            "physical_view_mode": spec.physical_view_mode,
            "support_count": str(spec.support_count),
            "candidate_count": str(len(candidates)),
            "target_position": target.detach().cpu().tolist(),
            "synthetic_persona": composition["synthetic_persona"],
        }
        telemetry.update(
            telemetry_metrics,
            categories=category_values,
            strata={key: category_values[key] for key in (
                "episode_type", "label_mode", "physical_view_mode",
                "support_count", "candidate_count",
            )},
        )

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
                if online_encoder is not None:
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
            if online_encoder is not None:
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
                "physical_view_mode": spec.physical_view_mode,
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
                "soft_backward_scale": SOFT_BACKWARD_SCALE,
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
                "this_batch_enrolled": composition,
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
        if step % args.save_every == 0 or step == args.steps:
            save_trainer_state(step)

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
            "n_retrieval_heads": RETRIEVAL_SUBSPACES,
            "subspace_dim": RETRIEVAL_SUBSPACE_DIM,
            "subspace_ema": RETRIEVAL_PROJECTION_EMA,
            "tokenizer_mode": policy.tokenizer_mode,
        },
        "episode_cfg": {
            "episode_types": list(EPISODE_TYPES),
            "candidate_counts": list(CANDIDATE_COUNTS),
            "support_counts": list(SUPPORT_COUNTS),
            "physical_view_modes": list(PHYSICAL_VIEW_MODES),
            "clean_physical_view_share": 0.5,
            "label_text_modes": list(LABEL_TEXT_MODES),
            "mixture": "balanced_cycle",
            "distractor_modes": list(DISTRACTOR_MODES),
            "query_balance": "hierarchical",
            "query_subject_alpha": 0.5,
            "candidate_support_policy": "equal_count_remove_canonical_background",
            "physical_views": "balanced_exact_clean_or_subject_style_then_phase_b_generic",
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
            "validation_query_policy": VALIDATION_QUERY_POLICY,
            "window_counts": {
                "train_base": int(fold_masks.train_base.sum()),
                "validation_subject_only": int(fold_masks.subject_only.sum()),
                "validation_configuration_only": int(fold_masks.configuration_only.sum()),
                "validation_joint": int(fold_masks.joint.sum()),
                "validation_query_pool": int(len(val_pool)),
            },
            "validation_query_pool": {
                "labels": int(torch.unique(y[val_pool]).numel()),
                "subjects": int(torch.unique(subj[val_pool]).numel()),
                "configurations": int(torch.unique(cfg[val_pool]).numel()),
            },
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
        "phase_b_schema_version": 3,
        "training_regime": PHASE_B_TRAINING_REGIME,
        "best_step": best_step, "best_metrics": best,
    }
    atomic_torch_save(payload, args.out)
    telemetry.emit(
        step=args.steps, elapsed_seconds=time.time() - t0, force=True, final=True
    )
    print(f"[patch-dec] best step {best_step}: {best} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
