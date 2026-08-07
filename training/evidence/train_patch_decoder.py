"""Train the Phase-B patch evidence predictor on answerable candidate episodes.

The sole predictor objective is cross-entropy over the runtime candidate set. Memory composition,
true-label support, acquisition configuration, and distractor difficulty vary as episode inputs.
Reject confidence is calibrated later by ``train_patch_confidence`` with this predictor frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.scoring import get_sbert_encoder
from model.evidence.confidence import confidence_features
from model.evidence.decoder import DecoderConfig, EvidenceDecoder
from model.evidence.patch_retrieval import PatchSubspaceRetriever
from training.evidence.bank_guard import (
    assert_bank_current,
    assert_patch_bank,
    bank_fingerprint,
)
from training.evidence.labeltext import build_label_variants, ensemble_text
from training.evidence.patch_episodes import (
    PatchTable,
    assemble_evidence,
    build_allowed_mask,
    realized_support_examples,
)
from training.tokenizer.pretrain_data import stream_channel_descriptions

_DIR = Path(__file__).resolve().parent / "outputs"
_DEFAULT_BANK = _DIR / "memory_bank.pt"
_DEFAULT_OUT = _DIR / "patch_evidence_predictor.pt"
_FAMILY_PATH = Path(__file__).resolve().parents[2] / "data/labels/activity_families.json"
SEED = 20260725
SUPPORT_CHOICES = (0, 1, 2, 4, 8, None)
TRUE_SUPPORT_PROBS = (0.35, 0.25, 0.15, 0.10, 0.05, 0.10)
OTHER_SUPPORT_PROBS = (0.05, 0.10, 0.15, 0.20, 0.20, 0.30)


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


def sample_support(
    rng: np.random.Generator,
    choices: tuple[int | None, ...],
    probabilities: tuple[float, ...],
) -> int | None:
    if len(choices) != len(probabilities) or not np.isclose(sum(probabilities), 1.0):
        raise ValueError("support choices and probabilities must align and sum to one")
    return choices[int(rng.choice(len(choices), p=probabilities))]


def choose_memory_labels(
    allowed_vocab: torch.Tensor,
    query_labels: torch.Tensor,
    fraction: float,
    rng: np.random.Generator,
    *,
    require_truth: bool,
) -> torch.Tensor:
    """Choose the labels represented in memory independently of the candidate roster."""
    allowed = allowed_vocab.detach().cpu().numpy()
    required = np.unique(query_labels.detach().cpu().numpy()) if require_truth else np.empty(0, int)
    target = max(len(required), min(len(allowed), max(1, int(round(fraction * len(allowed))))))
    pool = np.setdiff1d(allowed, required, assume_unique=False)
    extra = rng.choice(pool, size=min(target - len(required), len(pool)), replace=False)
    labels = np.unique(np.concatenate([required, np.asarray(extra, dtype=np.int64)]))
    return torch.as_tensor(labels, device=allowed_vocab.device, dtype=torch.long)


def synthetic_smoke_bank() -> dict:
    """Small schema-v2 bank spanning every current label, config, and subject split."""
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
    }
    bank = {
        "schema_version": 2, "Z": Z.half(), "y": y, "subj": subj, "cfg": cfg,
        "event": event, "event_verified": event_verified, "patch": patch,
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
        "max_per_stream": None, "max_per_label": None,
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


def candidate_size(n_vocab: int, fractions: tuple[float, ...],
                   rng: np.random.Generator) -> int:
    fraction = float(fractions[int(rng.integers(len(fractions)))])
    return max(2, min(n_vocab, int(round(fraction * n_vocab))))


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
    topk_per_head: int,
    max_evidence: int,
    max_per_window: int,
    max_per_label: int,
    tau: float,
    memory_config_text: torch.Tensor | None = None,
    query_config_text: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Retrieve and decode an already-packed query patch set."""
    device = next(dec.parameters()).device
    patch = bank["patch"]
    query.Z = F.normalize(query.Z, dim=-1)
    retrieval = retriever.retrieve(
        query.Z, memory_index, allowed, topk_per_head, query_mask=query.mask
    )
    memory_Z = F.normalize(
        torch.as_tensor(patch["Z"])[index_rows.cpu()].float().to(device), dim=-1
    )
    online_score = retriever.score_selected(query.Z, memory_Z, retrieval.index)
    evidence = assemble_evidence(
        retrieval, online_score, index_rows, patch, max_evidence=max_evidence,
        max_per_window=max_per_window, max_per_label=max_per_label, tau=tau,
    )
    ev_idx = evidence.index

    def ev_field(name, dtype=None):
        value = torch.as_tensor(patch[name])[ev_idx.detach().cpu()].to(device)
        return value.to(dtype=dtype) if dtype is not None else value

    ev_y = ev_field("y", torch.long)
    ev_window = ev_field("window", torch.long)
    ev_cfg = ev_field("cfg", torch.long)
    ev_sensor = ev_field("sensor", torch.long)
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
    memory_labels: torch.Tensor,
    t_ev: torch.Tensor,
    t_cand: torch.Tensor,
    *,
    truth_present: bool,
    true_support: int | None,
    other_support: int | None,
    config_mode: str,
    rng: np.random.Generator,
    topk_per_head: int,
    max_evidence: int,
    max_per_window: int,
    max_per_label: int,
    tau: float,
    query_window_mask: torch.Tensor | None = None,
    config_text: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """One patch-level training episode with all leakage exclusions."""
    device = next(dec.parameters()).device
    query = table.gather_queries(
        qi, device, expand_verified_events=True, allowed_window_mask=query_window_mask
    )
    y_window = torch.as_tensor(bank["y"], device=device, dtype=torch.long)
    query_label = y_window[qi]
    allowed = build_allowed_mask(
        bank["patch"], index_rows, query, query_label, memory_labels,
        truth_present=truth_present, true_support=true_support,
        other_support=other_support, config_mode=config_mode, rng=rng,
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
        candidates, t_ev, t_cand, topk_per_head=topk_per_head,
        max_evidence=max_evidence, max_per_window=max_per_window,
        max_per_label=max_per_label, tau=tau, memory_config_text=config_text,
    )
    aux["query_label"] = query_label
    aux["realized_true_support"] = support
    aux["memory_labels"] = memory_labels
    return logits, aux


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, default=_DEFAULT_BANK)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--subspaces", type=int, default=4)
    ap.add_argument("--subspace-dim", type=int, default=64)
    ap.add_argument("--subspace-ema", type=float, default=0.995)
    ap.add_argument("--index-per-label", type=int, default=256,
                    help="config/resolution-stratified EMA-index patches per label; -1 = full bank")
    ap.add_argument("--index-refresh", type=int, default=100)
    ap.add_argument("--topk-per-head", type=int, default=8)
    ap.add_argument("--max-evidence", type=int, default=64)
    ap.add_argument("--max-per-window", type=int, default=4)
    ap.add_argument("--max-per-label", type=int, default=12)
    ap.add_argument("--tau-retr", type=float, default=0.07)
    ap.add_argument("--candidate-fractions", type=float, nargs="+",
                    default=(0.10, 0.25, 0.50, 1.0))
    ap.add_argument("--memory-fractions", type=float, nargs="+", default=(0.25, 0.50, 1.0),
                    help="fraction of training labels represented in an episode's memory")
    ap.add_argument("--config-modes", nargs="+", choices=("any", "same", "cross", "query_absent"),
                    default=("same", "cross", "query_absent"))
    ap.add_argument("--distractor-modes", nargs="+",
                    choices=("random", "language", "motion_family", "physical", "mixed"),
                    default=("random", "language", "motion_family", "physical", "mixed"))
    ap.add_argument(
        "--query-balance",
        choices=("hierarchical", "legacy_sqrt"),
        default="hierarchical",
        help="hierarchical balances selected labels/configs and tempers subjects; legacy_sqrt "
             "uses the historical inverse-sqrt label-frequency draw",
    )
    ap.add_argument("--query-subject-alpha", type=float, default=0.5,
                    help="subject-size exponent inside each label/config query bucket")
    ap.add_argument("--explicit-config-text", action="store_true",
                    help="ablation: re-inject acquisition/config text already conditioned in Phase A")
    ap.add_argument("--val-families", type=int, default=1,
                    help="complete canonical activity families excluded from every training role")
    ap.add_argument("--val-frac-cfg", type=float, default=0.2)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-episodes", type=int, default=16)
    ap.add_argument("--val-queries", type=int, default=32)
    ap.add_argument("--label-variants", type=int, default=16)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true",
                    help="run a two-step CPU integration test on a synthetic schema-v2 bank")
    args = ap.parse_args()
    if args.smoke:
        args.device = "cpu"
        args.steps = 2
        args.batch = 4
        args.index_per_label = 4
        args.index_refresh = 1
        args.topk_per_head = 2
        args.max_evidence = 8
        args.max_per_window = 2
        args.max_per_label = 4
        args.val_every = 1
        args.val_episodes = 2
        args.val_queries = 4
        args.val_frac_cfg = 0.5
        args.warmup_steps = 1
        args.out = Path("/tmp/halo_phase_b_predictor_smoke.pt") if args.out == _DEFAULT_OUT else args.out
    if args.steps < 1 or args.batch < 1 or args.val_every < 1:
        ap.error("steps, batch, and val-every must be positive")
    if any(not 0 < value <= 1 for value in args.candidate_fractions):
        ap.error("candidate fractions must be in (0,1]")
    if any(not 0 < value <= 1 for value in args.memory_fractions):
        ap.error("memory fractions must be in (0,1]")
    if args.index_per_label == 0 or args.index_per_label < -1:
        ap.error("--index-per-label must be positive or -1 for the full bank")
    if not 0 <= args.query_subject_alpha <= 1:
        ap.error("--query-subject-alpha must be in [0,1]")
    if args.warmup_steps < 0 or args.grad_clip <= 0:
        ap.error("warmup-steps must be nonnegative and grad-clip must be positive")

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
    allowed_train_vocab = torch.arange(n_vocab, device=device)
    allowed_train_vocab = allowed_train_vocab[~torch.isin(allowed_train_vocab, heldout_labels)]

    memory_window_mask = torch.zeros(len(Z), dtype=torch.bool, device=device)
    memory_window_mask[train_pool] = True
    val_memory_window_mask = torch.zeros(len(Z), dtype=torch.bool, device=device)
    val_memory_window_mask[base_train_pool] = True
    val_window_mask = torch.zeros(len(Z), dtype=torch.bool, device=device)
    val_window_mask[val_pool] = True
    retriever = PatchSubspaceRetriever(
        d, args.subspaces, args.subspace_dim, args.subspace_ema
    ).to(device)
    dec = EvidenceDecoder(DecoderConfig(
        d_model=d, n_layers=args.layers, n_heads=args.heads,
        candidate_tokens=True, structural_metadata=True,
    )).to(device)
    params = dec.param_groups(args.weight_decay) + [
        {"params": retriever.parameters(), "weight_decay": args.weight_decay},
    ]
    opt = torch.optim.AdamW(params, lr=args.lr)

    def lr_factor(step_index: int) -> float:
        if args.warmup_steps and step_index < args.warmup_steps:
            return float(step_index + 1) / args.warmup_steps
        progress = (step_index - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    index_rows = memory_index = None
    index_rng = np.random.default_rng(args.seed + 2)
    val_index_rows = table.sample_index_rows(
        val_memory_window_mask, args.index_per_label, np.random.default_rng(args.seed + 3)
    )

    def rebuild_index(rows=None):
        if rows is None:
            rows = table.sample_index_rows(
                memory_window_mask, args.index_per_label, index_rng
            )
        memory = F.normalize(
            torch.as_tensor(bank["patch"]["Z"])[rows].float().to(device), dim=-1
        )
        return rows, retriever.build_index(memory)

    def make_episode(
        pool, *, count, local_rng, validation=False, candidate_fraction=None,
        distractor_mode=None,
    ):
        present = torch.unique(y[pool])
        n_seed = min(4, len(present))
        seed = local_rng.choice(present.cpu().numpy(), size=n_seed, replace=False)
        seed_labels = torch.tensor(seed, device=device, dtype=torch.long)
        if args.query_balance == "hierarchical":
            qi = sample_queries(
                pool, seed_labels, y, count, local_rng,
                config_ids=cfg, subject_ids=subj, label_alpha=0.0,
                subject_alpha=args.query_subject_alpha,
            )
        else:
            qi = sample_queries(pool, seed_labels, y, count, local_rng)
        fractions = tuple(args.candidate_fractions) if candidate_fraction is None \
            else (float(candidate_fraction),)
        size = candidate_size(n_vocab, fractions, local_rng)
        mode = distractor_mode or str(
            args.distractor_modes[int(local_rng.integers(len(args.distractor_modes)))]
        )
        candidates = choose_candidates(
            y[qi], size, n_vocab, text, physical, truth_present=True,
            mode=mode, rng=local_rng,
            allowed_vocab=(None if validation else allowed_train_vocab),
            family_ids=family_ids,
        )
        return qi, candidates, mode

    # Fixed validation matrix over strict-zero-shot and low/full-support held-out-family episodes.
    val_specs = []
    val_rng = np.random.default_rng(args.seed + 1)
    for i in range(args.val_episodes):
        support_index, block = i % 4, i // 4
        support = (0, 1, 4, None)[support_index]
        candidate_fraction = args.candidate_fractions[
            (support_index + block) % len(args.candidate_fractions)
        ]
        config_mode = ("cross", "query_absent")[(support_index + block) % 2]
        distractor_mode = args.distractor_modes[
            (support_index + 2 * block) % len(args.distractor_modes)
        ]
        memory_fraction = args.memory_fractions[
            (support_index + 2 * block) % len(args.memory_fractions)
        ]
        episode_seed = args.seed + 1000 + i
        for _attempt in range(50):
            qi, candidates, _ = make_episode(
                val_pool, count=args.val_queries, local_rng=val_rng, validation=True,
                candidate_fraction=candidate_fraction, distractor_mode=distractor_mode,
            )
            memory_labels = choose_memory_labels(
                torch.arange(n_vocab, device=device), y[qi], memory_fraction, val_rng,
                require_truth=support != 0,
            )
            query = table.gather_queries(
                qi, device, expand_verified_events=True,
                allowed_window_mask=val_window_mask,
            )
            allowed = build_allowed_mask(
                bank["patch"], val_index_rows, query, y[qi], memory_labels,
                truth_present=True, true_support=support, other_support=None,
                config_mode=config_mode, rng=np.random.default_rng(episode_seed),
            )
            realized = realized_support_examples(
                bank["patch"], val_index_rows, allowed, y[qi]
            )
            support_ok = bool((realized > 0).all()) if support is None \
                else bool((realized >= support).all())
            memory_ok = bool((allowed.any(-1) | ~query.mask).all())
            if support_ok and memory_ok:
                break
        else:
            raise RuntimeError(
                f"could not construct validation cell support={support}, "
                f"config={config_mode}, candidate_fraction={candidate_fraction}"
            )
        val_specs.append({
            "qi": qi, "candidates": candidates, "memory_labels": memory_labels,
            "true_support": support, "other_support": None,
            "candidate_fraction": float(candidate_fraction),
            "memory_fraction": float(memory_fraction), "config_mode": config_mode,
            "distractor_mode": str(distractor_mode), "seed": episode_seed,
        })

    @torch.no_grad()
    def evaluate():
        dec.eval(); retriever.eval()
        all_pred, all_identity, all_true = [], [], []
        per_cell, supports, true_mass = [], [], []
        _, val_memory_index = rebuild_index(val_index_rows)
        for spec in val_specs:
            qi, candidates = spec["qi"], spec["candidates"]
            logits, aux = run_patch_episode(
                dec, retriever, table, bank, val_index_rows, val_memory_index,
                qi, candidates, spec["memory_labels"], text, text, truth_present=True,
                true_support=spec["true_support"], other_support=spec["other_support"],
                config_mode=spec["config_mode"], rng=np.random.default_rng(spec["seed"]),
                topk_per_head=args.topk_per_head, max_evidence=args.max_evidence,
                max_per_window=args.max_per_window, max_per_label=args.max_per_label,
                tau=args.tau_retr,
                query_window_mask=val_window_mask,
                config_text=config_text,
            )
            true = y[qi]
            pred = candidates[logits.argmax(1)]
            identity = candidates[aux["identity_logits"].argmax(1)]
            cell_ba = balanced_accuracy(pred.cpu().numpy(), true.cpu().numpy())
            per_cell.append((spec["true_support"], cell_ba))
            all_pred.extend(pred.cpu().tolist()); all_identity.extend(identity.cpu().tolist())
            all_true.extend(true.cpu().tolist())
            supports.extend(aux["realized_true_support"].cpu().tolist())
            mass = (
                aux["pool_weights"]
                * aux["evidence_label"].eq(true.unsqueeze(1)).to(aux["pool_weights"].dtype)
            ).sum(1)
            true_mass.extend(mass.cpu().tolist())
        zero = [score for support, score in per_cell if support == 0]
        low = [score for support, score in per_cell if support != 0]
        return {
            "macro_cell_ba": float(np.mean([score for _, score in per_cell])),
            "ba": balanced_accuracy(np.asarray(all_pred), np.asarray(all_true)),
            "identity_ba": balanced_accuracy(np.asarray(all_identity), np.asarray(all_true)),
            "zero_support_ba": float(np.mean(zero)) if zero else float("nan"),
            "positive_support_ba": float(np.mean(low)) if low else float("nan"),
            "mean_realized_true_support": float(np.mean(supports)),
            "mean_retrieved_true_mass": float(np.mean(true_mass)),
        }

    best = {"macro_cell_ba": -float("inf")}
    best_step = 0
    best_state = None
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if index_rows is None or (step - 1) % args.index_refresh == 0:
            index_rows, memory_index = rebuild_index()
        dec.train(); retriever.train()
        episode_error = None
        for _attempt in range(10):
            true_support = sample_support(rng, SUPPORT_CHOICES, TRUE_SUPPORT_PROBS)
            other_support = sample_support(rng, SUPPORT_CHOICES, OTHER_SUPPORT_PROBS)
            memory_fraction = float(
                args.memory_fractions[int(rng.integers(len(args.memory_fractions)))]
            )
            qi, candidates, distractor_mode = make_episode(
                train_pool, count=args.batch, local_rng=rng
            )
            memory_labels = choose_memory_labels(
                allowed_train_vocab, y[qi], memory_fraction, rng,
                require_truth=true_support != 0,
            )
            config_mode = str(args.config_modes[int(rng.integers(len(args.config_modes)))])
            t_ev, t_cand = (
                (text, text) if variants is None else sample_text_tables(variants, text_gen)
            )
            try:
                logits, aux = run_patch_episode(
                    dec, retriever, table, bank, index_rows, memory_index,
                    qi, candidates, memory_labels, t_ev, t_cand, truth_present=True,
                    true_support=true_support, other_support=other_support,
                    config_mode=config_mode, rng=rng, topk_per_head=args.topk_per_head,
                    max_evidence=args.max_evidence, max_per_window=args.max_per_window,
                    max_per_label=args.max_per_label, tau=args.tau_retr,
                    query_window_mask=memory_window_mask,
                    config_text=config_text,
                )
                break
            except ValueError as exc:
                if not any(token in str(exc) for token in (
                    "no eligible", "no evidence", "requested true support",
                )):
                    raise
                episode_error = exc
        else:
            raise RuntimeError(
                "could not draw a feasible support/config episode in 10 attempts"
            ) from episode_error
        target = label_index(candidates, n_vocab, device)[y[qi]]
        if bool((target < 0).any()):
            raise RuntimeError("answerable episode omitted a query label from candidates")
        loss = F.cross_entropy(logits, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite Phase-B loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        dec_grad = torch.sqrt(sum(
            parameter.grad.float().square().sum()
            for parameter in dec.parameters() if parameter.grad is not None
        ))
        retrieval_grad = torch.sqrt(sum(
            parameter.grad.float().square().sum()
            for parameter in retriever.parameters() if parameter.grad is not None
        ))
        preclip_grad = torch.nn.utils.clip_grad_norm_(
            list(dec.parameters()) + list(retriever.parameters()), args.grad_clip,
        )
        opt.step(); sched.step(); retriever.update_ema()

        if step == 1 or step % args.val_every == 0:
            metrics = evaluate()
            better = metrics["macro_cell_ba"] > best["macro_cell_ba"]
            if better:
                best, best_step = dict(metrics), step
                best_state = {
                    "decoder": {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()},
                    "retriever": {
                        k: v.detach().cpu().clone() for k, v in retriever.state_dict().items()
                    },
                }
            mask = aux["evidence_mask"]
            usage = torch.zeros(args.subspaces, device=device)
            usage.scatter_add_(
                0, aux["evidence_head"][mask], aux["pool_weights"][mask]
            )
            usage = usage / usage.sum().clamp_min(1e-8)
            true_mass = (
                aux["pool_weights"]
                * aux["evidence_label"].eq(y[qi].unsqueeze(1)).to(aux["pool_weights"].dtype)
            ).sum(1).mean()
            print(json.dumps({
                "step": step, "loss": round(float(loss.detach()), 5),
                "candidate_count": len(candidates), "memory_label_count": len(memory_labels),
                "true_support": "all" if true_support is None else true_support,
                "other_support": "all" if other_support is None else other_support,
                "realized_true_support_mean": round(
                    float(aux["realized_true_support"].float().mean()), 3
                ),
                "retrieved_true_mass": round(float(true_mass.detach()), 4),
                "head_usage": [round(float(value.detach()), 4) for value in usage],
                "decoder_grad_norm": round(float(dec_grad), 4),
                "retriever_grad_norm": round(float(retrieval_grad), 4),
                "preclip_grad_norm": round(float(preclip_grad), 4),
                "delta_norm": round(float(aux["delta_norm"]), 4),
                "candidate_delta_norm": round(float(aux["candidate_delta_norm"]), 4),
                "lr": opt.param_groups[0]["lr"],
                "memory_fraction": memory_fraction,
                "config_mode": config_mode, "distractor_mode": distractor_mode,
                **{key: round(value, 4) for key, value in metrics.items()},
                "best_macro_cell_ba": round(best["macro_cell_ba"], 4),
                "elapsed_s": round(time.time() - t0, 1),
            }), flush=True)

    if best_state is None:
        raise RuntimeError("training completed without a valid checkpoint")
    payload = {
        **best_state,
        "cfg": {
            "d_model": d, "n_layers": args.layers, "n_heads": args.heads,
            "candidate_tokens": True, "structural_metadata": True,
            "explicit_config_text": bool(args.explicit_config_text),
            "n_subspaces": args.subspaces, "subspace_dim": args.subspace_dim,
            "subspace_ema": args.subspace_ema,
        },
        "episode_cfg": {
            "candidate_fractions": list(args.candidate_fractions),
            "memory_fractions": list(args.memory_fractions),
            "support_choices": list(SUPPORT_CHOICES),
            "true_support_probabilities": list(TRUE_SUPPORT_PROBS),
            "other_support_probabilities": list(OTHER_SUPPORT_PROBS),
            "config_modes": list(args.config_modes),
            "distractor_modes": list(args.distractor_modes),
            "query_balance": args.query_balance,
            "query_subject_alpha": args.query_subject_alpha,
        },
        "retrieval_cfg": {
            "index_per_label": args.index_per_label,
            "index_seed": args.seed + 3,
            "topk_per_head": args.topk_per_head,
            "max_evidence": args.max_evidence,
            "max_per_window": args.max_per_window,
            "max_per_label": args.max_per_label,
            "tau_retr": args.tau_retr,
        },
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
        },
        "objective": "candidate_cross_entropy",
        "best_step": best_step, "best_metrics": best,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"[patch-dec] best step {best_step}: {best} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
