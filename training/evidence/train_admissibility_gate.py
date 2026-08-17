"""Stage-2 refinement of the warm-started admissibility gate.

Only the low-rank gate is learned. Phase-A features, frozen text embeddings, memory contents,
compatibility rules, cosine ranking, voting, and deployment top-k remain fixed. Training uses a
fully soft vote over a bounded label-balanced working set so every compatible evidence row receives
credit; validation uses top-k on the same continuous admissibility-adjusted score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.evidence.admissibility_gate import AdmissibilityGate, observations_from_table
from training.evidence.admissibility_text import (
    embed_text_views, observation_text_views,
)
from training.evidence.admissible_retrieval import (
    admissibility_from_unique, compatibility_mask, merge_sensors, rank_scores, soft_vote_all, vote,
)
from training.evidence.bank_guard import (
    assert_artifact_matches_bank, assert_bank_current, assert_sensor_bank,
)
from training.evidence.gate_extrapolation import _embed
from training.evidence.gate_predictor import (
    ADMISSIBILITY_REFINEMENT_REGIME, BankRows, SensorBankStore, concatenate_bank_rows, load_gate,
    save_gate, subset_bank_rows,
)
from training.evidence.labeltext import build_label_variants

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_BANK = _REPO / "training/evidence/outputs/memory_bank.pt"
DEFAULT_GATE = _REPO / "training/evidence/outputs/admissibility_gate.pt"
DEFAULT_OUT = _REPO / "training/evidence/outputs/admissibility_stage2"


@dataclass(frozen=True)
class ArchiveIndex:
    """Window-level label index plus O(total rows) precomputed sensor-row lookup."""

    windows_by_label: dict[int, np.ndarray]
    sensor_order: torch.Tensor
    sensor_offsets: torch.Tensor
    execution_by_window: np.ndarray

    @classmethod
    def from_bank(cls, bank: dict) -> "ArchiveIndex":
        labels = torch.as_tensor(bank["y"], dtype=torch.long)
        windows_by_label = {
            int(label): torch.where(labels.eq(label))[0].cpu().numpy()
            for label in torch.unique(labels).tolist() if int(label) >= 0
        }
        sensor_window = torch.as_tensor(bank["sensor"]["window"], dtype=torch.long)
        order = torch.argsort(sensor_window, stable=True)
        counts = torch.bincount(sensor_window, minlength=len(labels))
        offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        subject = torch.as_tensor(bank["subj"], dtype=torch.long).numpy()
        event = torch.as_tensor(bank["event"], dtype=torch.long).numpy()
        n_events = int(event.max()) + 1
        # Subject ids are globally keyed as ``dataset:subject`` by build_memory. Event strings are
        # not guaranteed to be globally unique within a dataset, so event alone (or dataset,event)
        # can collapse executions from different participants into one sampling unit.
        execution = subject.astype(np.int64) * n_events + event.astype(np.int64)
        return cls(windows_by_label, order, offsets, execution)

    def sensor_rows(self, windows: list[int] | np.ndarray) -> torch.Tensor:
        pieces = [self.sensor_order[self.sensor_offsets[int(window)]:
                                    self.sensor_offsets[int(window) + 1]]
                  for window in windows]
        return torch.cat(pieces) if pieces else torch.empty(
            0, dtype=torch.long, device=self.sensor_order.device,
        )

    def row_lookup_to(self, device: torch.device | str) -> "ArchiveIndex":
        """Move only the row-order lookup; label/execution sampling remains CPU-side."""
        return ArchiveIndex(
            self.windows_by_label,
            self.sensor_order.to(device),
            self.sensor_offsets,
            self.execution_by_window,
        )

    def sample_executions(
        self, windows: np.ndarray, count: int, rng: np.random.Generator,
    ) -> list[int]:
        """Sample distinct recorded executions, never adjacent windows from one execution."""
        shuffled = rng.permutation(windows)
        selected, seen = [], set()
        for window in shuffled.tolist():
            execution = int(self.execution_by_window[int(window)])
            if execution in seen:
                continue
            selected.append(int(window))
            seen.add(execution)
            if len(selected) == count:
                return selected
        raise ValueError("label has too few distinct executions for the requested episode")


@dataclass(frozen=True)
class Episode:
    candidates: list[int]
    queries: list[int]
    support_windows: list[int]
    support_positions: list[int]
    support_k: int
    enrolled_count: int


def _active_windows(index: ArchiveIndex, per_label: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in sorted(index.windows_by_label):
        windows = index.windows_by_label[label]
        count = min(per_label, len(windows))
        selected.extend(rng.choice(windows, count, replace=False).tolist())
    return selected


def _label_split(labels: list[int], validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    values = np.asarray(labels, dtype=np.int64)
    rng.shuffle(values)
    n_val = max(2, int(round(len(values) * validation_fraction)))
    n_val = min(n_val, len(values) - 2)
    return sorted(values[n_val:].tolist()), sorted(values[:n_val].tolist())


def _draw_episode(
    pool: list[int],
    index: ArchiveIndex,
    candidate_count: int,
    max_support: int,
    rng: np.random.Generator,
    *,
    fixed_support: int | None = None,
    full_enrollment: bool = False,
    partial_enrollment: bool = False,
) -> Episode:
    count = min(candidate_count, len(pool))
    if count < 2:
        raise ValueError("an episode needs at least two candidate labels")
    candidates = rng.choice(pool, count, replace=False).tolist()
    support_k = int(rng.integers(0, max_support + 1) if fixed_support is None else fixed_support)
    if full_enrollment and partial_enrollment:
        raise ValueError("an episode cannot request full and partial enrollment together")
    if support_k == 0:
        enrolled_count = 0
    elif full_enrollment:
        enrolled_count = count
    elif partial_enrollment:
        enrolled_count = int(rng.integers(1, count))
    else:
        enrolled_count = int(rng.integers(1, count + 1))
    enrolled = set(
        rng.choice(count, enrolled_count, replace=False).tolist()
        if enrolled_count else []
    )
    queries, support_windows, support_positions = [], [], []
    for position, label in enumerate(candidates):
        windows = index.windows_by_label[label]
        needed = 1 + (support_k if position in enrolled else 0)
        selected = index.sample_executions(windows, needed, rng)
        queries.append(int(selected[0]))
        for window in selected[1:]:
            support_windows.append(int(window))
            support_positions.append(position)
    return Episode(candidates, queries, support_windows, support_positions,
                   support_k, enrolled_count)


def _episode_memory(
    store: SensorBankStore,
    index: ArchiveIndex,
    active: BankRows,
    episode: Episode,
) -> BankRows:
    candidate_ids = torch.tensor(episode.candidates, device=active.rows.feature.device)
    corpus = subset_bank_rows(active, ~torch.isin(active.rows.label, candidate_ids))
    if not episode.support_windows:
        return corpus
    row_index = index.sensor_rows(episode.support_windows)
    positions = []
    for window, position in zip(episode.support_windows, episode.support_positions):
        positions.extend([position] * int(index.sensor_offsets[window + 1] -
                                          index.sensor_offsets[window]))
    support = store.select(row_index, enrolled_candidate=torch.tensor(positions))
    return concatenate_bank_rows(corpus, support)


def _query_batch(
    store: SensorBankStore, index: ArchiveIndex, windows: list[int],
):
    """Gather all query sensors once and retain their source-window ownership."""
    counts = torch.tensor([
        int(index.sensor_offsets[window + 1] - index.sensor_offsets[window])
        for window in windows
    ], dtype=torch.long, device=store.feature.device)
    owner = torch.repeat_interleave(
        torch.arange(len(windows), device=store.feature.device), counts,
    )
    return store.select(index.sensor_rows(windows)).rows, owner


def _merge_query_windows(
    per_sensor: torch.Tensor, owner: torch.Tensor, window_count: int,
) -> torch.Tensor:
    merged = per_sensor.new_zeros((window_count, per_sensor.shape[-1]))
    return merged.index_add(0, owner, per_sensor)


def _candidate_text(
    variants: torch.Tensor,
    candidates: list[int],
    rng: np.random.Generator,
    *,
    canonical: bool,
) -> torch.Tensor:
    label = torch.tensor(candidates, device=variants.device)
    if canonical:
        view = torch.zeros(len(candidates), dtype=torch.long, device=variants.device)
    else:
        view = torch.from_numpy(
            rng.integers(0, variants.shape[1], size=len(candidates), dtype=np.int64)
        ).to(variants.device)
    return variants[label, view]


def _admissibility_from_row_values(
    gate: AdmissibilityGate,
    row_admissibility: torch.Tensor,
    query_descriptor: torch.Tensor,
    candidate_text: torch.Tensor,
) -> torch.Tensor:
    """Combine cached evidence-side values with the current query, exactly as the full helper does."""
    query_admissibility = gate(query_descriptor, candidate_text)
    return torch.exp(0.5 * (
        query_admissibility.unsqueeze(1).clamp_min(1e-8).log()
        + row_admissibility.unsqueeze(0).clamp_min(1e-8).log()
    ))


def _predict_window(
    gate: AdmissibilityGate,
    memory: BankRows,
    query,
    candidate_text: torch.Tensor,
    label_text: torch.Tensor,
    *,
    soft: bool,
    top_k: int,
    gate_enabled: bool = True,
    return_stats: bool = False,
    row_admissibility: torch.Tensor | None = None,
    query_owner: torch.Tensor | None = None,
    query_count: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rows = memory.rows
    if row_admissibility is None:
        adm = admissibility_from_unique(
            gate, memory.unique_descriptor, memory.descriptor_id, candidate_text,
            query.descriptor,
        )
    else:
        adm = _admissibility_from_row_values(
            gate, row_admissibility, query.descriptor, candidate_text,
        )
    if not gate_enabled:
        adm = torch.ones_like(adm)
    scores = rank_scores(
        query.feature, query.bias, rows,
        compatibility_mask(query.modality, query.gravity, rows),
    )
    if soft:
        soft_result = soft_vote_all(
            scores, rows, candidate_text, label_text, adm, return_stats=return_stats,
        )
        if return_stats:
            per_sensor, stats = soft_result
        else:
            per_sensor = soft_result
    else:
        per_sensor = vote(
            scores, rows, candidate_text, label_text, adm,
            top_k=top_k,
        )
    result = (
        merge_sensors(per_sensor)
        if query_owner is None
        else _merge_query_windows(per_sensor, query_owner, int(query_count))
    )
    return (result, stats) if soft and return_stats else result


def _macro_f1(truth: list[int], prediction: list[int]) -> float:
    scores = []
    for label in sorted(set(truth)):
        tp = sum(t == label and p == label for t, p in zip(truth, prediction))
        fp = sum(t != label and p == label for t, p in zip(truth, prediction))
        fn = sum(t == label and p != label for t, p in zip(truth, prediction))
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def evaluate(
    gate: AdmissibilityGate,
    store: SensorBankStore,
    index: ArchiveIndex,
    active: BankRows,
    pool: list[int],
    variants: torch.Tensor,
    label_text: torch.Tensor,
    *,
    candidate_counts: list[int],
    support_curve: list[int],
    episodes_per_k: int,
    max_support: int,
    top_k: int,
    seed: int,
) -> dict:
    gate.eval()
    results = {}
    for candidate_count in candidate_counts:
        for support_k in support_curve:
            enrollment_modes = ("none",) if support_k == 0 else ("partial", "full")
            for enrollment_mode in enrollment_modes:
                truth, learned, soft_learned, disabled = [], [], [], []
                mode_seed = {"none": 0, "partial": 1, "full": 2}[enrollment_mode]
                rng = np.random.default_rng(
                    seed + 1009 * support_k + 65_537 * candidate_count + mode_seed
                )
                for _ in range(episodes_per_k):
                    episode = _draw_episode(
                        pool, index, candidate_count, max_support, rng,
                        fixed_support=support_k,
                        full_enrollment=enrollment_mode == "full",
                        partial_enrollment=enrollment_mode == "partial",
                    )
                    memory = _episode_memory(store, index, active, episode)
                    text = _candidate_text(variants, episode.candidates, rng, canonical=True)
                    row_admissibility = gate(
                        memory.unique_descriptor, text
                    ).index_select(0, memory.descriptor_id)
                    query, owner = _query_batch(store, index, episode.queries)
                    adm = _admissibility_from_row_values(
                        gate, row_admissibility, query.descriptor, text,
                    )
                    scores = rank_scores(
                        query.feature, query.bias, memory.rows,
                        compatibility_mask(query.modality, query.gravity, memory.rows),
                    )
                    learned_vote = _merge_query_windows(vote(
                        scores, memory.rows, text, label_text, adm, top_k=top_k,
                    ), owner, len(episode.queries))
                    soft_vote = _merge_query_windows(soft_vote_all(
                        scores, memory.rows, text, label_text, adm,
                    ), owner, len(episode.queries))
                    disabled_vote = _merge_query_windows(vote(
                        scores, memory.rows, text, label_text, torch.ones_like(adm),
                        top_k=top_k,
                    ), owner, len(episode.queries))
                    truth.extend(episode.candidates)
                    learned.extend(
                        episode.candidates[int(value)] for value in learned_vote.argmax(1)
                    )
                    soft_learned.extend(
                        episode.candidates[int(value)] for value in soft_vote.argmax(1)
                    )
                    disabled.extend(
                        episode.candidates[int(value)] for value in disabled_vote.argmax(1)
                    )
                learned_f1 = _macro_f1(truth, learned)
                soft_f1 = _macro_f1(truth, soft_learned)
                disabled_f1 = _macro_f1(truth, disabled)
                results[f"c={candidate_count}/k={support_k}/enrollment={enrollment_mode}"] = {
                    "candidate_count": candidate_count, "support_k": support_k,
                    "enrollment": enrollment_mode,
                    "f1_macro": learned_f1,
                    "soft_training_rule_f1_macro": soft_f1,
                    "soft_minus_topk_f1_macro": soft_f1 - learned_f1,
                    "gate_disabled_f1_macro": disabled_f1,
                    "delta_f1_macro": learned_f1 - disabled_f1,
                    "queries": len(truth),
                }
    gate.train()
    return results


def _save_refined(
    gate: AdmissibilityGate,
    path: Path,
    bank: dict,
    base_provenance: dict,
    training: dict,
) -> None:
    provenance = dict(base_provenance)
    provenance["stage2"] = training
    provenance["training_concepts"] = training["training_concepts"]
    save_gate(
        gate, path, provenance, bank=bank,
        training_regime=ADMISSIBILITY_REFINEMENT_REGIME,
    )


def _configuration(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--episodes-per-step", type=int, default=4)
    parser.add_argument("--candidate-counts", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--max-support", type=int, default=4)
    parser.add_argument("--active-windows-per-label", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--replay-weight", type=float, default=0.2)
    parser.add_argument("--replay-batch", type=int, default=128)
    parser.add_argument("--text-views", type=int, default=4)
    parser.add_argument("--label-variants", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    if args.steps < 1 or args.episodes_per_step < 1 or min(args.candidate_counts) < 2:
        parser.error("steps and episodes must be positive; candidate counts must be at least two")
    if args.max_support < 0:
        parser.error("max-support cannot be negative")
    if args.active_windows_per_label < 1 or args.replay_batch < 1:
        parser.error("active-windows-per-label and replay-batch must be positive")
    if args.text_views < 1 or args.label_variants < 1:
        parser.error("text-views and label-variants must be positive")
    if args.eval_every < 1 or args.eval_episodes < 1:
        parser.error("evaluation cadence and episode count must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or args.replay_weight < 0:
        parser.error("lr must be positive; weight-decay and replay-weight cannot be negative")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must be between zero and one")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    bank = torch.load(args.bank, map_location="cpu", weights_only=True, mmap=True)
    assert_bank_current(bank, context="train_admissibility_gate")
    assert_sensor_bank(bank, context="train_admissibility_gate")
    artifact = torch.load(args.gate, map_location="cpu", weights_only=True)
    assert_artifact_matches_bank(
        artifact, bank, context="train_admissibility_gate", artifact_name="admissibility gate",
    )
    gate = load_gate(args.gate).to(device).train()
    base_provenance = dict(artifact.get("provenance") or {})

    index = ArchiveIndex.from_bank(bank).row_lookup_to(device)
    store = SensorBankStore.from_bank(bank, device)
    eligible = sorted(
        label for label, windows in index.windows_by_label.items()
        if len(np.unique(index.execution_by_window[windows])) >= args.max_support + 1
    )
    train_labels, val_labels = _label_split(eligible, args.validation_fraction, args.seed)
    if min(len(train_labels), len(val_labels)) < 2:
        raise SystemExit("too few represented labels for a train/validation concept split")

    active_windows = _active_windows(index, args.active_windows_per_label, args.seed + 1)
    active = store.select(index.sensor_rows(active_windows))
    from eval.scoring import get_sbert_encoder
    label_variants = build_label_variants(
        list(bank["vocab"]), get_sbert_encoder(), K=args.label_variants,
        seed=args.seed, use_descriptions=False, train_only=True,
    ).to(device)
    label_text = F.normalize(torch.as_tensor(bank["label_text"]).float(), dim=-1).to(device)

    from training.evidence.resolvability import load as load_resolvability
    table = load_resolvability()
    table_hash = hashlib.sha256(
        json.dumps(table, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if table_hash != base_provenance.get("resolvability_table_sha256"):
        raise SystemExit("warm-start gate and current resolvability table differ; refit Stage 1")
    observations = observations_from_table(table)
    text_views = observation_text_views(observations, args.text_views, args.seed)
    replay_sensor, replay_concept = embed_text_views(text_views, _embed)
    replay_sensor = replay_sensor.to(device)
    replay_concept = replay_concept.to(device)
    replay_target = torch.from_numpy(observations.value).float().to(device)
    training_concept_names = {bank["vocab"][label] for label in train_labels}
    replay_cells = torch.tensor([
        index for index, concept in enumerate(observations.concept)
        if concept in training_concept_names
    ], dtype=torch.long, device=device)
    if len(replay_cells) < 20:
        raise SystemExit("too few train-concept resolvability cells for Stage-2 replay")

    # These are legitimate Stage-2 training supports. Register them before training so abstention
    # does not suppress gradients for wording/configurations the optimizer is explicitly shown.
    gate.extend_known_support(
        active.unique_descriptor,
        label_variants[torch.tensor(train_labels, device=device)].reshape(-1, label_variants.shape[-1]),
    )
    initial = {name: value.detach().clone() for name, value in gate.named_parameters()}
    optimizer = torch.optim.AdamW(
        [parameter for parameter in gate.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    warmup = min(100, max(1, args.steps // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup)
        * 0.5 * (1.0 + np.cos(np.pi * max(0, step - warmup) /
                              max(1, args.steps - warmup))),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    telemetry_path = args.out / "telemetry.jsonl"
    telemetry_path.write_text("")
    support_curve = [value for value in (0, 1, 2, 4) if value <= args.max_support]
    best_metric = float("-inf")
    best_step = 0
    rng = np.random.default_rng(args.seed + 2)
    started = time.perf_counter()

    def run_validation(step: int) -> dict:
        nonlocal best_metric, best_step
        curves = evaluate(
            gate, store, index, active, val_labels, label_variants, label_text,
            candidate_counts=[count for count in args.candidate_counts if count <= len(val_labels)],
            support_curve=support_curve,
            episodes_per_k=args.eval_episodes, max_support=args.max_support,
            top_k=args.top_k, seed=args.seed + 10_000,
        )
        metric = float(np.mean([cell["f1_macro"] for cell in curves.values()]))
        if metric > best_metric:
            best_metric, best_step = metric, step
            _save_refined(gate, args.out / "best.pt", bank, base_provenance, {
                "step": step, "validation_mean_f1_macro": metric,
                "validation_curve": curves,
                "training_concepts": [bank["vocab"][label] for label in train_labels],
                "heldout_concepts": [bank["vocab"][label] for label in val_labels],
                "configuration": _configuration(args),
            })
        return {"validation_mean_f1_macro": metric, "validation_curve": curves}

    initial_validation = run_validation(0)
    print(json.dumps({"step": 0, **initial_validation}), flush=True)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        ce_losses = []
        support_k, enrolled = [], []
        last_admissibility = None
        retrieval_stats: list[dict[str, torch.Tensor]] = []
        collect_stats = step == 1 or step % 20 == 0
        for _ in range(args.episodes_per_step):
            episode = _draw_episode(
                train_labels, index,
                int(rng.choice([count for count in args.candidate_counts
                                if count <= len(train_labels)])),
                args.max_support, rng,
            )
            support_k.append(episode.support_k)
            enrolled.append(episode.enrolled_count)
            memory = _episode_memory(store, index, active, episode)
            candidate_text = _candidate_text(
                label_variants, episode.candidates, rng, canonical=False,
            )
            unique_admissibility = gate(memory.unique_descriptor, candidate_text)
            row_admissibility = unique_admissibility.index_select(0, memory.descriptor_id)
            query, owner = _query_batch(store, index, episode.queries)
            prediction = _predict_window(
                gate, memory, query, candidate_text, label_text,
                soft=True, top_k=args.top_k, return_stats=collect_stats,
                row_admissibility=row_admissibility,
                query_owner=owner, query_count=len(episode.queries),
            )
            if collect_stats:
                votes, stats = prediction
                retrieval_stats.append(stats)
            else:
                votes = prediction
            episode_loss = F.cross_entropy(
                (votes + 1e-8).log(),
                torch.arange(len(episode.candidates), device=device),
            )
            (episode_loss / args.episodes_per_step).backward()
            ce_losses.append(episode_loss.detach())
            last_admissibility = unique_admissibility.detach()

        ce_loss = torch.stack(ce_losses).mean()
        physical = replay_cells[torch.randint(
            len(replay_cells), (args.replay_batch,), device=device
        )]
        view = torch.randint(args.text_views, (args.replay_batch,), device=device)
        replay_prediction = gate.paired(
            replay_sensor[physical, view], replay_concept[physical, view]
        )
        replay_loss = F.mse_loss(replay_prediction, replay_target[physical])
        (args.replay_weight * replay_loss).backward()
        loss = ce_loss + args.replay_weight * replay_loss.detach()
        module_gradients = {
            f"gradient/{name}": (
                float(parameter.grad.detach().norm()) if parameter.grad is not None else 0.0
            )
            for name, parameter in gate.named_parameters()
        }
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0))
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 20 == 0:
            drift = float(torch.sqrt(sum(
                (parameter.detach() - initial[name]).square().sum()
                for name, parameter in gate.named_parameters()
            )))
            row = {
                "step": step, "loss": float(loss.detach()),
                "candidate_ce": float(ce_loss.detach()),
                "resolvability_replay_mse": float(replay_loss.detach()),
                "gradient_norm": gradient_norm, "parameter_drift_l2": drift,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "mean_support_k": float(np.mean(support_k)),
                "mean_enrolled_candidates": float(np.mean(enrolled)),
                "elapsed_seconds": time.perf_counter() - started,
                **AdmissibilityGate.spread(last_admissibility),
                **module_gradients,
            }
            if retrieval_stats:
                row.update({
                    key: float(torch.stack([values[key] for values in retrieval_stats]).mean())
                    for key in retrieval_stats[0]
                })
            with telemetry_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            validation = run_validation(step)
            with telemetry_path.open("a") as handle:
                handle.write(json.dumps({"step": step, **validation}) + "\n")

    training = {
        "step": args.steps, "best_step": best_step,
        "best_validation_mean_f1_macro": best_metric,
        "training_concepts": [bank["vocab"][label] for label in train_labels],
        "heldout_concepts": [bank["vocab"][label] for label in val_labels],
        "configuration": _configuration(args),
    }
    _save_refined(gate, args.out / "last.pt", bank, base_provenance, training)
    summary = {
        **training, "initial_validation": initial_validation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
