"""Train the compact HALO evidence engine on independent adaptation episodes.

Each episode has its own candidate set, query executions, enrolled support, and fixed-size
stratified memory bank.  Several episodes share one encoder forward for throughput, but never share
candidate identities or attention.  The exact compact deployment path is optimized end to end:

    temporal sensor encoder -> learned pair scorer -> hard top-k -> evidence mixer -> text vote

The default input is clean.  Signal augmentation is an explicit experiment, not hidden in the
reference recipe.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.scripts.augmentations import AugmentationConfig
from model.blocks import AttentionSpec
from model.evidence.engine import EngineConfig, EvidenceEngine
from model.evidence.evidence_mixer import EvidenceMixerConfig
from model.evidence.retrieval_scorer import PairScorerConfig
from model.tokenizer.encoder import SetTokenizerEncoder
from training.evidence.episode_labels import encode_neutral_aliases, episode_label_set
from training.tokenizer.episodic import (
    BankSpec,
    EpisodePlan,
    EpisodeSpec,
    EpisodicCollate,
    GroupedEpisodicBatchSampler,
    bank_index,
    build_episode_plans,
    build_shared_stream_plans,
    eligible_labels,
    episode_binding,
    episode_row_roles,
    label_window_table,
    live_sensor_rows,
    macro_f1,
    matched_support_variants,
    retrieval_alignment_loss,
    provenance_lift,
    sample_bank_positions,
    stream_label_table,
)
from training.tokenizer.eval_transfer import build_encoder
from training.tokenizer.pretrain import (
    capture_source_provenance,
    corpus_fingerprint,
    representation_health,
)
from training.tokenizer.pretrain_data import (
    DFT_SIZE,
    PATCH_SECONDS,
    TRAIN_DATASETS,
    CorpusIndex,
    MultiScaleCollate,
    PretrainDataset,
    _seed_worker,
)

SEED = 20260818
#: The validation draw is deliberately NOT tied to the training seed. Measured 2026-08-20: four
#: replicates of one configuration differing only in seed scored 0.4499 / 0.5881 / 0.4545 / 0.4519,
#: between-run sd 0.068 -- and the step-0 baseline alone had sd 0.073, so essentially all of it was
#: WHICH CONCEPTS the seed happened to hold out (16 to 23 of them across those runs). Pinning the
#: draw makes two runs comparable; leaving it tied to --seed made every arm comparison read noise.
VAL_SEED = 20260901


def subject_ids_for(index: CorpusIndex, keys) -> np.ndarray:
    """Global subject id per key position."""
    seen: dict[tuple[str, str], int] = {}
    values = []
    for key in keys:
        ref = index.refs[key.stream_i]
        identity = (ref.dataset, str(ref.subjects[key.window_i]))
        values.append(seen.setdefault(identity, len(seen)))
    return np.asarray(values, dtype=np.int64)


def stream_ids_for(index: CorpusIndex, keys) -> tuple[np.ndarray, dict[str, int]]:
    """Acquisition stream id per key position."""
    names: dict[str, int] = {}
    values = [names.setdefault(index.refs[key.stream_i].key, len(names)) for key in keys]
    return np.asarray(values, dtype=np.int64), names


def execution_ids_for(index: CorpusIndex, keys) -> np.ndarray:
    """Physical execution id; synchronous streams from one event share an id."""
    names: dict[tuple[str, str, str], int] = {}
    values = []
    for key in keys:
        ref = index.refs[key.stream_i]
        identity = (
            ref.dataset,
            str(ref.subjects[key.window_i]),
            str(ref.event_ids[key.window_i]),
        )
        values.append(names.setdefault(identity, len(names)))
    return np.asarray(values, dtype=np.int64)


def label_text_matrix(encoder: SetTokenizerEncoder, label_ids: dict, device) -> torch.Tensor:
    """Frozen MiniLM embedding for every canonical corpus label."""
    from eval.scoring import get_sbert_encoder

    id_to_label = {value: key for key, value in label_ids.items()}
    labels = [id_to_label[i] for i in range(len(id_to_label))]
    return torch.from_numpy(get_sbert_encoder()(labels)).to(device)


def split_label_pool(pool: list[int], fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic, disjoint objective-train and validation concept pools."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("holdout fraction must be in (0,1)")
    values = np.asarray(sorted(pool), dtype=np.int64)
    if len(values) < 4:
        raise ValueError("at least four eligible labels are needed for a concept holdout")
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    n_val = min(max(2, int(round(len(values) * fraction))), len(values) - 2)
    return sorted(values[n_val:].tolist()), sorted(values[:n_val].tolist())


def replace_counts(spec: EpisodeSpec, pool_size: int) -> EpisodeSpec:
    """Drop candidate counts that cannot fit while preserving every other episode field."""
    ceiling = pool_size - (1 if spec.background_windows else 0)
    counts = tuple(count for count in spec.candidate_counts if count <= ceiling)
    if not counts:
        raise ValueError(
            f"no candidate count {spec.candidate_counts} fits pool size {pool_size} "
            f"with ceiling {ceiling}"
        )
    return dataclasses.replace(spec, candidate_counts=counts)


def encode_batch(encoder: SetTokenizerEncoder, batch: dict, device: torch.device) -> dict:
    """Encode every window from several independent episodes in one heterogeneous forward."""
    if encoder.trunk != "temporal" or encoder.token_granularity != "sensor":
        raise ValueError("compact Phase B requires a temporal, sensor-granularity encoder")
    return encoder(
        batch["patches"].to(device, non_blocking=True),
        batch["rates"].to(device, non_blocking=True),
        batch["patch_len"].to(device, non_blocking=True),
        batch["role_texts"],
        batch["positions"].to(device, non_blocking=True),
        patch_durations=batch["patch_durations"].to(device, non_blocking=True),
        channel_mask=batch["channel_mask"].to(device, non_blocking=True),
        patch_padding_mask=batch["patch_padding_mask"].to(device, non_blocking=True),
        sensor_texts=batch["sensor_texts"],
        sensor_id=batch["sensor_id"].to(device, non_blocking=True),
        source_rate_hz=batch["source_rates"].to(device, non_blocking=True),
        return_retrieval_tokens=True,
    )


def _offset_plan(plan: EpisodePlan, offset: int) -> EpisodePlan:
    """Move query/support positions into a combined train+validation dataset."""
    return dataclasses.replace(
        plan,
        query_positions=tuple(value + offset for value in plan.query_positions),
        support_positions=tuple(value + offset for value in plan.support_positions),
    )


def _attach_training_banks(
    plans: list[EpisodePlan], bank, bank_spec: BankSpec, seed: int,
) -> list[EpisodePlan]:
    rng = np.random.default_rng(seed)
    out = []
    for plan in plans:
        n_background = bank_spec.n_windows - plan.n_support
        positions = sample_bank_positions(
            bank, spec=bank_spec, n_windows=n_background,
            exclude_labels=plan.candidates, rng=rng,
        )
        out.append(dataclasses.replace(plan, background_positions=tuple(positions)))
    return out


def _matched_validation_plans(
    base_plans: list[EpisodePlan], support_grid: tuple[int, ...], bank,
    bank_spec: BankSpec, train_offset: int, seed: int,
) -> list[EpisodePlan]:
    """Build matched k-curves: same query, candidates, and nested background bank at every k."""
    rng = np.random.default_rng(seed)
    out = []
    for raw_base in base_plans:
        base = _offset_plan(raw_base, train_offset)
        full_background = sample_bank_positions(
            bank, spec=bank_spec, n_windows=bank_spec.n_windows,
            exclude_labels=base.candidates, rng=rng,
        )
        for variant in matched_support_variants(base, support_grid):
            keep = bank_spec.n_windows - variant.n_support
            out.append(dataclasses.replace(
                variant, background_positions=tuple(full_background[:keep]),
            ))
    return out


def _make_plans(
    table, stream_table, pool, spec, streams, executions, *, n_episodes, seed, schedule=None,
) -> list[EpisodePlan]:
    if spec.shared_query_stream:
        return build_shared_stream_plans(
            table, stream_table, pool, n_episodes=n_episodes, spec=spec, seed=seed,
            stream_ids=streams, support_schedule=schedule, execution_ids=executions,
        )
    return build_episode_plans(
        table, pool, n_episodes=n_episodes, spec=spec, seed=seed,
        support_schedule=schedule, stream_ids=streams, execution_ids=executions,
    )


def _grad_norm(parameters) -> float:
    gradients = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(gradients))) if gradients else 0.0


def run_episode(
    engine: EvidenceEngine,
    batch: dict,
    plan: EpisodePlan,
    label_text: torch.Tensor,
    alias_text: torch.Tensor,
    device: torch.device,
    *,
    encoded_out: dict | None = None,
    row_offset: int = 0,
    seed: int = 0,
    collect_stats: bool = False,
    aux_weight: float = 0.0,
    aux_temperature: float = 0.1,
) -> dict:
    """Run one independent episode through the exact compact deployment rule."""
    if engine.encoder is None:
        raise ValueError("training requires an engine with an attached encoder")
    labels = batch["labels"].to(device)
    expected = torch.tensor(plan.expected_labels(), dtype=labels.dtype, device=device)
    actual = labels[row_offset:row_offset + len(expected)]
    if not torch.equal(actual, expected):
        raise RuntimeError("episode plan and DataLoader batch disagree on query/support labels")

    encoded = encode_batch(engine.encoder, batch, device) if encoded_out is None else encoded_out
    query_i, support_i, background_i = episode_row_roles(plan, row_offset=row_offset)
    binding = episode_binding(plan, len(labels), row_offset=row_offset).to(device)
    query = live_sensor_rows(
        encoded, batch, labels=labels, enrolled_candidate=binding, select=query_i,
    )
    memory = live_sensor_rows(
        encoded, batch, labels=labels, enrolled_candidate=binding,
        select=torch.cat([support_i, background_i]),
    )
    if len(memory.rows.feature) == 0:
        raise RuntimeError("episode produced no live memory rows")

    candidate_ids = torch.tensor(plan.candidates, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed)
    labels_local = episode_label_set(
        candidate_ids, label_text, mode=plan.label_mode, rng=rng,
        alias_embeddings=alias_text,
    )
    generator = torch.Generator().manual_seed(seed)
    result = engine(
        query.rows, memory.rows, labels_local.embeddings, label_text,
        generator=generator, collect_stats=collect_stats,
    )

    # A query window contributes one row per valid (patch, sensor). Summing is the deployment vote
    # over all observed evidence; the common scale cancels after taking log-mass for cross entropy.
    window_mass = result["logits"].new_zeros((len(labels), len(plan.candidates)))
    window_mass.index_add_(0, query.window, result["logits"])
    mass = window_mass[query_i.to(device)]
    target = torch.tensor(plan.query_slot, dtype=torch.long, device=device)
    loss = F.cross_entropy((mass + 1e-8).log(), target)
    aux = loss.new_zeros(())
    if aux_weight:
        # Supervises RETRIEVAL directly. The vote gradient alone reaches the scorer only through a
        # near-uniform average over 64 rows, which measurement showed carries almost no ranking
        # information; this is the signal that makes retrieval semantic rather than merely similar.
        aux = retrieval_alignment_loss(
            result["scores"], query.rows.label, memory.rows.label, label_text,
            temperature=aux_temperature,
        )
        loss = loss + aux_weight * aux
    prediction = mass.argmax(dim=1)
    out = {
        "loss": loss,
        "task_loss": float(F.cross_entropy((mass + 1e-8).log(), target).detach()),
        "aux_loss": float(aux.detach()),
        "truth": target.detach().cpu().tolist(),
        "prediction": prediction.detach().cpu().tolist(),
        "truth_label": [plan.candidates[int(slot)] for slot in target.detach().cpu()],
        "prediction_label": [plan.candidates[int(slot)] for slot in prediction.detach().cpu()],
        "support_k": plan.support_k,
        "label_mode": plan.label_mode,
        "memory_rows": len(memory.rows.feature),
    }
    if collect_stats:
        stats = dict(result.get("stats", {}))
        selected_bound = memory.rows.enrolled_candidate[result["selected"]]
        query_target = torch.full((len(labels),), -1, dtype=torch.long, device=device)
        query_target[query_i.to(device)] = target
        row_target = query_target[query.window]
        stats["retrieval/true_support_selected_share"] = float(
            selected_bound.eq(row_target.unsqueeze(1)).float().mean()
        )
        stats["episode/accuracy"] = float(prediction.eq(target).float().mean())
        stats.update(representation_health(query.rows.feature, "query_repr"))
        out["stats"] = stats
    return out


@torch.no_grad()
def validate(
    engine: EvidenceEngine, loader: DataLoader, plans: list[EpisodePlan],
    label_text: torch.Tensor, alias_text: torch.Tensor, device: torch.device, seed: int,
    episodes_per_batch: int,
) -> dict:
    engine.eval()
    cells: dict[tuple[str, int], tuple[list[int], list[int]]] = {}
    losses: list[float] = []
    episode = 0
    for batch in loader:
        with _autocast(device):
            encoded = encode_batch(engine.encoder, batch, device)
            step_plans = plans[episode:episode + episodes_per_batch]
            offsets = np.cumsum([0] + [len(plan.flat_positions())
                                        for plan in step_plans[:-1]])
            results = [
                run_episode(
                    engine, batch, plan, label_text, alias_text, device,
                    encoded_out=encoded, row_offset=int(offset), seed=seed + episode + local,
                )
                for local, (plan, offset) in enumerate(zip(step_plans, offsets))
            ]
        for plan, result in zip(step_plans, results):
            losses.append(float(result["loss"]))
            bucket = cells.setdefault((plan.label_mode, plan.support_k), ([], []))
            # Candidate slots are randomized episode-local identities. Scoring those slots as if
            # slot 0 meant one stable class across episodes produces a plausible but meaningless F1.
            bucket[0].extend(result["truth_label"])
            bucket[1].extend(result["prediction_label"])
        episode += len(step_plans)
    if episode != len(plans):
        raise RuntimeError(f"validation consumed {episode} plans, expected {len(plans)}")
    curve = {
        f"{mode}/k={support}": {
            "macro_f1": macro_f1(truth, prediction),
            "queries": len(truth),
        }
        for (mode, support), (truth, prediction) in sorted(cells.items())
    }
    coherent = [row["macro_f1"] for key, row in curve.items() if key.startswith("coherent/")]
    aliases = [row["macro_f1"] for key, row in curve.items() if key.startswith("random_alias/")]
    report = {
        "loss": float(np.mean(losses)),
        "curve": curve,
        "coherent_mean_macro_f1": float(np.mean(coherent)) if coherent else 0.0,
        "alias_mean_macro_f1": float(np.mean(aliases)) if aliases else 0.0,
    }
    # Coherent recognition and arbitrary-label enrollment are co-primary; neither may disappear
    # inside one pooled metric.  Their equal mean is used only for checkpoint ordering.
    report["selection_score"] = 0.5 * (
        report["coherent_mean_macro_f1"] + report["alias_mean_macro_f1"]
    )
    engine.train()
    if engine.encoder is not None and not any(
        parameter.requires_grad for parameter in engine.encoder.parameters()
    ):
        # A frozen encoder is a deterministic feature extractor. Recursive engine.train() would
        # otherwise reactivate its dropout even though its parameters cannot adapt to that noise.
        engine.encoder.eval()
    return report


def _autocast(device: torch.device):
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else nullcontext())


def profile_training(
    engine, train_loader, train_plans, label_text, alias_text, optimizer, scheduler,
    device, args,
) -> None:
    """Where a training step actually goes: phase timers over real steps, then an op-level table.

    Phase timers synchronise at section boundaries, which inflates the total a little but makes
    the attribution honest. The op table runs unsynchronised so overlap is visible there instead.
    """
    encoder = engine.encoder

    def mark():
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    phases = {name: 0.0 for name in
              ("data_wait", "encode", "episodes", "backward", "optimizer")}
    measured = 0
    iterator = iter(train_loader)
    for step in range(1, args.profile_steps + 6):
        timing = step > 5                                   # warmup steps excluded
        t0 = mark()
        batch = next(iterator)
        t1 = mark()
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            encoded = encode_batch(encoder, batch, device)
            t2 = mark()
            start = (step - 1) * args.episodes_per_step
            plans = train_plans[start:start + args.episodes_per_step]
            offsets = np.cumsum([0] + [len(plan.flat_positions()) for plan in plans[:-1]])
            results = [
                run_episode(engine, batch, plan, label_text, alias_text, device,
                            encoded_out=encoded, row_offset=int(offset),
                            seed=args.seed + step * 10_000 + episode)
                for episode, (plan, offset) in enumerate(zip(plans, offsets))
            ]
            loss = torch.stack([result["loss"] for result in results]).mean()
        t3 = mark()
        loss.backward()
        t4 = mark()
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]], args.grad_clip)
        optimizer.step()
        scheduler.step()
        t5 = mark()
        if timing:
            measured += 1
            for name, span in zip(phases, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                phases[name] += span

    total = sum(phases.values())
    print(f"\n=== phase breakdown over {measured} steps "
          f"(bank {args.bank_windows}, {args.episodes_per_step} episodes/step) ===")
    for name, span in phases.items():
        print(f"  {name:10s} {span / measured * 1000:7.1f} ms  {span / total * 100:5.1f}%")
    print(f"  {'TOTAL':10s} {total / measured * 1000:7.1f} ms")

    from torch.profiler import ProfilerActivity, profile
    activities = [ProfilerActivity.CPU] + (
        [ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with profile(activities=activities) as prof:
        for step in range(3):
            batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                encoded = encode_batch(encoder, batch, device)
                start = (args.profile_steps + 5 + step) * args.episodes_per_step
                plans = train_plans[start:start + args.episodes_per_step]
                offsets = np.cumsum([0] + [len(plan.flat_positions())
                                            for plan in plans[:-1]])
                loss = torch.stack([
                    run_episode(engine, batch, plan, label_text, alias_text, device,
                                encoded_out=encoded, row_offset=int(offset),
                                seed=args.seed + step)["loss"]
                    for episode, (plan, offset) in enumerate(zip(plans, offsets))
                ]).mean()
            loss.backward()
            optimizer.step()
    sort = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort, row_limit=14))


def _random_encoder(device: torch.device) -> tuple[SetTokenizerEncoder, dict]:
    config = {
        "d_model": 128,
        "num_layers": 3,
        "num_heads": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "frontend": "fixed",
        "trunk": "temporal",
        "descriptor_prediction": False,
        "text_conditioning": "factored",
        "token_granularity": "sensor",
        "sensor_bias_dim": 14,
        "use_sensor_bias_conditioning": False,
        "use_sensor_isolated_retrieval": False,
        "multiresolution": False,
        # Constructor bounds remain valid even though multiresolution is disabled. Keeping the
        # ordinary bounds makes this checkpoint reconstructible by the shared loader.
        "short_patch_choices": [0.4],
        "long_patch_choices": [1.5],
        "val_resolution_pair": [0.5, 1.5],
        "train_datasets": list(TRAIN_DATASETS),
    }
    encoder = SetTokenizerEncoder(
        d_model=128, num_layers=3, num_heads=4, dim_feedforward=256, dropout=0.1,
        dft_size=DFT_SIZE, frontend="fixed", trunk="temporal",
        descriptor_prediction=False, text_conditioning="factored",
        token_granularity="sensor", use_sensor_bias_conditioning=False,
    ).to(device)
    return encoder, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=list(TRAIN_DATASETS))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--episodes-per-step", type=int, default=4)
    parser.add_argument("--candidate-counts", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--queries-per-candidate", type=int, default=2)
    parser.add_argument("--max-support", type=int, default=4)
    parser.add_argument("--bank-windows", type=int, default=512)
    parser.add_argument("--alias-episode-fraction", type=float, default=0.5)
    parser.add_argument("--disjointness", choices=("subject", "stream"), default="stream")
    parser.add_argument("--no-shared-query-stream", action="store_true")
    parser.add_argument("--holdout-label-fraction", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--readout", choices=("weights", "semantic"), default="weights")
    # --- ablation ladder: substitute a fixed stand-in for one learnable stage at a time
    parser.add_argument("--retrieval", choices=("learned", "cosine"), default="cosine",
                        help="'cosine' replaces the learned pair scorer with the plain feature "
                             "cosine at a fixed temperature, registering no parameters")
    parser.add_argument("--vote-scope", choices=("topk", "bank"), default="topk",
                        help="'bank' votes over the whole memory while the mixer still sees only "
                             "the top-k, so every row receives gradient instead of the ~1.3% per "
                             "query that hard selection reaches")
    parser.add_argument("--mixing", choices=("attention", "off"), default="attention",
                        help="'off' removes the evidence mixer entirely: the weight is the "
                             "retrieval score and the label vectors are the frozen row text")
    parser.add_argument("--identity-gain", type=float, default=1.0,
                        help="initial gain on the mixer's role/slot/group channels relative to "
                             "content at 1.0; at 1.0 content is only a quarter of every token")
    parser.add_argument("--mixer-layers", type=int, default=2,
                        help="0 keeps the readout but gives it uncontextualized tokens, which "
                             "separates 'the readout helps' from 'attention helps'")
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--engine-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--retrieval-aux-weight", type=float, default=0.0,
                        help="weight on the retrieval alignment loss (0 = the measured baseline)")
    parser.add_argument("--retrieval-aux-temperature", type=float, default=0.1)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--calib-batches", type=int, default=5,
                        help="episode batches used to fit frozen filterbank normalization for "
                             "--random-init; checkpoints must already carry fitted statistics")
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--val-episodes", type=int, default=32)
    parser.add_argument("--telemetry-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-seed", type=int, default=VAL_SEED,
                        help="seed for the concept holdout and the validation episodes, kept "
                             "SEPARATE from --seed so every run is scored on the same draw")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=0,
                        help="measure this many real steps (phase timers + op table), then exit")
    args = parser.parse_args()

    if (args.checkpoint is None) == (not args.random_init):
        parser.error("choose exactly one of --checkpoint or --random-init")
    if (args.steps < 1 or args.episodes_per_step < 1 or args.bank_windows < 1
            or args.calib_batches < 1):
        parser.error("steps, episodes-per-step, bank-windows, and calib-batches must be positive")
    if args.top_k > args.bank_windows:
        parser.error("--top-k cannot exceed --bank-windows")
    # The mixer keeps every attention activation of a (C + 2 + 3k)-token sequence per query row for
    # backward. At k=512 that OOMed a 24 GiB card 21 steps in; fail at parse time with arithmetic
    # instead. Rough bound: rows_per_step * seq_len^2 * heads * 2 bytes for the attention maps.
    seq = max(args.candidate_counts) + 2 + 3 * args.top_k
    rows = args.episodes_per_step * max(args.candidate_counts) * args.queries_per_candidate * 2
    attention_gib = rows * seq * seq * 4 * 2 * args.mixer_layers / 2 ** 30
    if attention_gib > 16:
        parser.error(
            f"top_k={args.top_k} gives {seq}-token mixer sequences; the retained attention maps "
            f"alone are ~{attention_gib:.0f} GiB for backward. Use --vote-scope bank for full "
            "gradient coverage at top-k cost instead of raising k."
        )
    if args.warmup_steps < 0 or args.warmup_steps > args.steps:
        parser.error("--warmup-steps must lie in [0, steps]")
    if args.smoke:
        args.steps, args.val_every, args.val_episodes = 3, 3, 2
        args.warmup_steps, args.num_workers, args.calib_batches = 1, 0, 1
        args.datasets = ["uci_har", "wisdm", "mhealth", "pamap2"]
        args.candidate_counts, args.max_support = [2, 4], 2
        args.bank_windows, args.top_k = 32, 16

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    index = CorpusIndex(seed=args.seed, datasets=tuple(args.datasets))
    print(f"[phase-b] corpus: {index.summary()}", flush=True)
    combined_keys = list(index.train) + list(index.val)
    val_offset = len(index.train)

    base_spec = EpisodeSpec(
        candidate_counts=tuple(args.candidate_counts),
        queries_per_candidate=args.queries_per_candidate,
        max_support=args.max_support,
        # The episode planner requires a nonempty fallback on k=0. It is replaced below by the
        # fixed-size bank, and never reaches the DataLoader.
        background_windows=1,
        alias_episode_fraction=args.alias_episode_fraction,
        disjointness=args.disjointness,
        shared_query_stream=(args.disjointness == "stream" and not args.no_shared_query_stream),
    )
    train_subject = subject_ids_for(index, index.train)
    val_subject = subject_ids_for(index, index.val)
    train_stream, _ = stream_ids_for(index, index.train)
    val_stream, _ = stream_ids_for(index, index.val)
    train_execution = execution_ids_for(index, index.train)
    val_execution = execution_ids_for(index, index.val)
    train_table = label_window_table(index.train, train_subject)
    val_table = label_window_table(index.val, val_subject)
    train_stream_table = stream_label_table(index.train, train_subject, train_stream)
    val_stream_table = stream_label_table(index.val, val_subject, val_stream)
    eligible = eligible_labels(
        train_table, base_spec, train_stream, execution_ids=train_execution,
    )
    train_pool, heldout = split_label_pool(eligible, args.holdout_label_fraction, args.val_seed)
    val_eligible = set(eligible_labels(
        val_table, base_spec, val_stream, execution_ids=val_execution,
    ))
    val_pool = sorted(set(heldout) & val_eligible)
    train_spec = replace_counts(base_spec, len(train_pool))
    val_spec = replace_counts(base_spec, len(val_pool))
    print(f"[phase-b] concepts: {len(train_pool)} train / {len(val_pool)} held out", flush=True)

    n_train = args.steps * args.episodes_per_step
    train_plans = _make_plans(
        train_table, train_stream_table, train_pool, train_spec, train_stream, train_execution,
        n_episodes=n_train, seed=args.seed,
    )
    bank_spec = BankSpec(n_windows=args.bank_windows)
    # A held-out concept is absent from the whole Phase-B objective, not merely absent as a query.
    # Letting its labeled rows enter the background bank would expose its name and representation
    # to attention before validation and turn "held out" into "seen only as a distractor".
    train_pool_set = set(train_pool)
    objective_bank_table = {
        stream: {label: by_subject for label, by_subject in by_label.items()
                 if label in train_pool_set}
        for stream, by_label in train_stream_table.items()
    }
    objective_bank_table = {
        stream: by_label for stream, by_label in objective_bank_table.items() if by_label
    }
    train_bank = bank_index(objective_bank_table)
    train_plans = _attach_training_banks(train_plans, train_bank, bank_spec, args.seed + 1)

    support_grid = tuple(value for value in (0, 1, 2, 4) if value <= args.max_support)
    max_support = max(support_grid)
    coherent_base = _make_plans(
        val_table, val_stream_table, val_pool,
        dataclasses.replace(val_spec, alias_episode_fraction=0.0),
        val_stream, val_execution, n_episodes=args.val_episodes,
        seed=args.val_seed + 2, schedule=(max_support,),
    )
    coherent_val = _matched_validation_plans(
        coherent_base, support_grid, train_bank, bank_spec, val_offset, args.val_seed + 3,
    )
    positive_grid = tuple(value for value in support_grid if value > 0)
    alias_base = _make_plans(
        val_table, val_stream_table, val_pool,
        dataclasses.replace(val_spec, alias_episode_fraction=1.0),
        val_stream, val_execution, n_episodes=args.val_episodes,
        seed=args.val_seed + 4, schedule=(max_support,),
    )
    alias_val = _matched_validation_plans(
        alias_base, positive_grid, train_bank, bank_spec, val_offset, args.val_seed + 5,
    ) if positive_grid else []
    val_plans = coherent_val + alias_val

    augmentation = AugmentationConfig.phase_b_generic() if args.augment else AugmentationConfig.none()
    dataset = PretrainDataset(
        index, combined_keys, augment=args.augment, augmentation_config=augmentation,
    )
    collate = EpisodicCollate(MultiScaleCollate(fixed_patch_seconds=PATCH_SECONDS))
    loader_kwargs = dict(
        dataset=dataset, collate_fn=collate, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", worker_init_fn=_seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        batch_sampler=GroupedEpisodicBatchSampler(train_plans, args.episodes_per_step),
        **loader_kwargs,
    )
    val_group = math.gcd(len(val_plans), args.episodes_per_step)
    val_loader = DataLoader(
        batch_sampler=GroupedEpisodicBatchSampler(val_plans, val_group), **loader_kwargs,
    )

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        encoder = build_encoder(checkpoint, device, training=True)
        config = dict(checkpoint["config"])
        source = str(args.checkpoint)
    else:
        encoder, config = _random_encoder(device)
        source = "random-init"
    if encoder.trunk != "temporal" or encoder.token_granularity != "sensor":
        raise SystemExit(
            "Phase B accepts only the compact temporal sensor encoder; train/reload Phase A with "
            "--trunk temporal --token-granularity sensor"
        )
    if encoder.use_sensor_bias_conditioning:
        raise SystemExit("compact Phase B does not consume the legacy source-specific sensor bias")
    frontend = encoder.filterbank
    if args.random_init:
        print(f"[phase-b] calibrating filterbank on {args.calib_batches} episode batches", flush=True)
        frontend.reset_norm_accumulator()
        for batch_index, batch in enumerate(train_loader):
            frontend.accumulate_norm_stats(
                batch["patches"].to(device, non_blocking=True),
                batch["rates"].to(device, non_blocking=True),
                batch["patch_len"].to(device, non_blocking=True),
                patch_mask=batch["patch_padding_mask"].to(device, non_blocking=True),
                channel_mask=batch["channel_mask"].to(device, non_blocking=True),
                source_rate_hz=batch["source_rates"].to(device, non_blocking=True),
            )
            if batch_index + 1 >= args.calib_batches:
                break
        frontend.finalize_norm_stats()
    elif not bool(frontend._norm_fitted.item()):
        raise SystemExit(
            "the supplied Phase-A checkpoint has no fitted filterbank normalization; refusing "
            "to train Phase B on identity-scaled physical features"
        )
    encoder.mask_token.requires_grad_(False)
    if args.freeze_encoder:
        encoder.requires_grad_(False)

    spec = encoder.attention_spec
    engine_config = EngineConfig(
        spec=spec,
        trunk_layers=int(encoder.transformer.num_layers),
        top_k=args.top_k,
        mixing=args.mixing,
        vote_scope=args.vote_scope,
        scorer=PairScorerConfig(learned=args.retrieval == "learned"),
        mixer=EvidenceMixerConfig(n_groups=max(96, args.top_k + 2), readout=args.readout,
                                  n_layers=args.mixer_layers,
                                  identity_gain_init=args.identity_gain),
    )
    engine = EvidenceEngine(encoder, engine_config).to(device).train()
    if args.freeze_encoder:
        encoder.eval()
    label_text = F.normalize(label_text_matrix(encoder, index.label_ids, device).float(), dim=-1)
    from eval.scoring import get_sbert_encoder

    alias_text = F.normalize(
        encode_neutral_aliases(get_sbert_encoder(), device).float(), dim=-1,
    )

    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    scorer_params = [p for p in engine.scorer.parameters() if p.requires_grad]
    mixer_params = ([p for p in engine.mixer.parameters() if p.requires_grad]
                    if engine.mixer is not None else [])
    # A ladder rung that substitutes a fixed stand-in leaves a stage with no parameters at all;
    # AdamW rejects an empty group, so only non-empty ones are handed over.
    groups = [{"params": params, "lr": lr} for params, lr in
              ((encoder_params, args.encoder_lr), (scorer_params, args.engine_lr),
               (mixer_params, args.engine_lr)) if params]
    if not groups:
        raise SystemExit("every stage is frozen; there is nothing to train")
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    def lr_factor(step: int) -> float:
        if args.warmup_steps and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    provenance = capture_source_provenance(
        args.out, write=True,
        roots=("training/tokenizer", "training/evidence", "model/tokenizer",
               "model/evidence", "model/blocks.py", "data/datasets", "data/scripts"),
    )
    provenance.pop("_patch", None)
    config.update({
        "engine_config": dataclasses.asdict(engine_config),
        "train_datasets": list(args.datasets),
        "phase_b": {key: (str(value) if isinstance(value, Path) else value)
                    for key, value in vars(args).items()},
    })
    run_info = {
        "source_checkpoint": source,
        "corpus": index.summary(),
        "corpus_fingerprint": corpus_fingerprint(index),
        "train_concepts": train_pool,
        "heldout_concepts": val_pool,
        "episode_spec": dataclasses.asdict(train_spec),
        "bank_spec": dataclasses.asdict(bank_spec),
        "train_provenance_lift": provenance_lift(train_plans, train_stream),
        "parameters": engine.parameter_report(),
        "config": config,
        "source_provenance": provenance,
    }
    (args.out / "run_config.json").write_text(json.dumps(run_info, indent=2, default=str))
    telemetry_path = args.out / "log.jsonl"
    telemetry_path.write_text("")

    def record(row: dict) -> None:
        with telemetry_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    best = {"score": -1.0, "step": 0}

    def payload(step: int, report: dict) -> dict:
        return {
            "encoder": encoder.state_dict(),
            "evidence_engine": engine.state_dict(),
            "config": config,
            "label_ids": index.label_ids,
            "step": step,
            "selection_metric": "mean(coherent_curve_mean, alias_positive_curve_mean)",
            "selection_value": report["selection_score"],
            "validation": report,
            "source_provenance": provenance,
            "corpus_fingerprint": run_info["corpus_fingerprint"],
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }

    def run_validation(step: int) -> dict:
        report = validate(
            engine, val_loader, val_plans, label_text, alias_text, device, args.val_seed + 10_000,
            val_group,
        )
        record({"step": step, "kind": "validation", **report})
        if report["selection_score"] > best["score"]:
            best.update(score=report["selection_score"], step=step)
            torch.save(payload(step, report), args.out / "best.pt")
        return report

    if args.profile_steps:
        profile_training(engine, train_loader, train_plans, label_text, alias_text,
                         optimizer, scheduler, device, args)
        return

    initial = run_validation(0)
    for step, batch in enumerate(train_loader, 1):
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            encoded = encode_batch(encoder, batch, device)
            start = (step - 1) * args.episodes_per_step
            plans = train_plans[start:start + args.episodes_per_step]
            offsets = np.cumsum([0] + [len(plan.flat_positions()) for plan in plans[:-1]])
            results = [
                run_episode(
                    engine, batch, plan, label_text, alias_text, device,
                    encoded_out=encoded, row_offset=int(offset),
                    seed=args.seed + step * 10_000 + episode,
                    collect_stats=(step == 1 or step % args.telemetry_every == 0),
                    aux_weight=args.retrieval_aux_weight,
                    aux_temperature=args.retrieval_aux_temperature,
                )
                for episode, (plan, offset) in enumerate(zip(plans, offsets))
            ]
            loss = torch.stack([result["loss"] for result in results]).mean()
        loss.backward()
        encoder_grad = _grad_norm(encoder_params)
        scorer_grad = _grad_norm(scorer_params)
        mixer_grad = _grad_norm(mixer_params)
        preclip = float(torch.nn.utils.clip_grad_norm_(
            [p for group in groups for p in group["params"]], args.grad_clip,
        ))
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.telemetry_every == 0:
            stats = [result.get("stats", {}) for result in results]
            keys = sorted({key for row in stats for key in row})
            row = {
                "step": step,
                "kind": "train",
                "loss": float(loss.detach()),
                "task_loss": float(np.mean([r["task_loss"] for r in results])),
                "aux_loss": float(np.mean([r["aux_loss"] for r in results])),
                "accuracy": float(np.mean([
                    np.mean(np.equal(result["truth"], result["prediction"])) for result in results
                ])),
                "encoder_grad_norm": encoder_grad,
                "scorer_grad_norm": scorer_grad,
                "mixer_grad_norm": mixer_grad,
                "total_preclip_grad_norm": preclip,
                "clip_coefficient": min(1.0, args.grad_clip / max(preclip, 1e-12)),
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            row.update({key: float(np.mean([value[key] for value in stats if key in value]))
                        for key in keys})
            record(row)
        if step % args.val_every == 0 or step == args.steps:
            final = run_validation(step)

    torch.save(payload(args.steps, final), args.out / "last.pt")
    summary = {
        "steps": args.steps,
        "best": best,
        "initial_validation": initial,
        "final_validation": final,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
