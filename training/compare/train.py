"""Fine-tune the encoder and the support-only comparator end to end.

THE LOSS
--------
Two regimes, interleaved in one stream at the sampler's rate ``p``:

* **few-shot** (the answer is among the candidates): plain cross-entropy on the ground-truth slot.
* **zero-shot** (the answer is absent): a one-hot target is undefined, so the target is the
  label-text similarity distribution

      t = softmax( cos(text(query label), text(candidate)) / tau_text ),   tau_text = 0.1

  and the loss is ``KL(t || softmax(z))``. Without this the closed-vocabulary objective drives
  zero-shot *below chance* — measured, not feared: 16.18 to 9.44 macro-F1 on the previous design.

Each group is averaged within itself and the two are summed with weight 1. They are not weighted by
episode count, because the mix is already set by ``p`` and weighting twice would make ``p`` do two
jobs at once.

WARM START IS NOT OPTIONAL
--------------------------
From random initialisation the encoder's effective rank collapsed from 24 to about 9 inside 300
steps on the previous design, and the run never recovered. ``--phase-a`` is required unless
``--allow-random-init`` is passed explicitly, and ``encoder/effective_rank`` is logged every
validation so the same failure is visible immediately rather than at the end of a 3-hour run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.scripts.curate import deployment_policy
from model.blocks import AttentionSpec
from model.evidence.comparator import ComparatorConfig, SupportComparator, comparator_logits
from training.compare.corpus import support_corpus_from_index
from training.compare.sampling import (
    DEFAULT_LABEL_SUBSET,
    DEFAULT_P_GT_PRESENT,
    DEFAULT_SUPPORT,
    Episode,
    SupportCorpus,
    draw_batch,
)
from training.tokenizer.episodic import EpisodicCollate
from training.tokenizer.eval_transfer import build_encoder
from training.tokenizer.pretrain_data import (
    PATCH_SECONDS,
    CorpusIndex,
    MultiScaleCollate,
    PretrainDataset,
)
from training.tokenizer.pretrain_episodic import encode_batch

TAU_TEXT = 0.1          # a priori; matches the retrieval temperature used elsewhere in the repo
TAU_SUPPORT = 0.07      # closed-form weighting temperature
VOTE_SCALE = 10.0


# ------------------------------------------------------------------ text tower
def label_text_matrix(labels: list[str], device) -> torch.Tensor:
    """Frozen MiniLM embeddings for verbatim label strings.

    The same frozen sentence encoder every scored path in the repo uses, so a label's vector here
    is the vector the evaluation harness would give it.
    """
    from eval.scoring import get_sbert_encoder

    embeddings = torch.from_numpy(get_sbert_encoder()(list(labels))).to(device)
    return F.normalize(embeddings.float(), dim=-1)


# ------------------------------------------------------------------ batching
def recording_rows(encoded: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """One pooled feature and one acquisition descriptor per encoded recording.

    The descriptor is the normalised mean over the sensors that are actually present, matching
    ``training.tokenizer.episodic.live_recording_rows`` exactly. A recording may carry both an
    accelerometer and a gyroscope, so the acquisition configuration lives in this vector rather
    than in a scalar modality code.
    """
    pooled = encoded.get("pooled")
    descriptor = encoded.get("descriptor")
    present = encoded.get("sensor_present")
    if pooled is None or descriptor is None or present is None:
        raise KeyError("the comparator needs pooled, descriptor and sensor_present outputs")
    if bool((~present.any(dim=1)).any()):
        raise ValueError("an encoded recording carries no real sensor")
    weight = present.unsqueeze(-1).to(descriptor.dtype)
    merged = (descriptor * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
    return pooled, F.normalize(merged.float(), dim=-1)


def episode_positions(episodes: list[Episode], corpus: SupportCorpus) -> list[int]:
    """Every dataset position one step needs, query rows first within each episode."""
    positions: list[int] = []
    for episode in episodes:
        positions.append(corpus.recordings[episode.query].window_index)
        positions.extend(corpus.recordings[i].window_index for i in episode.support)
    return positions


def split_encoded(
    pooled: torch.Tensor,
    descriptor: torch.Tensor,
    episodes: list[Episode],
) -> dict[str, torch.Tensor]:
    """Regroup one flat encoder output into padded per-episode query and support tensors."""
    device = pooled.device
    B = len(episodes)
    K = max((len(episode.support) for episode in episodes), default=0)
    d = pooled.shape[-1]
    z = descriptor.shape[-1]

    query_feature = pooled.new_zeros((B, 1, d))
    query_descriptor = descriptor.new_zeros((B, 1, z))
    support_feature = pooled.new_zeros((B, K, d))
    support_descriptor = descriptor.new_zeros((B, K, z))
    support_mask = torch.zeros((B, K), dtype=torch.bool, device=device)

    cursor = 0
    for row, episode in enumerate(episodes):
        query_feature[row, 0] = pooled[cursor]
        query_descriptor[row, 0] = descriptor[cursor]
        cursor += 1
        for slot in range(len(episode.support)):
            support_feature[row, slot] = pooled[cursor]
            support_descriptor[row, slot] = descriptor[cursor]
            support_mask[row, slot] = True
            cursor += 1
    if cursor != pooled.shape[0]:
        raise RuntimeError(
            f"episodes cover {cursor} rows but the encoder returned {pooled.shape[0]}"
        )
    return {
        "query_feature": query_feature,
        "query_descriptor": query_descriptor,
        "query_mask": torch.ones((B, 1), dtype=torch.bool, device=device),
        "support_feature": support_feature,
        "support_descriptor": support_descriptor,
        "support_mask": support_mask,
    }


def episode_text(
    episodes: list[Episode],
    corpus: SupportCorpus,
    text_of,
    device,
) -> dict[str, torch.Tensor]:
    """Candidate text, support-label text, bindings and slots for one batch of episodes."""
    B = len(episodes)
    C = max(len(episode.candidates) for episode in episodes)
    K = max((len(episode.support) for episode in episodes), default=0)
    z = text_of("walking").shape[-1]

    candidate_text = torch.zeros((B, C, z), device=device)
    support_label_text = torch.zeros((B, K, z), device=device)
    support_bound = torch.full((B, K), -1, dtype=torch.long, device=device)
    candidate_slot = torch.zeros((B, C), dtype=torch.long, device=device)
    candidate_mask = torch.zeros((B, C), dtype=torch.bool, device=device)
    query_label_text = torch.zeros((B, z), device=device)

    for row, episode in enumerate(episodes):
        for slot, label in enumerate(episode.candidates):
            candidate_text[row, slot] = text_of(label)
            candidate_mask[row, slot] = True
        # Coreference tags are redrawn per episode so they carry no label identity across episodes.
        candidate_slot[row, : len(episode.candidates)] = torch.randperm(
            len(episode.candidates), device=device,
        )
        for slot, index in enumerate(episode.support):
            support_label_text[row, slot] = text_of(corpus.recordings[index].label)
            support_bound[row, slot] = episode.support_candidate[slot]
        query_label_text[row] = text_of(corpus.recordings[episode.query].label)
    return {
        "candidate_text": candidate_text,
        "support_label_text": support_label_text,
        "support_bound": support_bound,
        "candidate_slot": candidate_slot,
        "candidate_mask": candidate_mask,
        "query_label_text": query_label_text,
    }


# ------------------------------------------------------------------ loss
def episode_loss(
    logits: torch.Tensor,           # (B, C)
    episodes: list[Episode],
    text: dict[str, torch.Tensor],
    *,
    tau_text: float = TAU_TEXT,
) -> dict[str, torch.Tensor]:
    """Few-shot cross-entropy plus zero-shot soft-target KL, each averaged within its group."""
    device = logits.device
    masked = logits.masked_fill(~text["candidate_mask"], float("-inf"))
    log_probability = torch.log_softmax(masked, dim=-1)

    few_shot = torch.tensor(
        [index for index, e in enumerate(episodes) if not e.is_zero_shot],
        dtype=torch.long, device=device,
    )
    zero_shot = torch.tensor(
        [index for index, e in enumerate(episodes) if e.is_zero_shot],
        dtype=torch.long, device=device,
    )

    ce = logits.new_zeros(())
    if len(few_shot):
        target = torch.tensor(
            [episodes[int(i)].gt_slot for i in few_shot], dtype=torch.long, device=device,
        )
        ce = F.nll_loss(log_probability[few_shot], target)

    kl = logits.new_zeros(())
    if len(zero_shot):
        similarity = torch.einsum(
            "bz,bcz->bc", text["query_label_text"], text["candidate_text"],
        )
        similarity = similarity.masked_fill(~text["candidate_mask"], float("-inf"))
        soft_target = torch.softmax(similarity[zero_shot] / tau_text, dim=-1)
        rows = log_probability[zero_shot]
        # NOT F.kl_div: episodes carry different candidate counts, so padded slots hold
        # target 0 against log-probability -inf, and their product is NaN rather than the 0 the
        # limit gives. Summing only over slots with positive target mass is the same quantity
        # without the indeterminate form.
        term = soft_target * (soft_target.clamp_min(1e-12).log() - rows)
        term = torch.where(soft_target > 0, term, torch.zeros_like(term))
        kl = term.sum(dim=-1).mean()
    return {"loss": ce + kl, "ce": ce.detach(), "kl": kl.detach()}


def effective_rank(features: torch.Tensor) -> float:
    """exp(entropy of the normalised singular spectrum) — the collapse watchdog."""
    matrix = features.detach().float().flatten(0, -2)
    if matrix.shape[0] < 2:
        return float("nan")
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(matrix)
    share = singular / singular.sum().clamp_min(1e-12)
    entropy = -(share * share.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


# ------------------------------------------------------------------ training
def build_dataset(index: CorpusIndex, args) -> PretrainDataset:
    return PretrainDataset(
        index, index.train, augment=False, two_view=False,
        neutral_acquisition_text=args.neutral_acquisition_text,
    )


def run_step(
    *,
    episodes: list[Episode],
    corpus: SupportCorpus,
    dataset: PretrainDataset,
    collate,
    encoder,
    comparator: SupportComparator | None,
    text_of,
    device: torch.device,
) -> dict:
    positions = episode_positions(episodes, corpus)
    batch = collate([dataset[position] for position in positions])
    encoded = encode_batch(encoder, batch, device)
    pooled, descriptor = recording_rows(encoded)

    rows = split_encoded(pooled, descriptor, episodes)
    text = episode_text(episodes, corpus, text_of, device)
    output = comparator_logits(
        comparator,
        candidate_text=text["candidate_text"],
        query_feature=rows["query_feature"],
        query_descriptor=rows["query_descriptor"],
        query_mask=rows["query_mask"],
        support_feature=rows["support_feature"],
        support_descriptor=rows["support_descriptor"],
        support_label_text=text["support_label_text"],
        support_bound=text["support_bound"],
        support_mask=rows["support_mask"],
        candidate_slot=text["candidate_slot"],
        temperature=TAU_SUPPORT,
        vote_scale=VOTE_SCALE,
    )
    loss = episode_loss(output["logits"], episodes, text)
    return {**loss, **output, "pooled": pooled, "text": text}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a", type=Path, default=None,
                        help="Phase-A checkpoint to warm-start the encoder from (required "
                             "unless --allow-random-init)")
    parser.add_argument("--allow-random-init", action="store_true",
                        help="train the encoder from scratch. Measured to collapse effective "
                             "rank 24 -> 9 within 300 steps; for the documented ablation only")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=35_000)
    parser.add_argument("--episodes-per-step", type=int, default=8)
    parser.add_argument("--support-size", type=int, default=DEFAULT_SUPPORT)
    parser.add_argument("--p-gt-present", type=float, default=DEFAULT_P_GT_PRESENT)
    parser.add_argument("--label-subset", type=int, nargs=2, default=list(DEFAULT_LABEL_SUBSET))
    parser.add_argument("--mode", choices=("compatible", "near_miss", "unfiltered"),
                        default="compatible")
    parser.add_argument("--neutral-acquisition-text", action="store_true",
                        help="Arm A: the encoder is told nothing about the acquisition config")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-per-stream", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="~50 steps on real data with telemetry; launches nothing long")
    args = parser.parse_args()

    if args.phase_a is None and not args.allow_random_init:
        parser.error(
            "--phase-a is required. Warm-starting is not a preference: from random init the "
            "encoder's effective rank collapsed 24 -> 9 in 300 steps and never recovered. Pass "
            "--allow-random-init only to reproduce that ablation."
        )
    if args.smoke:
        args.steps = min(args.steps, 50)
        args.log_every = 10
        args.max_per_stream = args.max_per_stream or 400

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    index = CorpusIndex(
        max_per_stream=args.max_per_stream, seed=args.seed,
        datasets=deployment_policy.EXPANDED_PHASE_A_TRAIN_DATASETS, alignment="native",
    )
    print(f"[compare] corpus: {index.summary()}", flush=True)
    corpus = support_corpus_from_index(index)
    print(f"[compare] support corpus: {corpus.summary()}", flush=True)

    dataset = build_dataset(index, args)
    collate = EpisodicCollate(MultiScaleCollate(fixed_patch_seconds=PATCH_SECONDS))

    if args.phase_a is None:
        raise SystemExit(
            "random-init construction is intentionally not wired here: an encoder still needs a "
            "Phase-A config to be built from. Pass --phase-a."
        )
    # Local checkpoint written by our own Phase-A trainer; its `config` holds plain Python values
    # alongside the tensors, which weights_only=True refuses to unpickle.
    checkpoint = torch.load(args.phase_a, map_location="cpu", weights_only=False)
    if bool(checkpoint.get("config", {}).get("neutral_acquisition_text", False)) != \
            args.neutral_acquisition_text:
        raise SystemExit(
            "--neutral-acquisition-text must match the Phase-A checkpoint. Arm A's claim is that "
            "the encoder never saw acquisition text at ANY stage; mixing the two stages would "
            "quietly void it."
        )
    encoder = build_encoder(checkpoint, device, training=True)
    print(f"[compare] warm-started from {args.phase_a}", flush=True)

    spec = AttentionSpec(d_model=encoder.d_model, n_heads=4, ffn_mult=2, dropout=0.1)
    comparator = SupportComparator(spec, ComparatorConfig()).to(device)

    text_cache: dict[str, torch.Tensor] = {}

    def text_of(label: str) -> torch.Tensor:
        if label not in text_cache:
            text_cache[label] = label_text_matrix([label], device)[0]
        return text_cache[label]

    optimizer = torch.optim.AdamW([
        {"params": list(comparator.parameters()), "lr": args.lr},
        {"params": [p for p in encoder.parameters() if p.requires_grad],
         "lr": args.lr * args.encoder_lr_scale},
    ], weight_decay=args.weight_decay)

    log_path = args.out / "log.jsonl"
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        warm = min(1.0, step / max(1, args.warmup_steps))
        for group, base in zip(optimizer.param_groups,
                               (args.lr, args.lr * args.encoder_lr_scale)):
            group["lr"] = base * warm

        episodes, telemetry = draw_batch(
            corpus, rng,
            batch_size=args.episodes_per_step,
            support_size=args.support_size,
            p_gt_present=args.p_gt_present,
            label_subset=tuple(args.label_subset),
            mode=args.mode,
        )
        result = run_step(
            episodes=episodes, corpus=corpus, dataset=dataset, collate=collate,
            encoder=encoder, comparator=comparator, text_of=text_of, device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(comparator.parameters()) + list(encoder.parameters()), args.grad_clip,
        )
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            row = {
                "step": step,
                "loss": float(result["loss"].detach()),
                "loss/ce": float(result["ce"]),
                "loss/kl": float(result["kl"]),
                "encoder/effective_rank": effective_rank(result["pooled"]),
                "elapsed_s": round(time.perf_counter() - started, 1),
                **telemetry,
                **comparator.telemetry(),
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                f"[compare] step {step:>6} loss {row['loss']:.4f} "
                f"(ce {row['loss/ce']:.4f} kl {row['loss/kl']:.4f}) "
                f"rank {row['encoder/effective_rank']:.1f} "
                f"gt {row['sampler/realised_gt_rate']:.2f} "
                f"K {row['sampler/mean_support_size']:.1f} "
                f"shrunk {row['sampler/shrunk_episode_fraction']:.2f}",
                flush=True,
            )

    payload = {
        "encoder": encoder.state_dict(),
        "comparator": comparator.state_dict(),
        "comparator_config": dataclasses.asdict(comparator.cfg),
        "attention_spec": dataclasses.asdict(spec),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "step": args.steps,
    }
    torch.save(payload, args.out / "last.pt")
    print(f"[compare] wrote {args.out / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
