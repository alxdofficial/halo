"""Decompose the k=0 gap: is it the bank, retrieval, or the weighting?

The first end-to-end run plateaus at coherent k=0 macro-F1 about 0.32 while the text geometry
admits far more: hand-picking one bank row per query wins 95-100% of episodes, because held-out
concept names ARE separable against the training vocabulary. So the loss is somewhere between
"a useful row exists in the bank" and "the model's vote picks the right candidate".

Three nested oracles localize it, each relaxing exactly one stage and leaving the others as the
model actually ran them:

  ACTUAL          the model's retrieved top-k and the model's own weights          <- what we get
  ORACLE-WEIGHT   the model's SAME retrieved top-k, weights chosen to maximize
                  the true candidate's margin                                      <- is weighting the loss?
  ORACLE-RETRIEVE the whole bank, best single row                                  <- is retrieval the loss?
  ORACLE-BANK     as above, and the ceiling is what the bank contains at all       <- is the bank the loss?

Read the result as a waterfall. If ORACLE-WEIGHT is barely above ACTUAL but ORACLE-RETRIEVE is far
above both, the retrieved set does not contain the rows the vote needs and retrieval is the thing
to fix. If ORACLE-WEIGHT recovers most of the gap, the rows are there and the mixer is failing to
weight them.

Run:  python -m training.evidence.phase_b_bottleneck --run <dir> [--episodes 48]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.evidence.rows import evidence_label_tokens
from training.evidence.gate_predictor import load_evidence_engine
from training.tokenizer.episodic import (
    BankSpec, EpisodeSpec, EpisodicCollate, bank_index, episode_binding, episode_row_roles,
    label_window_table, live_sensor_rows, macro_f1, sample_bank_positions, stream_label_table,
)
from training.tokenizer.eval_transfer import build_encoder
from training.tokenizer.pretrain_data import (
    PATCH_SECONDS, TRAIN_DATASETS, CorpusIndex, MultiScaleCollate, PretrainDataset,
)
from training.tokenizer.pretrain_episodic import (
    encode_batch, eligible_labels, execution_ids_for, label_text_matrix, replace_counts,
    split_label_pool, stream_ids_for, subject_ids_for, _make_plans,
)


def oracle_margins(evidence: torch.Tensor, target: int) -> dict[str, float]:
    """Best achievable margin for the true candidate over a set of rows.

    ``evidence`` is (rows, candidates) rectified label-text cosine — exactly what the vote consumes.
    The best single-row choice is the tightest honest upper bound on any convex weighting, because
    a convex combination of rows can never exceed the best row's margin.
    """
    others = evidence.clone()
    others[:, target] = float("-inf")
    margin = evidence[:, target] - others.max(dim=1).values
    return {"best_margin": float(margin.max()), "wins": bool(margin.max() > 0)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--datasets", nargs="+", default=list(TRAIN_DATASETS))
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    device = torch.device(args.device)

    blob = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    encoder = build_encoder(blob, device, training=False).eval()
    engine = load_evidence_engine(args.run / "best.pt", encoder=encoder, device=device)
    engine.eval()
    print(f"loaded step {blob['step']}, top_k={engine.cfg.top_k}, "
          f"readout={engine.cfg.mixer.readout}")

    index = CorpusIndex(seed=args.seed, datasets=tuple(args.datasets))
    combined = list(index.train) + list(index.val)
    val_offset = len(index.train)
    spec = EpisodeSpec(candidate_counts=(2, 4, 8, 16), queries_per_candidate=2, max_support=4,
                       background_windows=1, alias_episode_fraction=0.0,
                       disjointness="stream", shared_query_stream=True)
    train_subject, val_subject = subject_ids_for(index, index.train), subject_ids_for(index, index.val)
    train_stream, _ = stream_ids_for(index, index.train)
    val_stream, _ = stream_ids_for(index, index.val)
    train_execution = execution_ids_for(index, index.train)
    val_execution = execution_ids_for(index, index.val)
    train_table = label_window_table(index.train, train_subject)
    val_table = label_window_table(index.val, val_subject)
    train_stream_table = stream_label_table(index.train, train_subject, train_stream)
    val_stream_table = stream_label_table(index.val, val_subject, val_stream)
    eligible = eligible_labels(train_table, spec, train_stream, execution_ids=train_execution)
    train_pool, heldout = split_label_pool(eligible, 0.2, args.seed)
    val_eligible = set(eligible_labels(val_table, spec, val_stream, execution_ids=val_execution))
    val_pool = sorted(set(heldout) & val_eligible)
    val_spec = replace_counts(spec, len(val_pool))

    plans = _make_plans(val_table, val_stream_table, val_pool, val_spec, val_stream,
                        val_execution, n_episodes=args.episodes, seed=args.seed + 2,
                        schedule=(0,))
    train_pool_set = set(train_pool)
    objective_table = {
        stream: {label: by_subject for label, by_subject in by_label.items()
                 if label in train_pool_set}
        for stream, by_label in train_stream_table.items()
    }
    bank = bank_index({s: v for s, v in objective_table.items() if v})
    bank_spec = BankSpec(n_windows=512)
    rng = np.random.default_rng(args.seed + 3)
    plans = [
        dataclasses.replace(
            dataclasses.replace(
                plan,
                query_positions=tuple(v + val_offset for v in plan.query_positions),
                support_positions=tuple(v + val_offset for v in plan.support_positions),
            ),
            background_positions=tuple(sample_bank_positions(
                bank, spec=bank_spec, n_windows=bank_spec.n_windows,
                exclude_labels=plan.candidates, rng=rng)),
        )
        for plan in plans
    ]

    label_text = F.normalize(label_text_matrix(encoder, index.label_ids, device).float(), dim=-1)
    dataset = PretrainDataset(index, combined, augment=False)
    collate = EpisodicCollate(MultiScaleCollate(fixed_patch_seconds=PATCH_SECONDS))

    truth, actual_pred, weight_pred, retrieve_pred = [], [], [], []
    stats = {"n_retrieved_useful": [], "n_bank_useful": [], "actual_margin": [],
             "weight_margin": [], "retrieve_margin": []}

    for plan in plans:
        batch = collate([dataset[p] for p in plan.flat_positions()])
        with torch.no_grad():
            encoded = encode_batch(encoder, batch, device)
            labels = batch["labels"].to(device)
            query_i, support_i, background_i = episode_row_roles(plan)
            binding = episode_binding(plan, len(labels)).to(device)
            query = live_sensor_rows(encoded, batch, labels=labels,
                                     enrolled_candidate=binding, select=query_i)
            memory = live_sensor_rows(encoded, batch, labels=labels,
                                      enrolled_candidate=binding,
                                      select=torch.cat([support_i, background_i]))
            candidate_text = label_text[torch.tensor(plan.candidates, device=device)]
            result = engine(query.rows, memory.rows, candidate_text, label_text,
                            generator=torch.Generator().manual_seed(0))

            # Per-window model prediction, exactly as the trainer scores it.
            mass = result["logits"].new_zeros((len(labels), len(plan.candidates)))
            mass.index_add_(0, query.window, result["logits"])
            per_window = mass[query_i.to(device)]

            row_text = evidence_label_tokens(memory.rows, candidate_text, label_text)
            bank_evidence = (row_text @ candidate_text.T).clamp_min(0)      # (M, C)

        for local, slot in enumerate(plan.query_slot):
            truth.append(plan.candidates[slot])
            actual_pred.append(plan.candidates[int(per_window[local].argmax())])
            rows_for_window = (query.window == query_i[local].to(device)).nonzero().flatten()
            if not len(rows_for_window):
                weight_pred.append(actual_pred[-1]); retrieve_pred.append(actual_pred[-1]); continue
            retrieved = result["selected"][rows_for_window].reshape(-1).unique()

            got = oracle_margins(bank_evidence[retrieved], slot)
            whole = oracle_margins(bank_evidence, slot)
            best_retrieved = int((bank_evidence[retrieved][:, slot]
                                  - bank_evidence[retrieved].clone().index_fill_(
                                      1, torch.tensor([slot], device=device), -9
                                  ).max(1).values).argmax())
            weight_pred.append(plan.candidates[int(bank_evidence[retrieved][best_retrieved].argmax())])
            best_bank = int((bank_evidence[:, slot] - bank_evidence.clone().index_fill_(
                1, torch.tensor([slot], device=device), -9).max(1).values).argmax())
            retrieve_pred.append(plan.candidates[int(bank_evidence[best_bank].argmax())])

            stats["n_retrieved_useful"].append(float((bank_evidence[retrieved][:, slot]
                                                      > bank_evidence[retrieved].clone().index_fill_(
                                                          1, torch.tensor([slot], device=device), -9
                                                      ).max(1).values).float().mean()))
            stats["n_bank_useful"].append(float((bank_evidence[:, slot]
                                                 > bank_evidence.clone().index_fill_(
                                                     1, torch.tensor([slot], device=device), -9
                                                 ).max(1).values).float().mean()))
            stats["weight_margin"].append(got["best_margin"])
            stats["retrieve_margin"].append(whole["best_margin"])

    print(f"\n{'=' * 78}")
    print(f"k=0 BOTTLENECK WATERFALL — {len(truth)} query windows from {len(plans)} episodes")
    print(f"{'=' * 78}")
    print(f"  {'ACTUAL (model retrieval + model weights)':52s} macro-F1 {macro_f1(truth, actual_pred):.4f}")
    print(f"  {'ORACLE-WEIGHT (model top-k, best row in it)':52s} macro-F1 {macro_f1(truth, weight_pred):.4f}")
    print(f"  {'ORACLE-RETRIEVE (whole 512-window bank, best row)':52s} macro-F1 {macro_f1(truth, retrieve_pred):.4f}")
    print(f"\n  useful rows (this row alone picks the right answer):")
    print(f"    inside the model's retrieved top-k : {np.mean(stats['n_retrieved_useful']) * 100:5.2f}%")
    print(f"    inside the whole bank              : {np.mean(stats['n_bank_useful']) * 100:5.2f}%")
    print(f"  best achievable margin  in top-k {np.mean(stats['weight_margin']):+.4f}"
          f"   in whole bank {np.mean(stats['retrieve_margin']):+.4f}")
    out = args.run / "bottleneck.json"
    out.write_text(json.dumps({
        "actual": macro_f1(truth, actual_pred),
        "oracle_weight": macro_f1(truth, weight_pred),
        "oracle_retrieve": macro_f1(truth, retrieve_pred),
        "useful_in_topk": float(np.mean(stats["n_retrieved_useful"])),
        "useful_in_bank": float(np.mean(stats["n_bank_useful"])),
    }, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
