"""Autopsy of a trained Phase-B checkpoint: where does the representation go wrong?

The first end-to-end run peaked at step 1750 and the encoder's query representation fell from 24
effective dimensions to about 9 inside the first 300 steps. That is a symptom with several possible
causes, and they call for different fixes:

  A. THE TRUNK DESTROYS IT.       Rank is healthy after the filterbank/fold and collapses inside the
                                  temporal layers -> the trunk is the thing to regularize.
  B. THE FRONT END NEVER HAD IT.  Rank is already low at the fold -> the filterbank normalization or
                                  the fold is the bottleneck and no trunk fix will help.
  C. LOW RANK IS FINE.            Rank falls but retrieval quality RISES -> collapse is the model
                                  discarding nuisance variation, and rank is the wrong alarm.
  D. THE OBJECTIVE ERODES IT.     Rank and retrieval quality both fall below the RANDOM-INIT floor
                                  -> episodic cross-entropy is actively harmful to the encoder.

Distinguishing them needs the same measurements at every stage, on the same data, for three
encoders: random init (the floor), the best checkpoint, and the last checkpoint.

Rank alone cannot settle it, so every stage is also scored for what retrieval actually needs:

  * ``retrieval_precision`` — of the top-64 rows a query pulls from a held-out bank, what fraction
    carry the query's own label. This is the quantity the vote consumes, so it is the honest
    measure of representation quality for this model.
  * ``knn_accuracy``        — 1-NN label agreement, the same signal without the top-k structure.
  * ``between_over_within`` — ratio of between-label to within-label distance. Rank can fall while
    this RISES, which is exactly case C.

Run:  python -m training.evidence.phase_b_autopsy --run <dir> [--windows 1500]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.tokenizer.eval_transfer import build_encoder
from training.tokenizer.pretrain_data import (
    PATCH_SECONDS, TRAIN_DATASETS, CorpusIndex, MultiScaleCollate, PretrainDataset,
)
from training.tokenizer.pretrain_episodic import _random_encoder, encode_batch


def effective_rank(x: torch.Tensor) -> float:
    """exp(entropy of the normalized eigenspectrum) — dimensions actually carrying variance."""
    x = x.float()
    x = x - x.mean(0, keepdim=True)
    if len(x) < 2:
        return 0.0
    eigenvalues = torch.linalg.svdvals(x) ** 2
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    p = (eigenvalues / total).clamp_min(1e-12)
    return float(torch.exp(-(p * p.log()).sum()))


def separation(x: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Retrieval-shaped quality of a representation, label-aware."""
    x = F.normalize(x.float(), dim=-1)
    similarity = x @ x.t()
    similarity.fill_diagonal_(-2.0)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(len(x), dtype=torch.bool, device=x.device)

    nearest = similarity.argmax(dim=1)
    knn = float((labels[nearest] == labels).float().mean())

    k = min(64, len(x) - 1)
    top = similarity.topk(k, dim=1).indices
    precision = float((labels[top] == labels.unsqueeze(1)).float().mean())

    within = float(similarity[same & ~eye].mean())
    between = float(similarity[~same].mean())
    return {"knn_accuracy": knn, "retrieval_precision": precision,
            "within": within, "between": between, "gap": within - between}


@torch.no_grad()
def stage_activations(encoder, batch, device) -> dict[str, torch.Tensor]:
    """One row per (window, patch, sensor) at each stage, plus a per-window mean."""
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def grab(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(value):
                captured[name] = value.detach()
        return hook

    handles.append(encoder.sensor_fold.register_forward_hook(grab("1_sensor_fold")))
    handles.append(encoder.descriptor_proj.register_forward_hook(grab("2_descriptor_proj")))
    for i, block in enumerate(encoder.transformer.attn):
        handles.append(block.register_forward_hook(grab(f"3_trunk_attn{i}")))
    handles.append(encoder.transformer.register_forward_hook(grab("4_trunk_out")))
    out = encode_batch(encoder, batch, device)
    for handle in handles:
        handle.remove()
    captured["5_retrieval_rows"] = out["retrieval_tokens"].detach()
    return captured, out


def flatten_rows(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(B,P,S,d) -> (R,d) keeping only live (patch, sensor) slots; other shapes pass through."""
    if value.dim() == 4:
        return value[valid]
    if value.dim() == 3:                                   # (B*S, P, d) from inside the trunk
        return value.reshape(-1, value.shape[-1])
    return value.reshape(-1, value.shape[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--datasets", nargs="+", default=list(TRAIN_DATASETS))
    args = parser.parse_args()
    device = torch.device(args.device)

    index = CorpusIndex(seed=20260818, datasets=tuple(args.datasets))
    dataset = PretrainDataset(index, list(index.val), augment=False)
    collate = MultiScaleCollate(fixed_patch_seconds=PATCH_SECONDS)
    rng = np.random.default_rng(0)
    picks = rng.choice(len(index.val), size=min(args.windows, len(index.val)), replace=False)
    labels_all = torch.tensor([int(index.val[int(p)].label_id) for p in picks])

    encoders = {}
    random_encoder, _ = _random_encoder(device)
    encoders["random_init"] = random_encoder.eval()
    for name in ("best", "last"):
        path = args.run / f"{name}.pt"
        if path.exists():
            # Our own trainer wrote these and they carry a config dict alongside tensors, so
            # weights_only=False is required (same as build_encoder's own loader).
            blob = torch.load(path, map_location="cpu", weights_only=False)
            encoders[f"{name}(step {blob['step']})"] = build_encoder(
                blob, device, training=False).eval()

    # The filterbank normalization of the random encoder is unfitted, which would make its numbers
    # meaningless rather than a floor. Borrow the trained run's fitted statistics: the front end is
    # fixed physical filters, so this compares the same input scaling under different LEARNED
    # weights, which is the comparison the question needs.
    trained = next((e for n, e in encoders.items() if n != "random_init"), None)
    if trained is not None:
        random_encoder.filterbank.load_state_dict(trained.filterbank.state_dict())

    report: dict[str, dict] = {}
    chunk = 128
    for name, encoder in encoders.items():
        stages: dict[str, list[torch.Tensor]] = {}
        row_labels: list[torch.Tensor] = []
        window_rows: dict[str, list[torch.Tensor]] = {}
        for start in range(0, len(picks), chunk):
            batch_positions = picks[start:start + chunk]
            batch = collate([dataset[int(p)] for p in batch_positions])
            captured, out = stage_activations(encoder, batch, device)
            valid = (batch["patch_padding_mask"].to(device).unsqueeze(2)
                     & out["sensor_present"].unsqueeze(1))
            window_index = torch.nonzero(valid, as_tuple=True)[0]
            labels_chunk = torch.tensor(
                [int(index.val[int(p)].label_id) for p in batch_positions], device=device,
            )
            row_labels.append(labels_chunk[window_index].cpu())
            for stage, value in captured.items():
                rows = flatten_rows(value, valid)
                stages.setdefault(stage, []).append(rows.float().cpu())
                if value.dim() == 4:                       # per-window mean for a window-level view
                    weights = valid.unsqueeze(-1).float()
                    pooled = (value.float() * weights).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1)
                    window_rows.setdefault(stage, []).append(pooled.cpu())

        labels_rows = torch.cat(row_labels)
        entry = {}
        for stage in sorted(stages):
            rows = torch.cat(stages[stage])
            record = {"dim": rows.shape[-1], "rows": len(rows),
                      "effective_rank": effective_rank(rows[:4000])}
            if len(rows) == len(labels_rows):
                sample = torch.randperm(len(rows))[:3000]
                record.update(separation(rows[sample], labels_rows[sample]))
            entry[stage] = record
        report[name] = entry

    print(f"\n{'=' * 108}")
    print("PER-STAGE REPRESENTATION AUTOPSY — rows are (window, patch, sensor) retrieval units")
    print(f"{'=' * 108}")
    stage_names = sorted({s for e in report.values() for s in e})
    for stage in stage_names:
        print(f"\n{stage}")
        header = (f"  {'encoder':22s} {'dim':>5s} {'eff.rank':>9s} {'rank%':>7s} "
                  f"{'knn':>7s} {'ret.prec':>9s} {'gap':>8s}")
        print(header)
        for name, entry in report.items():
            row = entry.get(stage)
            if row is None:
                continue
            share = row["effective_rank"] / row["dim"] * 100
            print(f"  {name:22s} {row['dim']:>5d} {row['effective_rank']:>9.2f} {share:>6.1f}% "
                  f"{row.get('knn_accuracy', float('nan')):>7.3f} "
                  f"{row.get('retrieval_precision', float('nan')):>9.3f} "
                  f"{row.get('gap', float('nan')):>8.3f}")

    out_path = args.run / "autopsy.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
