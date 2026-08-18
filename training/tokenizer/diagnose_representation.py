"""Phase-A representation diagnostics: WHERE activity information lives, and WHAT the trunk encodes.

Two questions the aggregate transfer score cannot answer.

D1 — DEPTH. `retrieval_tokens` (one sensor-isolated temporal layer) beat the full six-layer token by
+0.071 development transfer on the old-good checkpoint, and by -0.001 at random init. The gap is
therefore CREATED BY TRAINING: something in the objective makes the deeper layers worse for
retrieval than the shallow one. Probing every layer localises where that happens.

D2 — SELECTIVITY. A representation can be "good" by encoding what activity is happening, or it can
look busy while encoding which device recorded it. HALO's readouts are near-identical
(prototype 48.47 vs ridge 48.85 at k=4) where LiMU-BERT's differ by 8.9 points, which is the
signature of diffuse class structure rather than structure the wrong readout is missing. The
existing provenance probe measured nearest-neighbour dataset purity 0.4211 against 0.0833 chance.
So for every layer we probe activity AND the two nuisance factors (source stream, subject) with the
same ridge probe, and report the margin. A trunk whose stream-probe rises while its activity-probe
falls is spending depth on acquisition identity.

Both probes are deliberately the SAME estimator (multiclass ridge, closed form) so that
activity/stream/subject numbers are commensurable. Activity uses a subject-disjoint split; the
nuisance probes use a random split because subject identity is the thing being predicted.

    python -m training.tokenizer.diagnose_representation \
        --checkpoint training/tokenizer/outputs/phase_a_a_clean_20260818/best.pt \
        --label clean_7500
    python -m training.tokenizer.diagnose_representation --random-init --label step0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.scripts.eda.grid_io import discover_grids
from model.tokenizer.encoder import SetTokenizerEncoder
from training.tokenizer.eval_transfer import (
    PHASE_A_SELECTION_STREAMS,
    SEED,
    _selection_subsample,
    build_encoder,
    subject_holdout,
)
from training.tokenizer.pretrain_data import (
    stream_channel_descriptions,
    _stream_gravity_state,
)

MAX_WINDOWS = 4_000
RIDGE_LAMBDA = 1.0


def _ridge_probe(train_x, train_y, test_x, test_y, n_classes: int) -> float:
    """Balanced accuracy of a closed-form multiclass ridge probe on frozen features.

    Dual form: the feature dimension (256) and the sample count are both small here, but the dual
    keeps it correct if a caller ever probes a wider representation than it has samples.
    """
    if len(train_x) == 0 or len(test_x) == 0:
        return float("nan")
    x = F.normalize(train_x.float(), dim=-1)
    q = F.normalize(test_x.float(), dim=-1)
    target = F.one_hot(train_y, n_classes).float()
    gram = x @ x.T
    alpha = torch.linalg.solve(
        gram + RIDGE_LAMBDA * torch.eye(len(x), device=x.device, dtype=x.dtype), target
    )
    predicted = (q @ (x.T @ alpha)).argmax(1)
    per_class = [
        float((predicted[test_y == c] == c).float().mean())
        for c in range(n_classes) if bool((test_y == c).any())
    ]
    return float(np.mean(per_class)) if per_class else float("nan")


@torch.no_grad()
def _encode_at_depth(enc, ref, device, data, lengths, n_windows: int, depth: int | None):
    """Encode with the trunk TRUNCATED to `depth` layers (None = full trunk).

    Truncating and reusing the ordinary encode path is preferred over forward hooks: hooks fire per
    batch and would need window-ownership bookkeeping to reassemble, whereas truncation exercises
    exactly the tested pooling/export code at every depth, so the depths stay commensurable.
    """
    from training.tokenizer.eval_transfer import encode_dataset_detailed

    original = enc.transformer.layers
    compiled = getattr(enc, "_compiled_transformer_forward", None)
    if depth is not None:
        enc.transformer.layers = original[:depth]
        enc._compiled_transformer_forward = None      # shape/'depth' change invalidates the graph
    try:
        encoded = encode_dataset_detailed(
            enc, data, stream_channel_descriptions(ref.dataset, ref.stream), device,
            ref.rate_hz, _stream_gravity_state(ref.dataset, ref.stream), channel_mask=ref.mask,
            dataset=ref.dataset, stream=ref.stream, lengths=lengths,
            eval_patching="checkpoint", export_sensor_rows=True,
        )
    finally:
        enc.transformer.layers = original
        enc._compiled_transformer_forward = compiled

    rows = encoded["sensor_Z"].float()
    owners = encoded["sensor_window"].long().to(rows.device)
    acc = rows.new_zeros((n_windows, rows.shape[1]))
    count = rows.new_zeros((n_windows, 1))
    acc.index_add_(0, owners, rows)
    count.index_add_(0, owners, rows.new_ones((len(rows), 1)))
    return (acc / count.clamp_min(1.0)).cpu(), encoded["pooled"].float().cpu()


@torch.no_grad()
def layer_embeddings(enc, ref, device, max_windows: int):
    """Per-window embedding at every trunk depth, plus the deployed retrieval row and pooled vector.

    `retrieval_row` is what Phase B stores (one sensor-isolated temporal layer, pre-conditioning).
    `depth{d}` pools the sensor rows exported after d full dual-branch layers. `pooled` is the
    session vector nothing downstream consumes, included to show what the objective optimises.
    """
    labels_all = np.asarray(ref.labels)
    subjects_all = np.asarray(ref.subjects)
    take = _selection_subsample(labels_all, subjects_all, max_windows)
    labels, subjects = labels_all[take], subjects_all[take]
    data = ref.load_data()[take]
    lengths = ref.load_lengths()[take]
    n = len(labels)

    out: dict[str, torch.Tensor] = {}
    was_isolated = enc.use_sensor_isolated_retrieval
    enc.use_sensor_isolated_retrieval = True
    out["retrieval_row"], _ = _encode_at_depth(enc, ref, device, data, lengths, n, None)
    enc.use_sensor_isolated_retrieval = False          # export post-trunk rows for the depth sweep
    try:
        for depth in range(1, len(enc.transformer.layers) + 1):
            rowz, pooled = _encode_at_depth(enc, ref, device, data, lengths, n, depth)
            out[f"depth{depth}"] = rowz
            if depth == len(enc.transformer.layers):
                out["pooled"] = pooled
    finally:
        enc.use_sensor_isolated_retrieval = was_isolated
    return out, labels, subjects


def diagnose(enc, device, max_windows: int) -> dict:
    refs = {(r.dataset, r.stream): r for r in discover_grids("native")}
    results: dict[str, dict[str, list]] = {}
    for dataset, stream, scored in PHASE_A_SELECTION_STREAMS:
        if not scored:                      # diagnose on the clean-label cohorts only
            continue
        ref = refs.get((dataset, stream))
        if ref is None:
            continue
        embeddings, labels, subjects = layer_embeddings(enc, ref, device, max_windows)

        hold = subject_holdout(subjects, dataset)
        is_test = np.asarray([s in hold for s in subjects])
        activity_names = sorted(set(labels.tolist()))
        activity_id = torch.tensor([activity_names.index(v) for v in labels])
        subject_names = sorted(set(subjects.tolist()))
        subject_id = torch.tensor([subject_names.index(v) for v in subjects])

        # Nuisance probes need a split that is not subject-disjoint (subject IS the target), so use
        # a deterministic random half. Activity keeps the subject-disjoint split.
        rng = np.random.default_rng(SEED)
        random_test = rng.random(len(labels)) < 0.5

        for name, z in embeddings.items():
            row = results.setdefault(name, {"activity": [], "subject": []})
            row["activity"].append(_ridge_probe(
                z[~is_test], activity_id[~is_test], z[is_test], activity_id[is_test],
                len(activity_names),
            ))
            row["subject"].append(_ridge_probe(
                z[~random_test], subject_id[~random_test], z[random_test], subject_id[random_test],
                len(subject_names),
            ))
    return {
        name: {k: float(np.nanmean(v)) for k, v in row.items()}
        for name, row in results.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-windows", type=int, default=MAX_WINDOWS)
    parser.add_argument("--out", type=Path,
                        default=Path("training/tokenizer/outputs/representation_diagnostics"))
    args = parser.parse_args()
    device = torch.device(args.device)

    if args.random_init:
        torch.manual_seed(SEED)
        enc = SetTokenizerEncoder(
            d_model=256, num_layers=6, num_heads=8, dim_feedforward=1024, dropout=0.1,
            frontend="fixed", text_conditioning="factored", token_granularity="sensor",
            use_sensor_bias_conditioning=False, use_sensor_isolated_retrieval=True,
        ).to(device).eval()
    else:
        if args.checkpoint is None:
            parser.error("pass --checkpoint or --random-init")
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        enc = build_encoder(ck, device)
        enc.use_sensor_isolated_retrieval = True

    report = diagnose(enc, device, args.max_windows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.label}.json").write_text(json.dumps(report, indent=2))

    print(f"\n{args.label}: ridge-probe balanced accuracy, mean over the 3 scored dev cohorts")
    print(f"{'representation':18} {'activity':>9} {'subject':>9} {'act-subj':>9}")
    print("-" * 48)
    for name, row in report.items():
        margin = row["activity"] - row["subject"]
        print(f"{name:18} {row['activity']:9.4f} {row['subject']:9.4f} {margin:+9.4f}")


if __name__ == "__main__":
    main()
