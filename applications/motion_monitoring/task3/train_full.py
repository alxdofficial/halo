"""Fit the Task-3 recurrence metric and fix its operating point a priori."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.data.manifests import read_cohort_manifest
from applications.motion_monitoring.evaluation_manifests import (
    Task3EvaluationUnit,
    read_task_manifest,
)
from applications.motion_monitoring.representation_cache import CachedMotionSequenceDataset
from applications.motion_monitoring.task3.candidates import (
    assign_event_targets,
    pool_multiscale_candidates,
)
from applications.motion_monitoring.task3.data import (
    collate_motion_sequences,
    event_batch_from_recordings,
)
from applications.motion_monitoring.task3.losses import scoped_pair_indices, scoped_pair_loss
from applications.motion_monitoring.task3.sampling import (
    build_event_index,
    crop_around,
    sample_batch_instances,
    split_identities,
)
from applications.motion_monitoring.task3.model import RecurrentMotionMetric
from applications.motion_monitoring.task3.training import (
    initialize_affinity_threshold,
    train_step,
)


def _crop_sequence(sequence, start: float, end: float):
    centers = sequence.intervals_sec.mean(dim=1)
    selected = torch.nonzero((centers >= start) & (centers < end), as_tuple=False).flatten()
    if not len(selected):
        raise ValueError("Task-3 crop contains no representation patches")
    left, right = int(selected[0]), int(selected[-1]) + 1
    return replace(
        sequence,
        embeddings=sequence.embeddings[left:right],
        intervals_sec=sequence.intervals_sec[left:right],
        valid=sequence.valid[left:right],
        physical_features=sequence.physical_features[left:right],
        physical_feature_mask=sequence.physical_feature_mask[left:right],
    )


def _batch_for_units(
    units,
    recording_caches,
    representations,
    *,
    crop_seconds: float | None,
    rng: random.Random,
    device,
    durations_sec,
    candidate_stride_sec,
    crops: Sequence[tuple[float, float] | None] | None = None,
):
    recordings = []
    sequences = []
    exhaustive = []
    backgrounds = set()
    annotation_kinds = {unit.annotation_kind for unit in units}
    if len(annotation_kinds) != 1:
        raise ValueError("one Task-3 batch must use one annotation kind")
    for position, unit in enumerate(units):
        recording = recording_caches[unit.dataset][unit.cache_index]
        sequence = representations.get(unit.dataset, unit.recording_id, unit.stream_id)
        window = None if crops is None else crops[position]
        if window is not None:
            sequence = _crop_sequence(sequence, window[0], window[1])
        elif crop_seconds is not None and sequence.duration_sec > crop_seconds:
            low = float(sequence.intervals_sec[0, 0])
            high = float(sequence.intervals_sec[-1, 1]) - crop_seconds
            start = rng.uniform(low, high)
            sequence = _crop_sequence(sequence, start, start + crop_seconds)
        recordings.append(recording)
        sequences.append(sequence)
        exhaustive.append(unit.exhaustive)
        backgrounds.update(unit.background_labels)
    timeline = collate_motion_sequences(sequences).to(device)
    events = event_batch_from_recordings(
        recordings,
        sequences,
        annotation_kind=next(iter(annotation_kinds)),
        exhaustive=exhaustive,
        background_labels=frozenset(backgrounds),
    ).to(device)
    candidates = pool_multiscale_candidates(
        timeline.embeddings,
        timeline.intervals_sec,
        timeline.valid,
        durations_sec=durations_sec,
        candidate_stride_sec=candidate_stride_sec,
    )
    targets = assign_event_targets(candidates, events, positive_iou=0.5)
    return candidates, targets


OPERATING_POINT_PROTOCOL = {
    "schema_version": 1,
    "false_edge_rate": 0.05,
    "holdout_identity_fraction": 0.25,
    "holdout_unit": "training identity",
    "rule": (
        "largest affinity threshold whose false-edge rate over held-out training "
        "identities does not exceed the budget"
    ),
}


def fix_operating_point(
    scores: np.ndarray, targets: np.ndarray, *, false_edge_rate: float
) -> dict[str, Any]:
    """Threshold from a declared false-edge budget on held-out training identities."""

    if not 0 < false_edge_rate < 1:
        raise ValueError("false-edge rate must lie in (0, 1)")
    if scores.shape != targets.shape or not len(scores):
        raise ValueError("scores and targets must be aligned and non-empty")
    negative = np.sort(scores[targets == 0])
    if not len(negative):
        raise ValueError("the hold-out produced no negative pairs")
    allowed = int(np.floor(false_edge_rate * len(negative)))
    # Scores are same-motion affinities, so an edge exists when score >= threshold.
    # To admit at most ``allowed`` negatives the threshold must sit just above the
    # (allowed+1)-th largest negative; ties are resolved upwards so the budget is
    # never exceeded.
    if allowed >= len(negative):
        threshold = float(np.nextafter(negative[0], -np.inf))
    else:
        threshold = float(np.nextafter(negative[len(negative) - allowed - 1], np.inf))
        while int((negative >= threshold).sum()) > allowed:
            threshold = float(np.nextafter(threshold, np.inf))
    accepted_negative = int((scores[targets == 0] >= threshold).sum())
    accepted_positive = int((scores[targets == 1] >= threshold).sum())
    positives = int((targets == 1).sum())
    return {
        "threshold": threshold,
        "false_edge_rate_budget": float(false_edge_rate),
        "holdout_false_edge_rate": accepted_negative / len(negative),
        "holdout_recall": accepted_positive / positives if positives else float("nan"),
        "holdout_pairs": int(len(scores)),
        "holdout_positive_pairs": positives,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    cohort = read_cohort_manifest(args.cohort)
    train_manifest = read_task_manifest(args.train_manifest)
    representations = CachedMotionSequenceDataset(
        args.representations, manifest_fingerprint=cohort.fingerprint
    )
    datasets = sorted({str(row["dataset"]) for row in train_manifest.units})
    recording_caches = {dataset: open_cache(dataset) for dataset in datasets}
    first = representations[0]
    model = RecurrentMotionMetric(
        first.embeddings.shape[1], projection_dim=args.projection_dim
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    units = [Task3EvaluationUnit(**row) for row in train_manifest.units]
    # Section 2.3: positives are independent executions of one identity. Index the
    # events first and crop around them; a random crop per recording almost never
    # contains two executions of the same identity (measured 0.34 for Opportunity,
    # 1.0 for CrossFit clips) and the loss then sees no positive pairs at all.
    full_index = build_event_index(units, recording_caches)
    index_summary = full_index.summary()
    # Two data roles only: the operating point is fixed on held-out TRAINING
    # identities, never on a development split.
    index, holdout_index = split_identities(
        full_index,
        holdout_fraction=float(OPERATING_POINT_PROTOCOL["holdout_identity_fraction"]),
        seed=args.seed,
    )
    if not index.recurring_identities(minimum=2):
        raise ValueError(
            "the Task-3 train manifest contains no identity with two independent executions"
        )
    telemetry = []
    skipped = 0
    initialized = False
    step = 0
    while step < args.steps:
        try:
            drawn = sample_batch_instances(
                index,
                batch_size=args.batch_size,
                positives=args.positives_per_batch,
                rng=rng,
            )
            subset = []
            crops = []
            for instance, _role in drawn:
                unit = units[instance.unit_index]
                sequence = representations.get(unit.dataset, unit.recording_id, unit.stream_id)
                subset.append(unit)
                crops.append(
                    crop_around(
                        instance,
                        crop_seconds=args.crop_seconds,
                        timeline_start=float(sequence.intervals_sec[0, 0]),
                        timeline_end=float(sequence.intervals_sec[-1, 1]),
                        rng=rng,
                    )
                )
            if len({unit.annotation_kind for unit in subset}) != 1:
                raise ValueError("one Task-3 batch must use one annotation kind")
            candidates, targets = _batch_for_units(
                subset,
                recording_caches,
                representations,
                crop_seconds=args.crop_seconds,
                rng=rng,
                device=args.device,
                durations_sec=args.durations,
                candidate_stride_sec=args.candidate_stride,
                crops=crops,
            )
            positive, negative = scoped_pair_indices(candidates, targets)
            if not len(positive) or not len(negative):
                raise ValueError("batch lacks pair classes")
        except ValueError:
            skipped += 1
            if skipped > args.steps * 20:
                raise RuntimeError("Task-3 sampling cannot produce eligible batches")
            continue
        if not initialized:
            initialize_affinity_threshold(model, candidates, targets)
            initialized = True
        result = train_step(
            model, optimizer, candidates, targets, max_grad_norm=args.grad_clip
        )
        step += 1
        if step == 1 or step % args.telemetry_every == 0 or step == args.steps:
            telemetry.append({"step": step, "loss": result.loss, **result.telemetry})

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "feature_dim": first.embeddings.shape[1],
        "projection_dim": args.projection_dim,
        "model_state_dict": model.state_dict(),
        "cohort_fingerprint": cohort.fingerprint,
        "representation_provenance": representations.metadata["encoder_provenance"],
        "durations": tuple(args.durations),
        "candidate_stride": args.candidate_stride,
    }
    torch.save(checkpoint, args.output / "task3_head.pt")

    # Operating point on held-out TRAINING identities, never on natural test data.
    holdout_scores: list[float] = []
    holdout_targets: list[int] = []
    direct_scores: list[float] = []
    direct_targets: list[int] = []
    holdout_rng = random.Random(args.seed + 1)
    model.eval()
    with torch.no_grad():
        for _ in range(args.operating_point_batches):
            try:
                drawn = sample_batch_instances(
                    holdout_index,
                    batch_size=args.batch_size,
                    positives=args.positives_per_batch,
                    rng=holdout_rng,
                )
                subset, crops = [], []
                for instance, _role in drawn:
                    unit = units[instance.unit_index]
                    sequence = representations.get(
                        unit.dataset, unit.recording_id, unit.stream_id
                    )
                    subset.append(unit)
                    crops.append(
                        crop_around(
                            instance,
                            crop_seconds=args.crop_seconds,
                            timeline_start=float(sequence.intervals_sec[0, 0]),
                            timeline_end=float(sequence.intervals_sec[-1, 1]),
                            rng=holdout_rng,
                        )
                    )
                if len({unit.annotation_kind for unit in subset}) != 1:
                    continue
                candidates, targets = _batch_for_units(
                    subset,
                    recording_caches,
                    representations,
                    crop_seconds=args.crop_seconds,
                    rng=holdout_rng,
                    device=args.device,
                    durations_sec=args.durations,
                    candidate_stride_sec=args.candidate_stride,
                    crops=crops,
                )
                output = scoped_pair_loss(model, candidates, targets)
                holdout_scores.extend(
                    output.positive_logits.detach().cpu().reshape(-1).tolist()
                )
                holdout_targets.extend([1] * output.positive_logits.numel())
                holdout_scores.extend(
                    output.negative_logits.detach().cpu().reshape(-1).tolist()
                )
                holdout_targets.extend([0] * output.negative_logits.numel())
                # The untrained floor readout scores the same held-out pairs with
                # plain cosine, so it gets its own threshold under the same budget
                # rather than borrowing the learned metric's.
                positive, negative = scoped_pair_indices(candidates, targets)
                flat = candidates.embeddings.reshape(
                    -1, candidates.embeddings.shape[-1]
                )
                unit_norm = F.normalize(flat, dim=-1, eps=1e-8)
                for indices, label in ((positive, 1), (negative, 0)):
                    cosine = (
                        unit_norm[indices[:, 0]] * unit_norm[indices[:, 1]]
                    ).sum(-1)
                    direct_scores.extend(cosine.detach().cpu().tolist())
                    direct_targets.extend([label] * len(indices))
            except ValueError:
                continue
    budget = float(OPERATING_POINT_PROTOCOL["false_edge_rate"])
    unsupported = {"unsupported": "held-out identities produced no scorable pairs"}
    operating_point = {
        "learned_metric_recurrence": (
            fix_operating_point(
                np.asarray(holdout_scores),
                np.asarray(holdout_targets),
                false_edge_rate=budget,
            )
            if holdout_scores
            else dict(unsupported)
        ),
        "direct_cosine_recurrence": (
            fix_operating_point(
                np.asarray(direct_scores),
                np.asarray(direct_targets),
                false_edge_rate=budget,
            )
            if direct_scores
            else dict(unsupported)
        ),
    }
    checkpoint["operating_point"] = operating_point
    checkpoint["operating_point_protocol"] = dict(OPERATING_POINT_PROTOCOL)
    torch.save(checkpoint, args.output / "task3_head.pt")
    report = {
        "task": "task3",
        "status": "trained_and_operating_point_fixed",
        "cohort_fingerprint": cohort.fingerprint,
        "train_manifest_fingerprint": train_manifest.fingerprint,
        "representation_provenance": representations.metadata["encoder_provenance"],
        "train_units": len(train_manifest.units),
        "event_index": index_summary,
        "fit_identities": len(index.identities),
        "holdout_identities": len(holdout_index.identities),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "positives_per_batch": args.positives_per_batch,
        "skipped_batches": skipped,
        "telemetry": telemetry,
        "operating_point_protocol": dict(OPERATING_POINT_PROTOCOL),
        "operating_point": operating_point,
    }
    (args.output / "task3_training.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=root / "manifests/COHORT_TASK3_V2.json")
    parser.add_argument(
        "--train-manifest", type=Path, default=root / "manifests/TASK3_TRAIN_V2.json"
    )
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--positives-per-batch",
        type=int,
        default=2,
        help="independent executions of one identity guaranteed in every batch",
    )
    parser.add_argument("--operating-point-batches", type=int, default=64)
    parser.add_argument("--crop-seconds", type=float, default=120.0)
    # Derived from the retained training executions (Opportunity gestures and
    # assembled CrossFit repetitions): p5/p50/p95 = 1.5 / 2.83 / 4.98 s. The first
    # four scales cover that distribution densely; 10 and 16 extend upward so
    # longer evaluation events are not structurally unreachable. Measured reach at
    # IoU >= 0.5: training 99.7 %, OpenPack 79.8 %, OCA 98.7 %. The previous
    # default started at 2 s and reached only 71.7 % of OpenPack, whose median
    # fine action is 1.6 s.
    parser.add_argument(
        "--durations", type=float, nargs="+", default=(1.5, 2.5, 4, 6, 10, 16)
    )
    parser.add_argument("--candidate-stride", type=float, default=1.0)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--telemetry-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
