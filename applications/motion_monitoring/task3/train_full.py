"""Fit and calibrate the Task-3 recurrence metric on signed manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from applications.motion_monitoring.task3.losses import scoped_pair_indices
from applications.motion_monitoring.task3.metrics import binary_auroc
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
):
    recordings = []
    sequences = []
    exhaustive = []
    backgrounds = set()
    annotation_kinds = {unit.annotation_kind for unit in units}
    if len(annotation_kinds) != 1:
        raise ValueError("one Task-3 batch must use one annotation kind")
    for unit in units:
        recording = recording_caches[unit.dataset][unit.cache_index]
        sequence = representations.get(unit.dataset, unit.recording_id, unit.stream_id)
        if crop_seconds is not None and sequence.duration_sec > crop_seconds:
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


def _balanced_threshold(scores: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    if not targets.any() or not (~targets).any():
        raise ValueError("threshold calibration requires both pair classes")
    unique = np.unique(scores)
    candidates = np.unique(np.quantile(unique, np.linspace(0.0, 1.0, min(512, len(unique)))))
    candidates = np.concatenate(
        (candidates, [np.nextafter(candidates[-1], np.inf)])
    )
    best = None
    for threshold in candidates:
        predicted = scores >= threshold
        sensitivity = float(predicted[targets].mean())
        specificity = float((~predicted[~targets]).mean())
        balanced = 0.5 * (sensitivity + specificity)
        key = (balanced, specificity, float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), sensitivity, specificity)
    assert best is not None
    return {
        "threshold": best[1],
        "balanced_accuracy": best[0][0],
        "sensitivity": best[2],
        "specificity": best[3],
        "auroc": binary_auroc(torch.from_numpy(scores), torch.from_numpy(targets)),
        "positive_pairs": int(targets.sum()),
        "negative_pairs": int((~targets).sum()),
    }


@torch.no_grad()
def calibrate(
    manifest,
    recording_caches,
    representations,
    model,
    *,
    batch_size,
    durations_sec,
    candidate_stride_sec,
    seed,
    device,
):
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[Task3EvaluationUnit]] = defaultdict(list)
    for row in manifest.units:
        unit = Task3EvaluationUnit(**row)
        grouped[(unit.dataset, unit.annotation_kind)].append(unit)
    direct_positive = []
    direct_negative = []
    learned_positive = []
    learned_negative = []
    rejected = 0
    for units in grouped.values():
        for start in range(0, len(units), batch_size):
            subset = units[start : start + batch_size]
            try:
                candidates, targets = _batch_for_units(
                    subset,
                    recording_caches,
                    representations,
                    crop_seconds=None,
                    rng=rng,
                    device=device,
                    durations_sec=durations_sec,
                    candidate_stride_sec=candidate_stride_sec,
                )
                positive, negative = scoped_pair_indices(candidates, targets)
                if not len(positive) or not len(negative):
                    rejected += len(subset)
                    continue
            except ValueError:
                rejected += len(subset)
                continue
            flat = candidates.embeddings.reshape(-1, candidates.embeddings.shape[-1])
            raw = F.normalize(flat, dim=-1, eps=1e-8)
            learned = model.embed(flat)
            for indices, direct_rows, learned_rows in (
                (positive, direct_positive, learned_positive),
                (negative, direct_negative, learned_negative),
            ):
                direct_rows.extend(
                    (raw[indices[:, 0]] * raw[indices[:, 1]]).sum(-1).cpu().tolist()
                )
                learned_rows.extend(
                    (learned[indices[:, 0]] * learned[indices[:, 1]]).sum(-1).cpu().tolist()
                )
    def fit(positive, negative):
        scores = np.asarray([*positive, *negative], dtype=np.float32)
        labels = np.asarray([True] * len(positive) + [False] * len(negative))
        return _balanced_threshold(scores, labels)
    return {
        "direct": fit(direct_positive, direct_negative),
        "learned": fit(learned_positive, learned_negative),
        "rejected_units": rejected,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    cohort = read_cohort_manifest(args.cohort)
    train_manifest = read_task_manifest(args.train_manifest)
    development_manifest = read_task_manifest(args.development_manifest)
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
    grouped: dict[tuple[str, str], list[Task3EvaluationUnit]] = defaultdict(list)
    for row in train_manifest.units:
        unit = Task3EvaluationUnit(**row)
        grouped[(unit.dataset, unit.annotation_kind)].append(unit)
    eligible_groups = [units for units in grouped.values() if len(units) >= args.batch_size]
    if not eligible_groups:
        raise ValueError("no Task-3 source has enough units for one batch")
    telemetry = []
    skipped = 0
    initialized = False
    step = 0
    while step < args.steps:
        units = rng.choice(eligible_groups)
        subset = rng.sample(units, args.batch_size)
        try:
            candidates, targets = _batch_for_units(
                subset,
                recording_caches,
                representations,
                crop_seconds=args.crop_seconds,
                rng=rng,
                device=args.device,
                durations_sec=args.durations,
                candidate_stride_sec=args.candidate_stride,
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

    development_caches = {
        dataset: open_cache(dataset)
        for dataset in sorted({str(row["dataset"]) for row in development_manifest.units})
    }
    calibration = calibrate(
        development_manifest,
        development_caches,
        representations,
        model,
        batch_size=args.batch_size,
        durations_sec=args.durations,
        candidate_stride_sec=args.candidate_stride,
        seed=args.seed,
        device=args.device,
    )
    report = {
        "task": "task3",
        "status": "trained_and_development_calibrated",
        "cohort_fingerprint": cohort.fingerprint,
        "representation_provenance": representations.metadata["encoder_provenance"],
        "train_units": len(train_manifest.units),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "skipped_batches": skipped,
        "telemetry": telemetry,
        "calibration": calibration,
    }
    (args.output / "task3_training.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=root / "manifests/COHORT_V1.json")
    parser.add_argument(
        "--train-manifest", type=Path, default=root / "manifests/TASK3_TRAIN_V1.json"
    )
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=root / "manifests/TASK3_DEVELOPMENT_V1.json",
    )
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-seconds", type=float, default=120.0)
    parser.add_argument("--durations", type=float, nargs="+", default=(2, 4, 8, 16, 32))
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
