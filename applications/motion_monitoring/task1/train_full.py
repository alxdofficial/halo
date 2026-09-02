"""Fit and calibrate the Task-1 metric head on signed application manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.data.manifests import read_cohort_manifest
from applications.motion_monitoring.evaluation import fingerprint_protocol
from applications.motion_monitoring.evaluation_manifests import (
    Task1EvaluationUnit,
    read_task_manifest,
)
from applications.motion_monitoring.representation_cache import open_representations
from applications.motion_monitoring.task1.episodes import (
    EmbeddingSequence,
    collate_detection_episodes,
    episode_from_recordings,
    from_motion_sequence,
)
from applications.motion_monitoring.task1.full_evaluation import unit_matches
from applications.motion_monitoring.task1.model import DifferentiableSubsequenceMatcher
from applications.motion_monitoring.task1.training import train_step


def _crop_sequence(sequence: EmbeddingSequence, start: float, end: float) -> EmbeddingSequence:
    centers = sequence.intervals_sec.mean(dim=1)
    selected = torch.nonzero(
        (centers >= start) & (centers < end), as_tuple=False
    ).flatten()
    if not len(selected):
        raise ValueError("training crop contains no representation patches")
    left, right = int(selected[0]), int(selected[-1]) + 1
    return EmbeddingSequence(
        sequence.embeddings[left:right],
        sequence.intervals_sec[left:right],
        sequence.valid[left:right],
        metadata=sequence.metadata,
    )


def _training_episode(unit, recordings, representations, *, seconds: float, rng):
    reference_recording = recordings[unit.reference_cache_index]
    query_recording = recordings[unit.query_cache_index]
    reference = from_motion_sequence(
        representations.get(
            unit.dataset, unit.reference_recording_id, unit.reference_stream_id
        )
    )
    query = from_motion_sequence(
        representations.get(unit.dataset, unit.query_recording_id, unit.query_stream_id)
    )
    timeline_start = float(query.intervals_sec[0, 0])
    timeline_end = float(query.intervals_sec[-1, 1])
    width = min(seconds, timeline_end - timeline_start)
    if unit.target_intervals_sec:
        target_start, target_end = rng.choice(unit.target_intervals_sec)
        low = max(timeline_start, target_end - width)
        high = min(target_start, timeline_end - width)
        crop_start = low if high <= low else rng.uniform(low, high)
    else:
        high = max(timeline_start, timeline_end - width)
        crop_start = timeline_start if high <= timeline_start else rng.uniform(timeline_start, high)
    query = _crop_sequence(query, crop_start, crop_start + width)
    return episode_from_recordings(
        reference_recording,
        query_recording,
        reference,
        query,
        label=unit.label,
        reference_event_index=unit.reference_event_index,
        target_intervals_sec=unit.target_intervals_sec,
        reference_interval_sec=getattr(unit, "reference_interval_sec", None),
        # Training re-draws the grid-snap context side per sample; evaluation
        # paths leave this None for the deterministic draw.
        reference_rng=rng,
    )


def _pooled_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    tp = sum(row["true_positive_count"] for row in rows)
    fp = sum(row["false_positive_count"] for row in rows)
    fn = sum(row["false_negative_count"] for row in rows)
    hours = sum(row["query_hours"] for row in rows)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "false_alarms_per_hour": fp / max(hours, 1e-12),
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
    }


def _event_prefix(matches, targets_sec, *, iou_threshold: float):
    """Return exact greedy TP counts for every score-ranked detection prefix."""

    ranked = sorted(matches, key=lambda item: item.score)
    targets = torch.as_tensor(targets_sec, dtype=torch.float64)
    unmatched = set(range(len(targets)))
    true_positive = 0
    cumulative = []
    for match in ranked:
        best_target = None
        best_iou = 0.0
        for target_index in unmatched:
            target_start, target_end = targets[target_index].tolist()
            intersection = max(
                0.0,
                min(match.end_sec, target_end) - max(match.start_sec, target_start),
            )
            union = max(match.end_sec, target_end) - min(match.start_sec, target_start)
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_target = target_index
        if best_target is not None and best_iou >= iou_threshold:
            unmatched.remove(best_target)
            true_positive += 1
        cumulative.append(true_positive)
    return (
        np.asarray([match.score for match in ranked], dtype=np.float64),
        np.asarray(cumulative, dtype=np.int64),
        len(targets),
    )


def calibrate(
    manifest,
    recording_caches,
    representations,
    model,
    *,
    nms_iou: float = 0.3,
    match_iou: float = 0.5,
) -> dict[str, Any]:
    evaluated = []
    rejected = []
    all_scores: list[float] = []
    for index, raw in enumerate(manifest.units):
        unit = Task1EvaluationUnit(**raw)
        try:
            matches, episode = unit_matches(
                unit,
                recording_caches[unit.dataset],
                representations,
                score_threshold=float("inf"),
                model=model,
                nms_iou=nms_iou,
            )
        except ValueError as error:
            rejected.append({"unit": index, "reason": str(error)})
            continue
        duration_hours = float(
            episode.query.intervals_sec[-1, 1]
            - episode.query.intervals_sec[0, 0]
        ) / 3600.0
        evaluated.append(
            (
                *_event_prefix(matches, episode.targets_sec, iou_threshold=match_iou),
                duration_hours,
                unit.dataset,
            )
        )
        all_scores.extend(match.score for match in matches)
    if not evaluated or not all_scores:
        raise ValueError("development calibration has no eligible scored matches")
    unique = np.unique(np.asarray(all_scores, dtype=np.float64))
    if len(unique) > 256:
        candidates = np.unique(np.quantile(unique, np.linspace(0.0, 1.0, 256)))
    else:
        candidates = unique
    candidates = np.concatenate(
        ([np.nextafter(candidates[0], -np.inf)], candidates)
    )
    def rows_at(threshold: float) -> list[dict[str, Any]]:
        rows = []
        for scores, cumulative_tp, target_count, duration_hours, dataset in evaluated:
            detection_count = int(np.searchsorted(scores, threshold, side="right"))
            true_positive = (
                int(cumulative_tp[detection_count - 1]) if detection_count else 0
            )
            rows.append(
                {
                    "true_positive_count": float(true_positive),
                    "false_positive_count": float(detection_count - true_positive),
                    "false_negative_count": float(target_count - true_positive),
                    "query_hours": duration_hours,
                    "dataset": dataset,
                }
            )
        return rows

    best = None
    for threshold in candidates:
        rows = rows_at(float(threshold))
        metrics = _pooled_metrics(rows)
        key = (
            metrics["event_f1"],
            -metrics["false_alarms_per_hour"],
            -float(threshold),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    assert best is not None
    chosen_rows = rows_at(best[1])
    per_dataset = {
        dataset: _pooled_metrics([row for row in chosen_rows if row["dataset"] == dataset])
        for dataset in sorted({row["dataset"] for row in chosen_rows})
    }
    return {
        "threshold": best[1],
        "metrics": best[2],
        "per_dataset": per_dataset,
        "eligible_units": len(evaluated),
        "rejected_units": len(rejected),
        "rejections": rejected,
        "candidate_thresholds": len(candidates),
    }


def select_common_units(manifest, path: Path | None):
    if path is None:
        return manifest, None
    common = json.loads(path.read_text(encoding="utf-8"))
    if common["task_manifest_fingerprint"] != manifest.fingerprint:
        raise ValueError("common Task-1 units belong to another development manifest")
    selected = [int(index) for index in common["selected_unit_indices"]]
    if not selected:
        raise ValueError("common Task-1 development-unit intersection is empty")
    return (
        replace(manifest, units=tuple(manifest.units[index] for index in selected)),
        common,
    )


def fit_head(
    units: Sequence[Task1EvaluationUnit],
    recording_caches,
    representations,
    *,
    feature_dim: int,
    steps: int,
    batch_size: int,
    query_seconds: float,
    projection_dim: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    telemetry_every: int,
    seed: int,
    device: str,
) -> tuple[DifferentiableSubsequenceMatcher, list[dict[str, Any]], int]:
    """Train one head on ``units``; return (model, telemetry, episode rejections)."""

    if steps <= 0 or batch_size <= 0 or query_seconds <= 0:
        raise ValueError("steps, batch size, and query duration must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    model = DifferentiableSubsequenceMatcher(feature_dim, projection_dim=projection_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    groups: dict[tuple[bool, str], list[Task1EvaluationUnit]] = defaultdict(list)
    for unit in units:
        groups[(unit.target_present, unit.dataset)].append(unit)
    datasets_by_status = {
        status: sorted(dataset for present, dataset in groups if present == status)
        for status in (False, True)
    }
    if any(not datasets for datasets in datasets_by_status.values()):
        raise ValueError("Task-1 training requires present and absent source groups")
    telemetry = []
    rejection_count = 0
    for step in range(steps):
        episodes = []
        attempts = 0
        while len(episodes) < batch_size and attempts < batch_size * 20:
            status = bool(len(episodes) % 2)
            dataset = rng.choice(datasets_by_status[status])
            unit = rng.choice(groups[(status, dataset)])
            attempts += 1
            try:
                episodes.append(
                    _training_episode(
                        unit,
                        recording_caches[unit.dataset],
                        representations,
                        seconds=query_seconds,
                        rng=rng,
                    )
                )
            except ValueError:
                rejection_count += 1
        if len(episodes) < batch_size:
            raise RuntimeError("could not assemble a complete eligible Task-1 batch")
        batch = collate_detection_episodes(episodes).to(device)
        result = train_step(model, batch, optimizer, grad_clip=grad_clip)
        if step == 0 or (step + 1) % telemetry_every == 0 or step + 1 == steps:
            telemetry.append({"step": step + 1, "loss": result.loss, **result.telemetry})
    model.eval()
    return model, telemetry, rejection_count


def train(args: argparse.Namespace) -> dict[str, Any]:
    cohort = read_cohort_manifest(args.cohort)
    train_manifest = read_task_manifest(args.train_manifest)
    development_manifest = read_task_manifest(args.development_manifest)
    development_manifest, development_common = select_common_units(
        development_manifest, args.common_development_units
    )
    representations = open_representations(args.representations, cohort=cohort)
    datasets = sorted({str(row["dataset"]) for row in train_manifest.units})
    recording_caches = {dataset: open_cache(dataset) for dataset in datasets}
    first = representations.get(
        datasets[0],
        str(train_manifest.units[0]["query_recording_id"]),
        str(train_manifest.units[0]["query_stream_id"]),
    )
    units = [Task1EvaluationUnit(**row) for row in train_manifest.units]
    model, telemetry, rejection_count = fit_head(
        units,
        recording_caches,
        representations,
        feature_dim=first.embeddings.shape[1],
        steps=args.steps,
        batch_size=args.batch_size,
        query_seconds=args.query_seconds,
        projection_dim=args.projection_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        telemetry_every=args.telemetry_every,
        seed=args.seed,
        device=args.device,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "feature_dim": first.embeddings.shape[1],
        "projection_dim": args.projection_dim,
        "model_state_dict": model.state_dict(),
        "cohort_fingerprint": cohort.fingerprint,
        "representation_provenance": representations.metadata["encoder_provenance"],
        "development_common_unit_count": len(development_manifest.units),
        "development_common_unit_fingerprint": (
            None if development_common is None else fingerprint_protocol(development_common)
        ),
        "config": vars(args),
    }
    torch.save(checkpoint, args.output / "task1_head.pt")

    development_caches = {
        dataset: open_cache(dataset)
        for dataset in sorted({str(row["dataset"]) for row in development_manifest.units})
    }
    direct = calibrate(
        development_manifest, development_caches, representations, None
    )
    learned = calibrate(
        development_manifest, development_caches, representations, model
    )
    # Natural-development gate (spec section C.4): the learned head only counts
    # if it beats the untrained direct floor on natural, per-execution data.
    # Training data (synthetic or otherwise) never touches this comparison.
    gate = {
        "direct_event_f1": direct["metrics"]["event_f1"],
        "learned_event_f1": learned["metrics"]["event_f1"],
        "delta_event_f1": learned["metrics"]["event_f1"] - direct["metrics"]["event_f1"],
        "passed": bool(learned["metrics"]["event_f1"] > direct["metrics"]["event_f1"]),
        "per_dataset_delta_event_f1": {
            dataset: learned["per_dataset"][dataset]["event_f1"]
            - direct["per_dataset"][dataset]["event_f1"]
            for dataset in sorted(direct["per_dataset"])
            if dataset in learned["per_dataset"]
        },
        "development_datasets": sorted(development_caches),
        "training_datasets": datasets,
    }
    report = {
        "task": "task1",
        "status": "trained_and_development_calibrated",
        "cohort_fingerprint": cohort.fingerprint,
        "representation_provenance": representations.metadata["encoder_provenance"],
        "representation_roots": [str(path) for path in args.representations],
        "train_units": len(units),
        "train_episode_rejections": rejection_count,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "telemetry": telemetry,
        "direct": direct,
        "learned": learned,
        "natural_dev_gate": gate,
    }
    (args.output / "task1_training.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=root / "manifests/COHORT_V1.json")
    parser.add_argument(
        "--train-manifest", type=Path, default=root / "manifests/TASK1_TRAIN_V1.json"
    )
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=root / "manifests/TASK1_DEVELOPMENT_V1.json",
    )
    parser.add_argument("--common-development-units", type=Path)
    parser.add_argument(
        "--representations",
        type=Path,
        nargs="+",
        required=True,
        help="one or more representation caches (e.g. natural + synthetic) for one encoder",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query-seconds", type=float, default=60.0)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--telemetry-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
