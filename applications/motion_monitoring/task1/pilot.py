"""Bounded Task-1 HALO/DTW pilot; diagnostic only, never a promoted result."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from applications.motion_monitoring.data.examples import crop_recording, open_cache
from applications.motion_monitoring.data.manifests import CohortManifest, read_cohort_manifest
from applications.motion_monitoring.representation_cache import CachedMotionSequenceDataset
from applications.motion_monitoring.sequence import MotionSequence
from applications.motion_monitoring.task1.episodes import (
    DetectionEpisode,
    EmbeddingSequence,
    collate_detection_episodes,
    episode_from_recordings,
)
from applications.motion_monitoring.task1.matcher import best_full_timeline_match
from applications.motion_monitoring.task1.model import DifferentiableSubsequenceMatcher
from applications.motion_monitoring.task1.training import train_step


def _embedding_slice(
    sequence: MotionSequence, start_sec: float, end_sec: float
) -> EmbeddingSequence:
    centers = sequence.intervals_sec.mean(dim=1)
    selected = torch.nonzero(
        (centers >= start_sec) & (centers < end_sec), as_tuple=False
    ).flatten()
    if not len(selected):
        raise ValueError("query crop contains no representation patches")
    start, stop = int(selected[0]), int(selected[-1]) + 1
    return EmbeddingSequence(
        sequence.embeddings[start:stop],
        sequence.intervals_sec[start:stop],
        sequence.valid[start:stop],
    )


def _crop_bounds(recording, event, duration_sec: float) -> tuple[float, float]:
    starts = [float(stream.timestamps_sec[0]) for stream in recording.streams]
    stops = [
        float(stream.timestamps_sec[-1]) + 1.0 / float(stream.nominal_rate_hz or 1.0)
        for stream in recording.streams
    ]
    timeline_start, timeline_stop = max(starts), min(stops)
    duration = min(duration_sec, timeline_stop - timeline_start)
    center = 0.5 * (event.start_sec + event.end_sec)
    start = max(timeline_start, min(center - duration / 2, timeline_stop - duration))
    stop = start + duration
    if event.start_sec < start or event.end_sec > stop:
        raise ValueError("positive event does not fit inside the requested query crop")
    return start, stop


def _negative_bounds(recording, label: str, duration_sec: float) -> tuple[float, float]:
    starts = [float(stream.timestamps_sec[0]) for stream in recording.streams]
    stops = [
        float(stream.timestamps_sec[-1]) + 1.0 / float(stream.nominal_rate_hz or 1.0)
        for stream in recording.streams
    ]
    timeline_start, timeline_stop = max(starts), min(stops)
    duration = min(duration_sec, timeline_stop - timeline_start)
    target_events = [event for event in recording.events if event.label == label]
    for start in np.arange(timeline_start, timeline_stop - duration + 1e-9, duration / 2):
        stop = float(start + duration)
        if all(event.end_sec <= start or event.start_sec >= stop for event in target_events):
            return float(start), stop
    raise ValueError(f"recording has no {duration:g}s target-absent crop for {label!r}")


def _episodes_for_split(
    manifest: CohortManifest,
    representations: CachedMotionSequenceDataset,
    *,
    dataset: str,
    split: str,
    annotation_kind: str,
    query_seconds: float,
    max_labels: int,
    excluded_labels: set[str],
    seed: int,
) -> tuple[tuple[DetectionEpisode, ...], dict[str, object]]:
    cache = open_cache(dataset)
    entries = {
        entry.recording_id: entry
        for entry in manifest.entries_for(dataset=dataset, split=split)
        if (dataset, entry.recording_id, entry.stream_ids[0])
        in representations.index_by_key
    }
    events_by_label: dict[str, list[tuple[str, int]]] = defaultdict(list)
    recordings = {}
    for recording_id, entry in entries.items():
        recording = cache[entry.cache_index]
        recordings[recording_id] = recording
        for event_index, event in enumerate(recording.events):
            if (
                event.annotation_kind == annotation_kind
                and not bool(event.metadata.get("clipped_to_observed_sensor_span", False))
            ):
                events_by_label[event.label].append((recording_id, event_index))

    episodes: list[DetectionEpisode] = []
    selected_labels: list[str] = []
    rejected: list[dict[str, str]] = []
    ordered_labels = sorted(
        (label for label in events_by_label if label not in excluded_labels),
        key=lambda label: hashlib.sha256(
            f"{seed}:{split}:{label}".encode("utf-8")
        ).hexdigest(),
    )
    for label in ordered_labels:
        by_recording: dict[str, int] = {}
        for recording_id, event_index in events_by_label[label]:
            by_recording.setdefault(recording_id, event_index)
        by_subject: dict[str, list[str]] = defaultdict(list)
        for recording_id in by_recording:
            by_subject[recordings[recording_id].subject_id].append(recording_id)
        eligible_subjects = sorted(
            (subject for subject, ids in by_subject.items() if len(ids) >= 2),
            key=lambda subject: hashlib.sha256(
                f"{seed}:{split}:{label}:{subject}".encode("utf-8")
            ).hexdigest(),
        )
        if not eligible_subjects:
            continue
        reference_id, query_id = sorted(by_subject[eligible_subjects[0]])[:2]
        try:
            reference = recordings[reference_id]
            query = recordings[query_id]
            reference_index = by_recording[reference_id]
            query_event = query.events[by_recording[query_id]]
            reference_stream = entries[reference_id].stream_ids[0]
            query_stream = entries[query_id].stream_ids[0]
            reference_sequence = representations.get(
                dataset, reference_id, reference_stream
            )
            query_sequence = representations.get(dataset, query_id, query_stream)
            positive_start, positive_stop = _crop_bounds(
                query, query_event, query_seconds
            )
            negative_start, negative_stop = _negative_bounds(
                query, label, query_seconds
            )
            pair_episodes: list[DetectionEpisode] = []
            for kind, start, stop in (
                ("positive", positive_start, positive_stop),
                ("target_absent", negative_start, negative_stop),
            ):
                query_crop = crop_recording(
                    query,
                    start,
                    stop,
                    recording_suffix=f"pilot-{kind}",
                )
                episode = episode_from_recordings(
                    reference,
                    query_crop,
                    EmbeddingSequence(
                        reference_sequence.embeddings,
                        reference_sequence.intervals_sec,
                        reference_sequence.valid,
                    ),
                    _embedding_slice(query_sequence, start, stop),
                    label=label,
                    reference_event_index=reference_index,
                )
                pair_episodes.append(
                    DetectionEpisode(
                        reference=episode.reference,
                        query=episode.query,
                        targets_sec=episode.targets_sec,
                        loss_valid=episode.loss_valid,
                        metadata={**episode.metadata, "pilot_case": kind},
                    )
                )
        except ValueError as error:
            rejected.append({"label": label, "reason": str(error)})
            continue
        episodes.extend(pair_episodes)
        selected_labels.append(label)
        if len(selected_labels) >= max_labels:
            break
    if len(selected_labels) < 2:
        raise ValueError(f"split {split!r} has too few independent repeated labels")
    return tuple(episodes), {
        "selected_labels": selected_labels,
        "rejected": rejected,
    }


def _hard_scores(
    episodes: tuple[DetectionEpisode, ...],
    model: DifferentiableSubsequenceMatcher | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores: list[float] = []
    labels: list[bool] = []
    overlaps: list[float] = []
    for episode in episodes:
        reference = episode.reference.embeddings[episode.reference.valid]
        query = episode.query.embeddings
        query_intervals = episode.query.intervals_sec
        if model is not None:
            with torch.no_grad():
                reference = model.project(reference.to(model.projection.weight.device)).cpu()
                query = model.project(query.to(model.projection.weight.device)).cpu()
        valid = episode.query.valid.detach().cpu().numpy()
        boundaries = np.flatnonzero(
            np.diff(np.pad(valid.astype(np.int8), (1, 1)))
        )
        matches = []
        for start, stop in zip(boundaries[::2], boundaries[1::2], strict=True):
            if stop - start < (len(reference) + 1) // 2:
                continue
            matches.append(
                best_full_timeline_match(
                    reference.numpy(),
                    query[start:stop].numpy(),
                    query_intervals[start:stop].numpy(),
                )
            )
        if not matches:
            raise ValueError("query has no quality-contiguous run long enough for DTW")
        match = min(matches, key=lambda item: item.score)
        scores.append(match.score)
        present = bool(len(episode.targets_sec))
        labels.append(present)
        if present:
            ious = []
            for target_start, target_stop in episode.targets_sec.tolist():
                intersection = max(
                    0.0,
                    min(match.end_sec, target_stop) - max(match.start_sec, target_start),
                )
                union = max(match.end_sec, target_stop) - min(match.start_sec, target_start)
                ious.append(intersection / union if union else 0.0)
            overlaps.append(max(ious))
    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(labels, dtype=np.bool_),
        np.asarray(overlaps, dtype=np.float64),
    )


def _fit_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(scores)
    candidates = np.concatenate(
        ([np.nextafter(unique[0], -np.inf)], (unique[:-1] + unique[1:]) / 2, [unique[-1]])
    )
    return float(
        min(
            candidates,
            key=lambda threshold: (
                -_presence_metrics(scores, labels, float(threshold))["balanced_accuracy"],
                threshold,
            ),
        )
    )


def _presence_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    predicted = scores <= threshold
    positive = labels
    negative = ~labels
    sensitivity = float(np.mean(predicted[positive])) if positive.any() else 0.0
    specificity = float(np.mean(~predicted[negative])) if negative.any() else 0.0
    positive_scores = scores[positive]
    negative_scores = scores[negative]
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    auc = float(np.mean(comparisons < 0) + 0.5 * np.mean(comparisons == 0))
    return {
        "threshold": threshold,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": auc,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.max_labels <= 0 or args.steps <= 0 or args.query_seconds <= 0:
        raise ValueError("max labels, steps, and query duration must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest = read_cohort_manifest(args.manifest)
    representations = CachedMotionSequenceDataset(
        args.representations, manifest_fingerprint=manifest.fingerprint
    )
    train_episodes, train_audit = _episodes_for_split(
        manifest,
        representations,
        dataset=args.dataset,
        split="train",
        annotation_kind=args.annotation_kind,
        query_seconds=args.query_seconds,
        max_labels=args.max_labels,
        excluded_labels=set(args.exclude_label),
        seed=args.seed,
    )
    development_episodes, development_audit = _episodes_for_split(
        manifest,
        representations,
        dataset=args.dataset,
        split="development",
        annotation_kind=args.annotation_kind,
        query_seconds=args.query_seconds,
        max_labels=args.max_labels,
        excluded_labels=set(args.exclude_label),
        seed=args.seed,
    )
    base_train_scores, train_labels, base_train_iou = _hard_scores(train_episodes, None)
    base_dev_scores, dev_labels, base_dev_iou = _hard_scores(development_episodes, None)
    base_threshold = _fit_threshold(base_train_scores, train_labels)

    feature_dim = train_episodes[0].reference.feature_dim
    model = DifferentiableSubsequenceMatcher(feature_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    batch = collate_detection_episodes(train_episodes).to(args.device)
    telemetry = []
    for step in range(args.steps):
        result = train_step(model, batch, optimizer)
        if step in {0, args.steps - 1}:
            telemetry.append({"step": step + 1, "loss": result.loss, **result.telemetry})

    learned_train_scores, _, learned_train_iou = _hard_scores(train_episodes, model)
    learned_dev_scores, _, learned_dev_iou = _hard_scores(development_episodes, model)
    learned_threshold = _fit_threshold(learned_train_scores, train_labels)
    result = {
        "status": "bounded_diagnostic_not_reportable",
        "cohort_fingerprint": manifest.fingerprint,
        "representation_provenance": representations.metadata["encoder_provenance"],
        "dataset": args.dataset,
        "annotation_kind": args.annotation_kind,
        "train_episode_count": len(train_episodes),
        "development_episode_count": len(development_episodes),
        "train_episode_audit": train_audit,
        "development_episode_audit": development_audit,
        "query_seconds": args.query_seconds,
        "steps": args.steps,
        "base": {
            "train": _presence_metrics(base_train_scores, train_labels, base_threshold),
            "development": _presence_metrics(base_dev_scores, dev_labels, base_threshold),
            "train_positive_mean_iou": float(base_train_iou.mean()),
            "development_positive_mean_iou": float(base_dev_iou.mean()),
        },
        "learned_projection": {
            "train": _presence_metrics(
                learned_train_scores, train_labels, learned_threshold
            ),
            "development": _presence_metrics(
                learned_dev_scores, dev_labels, learned_threshold
            ),
            "train_positive_mean_iou": float(learned_train_iou.mean()),
            "development_positive_mean_iou": float(learned_dev_iou.mean()),
            "telemetry": telemetry,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--dataset", default="openpack")
    parser.add_argument("--annotation-kind", default="fine_action")
    parser.add_argument("--query-seconds", type=float, default=60.0)
    parser.add_argument("--max-labels", type=int, default=8)
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=["Ignore", "Others", "System Error", "Unknown"],
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
