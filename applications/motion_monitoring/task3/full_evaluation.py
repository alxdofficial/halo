"""Complete-timeline Task-3 recurrence evaluation with bounded graph memory."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from applications.motion_monitoring.data.cache import CachedRecordingDataset
from applications.motion_monitoring.evaluation_manifests import (
    Task3EvaluationUnit,
    TaskEvaluationManifest,
)
from applications.motion_monitoring.representation_cache import CachedMotionSequenceDataset
from applications.motion_monitoring.sequence import localization_intervals
from applications.motion_monitoring.task3.candidates import pool_multiscale_candidates
from applications.motion_monitoring.task3.consolidation import (
    recurrence_clusters_blockwise,
    temporal_nms,
)
from applications.motion_monitoring.task3.metrics import clustering_metrics
from applications.motion_monitoring.task3.model import RecurrentMotionMetric


@dataclass(frozen=True)
class Task3DatasetResult:
    dataset: str
    metrics: Mapping[str, float]
    streams: int
    target_occurrences: int


def _targets(recording, unit: Task3EvaluationUnit, timeline) -> tuple[torch.Tensor, torch.Tensor]:
    start = float(timeline.intervals_sec[0, 0])
    end = float(timeline.intervals_sec[-1, 1])
    rows = [
        event
        for event in recording.events
        if event.annotation_kind == unit.annotation_kind
        and event.label not in set(unit.background_labels)
        and not bool(event.metadata.get("clipped_by_recording_crop", False))
        and min(end, event.end_sec) > max(start, event.start_sec)
    ]
    labels = {label: index for index, label in enumerate(sorted({row.label for row in rows}))}
    intervals = torch.tensor(
        [[max(start, row.start_sec), min(end, row.end_sec)] for row in rows],
        dtype=timeline.intervals_sec.dtype,
    ).reshape(-1, 2)
    label_ids = torch.tensor([labels[row.label] for row in rows], dtype=torch.long)
    return intervals, label_ids


def _match_predictions(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    *,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    if not len(predicted) or not len(targets):
        return []
    intersection = (
        torch.minimum(predicted[:, None, 1], targets[None, :, 1])
        - torch.maximum(predicted[:, None, 0], targets[None, :, 0])
    ).clamp_min(0)
    union = torch.maximum(predicted[:, None, 1], targets[None, :, 1]) - torch.minimum(
        predicted[:, None, 0], targets[None, :, 0]
    )
    iou = intersection / union.clamp_min(torch.finfo(union.dtype).eps)
    pairs: list[tuple[int, int, float]] = []
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    for flat_index in torch.argsort(iou.reshape(-1), descending=True).tolist():
        prediction = flat_index // len(targets)
        target = flat_index % len(targets)
        value = float(iou[prediction, target])
        if value < iou_threshold:
            break
        if prediction in used_predictions or target in used_targets:
            continue
        used_predictions.add(prediction)
        used_targets.add(target)
        pairs.append((prediction, target, value))
    return pairs


def _stream_metrics(
    unit: Task3EvaluationUnit,
    recordings: CachedRecordingDataset,
    representations: CachedMotionSequenceDataset,
    *,
    threshold: float,
    model: RecurrentMotionMetric | None,
    durations_sec: Sequence[float],
    candidate_stride_sec: float,
    mutual_k: int,
    min_occurrences: int,
    match_iou: float,
    block_size: int,
    device: torch.device,
) -> dict[str, float]:
    recording = recordings[unit.cache_index]
    sequence = representations.get(unit.dataset, unit.recording_id, unit.stream_id)
    candidates = pool_multiscale_candidates(
        sequence.embeddings.unsqueeze(0).to(device),
        localization_intervals(sequence).unsqueeze(0).to(device),
        sequence.valid.unsqueeze(0).to(device),
        durations_sec=durations_sec,
        candidate_stride_sec=candidate_stride_sec,
    )
    mask = candidates.candidate_mask[0]
    embeddings = candidates.embeddings[0, mask]
    if model is None:
        projected = F.normalize(embeddings, dim=-1, eps=1e-8)
    else:
        projected = model.embed(embeddings)
    starts = candidates.start_sec[0, mask]
    ends = candidates.end_sec[0, mask]
    clusters = recurrence_clusters_blockwise(
        projected,
        starts,
        ends,
        threshold=threshold,
        mutual_k=mutual_k,
        min_occurrences=min_occurrences,
        block_size=block_size,
    )
    prediction_rows: list[list[float]] = []
    prediction_clusters: list[int] = []
    for cluster_id, cluster in enumerate(clusters):
        members = torch.tensor(
            cluster.member_indices, dtype=torch.long, device=projected.device
        )
        medoid = projected[cluster.medoid_index]
        member_scores = projected[members] @ medoid
        kept = temporal_nms(
            starts[members], ends[members], member_scores, iou_threshold=0.5
        )
        consolidated = members[kept]
        if len(consolidated) < min_occurrences:
            continue
        for index in consolidated.tolist():
            prediction_rows.append([float(starts[index]), float(ends[index])])
            prediction_clusters.append(cluster_id)
    predicted = torch.tensor(prediction_rows, dtype=torch.float64).reshape(-1, 2)
    target, target_labels = _targets(recording, unit, sequence)
    pairs = _match_predictions(predicted, target, iou_threshold=match_iou)
    true_positive = len(pairs)
    false_positive = len(predicted) - true_positive
    false_negative = len(target) - true_positive
    duration_hours = sequence.duration_sec / 3600.0
    row = {
        "true_positive_count": float(true_positive),
        "false_positive_count": float(false_positive),
        "false_negative_count": float(false_negative),
        "recording_hours": duration_hours,
        "count_absolute_error": float(abs(len(predicted) - len(target))),
        "matched_iou_sum": sum(value for _, _, value in pairs),
        "matched_count": float(true_positive),
        "bcubed_f1_sum": 0.0,
        "fragments_sum": 0.0,
        "cluster_metric_weight": 0.0,
    }
    if pairs:
        predicted_ids = torch.tensor(
            [prediction_clusters[prediction] for prediction, _, _ in pairs]
        )
        true_ids = torch.tensor([target_labels[target_index] for _, target_index, _ in pairs])
        cluster = clustering_metrics(predicted_ids, true_ids)
        row["bcubed_f1_sum"] = cluster["cluster/bcubed_f1"] * len(pairs)
        row["fragments_sum"] = (
            cluster["cluster/mean_fragments_per_true_motif"] * len(pairs)
        )
        row["cluster_metric_weight"] = float(len(pairs))
    return row


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    tp = sum(row["true_positive_count"] for row in rows)
    fp = sum(row["false_positive_count"] for row in rows)
    fn = sum(row["false_negative_count"] for row in rows)
    hours = sum(row["recording_hours"] for row in rows)
    cluster_weight = sum(row["cluster_metric_weight"] for row in rows)
    return {
        "occurrence_precision": tp / max(tp + fp, 1.0),
        "occurrence_recall": tp / max(tp + fn, 1.0),
        "bcubed_f1": sum(row["bcubed_f1_sum"] for row in rows)
        / max(cluster_weight, 1.0),
        "mean_fragments_per_true_motif": sum(row["fragments_sum"] for row in rows)
        / max(cluster_weight, 1.0),
        "false_occurrences_per_hour": fp / max(hours, 1e-12),
        "mean_absolute_count_error": sum(row["count_absolute_error"] for row in rows)
        / len(rows),
        "matched_mean_iou": sum(row["matched_iou_sum"] for row in rows)
        / max(tp, 1.0),
        "matched_occurrence_count": tp,
        "target_occurrence_count": tp + fn,
        "recording_hours": hours,
    }


@torch.no_grad()
def evaluate_task3_test(
    manifest: TaskEvaluationManifest,
    recording_caches: Mapping[str, CachedRecordingDataset],
    representations: CachedMotionSequenceDataset,
    *,
    score_threshold: float,
    model: RecurrentMotionMetric | None = None,
    durations_sec: Sequence[float] = (2.0, 4.0, 6.0),
    candidate_stride_sec: float = 1.0,
    mutual_k: int = 5,
    min_occurrences: int = 2,
    match_iou: float = 0.5,
    block_size: int = 1024,
    device: torch.device | str = "cpu",
) -> tuple[Task3DatasetResult, ...]:
    if manifest.task != "task3" or not manifest.units:
        raise ValueError("a non-empty Task-3 manifest is required")
    if not np.isfinite(score_threshold):
        raise ValueError("Task-3 threshold must be a finite development value")
    device = torch.device(device)
    if model is not None:
        model = model.to(device).eval()
    rows_by_dataset: dict[str, list[dict[str, float]]] = defaultdict(list)
    target_by_dataset: dict[str, int] = defaultdict(int)
    for raw in manifest.units:
        unit = Task3EvaluationUnit(**raw)
        row = _stream_metrics(
            unit,
            recording_caches[unit.dataset],
            representations,
            threshold=float(score_threshold),
            model=model,
            durations_sec=durations_sec,
            candidate_stride_sec=candidate_stride_sec,
            mutual_k=mutual_k,
            min_occurrences=min_occurrences,
            match_iou=match_iou,
            block_size=block_size,
            device=device,
        )
        rows_by_dataset[unit.dataset].append(row)
        target_by_dataset[unit.dataset] += int(
            row["true_positive_count"] + row["false_negative_count"]
        )
    return tuple(
        Task3DatasetResult(
            dataset=dataset,
            metrics=_aggregate(rows),
            streams=len(rows),
            target_occurrences=target_by_dataset[dataset],
        )
        for dataset, rows in sorted(rows_by_dataset.items())
    )
