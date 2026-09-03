"""Complete-dataset Task-1 evaluation over immutable enrolled-detection units."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from applications.motion_monitoring.data.cache import CachedRecordingDataset
from applications.motion_monitoring.evaluation_manifests import (
    Task1EvaluationUnit,
    TaskEvaluationManifest,
)
from applications.motion_monitoring.representation_cache import (
    CachedMotionSequenceDataset,
    bounded_representation_id,
)
from applications.motion_monitoring.task1.episodes import (
    crop_sequence,
    episode_from_recordings,
    from_motion_sequence,
)
from applications.motion_monitoring.task1.matcher import full_timeline_matches
from applications.motion_monitoring.task1.model import DifferentiableSubsequenceMatcher
from applications.motion_monitoring.task1.training import event_detection_metrics


@dataclass(frozen=True)
class Task1DatasetResult:
    dataset: str
    metrics: Mapping[str, float]
    units: int
    target_events: int


def unit_matches(
    unit: Task1EvaluationUnit,
    recordings: CachedRecordingDataset,
    representations: CachedMotionSequenceDataset,
    *,
    score_threshold: float,
    model: DifferentiableSubsequenceMatcher | None,
    nms_iou: float,
) -> tuple[list, object]:
    reference_recording = recordings[unit.reference_cache_index]
    query_recording = recordings[unit.query_cache_index]
    reference_sequence = representations.get(
        unit.dataset,
        bounded_representation_id(unit.reference_recording_id, unit.reference_event_index),
        unit.reference_stream_id,
    )
    query_sequence = from_motion_sequence(
        representations.get(
            unit.dataset, unit.query_recording_id, unit.query_stream_id
        )
    )
    if getattr(unit, "query_interval_sec", None) is not None:
        block_start, block_end = map(float, unit.query_interval_sec)
        query_sequence = crop_sequence(query_sequence, block_start, block_end)
    episode = episode_from_recordings(
        reference_recording,
        query_recording,
        from_motion_sequence(reference_sequence),
        query_sequence,
        label=unit.label,
        reference_event_index=unit.reference_event_index,
        target_intervals_sec=unit.target_intervals_sec,
        reference_interval_sec=getattr(unit, "reference_interval_sec", None),
        guard_intervals_sec=unit.guard_intervals_sec,
    )
    reference = episode.reference.embeddings[episode.reference.valid]
    query = episode.query.embeddings
    valid = episode.query.valid.detach().cpu().numpy()
    intervals = episode.query.intervals_sec.detach().cpu().numpy()
    if model is None:
        matches = []
        changes = np.flatnonzero(np.diff(np.pad(valid.astype(np.int8), (1, 1))))
        for start, stop in zip(changes[::2], changes[1::2], strict=True):
            if stop - start < (len(reference) + 1) // 2:
                continue
            for match in full_timeline_matches(
                reference.detach().cpu().numpy(),
                query[start:stop].detach().cpu().numpy(),
                intervals[start:stop],
                score_threshold=score_threshold,
                nms_iou=nms_iou,
            ):
                matches.append(match)
    else:
        matches = model.detect(
            reference,
            query,
            intervals,
            score_threshold=score_threshold,
            query_valid=valid,
            nms_iou=nms_iou,
        )
    return matches, episode


def coalesce_matches(matches: list, gap_sec: float) -> list:
    """Chain adjacent accepted matches into one detection per physical bout.

    An excerpt reference enrolled from a periodic activity (spec section A, WEAR
    leading excerpts) legitimately matches repeatedly through one long bout;
    scoring each repetition separately fails the IoU criterion against the
    full-bout target by construction. Matches whose gap is at most ``gap_sec``
    merge into one interval carrying the best (minimum) member score.
    """

    if gap_sec < 0:
        raise ValueError("coalesce gap must be non-negative")
    if not matches:
        return []
    ordered = sorted(matches, key=lambda item: item.start_sec)
    merged = [ordered[0]]
    for match in ordered[1:]:
        previous = merged[-1]
        if match.start_sec - previous.end_sec <= gap_sec:
            merged[-1] = type(previous)(
                start_patch=previous.start_patch,
                end_patch=max(previous.end_patch, match.end_patch),
                start_sec=previous.start_sec,
                end_sec=max(previous.end_sec, match.end_sec),
                score=min(previous.score, match.score),
                path_length=previous.path_length + match.path_length,
                duration_ratio=previous.duration_ratio,
            )
        else:
            merged.append(match)
    return merged


DURATION_STRATA = (("lt1s", 0.0, 1.0), ("1to2s", 1.0, 2.0), ("ge2s", 2.0, float("inf")))
REFERENCE_POSITION_STRATA = (("2to3", 2, 3), ("4to15", 4, 15), ("ge16", 16, 10**9))


def _duration_bucket(duration: float) -> str:
    for name, low, high in DURATION_STRATA:
        if low <= duration < high:
            return name
    return DURATION_STRATA[-1][0]


def _reference_bucket(positions: int) -> str:
    for name, low, high in REFERENCE_POSITION_STRATA:
        if low <= positions <= high:
            return name
    raise ValueError(f"reference has no declared resolution stratum: {positions} positions")


def _duration_stratified_recall(
    matches: list, targets_sec, *, iou_threshold: float
) -> dict[str, dict[str, float]]:
    """Greedy per-target matching, counted per target-duration stratum."""

    targets = [tuple(map(float, row)) for row in targets_sec.tolist()]
    unmatched = set(range(len(targets)))
    matched: set[int] = set()
    for match in sorted(matches, key=lambda item: item.score):
        best_index, best_iou = None, 0.0
        for index in unmatched:
            left, right = targets[index]
            intersection = max(0.0, min(match.end_sec, right) - max(match.start_sec, left))
            union = max(match.end_sec, right) - min(match.start_sec, left)
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_index is not None and best_iou >= iou_threshold:
            unmatched.remove(best_index)
            matched.add(best_index)
    counts: dict[str, dict[str, float]] = {
        name: {"matched": 0.0, "total": 0.0} for name, _, _ in DURATION_STRATA
    }
    for index, (left, right) in enumerate(targets):
        bucket = _duration_bucket(right - left)
        counts[bucket]["total"] += 1.0
        if index in matched:
            counts[bucket]["matched"] += 1.0
    return counts


def _unit_metrics(
    unit: Task1EvaluationUnit,
    recordings: CachedRecordingDataset,
    representations: CachedMotionSequenceDataset,
    *,
    score_threshold: float | Mapping[str, float],
    model: DifferentiableSubsequenceMatcher | None,
    nms_iou: float,
    match_iou: float,
    coalesce_gap_sec: float | None = None,
) -> dict[str, float]:
    matches, episode = unit_matches(
        unit,
        recordings,
        representations,
        score_threshold=(float("inf") if isinstance(score_threshold, Mapping) else score_threshold),
        model=model,
        nms_iou=nms_iou,
    )
    resolved_threshold = (
        float(
            score_threshold.get(
                _reference_bucket(
                    int(
                        episode.metadata.get(
                            "reference_positions", len(episode.reference.embeddings)
                        )
                    )
                ),
                score_threshold.get("global", float("nan")),
            )
        )
        if isinstance(score_threshold, Mapping)
        else float(score_threshold)
    )
    if not np.isfinite(resolved_threshold):
        raise ValueError("Task-1 threshold mapping lacks a finite stratum fallback")
    accepted_matches = [
        match for match in matches if match.score <= resolved_threshold
    ]
    if coalesce_gap_sec is not None:
        accepted_matches = coalesce_matches(accepted_matches, coalesce_gap_sec)
    metrics = event_detection_metrics(
        accepted_matches,
        episode.targets_sec,
        query_duration_sec=episode.query.intervals_sec[-1, 1].item()
        - episode.query.intervals_sec[0, 0].item(),
        iou_threshold=match_iou,
        score_threshold=float("inf"),
    )
    metrics["query_subject_id"] = unit.query_subject_id
    metrics["reference_positions"] = int(
        episode.metadata.get("reference_positions", len(episode.reference.embeddings))
    )
    metrics["duration_strata"] = _duration_stratified_recall(
        accepted_matches, episode.targets_sec, iou_threshold=match_iou
    )
    return metrics


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    true_positive = sum(row["true_positive_count"] for row in rows)
    false_positive = sum(row["false_positive_count"] for row in rows)
    false_negative = sum(row["false_negative_count"] for row in rows)
    hours = sum(row["query_hours"] for row in rows)
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "false_alarms_per_hour": false_positive / max(hours, 1e-12),
        "mean_onset_error_sec": sum(
            row["onset_absolute_error_sum_sec"] for row in rows
        )
        / max(true_positive, 1.0),
        "mean_offset_error_sec": sum(
            row["offset_absolute_error_sum_sec"] for row in rows
        )
        / max(true_positive, 1.0),
        "mean_absolute_count_error": sum(
            row["mean_absolute_count_error"] for row in rows
        )
        / len(rows),
        "matched_event_count": true_positive,
        "target_event_count": true_positive + false_negative,
        "query_hours": hours,
    }


def _subject_bootstrap(
    rows: list[dict[str, float]],
    *,
    resamples: int = 1000,
    seed: int = 20260831,
) -> dict[str, object]:
    """Cluster bootstrap over query subjects (protocol: the unit of uncertainty)."""

    by_subject: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row.get("query_subject_id", "unknown"))].append(row)
    subjects = sorted(by_subject)
    if len(subjects) < 2:
        return {"subjects": len(subjects), "note": "too few subjects for a CI"}
    rng = np.random.default_rng(seed)
    f1s, fa_rates = [], []
    for _ in range(resamples):
        drawn = rng.choice(len(subjects), size=len(subjects), replace=True)
        pooled: list[dict[str, float]] = []
        for index in drawn:
            pooled.extend(by_subject[subjects[index]])
        stats = _aggregate(pooled)
        f1s.append(stats["event_f1"])
        fa_rates.append(stats["false_alarms_per_hour"])
    def interval(values: list[float]) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
    return {
        "subjects": len(subjects),
        "resamples": resamples,
        "event_f1_ci95": interval(f1s),
        "false_alarms_per_hour_ci95": interval(fa_rates),
    }


def _strata_summary(rows: list[dict[str, float]]) -> dict[str, object]:
    reference_strata: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        reference_strata[_reference_bucket(int(row.get("reference_positions", 0)))].append(row)
    duration_totals: dict[str, dict[str, float]] = {
        name: {"matched": 0.0, "total": 0.0} for name, _, _ in DURATION_STRATA
    }
    for row in rows:
        for bucket, counts in row.get("duration_strata", {}).items():
            duration_totals[bucket]["matched"] += counts["matched"]
            duration_totals[bucket]["total"] += counts["total"]
    return {
        "by_reference_positions": {
            bucket: {"units": len(bucket_rows), **_aggregate(bucket_rows)}
            for bucket, bucket_rows in sorted(reference_strata.items())
        },
        "target_recall_by_duration": {
            bucket: {
                "targets": counts["total"],
                "recall": (
                    counts["matched"] / counts["total"] if counts["total"] else None
                ),
            }
            for bucket, counts in duration_totals.items()
        },
    }


def evaluate_task1_test(
    manifest: TaskEvaluationManifest,
    recording_caches: Mapping[str, CachedRecordingDataset],
    representations: CachedMotionSequenceDataset,
    *,
    score_threshold: float | Mapping[str, float],
    model: DifferentiableSubsequenceMatcher | None = None,
    nms_iou: float = 0.3,
    match_iou: float = 0.5,
    coalesce_gap_by_dataset: Mapping[str, float] | None = None,
) -> tuple[Task1DatasetResult, ...]:
    """Evaluate every frozen unit and aggregate independently per test dataset.

    The threshold is mandatory and must be selected once on development
    sources. It may be a global scalar or a frozen mapping by declared
    reference-resolution stratum. This function never tunes against test
    annotations.
    ``coalesce_gap_by_dataset`` enables bout coalescing (spec section D.4) for
    datasets whose targets are long periodic bouts detected by excerpt
    references; it is part of the declared protocol, never tuned on test.
    """

    if manifest.task != "task1" or not manifest.units:
        raise ValueError("a non-empty Task-1 manifest is required")
    if isinstance(score_threshold, Mapping):
        if not score_threshold or any(not np.isfinite(float(value)) for value in score_threshold.values()):
            raise ValueError("Task-1 thresholds must be finite development values")
    elif not np.isfinite(score_threshold):
        raise ValueError("Task-1 threshold must be a finite development value")

    metrics_by_dataset: dict[str, list[dict[str, float]]] = defaultdict(list)
    targets_by_dataset: dict[str, int] = defaultdict(int)
    rejected_by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for index, row in enumerate(manifest.units):
        unit = Task1EvaluationUnit(**row)
        try:
            metrics = _unit_metrics(
                unit,
                recording_caches[unit.dataset],
                representations,
                score_threshold=score_threshold,
                model=model,
                nms_iou=nms_iou,
                match_iou=match_iou,
                coalesce_gap_sec=(
                    None
                    if coalesce_gap_by_dataset is None
                    else coalesce_gap_by_dataset.get(unit.dataset)
                ),
            )
        except ValueError as error:
            # An encoder-specific ineligibility (for example every embedding
            # position over the reference is invalid for this representation)
            # is recorded loudly, never silently skipped or fatal.
            rejected_by_dataset[unit.dataset].append(
                {"unit_index": index, "label": unit.label, "reason": str(error)}
            )
            continue
        metrics_by_dataset[unit.dataset].append(metrics)
        targets_by_dataset[unit.dataset] += len(unit.target_intervals_sec)
    return tuple(
        Task1DatasetResult(
            dataset=dataset,
            metrics={
                **_aggregate(rows),
                "subject_uncertainty": _subject_bootstrap(rows),
                "strata": _strata_summary(rows),
                "rejected_units": rejected_by_dataset.get(dataset, []),
            },
            units=len(rows),
            target_events=targets_by_dataset[dataset],
        )
        for dataset, rows in sorted(metrics_by_dataset.items())
    )
