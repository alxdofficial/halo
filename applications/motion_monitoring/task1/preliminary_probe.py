"""CPU-only feasibility probes for Task 1.

This script deliberately uses fixed physical features rather than a learned encoder. It answers two
questions before application model work begins:

1. Which converted sources retain independent executions and reconstructable continuous recordings?
2. Can a simple subsequence matcher localize an exercise from one MoniPar week in a later week?

Its output is diagnostic evidence, not a promoted application result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from applications.motion_monitoring.task1.matcher import best_full_timeline_match


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "datasets"

DATASET_ROLES = {
    "harmes": "development/action",
    "xrf_v2": "development/action+continuous",
    "opportunity": "development/action+continuous",
    "sp_sw_har": "development/transition",
    "forth_trace": "development/action+continuous",
    "unimib_shar": "development/isolated",
    "mmfit": "development/bout+continuous",
    "phytmo": "development/bout",
    "kuhar": "development/isolated",
    "wisdm": "development/bout",
    "nfi_fared": "development/action+background",
    "capture24": "development/background",
    "monipar": "held-out/cross-week",
    "spar": "held-out/within-bout",
    "upper_limb_use": "held-out/task-bout",
    "usc_had": "held-out/repeated-trial",
    "motionsense": "held-out/repeated-trial",
}

BACKGROUND_NAMES = {
    "background",
    "idle",
    "no_activity",
    "null",
    "other",
    "unlabeled",
    "unknown",
}

MONIPAR_ACTIONS = {
    "arising_from_a_chair",
    "finger_tapping",
    "gait",
    "moving_hands_to_the_chest",
    "postural_hand_tremor",
    "pronation_supination",
    "rapid_hand_opening_and_closing",
}


@dataclass(frozen=True)
class DatasetInventory:
    dataset: str
    role: str
    sessions: int
    source_recordings: int
    physical_executions: int
    subjects: int
    labels: int
    stream_view_hours: float
    unique_execution_hours: float
    median_session_seconds: float
    max_session_seconds: float
    multi_label_recordings: int
    subject_label_pairs: int
    pairs_with_two_executions: int
    pairs_with_two_source_recordings: int
    explicit_background_labels: tuple[str, ...]
    has_recording_map: bool


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _column_stats(path: Path, column: str) -> tuple[object | None, object | None]:
    """Return Parquet min/max without loading a potentially day-long sensor column."""
    parquet = pq.ParquetFile(path)
    try:
        index = parquet.schema_arrow.names.index(column)
    except ValueError:
        return None, None
    minima: list[object] = []
    maxima: list[object] = []
    for row_group in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(row_group).column(index).statistics
        if stats is None or not stats.has_min_max:
            continue
        minima.append(stats.min)
        maxima.append(stats.max)
    return (min(minima), max(maxima)) if minima else (None, None)


def _subject_from_session(session: str) -> str:
    patterns = (
        r"^(sub\d+)",
        r"^(part\d+)",
        r"^(s\d+)",
        r"^(p\d+)",
        r"^(w\d+)",
        r"^((?:hc|rem|sup)\d+)",
        r"^(control_c\d+|patient_p\d+)",
    )
    for pattern in patterns:
        match = re.match(pattern, session)
        if match:
            return match.group(1)
    return session.split("_")[0]


def _session_summary(path: Path, nominal_rate: float | None) -> tuple[str, float]:
    subject_min, subject_max = _column_stats(path, "subject")
    subject = (
        str(subject_min)
        if subject_min is not None and subject_min == subject_max
        else ""
    )
    time_min, time_max = _column_stats(path, "timestamp_sec")
    if time_min is not None and time_max is not None:
        duration = max(0.0, float(time_max) - float(time_min))
        if nominal_rate:
            duration += 1.0 / nominal_rate
    else:
        rows = pq.ParquetFile(path).metadata.num_rows
        duration = rows / nominal_rate if nominal_rate else float("nan")
    return subject, duration


def inventory_dataset(dataset: str) -> DatasetInventory:
    root = DATA_ROOT / dataset
    paths = sorted((root / "sessions").glob("*/data.parquet"))
    labels_by_session = _read_json(root / "labels.json", {})
    recordings = _read_json(root / "recordings.json", {})
    events = _read_json(root / "events.json", {})
    metadata = _read_json(root / "metadata.json", {})
    nominal_rate = metadata.get("sampling_rate_hz")
    nominal_rate = float(nominal_rate) if nominal_rate else None

    recording_labels: dict[str, set[str]] = defaultdict(set)
    subjects: set[str] = set()
    all_labels: set[str] = set()
    durations: list[float] = []
    subject_label_executions: dict[tuple[str, str], set[str]] = defaultdict(set)
    subject_label_recordings: dict[tuple[str, str], set[str]] = defaultdict(set)
    physical_executions: set[str] = set()
    execution_durations: dict[str, float] = {}

    for path in paths:
        session = path.parent.name
        subject, duration = _session_summary(path, nominal_rate)
        subject = subject or _subject_from_session(session)
        recording = str(recordings.get(session, session))
        execution = str(events.get(session, session))
        labels = labels_by_session.get(session, [])
        if isinstance(labels, str):
            labels = [labels]
        labels = [str(label) for label in labels]

        subjects.add(subject)
        durations.append(duration)
        all_labels.update(labels)
        recording_labels[recording].update(labels)
        physical_executions.add(execution)
        execution_durations[execution] = max(
            duration, execution_durations.get(execution, 0.0)
        )
        for label in labels:
            subject_label_executions[(subject, label)].add(execution)
            subject_label_recordings[(subject, label)].add(recording)

    repeatable_executions = sum(
        len(items) >= 2 for items in subject_label_executions.values()
    )
    repeatable_recordings = sum(
        len(items) >= 2 for items in subject_label_recordings.values()
    )
    explicit_background = tuple(
        sorted(
            label
            for label in all_labels
            if label.lower().replace(" ", "_") in BACKGROUND_NAMES
        )
    )
    finite_durations = np.asarray(
        [value for value in durations if math.isfinite(value)]
    )
    return DatasetInventory(
        dataset=dataset,
        role=DATASET_ROLES[dataset],
        sessions=len(paths),
        source_recordings=len(recording_labels),
        physical_executions=len(physical_executions),
        subjects=len(subjects),
        labels=len(all_labels),
        stream_view_hours=(
            float(finite_durations.sum() / 3600.0) if len(finite_durations) else 0.0
        ),
        unique_execution_hours=float(sum(execution_durations.values()) / 3600.0),
        median_session_seconds=(
            float(np.median(finite_durations)) if len(finite_durations) else 0.0
        ),
        max_session_seconds=(
            float(np.max(finite_durations)) if len(finite_durations) else 0.0
        ),
        multi_label_recordings=sum(
            len(labels) > 1 for labels in recording_labels.values()
        ),
        subject_label_pairs=len(subject_label_executions),
        pairs_with_two_executions=repeatable_executions,
        pairs_with_two_source_recordings=repeatable_recordings,
        explicit_background_labels=explicit_background,
        has_recording_map=bool(recordings),
    )


def _majority(values: Iterable[str]) -> str:
    series = pd.Series(list(values), dtype="string")
    return str(series.value_counts().index[0])


def physical_patch_features(frame: pd.DataFrame, patch_seconds: float = 1.0):
    """Extract fixed, dimensionless one-second features from an accelerometer stream."""
    required = ["timestamp_sec", "acc_x", "acc_y", "acc_z", "activity"]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    timestamp = frame["timestamp_sec"].to_numpy(dtype=np.float64)
    acc = frame[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=np.float64) / 9.80665
    labels = frame["activity"].astype(str).to_numpy()
    bins = np.floor((timestamp - timestamp[0]) / patch_seconds).astype(np.int64)
    rate = 1.0 / np.median(np.diff(timestamp))

    features: list[np.ndarray] = []
    patch_labels: list[str] = []
    intervals: list[tuple[float, float]] = []
    for index in range(int(bins[-1]) + 1):
        keep = bins == index
        values = acc[keep]
        if len(values) < max(3, int(0.7 * rate * patch_seconds)):
            continue
        magnitude = np.linalg.norm(values, axis=1)
        vector = np.concatenate(
            [
                values.mean(axis=0),
                values.std(axis=0),
                [magnitude.mean(), magnitude.std()],
            ]
        )
        features.append(vector)
        patch_labels.append(_majority(labels[keep]))
        intervals.append(
            (float(timestamp[keep][0]), float(timestamp[keep][-1] + 1.0 / rate))
        )

    result = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / np.maximum(norms, 1e-8)
    return result, np.asarray(patch_labels), np.asarray(intervals)


def subsequence_dtw(
    reference: np.ndarray, query: np.ndarray, warp_penalty: float = 0.05
):
    """Compatibility wrapper around the promoted full-timeline matcher."""
    match = best_full_timeline_match(reference, query, warp_penalty=warp_penalty)
    return {
        "start_patch": match.start_patch,
        "end_patch": match.end_patch,
        "score": match.score,
        "path_length": match.path_length,
        "duration_ratio": match.duration_ratio,
    }


def pooled_cosine(reference: np.ndarray, query: np.ndarray):
    """Fixed-duration pooled-vector control."""
    length = len(reference)
    if len(query) < length:
        raise ValueError("query is shorter than reference")
    ref = reference.mean(axis=0)
    ref /= max(np.linalg.norm(ref), 1e-8)
    cumulative = np.vstack([np.zeros((1, query.shape[1])), np.cumsum(query, axis=0)])
    windows = (cumulative[length:] - cumulative[:-length]) / length
    windows /= np.maximum(np.linalg.norm(windows, axis=1, keepdims=True), 1e-8)
    similarities = windows @ ref
    start = int(np.argmax(similarities))
    return {
        "start_patch": start,
        "end_patch": start + length,
        "score": float(1.0 - similarities[start]),
    }


def _runs(labels: np.ndarray, target: str) -> list[tuple[int, int]]:
    keep = labels == target
    padded = np.pad(keep.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2])]


def _best_iou(prediction: tuple[int, int], targets: list[tuple[int, int]]) -> float:
    start, end = prediction
    best = 0.0
    for target_start, target_end in targets:
        intersection = max(0, min(end, target_end) - max(start, target_start))
        union = max(end, target_end) - min(start, target_start)
        best = max(best, intersection / union if union else 0.0)
    return best


def _positive_crop(length: int, targets: list[tuple[int, int]], crop_length: int):
    target_start, target_end = max(targets, key=lambda item: item[1] - item[0])
    center = (target_start + target_end) // 2
    start = max(0, min(center - crop_length // 2, length - crop_length))
    end = min(length, start + crop_length)
    shifted = [
        (max(0, a - start), min(end - start, b - start))
        for a, b in targets
        if b > start and a < end
    ]
    return start, end, shifted


def _negative_crop(
    length: int,
    targets: list[tuple[int, int]],
    crop_length: int,
    *,
    seed_material: str,
):
    if length < crop_length:
        return None
    eligible = []
    for start in range(0, length - crop_length + 1, 5):
        end = start + crop_length
        if all(
            end <= target_start or start >= target_end
            for target_start, target_end in targets
        ):
            eligible.append((start, end))
    if not eligible:
        return None
    seed = int.from_bytes(
        hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "little"
    )
    return eligible[int(np.random.default_rng(seed).integers(len(eligible)))]


def monipar_cross_week_probe(max_cases: int = 80) -> dict:
    """Localize a prior-week action block inside the next available weekly protocol."""
    root = DATA_ROOT / "monipar" / "sessions"
    by_subject: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.glob("*/data.parquet")):
        subject = path.parent.name.split("_w", 1)[0]
        by_subject[subject].append(path)

    cases: list[tuple[Path, Path, str]] = []
    for sessions in by_subject.values():
        for reference_path, query_path in zip(sessions[:-1], sessions[1:]):
            reference_labels = set(
                pd.read_parquet(reference_path, columns=["activity"])["activity"]
            )
            query_labels = set(
                pd.read_parquet(query_path, columns=["activity"])["activity"]
            )
            for label in sorted(MONIPAR_ACTIONS & reference_labels & query_labels):
                cases.append((reference_path, query_path, label))
    order = np.random.default_rng(20260828).permutation(len(cases))
    if max_cases > 0:
        order = order[:max_cases]
    cases = [cases[int(index)] for index in order]

    cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    records: list[dict] = []
    started = time.perf_counter()
    for reference_path, query_path, label in cases:
        for path in (reference_path, query_path):
            if path not in cache:
                cache[path] = physical_patch_features(pd.read_parquet(path))
        reference_features, reference_labels, _ = cache[reference_path]
        query_features, query_labels, _ = cache[query_path]
        reference_runs = _runs(reference_labels, label)
        target_runs = _runs(query_labels, label)
        if not reference_runs or not target_runs:
            continue
        reference_start, reference_end = max(
            reference_runs, key=lambda item: item[1] - item[0]
        )
        reference = reference_features[reference_start:reference_end]
        if len(reference) < 3:
            continue

        dtw = subsequence_dtw(reference, query_features)
        pooled = pooled_cosine(reference, query_features)
        dtw_interval = (dtw["start_patch"], dtw["end_patch"])
        pooled_interval = (pooled["start_patch"], pooled["end_patch"])
        dtw_center = min((sum(dtw_interval) // 2), len(query_labels) - 1)
        pooled_center = min((sum(pooled_interval) // 2), len(query_labels) - 1)
        crop_length = min(120, len(query_features))
        positive_start, positive_end, positive_targets = _positive_crop(
            len(query_features), target_runs, crop_length
        )
        negative_bounds = _negative_crop(
            len(query_features),
            target_runs,
            crop_length,
            seed_material=f"{reference_path}:{query_path}:{label}",
        )
        positive_query = query_features[positive_start:positive_end]
        positive_dtw = subsequence_dtw(reference, positive_query)
        positive_pooled = pooled_cosine(reference, positive_query)
        negative_dtw_score = None
        negative_pooled_score = None
        if negative_bounds is not None:
            negative_start, negative_end = negative_bounds
            negative_query = query_features[negative_start:negative_end]
            negative_dtw_score = subsequence_dtw(reference, negative_query)["score"]
            negative_pooled_score = pooled_cosine(reference, negative_query)["score"]

        positive_interval = (positive_dtw["start_patch"], positive_dtw["end_patch"])
        records.append(
            {
                "subject": reference_path.parent.name.split("_w", 1)[0],
                "reference_session": reference_path.parent.name,
                "query_session": query_path.parent.name,
                "label": label,
                "reference_patches": len(reference),
                "dtw_score": dtw["score"],
                "dtw_center_correct": bool(query_labels[dtw_center] == label),
                "dtw_iou": _best_iou(dtw_interval, target_runs),
                "pooled_center_correct": bool(query_labels[pooled_center] == label),
                "pooled_iou": _best_iou(pooled_interval, target_runs),
                "positive_120s_dtw_score": positive_dtw["score"],
                "positive_120s_dtw_iou": _best_iou(positive_interval, positive_targets),
                "negative_120s_dtw_score": negative_dtw_score,
                "positive_120s_pooled_score": positive_pooled["score"],
                "negative_120s_pooled_score": negative_pooled_score,
            }
        )

    elapsed = time.perf_counter() - started
    if not records:
        return {"cases": 0, "elapsed_seconds": elapsed, "records": []}
    paired = [row for row in records if row["negative_120s_dtw_score"] is not None]
    subjects = sorted({row["subject"] for row in paired})
    shuffled_subjects = np.random.default_rng(20260830).permutation(subjects)
    split_index = max(1, len(shuffled_subjects) // 2)
    calibration_subjects = set(shuffled_subjects[:split_index])
    evaluation_subjects = set(shuffled_subjects[split_index:])
    calibration = [row for row in paired if row["subject"] in calibration_subjects]
    evaluation = [row for row in paired if row["subject"] in evaluation_subjects]
    dtw_negative = np.asarray([row["negative_120s_dtw_score"] for row in calibration])
    pooled_negative = np.asarray(
        [row["negative_120s_pooled_score"] for row in calibration]
    )
    dtw_threshold = (
        float(np.percentile(dtw_negative, 5)) if len(paired) else float("nan")
    )
    pooled_threshold = (
        float(np.percentile(pooled_negative, 5)) if len(paired) else float("nan")
    )
    return {
        "cases": len(records),
        "elapsed_seconds": elapsed,
        "cases_per_second": len(records) / elapsed,
        "dtw_center_accuracy": float(
            np.mean([row["dtw_center_correct"] for row in records])
        ),
        "dtw_mean_iou": float(np.mean([row["dtw_iou"] for row in records])),
        "dtw_median_score": float(np.median([row["dtw_score"] for row in records])),
        "pooled_center_accuracy": float(
            np.mean([row["pooled_center_correct"] for row in records])
        ),
        "pooled_mean_iou": float(np.mean([row["pooled_iou"] for row in records])),
        "paired_120s_cases": len(paired),
        "calibration_subjects": sorted(calibration_subjects),
        "evaluation_subjects": sorted(evaluation_subjects),
        "calibration_cases": len(calibration),
        "evaluation_cases": len(evaluation),
        "dtw_pairwise_positive_better": (
            float(
                np.mean(
                    [
                        row["positive_120s_dtw_score"] < row["negative_120s_dtw_score"]
                        for row in paired
                    ]
                )
            )
            if paired
            else float("nan")
        ),
        "dtw_eval_recall_at_calibrated_95pct_negative_specificity": (
            float(
                np.mean(
                    [
                        row["positive_120s_dtw_score"] < dtw_threshold
                        for row in evaluation
                    ]
                )
            )
            if evaluation and calibration
            else float("nan")
        ),
        "dtw_eval_negative_specificity": (
            float(
                np.mean(
                    [
                        row["negative_120s_dtw_score"] >= dtw_threshold
                        for row in evaluation
                    ]
                )
            )
            if evaluation and calibration
            else float("nan")
        ),
        "pooled_pairwise_positive_better": (
            float(
                np.mean(
                    [
                        row["positive_120s_pooled_score"]
                        < row["negative_120s_pooled_score"]
                        for row in paired
                    ]
                )
            )
            if paired
            else float("nan")
        ),
        "pooled_eval_recall_at_calibrated_95pct_negative_specificity": (
            float(
                np.mean(
                    [
                        row["positive_120s_pooled_score"] < pooled_threshold
                        for row in evaluation
                    ]
                )
            )
            if evaluation and calibration
            else float("nan")
        ),
        "pooled_eval_negative_specificity": (
            float(
                np.mean(
                    [
                        row["negative_120s_pooled_score"] >= pooled_threshold
                        for row in evaluation
                    ]
                )
            )
            if evaluation and calibration
            else float("nan")
        ),
        "records": records,
    }


def run(max_cases: int) -> dict:
    inventories = [inventory_dataset(dataset) for dataset in DATASET_ROLES]
    return {
        "status": "preliminary_not_promoted",
        "query_duration_seconds": 120,
        "inventory": [asdict(item) for item in inventories],
        "monipar_cross_week_physical_feature_probe": monipar_cross_week_probe(
            max_cases=max_cases
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(max_cases=args.max_cases)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
