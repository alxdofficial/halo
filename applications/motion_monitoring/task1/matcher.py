"""Encoder-agnostic subsequence matching over a complete recording timeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalMatch:
    """One reference match localized on the query recording's physical clock."""

    start_patch: int
    end_patch: int
    start_sec: float
    end_sec: float
    score: float
    path_length: int
    duration_ratio: float

    def __post_init__(self) -> None:
        if self.start_patch < 0 or self.end_patch <= self.start_patch:
            raise ValueError("a temporal match must cover at least one query patch")
        if not np.isfinite((self.start_sec, self.end_sec, self.score)).all():
            raise ValueError("temporal match times and score must be finite")
        if self.end_sec <= self.start_sec or self.path_length <= 0:
            raise ValueError("temporal match duration and path length must be positive")


def _validate_inputs(
    reference: np.ndarray,
    query: np.ndarray,
    query_intervals_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    intervals = np.asarray(query_intervals_sec, dtype=np.float64)
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError(
            "reference and query must be [time, feature] with matching features"
        )
    if not len(reference) or not len(query):
        raise ValueError("reference and query must be non-empty")
    if intervals.shape != (len(query), 2):
        raise ValueError("query intervals must have shape [query_time, 2]")
    if not np.isfinite(reference).all() or not np.isfinite(query).all():
        raise ValueError("reference and query features must be finite")
    reference_norm = np.linalg.norm(reference, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    if np.any(reference_norm <= 1e-12) or np.any(query_norm <= 1e-12):
        raise ValueError("reference and query features must have non-zero row norms")
    if not np.isfinite(intervals).all() or np.any(intervals[:, 1] <= intervals[:, 0]):
        raise ValueError("query intervals must contain finite positive durations")
    if np.any(np.diff(intervals[:, 0]) < 0) or np.any(np.diff(intervals[:, 1]) < 0):
        raise ValueError("query intervals must be ordered in physical time")
    return reference / reference_norm, query / query_norm, intervals


def _dtw_tables(
    reference: np.ndarray,
    query: np.ndarray,
    *,
    warp_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if warp_penalty < 0:
        raise ValueError("warp penalty must be non-negative")
    cost = 1.0 - np.clip(reference @ query.T, -1.0, 1.0)
    n_ref, n_query = cost.shape

    # State 0 is diagonal, 1 vertical, and 2 horizontal. Two consecutive
    # non-diagonal moves are forbidden, bounding the local warp slope.
    accumulated = np.full((3, n_ref + 1, n_query + 1), np.inf, dtype=np.float64)
    previous = np.full((3, n_ref + 1, n_query + 1), -1, dtype=np.int8)
    path_lengths = np.full((3, n_ref + 1, n_query + 1), -1, dtype=np.int32)
    accumulated[0, 0, :] = 0.0
    path_lengths[0, 0, :] = 0

    for i in range(1, n_ref + 1):
        for j in range(1, n_query + 1):
            diagonal_choices = accumulated[:, i - 1, j - 1]
            previous[0, i, j] = int(np.argmin(diagonal_choices))
            accumulated[0, i, j] = cost[i - 1, j - 1] + float(
                diagonal_choices[previous[0, i, j]]
            )
            path_lengths[0, i, j] = path_lengths[previous[0, i, j], i - 1, j - 1] + 1

            vertical_choices = accumulated[[0, 2], i - 1, j]
            vertical_choice = int(np.argmin(vertical_choices))
            previous[1, i, j] = (0, 2)[vertical_choice]
            accumulated[1, i, j] = (
                cost[i - 1, j - 1]
                + warp_penalty
                + float(vertical_choices[vertical_choice])
            )
            path_lengths[1, i, j] = path_lengths[previous[1, i, j], i - 1, j] + 1

            horizontal_choices = accumulated[[0, 1], i, j - 1]
            horizontal_choice = int(np.argmin(horizontal_choices))
            previous[2, i, j] = (0, 1)[horizontal_choice]
            accumulated[2, i, j] = (
                cost[i - 1, j - 1]
                + warp_penalty
                + float(horizontal_choices[horizontal_choice])
            )
            path_lengths[2, i, j] = path_lengths[previous[2, i, j], i, j - 1] + 1
    return accumulated, previous, path_lengths


def _trace_endpoint(
    previous: np.ndarray, *, n_ref: int, end_patch: int, state: int
) -> int | None:
    i, j = n_ref, end_patch
    while i > 0:
        prior_state = int(previous[state, i, j])
        if prior_state < 0:
            return None
        if state == 0:
            i -= 1
            j -= 1
        elif state == 1:
            i -= 1
        else:
            j -= 1
        state = prior_state
    if j < 0 or end_patch <= j:
        return None
    return j


def _normalized_endpoint_costs(
    accumulated: np.ndarray, *, n_ref: int
) -> np.ndarray:
    """Use the same reference-length scale as the differentiable training path."""

    return accumulated[:, n_ref, 1:] / n_ref


def _interval_iou(left: TemporalMatch, right: TemporalMatch) -> float:
    intersection = max(
        0.0, min(left.end_sec, right.end_sec) - max(left.start_sec, right.start_sec)
    )
    union = max(left.end_sec, right.end_sec) - min(left.start_sec, right.start_sec)
    return intersection / union if union > 0 else 0.0


def full_timeline_matches(
    reference: np.ndarray,
    query: np.ndarray,
    query_intervals_sec: np.ndarray,
    *,
    warp_penalty: float = 0.05,
    score_threshold: float = np.inf,
    nms_iou: float = 0.3,
    max_detections: int | None = None,
) -> list[TemporalMatch]:
    """Locate zero or more reference occurrences without pre-segmenting the query.

    Lower scores are better. ``score_threshold`` must be calibrated on development
    subjects with target-absent recordings. Temporal NMS consolidates the many
    nearby DTW endpoints generated by one physical occurrence.
    """

    reference, query, intervals = _validate_inputs(
        reference, query, query_intervals_sec
    )
    if not np.isfinite(score_threshold) and score_threshold != np.inf:
        raise ValueError("score threshold must be finite or positive infinity")
    if not 0 <= nms_iou <= 1:
        raise ValueError("nms_iou must be in [0, 1]")
    if max_detections is not None and max_detections <= 0:
        raise ValueError("max_detections must be positive when provided")

    accumulated, previous, path_lengths = _dtw_tables(
        reference, query, warp_penalty=warp_penalty
    )
    endpoint_costs = _normalized_endpoint_costs(accumulated, n_ref=len(reference))
    endpoint_states = np.argmin(endpoint_costs, axis=0)
    endpoint_scores = endpoint_costs[
        endpoint_states, np.arange(endpoint_costs.shape[1])
    ]
    eligible = np.flatnonzero(endpoint_scores <= score_threshold)
    ranked_endpoints = eligible[np.argsort(endpoint_scores[eligible], kind="stable")]

    selected: list[TemporalMatch] = []
    for end_offset in ranked_endpoints:
        end_patch = int(end_offset) + 1
        state = int(endpoint_states[end_offset])
        start_patch = _trace_endpoint(
            previous, n_ref=len(reference), end_patch=end_patch, state=state
        )
        if start_patch is None:
            continue
        candidate = TemporalMatch(
            start_patch=start_patch,
            end_patch=end_patch,
            start_sec=float(intervals[start_patch, 0]),
            end_sec=float(intervals[end_patch - 1, 1]),
            score=float(endpoint_scores[end_offset]),
            path_length=int(path_lengths[state, len(reference), end_patch]),
            duration_ratio=(end_patch - start_patch) / len(reference),
        )
        if any(_interval_iou(candidate, kept) > nms_iou for kept in selected):
            continue
        selected.append(candidate)
        if max_detections is not None and len(selected) >= max_detections:
            break
    return sorted(selected, key=lambda item: item.start_sec)


def best_full_timeline_match(
    reference: np.ndarray,
    query: np.ndarray,
    query_intervals_sec: np.ndarray | None = None,
    *,
    warp_penalty: float = 0.05,
) -> TemporalMatch:
    """Return the best open-begin/open-end match over a complete query."""

    if query_intervals_sec is None:
        query_intervals_sec = np.column_stack(
            [np.arange(len(query), dtype=np.float64), np.arange(1, len(query) + 1)]
        )
    reference, query, intervals = _validate_inputs(
        reference, query, query_intervals_sec
    )
    accumulated, previous, path_lengths = _dtw_tables(
        reference, query, warp_penalty=warp_penalty
    )
    endpoint_costs = _normalized_endpoint_costs(accumulated, n_ref=len(reference))
    flat_index = int(np.argmin(endpoint_costs))
    state, end_offset = np.unravel_index(flat_index, endpoint_costs.shape)
    end_patch = int(end_offset) + 1
    start_patch = _trace_endpoint(
        previous,
        n_ref=len(reference),
        end_patch=end_patch,
        state=int(state),
    )
    if start_patch is None:
        raise ValueError("no DTW path satisfies the bounded local warp constraint")
    return TemporalMatch(
        start_patch=start_patch,
        end_patch=end_patch,
        start_sec=float(intervals[start_patch, 0]),
        end_sec=float(intervals[end_patch - 1, 1]),
        score=float(endpoint_costs[state, end_offset]),
        path_length=int(path_lengths[state, len(reference), end_patch]),
        duration_ratio=(end_patch - start_patch) / len(reference),
    )
