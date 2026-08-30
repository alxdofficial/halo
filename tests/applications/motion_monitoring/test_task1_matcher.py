import numpy as np
import pytest

from applications.motion_monitoring.task1.matcher import (
    best_full_timeline_match,
    full_timeline_matches,
)


def _normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_full_timeline_match_finds_two_occurrences_without_proposals():
    reference = _normalized(
        np.asarray([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0], [-0.7, 0.7]])
    )
    background = _normalized(np.tile([[-1.0, 0.0]], (5, 1)))
    query = np.vstack([background, reference, background, reference, background])
    intervals = np.column_stack(
        [np.arange(len(query), dtype=float) * 0.5, np.arange(1, len(query) + 1) * 0.5]
    )

    matches = full_timeline_matches(
        reference,
        query,
        intervals,
        score_threshold=0.01,
        nms_iou=0.3,
    )

    assert [(match.start_patch, match.end_patch) for match in matches] == [
        (5, 9),
        (14, 18),
    ]
    assert [(match.start_sec, match.end_sec) for match in matches] == [
        (2.5, 4.5),
        (7.0, 9.0),
    ]


def test_best_match_uses_open_query_boundaries_and_bounded_warp():
    reference = _normalized(np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
    query = _normalized(
        np.asarray(
            [
                [0.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        )
    )

    match = best_full_timeline_match(reference, query)

    assert (match.start_patch, match.end_patch) == (1, 5)
    assert match.duration_ratio == pytest.approx(4 / 3)
    assert match.score < 0.02


def test_matcher_rejects_malformed_physical_intervals():
    reference = np.ones((2, 2))
    query = np.ones((3, 2))
    with pytest.raises(ValueError, match="query intervals"):
        full_timeline_matches(reference, query, np.ones((2, 2)))

    intervals = np.asarray([[0.0, 1.0], [2.0, 1.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="positive durations"):
        full_timeline_matches(reference, query, intervals)


def test_cosine_matching_is_invariant_to_embedding_scale():
    reference = np.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    query = np.asarray([[0.0, -2.0], [4.0, 0.0], [8.0, 8.0], [0.0, 6.0]])

    unscaled = best_full_timeline_match(reference, query)
    scaled = best_full_timeline_match(reference * 17.0, query * 0.03)

    assert (scaled.start_patch, scaled.end_patch) == (
        unscaled.start_patch,
        unscaled.end_patch,
    )
    assert scaled.score == pytest.approx(unscaled.score)


def test_one_best_and_multi_detection_rank_the_same_normalized_path():
    reference = _normalized(np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
    query = _normalized(
        np.asarray([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, 0.0]])
    )
    intervals = np.column_stack([np.arange(len(query)), np.arange(1, len(query) + 1)])

    best = best_full_timeline_match(reference, query, intervals)
    [multi_best] = full_timeline_matches(
        reference, query, intervals, nms_iou=1.0, max_detections=1
    )

    assert (best.start_patch, best.end_patch) == (
        multi_best.start_patch,
        multi_best.end_patch,
    )
    assert best.score == pytest.approx(multi_best.score)
