import numpy as np
import pytest

from applications.motion_monitoring.task1.preliminary_probe import subsequence_dtw


def test_subsequence_dtw_finds_retimed_pattern() -> None:
    reference = np.asarray([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    query = np.asarray(
        [
            [-1.0, 0.0],
            [-0.8, -0.2],
            [1.0, 0.0],
            [0.7, 0.7],
            [0.7, 0.7],
            [0.0, 1.0],
            [-0.5, -0.5],
        ]
    )
    query /= np.linalg.norm(query, axis=1, keepdims=True)

    match = subsequence_dtw(reference, query)

    assert match["start_patch"] == 2
    assert match["end_patch"] == 6
    assert match["score"] < 0.02
    assert 0.5 <= match["duration_ratio"] <= 2.0


def test_subsequence_dtw_rejects_degenerate_single_patch_warp() -> None:
    reference = np.repeat(np.asarray([[1.0, 0.0]]), 20, axis=0)
    query = np.asarray([[1.0, 0.0]])

    with pytest.raises(ValueError, match="warp constraint"):
        subsequence_dtw(reference, query)
