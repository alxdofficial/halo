from __future__ import annotations

import numpy as np
import pytest
import torch

from applications.motion_monitoring.task1.episodes import DetectionEpisode, EmbeddingSequence
from applications.motion_monitoring.task1.pilot import (
    _fit_threshold,
    _hard_scores,
    _presence_metrics,
)


def test_pilot_hard_scoring_preserves_the_source_clock_for_boundary_iou():
    reference = EmbeddingSequence(
        torch.eye(2),
        torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float64),
        torch.ones(2, dtype=torch.bool),
    )
    query = EmbeddingSequence(
        torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        torch.tensor(
            [[100.0, 101.0], [101.0, 102.0], [102.0, 103.0], [103.0, 104.0]],
            dtype=torch.float64,
        ),
        torch.ones(4, dtype=torch.bool),
    )
    episode = DetectionEpisode(
        reference,
        query,
        torch.tensor([[101.0, 103.0]], dtype=torch.float64),
    )

    scores, labels, overlaps = _hard_scores((episode,), None)

    assert scores.shape == (1,)
    assert labels.tolist() == [True]
    assert overlaps.tolist() == [1.0]


def test_pilot_threshold_is_fit_only_from_supplied_calibration_scores():
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    labels = np.asarray([True, True, False, False])
    threshold = _fit_threshold(scores, labels)
    metrics = _presence_metrics(scores, labels, threshold)
    assert metrics["balanced_accuracy"] == 1.0
    assert 0.2 <= threshold < 0.8


def test_pilot_hard_scoring_does_not_bridge_invalid_query_gaps():
    reference = EmbeddingSequence(
        torch.eye(3),
        torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=torch.float64),
        torch.ones(3, dtype=torch.bool),
    )
    query = EmbeddingSequence(
        torch.eye(3),
        torch.tensor([[10.0, 11.0], [11.0, 12.0], [12.0, 13.0]], dtype=torch.float64),
        torch.tensor([True, False, True]),
    )
    episode = DetectionEpisode(reference, query, torch.empty(0, 2))

    with pytest.raises(ValueError, match="quality-contiguous run"):
        _hard_scores((episode,), None)
