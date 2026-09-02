import math

import numpy as np
import torch

from applications.motion_monitoring.task1 import (
    DifferentiableSubsequenceMatcher,
    SyntheticDetectionDataset,
    collate_detection_episodes,
    event_detection_metrics,
    train_step,
)
from applications.motion_monitoring.task1.matcher import TemporalMatch
from applications.motion_monitoring.task1.train_full import _event_prefix


def test_synthetic_episodes_are_deterministic_and_include_absent_examples():
    first = SyntheticDetectionDataset(
        4, feature_dim=8, query_patches=24, reference_patches=4, seed=11
    )
    second = SyntheticDetectionDataset(
        4, feature_dim=8, query_patches=24, reference_patches=4, seed=11
    )
    for index in range(4):
        assert torch.equal(
            first[index].reference.embeddings, second[index].reference.embeddings
        )
        assert torch.equal(
            first[index].query.embeddings, second[index].query.embeddings
        )
        assert torch.equal(first[index].targets_sec, second[index].targets_sec)

    absent = SyntheticDetectionDataset(
        1,
        feature_dim=8,
        query_patches=24,
        reference_patches=4,
        target_present_probability=0.0,
    )[0]
    assert absent.targets_sec.shape == (0, 2)


def test_short_smoke_training_reports_loss_metrics_and_gradient_health():
    dataset = SyntheticDetectionDataset(
        12, feature_dim=8, query_patches=24, reference_patches=4, seed=19
    )
    batch = collate_detection_episodes([dataset[index] for index in range(12)])
    model = DifferentiableSubsequenceMatcher(8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    results = [train_step(model, batch, optimizer) for _ in range(4)]

    assert all(math.isfinite(result.loss) for result in results)
    telemetry = results[-1].telemetry
    assert telemetry["gradient_finite"] == 1.0
    assert telemetry["parameters_with_gradient_fraction"] == 1.0
    assert telemetry["parameters_with_nonzero_gradient_fraction"] == 1.0
    assert telemetry["grad_norm_projection"] > 0
    assert telemetry["grad_norm_score_bias"] > 0
    assert 0 <= telemetry["endpoint_f1"] <= 1
    assert 0 <= telemetry["target_absent_false_alarm_rate"] <= 1
    assert results[-1].loss < results[0].loss


def test_event_metrics_score_boundaries_and_target_absent_false_alarms():
    match = TemporalMatch(2, 5, 2.0, 5.0, 0.1, 3, 1.0)
    metrics = event_detection_metrics(
        [match], torch.tensor([[2.0, 5.0]]), query_duration_sec=120.0
    )
    assert metrics["event_f1"] == 1.0
    assert metrics["matched_event_count"] == 1.0
    assert metrics["event_average_precision"] == 1.0
    assert metrics["event_recall_at_operating_point"] == 1.0
    assert metrics["mean_absolute_count_error"] == 0.0
    assert metrics["mean_onset_error_sec"] == 0.0
    assert metrics["false_alarms_per_hour"] == 0.0

    absent = event_detection_metrics(
        [match], torch.empty(0, 2), query_duration_sec=120.0
    )
    assert absent["event_f1"] == 0.0
    assert absent["event_average_precision"] == 0.0
    assert absent["false_alarms_per_hour"] == 30.0


def test_calibration_prefix_matches_event_metric_at_every_threshold():
    matches = [
        TemporalMatch(0, 2, 0.0, 2.0, 0.1, 2, 1.0),
        TemporalMatch(1, 3, 1.0, 3.0, 0.2, 2, 1.0),
        TemporalMatch(5, 7, 5.0, 7.0, 0.3, 2, 1.0),
        TemporalMatch(8, 10, 8.0, 10.0, 0.4, 2, 1.0),
    ]
    targets = torch.tensor([[0.0, 2.0], [5.0, 7.0]])
    scores, cumulative, target_count = _event_prefix(
        matches, targets, iou_threshold=0.5
    )

    for threshold in (np.nextafter(0.1, -np.inf), 0.1, 0.2, 0.3, 0.4):
        detection_count = int(np.searchsorted(scores, threshold, side="right"))
        true_positive = int(cumulative[detection_count - 1]) if detection_count else 0
        fast = (
            true_positive,
            detection_count - true_positive,
            target_count - true_positive,
        )
        exact = event_detection_metrics(
            matches,
            targets,
            query_duration_sec=10.0,
            iou_threshold=0.5,
            score_threshold=threshold,
        )
        assert fast == (
            int(exact["true_positive_count"]),
            int(exact["false_positive_count"]),
            int(exact["false_negative_count"]),
        )
