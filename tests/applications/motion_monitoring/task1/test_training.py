import math

import torch

from applications.motion_monitoring.task1 import (
    DifferentiableSubsequenceMatcher,
    SyntheticDetectionDataset,
    collate_detection_episodes,
    event_detection_metrics,
    train_step,
)
from applications.motion_monitoring.task1.matcher import TemporalMatch


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
    assert metrics["mean_onset_error_sec"] == 0.0
    assert metrics["false_alarms_per_hour"] == 0.0

    absent = event_detection_metrics(
        [match], torch.empty(0, 2), query_duration_sec=120.0
    )
    assert absent["event_f1"] == 0.0
    assert absent["false_alarms_per_hour"] == 30.0
