from __future__ import annotations

import math

import torch

from applications.motion_monitoring.task2.metrics import (
    balanced_accuracy,
    binary_auroc,
    masked_regression_metrics,
)


def test_binary_metrics_handle_ties_masks_and_degenerate_classes() -> None:
    assert (
        binary_auroc(torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.tensor([0, 0, 1, 1]))
        == 1.0
    )
    assert binary_auroc(torch.ones(4), torch.tensor([0, 0, 1, 1])) == 0.5
    assert math.isclose(
        balanced_accuracy(
            torch.tensor([-2.0, 1.0, -1.0, 2.0]), torch.tensor([0, 1, 1, 1])
        ),
        5 / 6,
        rel_tol=1e-6,
    )
    assert math.isnan(binary_auroc(torch.ones(2), torch.ones(2)))


def test_regression_metrics_are_per_target_and_masked() -> None:
    metrics = masked_regression_metrics(
        torch.tensor([[1.0, 100.0], [3.0, 4.0]]),
        torch.tensor([[0.0, 0.0], [1.0, 2.0]]),
        torch.tensor([[True, False], [True, True]]),
        ("duration", "intensity"),
    )
    assert metrics.mae == {"duration": 1.5, "intensity": 2.0}
    assert metrics.counts == {"duration": 2, "intensity": 1}
    assert math.isclose(metrics.rmse["duration"], math.sqrt(2.5), rel_tol=1e-6)
