from __future__ import annotations

import math

import pytest
import torch

from applications.motion_monitoring.task2.metrics import (
    balanced_accuracy,
    binary_auroc,
    binary_operating_metrics,
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
    operating = binary_operating_metrics(
        torch.tensor([-2.0, 1.0, -1.0, 2.0]), torch.tensor([0, 1, 1, 1])
    )
    assert operating["sensitivity"] == pytest.approx(2 / 3)
    assert operating["specificity"] == 1.0
    assert operating["false_positive_rate"] == 0.0
    assert operating["accepted_false_alarm_rate"] == 0.0
    assert operating["balanced_accuracy"] == pytest.approx(5 / 6)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_binary_auroc_keeps_rank_tensors_on_score_device() -> None:
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8], device="cuda")
    targets = torch.tensor([0, 1, 0, 1], device="cuda")

    assert binary_auroc(scores, targets) == pytest.approx(1.0)
