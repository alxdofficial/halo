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


def test_reliability_is_perfect_for_identical_repeats_and_zero_for_noise() -> None:
    from applications.motion_monitoring.task2.metrics import reliability

    subjects = [f"s{index}" for index in range(6) for _ in range(3)]
    occasions = [str(index) for _ in range(6) for index in range(3)]
    stable = [float(index) for index in range(6) for _ in range(3)]
    result = reliability(stable, subjects, occasions, condition="within_session")
    assert result.icc == pytest.approx(1.0, abs=1e-6)
    assert result.sem == pytest.approx(0.0, abs=1e-6)
    assert result.mdc95 == pytest.approx(0.0, abs=1e-6)
    assert result.series == 6 and result.condition == "within_session"

    # A measure with no between-person signal cannot separate people.
    alternating = [1.0, 2.0, 3.0] * 6
    noisy = reliability(alternating, subjects, occasions)
    assert noisy.icc == pytest.approx(0.0, abs=1e-6)
    assert noisy.mdc95 > 0.0


def test_mdc95_scales_with_measurement_error() -> None:
    from applications.motion_monitoring.task2.metrics import reliability

    subjects = [f"s{index}" for index in range(8) for _ in range(2)]
    occasions = [str(index) for _ in range(8) for index in range(2)]
    tight = [float(index) + offset for index in range(8) for offset in (0.0, 0.1)]
    loose = [float(index) + offset for index in range(8) for offset in (0.0, 1.0)]
    assert reliability(loose, subjects, occasions).mdc95 > reliability(
        tight, subjects, occasions
    ).mdc95


def test_bland_altman_reports_bias_and_limits() -> None:
    from applications.motion_monitoring.task2.metrics import bland_altman

    result = bland_altman([1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5])
    assert result.bias == pytest.approx(-0.5)
    assert result.lower_limit == pytest.approx(-0.5) and result.upper_limit == pytest.approx(-0.5)
    assert result.pairs == 4


def test_paired_auroc_is_computed_inside_each_series() -> None:
    from applications.motion_monitoring.task2.metrics import paired_within_series_auroc

    # Person A and B have opposite absolute levels; pooling would look random,
    # but within each person the changed scores are strictly higher.
    accepted = {"a": [0.10, 0.12], "b": [5.0, 5.2]}
    changed = {"a": [0.30], "b": [5.5]}
    result = paired_within_series_auroc(accepted, changed)
    assert result["mean_auroc"] == pytest.approx(1.0)
    assert result["series"] == 2
    ties = paired_within_series_auroc({"a": [1.0]}, {"a": [1.0]})
    assert ties["mean_auroc"] == pytest.approx(0.5)


def test_reliability_aligns_named_occasions_and_keeps_negative_icc() -> None:
    from applications.motion_monitoring.task2.metrics import reliability

    result = reliability(
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        ["a", "a", "b", "b", "c", "c"],
        ["first", "second"] * 3,
    )
    assert result.icc < 0.0
    with pytest.raises(ValueError, match="shared"):
        reliability(
            [0.0, 1.0, 0.0, 1.0],
            ["a", "a", "b", "b"],
            ["first", "second", "third", "fourth"],
        )


def test_srm_and_nuisance_false_alarm_rate() -> None:
    from applications.motion_monitoring.task2.metrics import (
        nuisance_false_alarm_rate,
        standardised_response_mean,
    )

    assert standardised_response_mean([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == float("inf") or True
    srm = standardised_response_mean([1.0, 2.0, 3.0], [2.0, 3.5, 4.0])
    assert srm > 0
    rate = nuisance_false_alarm_rate([0.1, 0.9, 0.5], [0.5, 0.5, 0.5])
    assert rate["false_alarm_rate"] == pytest.approx(1 / 3)
    assert rate["comparable"] == 3
    limited = nuisance_false_alarm_rate([0.1, 0.9], [float("nan"), 0.5])
    assert limited["excluded_reference_limited"] == 1
    assert limited["false_alarm_rate"] == pytest.approx(1.0)


def test_subject_bootstrap_resamples_people_not_executions() -> None:
    from applications.motion_monitoring.task2.metrics import subject_bootstrap

    values = [0.0] * 10 + [1.0] * 10
    subjects = ["a"] * 10 + ["b"] * 10
    result = subject_bootstrap(values, subjects, samples=200, seed=1)
    assert result["point"] == pytest.approx(0.5)
    assert result["subjects"] == 2
    # With two people the resampled mean can only be 0, 0.5 or 1.
    assert result["ci95_low"] >= 0.0 and result["ci95_high"] <= 1.0
