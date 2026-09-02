from __future__ import annotations

import pytest

from applications.motion_monitoring.evaluation import (
    ApplicationEvaluation,
    fingerprint_protocol,
    render_markdown,
)


def _result(
    encoder: str,
    *,
    split: str = "test",
    protocol_fingerprint: str = "protocol",
) -> ApplicationEvaluation:
    protocol = {"version": protocol_fingerprint}
    return ApplicationEvaluation(
        task="task2",
        encoder=encoder,
        dataset="monipar",
        split=split,
        cohort_fingerprint="cohort",
        protocol_fingerprint=fingerprint_protocol(protocol),
        protocol=protocol,
        metrics={
            "change_sensitivity": 0.7,
            "change_specificity": 0.6,
            "change_balanced_accuracy": 0.65,
            "accepted_false_alarm_rate": 0.4,
        },
        counts={"subjects": 10, "pairs": 40},
        encoder_provenance={"kind": "released"},
    )


def test_comparison_requires_identical_protocol_dimensions() -> None:
    with pytest.raises(ValueError, match="different split"):
        render_markdown([_result("a"), _result("b", split="development")])
    with pytest.raises(ValueError, match="different protocol_fingerprint"):
        render_markdown([_result("a"), _result("b", protocol_fingerprint="other")])


def test_comparison_renders_primary_metrics() -> None:
    table = render_markdown([_result("harnet"), _result("unimts")])
    assert "change_sensitivity" in table
    assert "| harnet | 0.7000 | 0.6000 | 0.6500 | 0.4000 |" in table


def test_result_rejects_missing_primary_metric() -> None:
    with pytest.raises(ValueError, match="missing primary metrics"):
        ApplicationEvaluation(
            task="task3",
            encoder="harnet",
            dataset="openpack",
            split="test",
            cohort_fingerprint="cohort",
            protocol_fingerprint=fingerprint_protocol({"version": 1}),
            protocol={"version": 1},
            metrics={"occurrence_recall": 0.6},
            counts={"pairs": 10},
            encoder_provenance={"kind": "released"},
        )


def test_result_rejects_protocol_hash_that_does_not_match_record() -> None:
    with pytest.raises(ValueError, match="fingerprint does not match"):
        ApplicationEvaluation(
            task="task2",
            encoder="harnet",
            dataset="monipar",
            split="test",
            cohort_fingerprint="cohort",
            protocol_fingerprint="wrong",
            protocol={"version": 1},
            metrics={
                "change_sensitivity": 0.7,
                "change_specificity": 0.6,
                "change_balanced_accuracy": 0.65,
                "accepted_false_alarm_rate": 0.4,
            },
            counts={"subjects": 10},
            encoder_provenance={"kind": "released"},
        )
