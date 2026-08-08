"""Regression tests for run-safe Phase-B telemetry and health assessment."""

from __future__ import annotations

import json

from training.evidence.monitor_training import assess, write_report
from training.evidence.telemetry import PhaseBTelemetry
from training.evidence.policy import PHASE_B_DEV_DATASETS, PHASE_B_TEST_DATASETS
from data.scripts.curate.deployment_policy import PRIMARY_EVAL_DATASETS


def test_new_run_immediately_replaces_stale_latest_and_keeps_separate_history(tmp_path):
    first = PhaseBTelemetry(tmp_path, interval_seconds=60, run_id="first")
    first.start(step=10, elapsed_seconds=5, metadata={"planned_steps": 100})
    first.update({"loss": 1.0})
    first.emit(step=10, elapsed_seconds=5, force=True)

    second = PhaseBTelemetry(tmp_path, interval_seconds=60, run_id="second")
    second.start(step=0, elapsed_seconds=0, metadata={"planned_steps": 100})
    latest = json.loads((tmp_path / "phase_b_telemetry_latest.json").read_text())
    assert latest["run_id"] == "second"
    assert latest["event"] == "run_start"
    assert first.jsonl != second.jsonl
    assert first.jsonl.exists() and second.jsonl.exists()


def test_nonfinite_and_stratified_metrics_are_visible_to_monitor(tmp_path):
    telemetry = PhaseBTelemetry(tmp_path, interval_seconds=60, run_id="audit")
    telemetry.start(
        step=0,
        elapsed_seconds=0,
        metadata={"planned_steps": 100, "warmup_steps": 0, "grad_clip": 1.0},
    )
    telemetry.update(
        {
            "loss": 1.0,
            "bad": float("nan"),
            "decoder_grad_norm": 1.0,
            "retriever_grad_norm": 1.0,
            "hard_forward_max_abs_error": 0.0,
        },
        categories={"episode_type": "ordinary_few_support"},
        strata={"episode_type": "ordinary_few_support"},
    )
    payload = telemetry.emit(step=5, elapsed_seconds=2, force=True)
    assert payload["nonfinite"] == {"bad": 1}
    assert payload["strata"]["episode_type"]["ordinary_few_support"]["loss"]["mean"] == 1.0
    report = assess(tmp_path)
    assert report["status"] == "critical"
    assert any(item["code"] == "nonfinite_values" for item in report["alerts"])
    telemetry.update({"loss": 0.5})
    later = telemetry.emit(step=6, elapsed_seconds=3, force=True)
    assert later["nonfinite"] == {"bad": 1}
    final = telemetry.emit(step=6, elapsed_seconds=3, force=True, final=True)
    assert final["event"] == "run_end"
    assert final["metrics"] == later["metrics"]


def test_monitor_writes_human_machine_and_plot_outputs(tmp_path):
    telemetry = PhaseBTelemetry(tmp_path, interval_seconds=60, run_id="plot")
    telemetry.start(step=0, elapsed_seconds=0, metadata={"planned_steps": 10})
    telemetry.update({"loss": 1.0, "grad_norm": 0.2, "target_positive_rate": 0.5,
                      "predicted_confidence_mean": 0.5, "predicted_confidence_std": 0.1})
    telemetry.emit(step=1, elapsed_seconds=1, force=True)
    write_report(tmp_path, render=True)
    assert (tmp_path / "health.json").exists()
    assert "STATUS:" in (tmp_path / "health.txt").read_text()
    assert (tmp_path / "telemetry.png").stat().st_size > 1_000


def test_external_development_and_test_rosters_are_complete_and_disjoint():
    assert set(PHASE_B_DEV_DATASETS).isdisjoint(PHASE_B_TEST_DATASETS)
    assert set(PHASE_B_DEV_DATASETS) | set(PHASE_B_TEST_DATASETS) == set(PRIMARY_EVAL_DATASETS)
