"""Tests for the CPU-only Phase-A health monitor and dashboard."""

from __future__ import annotations

import json
import os
import time

from training.tokenizer.monitor_training import assess, write_report


def _write_run(tmp_path, train_rows, val_rows=(), *, steps=100):
    (tmp_path / "run_config.json").write_text(json.dumps({
        "steps": steps, "warmup_steps": 10, "d_model": 256,
    }))
    with (tmp_path / "log.jsonl").open("w") as handle:
        for row in [*train_rows, *val_rows]:
            handle.write(json.dumps(row) + "\n")


def _healthy(step=100):
    return {
        "step": step,
        "total": 2.0,
        "jepa": 0.4,
        "vicreg": 1.6,
        "grad/total_preclip": 1.2,
        "grad/encoder": 0.8,
        "grad/clip_coefficient": 0.8,
        "grad_objective/jepa_share": 0.45,
        "repr_encoder/effective_rank": 40.0,
        "repr_encoder/min_std": 0.1,
        "repr_projector/effective_rank": 50.0,
        "jepa/margin": 0.2,
        "vicreg/margin": 0.2,
        "jepa_zero_target_frac_window": 0.0,
        "data/input_finite_fraction": 1.0,
        "data/batches_window": 50,
        "data/examples_window": 12_800,
        "data/source_share_window": {"a": 0.5, "b": 0.5},
        "data/source_share_target": {"a": 0.5, "b": 0.5},
        "perf/steps_per_s": 5.0,
        "perf/examples_per_s": 1280.0,
        "perf/eta_minutes": 0.0,
        "amp/skipped_updates_window": 0,
        "amp/skipped_updates_total": 0,
        "amp/consecutive_skips": 0,
    }


def test_monitor_uses_label_stream_checkpoint_metric(tmp_path):
    val = {"step": 100, "val_knn_ba": 0.91, "val_knn_label_stream_ba": 0.63}
    _write_run(tmp_path, [_healthy()], [val])
    (tmp_path / "last.pt").touch()
    report = assess(tmp_path)
    assert report["status"] == "green"
    assert report["validation"]["latest_selection"] == 0.63


def test_monitor_flags_invalid_input_and_source_drift(tmp_path):
    row = _healthy()
    row["data/input_finite_fraction"] = 0.999
    row["data/source_share_window"] = {"a": 0.8, "b": 0.2}
    _write_run(tmp_path, [row])
    report = assess(tmp_path)
    assert report["status"] == "critical"
    assert {item["code"] for item in report["alerts"]} >= {
        "nonfinite_input", "source_balance_drift",
    }


def test_monitor_detects_a_stale_incomplete_run(tmp_path):
    _write_run(tmp_path, [_healthy(step=20)], steps=100)
    old = time.time() - 300
    os.utime(tmp_path / "log.jsonl", (old, old))
    report = assess(tmp_path, stale_seconds=120)
    assert report["status"] == "critical"
    assert any(item["code"] == "stale_heartbeat" for item in report["alerts"])


def test_monitor_writes_machine_and_human_snapshots_and_plot(tmp_path):
    _write_run(tmp_path, [_healthy()])
    report = write_report(tmp_path, stale_seconds=120, render=True)
    assert report["status"] == "green"
    assert json.loads((tmp_path / "health.json").read_text())["schema_version"] == 1
    assert "STATUS: GREEN" in (tmp_path / "health.txt").read_text()
    assert (tmp_path / "telemetry.png").stat().st_size > 1_000
