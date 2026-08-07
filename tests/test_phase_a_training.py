"""Focused tests for Phase-A evaluation and resume control paths."""

import pytest
import torch

from training.tokenizer.pretrain import (
    PretrainConfig,
    hydrate_calibrated_objective_weights,
    label_group_balanced_acc,
    prepare_output_dir,
    stratified_eval_subset,
)
from training.tokenizer.pretrain_data import WindowKey


def test_eval_subset_covers_streams_before_refilling_large_source():
    keys = (
        [WindowKey(0, i, 7) for i in range(100)]
        + [WindowKey(1, i, 7) for i in range(2)]
        + [WindowKey(2, 0, 7)]
    )
    chosen = stratified_eval_subset(keys, per_label=6, seed=3)
    assert len(chosen) == 6
    assert {key.stream_i for key in chosen} == {0, 1, 2}
    assert chosen == stratified_eval_subset(keys, per_label=6, seed=3)


def test_label_group_metric_does_not_let_large_cell_hide_failure():
    true = torch.tensor([0] * 10 + [0])
    pred = torch.tensor([0] * 10 + [1])
    groups = ["large"] * 10 + ["small"]
    assert label_group_balanced_acc(pred, true, groups) == pytest.approx(0.5)


def test_resume_hydrates_applied_calibration_weights_as_frozen_state():
    cfg = PretrainConfig()
    saved = {
        "objective_calibration_mode": "apply",
        "objective_calibration_at": 2000,
        "objective_calibration_batches": 73,
        "objective_target_jepa_share": 0.4,
        "jepa_weight": 19.0,
        "vicreg_weight": 0.67,
    }
    applied = hydrate_calibrated_objective_weights(cfg, saved, saved_step=2500)
    assert applied
    assert cfg.jepa_weight == 19.0
    assert cfg.vicreg_weight == 0.67
    assert cfg.objective_calibration_at == 2000
    assert cfg.objective_calibration_batches == 73
    assert cfg.objective_target_jepa_share == 0.4
    assert cfg.objective_calibration_mode == "apply"


def test_resume_hydrates_calibration_schedule_before_it_has_run():
    cfg = PretrainConfig()
    saved = {
        **vars(PretrainConfig()),
        "objective_calibration_mode": "apply",
        "objective_calibration_at": 2000,
        "objective_calibration_batches": 50,
    }
    assert hydrate_calibrated_objective_weights(cfg, saved, saved_step=1000)
    assert cfg.objective_calibration_at == 2000
    assert cfg.objective_calibration_mode == "apply"


def test_resume_leaves_explicit_calibration_override_for_validation_to_reject():
    cfg = PretrainConfig(objective_calibration_at=7)
    hydrate_calibrated_objective_weights(
        cfg, {"objective_calibration_at": 2000}, saved_step=100,
        explicit_fields={"objective_calibration_at"},
    )
    assert cfg.objective_calibration_at == 7


def test_force_removes_all_known_run_artifacts_but_preserves_unknown_files(tmp_path):
    for name in (
        "last.pt", "log.jsonl", "run_config.json", "objective_calibration.json",
        "objective_calibration_resolved.json", "source.patch", "source_provenance.json",
        "runtime_provenance.json", "health.json", "health.txt", "telemetry.png",
    ):
        (tmp_path / name).write_text("stale")
    (tmp_path / "notes.txt").write_text("keep")

    prepare_output_dir(tmp_path, force=True, smoke=False, resume=False)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["notes.txt"]
