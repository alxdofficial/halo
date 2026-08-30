from __future__ import annotations

from applications.motion_monitoring.task2.smoke import run_synthetic_smoke


def test_synthetic_smoke_trains_and_reports_gradient_health() -> None:
    result = run_synthetic_smoke(steps=50, batch_size=8, seed=13)
    assert result.final_loss < result.initial_loss
    assert result.auroc >= 0.9
    assert result.max_grad_norm > 0
    assert result.min_active_parameter_grad_norm > 0
    assert result.nonfinite_gradients == 0
    assert result.updated_parameter_count == 5
