from __future__ import annotations

import torch

from applications.motion_monitoring.training import smoke_train


def test_smoke_train_reports_connected_finite_gradients() -> None:
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    inputs = torch.eye(3)
    targets = torch.ones(3, 1)

    def step(_: int):
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        return loss, {"prediction_mean": model(inputs).detach().mean().item()}

    history = smoke_train(model, optimizer, step, steps=2)
    assert len(history) == 2
    assert all(row["finite"] for row in history)
    assert all(row["parameters_with_grad"] == 2 for row in history)
    assert all(row["total_grad_norm"] > 0 for row in history)
