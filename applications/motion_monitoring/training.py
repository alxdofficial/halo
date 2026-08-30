"""Low-overhead training health checks shared by the three application tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping

import torch
import torch.nn as nn


def optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> tuple[nn.Parameter, ...]:
    """Return each trainable optimizer parameter once, preserving group order."""

    seen: set[int] = set()
    parameters: list[nn.Parameter] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not isinstance(parameter, nn.Parameter) or not parameter.requires_grad:
                continue
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                parameters.append(parameter)
    return tuple(parameters)


def parameter_grad_norm(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    gradients = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return torch.zeros(())
    return torch.stack([gradient.square().sum() for gradient in gradients]).sum().sqrt()


@dataclass(frozen=True)
class GradientHealth:
    loss: float
    total_grad_norm: float
    max_parameter_grad_norm: float
    parameters_with_grad: int
    trainable_parameters: int
    finite: bool
    missing_gradient_names: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def gradient_health(model: nn.Module, loss: torch.Tensor) -> GradientHealth:
    """Summarize gradients after ``backward`` without synchronizing per tensor."""

    trainable: list[tuple[str, nn.Parameter]] = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    missing = tuple(name for name, parameter in trainable if parameter.grad is None)
    gradients = [parameter.grad.detach() for _, parameter in trainable if parameter.grad is not None]
    if gradients:
        squared = torch.stack(
            [gradient.float().square().sum() for gradient in gradients]
        )
        norms = torch.stack([gradient.float().norm() for gradient in gradients])
        total = squared.sum().sqrt()
        maximum = norms.max()
        finite = torch.isfinite(squared).all() & torch.isfinite(loss.detach())
    else:
        device = loss.device
        total = torch.zeros((), device=device)
        maximum = torch.zeros((), device=device)
        finite = torch.isfinite(loss.detach())
    # One transfer covers the scalar telemetry; this is a smoke/debug path, not per-step logging.
    values = torch.stack(
        [loss.detach().float(), total.float(), maximum.float(), finite.float()]
    ).cpu()
    return GradientHealth(
        loss=float(values[0]),
        total_grad_norm=float(values[1]),
        max_parameter_grad_norm=float(values[2]),
        parameters_with_grad=len(gradients),
        trainable_parameters=len(trainable),
        finite=bool(values[3]),
        missing_gradient_names=missing,
    )


def smoke_train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: Callable[[int], tuple[torch.Tensor, Mapping[str, float]]],
    *,
    steps: int = 3,
    grad_clip: float | None = 5.0,
) -> list[dict[str, object]]:
    """Run a few real optimizer steps and return auditable loss/gradient telemetry."""

    if steps <= 0:
        raise ValueError("smoke training requires at least one step")
    if grad_clip is not None and grad_clip <= 0:
        raise ValueError("gradient clipping threshold must be positive")
    history: list[dict[str, object]] = []
    model.train()
    for index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = step(index)
        if loss.ndim != 0:
            raise ValueError("training step must return a scalar loss")
        loss.backward()
        health = gradient_health(model, loss)
        if not health.finite:
            raise FloatingPointError(f"non-finite training state at smoke step {index}")
        if health.parameters_with_grad == 0:
            raise RuntimeError("smoke loss is disconnected from every trainable parameter")
        preclip_norm = health.total_grad_norm
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        history.append(
            {
                "step": index,
                **health.as_dict(),
                "preclip_grad_norm": preclip_norm,
                **{name: float(value) for name, value in metrics.items()},
            }
        )
    return history
