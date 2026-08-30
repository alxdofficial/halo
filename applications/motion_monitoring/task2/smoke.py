"""Short synthetic Task-2 trainer for mechanical and gradient verification."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from .contracts import (
    BoundedExecution,
    ChangeTargetSpec,
    ExecutionPair,
    ExecutionPairDataset,
    collate_execution_pairs,
)
from .losses import ChangeLossConfig, change_quantification_loss
from .metrics import balanced_accuracy, binary_auroc, masked_regression_metrics
from .model import ChangeMetricHead
from .training import StepTelemetry, initialize_change_threshold, train_step


SYNTHETIC_TARGETS = (
    ChangeTargetSpec("duration_log_ratio", 0.25, "log-ratio"),
    ChangeTargetSpec("acc_intensity_log_ratio", 0.30, "log-ratio"),
    ChangeTargetSpec("gyro_intensity_log_ratio", 0.30, "log-ratio"),
    ChangeTargetSpec("pause_fraction_delta", 0.20, "fraction"),
)


@dataclass(frozen=True)
class SmokeResult:
    steps: int
    initial_loss: float
    final_loss: float
    auroc: float
    balanced_accuracy: float
    regression_mae: dict[str, float]
    max_grad_norm: float
    min_active_parameter_grad_norm: float
    nonfinite_gradients: int
    updated_parameter_count: int


def _trajectory(length: int, width: int, phase_shift: float = 0.0) -> torch.Tensor:
    phase = torch.linspace(0.0, 1.0, length)
    columns = []
    for index in range(width):
        frequency = 1 + index % 4
        columns.append(torch.sin(2 * torch.pi * frequency * (phase + phase_shift)))
    return torch.stack(columns, dim=-1)


def _execution(
    values: torch.Tensor,
    *,
    subject: int,
    execution: str,
    duration: float,
) -> BoundedExecution:
    count = len(values)
    edges = torch.linspace(0.0, duration, count + 1)
    intervals = torch.stack((edges[:-1], edges[1:]), dim=-1)
    return BoundedExecution(
        embeddings=values.float(),
        patch_intervals_sec=intervals.float(),
        patch_mask=torch.ones(count, dtype=torch.bool),
        dataset="synthetic_task2",
        subject_id=f"subject_{subject}",
        session_id=f"session_{execution}",
        execution_id=f"subject_{subject}_{execution}",
        task_id="repeated_reach",
    )


def make_synthetic_pairs(
    *,
    subjects: int = 16,
    embedding_dim: int = 12,
    seed: int = 7,
) -> list[ExecutionPair]:
    """Create independent repetitions with nuisance-only and known-change pairs."""

    if subjects < 2 or embedding_dim < 4:
        raise ValueError(
            "synthetic smoke requires at least two subjects and four features"
        )
    generator = torch.Generator().manual_seed(seed)
    pairs: list[ExecutionPair] = []
    for subject in range(subjects):
        length = 10 + subject % 5
        subject_offset = 0.05 * torch.randn(embedding_dim, generator=generator)
        reference_values = _trajectory(length, embedding_dim) + subject_offset
        reference_values += 0.015 * torch.randn(
            reference_values.shape, generator=generator
        )
        reference = _execution(
            reference_values,
            subject=subject,
            execution="reference",
            duration=4.0 + 0.1 * (subject % 3),
        )

        accepted_values = _trajectory(length + (subject % 2), embedding_dim, 0.005)
        accepted_values += subject_offset
        accepted_values += 0.025 * torch.randn(
            accepted_values.shape, generator=generator
        )
        accepted = _execution(
            accepted_values,
            subject=subject,
            execution="accepted",
            duration=4.05 + 0.1 * (subject % 3),
        )
        pairs.append(
            ExecutionPair(
                reference=reference,
                comparison=accepted,
                pair_kind="accepted_variation",
                change_targets=torch.tensor([0.01, 0.01, -0.01, 0.0]),
                target_mask=torch.ones(4, dtype=torch.bool),
                target_specs=SYNTHETIC_TARGETS,
            )
        )

        changed_values = _trajectory(length + 2, embedding_dim, 0.03) + subject_offset
        phase_start = len(changed_values) // 3
        phase_end = 2 * len(changed_values) // 3
        changed_values[phase_start:phase_end, :4] *= 0.35
        changed_values[phase_start:phase_end, 4:8] += 0.55
        changed_values += 0.02 * torch.randn(changed_values.shape, generator=generator)
        changed = _execution(
            changed_values,
            subject=subject,
            execution="changed",
            duration=5.0 + 0.1 * (subject % 3),
        )
        pairs.append(
            ExecutionPair(
                reference=reference,
                comparison=changed,
                pair_kind="known_change",
                change_targets=torch.tensor([0.22, -0.28, 0.24, 0.18]),
                target_mask=torch.tensor([True, True, subject % 4 != 0, True]),
                target_specs=SYNTHETIC_TARGETS,
            )
        )
    return pairs


@torch.no_grad()
def _evaluate(
    model: ChangeMetricHead, pairs: Sequence[ExecutionPair]
) -> dict[str, object]:
    batch = collate_execution_pairs(pairs)
    model.eval()
    output = model(batch)
    loss = change_quantification_loss(output, batch)
    regression = masked_regression_metrics(
        output.target_predictions,
        batch.change_targets,
        batch.target_mask,
        batch.target_names,
    )
    return {
        "loss": float(loss.total),
        "auroc": binary_auroc(
            output.change_scores,
            batch.classification_targets,
            batch.classification_mask,
        ),
        "balanced_accuracy": balanced_accuracy(
            output.change_logits,
            batch.classification_targets,
            batch.classification_mask,
        ),
        "regression_mae": regression.mae,
    }


def run_synthetic_smoke(
    *,
    steps: int = 40,
    batch_size: int = 8,
    embedding_dim: int = 12,
    device: str = "cpu",
    seed: int = 7,
) -> SmokeResult:
    if steps <= 0 or batch_size <= 1:
        raise ValueError("steps must be positive and batch size must exceed one")
    torch.manual_seed(seed)
    pairs = make_synthetic_pairs(embedding_dim=embedding_dim, seed=seed)
    train_pairs = [
        pair for pair in pairs if int(pair.reference.subject_id.split("_")[-1]) < 12
    ]
    validation_pairs = [
        pair for pair in pairs if int(pair.reference.subject_id.split("_")[-1]) >= 12
    ]
    loader = DataLoader(
        ExecutionPairDataset(train_pairs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_execution_pairs,
        generator=torch.Generator().manual_seed(seed),
    )
    model = ChangeMetricHead(
        embedding_dim=embedding_dim,
        target_dim=len(SYNTHETIC_TARGETS),
        phase_bins=8,
    ).to(device)
    initialize_change_threshold(model, collate_execution_pairs(train_pairs).to(device))
    initial_loss = float(_evaluate(model.cpu(), train_pairs)["loss"])
    model.to(device)
    initial_parameters = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_config = ChangeLossConfig(classification_weight=1.0, regression_weight=1.0)
    telemetry: list[StepTelemetry] = []
    iterator = iter(loader)
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        telemetry.append(
            train_step(
                model,
                batch.to(device),
                optimizer,
                loss_config=loss_config,
                grad_clip=5.0,
            )
        )
    evaluation = _evaluate(model.cpu(), validation_pairs)
    final_loss = float(_evaluate(model, train_pairs)["loss"])
    active_gradients = [
        value
        for step_telemetry in telemetry
        for value in step_telemetry.parameter_grad_norms.values()
        if value > 0
    ]
    updated = sum(
        not torch.equal(initial_parameters[name].cpu(), value.detach().cpu())
        for name, value in model.named_parameters()
    )
    return SmokeResult(
        steps=steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        auroc=float(evaluation["auroc"]),
        balanced_accuracy=float(evaluation["balanced_accuracy"]),
        regression_mae=dict(evaluation["regression_mae"]),
        max_grad_norm=max(item.total_grad_norm_preclip for item in telemetry),
        min_active_parameter_grad_norm=min(active_gradients),
        nonfinite_gradients=sum(item.nonfinite_gradients for item in telemetry),
        updated_parameter_count=updated,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(
        json.dumps(
            asdict(
                run_synthetic_smoke(
                    steps=args.steps, batch_size=args.batch_size, device=args.device
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
