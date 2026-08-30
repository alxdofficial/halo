from __future__ import annotations

from dataclasses import replace

import torch

from applications.motion_monitoring.task2.contracts import (
    BoundedExecution,
    ChangeTargetSpec,
    ExecutionPair,
    collate_execution_pairs,
)
from applications.motion_monitoring.task2.losses import (
    ChangeLossConfig,
    change_quantification_loss,
)
from applications.motion_monitoring.task2.model import (
    ChangeHeadOutput,
    ChangeMetricHead,
    resample_to_phase,
)
from applications.motion_monitoring.task2.training import (
    initialize_change_threshold,
    train_step,
)


SCHEMA = (
    ChangeTargetSpec("duration", 2.0, "seconds"),
    ChangeTargetSpec("intensity", 0.5, "log-ratio"),
)


def execution(execution_id: str, values: torch.Tensor) -> BoundedExecution:
    edges = torch.arange(len(values) + 1, dtype=torch.float32)
    return BoundedExecution(
        embeddings=values.float(),
        patch_intervals_sec=torch.stack((edges[:-1], edges[1:]), dim=-1),
        patch_mask=torch.ones(len(values), dtype=torch.bool),
        dataset="unit",
        subject_id="s1",
        session_id=execution_id,
        execution_id=execution_id,
        task_id="reach",
    )


def make_pair(kind: str, offset: float, suffix: str) -> ExecutionPair:
    reference = torch.tensor(
        [[1.0, 0.2, 0.1, 0.4], [0.4, 1.0, 0.2, 0.1], [0.2, 0.3, 1.0, 0.5]]
    )
    comparison = reference + offset * torch.tensor(
        [[0.0, 1.0, -0.5, 0.3], [0.8, 0.0, 0.4, -0.2], [-0.4, 0.6, 0.0, 0.5]]
    )
    return ExecutionPair(
        reference=execution(f"r{suffix}", reference),
        comparison=execution(f"c{suffix}", comparison),
        pair_kind=kind,
        change_targets=torch.tensor([offset * 2.0, offset * 0.5]),
        target_mask=torch.tensor([True, suffix != "masked"]),
        target_specs=SCHEMA,
    )


def test_phase_resampling_uses_physical_time_and_handles_one_patch() -> None:
    values = torch.tensor([[[0.0], [2.0], [4.0]]])
    intervals = torch.tensor([[[0.0, 1.0], [1.0, 3.0], [3.0, 5.0]]])
    result = resample_to_phase(
        values, intervals, torch.ones(1, 3, dtype=torch.bool), bins=4
    )
    assert result.shape == (1, 4, 1)
    assert torch.allclose(result[0, [0, -1], 0], torch.tensor([0.0, 4.0]))

    single = resample_to_phase(
        values[:, :1], intervals[:, :1], torch.ones(1, 1, dtype=torch.bool), bins=4
    )
    assert torch.equal(single, torch.zeros(1, 4, 1))


def test_masked_regression_ignores_missing_targets_and_respects_scales() -> None:
    batch = collate_execution_pairs([make_pair("accepted_variation", 0.05, "masked")])
    output = ChangeHeadOutput(
        change_logits=torch.tensor([0.0]),
        change_scores=torch.tensor([0.0]),
        target_predictions=torch.tensor([[2.1, 1_000.0]]),
        phase_residuals=torch.zeros(1, 4),
        reference_phase=torch.zeros(1, 4, 4),
        comparison_phase=torch.zeros(1, 4, 4),
    )
    loss = change_quantification_loss(
        output,
        batch,
        ChangeLossConfig(classification_weight=0.0, regression_weight=1.0),
    )
    # Target zero is 0.1, so the scaled error is exactly one: Huber(1) = 0.5.
    assert torch.allclose(loss.regression, torch.tensor(0.5))
    assert loss.regression_count == 1


def test_padding_values_do_not_change_outputs() -> None:
    short = make_pair("accepted_variation", 0.02, "a")
    short = replace(
        short,
        reference=execution("ra_short", short.reference.embeddings[:2]),
        comparison=execution("ca_short", short.comparison.embeddings[:2]),
    )
    pairs = [short, make_pair("known_change", 0.5, "b")]
    batch = collate_execution_pairs(pairs)
    model = ChangeMetricHead(embedding_dim=4, target_dim=2, phase_bins=5)
    expected = model(batch)
    corrupted = replace(
        batch,
        reference_embeddings=batch.reference_embeddings.masked_fill(
            ~batch.reference_mask.unsqueeze(-1), 1e9
        ),
        comparison_embeddings=batch.comparison_embeddings.masked_fill(
            ~batch.comparison_mask.unsqueeze(-1), -1e9
        ),
    )
    actual = model(corrupted)
    assert torch.allclose(actual.change_logits, expected.change_logits)
    assert torch.allclose(actual.target_predictions, expected.target_predictions)


def test_mixed_batch_updates_every_trainable_component_with_finite_gradients() -> None:
    batch = collate_execution_pairs(
        [
            make_pair("accepted_variation", 0.02, "a"),
            make_pair("known_change", 0.6, "b"),
            make_pair("known_change", 0.4, "c"),
        ]
    )
    model = ChangeMetricHead(embedding_dim=4, target_dim=2, phase_bins=5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    telemetry = train_step(model, batch, optimizer)

    assert telemetry.nonfinite_gradients == 0
    assert telemetry.total_grad_norm_preclip > 0
    assert 0 < telemetry.clip_coefficient <= 1
    assert all(value > 0 for value in telemetry.parameter_grad_norms.values())
    assert all(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )


def test_threshold_initialization_uses_training_pair_medians() -> None:
    batch = collate_execution_pairs(
        [
            make_pair("accepted_variation", 0.02, "a"),
            make_pair("known_change", 0.6, "b"),
        ]
    )
    model = ChangeMetricHead(embedding_dim=4, target_dim=2, phase_bins=5)
    threshold = initialize_change_threshold(model, batch)
    logits = model(batch).change_logits
    assert threshold > 0
    assert logits[0] < 0 < logits[1]


def test_half_precision_head_preserves_output_dtype_and_finiteness() -> None:
    batch = collate_execution_pairs(
        [
            make_pair("accepted_variation", 0.02, "a"),
            make_pair("known_change", 0.6, "b"),
        ]
    )
    batch = replace(
        batch,
        reference_embeddings=batch.reference_embeddings.half(),
        comparison_embeddings=batch.comparison_embeddings.half(),
    )
    model = ChangeMetricHead(embedding_dim=4, target_dim=2, phase_bins=5).half()
    output = model(batch)
    assert output.target_predictions.dtype == torch.float16
    assert torch.isfinite(output.target_predictions).all()
    assert torch.isfinite(output.change_logits).all()
