"""Task-2 data contract tests."""

from __future__ import annotations

import pytest
import torch

from applications.motion_monitoring.task2.contracts import (
    BoundedExecution,
    ChangeTargetSpec,
    ExecutionPair,
    collate_execution_pairs,
)


SCHEMA = (ChangeTargetSpec("duration", 0.5, "seconds"),)


def execution(
    execution_id: str,
    *,
    length: int = 3,
    subject: str = "s1",
    invalid_last: bool = False,
) -> BoundedExecution:
    values = torch.arange(length * 4, dtype=torch.float32).reshape(length, 4) + 1
    mask = torch.ones(length, dtype=torch.bool)
    if invalid_last:
        mask[-1] = False
        values[-1] = float("nan")
    edges = torch.arange(length + 1, dtype=torch.float32)
    return BoundedExecution(
        embeddings=values,
        patch_intervals_sec=torch.stack((edges[:-1], edges[1:]), dim=-1),
        patch_mask=mask,
        dataset="unit",
        subject_id=subject,
        session_id=execution_id,
        execution_id=execution_id,
        task_id="reach",
    )


def pair(reference: BoundedExecution, comparison: BoundedExecution) -> ExecutionPair:
    return ExecutionPair(
        reference=reference,
        comparison=comparison,
        pair_kind="known_change",
        change_targets=torch.tensor([0.2]),
        target_mask=torch.tensor([True]),
        target_specs=SCHEMA,
    )


def test_collate_preserves_variable_lengths_and_sanitizes_invalid_values() -> None:
    first = pair(
        execution("r1", length=2), execution("c1", length=4, invalid_last=True)
    )
    second = pair(execution("r2", length=3), execution("c2", length=2))
    batch = collate_execution_pairs([first, second])

    assert batch.reference_embeddings.shape == (2, 3, 4)
    assert batch.comparison_embeddings.shape == (2, 4, 4)
    assert batch.reference_mask.tolist() == [[True, True, False], [True, True, True]]
    assert batch.comparison_mask.tolist() == [
        [True, True, True, False],
        [True, True, False, False],
    ]
    assert torch.isfinite(batch.comparison_embeddings).all()
    assert torch.equal(
        batch.comparison_embeddings[~batch.comparison_mask],
        torch.zeros_like(batch.comparison_embeddings[~batch.comparison_mask]),
    )


def test_pair_requires_independent_within_subject_same_task_executions() -> None:
    with pytest.raises(ValueError, match="within subject"):
        pair(execution("r", subject="s1"), execution("c", subject="s2"))
    with pytest.raises(ValueError, match="independent"):
        pair(execution("same"), execution("same"))


def test_unlabeled_pair_has_no_classification_target_but_can_carry_measurements() -> (
    None
):
    unlabeled = ExecutionPair(
        reference=execution("r"),
        comparison=execution("c"),
        pair_kind="unlabeled",
        change_targets=torch.tensor([0.1]),
        target_mask=torch.tensor([True]),
        target_specs=SCHEMA,
    )
    batch = collate_execution_pairs([unlabeled])
    assert not bool(batch.classification_mask[0])
    assert bool(batch.target_mask[0, 0])


def test_collate_preserves_large_absolute_clock_precision() -> None:
    reference = execution("r").__class__(
        embeddings=execution("r").embeddings,
        patch_intervals_sec=execution("r").patch_intervals_sec.double() + 1_634_178_333.0,
        patch_mask=execution("r").patch_mask,
        dataset="unit",
        subject_id="s1",
        session_id="r",
        execution_id="r",
        task_id="reach",
    )
    batch = collate_execution_pairs([pair(reference, execution("c"))])
    assert batch.reference_intervals_sec.dtype == torch.float64
    assert torch.all(
        batch.reference_intervals_sec[0, :, 1]
        > batch.reference_intervals_sec[0, :, 0]
    )
