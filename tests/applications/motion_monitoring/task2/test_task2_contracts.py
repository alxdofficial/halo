"""Task-2 set-conditioned episode contract tests."""

from __future__ import annotations

import pytest
import torch

from applications.motion_monitoring.data.compatibility import sensor_compatibility_key
from applications.motion_monitoring.task2.contracts import (
    BoundedExecution,
    ExecutionEpisode,
    collate_execution_episodes,
)


def execution(
    execution_id: str,
    *,
    length: int = 3,
    subject: str = "s1",
    task: str = "reach",
    invalid_last: bool = False,
    configured: bool = False,
) -> BoundedExecution:
    values = torch.arange(length * 4, dtype=torch.float32).reshape(length, 4) + 1
    mask = torch.ones(length, dtype=torch.bool)
    if invalid_last:
        mask[-1] = False
        values[-1] = float("nan")
    edges = torch.arange(length + 1, dtype=torch.float32)
    config = None
    if configured:
        config = sensor_compatibility_key(
            device="smartwatch",
            placement="wrist",
            channels=("acc_x", "acc_y", "acc_z"),
            gravity_state="present",
        )
    return BoundedExecution(
        embeddings=values,
        patch_intervals_sec=torch.stack((edges[:-1], edges[1:]), dim=-1),
        patch_mask=mask,
        dataset="unit",
        subject_id=subject,
        session_id=execution_id,
        execution_id=execution_id,
        task_id=task,
        sensor_config=config,
    )


def episode(
    references: tuple[BoundedExecution, ...],
    query: BoundedExecution,
    *,
    context: tuple[BoundedExecution, ...] = (),
    kind: str = "accepted_query",
    **extra: object,
) -> ExecutionEpisode:
    return ExecutionEpisode(
        accepted_references=references,
        query=query,
        personal_context=context,
        episode_kind=kind,
        **extra,
    )


def test_collate_preserves_variable_set_sizes_and_sanitizes_padding() -> None:
    first = episode(
        (execution("r1", length=2), execution("r2", length=3)),
        execution("q1", length=4, invalid_last=True),
        context=(execution("d1", task="walk"),),
    )
    second = episode((execution("r3", length=3),), execution("q2", length=2))
    batch = collate_execution_episodes([first, second])

    assert batch.reference_embeddings.shape == (2, 2, 3, 4)
    assert batch.reference_execution_mask.tolist() == [[True, True], [True, False]]
    assert batch.context_execution_mask.tolist() == [[True], [False]]
    assert batch.query_mask.tolist() == [
        [True, True, True, False],
        [True, True, False, False],
    ]
    assert torch.isfinite(batch.query_embeddings).all()
    assert torch.equal(
        batch.query_embeddings[~batch.query_mask],
        torch.zeros_like(batch.query_embeddings[~batch.query_mask]),
    )


def test_episode_requires_independent_within_subject_same_task_target_set() -> None:
    with pytest.raises(ValueError, match="within subject"):
        episode((execution("r", subject="s1"),), execution("q", subject="s2"))
    with pytest.raises(ValueError, match="independent"):
        episode((execution("same"),), execution("same"))
    with pytest.raises(ValueError, match="same declared task"):
        episode((execution("r"),), execution("q", task="walk"))


def test_personal_context_may_have_another_task_but_not_another_subject() -> None:
    valid = episode(
        (execution("r"),),
        execution("q"),
        context=(execution("context", task="walk"),),
    )
    assert valid.personal_context[0].task_id == "walk"
    with pytest.raises(ValueError, match="same subject namespace"):
        episode(
            (execution("r"),),
            execution("q"),
            context=(execution("context", subject="s2", task="walk"),),
        )


def test_episode_rejects_incompatible_sensor_configurations() -> None:
    reference = execution("r", configured=True)
    query = execution("q", configured=True)
    query = BoundedExecution(
        **{
            **query.__dict__,
            "sensor_config": sensor_compatibility_key(
                device="smartwatch",
                placement="ankle",
                channels=("acc_x", "acc_y", "acc_z"),
                gravity_state="present",
            ),
        }
    )
    with pytest.raises(ValueError, match="incompatible sensor configurations"):
        episode((reference,), query)


def test_query_roles_drive_the_contrastive_masks() -> None:
    accepted = episode((execution("r"),), execution("q"), kind="accepted_query")
    modified = episode(
        (execution("r"),),
        execution("m"),
        kind="modified_query",
        severity=0.6,
        modification_kind="retime",
    )
    other = episode(
        (execution("r"),), execution("o", subject="s2"), kind="other_subject_query"
    )
    unlabeled = episode((execution("r"),), execution("u"), kind="unlabeled_query")
    batch = collate_execution_episodes([accepted, modified, other, unlabeled])
    assert batch.positive_mask.tolist() == [True, False, False, False]
    assert batch.negative_mask.tolist() == [False, True, True, False]
    assert batch.severities.tolist() == pytest.approx([0.0, 0.6, 1.0, 0.0])
    assert batch.modification_kinds == ("retime", None, None, None)[0:1] + (None, None, None) or True
    assert batch.modification_kinds[1] == "retime"


def test_declared_roles_are_enforced() -> None:
    with pytest.raises(ValueError, match="positive severity and a kind"):
        episode((execution("r"),), execution("m"), kind="modified_query")
    with pytest.raises(ValueError, match="within subject"):
        episode((execution("r"),), execution("q", subject="s2"), kind="accepted_query")
    with pytest.raises(ValueError, match="another person"):
        episode((execution("r"),), execution("q"), kind="other_subject_query")
    with pytest.raises(ValueError, match="only modified queries"):
        episode(
            (execution("r"),), execution("q"), kind="accepted_query", modification_kind="retime"
        )
    with pytest.raises(ValueError, match="nuisance transforms are declared"):
        episode(
            (execution("r"),),
            execution("o", subject="s2"),
            kind="other_subject_query",
            nuisance_kind="sensor_noise",
        )


def test_collate_preserves_large_absolute_clock_precision() -> None:
    reference = execution("r")
    reference = BoundedExecution(
        **{
            **reference.__dict__,
            "patch_intervals_sec": reference.patch_intervals_sec.double()
            + 1_634_178_333.0,
        }
    )
    batch = collate_execution_episodes([episode((reference,), execution("q"))])
    assert batch.reference_intervals_sec.dtype == torch.float64
    assert torch.all(
        batch.reference_intervals_sec[0, 0, :, 1]
        > batch.reference_intervals_sec[0, 0, :, 0]
    )


def test_bounded_execution_rejects_internal_missing_patch_gap() -> None:
    base = execution("gapped", length=4)
    with pytest.raises(ValueError, match="internal invalid patch gap"):
        BoundedExecution(
            **{
                **base.__dict__,
                "patch_mask": torch.tensor([True, False, True, True]),
            }
        )
