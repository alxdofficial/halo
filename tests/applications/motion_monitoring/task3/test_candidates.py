import pytest
import torch

from applications.motion_monitoring.task3.candidates import (
    assign_event_targets,
    pool_multiscale_candidates,
)
from applications.motion_monitoring.task3.contracts import EventBatch


def _intervals(batch: int, patches: int) -> torch.Tensor:
    one = torch.stack(
        [
            torch.arange(patches, dtype=torch.float32),
            torch.arange(1, patches + 1, dtype=torch.float32),
        ],
        dim=-1,
    )
    return one.expand(batch, -1, -1).clone()


def test_multiscale_pooling_uses_physical_spans_and_masks_padding():
    embeddings = torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2)
    embeddings.requires_grad_()
    intervals = _intervals(2, 6)
    mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])

    candidates = pool_multiscale_candidates(
        embeddings,
        intervals,
        mask,
        durations_sec=(2.0, 3.0),
        normalize=False,
    )

    assert candidates.candidate_mask.sum(dim=1).tolist() == [9, 5]
    assert candidates.embeddings[0, 0].tolist() == pytest.approx(
        embeddings.detach()[0, :2].mean(dim=0).tolist()
    )
    assert candidates.start_sec[0, :5].tolist() == [0, 1, 2, 3, 4]
    assert candidates.end_sec[0, :5].tolist() == [2, 3, 4, 5, 6]
    assert torch.all(candidates.scale_index[1, 5:] == -1)

    candidates.embeddings[candidates.candidate_mask].sum().backward()
    assert torch.all(embeddings.grad[1, :4] > 0)
    assert torch.all(embeddings.grad[1, 4:] == 0)


def test_pooling_does_not_bridge_a_hard_timestamp_gap():
    embeddings = torch.randn(1, 6, 4)
    intervals = _intervals(1, 6)
    intervals[0, 3:] += 7.0
    mask = torch.ones(1, 6, dtype=torch.bool)

    candidates = pool_multiscale_candidates(
        embeddings, intervals, mask, durations_sec=(2.0, 4.0), normalize=False
    )

    for start, end in zip(
        candidates.start_patch[candidates.candidate_mask].tolist(),
        candidates.end_patch[candidates.candidate_mask].tolist(),
        strict=True,
    ):
        assert not (start < 3 < end)


def test_pooling_rejects_unordered_physical_intervals():
    embeddings = torch.randn(1, 3, 4)
    intervals = _intervals(1, 3)
    intervals[0, 1, 0] = -1.0

    with pytest.raises(ValueError, match="ordered in physical time"):
        pool_multiscale_candidates(
            embeddings,
            intervals,
            torch.ones(1, 3, dtype=torch.bool),
            durations_sec=(2.0,),
        )


def test_event_assignment_ignores_partial_overlap_and_requires_exhaustive_background():
    embeddings = torch.randn(2, 6, 4)
    candidates = pool_multiscale_candidates(
        embeddings,
        _intervals(2, 6),
        torch.ones(2, 6, dtype=torch.bool),
        durations_sec=(2.0,),
    )
    events = EventBatch(
        start_sec=torch.tensor([[1.0], [1.0]]),
        end_sec=torch.tensor([[3.0], [3.0]]),
        label_id=torch.tensor([[4], [4]]),
        instance_id=torch.tensor([[10], [20]]),
        scope_id=torch.tensor([[0], [1]]),
        event_mask=torch.ones(2, 1, dtype=torch.bool),
        exhaustive=torch.tensor([True, False]),
    )

    targets = assign_event_targets(candidates, events, positive_iou=0.7)

    assert targets.assigned_mask[:, 1].tolist() == [True, True]
    assert targets.scope_id[:, 1].tolist() == [0, 1]
    assert targets.assigned_mask[:, 0].tolist() == [False, False]
    assert targets.background_mask[0, 3:].all()
    assert not targets.background_mask[1].any()


def test_event_assignment_supports_exhaustive_recording_with_no_events():
    candidates = pool_multiscale_candidates(
        torch.randn(1, 4, 3),
        _intervals(1, 4),
        torch.ones(1, 4, dtype=torch.bool),
        durations_sec=(2.0,),
    )
    events = EventBatch(
        start_sec=torch.empty(1, 0),
        end_sec=torch.empty(1, 0),
        label_id=torch.empty(1, 0, dtype=torch.long),
        instance_id=torch.empty(1, 0, dtype=torch.long),
        scope_id=torch.empty(1, 0, dtype=torch.long),
        event_mask=torch.empty(1, 0, dtype=torch.bool),
        exhaustive=torch.tensor([True]),
    )

    targets = assign_event_targets(candidates, events)

    assert not targets.assigned_mask.any()
    assert torch.equal(targets.background_mask, candidates.candidate_mask)


def test_multiscale_pooling_preserves_large_absolute_clock_precision():
    intervals = _intervals(1, 4).double() + 1_634_178_333.0
    candidates = pool_multiscale_candidates(
        torch.randn(1, 4, 3),
        intervals,
        torch.ones(1, 4, dtype=torch.bool),
        durations_sec=(2.0,),
    )
    assert candidates.start_sec.dtype == torch.float64
    assert torch.all(
        candidates.end_sec[candidates.candidate_mask]
        > candidates.start_sec[candidates.candidate_mask]
    )
