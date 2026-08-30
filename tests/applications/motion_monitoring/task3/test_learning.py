import torch

from applications.motion_monitoring.task3.candidates import (
    assign_event_targets,
    pool_multiscale_candidates,
)
from applications.motion_monitoring.task3.contracts import (
    CandidateBatch,
    CandidateTargets,
)
from applications.motion_monitoring.task3.losses import (
    scoped_pair_indices,
    scoped_pair_loss,
    scoped_pair_masks,
)
from applications.motion_monitoring.task3.model import RecurrentMotionMetric
from applications.motion_monitoring.task3.smoke_train import (
    run_smoke_training,
    synthetic_batch,
)


def _candidate_fixture() -> tuple[CandidateBatch, CandidateTargets]:
    embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
            [[-1.0, 0.0], [-0.9, 0.1], [0.0, -1.0], [0.1, -0.9]],
        ]
    )
    mask = torch.ones(2, 4, dtype=torch.bool)
    starts = torch.arange(4, dtype=torch.float32).expand(2, -1)
    candidates = CandidateBatch(
        embeddings=embeddings,
        candidate_mask=mask,
        start_sec=starts,
        end_sec=starts + 1,
        scale_index=torch.zeros(2, 4, dtype=torch.long),
        start_patch=torch.arange(4).expand(2, -1),
        end_patch=torch.arange(1, 5).expand(2, -1),
        recording_id=torch.arange(2)[:, None].expand(-1, 4),
    )
    # Both scopes reuse labels 0 and 1. Instances 10 and 11 are copies of one
    # execution and therefore may not form a positive pair with each other.
    targets = CandidateTargets(
        label_id=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
        instance_id=torch.tensor([[10, 10, 20, 21], [30, 31, 40, 41]]),
        scope_id=torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]]),
        assigned_mask=mask,
        background_mask=torch.zeros_like(mask),
        best_iou=torch.ones(2, 4),
    )
    return candidates, targets


def test_pair_masks_respect_scope_and_independent_instances():
    candidates, targets = _candidate_fixture()
    valid, positive, negative = scoped_pair_masks(candidates, targets)

    assert not positive[0, 1]  # same physical event instance
    assert positive[2, 3]
    assert positive[4, 5]
    assert not valid[:4, 4:].any()  # no cross-dataset assumptions
    assert negative[0, 2]


def test_metric_loss_is_finite_and_all_parameters_receive_gradients():
    candidates, targets = _candidate_fixture()
    model = RecurrentMotionMetric(2, projection_dim=2)

    output = scoped_pair_loss(model, candidates, targets)
    output.loss.backward()

    assert torch.isfinite(output.loss)
    assert output.positive_logits.numel() == 3
    assert output.negative_logits.numel() == 8
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.norm() > 0
        for parameter in model.parameters()
    )


def test_pair_construction_is_bounded_without_changing_class_balance():
    candidates, targets = _candidate_fixture()

    positive, negative = scoped_pair_indices(candidates, targets, max_pairs_per_class=2)

    assert positive.shape == (2, 2)
    assert negative.shape == (2, 2)


def test_complete_synthetic_smoke_training_improves_without_gradient_failure():
    result = run_smoke_training(steps=20, seed=7)

    assert result["final_loss"] < result["initial_loss"] * 0.3
    assert result["gradient/all_finite"] == 1.0
    assert result["gradient/nonzero_tensor_count"] == 3.0
    assert result["pair/probability_separation"] > 0.8


def test_loss_can_finetune_upstream_patch_embeddings_through_pooling():
    patch_embeddings, intervals, patch_mask, events = synthetic_batch(seed=11)
    patch_embeddings.requires_grad_()
    candidates = pool_multiscale_candidates(
        patch_embeddings,
        intervals,
        patch_mask,
        durations_sec=(4.0,),
    )
    targets = assign_event_targets(candidates, events, positive_iou=0.7)
    model = RecurrentMotionMetric(patch_embeddings.shape[-1], projection_dim=6)

    scoped_pair_loss(model, candidates, targets).loss.backward()

    assert patch_embeddings.grad is not None
    assert torch.isfinite(patch_embeddings.grad).all()
    assert patch_embeddings.grad.norm() > 0
