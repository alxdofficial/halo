import pytest
import torch

from applications.motion_monitoring.task2 import (
    ChangeRuler,
    collate_execution_episodes,
    personal_change_report,
)
from applications.motion_monitoring.task2.smoke import build_demo_batch
from applications.motion_monitoring.task3 import (
    assign_event_targets,
    direct_cosine_affinity,
    pool_multiscale_candidates,
)
from applications.motion_monitoring.task3.smoke_train import synthetic_batch


def test_task2_untrained_floor_uses_personal_references_without_parameters() -> None:
    episodes, batch = build_demo_batch(subjects=4, seed=13)
    floor = personal_change_report(batch, None)

    assert floor.joint_deviation.shape == (len(episodes),)
    assert torch.isfinite(floor.joint_deviation).all()
    assert not floor.reference_limited.any()
    # A declared physical modification must read as a larger raw residual than an
    # accepted repeat of the same person, with no learned parameter involved.
    assert (
        floor.raw_distance[batch.negative_mask].mean()
        > floor.raw_distance[batch.positive_mask].mean()
    )
    # An untrained ruler is exactly this floor: the refinement is zero-initialised.
    ruler = personal_change_report(batch, ChangeRuler(batch.query_embeddings.shape[-1]).eval())
    assert ruler.joint_deviation.tolist() == pytest.approx(
        floor.joint_deviation.tolist(), abs=1e-5
    )


def test_task3_direct_control_scores_exact_same_candidate_pairs() -> None:
    embeddings, intervals, patch_mask, events = synthetic_batch()
    candidates = pool_multiscale_candidates(
        embeddings,
        intervals,
        patch_mask,
        durations_sec=(2.0, 4.0, 6.0),
        candidate_stride_sec=1.0,
    )
    targets = assign_event_targets(candidates, events, positive_iou=0.7)
    output = direct_cosine_affinity(candidates, targets)

    assert len(output.positive_scores) > 0
    assert len(output.negative_scores) > 0
    assert 0.0 <= output.auroc <= 1.0
    assert 0.0 <= output.auprc <= 1.0
    assert output.positive_scores.mean() > output.negative_scores.mean()
