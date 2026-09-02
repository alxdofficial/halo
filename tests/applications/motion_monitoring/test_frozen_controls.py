import torch

from applications.motion_monitoring.task2 import (
    collate_execution_episodes,
    direct_change_scores,
)
from applications.motion_monitoring.task2.smoke import make_synthetic_episodes
from applications.motion_monitoring.task3 import (
    assign_event_targets,
    direct_cosine_affinity,
    pool_multiscale_candidates,
)
from applications.motion_monitoring.task3.smoke_train import synthetic_batch


def test_task2_direct_control_uses_personal_references_without_parameters() -> None:
    episodes = make_synthetic_episodes(subjects=4, embedding_dim=12)
    batch = collate_execution_episodes(episodes)
    output = direct_change_scores(batch, phase_bins=8)

    assert output.raw_change_scores.shape == (len(episodes),)
    assert output.personal_change_scores.shape == (len(episodes),)
    assert torch.isfinite(output.personal_change_scores).all()
    accepted = batch.classification_targets == 0
    changed = batch.classification_targets == 1
    assert output.raw_change_scores[changed].mean() > output.raw_change_scores[accepted].mean()
    assert not output.reference_limited.any()


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
