"""Regression tests for the Phase-B multi-axis holdout."""

import torch

from training.evidence.folds import phase_b_fold_masks


def test_phase_b_fold_uses_all_nontraining_quadrants_without_overlap():
    configuration = torch.tensor([0, 0, 1, 1])
    subject = torch.tensor([0, 1, 0, 1])
    fold = phase_b_fold_masks(
        configuration, subject, torch.tensor([1]), torch.tensor([1])
    )
    assert fold.train_base.tolist() == [True, False, False, False]
    assert fold.subject_only.tolist() == [False, True, False, False]
    assert fold.configuration_only.tolist() == [False, False, True, False]
    assert fold.joint.tolist() == [False, False, False, True]
    assert fold.validation.tolist() == [False, True, True, True]


def test_phase_b_fold_rejects_misaligned_metadata():
    try:
        phase_b_fold_masks(
            torch.tensor([0, 1]), torch.tensor([0]), torch.tensor([1]), torch.tensor([1])
        )
    except ValueError as error:
        assert "align" in str(error)
    else:
        raise AssertionError("misaligned fold metadata was accepted")
