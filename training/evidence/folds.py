"""Leakage-safe Phase-B train/validation fold construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch


VALIDATION_QUERY_POLICY = "heldout_subject_or_configuration_v1"


@dataclass(frozen=True)
class PhaseBFoldMasks:
    train_base: torch.Tensor
    validation: torch.Tensor
    subject_only: torch.Tensor
    configuration_only: torch.Tensor
    joint: torch.Tensor


def phase_b_fold_masks(
    configuration: torch.Tensor,
    subject: torch.Tensor,
    validation_configurations: torch.Tensor,
    validation_subjects: torch.Tensor,
) -> PhaseBFoldMasks:
    """Keep training in one quadrant and use every excluded quadrant for validation.

    Training rows have neither a held-out subject nor a held-out configuration. Validation rows
    have at least one held factor. This preserves strict row disjointness while avoiding the old
    behavior that discarded the subject-only and configuration-only quadrants.
    """
    if configuration.shape != subject.shape:
        raise ValueError("configuration and subject must align one-to-one")
    held_config = torch.isin(configuration, validation_configurations)
    held_subject = torch.isin(subject, validation_subjects)
    subject_only = held_subject & ~held_config
    configuration_only = held_config & ~held_subject
    joint = held_subject & held_config
    train_base = ~held_subject & ~held_config
    validation = subject_only | configuration_only | joint
    if bool((train_base & validation).any()) or not bool((train_base | validation).all()):
        raise RuntimeError("Phase-B fold masks do not partition the corpus")
    return PhaseBFoldMasks(
        train_base=train_base,
        validation=validation,
        subject_only=subject_only,
        configuration_only=configuration_only,
        joint=joint,
    )
