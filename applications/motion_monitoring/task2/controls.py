"""Transparent non-learned controls for Task-2 change quantification.

``direct_change_scores`` is the untrained floor: frozen embeddings, cosine
residual to the personal prototype, personal envelope. It is a mandatory row in
every Task-2 table (design doc section 7).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import EpisodeBatch
from .model import _resample_execution_set, resample_to_phase
from .personal import fit_personal_variation


@dataclass(frozen=True)
class DirectChangeOutput:
    """Phase residuals and personal-standardized change without a neural head."""

    raw_change_scores: Tensor
    personal_change_scores: Tensor
    phase_residuals: Tensor
    reference_limited: Tensor


@torch.no_grad()
def direct_change_scores(
    batch: EpisodeBatch,
    *,
    phase_bins: int = 8,
    measurement_floor: float = 1e-3,
) -> DirectChangeOutput:
    """Compare queries with robust personal reference trajectories.

    Each reference is compared with a leave-one-out reference prototype to form
    the person's ordinary residual distribution. The query is compared with the
    full reference prototype. No globally fitted projection or neural parameter
    participates in this control.
    """

    if phase_bins < 2 or measurement_floor <= 0:
        raise ValueError("phase_bins and measurement_floor must be positive")
    references = _resample_execution_set(
        batch.reference_embeddings,
        batch.reference_intervals_sec,
        batch.reference_patch_mask,
        batch.reference_execution_mask,
        bins=phase_bins,
    )
    query = resample_to_phase(
        batch.query_embeddings,
        batch.query_intervals_sec,
        batch.query_mask,
        bins=phase_bins,
    )
    references = F.normalize(references, dim=-1, eps=1e-8)
    query = F.normalize(query, dim=-1, eps=1e-8)

    raw_scores: list[Tensor] = []
    personal_scores: list[Tensor] = []
    phase_rows: list[Tensor] = []
    limited: list[bool] = []
    for row in range(len(query)):
        valid = batch.reference_execution_mask[row]
        accepted = references[row, valid]
        prototype = F.normalize(accepted.mean(dim=0), dim=-1, eps=1e-8)
        query_residual = 1.0 - (query[row] * prototype).sum(dim=-1)

        ordinary_rows: list[Tensor] = []
        for index in range(len(accepted)):
            if len(accepted) == 1:
                comparison = accepted[index]
            else:
                comparison = F.normalize(
                    torch.cat((accepted[:index], accepted[index + 1 :])).mean(dim=0),
                    dim=-1,
                    eps=1e-8,
                )
            ordinary_rows.append(1.0 - (accepted[index] * comparison).sum(dim=-1))
        personal = fit_personal_variation(
            torch.stack(ordinary_rows), measurement_floor=measurement_floor
        )
        personal_score = personal.score(query_residual.double()).joint_deviation
        raw_scores.append(query_residual.mean())
        personal_scores.append(personal_score.to(query.dtype))
        phase_rows.append(query_residual)
        limited.append(personal.reference_limited)

    return DirectChangeOutput(
        raw_change_scores=torch.stack(raw_scores),
        personal_change_scores=torch.stack(personal_scores),
        phase_residuals=torch.stack(phase_rows),
        reference_limited=torch.tensor(limited, dtype=torch.bool, device=query.device),
    )
