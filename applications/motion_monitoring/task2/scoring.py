"""The deployment scoring path shared by the untrained floor and the ruler.

Both arms are scored identically (design doc sections 6-7): align to
movement phase, take phase-local residuals against the personal prototype,
estimate the person's ordinary scatter by leave-one-execution-out over the
accepted references, and report the joint deviation together with that person's
own reference-only 95% operating limit. The only difference between arms is the space the residuals
live in, which is the whole point of the ruler.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import BoundedExecution, EpisodeBatch
from .model import ChangeRuler, _resample_execution_set, _normalize, _prototype, resample_to_phase
from .personal import fit_personal_variation, personal_operating_point, score_query


@dataclass(frozen=True)
class ChangeReport:
    """Per-episode personal deviation, threshold and its provenance."""

    joint_deviation: Tensor
    personal_limit95: Tensor
    exceeds_personal_limit: Tensor
    reference_limited: Tensor
    phase_residuals: Tensor
    raw_distance: Tensor


_PHYSICAL_FLOORS = {
    "duration_sec": 0.01,
    "acc_magnitude_mean_g": 1e-3,
    "acc_magnitude_std_g": 1e-3,
    "acc_dynamic_rms_g": 1e-3,
    "acc_jerk_rms_g_per_s": 1e-2,
    "gyro_magnitude_mean_rad_per_s": 1e-3,
    "gyro_magnitude_std_rad_per_s": 1e-3,
    "gyro_dynamic_rms_rad_per_s": 1e-3,
    "gyro_jerk_rms_rad_per_s2": 1e-2,
}


def _physical_summary(execution: BoundedExecution) -> tuple[Tensor, Tensor, tuple[str, ...]]:
    if execution.physical_features is None or execution.physical_feature_mask is None:
        raise ValueError("the representation cache does not contain physical summaries")
    valid_patch = execution.patch_mask[:, None]
    mask = execution.physical_feature_mask & valid_patch
    values = execution.physical_features.double()
    names = tuple(execution.physical_feature_names)
    keep = [index for index, name in enumerate(names) if name != "valid_fraction"]
    summaries = []
    available = []
    for index in keep:
        selected = mask[:, index]
        available.append(bool(selected.any()))
        summaries.append(values[selected, index].mean() if bool(selected.any()) else values.new_zeros(()))
    valid_intervals = execution.patch_intervals_sec[execution.patch_mask]
    duration = (valid_intervals[-1, 1] - valid_intervals[0, 0]).double()
    return (
        torch.stack((duration, *summaries)),
        torch.tensor((True, *available), dtype=torch.bool, device=values.device),
        ("duration_sec", *(names[index] for index in keep)),
    )


@torch.no_grad()
def physical_change_report(
    references: tuple[BoundedExecution, ...], query: BoundedExecution
) -> dict[str, float | bool]:
    """Non-learned personal ruler over auditable physical summaries."""

    if not references:
        raise ValueError("physical control needs accepted references")
    rows = [_physical_summary(item) for item in (*references, query)]
    names = rows[0][2]
    if any(item[2] != names for item in rows):
        raise ValueError("physical feature schemas differ inside one episode")
    common = torch.stack([item[1] for item in rows]).all(dim=0)
    if not bool(common.any()):
        raise ValueError("episode has no physical feature shared by every execution")
    reference_values = torch.stack([item[0] for item in rows[:-1]])[:, common]
    query_values = rows[-1][0][common]
    selected_names = tuple(name for name, keep in zip(names, common.tolist()) if keep)
    floors = torch.tensor(
        [_PHYSICAL_FLOORS[name] for name in selected_names],
        dtype=torch.float64,
        device=reference_values.device,
    )
    return score_query(reference_values, query_values, measurement_floor=floors)


def _phase_spaces(
    batch: EpisodeBatch, model: ChangeRuler | None, *, phase_bins: int
) -> tuple[Tensor, Tensor]:
    if model is None:
        references = _resample_execution_set(
            batch.reference_embeddings,
            batch.reference_intervals_sec,
            batch.reference_patch_mask,
            batch.reference_execution_mask,
            bins=phase_bins,
        )
        query = resample_to_phase(
            batch.query_embeddings, batch.query_intervals_sec, batch.query_mask, bins=phase_bins
        )
        return references, query
    output = model(batch)
    return output.reference_phase, output.query_phase


@torch.no_grad()
def personal_change_report(
    batch: EpisodeBatch,
    model: ChangeRuler | None = None,
    *,
    phase_bins: int = 8,
    measurement_floor: float = 1e-3,
    z: float = 1.96,
) -> ChangeReport:
    """Score every episode through its own person's accepted envelope.

    ``model=None`` is the mandatory untrained floor: frozen embeddings, cosine
    residual, personal envelope, no learned parameter anywhere.
    """

    if phase_bins < 2:
        raise ValueError("phase_bins must be at least two")
    bins = phase_bins if model is None else model.phase_bins
    references, query = _phase_spaces(batch, model, phase_bins=bins)
    prototype = _prototype(references, batch.reference_execution_mask)
    query_residual = 1.0 - (_normalize(query) * _normalize(prototype)).sum(dim=-1)
    learned_reference_residuals = None
    if model is not None:
        # Calibrate the personal envelope through the same learned query path.
        # Otherwise a trained refinement can move deployment queries while the
        # reference-only operating limit remains in the unrefined space.
        learned_reference_residuals, _ = model.reference_residuals(batch)

    deviations: list[float] = []
    limits: list[float] = []
    exceeds: list[bool] = []
    limited: list[bool] = []
    for row in range(query.shape[0]):
        valid = torch.nonzero(batch.reference_execution_mask[row], as_tuple=False).flatten().tolist()
        if learned_reference_residuals is not None:
            stacked = learned_reference_residuals[row, valid].double()
        else:
            ordinary = []
            for index in valid:
                others = [item for item in valid if item != index]
                comparison = (
                    references[row, index]
                    if not others
                    else references[row, others].mean(dim=0)
                )
                ordinary.append(
                    1.0
                    - (
                        _normalize(references[row, index])
                        * _normalize(comparison)
                    ).sum(dim=-1)
                )
            stacked = torch.stack(ordinary).double()
        personal = fit_personal_variation(stacked, measurement_floor=measurement_floor)
        deviation = float(personal.score(query_residual[row].double()).joint_deviation)
        point = personal_operating_point(stacked, measurement_floor=measurement_floor, z=z)
        deviations.append(deviation)
        limits.append(point.personal_limit95)
        limited.append(point.reference_limited)
        exceeds.append(
            (not point.reference_limited) and deviation > point.personal_limit95
        )
    device = query.device
    return ChangeReport(
        joint_deviation=torch.tensor(deviations, dtype=torch.float64, device=device),
        personal_limit95=torch.tensor(limits, dtype=torch.float64, device=device),
        exceeds_personal_limit=torch.tensor(exceeds, dtype=torch.bool, device=device),
        reference_limited=torch.tensor(limited, dtype=torch.bool, device=device),
        phase_residuals=query_residual,
        raw_distance=query_residual.mean(dim=-1),
    )
