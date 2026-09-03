"""Personal expected-variation fitting for Task 2.

The global metric head learns across development subjects.  This module performs
the separate deployment-time fit that asks which joint feature changes are usual
for one person, task, and acquisition setup.  It deliberately has no optimizer:
accepted executions define a small regularized statistical model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


MIN_OPERATING_POINT_REFERENCES = 4


@dataclass(frozen=True)
class PersonalDeviation:
    """One or more feature vectors scored against a personal baseline."""

    standardized_residuals: Tensor
    squared_mahalanobis: Tensor
    joint_deviation: Tensor


@dataclass(frozen=True)
class PersonalVariationModel:
    """Robust center and OAS-regularized joint variation for accepted executions."""

    center: Tensor
    feature_scale: Tensor
    covariance: Tensor
    precision: Tensor
    shrinkage: float
    sample_count: int
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        width = self.center.numel()
        if self.center.shape != (width,) or self.feature_scale.shape != (width,):
            raise ValueError("personal center and scale must be one-dimensional")
        if self.covariance.shape != (width, width) or self.precision.shape != (
            width,
            width,
        ):
            raise ValueError("personal covariance and precision have the wrong shape")
        if self.sample_count <= 0 or not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("invalid personal-model sample count or shrinkage")
        tensors = (self.center, self.feature_scale, self.covariance, self.precision)
        if not all(tensor.is_floating_point() for tensor in tensors):
            raise ValueError("personal-model tensors must be floating point")
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise ValueError("personal-model tensors must be finite")
        if bool((self.feature_scale <= 0).any()):
            raise ValueError("personal feature scales must be positive")
        if self.feature_names and len(self.feature_names) != width:
            raise ValueError("personal feature names must match the fitted width")

    @property
    def reference_limited(self) -> bool:
        """Whether too few accepted executions were available for joint covariance."""

        return self.sample_count < MIN_OPERATING_POINT_REFERENCES

    def score(self, features: Tensor) -> PersonalDeviation:
        """Return robust standardized and joint deviations from the accepted baseline."""

        values = torch.as_tensor(features, dtype=self.center.dtype, device=self.center.device)
        if values.shape[-1] != self.center.numel():
            raise ValueError("personal score feature width does not match the fitted model")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("personal score features must be finite")
        residuals = (values - self.center) / self.feature_scale
        squared = torch.einsum("...i,ij,...j->...", residuals, self.precision, residuals)
        squared = squared.clamp_min(0.0)
        joint = torch.sqrt(squared / self.center.numel())
        return PersonalDeviation(residuals, squared, joint)


def _oas_covariance(values: Tensor, *, eigenvalue_floor: float) -> tuple[Tensor, float]:
    """Oracle-approximating shrinkage covariance with an eigenvalue safety floor."""

    sample_count, width = values.shape
    if sample_count < 3:
        return torch.eye(width, dtype=values.dtype, device=values.device), 1.0
    centered = values - values.mean(dim=0, keepdim=True)
    empirical = centered.T @ centered / sample_count
    mu = torch.trace(empirical) / width
    alpha = empirical.square().mean()
    denominator = (sample_count + 1.0) * (alpha - mu.square() / width)
    if float(denominator) <= torch.finfo(values.dtype).eps:
        shrinkage = 1.0
    else:
        shrinkage = min(float((alpha + mu.square()) / denominator), 1.0)
    target_scale = mu.clamp_min(eigenvalue_floor)
    covariance = (1.0 - shrinkage) * empirical
    covariance = covariance + shrinkage * target_scale * torch.eye(
        width, dtype=values.dtype, device=values.device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    covariance = (eigenvectors * eigenvalues.clamp_min(eigenvalue_floor)) @ eigenvectors.T
    return covariance, shrinkage


def fit_personal_variation(
    accepted_features: Tensor,
    *,
    measurement_floor: Tensor | float = 1e-4,
    feature_names: Sequence[str] = (),
    eigenvalue_floor: float = 1e-4,
) -> PersonalVariationModel:
    """Fit expected joint variation from accepted independent executions.

    Features should contain a fixed ordered set of phase-local latent residuals and
    separately measured physical differences.  The measurement floor must be in
    those same units and should come from development-set test-retest or remounting
    measurements rather than a tuned test-set constant.
    """

    values = torch.as_tensor(accepted_features)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("accepted features must be non-empty [execution, feature]")
    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError("accepted features must be finite floating-point values")
    if eigenvalue_floor <= 0:
        raise ValueError("eigenvalue_floor must be positive")
    # Float64 keeps the small personal covariance fit stable; scoring remains cheap.
    values = values.detach().to(dtype=torch.float64)
    center = values.median(dim=0).values
    robust_scale = 1.4826 * (values - center).abs().median(dim=0).values
    floor = torch.as_tensor(measurement_floor, dtype=values.dtype, device=values.device)
    if floor.ndim == 0:
        floor = floor.expand(values.shape[1])
    if floor.shape != (values.shape[1],) or not bool(torch.isfinite(floor).all()):
        raise ValueError("measurement_floor must be finite and match the feature width")
    if bool((floor <= 0).any()):
        raise ValueError("measurement_floor must be positive")
    scale = torch.maximum(robust_scale, floor)
    standardized = (values - center) / scale
    covariance, shrinkage = _oas_covariance(
        standardized, eigenvalue_floor=eigenvalue_floor
    )
    precision = torch.linalg.inv(covariance)
    return PersonalVariationModel(
        center=center,
        feature_scale=scale,
        covariance=covariance,
        precision=precision,
        shrinkage=shrinkage,
        sample_count=len(values),
        feature_names=tuple(feature_names),
    )


@dataclass(frozen=True)
class PersonalOperatingPoint:
    """Per-person, per-task limit fixed from accepted references only.

    ``personal_limit95`` is the deployed threshold on the joint deviation: the mean plus
    1.96 standard deviations of the leave-one-execution-out deviations of the
    accepted references. It is not a measurement-science MDC because it is a
    reference-only operating limit rather than ``1.96 * sqrt(2) * SEM``. With
    fewer than four references, each leave-one-out fold has too little data for
    even a regularized joint scatter estimate, so no limit is reported.
    """

    personal_limit95: float
    loo_deviations: Tensor
    loo_mean: float
    loo_sd: float
    sample_count: int
    z: float

    @property
    def reference_limited(self) -> bool:
        return self.sample_count < MIN_OPERATING_POINT_REFERENCES


def personal_operating_point(
    reference_features: Tensor,
    *,
    measurement_floor: Tensor | float = 1e-3,
    z: float = 1.96,
    feature_names: Sequence[str] = (),
) -> PersonalOperatingPoint:
    """Leave-one-execution-out 95% operating limit from accepted references."""

    values = torch.as_tensor(reference_features).double()
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("reference features must be [reference, feature] with at least one row")
    if z <= 0:
        raise ValueError("z must be positive")
    count = int(values.shape[0])
    if count < MIN_OPERATING_POINT_REFERENCES:
        return PersonalOperatingPoint(
            personal_limit95=float("nan"),
            loo_deviations=values.new_zeros(0),
            loo_mean=float("nan"),
            loo_sd=float("nan"),
            sample_count=count,
            z=float(z),
        )
    deviations = []
    for index in range(count):
        others = torch.cat((values[:index], values[index + 1 :]))
        model = fit_personal_variation(
            others, measurement_floor=measurement_floor, feature_names=tuple(feature_names)
        )
        deviations.append(model.score(values[index]).joint_deviation)
    loo = torch.stack(deviations)
    mean = float(loo.mean())
    sd = float(loo.std(unbiased=True)) if count > 1 else 0.0
    sd = max(sd, 1e-6)
    return PersonalOperatingPoint(
        personal_limit95=mean + float(z) * sd,
        loo_deviations=loo,
        loo_mean=mean,
        loo_sd=sd,
        sample_count=count,
        z=float(z),
    )


def score_query(
    reference_features: Tensor,
    query_features: Tensor,
    *,
    measurement_floor: Tensor | float = 1e-3,
    z: float = 1.96,
) -> dict[str, float | bool]:
    """Deviation of one query from the person's accepted envelope, with its threshold."""

    references = torch.as_tensor(reference_features).double()
    model = fit_personal_variation(references, measurement_floor=measurement_floor)
    deviation = float(model.score(torch.as_tensor(query_features).double()).joint_deviation)
    point = personal_operating_point(references, measurement_floor=measurement_floor, z=z)
    exceeds = (not point.reference_limited) and deviation > point.personal_limit95
    return {
        "joint_deviation": deviation,
        "personal_limit95": point.personal_limit95,
        "exceeds_personal_limit": bool(exceeds),
        "reference_limited": bool(point.reference_limited),
    }
