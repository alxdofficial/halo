"""Measurement-science metrics for Task 2 (design doc section 7).

Reliability comes first: intraclass correlation, standard error of measurement
and minimum detectable change on accepted repeats, reported separately for the
within-session, cross-session and between-day conditions. Responsiveness is the
paired within-person AUROC and standardised response mean against a real change.
All intervals resample subjects, never executions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

import torch
from torch import Tensor


def binary_auroc(scores: Tensor, targets: Tensor, mask: Tensor | None = None) -> float:
    """Exact rank AUROC with average ranks for ties."""

    scores = torch.as_tensor(scores, dtype=torch.float64).flatten()
    targets = torch.as_tensor(targets).bool().flatten()
    if mask is not None:
        valid = torch.as_tensor(mask).bool().flatten()
        scores, targets = scores[valid], targets[valid]
    if len(scores) == 0 or not torch.isfinite(scores).all():
        return float("nan")
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    ranks = torch.arange(
        1, len(scores) + 1, dtype=torch.float64, device=scores.device
    )
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = ranks[start:end].mean()
        start = end
    original_ranks = torch.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = original_ranks[targets].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    )


def binary_operating_metrics(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor | None = None,
    *,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Return decisions at a threshold fixed on development subjects."""

    logits = torch.as_tensor(logits).flatten()
    targets = torch.as_tensor(targets).bool().flatten()
    valid = (
        torch.ones_like(targets)
        if mask is None
        else torch.as_tensor(mask).bool().flatten()
    )
    predictions = logits[valid] >= threshold
    truth = targets[valid]
    positive = truth
    negative = ~truth
    if not bool(positive.any()) or not bool(negative.any()):
        return {
            name: float("nan")
            for name in (
                "sensitivity",
                "specificity",
                "balanced_accuracy",
                "precision",
                "f1",
                "false_positive_rate",
                "accepted_false_alarm_rate",
            )
        }
    sensitivity = (predictions[positive] == truth[positive]).float().mean()
    specificity = (predictions[negative] == truth[negative]).float().mean()
    true_positive = (predictions & truth).sum().float()
    predicted_positive = predictions.sum().float()
    precision = true_positive / predicted_positive.clamp_min(1.0)
    f1 = 2 * precision * sensitivity / (precision + sensitivity).clamp_min(1e-12)
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "precision": float(precision),
        "f1": float(f1),
        "false_positive_rate": float(1.0 - specificity),
        "accepted_false_alarm_rate": float(1.0 - specificity),
    }


def balanced_accuracy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor | None = None,
    *,
    threshold: float = 0.0,
) -> float:
    """Compatibility wrapper for the complete operating-point metrics."""

    return binary_operating_metrics(
        logits, targets, mask, threshold=threshold
    )["balanced_accuracy"]


@dataclass(frozen=True)
class RegressionMetrics:
    mae: dict[str, float]
    rmse: dict[str, float]
    counts: dict[str, int]


def masked_regression_metrics(
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
    target_names: tuple[str, ...],
) -> RegressionMetrics:
    predictions = torch.as_tensor(predictions)
    targets = torch.as_tensor(targets)
    mask = torch.as_tensor(mask).bool()
    if predictions.shape != targets.shape or mask.shape != targets.shape:
        raise ValueError("predictions, targets, and mask must have identical shapes")
    if predictions.ndim != 2 or predictions.shape[1] != len(target_names):
        raise ValueError("target names must match the regression width")
    mae: dict[str, float] = {}
    rmse: dict[str, float] = {}
    counts: dict[str, int] = {}
    for index, name in enumerate(target_names):
        valid = mask[:, index]
        count = int(valid.sum())
        counts[name] = count
        if count == 0:
            mae[name] = float("nan")
            rmse[name] = float("nan")
            continue
        error = predictions[valid, index] - targets[valid, index]
        mae[name] = float(error.abs().mean())
        rmse[name] = float(error.square().mean().sqrt())
    return RegressionMetrics(mae=mae, rmse=rmse, counts=counts)


@dataclass(frozen=True)
class ReliabilityResult:
    """Test-retest reliability of one measure over repeated accepted executions."""

    icc: float
    sem: float
    mdc95: float
    within_subject_cv: float
    # Number of measurement SERIES entering the ICC. A series is one person, task
    # and stream, so this is not a count of people; callers report that separately.
    series: int
    observations: int
    condition: str


def reliability(
    values: Sequence[float],
    subjects: Sequence[str],
    occasions: Sequence[str],
    *,
    condition: str = "unspecified",
    z: float = 1.96,
) -> ReliabilityResult:
    """ICC(2,1), SEM and MDC95 over repeated measurements of the same people.

    ICC(2,1) is the two-way random-effects absolute-agreement single-measurement
    form; it is the coefficient the wearable reliability literature reports.
    """

    if not (len(values) == len(subjects) == len(occasions)):
        raise ValueError("values, subject ids, and occasion ids must align")
    groups: dict[str, dict[str, float]] = {}
    for value, subject, occasion in zip(values, subjects, occasions):
        rows = groups.setdefault(str(subject), {})
        key = str(occasion)
        if key in rows:
            raise ValueError("a measurement series has duplicate occasion ids")
        rows[key] = float(value)
    groups = {subject: rows for subject, rows in groups.items() if len(rows) >= 2}
    if len(groups) < 2:
        raise ValueError("reliability needs at least two series with repeated measurements")
    common_occasions = sorted(set.intersection(*(set(rows) for rows in groups.values())))
    if len(common_occasions) < 2:
        raise ValueError("reliability needs at least two named occasions shared by every series")
    matrix = np.asarray(
        [[rows[occasion] for occasion in common_occasions] for rows in groups.values()],
        dtype=np.float64,
    )
    series_count, measurement_count = matrix.shape
    grand = matrix.mean()
    series_means = matrix.mean(axis=1)
    measurement_means = matrix.mean(axis=0)
    between_series = measurement_count * float(((series_means - grand) ** 2).sum()) / max(series_count - 1, 1)
    between_measurement = series_count * float(((measurement_means - grand) ** 2).sum()) / max(measurement_count - 1, 1)
    residual = matrix - series_means[:, None] - measurement_means[None, :] + grand
    error = float((residual**2).sum()) / max((series_count - 1) * (measurement_count - 1), 1)
    denominator = (
        between_series
        + (measurement_count - 1) * error
        + measurement_count * (between_measurement - error) / series_count
    )
    icc = float("nan") if denominator == 0 else float((between_series - error) / denominator)
    total_variance = float(matrix.var(ddof=1)) if matrix.size > 1 else 0.0
    sem_variance = max(total_variance * (1.0 - icc), 0.0)
    sem = float(np.sqrt(sem_variance)) if np.isfinite(icc) else float("nan")
    mean = float(np.abs(matrix.mean()))
    within_sd = float(np.mean(matrix.std(axis=1, ddof=1)))
    return ReliabilityResult(
        icc=icc,
        sem=sem,
        mdc95=float(z * np.sqrt(2.0) * sem),
        within_subject_cv=float(within_sd / mean) if mean > 0 else float("nan"),
        series=series_count,
        observations=int(matrix.size),
        condition=condition,
    )


@dataclass(frozen=True)
class AgreementResult:
    """Bland-Altman bias and limits of agreement between paired measurements."""

    bias: float
    lower_limit: float
    upper_limit: float
    pairs: int


def bland_altman(first: Sequence[float], second: Sequence[float], *, z: float = 1.96) -> AgreementResult:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("Bland-Altman needs two aligned vectors with at least two pairs")
    difference = left - right
    bias = float(difference.mean())
    sd = float(difference.std(ddof=1))
    return AgreementResult(
        bias=bias, lower_limit=bias - z * sd, upper_limit=bias + z * sd, pairs=int(len(left))
    )


def paired_within_series_auroc(
    accepted: Mapping[str, Sequence[float]], changed: Mapping[str, Sequence[float]]
) -> dict[str, float]:
    """AUROC computed inside each person/task/stream series, then averaged.

    Pooling across people would let between-person offsets do the work, which is
    exactly the confound Task 2 exists to avoid.
    """

    per_series: dict[str, float] = {}
    for series in sorted(set(accepted) & set(changed)):
        negative = np.asarray(accepted[series], dtype=np.float64)
        positive = np.asarray(changed[series], dtype=np.float64)
        if not len(negative) or not len(positive):
            continue
        comparisons = (positive[:, None] > negative[None, :]).sum()
        ties = (positive[:, None] == negative[None, :]).sum()
        per_series[series] = float((comparisons + 0.5 * ties) / (len(positive) * len(negative)))
    if not per_series:
        raise ValueError("no person/task/stream series has both accepted and changed scores")
    values = np.asarray(list(per_series.values()))
    return {
        "mean_auroc": float(values.mean()),
        "series": float(len(values)),
        "min_auroc": float(values.min()),
        "max_auroc": float(values.max()),
    }


def standardised_response_mean(before: Sequence[float], after: Sequence[float]) -> float:
    """SRM: mean paired change divided by the standard deviation of that change."""

    left = np.asarray(before, dtype=np.float64)
    right = np.asarray(after, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("SRM needs two aligned vectors with at least two pairs")
    difference = right - left
    sd = float(difference.std(ddof=1))
    return float(difference.mean() / sd) if sd > 0 else float("nan")


def nuisance_false_alarm_rate(
    deviations: Sequence[float], thresholds: Sequence[float]
) -> dict[str, float]:
    """Fraction of accepted repeats that exceed their personal reference limit."""

    values = np.asarray(deviations, dtype=np.float64)
    limits = np.asarray(thresholds, dtype=np.float64)
    if values.shape != limits.shape or values.ndim != 1 or not len(values):
        raise ValueError("deviations and thresholds must be aligned non-empty vectors")
    finite = np.isfinite(values) & np.isfinite(limits)
    if not finite.any():
        raise ValueError("no comparable accepted repeat has a finite threshold")
    exceed = (values[finite] > limits[finite]).sum()
    return {
        "false_alarm_rate": float(exceed / finite.sum()),
        "comparable": float(finite.sum()),
        "excluded_reference_limited": float((~finite).sum()),
    }


def subject_bootstrap(
    values: Sequence[float],
    subjects: Sequence[str],
    statistic: Callable[[np.ndarray], float] = lambda rows: float(np.mean(rows)),
    *,
    samples: int = 2000,
    seed: int = 20260902,
) -> dict[str, float]:
    """Resample subjects (never executions) for a 95 % interval on a statistic."""

    groups = {}
    for value, subject in zip(values, subjects):
        groups.setdefault(str(subject), []).append(float(value))
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("a subject bootstrap needs at least two subjects")
    rng = np.random.default_rng(seed)
    point = statistic(np.asarray([item for key in keys for item in groups[key]]))
    draws = []
    for _ in range(samples):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        pooled = [item for index in chosen for item in groups[keys[index]]]
        draws.append(statistic(np.asarray(pooled)))
    array = np.asarray(draws)
    return {
        "point": float(point),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
        "subjects": float(len(keys)),
    }
