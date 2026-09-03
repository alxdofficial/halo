"""Declared physical modifications and nuisance transforms for the Task-2 ruler.

Two catalogues, applied at the level of an already-embedded execution's phase
trajectory is *not* enough: a modification must change the movement, so these
operate on the raw bounded execution before encoding. They are deterministic
functions of a seed and are frozen in the training manifest
(docs/tasks/TASK2_CHANGE_QUANTIFICATION.md section 4).

* ``MODIFICATIONS`` change the physiology and become negatives with a declared
  kind and severity in [0, 1].
* ``NUISANCES`` change the acquisition, not the movement, and ride on positives
  so the ruler is taught to ignore them. They are also applied to negatives so
  that "was transformed" carries no information about the label.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class TransformSpec:
    name: str
    kind: str
    description: str


def _resample(values: np.ndarray, factor: float) -> np.ndarray:
    """Uniformly retime a [T, C] execution by ``factor`` (>1 is slower)."""

    count = max(4, int(round(values.shape[0] * factor)))
    ratio = Fraction(count, values.shape[0]).limit_denominator(1000)
    output = resample_poly(values, ratio.numerator, ratio.denominator, axis=0)
    if len(output) > count:
        output = output[:count]
    elif len(output) < count:
        output = np.pad(output, ((0, count - len(output)), (0, 0)), mode="edge")
    return output


def retime(values: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Whole-execution speed change of up to +/-30 %."""

    direction = 1.0 if rng.random() < 0.5 else -1.0
    return _resample(values, 1.0 + direction * 0.30 * severity)


def amplitude_scale(values: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Phase-local reduction of the dynamic component over a contiguous third."""

    out = values.copy()
    count = out.shape[0]
    width = max(2, count // 3)
    start = int(rng.integers(0, max(1, count - width)))
    window = slice(start, start + width)
    baseline = out[window].mean(axis=0, keepdims=True)
    out[window] = baseline + (out[window] - baseline) * (1.0 - 0.6 * severity)
    return out


def reduced_range(values: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Whole-execution reduction of movement range about its own mean."""

    baseline = values.mean(axis=0, keepdims=True)
    return baseline + (values - baseline) * (1.0 - 0.5 * severity)


def added_tremor(
    values: np.ndarray,
    severity: float,
    rng: np.random.Generator,
    *,
    sampling_rate_hz: float,
    channels: tuple[str, ...],
) -> np.ndarray:
    """Additive 4-6 Hz oscillation on the accelerometer channels."""

    count, _ = values.shape
    frequency = float(rng.uniform(4.0, 6.0))
    time = np.arange(count) / sampling_rate_hz
    phase = float(rng.uniform(0.0, 2 * np.pi))
    accel = [index for index, name in enumerate(channels) if name.startswith("acc_")]
    if not accel:
        raise ValueError("added_tremor requires accelerometer channels")
    amplitude = 0.35 * severity * float(np.abs(values[:, accel]).std() + 1e-6)
    wave = amplitude * np.sin(2 * np.pi * frequency * time + phase)
    out = values.copy()
    out[:, accel] += wave[:, None]
    return out


def inserted_pause(values: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """A held posture inserted mid-execution."""

    count = values.shape[0]
    hold = max(2, int(round(count * 0.25 * severity)))
    position = int(rng.integers(count // 4, max(count // 4 + 1, 3 * count // 4)))
    held = np.repeat(values[position : position + 1], hold, axis=0)
    return np.concatenate((values[:position], held, values[position:]), axis=0)


MODIFICATIONS: Mapping[str, Callable[..., np.ndarray]] = {
    "retime": retime,
    "amplitude_scale": amplitude_scale,
    "reduced_range": reduced_range,
    "added_tremor": added_tremor,
    "inserted_pause": inserted_pause,
}


def remount_rotation(
    values: np.ndarray, rng: np.random.Generator, *, channels: tuple[str, ...]
) -> np.ndarray:
    """A small fixed sensor rotation, applied identically to acc and gyro triads."""

    angle = float(rng.uniform(-np.deg2rad(15.0), np.deg2rad(15.0)))
    axis = rng.normal(size=3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    out = values.copy()
    by_name = {name: index for index, name in enumerate(channels)}
    for prefix in ("acc", "gyro"):
        names = tuple(f"{prefix}_{axis}" for axis in "xyz")
        if all(name in by_name for name in names):
            indices = [by_name[name] for name in names]
            out[:, indices] = values[:, indices] @ rotation.T
    return out


def sensor_noise(
    values: np.ndarray, rng: np.random.Generator, *, channels: tuple[str, ...]
) -> np.ndarray:
    """Additive noise at the execution's own quiet-band level."""

    dynamic = np.median(np.abs(np.diff(values, axis=0)), axis=0)
    numerical_floor = np.asarray(
        [1e-4 if name.startswith(("acc_", "gyro_")) else 1e-6 for name in channels]
    )
    scale = np.maximum(0.02 * dynamic, numerical_floor)
    return values + rng.normal(scale=scale[None, :], size=values.shape)


def boundary_jitter(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Enrollment boundary error of up to 10 % at each end."""

    count = values.shape[0]
    left = int(rng.integers(0, max(1, int(0.1 * count))))
    right = int(rng.integers(0, max(1, int(0.1 * count))))
    trimmed = values[left : count - right]
    return trimmed if len(trimmed) >= 4 else values


NUISANCES: Mapping[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "remount_rotation": remount_rotation,
    "sensor_noise": sensor_noise,
    "boundary_jitter": boundary_jitter,
}

MODIFICATION_SPECS = tuple(
    TransformSpec(name, "modification", MODIFICATIONS[name].__doc__ or "")
    for name in sorted(MODIFICATIONS)
)
NUISANCE_SPECS = tuple(
    TransformSpec(name, "nuisance", NUISANCES[name].__doc__ or "") for name in sorted(NUISANCES)
)


def apply_modification(
    values: np.ndarray,
    *,
    kind: str,
    severity: float,
    seed: int,
    sampling_rate_hz: float,
    channels: tuple[str, ...],
) -> np.ndarray:
    """Deterministically apply one declared physical modification."""

    if kind not in MODIFICATIONS:
        raise KeyError(f"unknown modification {kind!r}; choose from {sorted(MODIFICATIONS)}")
    if not 0.0 < severity <= 1.0:
        raise ValueError("severity must lie in (0, 1]")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 4:
        raise ValueError("an execution must be [time, channel] with at least four samples")
    if len(channels) != array.shape[1] or sampling_rate_hz <= 0:
        raise ValueError("channels and a positive sampling rate must describe the execution")
    function = MODIFICATIONS[kind]
    if kind == "added_tremor":
        return function(
            array,
            float(severity),
            np.random.default_rng(seed),
            sampling_rate_hz=float(sampling_rate_hz),
            channels=tuple(channels),
        )
    return function(array, float(severity), np.random.default_rng(seed))


def apply_nuisance(
    values: np.ndarray,
    *,
    kind: str,
    seed: int,
    sampling_rate_hz: float,
    channels: tuple[str, ...],
) -> np.ndarray:
    """Deterministically apply one acquisition nuisance transform."""

    if kind not in NUISANCES:
        raise KeyError(f"unknown nuisance {kind!r}; choose from {sorted(NUISANCES)}")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 4:
        raise ValueError("an execution must be [time, channel] with at least four samples")
    if len(channels) != array.shape[1] or sampling_rate_hz <= 0:
        raise ValueError("channels and a positive sampling rate must describe the execution")
    function = NUISANCES[kind]
    if kind in {"remount_rotation", "sensor_noise"}:
        return function(array, np.random.default_rng(seed), channels=tuple(channels))
    return function(array, np.random.default_rng(seed))
