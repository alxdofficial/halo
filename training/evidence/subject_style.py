"""Conservative virtual-subject transforms for Phase-B adaptation episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class SubjectStyle:
    """Persistent motion traits shared by a virtual subject across executions."""

    pace: float
    accel_dynamic_gain: float
    gyro_gain: float
    smoothing: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def sample_subject_style(rng: np.random.Generator) -> SubjectStyle:
    """Draw deliberately mild defaults; the calibration script can tighten these ranges."""
    return SubjectStyle(
        pace=float(rng.uniform(0.88, 1.12)),
        accel_dynamic_gain=float(rng.uniform(0.85, 1.15)),
        gyro_gain=float(rng.uniform(0.85, 1.15)),
        smoothing=float(rng.uniform(0.0, 0.25)),
    )


@lru_cache(maxsize=512)
def _time_warp_coordinates(n: int, pace: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    progress = np.linspace(0.0, 1.0, n, dtype=np.float64)
    source = progress ** float(pace) * (n - 1)
    lo = np.floor(source).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    return lo, hi, source - lo


def _time_warp_fixed_length(x: np.ndarray, pace: float) -> np.ndarray:
    """Apply one monotone timing profile to a window or a batch of equal-length windows."""
    n = x.shape[-2]
    if n < 2 or abs(pace - 1.0) < 1e-6:
        return x
    lo, hi, frac = _time_warp_coordinates(n, float(pace))
    if x.ndim == 2:
        return (1.0 - frac[:, None]) * x[lo] + frac[:, None] * x[hi]
    return (
        (1.0 - frac[None, :, None]) * x[:, lo, :]
        + frac[None, :, None] * x[:, hi, :]
    )


@lru_cache(maxsize=64)
def _lowpass_sos(rate_hz: float, cutoff_hz: float) -> np.ndarray:
    return signal.butter(2, cutoff_hz, btype="low", fs=rate_hz, output="sos")


def apply_subject_style(
    data: np.ndarray,
    rate_hz: float,
    channel_mask,
    style: SubjectStyle,
) -> np.ndarray:
    """Apply one virtual subject's traits without changing units, shape, or gravity direction."""
    x = np.asarray(data, dtype=np.float64).copy()
    if x.ndim not in (2, 3) or x.shape[-1] != 6:
        raise ValueError(f"subject style expects canonical (...,T,6) IMU data, got {x.shape}")
    mask = np.asarray(channel_mask, dtype=bool)
    if mask.shape != (6,):
        raise ValueError("channel_mask must have six canonical IMU slots")
    x = _time_warp_fixed_length(x, style.pace)

    if mask[:3].all() and x.shape[-2] >= 16 and rate_hz > 1.0:
        cutoff = min(0.4, 0.2 * float(rate_hz))
        sos = _lowpass_sos(float(rate_hz), cutoff)
        gravity = signal.sosfiltfilt(sos, x[..., :3], axis=-2)
        x[..., :3] = gravity + style.accel_dynamic_gain * (x[..., :3] - gravity)
    if mask[3:].all():
        x[..., 3:] *= style.gyro_gain

    if style.smoothing > 0.0 and x.shape[-2] >= 9:
        # A fixed 6 Hz low-pass attenuates sharpness while preserving ordinary HAR dynamics.
        cutoff = min(6.0, 0.4 * float(rate_hz))
        if cutoff > 0:
            sos = _lowpass_sos(float(rate_hz), cutoff)
            smooth = signal.sosfiltfilt(sos, x, axis=-2)
            x = (1.0 - style.smoothing) * x + style.smoothing * smooth

    x[..., ~mask] = 0.0
    if not np.isfinite(x).all():
        raise FloatingPointError("subject-style transform produced non-finite IMU values")
    return x.astype(np.float32)
