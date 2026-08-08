"""Conservative virtual-subject transforms for Phase-B adaptation episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

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


def _time_warp_fixed_length(x: np.ndarray, pace: float) -> np.ndarray:
    """Apply a monotone timing profile while retaining duration, endpoints, and sample clock."""
    n = len(x)
    if n < 2 or abs(pace - 1.0) < 1e-6:
        return x
    progress = np.linspace(0.0, 1.0, n, dtype=np.float64)
    source = progress ** float(pace) * (n - 1)
    lo = np.floor(source).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    frac = (source - lo)[:, None]
    return (1.0 - frac) * x[lo] + frac * x[hi]


def apply_subject_style(
    data: np.ndarray,
    rate_hz: float,
    channel_mask,
    style: SubjectStyle,
) -> np.ndarray:
    """Apply one virtual subject's traits without changing units, shape, or gravity direction."""
    x = np.asarray(data, dtype=np.float64).copy()
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError(f"subject style expects canonical (T,6) IMU data, got {x.shape}")
    mask = np.asarray(channel_mask, dtype=bool)
    if mask.shape != (6,):
        raise ValueError("channel_mask must have six canonical IMU slots")
    x = _time_warp_fixed_length(x, style.pace)

    if mask[:3].all() and len(x) >= 16 and rate_hz > 1.0:
        cutoff = min(0.4, 0.2 * float(rate_hz))
        sos = signal.butter(2, cutoff, btype="low", fs=float(rate_hz), output="sos")
        gravity = signal.sosfiltfilt(sos, x[:, :3], axis=0)
        x[:, :3] = gravity + style.accel_dynamic_gain * (x[:, :3] - gravity)
    if mask[3:].all():
        x[:, 3:] *= style.gyro_gain

    if style.smoothing > 0.0 and len(x) >= 9:
        # A fixed 6 Hz low-pass attenuates sharpness while preserving ordinary HAR dynamics.
        cutoff = min(6.0, 0.4 * float(rate_hz))
        if cutoff > 0:
            sos = signal.butter(2, cutoff, btype="low", fs=float(rate_hz), output="sos")
            smooth = signal.sosfiltfilt(sos, x, axis=0)
            x = (1.0 - style.smoothing) * x + style.smoothing * smooth

    x[:, ~mask] = 0.0
    if not np.isfinite(x).all():
        raise FloatingPointError("subject-style transform produced non-finite IMU values")
    return x.astype(np.float32)
