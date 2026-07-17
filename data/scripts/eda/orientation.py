"""Orientation helpers for visualization-only gravity normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GravityAlignment:
    rotation: np.ndarray
    gravity_vector: np.ndarray
    gravity_norm: float
    alignment_error: float


def gravity_alignment(acceleration: np.ndarray) -> GravityAlignment:
    """Estimate one rigid rotation that maps mean acceleration onto positive z.

    The input must contain gravity and use g units. This removes pitch/roll from a
    window but cannot identify yaw around the gravity axis.
    """
    acceleration = np.asarray(acceleration, dtype=np.float64)
    if acceleration.ndim != 2 or acceleration.shape[1] != 3:
        raise ValueError(f"acceleration must have shape (T,3), got {acceleration.shape}")
    if not np.isfinite(acceleration).all():
        raise ValueError("acceleration contains non-finite values")

    gravity = acceleration.mean(axis=0)
    norm = float(np.linalg.norm(gravity))
    if not 0.5 <= norm <= 1.5:
        raise ValueError(
            f"mean acceleration norm {norm:.3f} g is not a credible gravity estimate"
        )

    source = gravity / norm
    target = np.asarray([0.0, 0.0, 1.0])
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.dot(source, target))

    if sine < 1e-10:
        if cosine > 0:
            rotation = np.eye(3)
        else:
            rotation = np.diag([1.0, -1.0, -1.0])
    else:
        skew = np.asarray([
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ])
        rotation = np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)

    error = float(np.linalg.norm(rotation @ source - target))
    return GravityAlignment(
        rotation=rotation.astype(np.float32),
        gravity_vector=gravity.astype(np.float32),
        gravity_norm=norm,
        alignment_error=error,
    )


def rotate_vectors(vectors: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply a column-vector rotation matrix to row-major xyz vectors."""
    vectors = np.asarray(vectors, dtype=np.float32)
    rotation = np.asarray(rotation, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or rotation.shape != (3, 3):
        raise ValueError("expected vectors (T,3) and rotation (3,3)")
    return vectors @ rotation.T


def centered_rms_normalize(vectors: np.ndarray) -> tuple[np.ndarray, float]:
    """Center a triad and scale it by vector RMS for unitless shape comparison."""
    centered = np.asarray(vectors, dtype=np.float32) - np.mean(vectors, axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if rms < 1e-6:
        return np.zeros_like(centered), rms
    return centered / rms, rms
