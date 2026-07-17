import numpy as np
import pytest

from data.scripts.eda.orientation import (
    centered_rms_normalize,
    gravity_alignment,
    rotate_vectors,
)


def test_gravity_alignment_maps_mean_to_positive_z_and_preserves_norms() -> None:
    base = np.tile(np.asarray([0.6, -0.3, 0.8], dtype=np.float32), (100, 1))
    base[:, 0] += np.sin(np.linspace(0, 4 * np.pi, 100)) * 0.05

    estimate = gravity_alignment(base)
    aligned = rotate_vectors(base, estimate.rotation)

    assert np.allclose(aligned.mean(axis=0)[:2], 0.0, atol=1e-6)
    assert aligned.mean(axis=0)[2] > 0.99
    assert np.allclose(np.linalg.norm(aligned, axis=1), np.linalg.norm(base, axis=1))
    assert estimate.alignment_error < 1e-6


def test_same_rotation_is_applied_to_gyro_without_changing_magnitude() -> None:
    acc = np.tile(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), (20, 1))
    gyro = np.stack([np.linspace(0, 1, 20), np.zeros(20), np.ones(20)], axis=1)
    estimate = gravity_alignment(acc)

    rotated = rotate_vectors(gyro, estimate.rotation)

    assert np.allclose(np.linalg.norm(rotated, axis=1), np.linalg.norm(gyro, axis=1))


def test_gravity_alignment_rejects_gravity_removed_signal() -> None:
    with pytest.raises(ValueError, match="credible gravity estimate"):
        gravity_alignment(np.zeros((100, 3), dtype=np.float32))


def test_centered_rms_normalization_has_zero_mean_and_unit_vector_rms() -> None:
    values = np.asarray([[1, 2, 3], [3, 4, 5], [5, 6, 7]], dtype=np.float32)

    normalized, rms = centered_rms_normalize(values)

    assert rms > 0
    assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-7)
    assert np.isclose(np.sqrt(np.mean(np.sum(normalized**2, axis=1))), 1.0)


def test_summed_frequency_power_is_rotation_invariant() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(256, 3)).astype(np.float32)
    rotation = gravity_alignment(np.tile([0.4, 0.8, 0.5], (64, 1))).rotation
    rotated = rotate_vectors(values, rotation)

    original_power = np.sum(np.abs(np.fft.rfft(values, axis=0)) ** 2, axis=1)
    rotated_power = np.sum(np.abs(np.fft.rfft(rotated, axis=0)) ** 2, axis=1)

    assert np.allclose(original_power, rotated_power, rtol=1e-5, atol=1e-5)
