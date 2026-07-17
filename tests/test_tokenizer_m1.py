"""M1 gate tests (build plan M1): the M0 invariances must hold in the PORTED code,
missing-channel paths must mask (never crash), and rate-invariance must hold across
20/50/100 Hz. Real-data tests skip when grids are absent.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from model.tokenizer.preprocess import (
    GRAVITY_MIN_G,
    accel_gyro_triads,
    estimate_gravity,
    find_triads,
    gravity_align,
)
from model.tokenizer.primitives import (
    CADENCE_MIN_MOTION_G,
    cadence,
    compute_primitives,
    eigen_ratios,
)
from model.tokenizer.scattering import build_frontend

CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
RATE = 60.0
GRIDS_PRESENT = (
    Path(__file__).resolve().parents[1]
    / "data/datasets/motionsense/grids/harmonised"
).exists()


# --------------------------------------------------------------------------- helpers
def random_so3(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(4, generator=g)
    w, x, y, z = (q / q.norm()).tolist()
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotate_window(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Rotate every xyz triad of (B, T, 6) by the same R (joint, physical)."""
    out = x.clone()
    for triad in find_triads(CHANNELS).values():
        idx = list(triad)
        out[:, :, idx] = torch.einsum("ij,btj->bti", r, x[:, :, idx])
    return out


def walking_like_batch(b: int = 3, t: int = 360, seed: int = 1) -> torch.Tensor:
    """Synthetic gravity-present 'gait': 2 Hz vertical step bounce (ALONG gravity — the
    physically-correct axis: components perpendicular to gravity only enter |acc| at
    second order) + 1 Hz stride asymmetry + noise."""
    g = torch.Generator().manual_seed(seed)
    time = torch.arange(t) / RATE
    x = 0.02 * torch.randn(b, t, 6, generator=g)
    x[:, :, 2] += 1.0 + 0.25 * torch.sin(2 * math.pi * 2.0 * time)  # gravity + step bounce (2 Hz)
    x[:, :, 0] += 0.10 * torch.sin(2 * math.pi * 1.0 * time)        # stride asymmetry (1 Hz)
    x[:, :, 2] += 0.08 * torch.sin(2 * math.pi * 1.0 * time + 0.3)  # vertical stride component
    x[:, :, 3] += 0.20 * torch.sin(2 * math.pi * 2.0 * time + 0.7)  # coherent gyro
    return x


def real_windows(n: int = 12) -> torch.Tensor:
    from data.scripts.eda.grid_io import discover_grids

    ref = next(r for r in discover_grids("harmonised") if r.dataset == "motionsense")
    walk = [i for i, l in enumerate(ref.labels) if l == "walking"][:n]
    return torch.tensor(ref.load_data()[walk].copy(), dtype=torch.float32)


# ------------------------------------------------------------------ channel handling
def test_triads_found_by_text_not_position():
    shuffled = ["gyro_z", "acc_y", "other", "acc_x", "gyro_x", "acc_z", "gyro_y"]
    acc, gyro = accel_gyro_triads(shuffled)
    assert acc == (3, 1, 5) and gyro == (4, 6, 0)


def test_missing_gyro_masks_coherence_only():
    x = walking_like_batch()[:, :, :3]
    prims = compute_primitives(x, CHANNELS[:3], RATE)
    assert not prims["coherence"].valid.any()
    for name in ("grav_band_energy", "eigen_ratios", "spectral_shape", "cadence"):
        assert prims[name].valid.all(), name


def test_no_accel_all_invalid_no_crash():
    x = torch.randn(2, 360, 3)
    prims = compute_primitives(x, ["gyro_x", "gyro_y", "gyro_z"], RATE)
    assert all(not p.valid.any() for p in prims.values())


# ----------------------------------------------------------------- gravity alignment
def test_gravity_align_is_canonicalizing():
    """align(R @ x) must equal align(x): rotation then re-alignment cancels (up to yaw,
    which the [vert, horiz-pooled] energies are blind to)."""
    x = walking_like_batch()
    rot = rotate_window(x, random_so3(3))
    p0 = compute_primitives(x, CHANNELS, RATE)["grav_band_energy"]
    p1 = compute_primitives(rot, CHANNELS, RATE)["grav_band_energy"]
    assert p0.valid.all() and p1.valid.all()
    cos = torch.nn.functional.cosine_similarity(p0.values, p1.values, dim=1)
    assert cos.min() > 0.999, f"grav_band_energy drifted under rotation: {cos}"


def test_gravity_absent_flagged_not_rotated():
    x = 0.05 * torch.randn(2, 360, 6)          # gravity-removed-like (|lowpass| << MIN_G)
    aligned, r, ok = gravity_align(x, CHANNELS, RATE)
    assert not ok.any()
    assert torch.allclose(aligned, x)
    assert torch.allclose(r, torch.eye(3).expand(2, 3, 3))


def test_estimate_gravity_units_are_g():
    x = walking_like_batch()
    gravity, present = estimate_gravity(x[:, :, :3], RATE)
    assert present.all()
    assert (gravity.norm(dim=1) - 1.0).abs().max() < 0.2      # ~1 g, not ~9.8
    assert GRAVITY_MIN_G < 1.0 < 2.0                          # threshold sanity


# ------------------------------------------------- invariances (the M0 result, in code)
@pytest.mark.parametrize("family", ["eigen_ratios", "coherence", "spectral_shape", "cadence"])
def test_rotation_invariance_synthetic(family):
    x = walking_like_batch()
    rot = rotate_window(x, random_so3(7))
    p0, p1 = (compute_primitives(v, CHANNELS, RATE)[family] for v in (x, rot))
    both = p0.valid & p1.valid
    assert both.any()
    a, b = p0.values[both], p1.values[both]
    finite = torch.isfinite(a) & torch.isfinite(b)
    assert torch.allclose(a[finite], b[finite], atol=1e-3), family


@pytest.mark.parametrize("gain", [0.7, 1.3])
def test_gain_invariance_synthetic(gain):
    x = walking_like_batch()
    for family in ("eigen_ratios", "spectral_shape", "cadence", "coherence"):
        p0 = compute_primitives(x, CHANNELS, RATE)[family]
        p1 = compute_primitives(x * gain, CHANNELS, RATE)[family]
        both = p0.valid & p1.valid
        a, b = p0.values[both], p1.values[both]
        finite = torch.isfinite(a) & torch.isfinite(b)
        assert torch.allclose(a[finite], b[finite], atol=1e-3), family


@pytest.mark.skipif(not GRIDS_PRESENT, reason="real grids not built")
def test_rotation_invariance_real_data():
    """The M1 gate on REAL windows: rotation drift below threshold for the kept set."""
    x = real_windows()
    rot = rotate_window(x, random_so3(11))
    for family in ("eigen_ratios", "spectral_shape", "grav_band_energy"):
        p0 = compute_primitives(x, CHANNELS, RATE)[family]
        p1 = compute_primitives(rot, CHANNELS, RATE)[family]
        both = p0.valid & p1.valid
        assert both.any(), family
        a = p0.values[both].reshape(both.sum(), -1)
        b = p1.values[both].reshape(both.sum(), -1)
        finite = torch.isfinite(a) & torch.isfinite(b)
        a, b = torch.where(finite, a, torch.zeros_like(a)), torch.where(finite, b, torch.zeros_like(b))
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
        assert cos.min() > 0.99, f"{family} drifted on real data: {cos.min()}"


# ------------------------------------------------------------------- rate invariance
@pytest.mark.parametrize("rate", [20.0, 50.0, 100.0])
def test_primitive_rate_invariance(rate):
    """Same 6 s continuous-time signal sampled at 20/50/100 Hz -> same primitives."""
    def sample(r: float) -> torch.Tensor:
        t = torch.arange(int(6 * r)) / r
        x = torch.zeros(1, len(t), 6)
        x[0, :, 0] = 0.3 * torch.sin(2 * math.pi * 2.0 * t)
        x[0, :, 1] = 0.1 * torch.sin(2 * math.pi * 1.0 * t + 0.4)
        x[0, :, 2] = 1.0
        x[0, :, 3] = 0.2 * torch.sin(2 * math.pi * 2.0 * t + 0.7)
        return x

    ref = compute_primitives(sample(60.0), CHANNELS, 60.0)
    got = compute_primitives(sample(rate), CHANNELS, rate)
    for family in ("eigen_ratios", "spectral_shape", "cadence", "grav_band_energy"):
        a, b = ref[family].values.flatten(), got[family].values.flatten()
        finite = torch.isfinite(a) & torch.isfinite(b)
        assert torch.allclose(a[finite], b[finite], atol=0.08), (
            f"{family} not rate-invariant at {rate} Hz: max diff "
            f"{(a[finite] - b[finite]).abs().max():.4f}"
        )


@pytest.mark.parametrize("rate", [20.0, 50.0, 100.0])
def test_filterbank_rate_invariance_multitone(rate):
    """Ported-tokenizer gate: one multi-tone signal at 20/50/100 Hz.

    The invariance claim is on SHARED OBSERVABLE bands — at 20 Hz the top bands are
    (correctly) Nyquist-masked and their mask features (deliberately) differ, so full
    tokens are only compared where every band is observable (50/100 Hz)."""
    tok = build_frontend("fixed", d_model=64)
    tok.eval()

    def patch(r: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = int(2.0 * r)
        t = torch.arange(n) / r
        sig = torch.sin(2 * math.pi * 2.0 * t) + 0.5 * torch.sin(2 * math.pi * 5.0 * t)
        out = torch.zeros(1, 1, tok.S, 1)
        out[0, 0, :n, 0] = sig
        return out, torch.tensor([r]), torch.tensor([n])

    with torch.no_grad():
        p60, r60, n60 = patch(60.0)
        pr, rr, nr = patch(rate)
        # Shared-observable-band energies: the physical invariance claim.
        e60, centers, sigma, _ = tok._band_energy(p60, r60, n60)
        er, _, _, _ = tok._band_energy(pr, rr, nr)
        o60, _ = tok._observability_masks(r60, n60, centers, sigma)
        orr, _ = tok._observability_masks(rr, nr, centers, sigma)
        both = (o60 * orr).bool().squeeze(0)
        assert both.sum() >= 20, "too few shared observable bands to compare"
        a = torch.log1p(e60)[0, 0, 0][both]
        b = torch.log1p(er)[0, 0, 0][both]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0)
        assert cos > 0.999, f"shared-band energies drift at {rate} Hz: cos={cos:.5f}"
        # Full tokens: only where ALL bands are observable at both rates.
        if bool(orr.all()):
            t60 = tok(p60, 60.0, n60)
            tr = tok(pr, rate, nr)
            cos_tok = torch.nn.functional.cosine_similarity(
                t60.flatten(), tr.flatten(), dim=0
            )
            assert cos_tok > 0.999, f"tokens drift at {rate} Hz: cos={cos_tok:.5f}"


# ------------------------------------------------------------------ cadence behavior
def test_cadence_motion_floor_blocks_static():
    x = torch.zeros(2, 360, 3)
    x[:, :, 2] = 1.0
    x += 0.5 * CADENCE_MIN_MOTION_G * torch.sin(
        2 * math.pi * 0.1 * torch.arange(360) / RATE
    ).view(1, -1, 1)                                             # slow drift, sub-floor
    assert not cadence(x, RATE).valid.any()


def test_cadence_octave_aware_prefers_step_rate():
    """Strong 2 Hz step + weaker 1 Hz stride asymmetry: raw autocorr peaks at the
    stride lag; the octave rule must recover the 2 Hz step rate."""
    x = walking_like_batch(b=4)
    p = cadence(x[:, :, :3], RATE)
    assert p.valid.all()
    hz = 2.0 ** p.values[:, 0]
    assert ((hz - 2.0).abs() < 0.3).all(), f"expected ~2 Hz step rate, got {hz}"


def test_eigen_ratios_live_on_simplex():
    p = eigen_ratios(walking_like_batch()[:, :, :3], RATE)
    vals = p.values[torch.isfinite(p.values).all(dim=2)]
    assert ((vals >= -1e-5) & (vals <= 1 + 1e-5)).all()
    assert torch.allclose(vals.sum(dim=1), torch.ones(len(vals)), atol=1e-4)


# ------------------------------------------------------------------ frontend factory
def test_frontend_factory_flags():
    assert not build_frontend("fixed").learnable
    assert build_frontend("sincnet").learnable
    with pytest.raises(NotImplementedError):
        build_frontend("scattering")
    with pytest.raises(NotImplementedError):
        build_frontend("free_conv")
    with pytest.raises(ValueError):
        build_frontend("nonsense")
