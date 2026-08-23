"""The continuous-kernel tokenizer must be RATE-COMPARABLE, or it has no reason to exist.

Design of record: docs/design/CONTINUOUS_KERNEL_FRONTEND.md. Every test here pins one property that,
if broken, makes the front end silently behave differently on a 20 Hz phone than on a 100 Hz watch —
which is the failure the module exists to prevent, and which a loss curve would never reveal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from model.tokenizer.continuous_kernel import ContinuousKernelTokenizer


def _band_limited_signal(rate_hz: float, duration_s: float = 6.0, seed: int = 0) -> np.ndarray:
    """One physical signal, sampled as a real ADC would: generate high, anti-alias DECIMATE.

    Point-sampling a high-rate signal is NOT the same as recording at a low rate; without the
    decimation this fixture would compare aliased garbage and the test would be meaningless.
    """
    from fractions import Fraction

    from scipy.signal import resample_poly

    high = 400.0
    t = np.arange(int(duration_s * high)) / high
    rng = np.random.default_rng(seed)
    x = np.zeros_like(t)
    for freq, amp in ((1.3, 1.0), (2.0, 0.7), (4.5, 0.4)):      # all below 9 Hz => 20 Hz-safe
        x += amp * np.sin(2 * math.pi * freq * t + rng.uniform(0, 2 * math.pi))
    ratio = Fraction(rate_hz / high).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator).astype(np.float32)


def _as_patches(signal: np.ndarray, rate_hz: float, patch_seconds: float = 1.0):
    """(1, P, S, C=1) padded patches + patch_len + mask, the loader's layout."""
    per = int(round(rate_hz * patch_seconds))
    n_patches = len(signal) // per
    trimmed = signal[: n_patches * per].reshape(n_patches, per, 1)
    patches = torch.from_numpy(trimmed).unsqueeze(0).float()
    lengths = torch.full((1, n_patches), per, dtype=torch.long)
    mask = torch.ones(1, n_patches, dtype=torch.bool)
    return patches, lengths, mask


def _analyze(module, rate_hz, **kw):
    patches, lengths, mask = _as_patches(_band_limited_signal(rate_hz, **kw), rate_hz)
    return module.analyze(patches, rate_hz, lengths, patch_mask=mask)


@pytest.fixture(scope="module")
def tokenizer():
    torch.manual_seed(0)
    return ContinuousKernelTokenizer().eval()


# ---------------------------------------------------------------------------------------------
# 1. THE test — the same physical signal must produce the same features at any sampling rate
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rate", [20.0, 25.0, 50.0])
def test_band_magnitudes_agree_across_sampling_rates(tokenizer, rate):
    """The property the whole design exists for.

    An earlier draft rounded each output frame's centre to the nearest input sample. At 20 Hz that
    is 25 ms of jitter — 45 degrees of phase error on a 5 Hz component — and it dropped correlation
    to 0.844. The kernel must instead be evaluated at the EXACT offsets of the real samples so it
    absorbs the sub-sample shift.
    """
    with torch.no_grad():
        reference = _analyze(tokenizer, 100.0)["compressed"]
        candidate = _analyze(tokenizer, rate)["compressed"]
    # only kernels whose harmonics this rate can represent are comparable; the rest are, correctly,
    # a different filter (and the nyquist mask says so).
    live = tokenizer.masks(rate, torch.tensor([6.0]))[0][0] >= 1.0
    assert bool(live.any())
    a = reference[..., live, :].flatten().double().numpy()
    b = candidate[..., live, :].flatten().double().numpy()
    n = min(len(a), len(b))
    correlation = float(np.corrcoef(a[:n], b[:n])[0, 1])
    assert correlation > 0.95, f"rate {rate}: correlation {correlation:.4f}"


def test_token_count_depends_on_duration_not_on_rate(tokenizer):
    """6 s must give the same number of tokens at 20 Hz and at 100 Hz. Shape invariance is half
    the contract; values are the other half."""
    shapes = set()
    for rate in (20.0, 50.0, 100.0):
        patches, lengths, mask = _as_patches(_band_limited_signal(rate), rate)
        with torch.no_grad():
            out = tokenizer(patches, rate, lengths, patch_mask=mask)
        shapes.add(tuple(out.shape))
    assert len(shapes) == 1, f"token shape varies with sampling rate: {shapes}"


@pytest.mark.parametrize("duration", [2.0, 4.0, 6.0])
def test_token_count_scales_with_duration(tokenizer, duration):
    patches, lengths, mask = _as_patches(_band_limited_signal(50.0, duration_s=duration), 50.0)
    with torch.no_grad():
        out = tokenizer(patches, 50.0, lengths, patch_mask=mask)
    assert out.shape[1] == int(duration), f"{duration}s -> {out.shape[1]} tokens"


# ---------------------------------------------------------------------------------------------
# 2. The three rules that make rates comparable
# ---------------------------------------------------------------------------------------------
def test_kernels_are_zero_mean_so_they_cannot_measure_gravity(tokenizer):
    """A DC offset is gravity/tilt, handled by the separate signed-DC feature. If kernels responded
    to it they would re-introduce the orientation confound through the back door."""
    for rate in (20.0, 100.0):
        offsets = (torch.arange(-int(rate), int(rate) + 1, dtype=torch.float32) / rate)
        with torch.no_grad():
            pair = tokenizer.kernel_at(offsets, rate)
        assert torch.allclose(pair.sum(-1), torch.zeros_like(pair.sum(-1)), atol=1e-5)


def test_response_magnitude_is_not_a_function_of_sampling_rate(tokenizer):
    """`dt = 1/r` makes the convolution an integral. Without it a 20 Hz recording returns about one
    FIFTH the response of the same motion at 100 Hz, and the model learns to identify the device."""
    with torch.no_grad():
        low = _analyze(tokenizer, 20.0)["compressed"]
        high = _analyze(tokenizer, 100.0)["compressed"]
    live = tokenizer.masks(20.0, torch.tensor([6.0]))[0][0] >= 1.0
    ratio = float(low[..., live, :].abs().mean() / high[..., live, :].abs().mean().clamp_min(1e-9))
    assert 0.6 < ratio < 1.6, f"magnitude ratio 20/100 Hz = {ratio:.3f}"


def test_harmonics_above_nyquist_are_dropped(tokenizer):
    """Band-limiting is exact: harmonic m sits at m/T_k Hz, so it is a coefficient mask. A kernel
    sampled without it would alias and behave differently at every rate."""
    nyq_low, _ = tokenizer.masks(20.0, torch.tensor([6.0]))
    nyq_high, _ = tokenizer.masks(100.0, torch.tensor([6.0]))
    assert (nyq_low[0] <= nyq_high[0] + 1e-6).all()
    assert nyq_low[0].min() < 1.0, "at 20 Hz some short kernels must lose harmonics"
    assert nyq_high[0].mean() > nyq_low[0].mean()


def test_amplitude_scales_linearly_before_compression(tokenizer):
    patches, lengths, mask = _as_patches(_band_limited_signal(50.0), 50.0)
    with torch.no_grad():
        one = tokenizer.analyze(patches, 50.0, lengths, patch_mask=mask)["compressed"]
        two = tokenizer.analyze(patches * 3.0, 50.0, lengths, patch_mask=mask)["compressed"]
    assert (torch.expm1(two) >= torch.expm1(one) - 1e-4).all()
    ratio = float(torch.expm1(two).mean() / torch.expm1(one).mean().clamp_min(1e-9))
    assert 2.0 < ratio < 4.0, ratio


# ---------------------------------------------------------------------------------------------
# 3. Robustness — the cases that have actually broken this codebase before
# ---------------------------------------------------------------------------------------------
def test_single_sample_window_produces_a_token_and_never_raises(tokenizer):
    """~0.1% of corpus windows carry ONE sample (wisdm, capture24, opportunity). This exact case
    crashed all four arms of the encoder-swap harness on 2026-08-22."""
    patches = torch.randn(2, 6, 50, 3)
    lengths = torch.zeros(2, 6, dtype=torch.long)
    lengths[:, 0] = 1
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 0] = True
    with torch.no_grad():
        out = tokenizer(patches, 50.0, lengths, patch_mask=mask)
    assert torch.isfinite(out).all()


def test_padded_region_cannot_leak_into_the_output(tokenizer):
    patches = torch.randn(1, 6, 60, 2)
    lengths = torch.full((1, 6), 60, dtype=torch.long)
    lengths[0, 3:] = 0
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, 3:] = False
    with torch.no_grad():
        before = tokenizer(patches, 60.0, lengths, patch_mask=mask)
        patches[0, 3:] = 1e4
        after = tokenizer(patches, 60.0, lengths, patch_mask=mask)
    assert torch.allclose(before, after, atol=1e-4)


def test_gabor_init_is_a_band_pass_bank_not_noise(tokenizer):
    """At init each kernel must peak near its own centre frequency — that is what makes step 0
    comparable to the physical filterbank instead of a random projection."""
    rate = 100.0
    for k in (8, 16, 24):
        span = float(tokenizer.spans[k])
        centre = float(tokenizer.centres[k])
        t = torch.arange(int(rate * 6.0), dtype=torch.float32) / rate
        best, best_freq = -1.0, None
        for probe in (centre / 3.0, centre, centre * 3.0):
            signal = torch.sin(2 * math.pi * probe * t).view(1, 1, -1, 1)
            lengths = torch.tensor([[signal.shape[2]]], dtype=torch.long)
            with torch.no_grad():
                energy = tokenizer.analyze(signal, rate, lengths)["compressed"][0, 0, k].mean()
            if float(energy) > best:
                best, best_freq = float(energy), probe
        assert best_freq == pytest.approx(centre, rel=1e-6), (
            f"kernel {k} (centre {centre:.2f} Hz) responded most to {best_freq:.2f} Hz")


def test_norm_statistics_round_trip(tokenizer):
    module = ContinuousKernelTokenizer().eval()
    assert float(module._norm_fitted) == 0.0
    module.reset_norm_accumulator()
    patches, lengths, mask = _as_patches(_band_limited_signal(50.0), 50.0)
    module.accumulate_norm_stats(patches, 50.0, lengths, patch_mask=mask)
    module.finalize_norm_stats()
    assert float(module._norm_fitted) == 1.0
    assert torch.isfinite(module.norm_mu).all() and (module.norm_sd > 0).all()


def test_gradients_reach_the_sixteen_coefficients(tokenizer):
    module = ContinuousKernelTokenizer()
    patches, lengths, mask = _as_patches(_band_limited_signal(50.0), 50.0)
    module(patches, 50.0, lengths, patch_mask=mask).pow(2).mean().backward()
    for name in ("cos_coeff", "sin_coeff", "log_sigma", "log_gain"):
        grad = getattr(module, name).grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, f"{name} received no gradient"


def test_is_not_yet_wired_into_the_encoder():
    """Deliberate: the module is ready to swap but NOT integrated, so an in-flight experiment
    cannot be perturbed by it. Delete this test in the commit that adds the --frontend arm."""
    import model.tokenizer.encoder as encoder_module

    source = open(encoder_module.__file__).read()
    assert "continuous_kernel" not in source
