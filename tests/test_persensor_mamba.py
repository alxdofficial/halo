"""Tests for the per-sensor, patch-free, causal continuous-scan Mamba tokenizer.

Pins the four design properties: per-sensor (B,P,1,d) output, CAUSAL readout (streaming-safe),
CONTINUOUS state across patch boundaries, and padding-carry — plus channel masking and calibration.
"""

import torch

from model.tokenizer.mamba_frontend import PerSensorMambaTokenizer
from model.tokenizer.scattering import build_frontend


def _tok(**kw):
    torch.manual_seed(0)
    return PerSensorMambaTokenizer(d_model=16, n_layers=2, d_state=8, d_inner=32, **kw).eval()


def test_output_shape_is_per_sensor_single_channel():
    tok = _tok()
    x = torch.randn(2, 3, 8, 6)
    y = tok(x, 50.0, torch.full((2, 3), 8))
    assert y.shape == (2, 3, 1, 16)                 # (B, P, 1, d): one token/patch, single sensor channel


def test_build_frontend_registers_mamba_sensor():
    fe = build_frontend("mamba_sensor", d_model=16, n_layers=1, d_state=8, d_inner=32)
    assert isinstance(fe, PerSensorMambaTokenizer)
    assert fe(torch.randn(1, 2, 8, 6), 50.0, torch.full((1, 2), 8)).shape == (1, 2, 1, 16)


def test_readout_is_CAUSAL_future_patches_do_not_change_earlier_tokens():
    """The streaming guarantee: altering the LAST patch must leave every earlier token bit-identical."""
    tok = _tok()
    B, P, S, C = 2, 4, 8, 6
    x = torch.randn(B, P, S, C)
    y0 = tok(x, 50.0, torch.full((B, P), S))
    x2 = x.clone()
    x2[:, -1] = torch.randn(B, S, C)                # perturb ONLY the final patch
    y1 = tok(x2, 50.0, torch.full((B, P), S))
    assert torch.allclose(y0[:, :-1], y1[:, :-1], atol=0), "earlier tokens must not see future patches"
    assert not torch.allclose(y0[:, -1], y1[:, -1]), "the perturbed patch's own token should change"


def test_scan_is_CONTINUOUS_state_carries_across_patch_boundaries():
    """Not per-patch: perturbing an EARLIER patch must change a LATER token (state propagates)."""
    tok = _tok()
    B, P, S, C = 2, 4, 8, 6
    x = torch.randn(B, P, S, C)
    y0 = tok(x, 50.0, torch.full((B, P), S))
    x2 = x.clone()
    x2[:, 0] = torch.randn(B, S, C)                 # perturb the FIRST patch
    y1 = tok(x2, 50.0, torch.full((B, P), S))
    assert not torch.allclose(y0[:, 2], y1[:, 2]), "a per-patch tokenizer would leave token 2 unchanged"


def test_padding_does_not_corrupt_and_output_is_finite():
    tok = _tok()
    B, P, S, C = 2, 3, 8, 6
    x = torch.randn(B, P, S, C)
    plen = torch.tensor([[5, 5, 5], [8, 8, 8]])     # first item has 3 padded steps per patch
    y = tok(x, 50.0, plen)
    assert torch.isfinite(y).all()
    # token for a fully-real patch equals its readout regardless of what sits in the padding region
    x2 = x.clone(); x2[0, :, 5:] = 999.0            # garbage in patch 0's padding of item 0
    y2 = tok(x2, 50.0, plen)
    assert torch.allclose(y[0], y2[0], atol=1e-4), "padded samples must not enter the readout"


def test_absent_channels_are_zeroed_before_the_stem():
    tok = _tok()
    B, P, S, C = 2, 2, 8, 6
    x = torch.randn(B, P, S, C)
    cmask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)  # item0 accel-only
    y = tok(x, 50.0, torch.full((B, P), S), channel_mask=cmask)
    # garbage in item0's absent gyro channels must not change item0's tokens
    x2 = x.clone(); x2[0, :, :, 3:] = 42.0
    y2 = tok(x2, 50.0, torch.full((B, P), S), channel_mask=cmask)
    assert torch.allclose(y[0], y2[0], atol=1e-5)


def test_rate_conditioning_changes_the_representation():
    tok = _tok()
    x = torch.randn(2, 3, 8, 6); N = torch.full((2, 3), 8)
    assert not torch.allclose(tok(x, 25.0, N), tok(x, 100.0, N)), "Δ=1/rate must make rate matter"


def test_calibration_fits_per_modality_stats():
    tok = _tok()
    tok.reset_norm_accumulator()
    tok.accumulate_norm_stats(torch.randn(4, 3, 16, 6) * torch.tensor([1., 1, 1, 3, 3, 3]),
                              50.0, torch.full((4, 3), 16),
                              patch_mask=torch.ones(4, 3, dtype=torch.bool),
                              channel_mask=torch.ones(4, 6, dtype=torch.bool))
    tok.finalize_norm_stats()
    assert tok._norm_fitted.item() == 1.0 and tok.norm_sd.numel() == 2      # accel vs gyro
    assert tok.norm_sd[1] > tok.norm_sd[0]                                  # gyro was scaled up


def test_gradients_flow_and_are_finite():
    torch.manual_seed(0)
    tok = PerSensorMambaTokenizer(d_model=16, n_layers=2, d_state=8, d_inner=32).train()
    x = torch.randn(2, 3, 8, 6)
    tok(x, 50.0, torch.full((2, 3), 8)).pow(2).mean().backward()
    grads = [p.grad for p in tok.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g is not None for g in grads) == sum(1 for _ in tok.parameters())


def test_adaptation_interface_parity():
    tok = _tok()
    reg = tok.adaptation_regularization()
    assert torch.isfinite(reg)
    assert "frontend/delta_mult_baseline_mean" in tok.adaptation_summary()


def test_kernel_matches_reference_on_cuda():
    import pytest
    from model.tokenizer.mamba_frontend import _HAS_KERNEL
    if not (torch.cuda.is_available() and _HAS_KERNEL):
        pytest.skip("mamba kernel / CUDA not available")
    dev = torch.device("cuda")
    torch.manual_seed(0)
    tok = PerSensorMambaTokenizer(d_model=32, n_layers=2, d_state=8, d_inner=64).to(dev).eval()
    x = torch.randn(2, 3, 16, 6, device=dev); N = torch.tensor([[12, 12, 12], [16, 16, 16]], device=dev)
    tok.use_kernel = True;  yk = tok(x, 50.0, N)
    tok.use_kernel = False; yr = tok(x, 50.0, N)
    assert torch.allclose(yk, yr, atol=1e-3), f"kernel vs ref max diff {(yk - yr).abs().max():.2e}"


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
