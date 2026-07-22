"""Tests for the per-channel selective-SSM (Mamba) tokenizer front end.

Pins the drop-in contract, the physical Δ=1/rate rate-conditioning (the whole point), gravity/DC
preservation, padding-masking, and encoder integration behind the `frontend='mamba'` switch.
"""

import math
import torch

from model.tokenizer.mamba_frontend import SelectiveSSMChannelTokenizer
from model.tokenizer.scattering import build_frontend
from model.tokenizer.encoder import SetTokenizerEncoder


def _tok(**kw):
    torch.manual_seed(0)
    return SelectiveSSMChannelTokenizer(d_model=16, d_state=8, d_inner=32, dft_size=256, **kw).eval()


def test_dropin_shape_and_finite():
    tok = _tok()
    B, P, S, C = 2, 3, 256, 6
    patches = torch.randn(B, P, S, C)
    out = tok(patches, sampling_rate_hz=50.0, patch_len_samples=torch.full((B, P), 50))
    assert out.shape == (B, P, C, 16) and torch.isfinite(out).all()


def test_build_frontend_mamba():
    tok = build_frontend("mamba", d_model=16, d_state=8)
    assert isinstance(tok, SelectiveSSMChannelTokenizer)


def test_rate_changes_the_token():
    """Δ depends on rate, so the SAME sample array at two rates must produce different tokens."""
    tok = _tok(standardize=False)
    B, P, S, C = 1, 1, 64, 1
    x = torch.randn(B, P, S, C)
    N = torch.full((B, P), 64)
    a = tok(x, 50.0, N)
    b = tok(x, 100.0, N)
    assert not torch.allclose(a, b, atol=1e-4), "rate is not actually conditioning the SSM"


def test_delta_equals_one_over_rate_gives_physical_alignment():
    """THE point: the same physical motion sampled at 50 vs 100 Hz should give CLOSER tokens under
    the correct native rate than a wrong-rate control that ignores the physical time step."""
    tok = _tok(standardize=False)
    S = 256
    t100 = torch.arange(100) / 100.0          # 1 s at 100 Hz
    t50 = torch.arange(50) / 50.0             # 1 s at 50 Hz
    freq = 2.0                                # a 2 Hz physical oscillation
    sig100 = torch.sin(2 * math.pi * freq * t100)
    sig50 = torch.sin(2 * math.pi * freq * t50)

    def pack(sig):
        x = torch.zeros(1, 1, S, 1); x[0, 0, :len(sig), 0] = sig
        return x

    N100 = torch.tensor([[100]]); N50 = torch.tensor([[50]])
    tok100 = tok(pack(sig100), 100.0, N100)
    tok50_right = tok(pack(sig50), 50.0, N50)          # correct native rate -> Δ=1/50
    tok50_wrong = tok(pack(sig50), 100.0, N50)         # LIE: same samples, claim 100 Hz -> Δ=1/100

    d_right = (tok100 - tok50_right).norm()
    d_wrong = (tok100 - tok50_wrong).norm()
    assert d_right < d_wrong, (
        f"physical Δ=1/rate should align cross-rate tokens: correct-rate dist {d_right:.4f} "
        f"should be < wrong-rate dist {d_wrong:.4f}")


def test_gravity_dc_is_preserved_not_normalized_away():
    """A DC offset (gravity) must change the token — instance-norm would wrongly erase it."""
    tok = _tok(standardize=False)
    B, P, S, C = 1, 1, 64, 1
    N = torch.full((B, P), 64)
    base = torch.randn(B, P, S, C)
    lifted = base + 1.0                        # add a 1 g DC offset on the (single) channel
    assert not torch.allclose(tok(base, 50.0, N), tok(lifted, 50.0, N), atol=1e-4), \
        "DC/gravity offset must affect the token"


def test_padding_beyond_N_is_ignored():
    """Garbage in the zero-pad region (t >= N) must not change the token."""
    tok = _tok(standardize=False)
    B, P, S, C = 1, 1, 128, 1
    N = torch.tensor([[40]])
    x = torch.zeros(B, P, S, C); x[0, 0, :40, 0] = torch.randn(40)
    y = x.clone(); y[0, 0, 40:, 0] = torch.randn(S - 40)   # noise only in the pad region
    assert torch.allclose(tok(x, 50.0, N), tok(y, 50.0, N), atol=1e-5)


def test_standardization_is_per_modality_not_global():
    """Accel and gyro must get SEPARATE scale — a single global scalar under-normalises one."""
    tok = _tok(standardize=True)
    assert tok.norm_mu.numel() == 2, "expected per-modality (accel/gyro) stats, not one global scalar"
    # accel channels ~unit scale, gyro channels ~5x scale
    cal = torch.randn(8, 4, 256, 6)
    cal[..., 3:6] *= 5.0
    tok.fit_norm_stats(cal, patch_len_samples=torch.full((8, 4), 200))
    accel_sd, gyro_sd = tok.norm_sd[0].item(), tok.norm_sd[1].item()
    assert gyro_sd > 3 * accel_sd, f"per-modality σ should track the 5x gyro scale (accel {accel_sd:.2f}, gyro {gyro_sd:.2f})"


def test_per_modality_standardization_preserves_relative_gravity():
    """A shared per-modality μ shifts the 3 accel axes equally -> relative gravity direction survives."""
    tok = _tok(standardize=True)
    cal = torch.randn(8, 4, 256, 6) * 2.0 + torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # gravity on acc_x
    tok.fit_norm_stats(cal, patch_len_samples=torch.full((8, 4), 200))
    B, P, S, C = 1, 1, 64, 6
    N = torch.full((B, P), 64)
    x = torch.randn(B, P, S, C)
    x_lift = x.clone(); x_lift[..., 0] += 1.0    # extra gravity offset on acc_x only
    assert not torch.allclose(tok(x, 50.0, N), tok(x_lift, 50.0, N), atol=1e-4), \
        "per-modality standardization must not erase a per-axis gravity difference"


def test_missing_patch_len_and_dft_size_raises():
    """The N=0 footgun: no patch_len and no dft_size must fail loud, not return a constant."""
    tok = SelectiveSSMChannelTokenizer(d_model=8)      # no dft_size
    try:
        tok(torch.randn(1, 1, 64, 6), 50.0, None)
        raise AssertionError("expected ValueError for unknown patch length")
    except ValueError:
        pass


def test_encoder_mamba_frontend_end_to_end():
    enc = SetTokenizerEncoder(d_model=32, num_layers=1, num_heads=4, dim_feedforward=64,
                              frontend="mamba", d_state=8).train()
    from model.tokenizer.mamba_frontend import SelectiveSSMChannelTokenizer
    assert isinstance(enc.filterbank, SelectiveSSMChannelTokenizer)
    B, P, C = 2, 3, 6; S = 256
    patches = torch.randn(B, P, S, C, requires_grad=True)
    texts = [["accelerometer x-axis worn at the wrist (watch)"] * C for _ in range(B)]
    out = enc(patches, 50.0, torch.full((B, P), 50), texts,
              torch.arange(P).float().unsqueeze(0).repeat(B, 1),
              channel_mask=torch.ones(B, C, dtype=torch.bool),
              patch_padding_mask=torch.ones(B, P, dtype=torch.bool))
    assert out["pooled"].shape == (B, 32) and torch.isfinite(out["pooled"]).all()
    out["pooled"].square().mean().backward()
    assert enc.filterbank.blocks[0].A_log.grad is not None and torch.isfinite(enc.filterbank.blocks[0].A_log.grad).all()


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))


def test_legacy_learnable_kwarg_still_builds_all_arms():
    """Regression (2026-07-22): routing the encoder through build_frontend made `learnable` get
    passed twice, breaking the DEFAULT fixed path used by pretrain.py / eval_transfer. The unit
    tests missed it because none passed `learnable=`. Pin all three construction paths."""
    from model.tokenizer.filterbank import PhysicalFilterbankTokenizer
    from model.tokenizer.mamba_frontend import SelectiveSSMChannelTokenizer
    k = dict(d_model=16, num_layers=1, num_heads=2, dim_feedforward=32, dft_size=256)
    fixed = SetTokenizerEncoder(learnable=False, **k)          # pretrain.py's fixed arm
    learn = SetTokenizerEncoder(learnable=True, **k)           # pretrain.py's learnable arm
    mamba = SetTokenizerEncoder(frontend="mamba", **k)
    assert isinstance(fixed.filterbank, PhysicalFilterbankTokenizer) and not fixed.filterbank.learnable
    assert isinstance(learn.filterbank, PhysicalFilterbankTokenizer) and learn.filterbank.learnable
    assert isinstance(mamba.filterbank, SelectiveSSMChannelTokenizer)


def test_kernel_matches_reference_scan_on_cuda():
    """The fused mamba kernel and the pure-PyTorch reference scan must agree (they are the same math)."""
    import pytest
    from model.tokenizer.mamba_frontend import _HAS_KERNEL
    if not (torch.cuda.is_available() and _HAS_KERNEL):
        pytest.skip("mamba kernel / CUDA not available")
    dev = torch.device("cuda")
    torch.manual_seed(0)
    tok = SelectiveSSMChannelTokenizer(d_model=32, d_state=8, d_inner=64, dft_size=256).to(dev).eval()
    x = torch.randn(4, 3, 256, 6, device=dev); N = torch.full((4, 3), 200, device=dev)
    tok.use_kernel = True;  yk = tok(x, 50.0, N)
    tok.use_kernel = False; yr = tok(x, 50.0, N)
    assert torch.allclose(yk, yr, atol=1e-3), f"kernel vs ref max diff {(yk-yr).abs().max():.2e}"
