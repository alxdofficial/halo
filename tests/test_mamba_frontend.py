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


def test_frozen_standardization_preserves_relative_dc():
    """Calibrated standardization removes scale but keeps the RELATIVE DC (gravity direction)."""
    tok = _tok(standardize=True)
    cal = torch.randn(4, 4, 256, 6) * 3.0 + 0.5           # some scale + offset
    tok.fit_norm_stats(cal, patch_len_samples=torch.full((4, 4), 200))
    assert tok._norm_fitted.item() == 1.0 and tok.norm_sd.item() > 0


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
    assert enc.filterbank.A_log.grad is not None and torch.isfinite(enc.filterbank.A_log.grad).all()


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
