"""Unit tests for the auxiliary TF-C TimeEncoder (training-only time-domain rail)."""

from __future__ import annotations

import torch

from training.tokenizer.time_encoder import TimeEncoder

B, P, S, C, D = 3, 4, 32, 6, 48


def _inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    patches = torch.randn(B, P, S, C, generator=g)
    # per-patch valid length in [8, S]; some patches padded out entirely (N=0)
    patch_len = torch.randint(8, S + 1, (B, P), generator=g)
    patch_len[0, P - 1] = 0                                   # a fully padded patch
    channel_mask = torch.ones(B, C, dtype=torch.bool)
    channel_mask[1, 3:] = False                              # accel-only window (gyro absent)
    patch_pad = patch_len > 0
    # zero the padded time steps + absent channels, as the real collate guarantees
    idx = torch.arange(S).view(1, 1, S)
    tvalid = idx < patch_len.unsqueeze(-1)
    patches = patches * tvalid.unsqueeze(-1) * channel_mask.view(B, 1, 1, C)
    return patches, patch_len, channel_mask, patch_pad


def test_output_shape_and_gradient():
    enc = TimeEncoder(D, n_channels=C)
    patches, patch_len, cmask, ppad = _inputs()
    patches.requires_grad_(True)
    out = enc(patches, patch_len, cmask, patch_padding_mask=ppad)
    assert out.shape == (B, D)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert patches.grad is not None and torch.isfinite(patches.grad).all()
    # some parameter must receive gradient (the rail is alive)
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in enc.parameters())


def test_ignores_padded_time_steps():
    """Garbage in the padded region (t >= patch_len) must not change the embedding — the encoder
    zeros it before the conv, so it cannot leak across the valid/pad boundary."""
    enc = TimeEncoder(D, n_channels=C).eval()
    patches, patch_len, cmask, ppad = _inputs(1)
    with torch.no_grad():
        base = enc(patches, patch_len, cmask, patch_padding_mask=ppad)
        poisoned = patches.clone()
        idx = torch.arange(S).view(1, 1, S)
        pad_region = ~(idx < patch_len.unsqueeze(-1))         # (B,P,S) True where padded
        poisoned[pad_region.unsqueeze(-1).expand_as(poisoned)] += 5.0
        after = enc(poisoned, patch_len, cmask, patch_padding_mask=ppad)
    assert torch.allclose(base, after, atol=1e-5), "padded time steps leaked into the embedding"


def test_ignores_absent_channels():
    """Values on an absent channel must not change the embedding (channel-independent conv +
    channel-masked pool)."""
    enc = TimeEncoder(D, n_channels=C).eval()
    patches, patch_len, cmask, ppad = _inputs(2)
    with torch.no_grad():
        base = enc(patches, patch_len, cmask, patch_padding_mask=ppad)
        poisoned = patches.clone()
        absent = ~cmask                                       # (B,C)
        poisoned += (absent.view(B, 1, 1, C).float() * 7.0)
        after = enc(poisoned, patch_len, cmask, patch_padding_mask=ppad)
    assert torch.allclose(base, after, atol=1e-5), "absent-channel signal leaked into the embedding"


def test_ignores_padded_patches():
    """Garbage in a fully padded patch (patch_padding_mask False) must not change the embedding."""
    enc = TimeEncoder(D, n_channels=C).eval()
    patches, patch_len, cmask, ppad = _inputs(3)
    assert (~ppad).any(), "test needs at least one padded patch"
    with torch.no_grad():
        base = enc(patches, patch_len, cmask, patch_padding_mask=ppad)
        poisoned = patches.clone()
        poisoned[~ppad] += 9.0
        after = enc(poisoned, patch_len, cmask, patch_padding_mask=ppad)
    assert torch.allclose(base, after, atol=1e-5), "padded patches leaked into the embedding"


def test_accepts_per_window_patch_len():
    """patch_len may be per-window (B,) (single-scale collate) as well as per-patch (B,P)."""
    enc = TimeEncoder(D, n_channels=C).eval()
    patches, _, cmask, _ = _inputs(4)
    per_window_len = torch.full((B,), S)
    ppad = torch.ones(B, P, dtype=torch.bool)
    out = enc(patches, per_window_len, cmask, patch_padding_mask=ppad)
    assert out.shape == (B, D) and torch.isfinite(out).all()


def test_stays_small():
    """The rail is auxiliary — keep it small (well under a second backbone)."""
    n = sum(p.numel() for p in TimeEncoder(256, n_channels=C).parameters())
    assert n < 600_000, f"TimeEncoder unexpectedly large: {n} params"
