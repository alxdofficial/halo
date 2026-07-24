"""Auxiliary TIME-domain window encoder for the Time-Frequency Consistency (TF-C) objective.

TF-C (Zhang et al. 2022) pulls together two *domain views* of the SAME window: the
frequency-domain embedding (our main filterbank/transformer encoder's pooled output) and a
time-domain embedding produced HERE, over the raw native-rate samples. Pull the two views of
one window together, push different windows apart (a plain NT-Xent). It is augmentation-free:
the two views are different *representations* of one signal, not two augmentations.

This module is TRAINING-ONLY — it is discarded at inference. Nothing in the eval / transfer /
evidence-engine path imports it; only ``training/tokenizer/pretrain.py`` constructs it, and the
saved checkpoint keeps it under ``heads`` purely so a warm --resume stays bit-consistent (the
inference loaders read ``ckpt["encoder"]`` alone).

Contract mirrors the filterbank's inputs so the two domain views see EXACTLY the same signal:
    patches       (B, P, S, C) native-rate zero-padded time samples
    patch_len     scalar | (B,) | (B, P)  — true valid length N over the S axis
    channel_mask  (B, C) bool — True = real channel (absent channels are zero-filled)
    patch_padding_mask (B, P) bool — True = real patch (only needed for the single-scale
                  collate whose patch_len is per-window, so padded patches are not flagged by N)

A depthwise-separable 1D-conv stack runs over the S time axis, INDEPENDENTLY per channel
(so absent channels can never leak into present ones), then the window embedding is formed by
masked pooling: over valid time, then over real channels, then over real patches. The padded
time steps and absent channels are zeroed BEFORE the conv, so garbage there cannot leak across
the valid/pad boundary — the embedding depends only on real signal.

The sample-based conv is intrinsically rate-BLIND (the same S samples at 50 vs 100 Hz are
DIFFERENT physical signals, which the filterbank correctly separates) and the patch-pool is
order-BLIND. Before pooling over patches, each patch's (layer-normed) content is therefore
FiLM-modulated by a physical embedding — [log physical duration, Fourier features of the
physical patch-center time]: ``pooled = ln(content)*(1+gamma(phys)) + beta(phys)``. Multiplicative
binding is non-separable under the mean pool, so the SAME samples at different rates get different
embeddings and permuting the patch order changes the output — matching the physical, position-aware
frequency view the TF-C loss pulls toward (F2/F3). When durations/positions are absent the
modulation is skipped (backward-compatible).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _DSConv1d(nn.Module):
    """Depthwise-separable 1D conv block: depthwise (temporal) + pointwise (feature mix)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, padding=pad, groups=in_ch)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.norm = nn.GroupNorm(1, out_ch)          # batch-size-independent (works at smoke B)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (N, in_ch, S)
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class TimeEncoder(nn.Module):
    """Compact time-domain encoder over raw patches -> one (B, d_model) window embedding.

    Small by design (~0.2M params) — it is an auxiliary rail, not a second backbone.
    """

    def __init__(self, d_model: int, n_channels: int = 6, hidden: int = 192,
                 kernel_size: int = 7, n_blocks: int = 4, n_pos_freq: int = 8):
        super().__init__()
        self.n_channels = n_channels
        blocks = [_DSConv1d(1, hidden, kernel_size)]
        blocks += [_DSConv1d(hidden, hidden, kernel_size) for _ in range(n_blocks - 1)]
        self.conv = nn.Sequential(*blocks)
        self.proj = nn.Linear(hidden, d_model)
        # Physical conditioning of the per-patch embedding (F2/F3): the sample-based conv is
        # rate-BLIND (the same S samples at 50 vs 100 Hz are DIFFERENT physical signals — the
        # filterbank sees them differently) and the patch-pool is ORDER-blind. A small embedding of
        # [log physical duration, Fourier features of the physical patch-center time] is added to
        # each patch before pooling, so the same samples at different rates get different embeddings
        # AND permuting patches changes the output. Geometric periods span the window (0.4 s .. 6 s).
        self.n_pos_freq = n_pos_freq
        periods = torch.logspace(torch.log10(torch.tensor(0.4)),
                                 torch.log10(torch.tensor(6.0)), n_pos_freq)
        self.register_buffer("pos_omega", 2.0 * torch.pi / periods)      # (n_pos_freq,)
        # FiLM-modulate each patch's (layer-normed) content by its physical duration/position before
        # pooling: pooled = ln(content) * (1 + gamma(phys)) + beta(phys). Multiplicative binding is
        # non-separable under the mean pool (an additive/GELU embedding lets content drown phys and
        # stays near-separable -> order-blind), so the SAME samples at different rates differ and the
        # patch ORDER matters.
        self.phys_ln = nn.LayerNorm(hidden)
        self.phys_film = nn.Linear(1 + 2 * n_pos_freq, 2 * hidden)

    @staticmethod
    def _valid_len(patch_len, B: int, P: int, device) -> torch.Tensor:
        """Normalize patch_len (scalar | (B,) | (B,P)) to a (B, P) long tensor of valid lengths."""
        if not torch.is_tensor(patch_len):
            patch_len = torch.as_tensor(patch_len)
        N = patch_len.to(device=device).long()
        if N.ndim == 0:
            N = N.view(1, 1).expand(B, P)
        elif N.ndim == 1:
            N = N.view(B, 1).expand(B, P)
        return N

    def forward(
        self,
        patches: torch.Tensor,                              # (B, P, S, C)
        patch_len,                                          # scalar | (B,) | (B, P)
        channel_mask: torch.Tensor,                         # (B, C) bool
        patch_padding_mask: Optional[torch.Tensor] = None,  # (B, P) bool
        patch_durations: Optional[torch.Tensor] = None,     # (B, P) physical seconds (F2)
        positions: Optional[torch.Tensor] = None,           # (B, P) physical patch-center s (F3)
    ) -> torch.Tensor:
        B, P, S, C = patches.shape
        device = patches.device
        N = self._valid_len(patch_len, B, P, device)                    # (B, P)
        idx = torch.arange(S, device=device).view(1, 1, S)
        time_valid = (idx < N.unsqueeze(-1))                            # (B, P, S) bool
        cmask = channel_mask.to(torch.bool)                             # (B, C)

        # Zero the padded time steps AND absent channels BEFORE the conv so nothing there can
        # leak across the valid/pad boundary (the conv has a receptive field); the embedding
        # then depends only on real signal.
        x = patches * time_valid.unsqueeze(-1).to(patches.dtype) \
            * cmask.view(B, 1, 1, C).to(patches.dtype)

        # Per-channel temporal conv: fold (B, P, C) into the batch, one time series per channel.
        x = x.permute(0, 1, 3, 2).reshape(B * P * C, 1, S)             # (B*P*C, 1, S)
        feat = self.conv(x)                                            # (B*P*C, H, S)
        H = feat.shape[1]

        # Masked mean over valid time.
        tv = time_valid.unsqueeze(2).expand(B, P, C, S).reshape(B * P * C, S).to(feat.dtype)
        denom_t = tv.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled_t = (feat * tv.unsqueeze(1)).sum(dim=2) / denom_t       # (B*P*C, H)
        pooled_t = pooled_t.view(B, P, C, H)

        # Masked pool over real channels.
        cw = cmask.view(B, 1, C, 1).to(feat.dtype)
        pooled_c = (pooled_t * cw).sum(dim=2) / cw.sum(dim=2).clamp(min=1.0)   # (B, P, H)

        # Physical conditioning (F2 rate-awareness + F3 order-awareness): add a per-patch embedding
        # of [log physical duration, Fourier(physical position)] so the SAME samples at different
        # rates get different embeddings and permuting patches changes the output. Backward-compatible
        # no-op when neither is supplied.
        if patch_durations is not None or positions is not None:
            dur = (patch_durations if patch_durations is not None
                   else patches.new_ones(B, P)).to(feat.dtype).clamp(min=1e-3)
            pos = (positions if positions is not None
                   else patches.new_zeros(B, P)).to(feat.dtype)
            ang = pos.unsqueeze(-1) * self.pos_omega.to(feat.dtype)           # (B, P, F)
            phys = torch.cat([dur.log().unsqueeze(-1),
                              torch.sin(ang), torch.cos(ang)], dim=-1)        # (B, P, 1+2F)
            # FiLM: modulate layer-normed content by the physical duration/position before pooling.
            gamma, beta = self.phys_film(phys).chunk(2, dim=-1)              # (B, P, H) each
            pooled_c = self.phys_ln(pooled_c) * (1.0 + gamma) + beta         # (B, P, H)

        # Masked pool over real patches. A patch is real if it has any valid time step (this
        # already excludes padding in the multi-resolution collate, where padded patches carry
        # N=0) AND is flagged real by patch_padding_mask (needed for the single-scale collate,
        # whose per-window patch_len marks every patch slot as full length).
        patch_valid = time_valid.any(dim=2)                            # (B, P) bool
        if patch_padding_mask is not None:
            patch_valid = patch_valid & patch_padding_mask.to(torch.bool)
        pw = patch_valid.unsqueeze(-1).to(feat.dtype)                  # (B, P, 1)
        pooled_p = (pooled_c * pw).sum(dim=1) / pw.sum(dim=1).clamp(min=1.0)   # (B, H)

        return self.proj(pooled_p)                                     # (B, d_model)
