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
                 kernel_size: int = 7, n_blocks: int = 4):
        super().__init__()
        self.n_channels = n_channels
        blocks = [_DSConv1d(1, hidden, kernel_size)]
        blocks += [_DSConv1d(hidden, hidden, kernel_size) for _ in range(n_blocks - 1)]
        self.conv = nn.Sequential(*blocks)
        self.proj = nn.Linear(hidden, d_model)

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
