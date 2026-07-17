"""The config-conditional set encoder (M3) — completes "the tokenizer, broadly".

Assembly (build plan M3; EVIDENCE_ENGINE.md §5.2.1):

    gravity-aligned patches ── PhysicalFilterbankTokenizer ──> sensor tokens (B,P,C,d)
                                                      │
    channel descriptions ──frozen LM──> text embeddings ──ChannelTextFusion──> identity
                                                      │
    A1 token_mask ──> learned [MASK] token (BEFORE fusion, so the model knows WHICH
                      channel is hidden — masked-channel modeling needs the identity
                      of the thing it must reconstruct)
                                                      │
    DualBranchTransformer: temporal attention with PHYSICAL-TIME RoPE (seconds, never
    patch index) + cross-channel attention (channel-mask aware). Channels carry NO
    positional index — identity is text, so channel count/order are free.
                                                      ▼
    {tokens (B,P,C,d) · per_patch (B,P,d) · pooled (B,d)}

Config conditioning IS the channel text ("accelerometer x-axis at the wrist") — this
replaces the M2 gate's per-stream config token and its UNKNOWN fallback: an unseen
config arrives with its own text and generalizes through language space.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .channel_text import ChannelTextFusion, TokenTextEncoder
from .filterbank import PhysicalFilterbankTokenizer
from .transformer import DualBranchTransformer, build_temporal_mask

# RoPE periods in SECONDS: fastest = finest patch spacing we draw (0.5 s multi-scale
# floor, §5.2.1); slowest comfortably above any session span we train on.
ROPE_MIN_PERIOD_S = 0.5
ROPE_MAX_PERIOD_S = 600.0


class SetTokenizerEncoder(nn.Module):
    """signal patches + channel TEXT + physical time -> representation.

    Permutation- and count-invariant over channels by construction (identity via text,
    attention via masks); rate- and patch-duration-agnostic via the physical-Hz
    filterbank + physical-time RoPE.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        text_model: str = "all-MiniLM-L6-v2",
        temporal_mode: str = "full",           # 'full' | 'causal' (streaming/world-model)
        **filterbank_kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.temporal_mode = temporal_mode
        self.filterbank = PhysicalFilterbankTokenizer(d_model=d_model, **filterbank_kwargs)
        self.text_encoder = TokenTextEncoder(model_name=text_model)   # frozen, cached
        self.fusion = ChannelTextFusion(d_model=d_model, text_dim=384)
        # MAE-style small-random init (NOT zeros: a zero mask token is symmetric across
        # masked positions and starts with no signal to distinguish "hidden here").
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.transformer = DualBranchTransformer(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_rope=True,
            rope_min_period=ROPE_MIN_PERIOD_S,
            rope_max_period=ROPE_MAX_PERIOD_S,
        )

    # ------------------------------------------------------------------ shareable stages
    # The forward splits into (tokenize · encode_texts · encode) so a training step that
    # needs two views (masked for A1, clean for A2/A3) computes the filterbank and the
    # text embeddings ONCE and only re-runs the cheap transformer tail.

    def tokenize(self, patches, sampling_rate_hz, patch_len_samples) -> torch.Tensor:
        """Filterbank sensor tokens (B, P, C, d) — identical across masked/clean views."""
        return self.filterbank(patches, sampling_rate_hz, patch_len_samples)

    def encode_texts(
        self, channel_texts: Sequence[Sequence[str]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B lists of C strings) -> token embeddings (B, C, S, 384) + mask (B, C, S).

        Dedupes to UNIQUE strings before the (cached) encoder + before the pad/stack —
        a batch has at most (#streams × 6) distinct descriptions, not B×C, so the
        per-step assembly cost is bounded by variety, not batch size.
        """
        B, C = len(channel_texts), len(channel_texts[0])
        for texts in channel_texts:
            assert len(texts) == C, "all samples in a batch must have the same channel count"
        flat = [t for texts in channel_texts for t in texts]
        unique = list(dict.fromkeys(flat))
        embs_u, masks_u = self.text_encoder.encode(unique, device=device)   # (U, S, 384)
        idx = {t: i for i, t in enumerate(unique)}
        gather = torch.tensor([idx[t] for t in flat], device=device)
        S = embs_u.shape[1]
        embs = embs_u.index_select(0, gather).reshape(B, C, S, -1)
        masks = masks_u.index_select(0, gather).reshape(B, C, S)
        return embs, masks

    def encode(
        self,
        sensor_tokens: torch.Tensor,                 # (B, P, C, d) from tokenize()
        text_embs: torch.Tensor,                     # (B, C, S, 384) from encode_texts()
        text_masks: torch.Tensor,
        positions: torch.Tensor,                     # (B, P) patch-center times in SECONDS
        token_mask: Optional[torch.Tensor] = None,   # (B, P, C) True = hide (A1)
        channel_mask: Optional[torch.Tensor] = None, # (B, C) True = channel exists
        patch_padding_mask: Optional[torch.Tensor] = None,  # (B, P) True = real patch
    ) -> dict[str, torch.Tensor]:
        B, P, C, _ = sensor_tokens.shape

        # A1 masking BEFORE fusion: the [MASK] token then receives its channel's text
        # identity, so the encoder knows *which* channel it must reconstruct.
        tokens = sensor_tokens
        if token_mask is not None:
            tokens = torch.where(
                token_mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens
            )
        tokens = self.fusion(tokens, text_embs, text_masks)

        temporal_mask = build_temporal_mask(positions, mode=self.temporal_mode)
        h = self.transformer(
            tokens,
            temporal_mask=temporal_mask,
            channel_mask=channel_mask,
            patch_padding_mask=patch_padding_mask,
            positions=positions,
        )                                                                # (B,P,C,d)

        # Pooling respects the masks: absent channels / padded patches contribute nothing.
        weights = h.new_ones(B, P, C)
        if channel_mask is not None:
            weights = weights * channel_mask.view(B, 1, C)
        if patch_padding_mask is not None:
            weights = weights * patch_padding_mask.view(B, P, 1)
        denom_c = weights.sum(dim=2, keepdim=True).clamp(min=1.0)
        per_patch = (h * weights.unsqueeze(-1)).sum(dim=2) / denom_c.squeeze(2).unsqueeze(-1)
        patch_w = weights.amax(dim=2)                                    # (B,P) patch validity
        pooled = (per_patch * patch_w.unsqueeze(-1)).sum(dim=1) \
            / patch_w.sum(dim=1, keepdim=True).clamp(min=1.0)

        return {"tokens": h, "per_patch": per_patch, "pooled": pooled}

    # ------------------------------------------------------------------------ forward
    def forward(
        self,
        patches: torch.Tensor,                       # (B, P, S, C) zero-padded native-rate
        sampling_rate_hz,                            # scalar | (B,)
        patch_len_samples,                           # scalar | (B,) true N
        channel_texts: Sequence[Sequence[str]],      # B lists of C descriptions
        positions: torch.Tensor,                     # (B, P) patch-center times in SECONDS
        token_mask: Optional[torch.Tensor] = None,   # (B, P, C) True = hide (A1)
        channel_mask: Optional[torch.Tensor] = None, # (B, C) True = channel exists
        patch_padding_mask: Optional[torch.Tensor] = None,  # (B, P) True = real patch
    ) -> dict[str, torch.Tensor]:
        sensor_tokens = self.tokenize(patches, sampling_rate_hz, patch_len_samples)
        text_embs, text_masks = self.encode_texts(channel_texts, sensor_tokens.device)
        return self.encode(sensor_tokens, text_embs, text_masks, positions,
                           token_mask=token_mask, channel_mask=channel_mask,
                           patch_padding_mask=patch_padding_mask)
