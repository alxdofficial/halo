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

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .channel_text import ChannelTextFusion, FactoredChannelTextFusion, TokenTextEncoder
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
        frontend: str = "fixed",                  # tokenizer front end: 'fixed'|'learnable'|'mamba' (see scattering.build_frontend)
        text_conditioning: str = "per_channel",  # 'per_channel' (legacy) | 'factored' (role+sensor)
        gate_bias_init: float = -2.0,             # factored: negative => identity lightly injected @ init
        temporal_mode: str = "full",           # 'full' | 'causal' (streaming/world-model)
        use_duration_embedding: bool = False,
        duration_min_seconds: float = 0.4,
        duration_max_seconds: float = 1.5,
        duration_gate_init: float = 0.1,
        rope_min_period: float = ROPE_MIN_PERIOD_S,
        **filterbank_kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.temporal_mode = temporal_mode
        self.use_duration_embedding = bool(use_duration_embedding)
        self.duration_min_seconds = float(duration_min_seconds)
        self.duration_max_seconds = float(duration_max_seconds)
        if not 0 < self.duration_min_seconds < self.duration_max_seconds:
            raise ValueError("duration bounds must satisfy 0 < min < max")
        if text_conditioning not in ("per_channel", "factored"):
            raise ValueError("text_conditioning must be 'per_channel' or 'factored'")
        self.text_conditioning = text_conditioning
        # Back-compat: callers (training/tokenizer/pretrain.py, older checkpoints) select the arm
        # via a legacy `learnable=bool` kwarg. Translate it into `frontend` and DROP it, so
        # build_frontend -- which sets `learnable` itself for the fixed/learnable arms -- does not
        # receive it twice (that collision broke the default fixed path; regression 2026-07-22).
        legacy_learnable = filterbank_kwargs.pop("learnable", None)
        if legacy_learnable is not None and frontend == "fixed":
            frontend = "learnable" if legacy_learnable else "fixed"
        self.frontend_kind = frontend
        # The tokenizer front end is swappable (fixed filterbank | mamba | ...) but every option
        # honours the same (B,P,S,C)+rate+N -> (B,P,C,d) contract, so the encoder body is identical
        # across the ablation. Attribute stays named `filterbank` for back-compat with checkpoints
        # and the tokenize() shim below.
        from .scattering import build_frontend
        self.filterbank = build_frontend(frontend, d_model=d_model, **filterbank_kwargs)
        self.text_encoder = TokenTextEncoder(model_name=text_model)   # frozen, cached
        if text_conditioning == "factored":
            # per-channel ROLE text + per-sensor IDENTITY text (docs/design/TEXT_CONDITIONING.md)
            self.fusion = FactoredChannelTextFusion(d_model=d_model, text_dim=384,
                                                    gate_bias_init=gate_bias_init)
        else:
            self.fusion = ChannelTextFusion(d_model=d_model, text_dim=384)
        if self.use_duration_embedding:
            self.duration_proj = nn.Sequential(
                nn.Linear(1, 16), nn.GELU(), nn.Linear(16, d_model),
            )
            if not 0.0 < duration_gate_init < 1.0:
                raise ValueError("duration_gate_init must be in (0, 1)")
            gate_logit = torch.logit(torch.tensor(float(duration_gate_init)))
            self.duration_gate_logit = nn.Parameter(gate_logit)
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
            rope_min_period=rope_min_period,
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

    def encode_texts_factored(self, role_texts, sensor_texts, device):
        """Encode the two factored text sources with the same frozen LM.

        role_texts:   B lists of C role strings   -> (role_embs (B,C,S,384), role_masks (B,C,S))
        sensor_texts: B lists of N_sensor strings -> (sensor_embs (B,N,S,384), sensor_masks (B,N,S))
        """
        role_embs, role_masks = self.encode_texts(role_texts, device)
        sensor_embs, sensor_masks = self.encode_texts(sensor_texts, device)
        return role_embs, role_masks, sensor_embs, sensor_masks

    def encode(
        self,
        sensor_tokens: torch.Tensor,                 # (B, P, C, d) from tokenize()
        text_embs: torch.Tensor,                     # (B, C, S, 384) from encode_texts()
        text_masks: torch.Tensor,
        positions: torch.Tensor,                     # (B, P) patch-center times in SECONDS
        patch_durations: Optional[torch.Tensor] = None,  # (B, P) true temporal support in seconds
        resolution_ids: Optional[torch.Tensor] = None,   # (B, P) 0=short, 1=long, -1=padding
        cross_resolution_attention: bool = True,
        token_mask: Optional[torch.Tensor] = None,   # (B, P, C) True = hide (A1)
        channel_mask: Optional[torch.Tensor] = None, # (B, C) True = channel exists
        patch_padding_mask: Optional[torch.Tensor] = None,  # (B, P) True = real patch
        # --- factored text conditioning (only when text_conditioning='factored') ---
        sensor_text_embs: Optional[torch.Tensor] = None,   # (B, N_sensors, S, 384)
        sensor_text_masks: Optional[torch.Tensor] = None,  # (B, N_sensors, S)
        sensor_id: Optional[torch.Tensor] = None,          # (B, C) long
    ) -> dict[str, torch.Tensor]:
        B, P, C, _ = sensor_tokens.shape

        # A1 masking BEFORE fusion: the [MASK] token then receives its channel's text
        # identity, so the encoder knows *which* channel it must reconstruct.
        tokens = sensor_tokens
        if token_mask is not None:
            tokens = torch.where(
                token_mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens
            )
        if self.use_duration_embedding:
            if patch_durations is None:
                raise ValueError("patch_durations are required when duration embedding is enabled")
            lo = math.log(self.duration_min_seconds)
            span = math.log(self.duration_max_seconds) - lo
            valid_duration = patch_durations > 0
            log_d = patch_durations.clamp(min=self.duration_min_seconds).log()
            normalized = (2.0 * (log_d - lo) / span - 1.0).clamp(-1.0, 1.0)
            duration_emb = self.duration_proj(normalized.unsqueeze(-1))
            duration_emb = duration_emb * valid_duration.unsqueeze(-1)
            tokens = tokens + torch.sigmoid(self.duration_gate_logit) * duration_emb.unsqueeze(2)
        if self.text_conditioning == "factored":
            if sensor_text_embs is None or sensor_id is None:
                raise ValueError("factored text_conditioning requires sensor_text_embs / "
                                 "sensor_text_masks / sensor_id; use encode_texts_factored()")
            # `text_embs`/`text_masks` carry the ROLE tokens in the factored path.
            tokens = self.fusion(tokens, text_embs, text_masks,
                                 sensor_text_embs, sensor_text_masks, sensor_id)
        else:
            tokens = self.fusion(tokens, text_embs, text_masks)

        temporal_mask = build_temporal_mask(positions, mode=self.temporal_mode)
        if not cross_resolution_attention:
            if resolution_ids is None:
                raise ValueError("resolution_ids are required to isolate resolution attention")
            same_resolution = resolution_ids.unsqueeze(2).eq(resolution_ids.unsqueeze(1))
            same_resolution &= resolution_ids.unsqueeze(2).ge(0)
            temporal_mask = (same_resolution if temporal_mask is None
                             else temporal_mask & same_resolution)
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
        if resolution_ids is None:
            pooled = (per_patch * patch_w.unsqueeze(-1)).sum(dim=1) \
                / patch_w.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            # Equal resolution weight: twelve short tokens must not outweigh four long
            # tokens merely because their temporal grid is denser.
            valid_r = (resolution_ids >= 0) & (resolution_ids < 2) & (patch_w > 0)
            one_hot = F.one_hot(resolution_ids.clamp(0, 1), num_classes=2).to(per_patch.dtype)
            scale_w = one_hot * valid_r.unsqueeze(-1).to(per_patch.dtype)  # (B,P,2)
            denom = scale_w.sum(dim=1)                                    # (B,2)
            summaries = torch.einsum("bpd,bps->bsd", per_patch, scale_w) \
                / denom.clamp(min=1.0).unsqueeze(-1)
            active = (denom > 0).to(per_patch.dtype)
            pooled = (summaries * active.unsqueeze(-1)).sum(dim=1) \
                / active.sum(dim=1, keepdim=True).clamp(min=1.0)

        return {"tokens": h, "per_patch": per_patch, "pooled": pooled}

    # ------------------------------------------------------------------------ forward
    def forward(
        self,
        patches: torch.Tensor,                       # (B, P, S, C) zero-padded native-rate
        sampling_rate_hz,                            # scalar | (B,)
        patch_len_samples,                           # scalar | (B,) true N
        channel_texts: Sequence[Sequence[str]],      # per_channel: B lists of C descriptions;
                                                     # factored: B lists of C ROLE strings
        positions: torch.Tensor,                     # (B, P) patch-center times in SECONDS
        patch_durations: Optional[torch.Tensor] = None,
        resolution_ids: Optional[torch.Tensor] = None,
        cross_resolution_attention: bool = True,
        token_mask: Optional[torch.Tensor] = None,   # (B, P, C) True = hide (A1)
        channel_mask: Optional[torch.Tensor] = None, # (B, C) True = channel exists
        patch_padding_mask: Optional[torch.Tensor] = None,  # (B, P) True = real patch
        sensor_texts: Optional[Sequence[Sequence[str]]] = None,  # factored: B lists of N_sensor strings
        sensor_id: Optional[torch.Tensor] = None,                # factored: (B, C) long
    ) -> dict[str, torch.Tensor]:
        sensor_tokens = self.tokenize(patches, sampling_rate_hz, patch_len_samples)
        device = sensor_tokens.device
        s_embs = s_masks = None
        if self.text_conditioning == "factored":
            if sensor_texts is None or sensor_id is None:
                raise ValueError("factored text_conditioning requires sensor_texts and sensor_id")
            text_embs, text_masks, s_embs, s_masks = self.encode_texts_factored(
                channel_texts, sensor_texts, device)
        else:
            text_embs, text_masks = self.encode_texts(channel_texts, device)
        return self.encode(sensor_tokens, text_embs, text_masks, positions,
                           patch_durations=patch_durations, resolution_ids=resolution_ids,
                           cross_resolution_attention=cross_resolution_attention,
                           token_mask=token_mask, channel_mask=channel_mask,
                           patch_padding_mask=patch_padding_mask,
                           sensor_text_embs=s_embs, sensor_text_masks=s_masks, sensor_id=sensor_id)
