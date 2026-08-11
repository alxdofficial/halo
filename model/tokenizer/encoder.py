"""The config-conditional set encoder (M3) — completes "the tokenizer, broadly".

Assembly (build plan M3; EVIDENCE_ENGINE.md §5.2.1):

    device-frame IMU patches ── PhysicalFilterbankTokenizer ──> sensor tokens (B,P,C,d)
                                                      │
    channel descriptions ──frozen LM──> text embeddings ──ChannelTextFusion──> identity
                                                      │
    JEPA token_mask ──> learned [MASK] token (BEFORE fusion, so the model knows WHICH
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
from .sensor_tokens import ConditioningProjection, DescriptorHead, SensorFold
from .transformer import DualBranchTransformer

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
        frontend: str = "fixed",                  # tokenizer front end: 'fixed'|'learnable'
        text_conditioning: str = "per_channel",  # 'per_channel' (legacy) | 'factored' (role+sensor)
        token_granularity: str = "channel",      # 'channel' (legacy) | 'sensor' (design of record)
        sensor_bias_dim: int = 9,                # width of the sensor_bias artifact
        gate_bias_init: float = -2.0,             # factored: negative => identity lightly injected @ init
        use_duration_embedding: bool = False,
        duration_min_seconds: float = 0.4,
        duration_max_seconds: float = 1.5,
        duration_gate_init: float = 0.1,
        rope_min_period: float = ROPE_MIN_PERIOD_S,
        **filterbank_kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_duration_embedding = bool(use_duration_embedding)
        self.duration_min_seconds = float(duration_min_seconds)
        self.duration_max_seconds = float(duration_max_seconds)
        if not 0 < self.duration_min_seconds < self.duration_max_seconds:
            raise ValueError("duration bounds must satisfy 0 < min < max")
        if text_conditioning not in ("per_channel", "factored"):
            raise ValueError("text_conditioning must be 'per_channel' or 'factored'")
        self.text_conditioning = text_conditioning
        if token_granularity not in ("channel", "sensor"):
            raise ValueError("token_granularity must be 'channel' or 'sensor'")
        self.token_granularity = token_granularity
        self.sensor_bias_dim = int(sensor_bias_dim)
        if frontend not in {"fixed", "learnable"}:
            raise ValueError("frontend must be 'fixed' or 'learnable'")
        # Attribute stays named `filterbank` for checkpoint compatibility.
        self.filterbank = PhysicalFilterbankTokenizer(
            learnable=frontend == "learnable", d_model=d_model, **filterbank_kwargs,
        )
        self.text_encoder = TokenTextEncoder(model_name=text_model)   # frozen, cached
        if token_granularity == "sensor":
            # DESIGN OF RECORD: a sensor is one modality triad. Folding xyz into one token makes the
            # role text ("x"/"y"/"z") redundant — axis identity becomes positional inside the token —
            # and the two conditioning artifacts get LEARNABLE projections over FROZEN inputs.
            self.sensor_fold = SensorFold(d_model=d_model, dropout=dropout)
            self.descriptor_proj = ConditioningProjection(384, d_model, dropout=dropout,
                                                          gate_bias_init=gate_bias_init)
            self.bias_proj = ConditioningProjection(self.sensor_bias_dim, d_model, dropout=dropout,
                                                    gate_bias_init=gate_bias_init)
            self.descriptor_head = DescriptorHead(d_model, text_dim=384, dropout=dropout)
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
        # Runtime-only acceleration hook. The trainer may install a compiled bound ``forward`` here;
        # keeping the actual module untouched preserves ordinary state_dict keys and eager eval loads.
        self._compiled_transformer_forward = None

    # ------------------------------------------------------------------ shareable stages
    # The forward splits into (tokenize · encode_texts · encode) so a training step that
    # needs masked and clean views computes the filterbank and the
    # text embeddings ONCE and only re-runs the cheap transformer tail.

    def tokenize(self, patches, sampling_rate_hz, patch_len_samples, channel_mask=None,
                 source_rate_hz=None) -> torch.Tensor:
        """Sensor tokens (B, P, C, d) — identical across masked/clean views. The filterbank emits
        per-channel tokens the encoder masks downstream, so channel_mask is unused here (accepted
        for a stable call signature)."""
        return self.filterbank(patches, sampling_rate_hz, patch_len_samples,
                               source_rate_hz=source_rate_hz)

    def analyze(self, patches, sampling_rate_hz, patch_len_samples, source_rate_hz=None):
        """Parameter-free (fixed arm) physical feature, shareable across encoder copies."""
        return self.filterbank.analyze(patches, sampling_rate_hz, patch_len_samples,
                                       source_rate_hz=source_rate_hz)

    def project_tokens(self, token_in: torch.Tensor) -> torch.Tensor:
        """Apply THIS encoder's learnable filterbank projection to a shared analysis."""
        return self.filterbank.project(token_in)

    def encode_texts(
        self, channel_texts: Sequence[Sequence[str]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B lists of C strings) -> token embeddings (B, C, S, 384) + mask (B, C, S).

        Dedupes to UNIQUE strings before the (cached) encoder + before the pad/stack —
        a batch has at most (#streams × 6) distinct descriptions, not B×C, so the
        per-step assembly cost is bounded by variety, not batch size.
        """
        embs_u, masks_u, gather = self.encode_texts_unique(channel_texts, device)
        B, C = gather.shape
        S = embs_u.shape[1]
        embs = embs_u.index_select(0, gather.reshape(-1)).reshape(B, C, S, -1)
        masks = masks_u.index_select(0, gather.reshape(-1)).reshape(B, C, S)
        return embs, masks

    def encode_texts_unique(
        self, channel_texts: Sequence[Sequence[str]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode unique strings and return ``(embeddings, masks, inverse_ids)``.

        ``inverse_ids`` has shape (B,C). Keeping text unique through the learnable pooler avoids
        repeating identical attention/MLP work for every batch row. With pooler dropout disabled (or
        in eval), values and gradients are exactly the dense computation; training shares one dropout
        realization among rows carrying the same description.
        """
        B, C = len(channel_texts), len(channel_texts[0])
        for texts in channel_texts:
            assert len(texts) == C, "all samples in a batch must have the same channel count"
        flat = [t for texts in channel_texts for t in texts]
        unique = list(dict.fromkeys(flat))
        embs_u, masks_u = self.text_encoder.encode(unique, device=device)   # (U, S, 384)
        idx = {t: i for i, t in enumerate(unique)}
        gather = torch.tensor([idx[t] for t in flat], device=device, dtype=torch.long).reshape(B, C)
        return embs_u, masks_u, gather

    def encode_texts_factored(self, role_texts, sensor_texts, device):
        """Encode the two factored text sources with the same frozen LM.

        role_texts:   B lists of C role strings   -> (role_embs (B,C,S,384), role_masks (B,C,S))
        sensor_texts: B ragged lists of sensor strings -> padded
                      (sensor_embs (B,N_max,S,384), sensor_masks (B,N_max,S)).
        """
        role_embs, role_masks = self.encode_texts(role_texts, device)
        if not sensor_texts or any(len(texts) == 0 for texts in sensor_texts):
            raise ValueError("factored text conditioning requires at least one sensor description "
                             "per sample")
        B, n_max = len(sensor_texts), max(map(len, sensor_texts))
        flat = [text for texts in sensor_texts for text in texts]
        unique = list(dict.fromkeys(flat))
        embs_u, masks_u = self.text_encoder.encode(unique, device=device)
        lookup = {text: i for i, text in enumerate(unique)}
        S, D = embs_u.shape[1], embs_u.shape[2]
        sensor_embs = embs_u.new_zeros(B, n_max, S, D)
        sensor_masks = torch.zeros(B, n_max, S, dtype=torch.bool, device=device)
        for b, texts in enumerate(sensor_texts):
            gather = torch.tensor([lookup[text] for text in texts], device=device)
            sensor_embs[b, :len(texts)] = embs_u.index_select(0, gather)
            sensor_masks[b, :len(texts)] = masks_u.index_select(0, gather)
        return role_embs, role_masks, sensor_embs, sensor_masks

    def encode_texts_factored_unique(self, role_texts, sensor_texts, device):
        """Unique-token equivalent of :meth:`encode_texts_factored` for the hot path.

        Returns unique role token rows + (B,C) inverse IDs and unique sensor token rows + a padded
        (B,N_max) inverse-ID table. ``-1`` marks a padded sensor slot that no channel may reference.
        """
        role_embs, role_masks, role_ids = self.encode_texts_unique(role_texts, device)
        if not sensor_texts or any(len(texts) == 0 for texts in sensor_texts):
            raise ValueError("factored text conditioning requires at least one sensor description "
                             "per sample")
        B, n_max = len(sensor_texts), max(map(len, sensor_texts))
        flat = [text for texts in sensor_texts for text in texts]
        unique = list(dict.fromkeys(flat))
        sensor_embs, sensor_masks = self.text_encoder.encode(unique, device=device)
        lookup = {text: i for i, text in enumerate(unique)}
        sensor_text_ids = torch.full((B, n_max), -1, dtype=torch.long, device=device)
        for b, texts in enumerate(sensor_texts):
            sensor_text_ids[b, :len(texts)] = torch.tensor(
                [lookup[text] for text in texts], dtype=torch.long, device=device
            )
        return (role_embs, role_masks, role_ids,
                sensor_embs, sensor_masks, sensor_text_ids)

    def encode(
        self,
        sensor_tokens: torch.Tensor,                 # (B, P, C, d) from tokenize()
        text_embs: torch.Tensor,                     # (B, C, S, 384) from encode_texts()
        text_masks: torch.Tensor,
        positions: torch.Tensor,                     # (B, P) patch-center times in SECONDS
        patch_durations: Optional[torch.Tensor] = None,  # (B, P) true temporal support in seconds
        resolution_ids: Optional[torch.Tensor] = None,   # (B, P) 0=short, 1=long, -1=padding
        token_mask: Optional[torch.Tensor] = None,   # (B, P, C) True = hide for JEPA
        channel_mask: Optional[torch.Tensor] = None, # (B, C) True = channel exists
        patch_padding_mask: Optional[torch.Tensor] = None,  # (B, P) True = real patch
        # --- factored text conditioning (only when text_conditioning='factored') ---
        sensor_text_embs: Optional[torch.Tensor] = None,   # (B, N_sensors, S, 384)
        sensor_text_masks: Optional[torch.Tensor] = None,  # (B, N_sensors, S)
        sensor_id: Optional[torch.Tensor] = None,          # (B, C) long
        role_text_ids: Optional[torch.Tensor] = None,      # (B,C), when text rows stay unique
        sensor_text_ids: Optional[torch.Tensor] = None,    # (B,N_sensors), -1 = padding
        # --- sensor granularity (token_granularity='sensor') ---
        sensor_bias: Optional[torch.Tensor] = None,        # (B,N_sensors,sensor_bias_dim) frozen
        descriptor_mask: Optional[torch.Tensor] = None,    # (B,N_sensors) True = hide the descriptor
    ) -> dict[str, torch.Tensor]:
        if self.token_granularity == "sensor":
            return self._encode_sensor(
                sensor_tokens, text_embs, text_masks, positions,
                patch_durations=patch_durations, resolution_ids=resolution_ids,
                token_mask=token_mask, channel_mask=channel_mask,
                patch_padding_mask=patch_padding_mask,
                sensor_text_embs=sensor_text_embs, sensor_text_masks=sensor_text_masks,
                sensor_id=sensor_id, sensor_text_ids=sensor_text_ids,
                sensor_bias=sensor_bias, descriptor_mask=descriptor_mask,
            )
        B, P, C, _ = sensor_tokens.shape

        # JEPA masking BEFORE fusion: the [MASK] token then receives its channel's text
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
            if sensor_text_embs is None or sensor_text_masks is None or sensor_id is None:
                raise ValueError("factored text_conditioning requires sensor_text_embs / "
                                 "sensor_text_masks / sensor_id; use encode_texts_factored()")
            sensor_id = sensor_id.to(device=tokens.device, dtype=torch.long)
            if sensor_id.shape != (B, C):
                raise ValueError(f"sensor_id must have shape {(B, C)}, got {tuple(sensor_id.shape)}")
            sensor_count = (sensor_text_ids.shape[1] if sensor_text_ids is not None
                            else sensor_text_embs.shape[1])
            if sensor_id.numel() and (
                int(sensor_id.min().item()) < 0
                or int(sensor_id.max().item()) >= sensor_count
            ):
                raise ValueError("sensor_id contains an index without a corresponding sensor description")
            valid_sensor = (sensor_text_ids.ge(0) if sensor_text_ids is not None
                            else sensor_text_masks.any(dim=2))
            selected_sensor_valid = torch.gather(valid_sensor, 1, sensor_id)
            if not bool(selected_sensor_valid.all()):
                raise ValueError(
                    "sensor_id points to a padded/missing sensor description for at least one sample"
                )
            # `text_embs`/`text_masks` carry the ROLE tokens in the factored path.
            tokens = self.fusion(tokens, text_embs, text_masks,
                                 sensor_text_embs, sensor_text_masks, sensor_id,
                                 role_text_ids=role_text_ids, sensor_text_ids=sensor_text_ids)
        else:
            tokens = self.fusion(tokens, text_embs, text_masks)

        transformer_forward = self._compiled_transformer_forward or self.transformer
        h = transformer_forward(
            tokens,
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
            # Within a resolution, weight each patch by its REPRESENTED duration so a partial tail
            # patch (a short window-crop remainder) contributes proportionally, not as a full-length
            # patch (F1). When every patch in a resolution has equal duration (e.g. eval's evenly
            # divided window) the constant factor cancels — this reduces to the uniform mean.
            dur_w = (patch_durations.to(per_patch.dtype) if patch_durations is not None
                     else patch_w.to(per_patch.dtype))
            scale_w = one_hot * (valid_r.to(per_patch.dtype) * dur_w).unsqueeze(-1)  # (B,P,2)
            denom = scale_w.sum(dim=1)                                    # (B,2)
            summaries = torch.einsum("bpd,bps->bsd", per_patch, scale_w) \
                / denom.clamp(min=1.0).unsqueeze(-1)
            active = (denom > 0).to(per_patch.dtype)
            pooled = (summaries * active.unsqueeze(-1)).sum(dim=1) \
                / active.sum(dim=1, keepdim=True).clamp(min=1.0)

        return {"tokens": h, "per_patch": per_patch, "pooled": pooled}

    # ------------------------------------------------------------------ sensor granularity
    def _encode_sensor(
        self,
        channel_tokens: torch.Tensor,                # (B,P,C,d) from tokenize()
        text_embs: torch.Tensor,                     # unused at sensor granularity (role text)
        text_masks: torch.Tensor,
        positions: torch.Tensor,
        patch_durations=None,
        resolution_ids=None,
        token_mask: Optional[torch.Tensor] = None,   # (B,P,S) True = hide this sensor-patch
        channel_mask: Optional[torch.Tensor] = None, # (B,C)
        patch_padding_mask: Optional[torch.Tensor] = None,
        sensor_text_embs: Optional[torch.Tensor] = None,   # (B,N,S_tok,384)
        sensor_text_masks: Optional[torch.Tensor] = None,  # (B,N,S_tok)
        sensor_id: Optional[torch.Tensor] = None,          # (B,C)
        sensor_text_ids: Optional[torch.Tensor] = None,
        sensor_bias: Optional[torch.Tensor] = None,        # (B,N,sensor_bias_dim)
        descriptor_mask: Optional[torch.Tensor] = None,    # (B,N) True = hide the descriptor
    ) -> dict[str, torch.Tensor]:
        """The design-of-record forward: fold to sensor tokens, condition, attend, pool.

        Order is load-bearing. The JEPA [MASK] is applied to the FOLDED sensor token but BEFORE the
        descriptor and bias are injected, so a masked sensor still carries its identity and the model
        knows *which* sensor it must reconstruct — the same principle as the per-channel path, one
        granularity up. Reversing it would ask the model to reconstruct an anonymous hole.
        """
        if sensor_id is None or sensor_text_embs is None or sensor_text_masks is None:
            raise ValueError("sensor granularity requires sensor_id, sensor_text_embs and "
                             "sensor_text_masks")
        B, P, C, d = channel_tokens.shape
        # `sensor_text_embs` may be UNIQUE rows (U,S_tok,384) with `sensor_text_ids` (B,N) indexing
        # them — the hot path dedupes text, so a batch holds at most a few distinct descriptions.
        # The per-sample sensor count therefore comes from the id table, never from the embedding.
        unique_text = sensor_text_ids is not None and sensor_text_embs.dim() == 3
        N = sensor_text_ids.shape[1] if sensor_text_ids is not None else sensor_text_embs.shape[1]
        sensor_id = sensor_id.to(device=channel_tokens.device, dtype=torch.long)

        tokens, folded_mask = self.sensor_fold(channel_tokens, sensor_id, channel_mask, n_sensors=N)
        # A sensor is present iff it has a live channel AND a real (non-padding) description.
        sensor_present = folded_mask
        if sensor_text_ids is not None:
            sensor_present = sensor_present & sensor_text_ids.ge(0).to(sensor_present.device)

        if token_mask is not None:
            if token_mask.shape != (B, P, N):
                raise ValueError(f"sensor-granularity token_mask must be {(B, P, N)}, "
                                 f"got {tuple(token_mask.shape)}")
            tokens = torch.where(token_mask.unsqueeze(-1),
                                 self.mask_token.expand_as(tokens), tokens)

        # The FROZEN descriptor artifact: mean-pooled SBERT over the sensor's valid text tokens.
        # Mean-pooling (not the learned pooler) is deliberate — this vector is also the TARGET of
        # descriptor-mask reconstruction, so it must not move as the model trains.
        tm = sensor_text_masks.to(sensor_text_embs.dtype).unsqueeze(-1)
        pooled_text = (sensor_text_embs * tm).sum(dim=-2) / tm.sum(dim=-2).clamp(min=1e-6)
        pooled_text = F.normalize(pooled_text, dim=-1)
        if unique_text:
            # (U,384) -> (B,N,384). Padding slots (-1) clamp to row 0 and are zeroed by
            # `sensor_present` downstream, so they contribute nothing.
            descriptor = pooled_text.index_select(0, sensor_text_ids.clamp_min(0).reshape(-1))
            descriptor = descriptor.reshape(B, N, -1)
        else:
            descriptor = pooled_text                                              # (B,N,384)

        descriptor_visible = sensor_present
        if descriptor_mask is not None:
            descriptor_visible = descriptor_visible & ~descriptor_mask.to(sensor_present.device)
        tokens = self.descriptor_proj(tokens, descriptor, descriptor_visible)
        if sensor_bias is not None:
            tokens = self.bias_proj(tokens, sensor_bias.to(tokens.dtype), sensor_present)

        transformer_forward = self._compiled_transformer_forward or self.transformer
        h = transformer_forward(tokens, channel_mask=sensor_present,
                                patch_padding_mask=patch_padding_mask, positions=positions)

        weights = h.new_ones(B, P, N) * sensor_present.view(B, 1, N).to(h.dtype)
        if patch_padding_mask is not None:
            weights = weights * patch_padding_mask.view(B, P, 1).to(h.dtype)
        denom = weights.sum(dim=2, keepdim=True).clamp(min=1.0)
        per_patch = (h * weights.unsqueeze(-1)).sum(dim=2) / denom.squeeze(2).unsqueeze(-1)
        patch_w = weights.amax(dim=2)
        if resolution_ids is None:
            pooled = (per_patch * patch_w.unsqueeze(-1)).sum(dim=1) \
                / patch_w.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            valid_r = (resolution_ids >= 0) & (resolution_ids < 2) & (patch_w > 0)
            one_hot = F.one_hot(resolution_ids.clamp(0, 1), num_classes=2).to(per_patch.dtype)
            one_hot = one_hot * valid_r.unsqueeze(-1).to(per_patch.dtype)
            counts = one_hot.sum(dim=1).clamp(min=1.0)                              # (B,2)
            sums = torch.einsum("bpd,bpr->brd", per_patch * patch_w.unsqueeze(-1), one_hot)
            means = sums / counts.unsqueeze(-1)
            present = (one_hot.sum(dim=1) > 0).to(per_patch.dtype)
            pooled = (means * present.unsqueeze(-1)).sum(dim=1) \
                / present.sum(dim=1, keepdim=True).clamp(min=1.0)

        # Per-sensor context, averaged over valid patches — the input the descriptor head predicts
        # from, and the per-sensor row the Phase-B bank stores.
        pw = (patch_padding_mask.to(h.dtype) if patch_padding_mask is not None
              else h.new_ones(B, P))
        sensor_context = (h * pw.view(B, P, 1, 1)).sum(dim=1) / pw.sum(dim=1).clamp(min=1.0).view(B, 1, 1)

        return {"tokens": h, "per_patch": per_patch, "pooled": pooled,
                "sensor_context": sensor_context, "sensor_present": sensor_present,
                "descriptor": descriptor,
                "descriptor_pred": self.descriptor_head(sensor_context)}

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
        token_mask: Optional[torch.Tensor] = None,   # (B, P, C) True = hide for JEPA
        channel_mask: Optional[torch.Tensor] = None, # (B, C) True = channel exists
        patch_padding_mask: Optional[torch.Tensor] = None,  # (B, P) True = real patch
        sensor_texts: Optional[Sequence[Sequence[str]]] = None,  # factored: B lists of N_sensor strings
        sensor_id: Optional[torch.Tensor] = None,                # factored: (B, C) long
        source_rate_hz=None,                         # scalar | (B,) acquisition bandwidth bound
        sensor_bias: Optional[torch.Tensor] = None,  # sensor granularity: (B, N, sensor_bias_dim)
        descriptor_mask: Optional[torch.Tensor] = None,  # sensor granularity: (B, N)
    ) -> dict[str, torch.Tensor]:
        sensor_tokens = self.tokenize(
            patches,
            sampling_rate_hz,
            patch_len_samples,
            source_rate_hz=source_rate_hz,
        )
        device = sensor_tokens.device
        s_embs = s_masks = None
        if self.token_granularity == "sensor":
            # Sensor granularity always uses the factored text sources, but only the SENSOR half:
            # role text is redundant once xyz is folded into one token (axis identity is positional).
            if sensor_texts is None or sensor_id is None:
                raise ValueError("sensor granularity requires sensor_texts and sensor_id")
            _, _, _, s_embs, s_masks, sensor_text_ids = \
                self.encode_texts_factored_unique(channel_texts, sensor_texts, device)
            return self._encode_sensor(
                sensor_tokens, None, None, positions,
                patch_durations=patch_durations, resolution_ids=resolution_ids,
                token_mask=token_mask, channel_mask=channel_mask,
                patch_padding_mask=patch_padding_mask,
                sensor_text_embs=s_embs, sensor_text_masks=s_masks,
                sensor_id=sensor_id, sensor_text_ids=sensor_text_ids,
                sensor_bias=sensor_bias, descriptor_mask=descriptor_mask,
            )
        if self.text_conditioning == "factored":
            if sensor_texts is None or sensor_id is None:
                raise ValueError("factored text_conditioning requires sensor_texts and sensor_id")
            text_embs, text_masks, role_ids, s_embs, s_masks, sensor_text_ids = \
                self.encode_texts_factored_unique(channel_texts, sensor_texts, device)
        else:
            text_embs, text_masks = self.encode_texts(channel_texts, device)
        return self.encode(sensor_tokens, text_embs, text_masks, positions,
                           patch_durations=patch_durations, resolution_ids=resolution_ids,
                           token_mask=token_mask, channel_mask=channel_mask,
                           patch_padding_mask=patch_padding_mask,
                           sensor_text_embs=s_embs, sensor_text_masks=s_masks, sensor_id=sensor_id,
                           role_text_ids=(role_ids if self.text_conditioning == "factored" else None),
                           sensor_text_ids=(sensor_text_ids
                                            if self.text_conditioning == "factored" else None))
