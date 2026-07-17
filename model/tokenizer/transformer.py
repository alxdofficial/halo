"""Dual-branch transformer (M3 port from legacy_code/model/transformer.py).

Temporal attention (per channel, physical-time RoPE over SECONDS not indices) +
cross-channel attention (per patch, channel-mask aware -> variable channel counts by
construction). `build_temporal_mask` provides full / causal / windowed modes — the
causal mode is the streaming / world-model (§5.1) path.

NOTE: this is FACTORIZED (dual-branch) attention, not the flat T×C attention sketched
in EVIDENCE_ENGINE.md §5.2.1 — ported as the battle-tested design; flat attention is
an ablation if ever needed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor):
    """positions (BC, P) in seconds, inv_freq (hd/2,) rad/s -> cos, sin (BC, 1, P, hd)."""
    ang = positions.to(inv_freq.dtype).unsqueeze(-1) * inv_freq        # (BC, P, hd/2)
    emb = torch.cat((ang, ang), dim=-1)                                # (BC, P, hd)
    return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)              # (BC, 1, P, hd)


def build_temporal_mask(positions, mode='full', window_sec=None,
                        lookahead_patches=0, attention_sink=False):
    """Temporal attention mask from per-patch physical times.

    positions: (B, P) seconds. Returns (B, P, P) bool (True = attend) or None for 'full'.
      full   -> None (unmasked bidirectional; offline/session mode)
      causal -> query i attends key j where j <= i + lookahead_patches (streaming, all past)
      window -> causal AND past key within `window_sec` seconds (bounded-memory streaming)
    attention_sink: also always attend to patch 0 (persistent anchor, StreamingLLM).
    The diagonal (self) is always allowed.
    """
    if mode == 'full':
        return None
    B, P = positions.shape
    dev = positions.device
    i = torch.arange(P, device=dev).view(1, P, 1)      # query index
    j = torch.arange(P, device=dev).view(1, 1, P)      # key index
    allowed = (j <= i + lookahead_patches)             # causal + K lookahead
    if mode == 'window' and window_sec is not None:
        ti = positions.unsqueeze(2)                    # (B,P,1)
        tj = positions.unsqueeze(1)                    # (B,1,P)
        past_ok = (ti - tj) <= window_sec              # past key within W seconds
        future = (j > i)                               # lookahead keys always allowed
        allowed = allowed & (past_ok | future)
    allowed = allowed.expand(B, P, P).clone()
    if attention_sink:
        allowed[:, :, 0] = True
    allowed |= torch.eye(P, dtype=torch.bool, device=dev).view(1, P, P)
    return allowed


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention over the temporal (patch) dimension.

    Processes each channel independently, allowing the model to learn
    temporal dependencies within each sensor channel.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_rope: bool = False,
        rope_min_period: float = 1.0,
        rope_max_period: float = 1000.0,
    ):
        """
        Args:
            d_model: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_rope: Apply rotary position embedding indexed by physical time (seconds)
            rope_min_period: Period (s) of the fastest rotary component (finest patch spacing)
            rope_max_period: Period (s) of the slowest component (> max session span)
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5

        # RoPE frequency band calibrated to HAR time-scales (seconds), not integer token
        # indices. Geometric periods: index 0 = fastest (min_period), last = slowest.
        self.use_rope = use_rope
        if use_rope:
            half = self.head_dim // 2
            k = torch.arange(half, dtype=torch.float32)
            periods = rope_min_period * (rope_max_period / rope_min_period) ** (k / max(half - 1, 1))
            self.register_buffer("rope_inv_freq", 2.0 * math.pi / periods, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply temporal self-attention.

        Args:
            x: Input tensor of shape (batch_size * num_channels, num_patches, d_model)
            mask: Optional attention mask of shape (num_patches, num_patches)
            key_padding_mask: Optional patch validity mask of shape (batch_channels, num_patches)
                             True = valid patch, False = padded patch

        Returns:
            Output tensor of same shape as input
        """
        batch_channels, num_patches, d_model = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch_channels, num_patches, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        # (batch_channels, num_patches, num_heads, head_dim)
        Q = Q.view(batch_channels, num_patches, self.num_heads, self.head_dim)
        K = K.view(batch_channels, num_patches, self.num_heads, self.head_dim)
        V = V.view(batch_channels, num_patches, self.num_heads, self.head_dim)

        # Transpose to (batch_channels, num_heads, num_patches, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Rotary position embedding over physical time, applied to Q,K before attention.
        if self.use_rope and positions is not None:
            cos, sin = _rope_cos_sin(positions, self.rope_inv_freq)     # (BC,1,P,hd)
            cos, sin = cos.to(Q.dtype), sin.to(Q.dtype)
            Q = Q * cos + _rotate_half(Q) * sin
            K = K * cos + _rotate_half(K) * sin

        # Build attention mask for SDPA (True = attend, False = masked). `mask` may be a
        # shared (P,P) mask or a per-sample (batch_channels, P, P) causal/windowed mask.
        attn_mask = None
        if mask is not None:
            attn_mask = mask.bool()
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)                     # (BC,1,P,P)
        if key_padding_mask is not None:
            # Column mask: prevent attention TO padded patches -> (BC,1,1,P)
            key_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = key_mask if attn_mask is None else (attn_mask & key_mask)

        # Use PyTorch's fused attention (dispatches to Flash Attention / memory-efficient backend)
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )

        # Reshape back and concatenate heads
        # (batch_channels, num_patches, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(batch_channels, num_patches, d_model)

        # Final projection
        output = self.out_proj(attn_output)

        return output


class CrossChannelSelfAttention(nn.Module):
    """
    Multi-head self-attention over the channel dimension.

    Allows different sensor channels to communicate and share information
    within each patch at the same temporal position.

    This enables the model to learn cross-channel dependencies and interactions
    (e.g., correlation between accelerometer and gyroscope).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-channel self-attention.

        Args:
            x: Input tensor of shape (batch_size * num_patches, num_channels, d_model)
            channel_mask: Optional channel validity mask of shape (batch_size * num_patches, num_channels)
                         True = valid channel, False = padded channel

        Returns:
            Output tensor of same shape as input
        """
        batch_patches, num_channels, d_model = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch_patches, num_channels, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        # (batch_patches, num_channels, num_heads, head_dim)
        Q = Q.view(batch_patches, num_channels, self.num_heads, self.head_dim)
        K = K.view(batch_patches, num_channels, self.num_heads, self.head_dim)
        V = V.view(batch_patches, num_channels, self.num_heads, self.head_dim)

        # Transpose to (batch_patches, num_heads, num_channels, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Build attention mask for SDPA (True = attend, False = masked)
        attn_mask = None
        if channel_mask is not None:
            # Column mask: prevent attention TO padded channels
            # (batch_patches, num_channels) -> (batch_patches, 1, 1, num_channels)
            attn_mask = channel_mask.unsqueeze(1).unsqueeze(2)

        # Use PyTorch's fused attention (dispatches to Flash Attention / memory-efficient backend)
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )

        # Reshape back and concatenate heads
        # (batch_patches, num_channels, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(batch_patches, num_channels, d_model)

        # Final projection
        output = self.out_proj(attn_output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Two-layer MLP with GELU activation.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Input/output dimension
            dim_feedforward: Hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (*, d_model)

        Returns:
            Output tensor of same shape
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TemporalTransformerBlock(nn.Module):
    """
    Single transformer block with temporal self-attention.

    Architecture:
    - Temporal self-attention with residual connection and layer norm
    - Feed-forward network with residual connection and layer norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Feature dimension
            num_heads: Number of attention heads
            dim_feedforward: Hidden dimension for feed-forward network
            dropout: Dropout probability
        """
        super().__init__()

        self.self_attn = TemporalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        self.feed_forward = FeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch_size * num_channels, num_patches, d_model)
            mask: Optional attention mask
            key_padding_mask: Optional patch validity mask (batch_channels, num_patches)
                             True = valid, False = padded

        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual
        attn_output = self.self_attn(x, mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class DualBranchTransformerBlock(nn.Module):
    """
    Dual-branch transformer block with both temporal and cross-channel attention.

    Architecture:
    1. Temporal self-attention (patches within each channel)
    2. Cross-channel self-attention (channels within each patch)
    3. Feed-forward network

    Each step includes residual connections and layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_rope: bool = False,
        rope_min_period: float = 1.0,
        rope_max_period: float = 1000.0,
    ):
        """
        Args:
            d_model: Feature dimension
            num_heads: Number of attention heads
            dim_feedforward: Hidden dimension for feed-forward network
            dropout: Dropout probability
            use_rope / rope_min_period / rope_max_period: physical-time RoPE for temporal attn
        """
        super().__init__()

        # Temporal attention (over patches within channel)
        self.temporal_attn = TemporalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            rope_min_period=rope_min_period,
            rope_max_period=rope_max_period,
        )

        # Cross-channel attention (over channels within patch)
        self.cross_channel_attn = CrossChannelSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # Feed-forward network
        self.feed_forward = FeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        patch_padding_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through dual-branch transformer block.

        Args:
            x: Input tensor of shape (batch_size, num_patches, num_channels, d_model)
            temporal_mask: Optional mask for temporal attention (num_patches, num_patches)
            channel_mask: Optional mask for channel attention (batch_size, num_channels)
                         True = valid channel, False = padded
            patch_padding_mask: Optional patch validity mask (batch_size, num_patches)
                               True = valid patch, False = padded

        Returns:
            Output tensor of shape (batch_size, num_patches, num_channels, d_model)
        """
        batch_size, num_patches, num_channels, d_model = x.shape

        # 1. Temporal self-attention (patches within each channel)
        # Reshape: (batch, patches, channels, d_model) -> (batch*channels, patches, d_model)
        x_temporal = x.permute(0, 2, 1, 3)  # (batch, channels, patches, d_model)
        x_temporal = x_temporal.reshape(batch_size * num_channels, num_patches, d_model)

        # Expand patch padding mask to (B*C, P) for temporal attention
        if patch_padding_mask is not None:
            temporal_key_padding_mask = patch_padding_mask.unsqueeze(1).expand(
                batch_size, num_channels, num_patches
            ).reshape(batch_size * num_channels, num_patches)
        else:
            temporal_key_padding_mask = None

        # Expand per-sample positions (B,P) -> (B*C,P) and a per-sample temporal mask
        # (B,P,P) -> (B*C,P,P); a shared (P,P) mask passes through unchanged.
        temporal_positions = None
        if positions is not None:
            temporal_positions = positions.unsqueeze(1).expand(
                batch_size, num_channels, num_patches
            ).reshape(batch_size * num_channels, num_patches)
        tmask = temporal_mask
        if tmask is not None and tmask.dim() == 3:
            tmask = tmask.unsqueeze(1).expand(
                batch_size, num_channels, num_patches, num_patches
            ).reshape(batch_size * num_channels, num_patches, num_patches)

        # Apply temporal attention
        temporal_output = self.temporal_attn(x_temporal, tmask,
                                             key_padding_mask=temporal_key_padding_mask,
                                             positions=temporal_positions)

        # Reshape back: (batch*channels, patches, d_model) -> (batch, patches, channels, d_model)
        temporal_output = temporal_output.reshape(batch_size, num_channels, num_patches, d_model)
        temporal_output = temporal_output.permute(0, 2, 1, 3)

        # Residual and norm
        x = x + self.dropout(temporal_output)
        x = self.norm1(x)

        # 2. Cross-channel self-attention (channels within each patch)
        # Reshape: (batch, patches, channels, d_model) -> (batch*patches, channels, d_model)
        x_channel = x.reshape(batch_size * num_patches, num_channels, d_model)

        # Expand channel mask if provided
        # (batch, channels) -> (batch*patches, channels)
        if channel_mask is not None:
            channel_mask_expanded = channel_mask.unsqueeze(1).expand(
                batch_size, num_patches, num_channels
            ).reshape(batch_size * num_patches, num_channels)
        else:
            channel_mask_expanded = None

        # Apply cross-channel attention
        channel_output = self.cross_channel_attn(x_channel, channel_mask_expanded)

        # Reshape back: (batch*patches, channels, d_model) -> (batch, patches, channels, d_model)
        channel_output = channel_output.reshape(batch_size, num_patches, num_channels, d_model)

        # Residual and norm
        x = x + self.dropout(channel_output)
        x = self.norm2(x)

        # 3. Feed-forward network
        ff_output = self.feed_forward(x)

        # Residual and norm
        x = x + self.dropout(ff_output)
        x = self.norm3(x)

        return x


class ChannelIndependentTemporalTransformer(nn.Module):
    """
    Temporal transformer that processes each channel independently.

    Each channel's patch sequence is processed through the same transformer,
    learning temporal dependencies without mixing information across channels.

    Input:  (batch, patches, channels, d_model)
    Process: Each channel independently through temporal attention
    Output: (batch, patches, channels, d_model)
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Feature dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dim_feedforward: Hidden dimension for feed-forward networks
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.gradient_checkpointing = False

        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TemporalTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patch_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process input through channel-independent temporal transformer.

        Args:
            x: Input tensor of shape (batch_size, num_patches, num_channels, d_model)
            mask: Optional attention mask of shape (num_patches, num_patches)
            patch_padding_mask: Optional patch validity mask (batch_size, num_patches)
                               True = valid patch, False = padded

        Returns:
            Output tensor of shape (batch_size, num_patches, num_channels, d_model)

        Processing:
            1. Reshape to (batch * channels, patches, d_model)
            2. Apply transformer layers
            3. Reshape back to (batch, patches, channels, d_model)
        """
        batch_size, num_patches, num_channels, d_model = x.shape

        # Reshape to process each channel independently
        # (batch, patches, channels, d_model) -> (batch * channels, patches, d_model)
        x = x.permute(0, 2, 1, 3)  # (batch, channels, patches, d_model)
        x = x.reshape(batch_size * num_channels, num_patches, d_model)

        # Expand patch padding mask to (B*C, P)
        if patch_padding_mask is not None:
            key_padding_mask = patch_padding_mask.unsqueeze(1).expand(
                batch_size, num_channels, num_patches
            ).reshape(batch_size * num_channels, num_patches)
        else:
            key_padding_mask = None

        # Apply transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, mask, key_padding_mask,
                    use_reentrant=False
                )
            else:
                x = layer(x, mask, key_padding_mask=key_padding_mask)

        # Reshape back to original format
        # (batch * channels, patches, d_model) -> (batch, patches, channels, d_model)
        x = x.reshape(batch_size, num_channels, num_patches, d_model)
        x = x.permute(0, 2, 1, 3)  # (batch, patches, channels, d_model)

        return x


class DualBranchTransformer(nn.Module):
    """
    Dual-branch transformer with both temporal and cross-channel attention.

    Processes patches with two types of attention:
    1. Temporal attention: Models dependencies across patches within each channel
    2. Cross-channel attention: Models interactions between channels within each patch

    Input:  (batch, patches, channels, d_model)
    Process: Temporal attention → Cross-channel attention → FFN (per block)
    Output: (batch, patches, channels, d_model)
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_rope: bool = False,
        rope_min_period: float = 1.0,
        rope_max_period: float = 1000.0,
    ):
        """
        Args:
            d_model: Feature dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dim_feedforward: Hidden dimension for feed-forward networks
            dropout: Dropout probability
            use_rope / rope_min_period / rope_max_period: physical-time RoPE for temporal attn
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Stack of dual-branch transformer blocks
        self.layers = nn.ModuleList([
            DualBranchTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_rope=use_rope,
                rope_min_period=rope_min_period,
                rope_max_period=rope_max_period,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        patch_padding_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process input through dual-branch transformer.

        Args:
            x: Input tensor of shape (batch_size, num_patches, num_channels, d_model)
            temporal_mask: Optional (num_patches, num_patches) or per-sample (B,P,P) mask
            channel_mask: Optional mask for channel attention (batch_size, num_channels)
            patch_padding_mask: Optional patch validity mask (batch_size, num_patches)
            positions: Optional per-patch physical times (batch_size, num_patches) for RoPE

        Returns:
            Output tensor of shape (batch_size, num_patches, num_channels, d_model)
        """
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, temporal_mask, channel_mask, patch_padding_mask, positions=positions)

        return x


class IMUTransformer(nn.Module):
    """
    Main transformer module for IMU encoder.

    Supports two modes:
    1. Temporal-only: Channel-independent temporal attention (backward compatible)
    2. Dual-branch: Temporal + cross-channel attention (enables channel interactions)
    """

    def __init__(
        self,
        d_model: int = 128,
        num_temporal_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_cross_channel: bool = False,
        use_rope: bool = False,
        rope_min_period: float = 1.0,
        rope_max_period: float = 1000.0,
    ):
        """
        Args:
            d_model: Feature dimension
            num_temporal_layers: Number of temporal transformer layers
            num_heads: Number of attention heads
            dim_feedforward: Hidden dimension for feed-forward networks
            dropout: Dropout probability
            use_cross_channel: Whether to use cross-channel attention (default: False)
            use_rope / rope_min_period / rope_max_period: physical-time RoPE (dual-branch only)
        """
        super().__init__()

        self.use_cross_channel = use_cross_channel
        self.use_rope = use_rope

        if use_cross_channel:
            # Use dual-branch transformer with temporal + cross-channel attention
            self.transformer = DualBranchTransformer(
                d_model=d_model,
                num_layers=num_temporal_layers,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_rope=use_rope,
                rope_min_period=rope_min_period,
                rope_max_period=rope_max_period,
            )
        else:
            # Use temporal-only transformer (backward compatible)
            self.transformer = ChannelIndependentTemporalTransformer(
                d_model=d_model,
                num_layers=num_temporal_layers,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        patch_padding_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process input through transformer.

        Args:
            x: Input tensor of shape (batch_size, num_patches, num_channels, d_model)
            temporal_mask: Optional (P,P) or per-sample (B,P,P) mask for temporal attention
            channel_mask: Optional mask for channel attention (only used if use_cross_channel=True)
            patch_padding_mask: Optional patch validity mask (batch_size, num_patches)
            positions: Optional per-patch physical times (batch_size, num_patches) for RoPE

        Returns:
            Output tensor of shape (batch_size, num_patches, num_channels, d_model)
        """
        if self.use_cross_channel:
            return self.transformer(x, temporal_mask, channel_mask, patch_padding_mask,
                                    positions=positions)
        else:
            # Temporal-only path does not support RoPE/positions (backward compatible).
            return self.transformer(x, temporal_mask, patch_padding_mask)
