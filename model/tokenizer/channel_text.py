"""Channel-text encoding + fusion (M3 port from legacy_code/model/token_text_encoder.py).

The channel axis is a TEXT-KEYED SET: each channel's identity comes from the frozen-LM
embedding of its free-form description ("accelerometer x-axis at the wrist"), fused into
the sensor tokens by ChannelTextFusion (pool text per channel -> gated broadcast). This
replaces every positional/index notion of channel identity — and the M2 gate's per-stream
config token: config conditioning IS the channel text, which generalizes to unseen
configs through language (no UNKNOWN-token fallback needed).

LearnableLabelBank was deliberately NOT ported — label-text encoding is a Pipeline B
concern (the evidence head's `t`), not a tokenizer concern.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Dict, Tuple


class TokenTextEncoder(nn.Module):
    """
    Frozen text encoder outputting token-level embeddings (not pooled).

    Supports configurable SentenceBERT backend:
    - all-MiniLM-L6-v2: 384-dim, 22M params (default)
    - all-mpnet-base-v2: 768-dim, 109M params (scaled)
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', max_length: int = 64):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.hidden_dim = None  # Set on lazy init

        # Lazy initialization
        self._model = None
        self._tokenizer = None

        # Cache for repeated strings (frozen embeddings)
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Sensor-granularity Phase A consumes only the frozen mean-pooled descriptor.  Keeping that
        # artifact separately avoids rebuilding ragged token batches and pooling the same text in
        # every clean/masked/teacher forward.
        self._pooled_cache: Dict[str, torch.Tensor] = {}

    def _init_model(self):
        """Lazy load the transformer model."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        # Load on CPU explicitly: SentenceTransformer auto-selects CUDA, silently
        # grabbing VRAM another job may own. Callers move the (frozen, cached)
        # OUTPUTS to their device; move the model itself only via an explicit .to.
        sbert = SentenceTransformer(self.model_name, device="cpu")
        self.hidden_dim = sbert.get_sentence_embedding_dimension()
        transformer = sbert[0]  # Get underlying transformer

        # Store without registering as submodule (keeps out of state_dict)
        object.__setattr__(self, '_model', transformer.auto_model)
        object.__setattr__(self, '_tokenizer', transformer.tokenizer)

        # Freeze
        for p in self._model.parameters():
            p.requires_grad = False

        print(f"Loaded {self.model_name} ({self.hidden_dim}-dim, frozen)")

    def encode(
        self,
        texts: List[str],
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get token-level embeddings with per-label caching.

        Only runs the model forward pass for texts not already in cache,
        then assembles the full batch from cache. This means after warmup,
        repeated labels (which are common across batches) are free.

        Args:
            texts: List of strings
            device: Target device

        Returns:
            token_embeddings: (batch, seq_len, 384)
            attention_mask: (batch, seq_len) bool - True for valid tokens
        """
        self._init_model()

        if device is None:
            device = next(self._model.parameters()).device

        # Find uncached texts and encode only those
        uncached_texts = [t for t in texts if t not in self._cache]
        if uncached_texts:
            # Deduplicate (same text may appear multiple times in batch)
            unique_uncached = list(dict.fromkeys(uncached_texts))

            encoded = self._tokenizer(
                unique_uncached,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Run the frozen LM on ITS device (CPU unless explicitly moved), then cache
            # on the CALLER's device so repeated lookups skip the transfer.
            model_device = next(self._model.parameters()).device
            input_ids = encoded['input_ids'].to(model_device)
            attention_mask = encoded['attention_mask'].to(model_device)

            with torch.no_grad():
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                token_embeddings = outputs.last_hidden_state

            for i, text in enumerate(unique_uncached):
                self._cache[text] = (token_embeddings[i].detach().to(device),
                                     attention_mask[i].detach().to(device))

        # Assemble full batch from cache (moving if a cached entry is on another device).
        #
        # Caching the assembled (padded batch, mask) by text tuple was tried and MEASURED: it saved
        # nothing (61.2 vs 61.8 ms/step, inside noise), because the language-model forward above is
        # already cached per string and the assembly is a small share of the step's ~870 kernel
        # launches. It also would have handed out frozen tensors by reference. Not worth the
        # aliasing risk for no gain.
        cached_embs = [self._cache[t][0].to(device) for t in texts]
        cached_masks = [self._cache[t][1].to(device) for t in texts]

        embs = pad_sequence(cached_embs, batch_first=True, padding_value=0.0)
        masks = pad_sequence(cached_masks, batch_first=True, padding_value=0).bool()
        return embs, masks

    def encode_pooled(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return frozen, L2-normalized mean-pooled descriptors for ``texts``.

        Token-level outputs remain available through :meth:`encode` for the channel-granularity
        conditioner.  Sensor granularity has no learnable text pooler, so caching its final frozen
        artifact is exact and removes ragged padding from the training hot path.
        """
        self._init_model()
        if device is None:
            device = next(self._model.parameters()).device
        missing = list(dict.fromkeys(text for text in texts if text not in self._pooled_cache))
        if missing:
            token_embeddings, attention_mask = self.encode(missing, device=device)
            weights = attention_mask.to(token_embeddings.dtype).unsqueeze(-1)
            pooled = (token_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            for text, descriptor in zip(missing, pooled):
                self._pooled_cache[text] = descriptor.detach()
        return torch.stack([self._pooled_cache[text].to(device) for text in texts], dim=0)

class _TextPooler(nn.Module):
    """Frozen-LM token embeddings (N, S, text_dim) + mask (N, S) -> one vector per item (N, d).

    The shared "text -> vector -> learnable projection" primitive (docs/design/TEXT_CONDITIONING.md):
    learned queries cross-attend the text tokens, the pooled queries are projected to one d_model
    vector. Extracted so the factored conditioner can pool ROLE text and SENSOR text with the same
    machinery ChannelTextFusion uses for per-channel text.
    """

    def __init__(self, d_model: int, text_dim: int, num_heads: int = 4,
                 num_queries: int = 4, dropout: float = 0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, d_model) if text_dim != d_model else nn.Identity()
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model * num_queries, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        # text_tokens (N, S, text_dim), text_mask (N, S) bool True=valid -> (N, d_model)
        N, S, _ = text_tokens.shape
        t = self.text_proj(text_tokens)
        m = text_mask.bool()
        # An all-masked row (a padding sensor/channel) would NaN the softmax; unmask position 0
        # as a dummy so it produces a finite (meaningless) vector the caller can ignore via masks.
        all_masked = ~m.any(dim=1)
        if all_masked.any():
            m = m.clone()
            m[all_masked, 0] = True
        q = self.queries.unsqueeze(0).expand(N, -1, -1)
        attn, _ = self.cross_attn(query=q, key=t, value=t, key_padding_mask=~m, need_weights=False)
        attn = self.norm(q + attn)
        return self.out_proj(attn.reshape(N, -1))


class FactoredChannelTextFusion(nn.Module):
    """Factored identity conditioning — per-channel ROLE text + per-sensor IDENTITY text.

    The clean factorization of docs/design/TEXT_CONDITIONING.md, replacing the per-channel
    placement-repeating text of ChannelTextFusion:

      * role text  ("accelerometer x-axis")        -> per-CHANNEL embedding  (axis/modality only)
      * sensor text ("a smartwatch on the left wrist") -> per-SENSOR embedding, broadcast to that
                                                          sensor's channels via ``sensor_id``
      * identity = role + sensor, injected as a GATED residual over the sensor tokens, broadcast
        across patches. The gate bias inits negative, so identity starts lightly injected and the
        model learns how much to use — making the whole conditioner do-no-harm at init and a clean
        on/off ablation (set the gate bias very negative to turn it off).

    Both text sources are pooled by the SAME ``_TextPooler`` (both are "text -> vector"). No
    placement in the role text and no axis in the sensor text -> each config fact is injected once,
    never compounded (docs/design/TEXT_CONDITIONING.md §6.1).
    """

    def __init__(self, d_model: int, text_dim: int = 384, num_heads: int = 4,
                 num_queries: int = 4, dropout: float = 0.1, gate_bias_init: float = -2.0):
        super().__init__()
        self.pool = _TextPooler(d_model, text_dim, num_heads=num_heads,
                                num_queries=num_queries, dropout=dropout)
        # Gate is a function of BOTH the sensor token and the identity (as in ChannelTextFusion),
        # split into two linears to avoid materializing a concat. Negative bias init => small gate.
        self.gate_sensor = nn.Linear(d_model, d_model, bias=False)
        self.gate_identity = nn.Linear(d_model, d_model, bias=True)
        nn.init.constant_(self.gate_identity.bias, float(gate_bias_init))

    def forward(
        self,
        sensor_tokens: torch.Tensor,      # (B, P, C, d)
        role_tokens: torch.Tensor,        # dense (B,C,S,text_dim) or unique (U_role,S,text_dim)
        role_mask: torch.Tensor,          # dense (B,C,S) or unique (U_role,S) bool
        sensor_text_tokens: torch.Tensor, # dense (B,N,S,text_dim) or unique (U_sensor,S,text_dim)
        sensor_text_mask: torch.Tensor,   # dense (B,N,S) or unique (U_sensor,S) bool
        sensor_id: torch.Tensor,          # (B, C) long: which sensor each channel belongs to
        role_text_ids: Optional[torch.Tensor] = None,    # (B,C) -> unique role row
        sensor_text_ids: Optional[torch.Tensor] = None,  # (B,N_sensors) -> unique sensor row
    ) -> torch.Tensor:
        B, P, C, D = sensor_tokens.shape
        if (role_text_ids is None) != (sensor_text_ids is None):
            raise ValueError("role_text_ids and sensor_text_ids must be provided together")

        if role_text_ids is None:
            Sr = role_tokens.shape[2]
            Ns = sensor_text_tokens.shape[1]
            Ss = sensor_text_tokens.shape[2]
            role = self.pool(role_tokens.reshape(B * C, Sr, -1),
                             role_mask.reshape(B * C, Sr)).reshape(B, C, D)
            sens = self.pool(sensor_text_tokens.reshape(B * Ns, Ss, -1),
                             sensor_text_mask.reshape(B * Ns, Ss)).reshape(B, Ns, D)
        else:
            # Factored descriptions repeat heavily: a live B=256 batch has 1,536 role rows but
            # only x/y/z/(unknown), and roughly 380 sensor rows but about 70 unique strings. Pool
            # each unique frozen-LM sequence once, then gather. index_select remains fully
            # differentiable and sums gradients from every repeated use into the shared pooler. In
            # training mode, repeats intentionally share that description's pooler-dropout draw for
            # this pass; deterministic/eval outputs and gradients are exactly the dense computation.
            if role_text_ids.shape != (B, C):
                raise ValueError(
                    f"role_text_ids must have shape {(B, C)}, got {tuple(role_text_ids.shape)}"
                )
            if sensor_text_ids.ndim != 2 or sensor_text_ids.shape[0] != B:
                raise ValueError("sensor_text_ids must have shape (B, N_sensors)")
            if role_tokens.ndim != 3 or role_mask.ndim != 2:
                raise ValueError("unique role text tensors must have shapes (U,S,D) and (U,S)")
            if sensor_text_tokens.ndim != 3 or sensor_text_mask.ndim != 2:
                raise ValueError("unique sensor text tensors must have shapes (U,S,D) and (U,S)")
            role_unique = self.pool(role_tokens, role_mask)
            sensor_unique = self.pool(sensor_text_tokens, sensor_text_mask)
            role = role_unique.index_select(0, role_text_ids.reshape(-1)).reshape(B, C, D)
            safe_sensor_ids = sensor_text_ids.clamp_min(0)
            sens = sensor_unique.index_select(0, safe_sensor_ids.reshape(-1)).reshape(
                B, sensor_text_ids.shape[1], D
            )
        # Broadcast each sensor's identity to its channels.
        sens_bc = torch.gather(sens, 1, sensor_id.unsqueeze(-1).expand(B, C, D))  # (B, C, d)

        identity = role + sens_bc                                                # (B, C, d)
        gate = torch.sigmoid(self.gate_sensor(sensor_tokens)
                             + self.gate_identity(identity).unsqueeze(1))        # (B, P, C, d)
        return sensor_tokens + gate * identity.unsqueeze(1)


class ChannelTextFusion(nn.Module):
    """
    Efficient per-channel text fusion with broadcast to patches.

    Instead of O(B×P×C) attention ops (attending for every sensor token),
    we do O(C) ops (pool each channel's text once) then broadcast.

    Flow:
    1. Pool each channel's text tokens → (C, D) channel embeddings
    2. Broadcast to all sensor tokens via learned gating
    """

    def __init__(
        self,
        d_model: int = 384,
        num_heads: int = 4,
        num_queries: int = 4,
        dropout: float = 0.1,
        text_dim: int = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        text_dim = text_dim or d_model

        # Project text tokens to d_model when text encoder dim differs (e.g. small_wide: 768→384)
        self.text_proj = nn.Linear(text_dim, d_model) if text_dim != d_model else nn.Identity()

        # Learnable queries to pool text tokens (one set shared across channels)
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)

        # Cross-attention: queries attend to (projected) text tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Project pooled queries to single channel embedding
        self.out_proj = nn.Sequential(
            nn.Linear(d_model * num_queries, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Gate: control how much text info to incorporate per sensor token
        # Split into two linear projections to avoid materializing (B, P, C, 2*D) concat tensor
        # Mathematically equivalent to Linear(cat(sensor, channel), d_model) since
        # W @ [a; b] = W_a @ a + W_b @ b when W is split column-wise
        self.gate_sensor = nn.Linear(d_model, d_model, bias=False)
        self.gate_channel = nn.Linear(d_model, d_model, bias=True)  # bias on one is sufficient

    def forward(
        self,
        sensor_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse sensor tokens with channel text descriptions (batched).

        Processes all samples in the batch in a single attention call by
        reshaping (B, C) into the batch dimension. Mathematically identical
        to the per-sample version since cross-attention is independent per channel.

        Args:
            sensor_tokens: (batch, patches, channels, d_model)
            text_tokens: (batch, channels, seq_len, d_model)
            text_mask: (batch, channels, seq_len) bool

        Returns:
            fused: (batch, patches, channels, d_model)
        """
        B, P, C, D = sensor_tokens.shape
        S = text_tokens.shape[2]  # seq_len

        # Flatten batch and channel dims for batched cross-attention: (B*C, ...)
        text_tokens_flat = text_tokens.reshape(B * C, S, -1)
        text_tokens_flat = self.text_proj(text_tokens_flat)  # (B*C, S, d_model)
        text_mask_flat = text_mask.reshape(B * C, S)
        text_mask_bool = text_mask_flat.bool()

        # Guard against all-masked channels (e.g. padding channels) which would
        # cause NaN in softmax (all -inf inputs). Unmask first position as a dummy
        # so attention produces a finite (if meaningless) output for those channels.
        all_masked = ~text_mask_bool.any(dim=1)  # (B*C,) True where entire row is masked
        if all_masked.any():
            text_mask_bool = text_mask_bool.clone()
            text_mask_bool[all_masked, 0] = True

        # Step 1: Pool each channel's text tokens to single embedding
        # B*C attention operations in one batched call
        queries = self.queries.unsqueeze(0).expand(B * C, -1, -1)  # (B*C, num_queries, D)

        attn_out, _ = self.cross_attn(
            query=queries,
            key=text_tokens_flat,
            value=text_tokens_flat,
            key_padding_mask=~text_mask_bool,
            need_weights=False
        )
        attn_out = self.norm1(queries + attn_out)

        # Combine queries: (B*C, num_queries, D) → (B*C, D)
        channel_embs = self.out_proj(attn_out.reshape(B * C, -1))

        # Step 2: Reshape and broadcast to all patches
        # (B*C, D) → (B, 1, C, D) for broadcasting across patches
        channel_embs = channel_embs.reshape(B, C, D).unsqueeze(1)

        # Gated fusion: sensor tokens control how much text to incorporate
        # Uses split linear projections to avoid materializing (B, P, C, 2*D) concat tensor
        gate = torch.sigmoid(self.gate_sensor(sensor_tokens) + self.gate_channel(channel_embs))
        fused = sensor_tokens + gate * channel_embs

        return fused
