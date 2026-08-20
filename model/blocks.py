"""ONE attention block specification, shared by every attention stack in HALO.

WHY THIS FILE EXISTS
--------------------
HALO has three places where attention happens: the encoder trunk (temporal, per sensor), the
retrieval scorer (a factored bilinear form, which is attention logits without the softmax), and the
evidence mixer (set attention over retrieved evidence). Historically each picked its own width,
head count and feed-forward multiplier, so the model had no single "size" — every capacity question
turned into three.

:class:`AttentionSpec` is the answer: one width, one head count, one FFN multiplier, one dropout,
instantiated once and passed everywhere. Growing the model means changing one number, and
:meth:`AttentionSpec.parameters_per_block` reports what that number costs before anything is built.

DESIGN OF THE BLOCK
-------------------
Pre-norm (``x + attn(norm(x))``), which is the current default for stability without warmup —
the post-norm arrangement in :class:`~model.tokenizer.transformer.DualBranchTransformerBlock`
predates this file and is kept only for checkpoint compatibility. GELU feed-forward at
``ffn_mult * d_model``, dropout on both sub-layer outputs, and a final norm at the end of a stack
so downstream consumers see a normalised representation regardless of depth.

SET ATTENTION, NOT SEQUENCE ATTENTION
-------------------------------------
:class:`SetAttentionStack` has no positional encoding of any kind. Its inputs are *sets* —
candidate labels, retrieved evidence rows — where order is an artifact of how the tensor was
assembled and must not be readable. Identity comes from additive role/coreference embeddings the
caller supplies, never from position. It accepts:

* ``key_padding_mask`` — variable-size sets in one batch (episodes differ in candidate count),
* ``attn_bias`` — an additive per-pair term in logit units, which is how the retrieval score
  reaches attention without becoming an input feature that a later layer could launder into a
  query-to-candidate path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AttentionSpec:
    """The size of every attention block in the model.

    ``d_model`` is also the width of the representation the encoder hands to retrieval, so this one
    number sets the feature dimension end to end.
    """

    d_model: int = 128
    n_heads: int = 4
    ffn_mult: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}"
            )
        if self.ffn_mult <= 0:
            raise ValueError("ffn_mult must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

    @property
    def dim_feedforward(self) -> int:
        return self.d_model * self.ffn_mult

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def parameters_per_block(self) -> int:
        """Learnable parameters in one pre-norm attention + FFN block.

        Stated in closed form rather than measured so a capacity decision can be made from the
        spec alone, before any module is constructed. ``tests`` asserts it against the real count.
        """
        d, f = self.d_model, self.dim_feedforward
        qkv = d * 3 * d + 3 * d
        out = d * d + d
        ffn = d * f + f + f * d + d
        norms = 2 * (2 * d)
        return qkv + out + ffn + norms

    def describe(self) -> str:
        return (f"d_model={self.d_model} heads={self.n_heads} ffn={self.dim_feedforward} "
                f"dropout={self.dropout} ({self.parameters_per_block():,} params/block)")


class SetAttentionStack(nn.Module):
    """Pre-norm self-attention over an unordered set, with optional additive pair bias.

    Shapes are ``(B, N, d_model)``. There is no positional information anywhere in this module: two
    inputs that differ only by a permutation of ``N`` produce outputs that differ by the same
    permutation. ``tests/test_blocks.py`` asserts that equivariance directly, because it is the
    property that lets an episode present a variable number of candidates and evidence rows without
    the model reading anything off their ordering.
    """

    def __init__(self, spec: AttentionSpec, n_layers: int):
        super().__init__()
        if n_layers < 0:
            raise ValueError("n_layers cannot be negative")
        self.spec = spec
        self.n_layers = int(n_layers)
        self.attn_norm = nn.ModuleList(
            [nn.LayerNorm(spec.d_model) for _ in range(n_layers)]
        )
        self.qkv = nn.ModuleList(
            [nn.Linear(spec.d_model, 3 * spec.d_model) for _ in range(n_layers)]
        )
        self.proj = nn.ModuleList(
            [nn.Linear(spec.d_model, spec.d_model) for _ in range(n_layers)]
        )
        self.ffn_norm = nn.ModuleList(
            [nn.LayerNorm(spec.d_model) for _ in range(n_layers)]
        )
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(spec.d_model, spec.dim_feedforward),
                nn.GELU(),
                nn.Dropout(spec.dropout),
                nn.Linear(spec.dim_feedforward, spec.d_model),
            )
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(spec.dropout)
        self.out_norm = nn.LayerNorm(spec.d_model)

    def forward(
        self,
        x: torch.Tensor,                               # (B, N, d)
        *,
        key_padding_mask: torch.Tensor | None = None,  # (B, N) True = real token
        attn_bias: torch.Tensor | None = None,         # (B, N, N) or (B, H, N, N), additive
    ) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != self.spec.d_model:
            raise ValueError(
                f"expected (B, N, {self.spec.d_model}), got {tuple(x.shape)}"
            )
        batch, n_tokens, _ = x.shape
        heads, head_dim = self.spec.n_heads, self.spec.head_dim

        mask = None
        if key_padding_mask is not None or attn_bias is not None:
            mask = x.new_zeros((batch, 1, 1, n_tokens))
            if attn_bias is not None:
                bias = attn_bias if attn_bias.dim() == 4 else attn_bias.unsqueeze(1)
                mask = mask + bias
            if key_padding_mask is not None:
                # A row that attends to nothing yields NaN from softmax. Padded tokens are still
                # visible to themselves so every row has at least one finite logit; their outputs
                # are meaningless and the caller drops them.
                blocked = ~key_padding_mask.view(batch, 1, 1, n_tokens)
                mask = mask.masked_fill(blocked, torch.finfo(x.dtype).min)
                eye = torch.eye(n_tokens, dtype=torch.bool, device=x.device)
                mask = torch.where(eye.view(1, 1, n_tokens, n_tokens), x.new_zeros(()), mask)

        for layer in range(self.n_layers):
            normed = self.attn_norm[layer](x)
            qkv = self.qkv[layer](normed).view(batch, n_tokens, 3, heads, head_dim)
            query, key, value = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
            attended = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=mask,
                dropout_p=self.spec.dropout if self.training else 0.0,
            )
            attended = attended.transpose(1, 2).reshape(batch, n_tokens, self.spec.d_model)
            x = x + self.dropout(self.proj[layer](attended))
            x = x + self.dropout(self.ffn[layer](self.ffn_norm[layer](x)))
        return self.out_norm(x)


class ScaledSum(nn.Module):
    """Sum unit-norm ingredients, each scaled by its own learned positive weight.

    Adding a raw ``nn.Embedding`` (norm about ``sqrt(d)``) to a projection of a unit-norm text
    vector lets the embedding drown the content. Normalising each ingredient to a direction and
    giving it a learned gain starting at one makes every additive channel start on equal footing
    and lets the model decide the mixture.
    """

    def __init__(self, n_terms: int):
        super().__init__()
        self.log_gain = nn.Parameter(torch.zeros(n_terms))

    def forward(self, *terms: torch.Tensor) -> torch.Tensor:
        if len(terms) != self.log_gain.numel():
            raise ValueError(
                f"ScaledSum built for {self.log_gain.numel()} terms, got {len(terms)}"
            )
        total = None
        for index, term in enumerate(terms):
            scaled = self.log_gain[index].exp() * F.normalize(term, dim=-1)
            total = scaled if total is None else total + scaled
        return total
