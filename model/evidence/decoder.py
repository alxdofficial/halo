"""Evidence decoder (Tier-2, T2.3) — the set-transformer that refines retrieved evidence
before it votes. Implements docs/design/EVIDENCE_ENGINE_TIER2.md §2 + §2.2 (transformer hygiene).

Per query we have already retrieved a set of top-k EVIDENCE tokens (labeled memory neighbours)
plus the QUERY token(s). This module mixes them with self-attention, then:

  (a) REFINES each evidence token's label text   t'_i = normalize( t(label_i) + Δ(state_i) )
      — a residual in frozen-SBERT space, EVIDENCE side only (never the target labels). This is
      what disambiguates fine-grained classes ("walking" state near stairs -> "walking upstairs").
  (b) POOLS the evidence as a residual on the RETRIEVAL prior   a_i = softmax( log w_retr + φ(state_i) ).
  Then votes against the FROZEN target text and accumulates:   e_c = Σ_i a_i · relu(⟨t'_i, t_frozen(c)⟩).

Identity-at-init (the do-no-harm gate, made exact):
  * LayerScale γ≈0 on every sublayer  → transformer output ≡ input embeddings at step 0.
  * refiner Δ zero-init                → t'_i = t(label_i) (no refinement until earned).
  * pooling φ zero-init                → a_i = w_retr (reproduces the untrained weighted-sum).
  ⇒ decoder@init == the untrained retrieval+text-ensemble mechanism (47.5). Any gain is additive.

Transformer hygiene (§2.2): pre-LN + final LN; position-wise GELU FFN (4×); MHA with 1/√d_head
scaling + key-padding mask (masked-softmax) + a learned *same-window* additive bias; positional =
**additive window-relative continuous-time Fourier features** (NOT a global sequence index / RoPE —
cross-window relative time is meaningless); structural = learned role, text-keyed config, projected
label text (query gets a learned no-label token). BatchNorm is never used (set sizes vary).

The encoder never runs here — inputs are frozen pooled/patch vectors + frozen SBERT text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.evidence.multisubspace import MultiSubspaceHead

ROLE_QUERY = 0
ROLE_EVIDENCE = 1


@dataclass
class DecoderConfig:
    d_model: int = 256          # matches the encoder so patch vectors need no up-projection
    text_dim: int = 384         # frozen SBERT dim
    n_layers: int = 3
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    n_time_freqs: int = 16      # continuous-time Fourier bands (window-relative)
    time_max_sec: float = 30.0  # longest window we scale frequencies for
    layerscale_init: float = 1e-4
    out_scale_init: float = 10.0
    n_subspaces: int = 0        # D2: multi-subspace pooling re-weighting (0 = off, byte-identical)
    subspace_dim: int = 64      # per-subspace projection dim for the multi-subspace head


def _fourier_time(t: torch.Tensor, n_freqs: int, time_max: float) -> torch.Tensor:
    """(...,) physical seconds (window-relative) -> (..., 2*n_freqs) sin/cos features.

    Frequencies are geometrically spaced across [1/time_max, ~Nyquist-ish], so the features
    resolve both slow (whole-window) and fast (patch-scale) temporal position. t may be 0 for
    pooled (whole-window) tokens -> a constant [0,1,0,1,...] vector (no positional info), which
    is exactly right: a pooled token has no within-window position.
    """
    device = t.device
    freqs = torch.exp(torch.linspace(math.log(1.0 / time_max), math.log(time_max),
                                     n_freqs, device=device))     # (F,)
    ang = t.unsqueeze(-1) * freqs * (2 * math.pi)                 # (..., F)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)   # (..., 2F)


class _Block(nn.Module):
    """Pre-LN transformer block with LayerScale (identity-at-init when γ≈0)."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        d = cfg.d_model
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ls1 = nn.Parameter(cfg.layerscale_init * torch.ones(d))
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.ffn_mult * d), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn_mult * d, d),
        )
        self.ls2 = nn.Parameter(cfg.layerscale_init * torch.ones(d))

    def forward(self, x, key_padding_mask, attn_bias):
        # attn_bias: (B*n_heads, T, T) additive float (same-window bias), or None.
        h = self.ln1(x)
        # torch deprecates mixing a bool key_padding_mask with a float attn_mask, so when an
        # additive bias is present convert the padding mask to the same float additive form.
        if attn_bias is not None and key_padding_mask is not None and key_padding_mask.dtype == torch.bool:
            key_padding_mask = torch.zeros_like(key_padding_mask, dtype=attn_bias.dtype).masked_fill(
                key_padding_mask, float("-inf"))
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         attn_mask=attn_bias, need_weights=False)
        x = x + self.ls1 * a
        x = x + self.ls2 * self.ffn(self.ln2(x))
        return x


class EvidenceDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig | None = None, **kw):
        super().__init__()
        cfg = cfg or DecoderConfig(**kw)
        self.cfg = cfg
        d = cfg.d_model

        # --- input embedding pieces (all summed into the token vector) ---
        self.proj_z = nn.Linear(cfg.d_model, d)          # frozen encoder patch/window vector
        self.proj_cfg = nn.Linear(cfg.text_dim, d)       # text-keyed config/placement (frozen SBERT)
        self.proj_lab = nn.Linear(cfg.text_dim, d)       # evidence known-label text (frozen SBERT)
        self.proj_time = nn.Linear(2 * cfg.n_time_freqs, d)
        self.role_emb = nn.Embedding(2, d)               # QUERY | EVIDENCE
        self.q_nolabel = nn.Parameter(torch.zeros(d))    # query stands in for "no label text"
        self.in_ln = nn.LayerNorm(d)

        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_layers))
        self.final_ln = nn.LayerNorm(d)
        self.same_window_bias = nn.Parameter(torch.zeros(()))   # learned; init 0 (no bias)

        # --- output heads (zero-init last layer => identity at init) ---
        self.refiner = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.text_dim))
        self.pool_phi = nn.Linear(d, 1)
        self.log_out_scale = nn.Parameter(torch.tensor(math.log(cfg.out_scale_init)))
        self._zero_init_heads()

        # --- D2: multi-subspace pooling residual (opt-in; identity-at-init via gamma_ms=0) ---
        # When n_subspaces == 0 (default) neither the head nor gamma_ms exist, so the state_dict
        # and forward are byte-identical to the pre-D2 decoder. When > 0 the head is built and its
        # contribution is gated by a zero-init scalar => still identity at init.
        self.ms_head = None
        if cfg.n_subspaces > 0:
            self.ms_head = MultiSubspaceHead(cfg.d_model, cfg.n_subspaces, cfg.subspace_dim)
            self.gamma_ms = nn.Parameter(torch.zeros(()))    # zero-init gate -> no effect @ init

    def _zero_init_heads(self):
        nn.init.zeros_(self.refiner[-1].weight); nn.init.zeros_(self.refiner[-1].bias)
        nn.init.zeros_(self.pool_phi.weight); nn.init.zeros_(self.pool_phi.bias)

    # ------------------------------------------------------------------ #
    def _embed(self, z, role, label_text, config_text, time_sec):
        """Assemble a token embedding. z:(...,d_model); role:(...,) long; label_text/config_text:
        (...,text_dim) or None; time_sec:(...,) or None."""
        e = self.proj_z(z) + self.role_emb(role)
        if config_text is not None:
            e = e + self.proj_cfg(config_text)
        if time_sec is not None:
            e = e + self.proj_time(_fourier_time(time_sec, self.cfg.n_time_freqs,
                                                 self.cfg.time_max_sec))
        if label_text is not None:
            e = e + self.proj_lab(label_text)
        return e

    def forward(
        self,
        zq,                 # (B, d)         query pooled/patch vector (single query token)
        zev,                # (B, k, d)      evidence vectors (retrieved)
        ev_label_text,      # (B, k, text)   frozen SBERT (ensembled) label text per evidence
        w_retr,             # (B, k)         retrieval weights (normalized over valid evidence)
        cand_text,          # (B, C, text) or (C, text)  frozen SBERT target label text
        ev_mask=None,       # (B, k) bool    True = valid evidence (else padding)
        q_config_text=None, ev_config_text=None,    # (B,text) / (B,k,text)  frozen SBERT config
        q_time=None, ev_time=None,                  # (B,) / (B,k)  window-relative seconds
        window_id=None,     # (B, 1+k) long  co-membership groups for the same-window bias
        return_aux=False,
    ):
        B, k, d = zev.shape
        dev = zev.device
        # ---- build the token set: [query] + [k evidence] ----
        q_tok = self._embed(zq, torch.full((B,), ROLE_QUERY, device=dev, dtype=torch.long),
                            None, q_config_text, q_time)                          # (B, d)
        q_tok = q_tok + self.q_nolabel
        ev_role = torch.full((B, k), ROLE_EVIDENCE, device=dev, dtype=torch.long)
        ev_tok = self._embed(zev, ev_role, ev_label_text, ev_config_text, ev_time)   # (B, k, d)
        x = torch.cat([q_tok.unsqueeze(1), ev_tok], dim=1)                       # (B, 1+k, d)
        x = self.in_ln(x)

        # ---- masks ----
        T = k + 1
        if ev_mask is None:
            ev_mask = torch.ones(B, k, dtype=torch.bool, device=dev)
        # key-padding: query (col 0) always valid; padded evidence keys ignored.
        key_pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=dev), ~ev_mask], dim=1)
        attn_bias = self._same_window_bias(window_id, T, B, key_pad, dev)

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_pad, attn_bias=attn_bias)
        x = self.final_ln(x)
        ev_state = x[:, 1:, :]                                                    # (B, k, d)

        # ---- (a) refine evidence label text (residual, evidence side, frozen target) ----
        delta = self.refiner(ev_state)                                           # (B, k, text) — 0 @ init
        t_ref = F.normalize(ev_label_text + delta, dim=-1)                       # (B, k, text)

        # ---- vote against FROZEN target text ----
        if cand_text.dim() == 2:
            cand_text = cand_text.unsqueeze(0).expand(B, -1, -1)                 # (B, C, text)
        cand = F.normalize(cand_text, dim=-1)
        votes = torch.relu(torch.einsum("bkt,bct->bkc", t_ref, cand))           # (B, k, C)

        # ---- (b) pool as a residual on the retrieval prior ----
        phi = self.pool_phi(ev_state).squeeze(-1)                                # (B, k) — 0 @ init
        a_logit = torch.log(w_retr.clamp_min(1e-12)) + phi
        # D2: multi-subspace re-weighting, gated by gamma_ms (0 @ init => a_logit UNCHANGED, so
        # a/e/logits stay byte-identical to the untrained mechanism). psi is finite (bounded
        # cosines) so 0*psi is exactly 0 — no NaN — and padded/dead rows are masked below as before.
        if self.ms_head is not None:
            psi = self.ms_head(zq, zev)                                          # (B, k)
            a_logit = a_logit + self.gamma_ms * psi
        a_logit = a_logit.masked_fill(~ev_mask, float("-inf"))
        # A row with NO valid evidence would be all -inf -> softmax = NaN -> one bad row silently
        # NaNs the whole batch's CE. Keep such rows finite and let them pool to zero evidence.
        dead = ~ev_mask.any(dim=1, keepdim=True)                                 # (B, 1)
        a_logit = a_logit.masked_fill(dead, 0.0)
        a = torch.softmax(a_logit, dim=1)                                        # (B, k) == w_retr @ init
        a = a.masked_fill(dead, 0.0)                                             # dead rows: no evidence
        e = torch.einsum("bk,bkc->bc", a, votes)                                 # (B, C) evidence
        logits = self.log_out_scale.exp() * e
        if return_aux:
            aux = {"evidence": e, "pool_weights": a, "delta": delta,
                   "delta_norm": float(delta.detach().norm(dim=-1).mean())}
            if self.ms_head is not None:
                # abs value so the trainer can reg-to-identity it (L1) like Δ/pool; keeps grad.
                aux["gamma_ms"] = self.gamma_ms.abs()
            return logits, aux
        return logits

    def _same_window_bias(self, window_id, T, B, key_pad, dev):
        """Additive (B*n_heads, T, T) bias: +γ where two tokens share a window; permutation-safe."""
        # Do NOT gate on the parameter's VALUE. It is initialised to exactly 0.0, so a value test
        # short-circuits on every forward, the bias never enters the autograd graph, and the
        # parameter is frozen at zero for all time (verified: .grad stays None). Gate on whether
        # the caller supplied window ids. (float(param) also forced a CUDA->CPU sync per forward
        # and emitted a requires_grad-to-scalar warning.)
        if window_id is None:
            return None
        same = (window_id.unsqueeze(1) == window_id.unsqueeze(2)).float()        # (B, T, T)
        bias = self.same_window_bias * same                                      # (B, T, T)
        return bias.repeat_interleave(self.cfg.n_heads, dim=0)                   # (B*nh, T, T)

    # -- convenience: split params for weight-decay grouping (exclude LN/bias/γ/embeddings) --
    # This iterates ALL named_parameters, so the D2 multi-subspace params join automatically:
    # ms_head.proj.weight (ndim 2) -> decay; ms_head.omega_logits (ndim 1) and the gamma_ms
    # scalar (ndim 0) both fall under `p.ndim <= 1` -> no_decay, as required for the gate scalar.
    def param_groups(self, weight_decay: float = 0.01):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or "ls1" in name or "ls2" in name or "emb" in name \
                    or "same_window_bias" in name or name.endswith("q_nolabel"):
                no_decay.append(p)
            else:
                decay.append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]
