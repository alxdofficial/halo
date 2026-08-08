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

ROLE_QUERY = 0
ROLE_EVIDENCE = 1
ROLE_SUPPORT = 2
ROLE_CANDIDATE = 3


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
    candidate_tokens: bool = False
    candidate_layers: int = 2
    structural_metadata: bool = False
    support_role: bool = False


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


class _CandidateBlock(nn.Module):
    """Permutation-equivariant candidate self-attention followed by physical cross-attention."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        d = cfg.d_model
        self.ln_self = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(
            d, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ls_self = nn.Parameter(cfg.layerscale_init * torch.ones(d))
        self.ln_cross = nn.LayerNorm(d)
        self.ln_memory = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(
            d, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ls_cross = nn.Parameter(cfg.layerscale_init * torch.ones(d))
        self.ln_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.ffn_mult * d), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn_mult * d, d),
        )
        self.ls_ffn = nn.Parameter(cfg.layerscale_init * torch.ones(d))

    def forward(self, candidate, memory, memory_key_padding_mask):
        h = self.ln_self(candidate)
        attended, _ = self.self_attn(h, h, h, need_weights=False)
        candidate = candidate + self.ls_self * attended
        attended, _ = self.cross_attn(
            self.ln_cross(candidate),
            self.ln_memory(memory),
            self.ln_memory(memory),
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        candidate = candidate + self.ls_cross * attended
        candidate = candidate + self.ls_ffn * self.ffn(self.ln_ffn(candidate))
        return candidate


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
        n_roles = 2 + int(cfg.support_role) + int(cfg.candidate_tokens)
        self.role_emb = nn.Embedding(n_roles, d)         # QUERY | EVIDENCE | optional CANDIDATE
        self.support_role_id = ROLE_SUPPORT if cfg.support_role else ROLE_EVIDENCE
        self.candidate_role_id = (
            ROLE_CANDIDATE if cfg.support_role else ROLE_SUPPORT
        )
        self.q_nolabel = nn.Parameter(torch.zeros(d))    # query stands in for "no label text"
        self.in_ln = nn.LayerNorm(d)

        # Acquisition values already shaped the frozen Phase-A vector. These optional fields encode
        # only structural relationships needed by Phase B: represented support and resolution
        # identity. Sensor membership is encoded as a relation bias below, never as a global ID.
        if cfg.structural_metadata:
            self.proj_duration = nn.Linear(1, d)
            self.resolution_emb = nn.Embedding(3, d)     # 0=unknown/pad, 1=short/single, 2=long

        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_layers))
        self.final_ln = nn.LayerNorm(d)
        self.same_window_bias = nn.Parameter(torch.zeros(()))   # learned; init 0 (no bias)
        if cfg.structural_metadata:
            self.same_sensor_bias = nn.Parameter(torch.zeros(()))

        # --- output heads (zero-init last layer => identity at init) ---
        self.refiner = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.text_dim))
        self.pool_phi = nn.Linear(d, 1)
        self.log_out_scale = nn.Parameter(torch.tensor(math.log(cfg.out_scale_init)))
        self._zero_init_heads()

        self.candidate_blocks = None
        if cfg.candidate_tokens:
            self.candidate_in_ln = nn.LayerNorm(d)
            self.candidate_blocks = nn.ModuleList(
                _CandidateBlock(cfg) for _ in range(cfg.candidate_layers)
            )
            self.candidate_final_ln = nn.LayerNorm(d)
            self.candidate_refiner = nn.Linear(d, cfg.text_dim)
            nn.init.zeros_(self.candidate_refiner.weight)
            nn.init.zeros_(self.candidate_refiner.bias)

    def _zero_init_heads(self):
        nn.init.zeros_(self.refiner[-1].weight); nn.init.zeros_(self.refiner[-1].bias)
        nn.init.zeros_(self.pool_phi.weight); nn.init.zeros_(self.pool_phi.bias)

    # ------------------------------------------------------------------ #
    def _embed(self, z, role, label_text, config_text, time_sec, duration_sec=None,
               resolution_id=None):
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
        if self.cfg.structural_metadata:
            if duration_sec is not None:
                duration_feature = torch.tanh(
                    duration_sec.to(dtype=e.dtype).clamp_min(1e-4).log() / 2.0
                ).unsqueeze(-1)
                e = e + self.proj_duration(duration_feature)
            if resolution_id is not None:
                resolution_index = resolution_id.to(dtype=torch.long).clamp(-1, 1) + 1
                e = e + self.resolution_emb(resolution_index)
        return e

    def forward(
        self,
        zq,                 # (B, d) or (B, q, d) query pooled/patch vector(s)
        zev,                # (B, k, d)      evidence vectors (retrieved)
        ev_label_text,      # (B, k, text)   frozen SBERT (ensembled) label text per evidence
        w_retr,             # (B, k)         retrieval weights (normalized over valid evidence)
        cand_text,          # (B, C, text) or (C, text)  frozen SBERT target label text
        ev_mask=None,       # (B, k) bool    True = valid evidence (else padding)
        ev_support_mask=None,  # (B, k) bool True = episode-provided labeled support
        q_mask=None,        # (B, q) bool    True = valid query patch
        q_config_text=None, ev_config_text=None,    # (B,text) / (B,k,text)  frozen SBERT config
        q_time=None, ev_time=None,                  # (B,q) / (B,k) window-relative seconds
        q_duration=None, ev_duration=None,          # (B,q) / (B,k) represented seconds
        q_resolution=None, ev_resolution=None,      # (B,q) / (B,k), -1/0/1
        q_sensor_id=None, ev_sensor_id=None,        # (B,q) / (B,k) structural sensor group
        window_id=None,     # (B, q+k) long  co-membership groups for the same-window bias
        return_aux=False,
    ):
        B, k, d = zev.shape
        dev = zev.device
        single_query = zq.dim() == 2
        if single_query:
            zq = zq.unsqueeze(1)
            if q_config_text is not None and q_config_text.dim() == 2:
                q_config_text = q_config_text.unsqueeze(1)
            for name in ("q_time", "q_duration", "q_resolution", "q_sensor_id"):
                value = locals()[name]
                if value is not None and value.dim() == 1:
                    if name == "q_time":
                        q_time = value.unsqueeze(1)
                    elif name == "q_duration":
                        q_duration = value.unsqueeze(1)
                    elif name == "q_resolution":
                        q_resolution = value.unsqueeze(1)
                    else:
                        q_sensor_id = value.unsqueeze(1)
        elif zq.dim() != 3:
            raise ValueError(f"zq must have shape (B,d) or (B,q,d), got {tuple(zq.shape)}")
        if zq.shape[0] != B or zq.shape[2] != d:
            raise ValueError(f"zq and zev batch/feature dimensions disagree: {zq.shape} vs {zev.shape}")
        q = zq.shape[1]
        if q_mask is None:
            q_mask = torch.ones(B, q, dtype=torch.bool, device=dev)
        elif q_mask.shape != (B, q):
            raise ValueError(f"q_mask must have shape {(B, q)}, got {tuple(q_mask.shape)}")

        # ---- build the token set: [q query patches] + [k evidence patches] ----
        q_tok = self._embed(
            zq, torch.full((B, q), ROLE_QUERY, device=dev, dtype=torch.long),
            None, q_config_text, q_time, q_duration, q_resolution,
        )
        q_tok = q_tok + self.q_nolabel
        ev_role = torch.full((B, k), ROLE_EVIDENCE, device=dev, dtype=torch.long)
        if ev_support_mask is not None:
            if not self.cfg.support_role:
                raise ValueError("ev_support_mask requires support_role=True")
            if ev_support_mask.shape != (B, k):
                raise ValueError(f"ev_support_mask must have shape {(B, k)}")
            ev_role = ev_role.masked_fill(ev_support_mask, self.support_role_id)
        ev_tok = self._embed(
            zev, ev_role, ev_label_text, ev_config_text, ev_time,
            ev_duration, ev_resolution,
        )
        x = torch.cat([q_tok, ev_tok], dim=1)                                   # (B, q+k, d)
        x = self.in_ln(x)

        # ---- masks ----
        T = q + k
        if ev_mask is None:
            ev_mask = torch.ones(B, k, dtype=torch.bool, device=dev)
        # key-padding: invalid query/evidence patches are ignored as keys.
        key_pad = torch.cat([~q_mask, ~ev_mask], dim=1)
        sensor_id = (
            torch.cat([q_sensor_id, ev_sensor_id], dim=1)
            if q_sensor_id is not None and ev_sensor_id is not None else None
        )
        attn_bias = self._structural_bias(window_id, sensor_id, T, B, key_pad, dev)

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_pad, attn_bias=attn_bias)
        x = self.final_ln(x)
        ev_state = x[:, q:, :]                                                    # (B, k, d)

        # ---- (a) refine evidence label text (residual, evidence side, frozen target) ----
        delta = self.refiner(ev_state)                                           # (B, k, text) — 0 @ init
        t_ref = F.normalize(ev_label_text + delta, dim=-1)                       # (B, k, text)

        # ---- vote against FROZEN target text ----
        if cand_text.dim() == 2:
            cand_text = cand_text.unsqueeze(0).expand(B, -1, -1)                 # (B, C, text)
        candidate_delta = None
        if self.candidate_blocks is not None:
            cand_role = torch.full(
                cand_text.shape[:2], self.candidate_role_id, device=dev, dtype=torch.long
            )
            candidate_state = self.candidate_in_ln(
                self.proj_lab(cand_text) + self.role_emb(cand_role)
            )
            for block in self.candidate_blocks:
                candidate_state = block(candidate_state, x, key_pad)
            candidate_state = self.candidate_final_ln(candidate_state)
            candidate_delta = self.candidate_refiner(candidate_state)
            cand_text = cand_text + candidate_delta
        cand = F.normalize(cand_text, dim=-1)
        votes = torch.relu(torch.einsum("bkt,bct->bkc", t_ref, cand))           # (B, k, C)

        # ---- (b) pool as a residual on the retrieval prior ----
        phi = self.pool_phi(ev_state).squeeze(-1)                                # (B, k) — 0 @ init
        a_logit = torch.log(w_retr.clamp_min(1e-12)) + phi
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
            aux = {"evidence": e, "votes": votes, "pool_weights": a, "delta": delta,
                   "delta_norm": float(delta.detach().norm(dim=-1).mean())}
            if candidate_delta is not None:
                aux["candidate_delta"] = candidate_delta
                aux["candidate_delta_norm"] = float(
                    candidate_delta.detach().norm(dim=-1).mean()
                )
            return logits, aux
        return logits

    def _structural_bias(self, window_id, sensor_id, T, B, key_pad, dev):
        """Permutation-safe learned biases for same-window and same-sensor co-membership."""
        # Do NOT gate on the parameter's VALUE. It is initialised to exactly 0.0, so a value test
        # short-circuits on every forward, the bias never enters the autograd graph, and the
        # parameter is frozen at zero for all time (verified: .grad stays None). Gate on whether
        # the caller supplied window ids. (float(param) also forced a CUDA->CPU sync per forward
        # and emitted a requires_grad-to-scalar warning.)
        if window_id is None and sensor_id is None:
            return None
        bias = torch.zeros(B, T, T, device=dev)
        if window_id is not None:
            if window_id.shape != (B, T):
                raise ValueError(f"window_id must have shape {(B, T)}, got {tuple(window_id.shape)}")
            same = (window_id.unsqueeze(1) == window_id.unsqueeze(2)).float()
            bias = bias + self.same_window_bias * same
        if sensor_id is not None:
            if not self.cfg.structural_metadata:
                raise ValueError("sensor_id requires structural_metadata=True")
            if sensor_id.shape != (B, T):
                raise ValueError(f"sensor_id must have shape {(B, T)}, got {tuple(sensor_id.shape)}")
            same = (sensor_id.unsqueeze(1) == sensor_id.unsqueeze(2)).float()
            bias = bias + self.same_sensor_bias * same
        return bias.repeat_interleave(self.cfg.n_heads, dim=0)                   # (B*nh, T, T)

    # -- convenience: split params for weight-decay grouping (exclude LN/bias/γ/embeddings) --
    def param_groups(self, weight_decay: float = 0.01):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or "ls1" in name or "ls2" in name or "emb" in name \
                    or "same_window_bias" in name or "same_sensor_bias" in name \
                    or name.endswith("q_nolabel"):
                no_decay.append(p)
            else:
                decay.append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]
