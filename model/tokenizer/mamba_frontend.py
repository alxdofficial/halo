"""Per-channel selective state-space (Mamba-style) tokenizer with physical Δ = 1/rate.

The learned-recurrent alternative to the physical-Hz filterbank, for the tokenizer ablation
(docs/design/TOKENIZER_ABLATION.md). Same drop-in contract as ``PhysicalFilterbankTokenizer``:

    forward(patches (B,P,S,C) native-rate zero-padded, sampling_rate_hz, patch_len_samples)
        -> tokens (B, P, C, d_model)

**The physics.** A state-space model is a discretised continuous-time ODE ``dh/dt = A h + B u``;
its discretisation step Δ is the *physical time between samples*. We set the base step to
``Δ_phys = 1/rate`` (seconds/sample) and let the selective (input-dependent) term modulate it. So
the SAME motion sampled at 20 Hz and 100 Hz advances the state at the same physical speed and
integrates to the same state over the same physical window — **rate-invariance by construction**,
the SSM analogue of the filterbank's rate-invariant physical-Hz bands. Rate is therefore not a
side "conditioning token" the model must learn to interpret; it is the discretisation step itself.

**Per channel, shared weights.** One SSM is applied independently to each channel (weights shared
across channels, exactly as the filterbank shares one bank), so channel count/order stay free and
identity still comes from text downstream. Gravity is preserved: we do NOT instance-normalise
(per-window mean removal would destroy the DC/gravity component that separates static postures).
Standardisation is frozen and **per-modality** (accel vs gyro have ~1.6× different scales in the
corpus; one shared scalar would let the larger dominate σ and the shared in_proj cannot compensate),
pooled within each modality's axes so the relative gravity direction survives.

**Perf note.** The scan below is SEQUENTIAL (exact, portable, kernel-free) — correct and fine for
tests / small runs, but O(S) Python steps. Full pretraining should swap in an associative parallel
scan or the ``mamba_ssm`` CUDA kernel; the module interface is unchanged by that.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSMChannelTokenizer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_inner: int | None = None,     # SSM width; default 2*d_model
        d_conv: int = 4,                # short causal depthwise conv (local mixing)
        dt_rank: int | None = None,     # low-rank Δ projection; default ceil(d_inner/16)
        mult_min: float = 0.5,          # bounds of the dimensionless learned Δ multiplier (~1)
        mult_max: float = 2.0,
        standardize: bool = True,       # frozen PER-MODALITY (mu,sd); preserves gravity DC
        channel_groups: tuple = (0, 0, 0, 1, 1, 1),  # channel -> modality group (accel=0, gyro=1)
        pool: str = "mean",            # 'mean' over valid steps | 'last' valid state
        dft_size: int | None = None,    # accepted+ignored: drop-in kwarg parity with the filterbank
        **_ignored,                     # tolerate filterbank-only kwargs so build_frontend can pass through
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = int(d_state)
        self.d_inner = int(d_inner or 2 * d_model)
        self.d_conv = int(d_conv)
        self.dt_rank = int(dt_rank or math.ceil(self.d_inner / 16))
        self.mult_min, self.mult_max = float(mult_min), float(mult_max)
        self.pool = pool
        self.S = int(dft_size) if dft_size is not None else None   # for the drop-in length assert only

        E, N = self.d_inner, self.d_state

        # scalar signal -> SSM width (x) + gate branch (z)
        self.in_proj = nn.Linear(1, E)
        self.gate_proj = nn.Linear(1, E)
        # depthwise causal conv over time for local mixing (Mamba's short conv)
        self.conv = nn.Conv1d(E, E, kernel_size=self.d_conv, groups=E, padding=self.d_conv - 1)
        # selective params from x: low-rank Δ, plus input-dependent B and C
        self.x_proj = nn.Linear(E, self.dt_rank + 2 * N, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, E, bias=True)
        # A (diagonal, stable): A = -exp(A_log). Init spread over state index (S4-style timescales).
        A = torch.arange(1, N + 1, dtype=torch.float32).repeat(E, 1)         # (E, N)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(E))                                # skip connection
        self.out_proj = nn.Linear(E, d_model)

        # Δ = (1/rate) · softplus(dt_proj(·)). The learned term is a DIMENSIONLESS multiplier around
        # 1 (not a second absolute step) — so at init Δ ≈ the physical step 1/rate, the SSM state
        # accumulates over a physical window, and doubling the rate halves Δ so the same motion
        # integrates to the same state (rate-invariance). Init softplus(bias) ∈ [mult_min, mult_max].
        m0 = torch.exp(torch.rand(E) * (math.log(mult_max) - math.log(mult_min)) + math.log(mult_min))
        self.dt_proj.bias.data.copy_(torch.log(torch.expm1(m0)))           # inverse-softplus(m0)

        # Frozen PER-MODALITY standardisation. Accel (g) and gyro (rad/s) have different natural
        # scales (measured ~1.6x), so one shared scalar would let the larger modality dominate σ and
        # under-normalise the other — and the shared in_proj cannot compensate per channel. Stats are
        # pooled WITHIN each modality group (shared across that modality's axes), so the scale
        # fingerprint is removed while the RELATIVE gravity direction across the accel axes is
        # preserved (a shared μ shifts the 3 accel axes equally). Global-per-window normalisation
        # (RevIN) is deliberately avoided — it would erase gravity. Identity until calibrated.
        self.standardize = bool(standardize)
        cg = torch.tensor(channel_groups, dtype=torch.long)
        self.register_buffer("channel_group", cg)                          # (C,) channel -> group
        self.n_groups = int(cg.max().item()) + 1
        self.register_buffer("norm_mu", torch.zeros(self.n_groups))
        self.register_buffer("norm_sd", torch.ones(self.n_groups))
        self.register_buffer("_norm_fitted", torch.zeros(1))
        self.register_buffer("_acc_n", torch.zeros(self.n_groups, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sum", torch.zeros(self.n_groups, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sqsum", torch.zeros(self.n_groups, dtype=torch.float64), persistent=False)

    def get_output_dim(self) -> int:
        return self.d_model

    # ------------------------------------------------------------------ calibration (like filterbank)
    def reset_norm_accumulator(self):
        self._acc_n.zero_(); self._acc_sum.zero_(); self._acc_sqsum.zero_()

    @torch.no_grad()
    def accumulate_norm_stats(self, patches, sampling_rate_hz=None, patch_len_samples=None,
                              patch_mask=None, channel_mask=None):
        """Fold one batch into the PER-MODALITY standardisation stats over REAL samples only.

        Stats are pooled within each modality group (accel / gyro), so scale is normalised per
        modality while the relative gravity DC survives. Padded patches / absent channels / zero-pad
        time are excluded.
        """
        B, P, S, C = patches.shape
        if C != self.channel_group.numel():
            raise ValueError(f"channel_groups has {self.channel_group.numel()} entries but C={C}")
        w = torch.ones(B, P, S, C, dtype=torch.float64)
        if patch_len_samples is not None:
            N = torch.as_tensor(patch_len_samples).to(patches.device)
            if N.ndim == 0:
                N = N.view(1, 1).expand(B, P)
            elif N.ndim == 1:
                N = N.view(B, 1).expand(B, P)
            idx = torch.arange(S, device=patches.device).view(1, 1, S)
            w = w * (idx < N.unsqueeze(-1)).view(B, P, S, 1).to(torch.float64)
        if patch_mask is not None:
            w = w * patch_mask.view(B, P, 1, 1).to(torch.float64)
        if channel_mask is not None:
            w = w * channel_mask.view(B, 1, 1, C).to(torch.float64)
        x = patches.to(torch.float64)
        for gi in range(self.n_groups):
            sel = (self.channel_group == gi)
            if not sel.any():
                continue
            xg, wg = x[..., sel], w[..., sel]
            self._acc_n[gi] += wg.sum()
            self._acc_sum[gi] += (xg * wg).sum()
            self._acc_sqsum[gi] += (xg * xg * wg).sum()

    @torch.no_grad()
    def finalize_norm_stats(self, eps: float = 1e-5):
        seen = self._acc_n > 0
        n = self._acc_n.clamp(min=1.0)
        mu = self._acc_sum / n
        var = (self._acc_sqsum / n) - mu * mu
        sd = var.clamp(min=eps).sqrt()
        self.norm_mu.copy_(torch.where(seen, mu, torch.zeros_like(mu)).to(self.norm_mu.dtype))
        self.norm_sd.copy_(torch.where(seen, sd, torch.ones_like(sd)).to(self.norm_sd.dtype))
        self._norm_fitted.fill_(1.0)

    @torch.no_grad()
    def fit_norm_stats(self, patches, sampling_rate_hz=None, patch_len_samples=None, eps: float = 1e-5):
        self.reset_norm_accumulator()
        self.accumulate_norm_stats(patches, patch_len_samples=patch_len_samples)
        self.finalize_norm_stats(eps)

    # ------------------------------------------------------------------------------- forward
    def _prep_rate_len(self, sampling_rate_hz, patch_len_samples, B, P, device, dtype):
        r = torch.as_tensor(sampling_rate_hz, device=device, dtype=dtype).reshape(-1)
        if r.numel() == 1:
            r = r.expand(B)
        assert r.numel() == B, f"sampling_rate_hz must be scalar or length B={B}"
        if patch_len_samples is None:
            if not self.S:
                raise ValueError("patch_len_samples is None and dft_size was not set: the true patch "
                                 "length is unknown, which would mask every timestep. Pass "
                                 "patch_len_samples, or set dft_size to treat all S steps as valid.")
            N = torch.full((B, P), self.S, dtype=torch.long, device=device)
        else:
            N = torch.as_tensor(patch_len_samples, device=device).long()
            if N.numel() == 1:
                N = N.view(1, 1).expand(B, P)
            elif N.numel() == B:
                N = N.view(B, 1).expand(B, P)
            else:
                N = N.reshape(B, P)
        return r, N

    def forward(self, patches, sampling_rate_hz, patch_len_samples=None) -> torch.Tensor:
        B, P, S, C = patches.shape
        device, dtype = patches.device, patches.dtype
        r, N = self._prep_rate_len(sampling_rate_hz, patch_len_samples, B, P, device, dtype)

        u = patches
        if self.standardize:
            if C != self.channel_group.numel():
                raise ValueError(f"standardize expects C={self.channel_group.numel()} channels "
                                 f"(channel_groups), got C={C}")
            mu = self.norm_mu[self.channel_group]          # (C,) per-modality
            sd = self.norm_sd[self.channel_group]          # (C,)
            u = (u - mu) / sd

        # Per-channel independent sequences with SHARED SSM weights: (B,P,S,C) -> (B*P*C, S, 1)
        seq = u.permute(0, 1, 3, 2).reshape(B * P * C, S, 1)                 # (M, S, 1)
        M = seq.shape[0]
        # valid-timestep mask (zero-pad beyond N): (B,P,1,C)->(M,S)
        idx = torch.arange(S, device=device).view(1, S)
        Nm = N.view(B, P, 1, 1).expand(B, P, 1, C).reshape(M, 1)
        valid = (idx < Nm).to(dtype)                                        # (M, S)
        # physical base step per sequence: dt = 1/rate (seconds/sample)
        dt_phys = (1.0 / r).view(B, 1, 1, 1).expand(B, P, 1, C).reshape(M, 1)  # (M,1)

        x = self.in_proj(seq)                                               # (M, S, E)
        z = self.gate_proj(seq)                                             # (M, S, E)
        # short causal conv over time (trim right padding), then SiLU
        xc = self.conv(x.transpose(1, 2))[..., :S].transpose(1, 2)
        x = F.silu(xc) * valid.unsqueeze(-1)

        # selective params (kept SMALL: (M,S,dt_rank+2N), never (M,S,E,N))
        proj = self.x_proj(x)
        dt_lr, Bm, Cm = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        A = -torch.exp(self.A_log)                                          # (E, N)

        # Sequential selective scan, computing the discretised (M,E,N) terms PER STEP. An earlier
        # version pre-materialised Abar/Bbar/Bx as (M,S,E,N) — ~39 GB at d_model=256 / batch 64, an
        # instant OOM. Per-step keeps the forward footprint at O(M·E·N). (Full-corpus pretraining
        # still wants a parallel scan / CUDA kernel for the backward graph; see the module docstring.)
        E, Nst = self.d_inner, self.d_state
        h = x.new_zeros(M, E, Nst)
        ys = []
        for t in range(S):
            dt_t = (F.softplus(self.dt_proj(dt_lr[:, t])) * dt_phys).unsqueeze(-1)   # (M,E,1) physical Δ
            Abar_t = torch.exp(dt_t * A.unsqueeze(0))                                # (M,E,N)
            Bx_t = dt_t * Bm[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1)              # (M,E,N)
            h = Abar_t * h + Bx_t
            y_t = (h * Cm[:, t].unsqueeze(1)).sum(-1) + self.D * x[:, t]             # (M,E)
            ys.append(y_t * valid[:, t:t + 1])
        y = torch.stack(ys, dim=1)                                                   # (M,S,E)
        y = y * F.silu(z)                                                   # gate

        # pool over valid timesteps -> one token per (channel, patch)
        if self.pool == "last":
            last = (valid.sum(1).clamp(min=1) - 1).long()                   # (M,)
            tok = y[torch.arange(M, device=device), last]                  # (M,E)
        else:
            denom = valid.sum(1, keepdim=True).clamp(min=1.0)
            tok = (y * valid.unsqueeze(-1)).sum(1) / denom                  # (M,E)
        tok = self.out_proj(tok)                                           # (M,d_model)
        return tok.reshape(B, P, C, self.d_model)
