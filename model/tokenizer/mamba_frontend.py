"""Per-channel STACKED selective-SSM (Mamba) tokenizer with physical Δ = 1/rate.

The learned-recurrent alternative to the physical-Hz filterbank, for the tokenizer ablation
(docs/design/TOKENIZER_ABLATION.md). Same drop-in contract as ``PhysicalFilterbankTokenizer``:

    forward(patches (B,P,S,C) native-rate zero-padded, sampling_rate_hz, patch_len_samples)
        -> tokens (B, P, C, d_model)

**Depth.** Each channel's patch is processed by ``n_layers`` residual Mamba blocks (norm → in_proj →
short conv → selective scan → gate → out_proj, with a residual), then pooled to one token. A single
block is a weak extractor (~one adaptive filter + gate); stacking gives hierarchical, nonlinear
intra-patch features (local oscillation → cycle → segment). Inter-patch temporal modelling is still
the downstream transformer's job — this is the *intra-patch* feature extractor.

**The physics.** An SSM is a discretised continuous ODE ``dh/dt = A h + B u``; its step Δ is the
physical time between samples. Each block inits Δ to ``1/rate``, giving the recurrence a rate-aware
inductive bias (same motion at 20/100 Hz advances the state at the same physical speed). This is a
learned bias seeded by the Δ init, NOT structural invariance — the fixed-tap conv and the D skip are
not rate-invariant (see docs/design/TOKENIZER_ABLATION.md #4).

**Per channel, shared weights.** The stack is applied independently per channel (weights shared
across channels, like the filterbank shares one bank), so channel count/order stay free and identity
comes from text downstream. Gravity is preserved: standardisation is frozen, **per-modality** (accel
vs gyro differ ~1.6× in scale), pooled within each modality's axes so relative gravity survives — NOT
per-window instance norm (which would erase the DC/gravity that separates static postures).

**Scan.** The fused ``mamba_ssm`` CUDA kernel is used when available (fast training path); a portable
pure-PyTorch chunked, gradient-checkpointed scan is the CPU/test fallback (the dual the mamba repo
ships). The two are numerically identical.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Official fused selective-scan kernel (state-spaces/mamba). CUDA-only; when absent (CPU / not
# installed) the pure-PyTorch reference scan below is used — the same dual the mamba repo ships.
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_fn
    _HAS_KERNEL = True
except Exception:      # pragma: no cover - import guard
    _selective_scan_fn = None
    _HAS_KERNEL = False


class _MambaBlock(nn.Module):
    """One residual Mamba block over (M, S, d_model). Selective SSM at inner width d_inner, Δ=1/rate."""

    def __init__(self, d_model: int, d_inner: int, d_state: int, d_conv: int, dt_rank: int,
                 mult_min: float, mult_max: float, scan_chunk: int):
        super().__init__()
        self.d_inner, self.d_state, self.dt_rank, self.scan_chunk = d_inner, d_state, dt_rank, scan_chunk
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)          # -> (x, z)
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)   # -> Δ_lr, B, C
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)   # S4D-real init
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)
        # Δ = (1/rate)·softplus(dt_proj(·)); init the multiplier baseline ∈ [mult_min, mult_max] (~1)
        m0 = torch.exp(torch.rand(d_inner) * (math.log(mult_max) - math.log(mult_min)) + math.log(mult_min))
        self.dt_proj.bias.data.copy_(torch.log(torch.expm1(m0)))

    def _scan_chunk(self, h, x_c, dtlr_c, Bm_c, Cm_c, dt_phys, valid_c, A):
        """Selective recurrence over one chunk of timesteps; carry h in, (h_out, y_chunk) out."""
        L = x_c.shape[1]
        ys = []
        for t in range(L):
            dt_t = (F.softplus(self.dt_proj(dtlr_c[:, t])) * dt_phys).unsqueeze(-1)   # (M,E,1) physical Δ
            Abar_t = torch.exp(dt_t * A.unsqueeze(0))                                 # (M,E,N)
            Bx_t = dt_t * Bm_c[:, t].unsqueeze(1) * x_c[:, t].unsqueeze(-1)           # (M,E,N)
            h = Abar_t * h + Bx_t
            y_t = (h * Cm_c[:, t].unsqueeze(1)).sum(-1) + self.D * x_c[:, t]          # (M,E)
            ys.append(y_t * valid_c[:, t:t + 1])
        return h, torch.stack(ys, dim=1)

    def forward(self, u, dt_phys, valid, use_kernel, allow_inner_ckpt=True):
        # u (M,S,d_model) residual stream; dt_phys (M,1); valid (M,S)
        M, S, _ = u.shape
        residual = u
        x = self.norm(u)
        x, z = self.in_proj(x).chunk(2, dim=-1)                              # (M,S,E) each
        xc = self.conv(x.transpose(1, 2))[..., :S].transpose(1, 2)           # causal short conv
        x = F.silu(xc) * valid.unsqueeze(-1)
        dt_lr, Bm, Cm = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        A = -torch.exp(self.A_log)                                           # (E,N)

        if use_kernel and _HAS_KERNEL and x.is_cuda:
            # Fused kernel: our physics is entirely in Δ (rate-scaled) -> delta_softplus=False. The
            # kernel applies the silu(z) gate and the D skip internally. Layout: (M,E,S)/(M,N,S).
            delta = F.softplus(self.dt_proj(dt_lr)) * dt_phys.unsqueeze(1)               # (M,S,E)
            y = _selective_scan_fn(
                x.transpose(1, 2).contiguous(), delta.transpose(1, 2).contiguous(), A,
                Bm.transpose(1, 2).contiguous(), Cm.transpose(1, 2).contiguous(),
                self.D, z=z.transpose(1, 2).contiguous(), delta_softplus=False,
            ).transpose(1, 2)                                                            # (M,S,E), gated
        else:
            # Portable chunked, gradient-checkpointed reference scan (CPU/tests). Backward stays
            # O(chunk·M·E·N); numerically matches the kernel.
            h = x.new_zeros(M, self.d_inner, self.d_state)
            do_ckpt = (allow_inner_ckpt and self.training
                       and torch.is_grad_enabled() and x.requires_grad)
            y_chunks = []
            for s0 in range(0, S, self.scan_chunk):
                s1 = min(s0 + self.scan_chunk, S)
                args = (h, x[:, s0:s1], dt_lr[:, s0:s1], Bm[:, s0:s1], Cm[:, s0:s1],
                        dt_phys, valid[:, s0:s1], A)
                if do_ckpt:
                    h, yc = torch.utils.checkpoint.checkpoint(self._scan_chunk, *args, use_reentrant=False)
                else:
                    h, yc = self._scan_chunk(*args)
                y_chunks.append(yc)
            y = torch.cat(y_chunks, dim=1) * F.silu(z)                                   # (M,S,E), gated
        return residual + self.out_proj(y * valid.unsqueeze(-1))            # (M,S,d_model)

    def delta_mult_baseline(self) -> torch.Tensor:
        return F.softplus(self.dt_proj.bias)


class SelectiveSSMChannelTokenizer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 3,              # stacked residual Mamba blocks per channel (intra-patch depth)
        d_state: int = 16,
        d_inner: int | None = None,     # SSM inner width per block; default 2*d_model
        d_conv: int = 4,                # short causal depthwise conv (local mixing)
        dt_rank: int | None = None,     # low-rank Δ projection; default ceil(d_inner/16)
        mult_min: float = 0.5,          # bounds of the dimensionless learned Δ multiplier init (~1)
        mult_max: float = 2.0,
        standardize: bool = True,       # frozen PER-MODALITY (mu,sd); preserves gravity DC
        channel_groups: tuple = (0, 0, 0, 1, 1, 1),  # channel -> modality group (accel=0, gyro=1)
        pool: str = "mean",            # 'mean' over valid steps | 'last' valid state
        scan_chunk: int = 32,           # timesteps per gradient-checkpointed scan chunk (ref path only)
        forward_chunk: int = 0,         # max per-channel sequences (M=B*P*C) processed at once; bounds
                                        # peak memory independent of caller batch. 0 = auto-scale from
                                        # d_inner*S (~2.5 GiB/chunk); >0 pins it. (4096 OOMs at d=256.)
        use_kernel: bool = True,        # use the fused CUDA kernel when available (else the ref scan)
        dft_size: int | None = None,    # accepted+ignored: drop-in kwarg parity with the filterbank
        **_ignored,                     # tolerate filterbank-only kwargs so build_frontend can pass through
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = int(n_layers)
        self.d_state = int(d_state)
        self.d_inner = int(d_inner or 2 * d_model)
        self.d_conv = int(d_conv)
        self.dt_rank = int(dt_rank or math.ceil(self.d_inner / 16))
        self.pool = pool
        self.forward_chunk = int(forward_chunk)
        self.use_kernel = bool(use_kernel)
        self.S = int(dft_size) if dft_size is not None else None   # for the drop-in length assert only

        self.stem = nn.Linear(1, d_model)                          # lift scalar signal -> residual width
        self.blocks = nn.ModuleList(
            _MambaBlock(d_model, self.d_inner, self.d_state, self.d_conv, self.dt_rank,
                        mult_min, mult_max, int(scan_chunk))
            for _ in range(self.n_layers))
        self.final_norm = nn.LayerNorm(d_model)
        self.learnable = True           # frontend-interface parity with the filterbank arm

        # Frozen PER-MODALITY standardisation (accel/gyro separate scale; preserves relative gravity).
        self.standardize = bool(standardize)
        cg = torch.tensor(channel_groups, dtype=torch.long)
        self.register_buffer("channel_group", cg)
        self.n_groups = int(cg.max().item()) + 1
        self.register_buffer("norm_mu", torch.zeros(self.n_groups))
        self.register_buffer("norm_sd", torch.ones(self.n_groups))
        self.register_buffer("_norm_fitted", torch.zeros(1))
        self.register_buffer("_acc_n", torch.zeros(self.n_groups, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sum", torch.zeros(self.n_groups, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sqsum", torch.zeros(self.n_groups, dtype=torch.float64), persistent=False)

    def get_output_dim(self) -> int:
        return self.d_model

    # ------------------------------------------------ frontend interface (parity with the filterbank)
    def adaptation_regularization(self) -> torch.Tensor:
        """Soft pull of every block's Δ-multiplier baseline toward 1 (the physical step 1/rate),
        so training keeps the physical clock near-honest (audit #5). Dimensionless."""
        terms = [b.delta_mult_baseline().clamp_min(1e-6).log().square().mean() for b in self.blocks]
        return torch.stack(terms).mean()

    @torch.no_grad()
    def adaptation_summary(self) -> dict[str, float]:
        base = torch.cat([b.delta_mult_baseline().detach() for b in self.blocks])
        return {
            "frontend/delta_mult_baseline_min": float(base.min()),
            "frontend/delta_mult_baseline_max": float(base.max()),
            "frontend/delta_mult_baseline_mean": float(base.mean()),
            "frontend/norm_sd_accel": float(self.norm_sd[0]) if self.norm_sd.numel() else 1.0,
            "frontend/norm_sd_gyro": float(self.norm_sd[-1]) if self.norm_sd.numel() > 1 else 1.0,
        }

    # ------------------------------------------------------------------ calibration (like filterbank)
    def reset_norm_accumulator(self):
        self._acc_n.zero_(); self._acc_sum.zero_(); self._acc_sqsum.zero_()

    @torch.no_grad()
    def accumulate_norm_stats(self, patches, sampling_rate_hz=None, patch_len_samples=None,
                              patch_mask=None, channel_mask=None):
        """Fold one batch into the PER-MODALITY standardisation stats over REAL samples only."""
        B, P, S, C = patches.shape
        if C != self.channel_group.numel():
            raise ValueError(f"channel_groups has {self.channel_group.numel()} entries but C={C}")
        dev = patches.device                                  # keep every factor on the patch device
        w = torch.ones(B, P, S, C, dtype=torch.float64, device=dev)  # was CPU -> mismatch on a CUDA run
        if patch_len_samples is not None:
            N = torch.as_tensor(patch_len_samples).to(dev)
            if N.ndim == 0:
                N = N.view(1, 1).expand(B, P)
            elif N.ndim == 1:
                N = N.view(B, 1).expand(B, P)
            idx = torch.arange(S, device=dev).view(1, 1, S)
            w = w * (idx < N.unsqueeze(-1)).view(B, P, S, 1).to(torch.float64)
        if patch_mask is not None:
            w = w * patch_mask.to(dev).view(B, P, 1, 1).to(torch.float64)
        if channel_mask is not None:
            w = w * channel_mask.to(dev).view(B, 1, 1, C).to(torch.float64)
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

    def _auto_chunk(self, S: int) -> int:
        """Per-chunk sequence count that keeps the fused kernel's peak ~constant across widths.
        Measured peak is ~linear in chunk*d_inner*S (RTX 4090, S=256: chunk=256,d_inner=512 -> 5.4 GiB;
        chunk=256,d_inner=128 -> 1.37 GiB). Budget ~2.5 GiB so the frontend leaves headroom for the
        transformer/heads in a full step. forward_chunk>0 overrides this."""
        d_inner = self.blocks[0].d_inner
        return max(64, 16_000_000 // (d_inner * max(int(S), 1)))

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
            sd = self.norm_sd[self.channel_group]
            u = (u - mu) / sd

        # Per-channel independent sequences with SHARED stack weights: (B,P,S,C) -> (B*P*C, S, 1)
        seq = u.permute(0, 1, 3, 2).reshape(B * P * C, S, 1)                 # (M, S, 1)
        M = seq.shape[0]
        idx = torch.arange(S, device=device).view(1, S)
        Nm = N.view(B, P, 1, 1).expand(B, P, 1, C).reshape(M, 1)
        valid = (idx < Nm).to(dtype)                                        # (M, S)
        dt_phys = (1.0 / r).view(B, 1, 1, 1).expand(B, P, 1, C).reshape(M, 1)  # (M,1) physical step

        # Each of the M=B*P*C sequences is INDEPENDENT, so process M in chunks: peak memory is bounded
        # by the chunk size regardless of the caller's batch (eval embeds B=256 blocks -> M~24k; the
        # real train batch gives M~45k at d_model=256 -> the fused kernel's per-chunk intermediates are
        # ~linear in chunk*d_inner*S and would OOM in one shot). Math is chunk-invariant (verified
        # bit-exact). The chunk auto-scales with d_inner*S so peak stays ~constant (~3 GiB) across
        # widths; forward_chunk>0 overrides. (Independent audit 2026-07-23: fc=4096 default was ~21 GiB
        # at d=256/fc=1024 -> OOM; measured peak is linear in chunk, so the budget picks the chunk.)
        chunk = self.forward_chunk if self.forward_chunk > 0 else self._auto_chunk(S)
        chunk = min(chunk, M)
        # Gate on train+grad ONLY, NOT seq.requires_grad: raw sensor inputs have requires_grad=False,
        # so gating on the input would DISABLE checkpointing in the real training path (the chunk's
        # graph is kept for the frontend PARAMETERS, not the input). use_reentrant=False correctly
        # differentiates the parameters even when no input requires grad. (Independent audit 2026-07-23:
        # the old gate never fired -> 21 GiB backward at tiny scale; verified fixed with a real step.)
        do_ckpt = self.training and torch.is_grad_enabled() and any(
            p.requires_grad for p in self.parameters())
        toks = []
        for s0 in range(0, M, chunk):
            s1 = min(s0 + chunk, M)
            sc, dc, vc = seq[s0:s1], dt_phys[s0:s1], valid[s0:s1]
            if do_ckpt:                        # bound TRAIN memory too: one chunk's activations live
                tok_c = torch.utils.checkpoint.checkpoint(
                    self._encode_chunk, sc, dc, vc, use_reentrant=False)
            else:                              # eval/no-grad: chunking alone bounds the forward working set
                tok_c = self._encode_chunk(sc, dc, vc)
            toks.append(tok_c)
        tok = torch.cat(toks, dim=0)                                        # (M,d_model)
        return tok.reshape(B, P, C, self.d_model)

    def _encode_chunk(self, seq, dt_phys, valid) -> torch.Tensor:
        """stem -> blocks -> final_norm -> pool over one chunk of M sequences. Returns (m, d_model).
        Blocks skip their own inner scan-checkpoint here (allow_inner_ckpt=False): when this whole
        chunk is wrapped in a checkpoint, the outer recompute already provides the memory bound, and
        nesting would double the recompute on the ref path."""
        h = self.stem(seq) * valid.unsqueeze(-1)                            # (m,S,d_model)
        for blk in self.blocks:
            h = blk(h, dt_phys, valid, self.use_kernel, allow_inner_ckpt=False)
        h = self.final_norm(h) * valid.unsqueeze(-1)
        if self.pool == "last":
            last = (valid.sum(1).clamp(min=1) - 1).long()
            return h[torch.arange(h.shape[0], device=h.device), last]
        denom = valid.sum(1, keepdim=True).clamp(min=1.0)
        return (h * valid.unsqueeze(-1)).sum(1) / denom                     # (m,d_model)
