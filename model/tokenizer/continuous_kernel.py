"""Continuous-time kernel tokenizer (CKT) — a rate-flexible, shape-preserving front end.

Design of record: `docs/design/CONTINUOUS_KERNEL_FRONTEND.md`. **Not yet wired into the encoder** —
this module is standalone and tested in isolation so it can be swapped in as a `--frontend` arm
without disturbing a running experiment.

WHAT IT IS
----------
A bank of `K` band-pass kernels, each defined as a CONTINUOUS function of real time and sampled at
whatever rate the recording arrives at. Kernel `k` spans `T_k` **seconds** and is parametrised in
normalised time `u = t/T_k` as a truncated Fourier series with `M` harmonics::

    w_k(u) = exp(-u^2 / 2 sigma_k^2) * sum_{m=1..M} [ a_km cos(2 pi m u) + b_km sin(2 pi m u) ]

`M = 8` gives **16 coefficients per kernel**. Three properties fall out of this basis, and each one
solves a problem a spline-through-control-points would leave open:

* harmonic `m` sits at exactly `m / T_k` Hz, so **band-limiting is a coefficient mask** rather than a
  smoothing heuristic — the kernel can never alias when sampled;
* there is no `m = 0` term, so `int w = 0` by construction and kernels **cannot measure gravity**
  (the signed DC component stays a separate feature, as in the filterbank);
* setting `a_k,carrier = 1` makes the kernel a Gabor wavelet, so **initialisation reproduces the
  physical filterbank** and any later difference is attributable to learning rather than to a
  different starting point.

WHY IT MIGHT BEAT THE FILTERBANK (and why it might not)
-------------------------------------------------------
The filterbank is already rate-invariant, by working in physical frequency. What it cannot represent
is *waveform shape within a band* (it keeps magnitude, discards phase) and *sub-second timing* (one
vector per patch). Three of the four motion scales that matter — impact transients ~60 ms, arm-swing
reversal ~150 ms, step ~500 ms — are invisible at a 1 s token boundary. This module keeps them by
emitting several ordered frames per token.

Against that: our own constrained-learnable filterbank arm measured **inert**, and a frozen
filterbank probe beat a raw-waveform CNN of the same budget. The prior is unfavourable, which is why
the design doc pre-registers a kill criterion on **cross-rate transfer** rather than accuracy.

THE THREE RULES THAT MAKE RATES COMPARABLE (all measured; see the design doc S12)
--------------------------------------------------------------------------------
1. **Evaluate the kernel at the exact offsets of the real samples.** For output time `t_f`, use
   `w_k((t_n - t_f) / T_k)` at the true sample times `t_n = n/r`. Rounding the frame centre to the
   nearest sample costs 25 ms of jitter at 20 Hz = 45 degrees of phase error at 5 Hz, and drops
   cross-rate correlation from 0.986 to 0.844. Never resample the signal; let the kernel absorb the
   sub-sample offset. That is the entire point of it being continuous.
2. **Convolution is an INTEGRAL: multiply by `dt = 1/r`.** A plain sum makes a 20 Hz recording return
   one fifth the response of the same physical motion at 100 Hz, and the model then learns to
   identify the device from response magnitude. L1 normalisation is equivalent; **L2 is actively
   harmful** (it rescales per rate and destroys amplitude comparability).
3. **Re-zero-mean after sampling.** Discretisation breaks the exact `int w = 0`.

Measured cross-rate agreement with all three applied, against 100 Hz: **0.986 at 20 Hz, 0.987 at
25 Hz, 0.998 at 50 Hz**. The residual is honest quadrature error (20 taps vs 100 taps approximating
the same integral), not a defect.

CONTRACT
--------
`forward(patches, sampling_rate_hz, patch_len_samples=None, source_rate_hz=None,
patch_mask=None) -> (B, P, C, d_model)` — identical to `PhysicalFilterbankTokenizer`, so it is a
drop-in swap. Token count is a function of DURATION only, never of sampling rate.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------
# Defaults — physically motivated, mirroring filterbank.py's constants where they correspond.
# ---------------------------------------------------------------------------------------------
CK_N_KERNELS = 32          # one per band; same count and log grid as FB_N_BANDS
CK_N_HARMONICS = 8         # -> 16 coefficients per kernel (8 cos + 8 sin)
CK_F_MIN_HZ = 0.3          # == FB_F_MIN_HZ; below this is quasi-DC, handled by the signed DC feature
CK_F_MAX_HZ = 15.0         # == FB_F_MAX_HZ; <= Nyquist of the lowest corpus rate (20 Hz)
CK_N_CYCLES = 4.0          # cycles of the centre frequency inside the span -> carrier at harmonic 4,
                           # so harmonics 1..8 span [f/4, 2f]: constant RELATIVE bandwidth (constant-Q
                           # arrived at from the time side rather than the frequency side)
CK_T_MIN_S = 0.05
CK_T_MAX_S = 2.0           # a 4 s kernel in a 6 s window is 67% edge-padded; 2.0 keeps two-thirds of
                           # frames fully supported for EVERY kernel and still reaches 2 cycles at 1 Hz
CK_FRAMES_PER_SECOND = 8   # output frame rate: 125 ms, resolves the arm-swing and step scales
CK_NYQUIST_MARGIN = 0.9    # == FB_NYQUIST_MARGIN
CK_ENVELOPE_SIGMA = 0.22   # Gaussian envelope width in normalised time
CK_FRAME_CHUNK = 8         # frames processed per gather; bounds peak memory, does not change results


class ContinuousKernelTokenizer(nn.Module):
    """Bank of continuous-time band-pass kernels + a small depthwise-separable stack."""

    def __init__(
        self,
        d_model: int = 128,
        n_kernels: int = CK_N_KERNELS,
        n_harmonics: int = CK_N_HARMONICS,
        f_min: float = CK_F_MIN_HZ,
        f_max: float = CK_F_MAX_HZ,
        n_cycles: float = CK_N_CYCLES,
        t_min: float = CK_T_MIN_S,
        t_max: float = CK_T_MAX_S,
        frames_per_second: int = CK_FRAMES_PER_SECOND,
        conv_channels: Tuple[int, int] = (64, 64),
        nyquist_margin: float = CK_NYQUIST_MARGIN,
        envelope_sigma: float = CK_ENVELOPE_SIGMA,
        gabor_init: bool = True,
        norm: str = "frozen",
    ):
        super().__init__()
        if n_kernels < 2 or n_harmonics < 1:
            raise ValueError("need at least 2 kernels and 1 harmonic")
        if not 0.0 < t_min < t_max:
            raise ValueError("require 0 < t_min < t_max")
        if frames_per_second < 2 or frames_per_second % 2:
            raise ValueError("frames_per_second must be even and >= 2 (one stride-2 stage)")
        self.K = int(n_kernels)
        self.M = int(n_harmonics)
        self.F = int(frames_per_second)
        self.d_model = int(d_model)
        self.nyquist_margin = float(nyquist_margin)
        self.envelope_sigma = float(envelope_sigma)
        self.norm = norm

        # --- band centres and spans (fixed physics, not learned) ---
        k = torch.arange(self.K, dtype=torch.float32)
        centres = f_min * (f_max / f_min) ** (k / (self.K - 1))
        spans = torch.clamp(n_cycles / centres, t_min, t_max)
        self.register_buffer("centres", centres)                 # (K,) Hz
        self.register_buffer("spans", spans)                     # (K,) seconds
        # carrier harmonic: the one nearest f_k within this kernel's own span
        carrier = torch.clamp(torch.round(centres * spans), 1, self.M).long()
        self.register_buffer("carrier", carrier)                 # (K,)

        # --- the 16 numbers per kernel, plus envelope width and gain ---
        self.cos_coeff = nn.Parameter(torch.zeros(self.K, self.M))
        self.sin_coeff = nn.Parameter(torch.zeros(self.K, self.M))
        self.log_sigma = nn.Parameter(torch.full((self.K,), math.log(envelope_sigma)))
        self.log_gain = nn.Parameter(torch.zeros(self.K))
        if gabor_init:
            # A Gabor wavelet at the carrier: the quadrature pair is (cos, sin) of that harmonic,
            # so at step 0 this bank approximates the physical filterbank we already trust.
            with torch.no_grad():
                self.cos_coeff[torch.arange(self.K), carrier - 1] = 1.0
        else:
            nn.init.normal_(self.cos_coeff, std=0.3)
            nn.init.normal_(self.sin_coeff, std=0.3)

        # --- frozen per-kernel standardisation of the compressed magnitude ---
        self.register_buffer("norm_mu", torch.zeros(self.K))
        self.register_buffer("norm_sd", torch.ones(self.K))
        self.register_buffer("_norm_fitted", torch.zeros(1))
        self.register_buffer("_acc_count", torch.zeros(self.K, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sum", torch.zeros(self.K, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sqsum", torch.zeros(self.K, dtype=torch.float64), persistent=False)

        # --- depthwise-separable stack on the fixed real-time grid (see design doc S10) ---
        c1, c2 = conv_channels
        self.dw1 = nn.Conv1d(self.K, self.K, 3, stride=2, padding=1, groups=self.K)
        self.pw1 = nn.Conv1d(self.K, c1, 1)
        self.gn1 = nn.GroupNorm(1, c1)
        self.dw2 = nn.Conv1d(c1, c1, 3, stride=1, padding=1, groups=c1)
        self.pw2 = nn.Conv1d(c1, c2, 1)
        self.gn2 = nn.GroupNorm(1, c2)
        self.frames_per_token = self.F // 2                      # after the single stride-2
        self.in_dim = (c2 * self.frames_per_token   # ORDERED frames — the point of the design
                       + self.K                     # nyquist mask
                       + self.K                     # resolution flag
                       + self.frames_per_token      # per-frame edge support
                       + 1                          # amplitude
                       + 1)                         # signed DC
        self.proj = nn.Linear(self.in_dim, self.d_model)

    # ------------------------------------------------------------------ kernel construction
    def kernel_at(self, offsets: torch.Tensor, rate_hz: float) -> torch.Tensor:
        """Evaluate every kernel at exact time offsets in SECONDS -> (K, 2, n) quadrature pair.

        `offsets` is (n,) seconds relative to an output-frame centre. Returns cos-phase and
        sin-phase kernels so the magnitude |z| is phase-invariant like a band energy.
        """
        spans = self.spans.to(offsets.device)                     # (K,)
        u = offsets.unsqueeze(0) / spans.unsqueeze(1)             # (K, n) normalised time
        inside = u.abs() <= 0.5
        m = torch.arange(1, self.M + 1, device=offsets.device, dtype=offsets.dtype)  # (M,)
        phase = 2 * math.pi * m.view(1, self.M, 1) * u.unsqueeze(1)                  # (K, M, n)
        # Rule 1 of the rate contract: per-harmonic band-limit. Harmonic m sits at m/T_k Hz, so
        # this is an exact mask on a coefficient, never a smoothing approximation.
        limit = self.nyquist_margin * rate_hz / 2.0
        live = (m.view(1, self.M) / spans.unsqueeze(1)) <= limit                     # (K, M)
        cos_c = self.cos_coeff.to(offsets.dtype) * live
        sin_c = self.sin_coeff.to(offsets.dtype) * live
        cos_k = (cos_c.unsqueeze(-1) * torch.cos(phase)).sum(1)
        sin_k = (sin_c.unsqueeze(-1) * torch.cos(phase)).sum(1)
        # quadrature partner: same coefficients, sine basis
        cos_q = (cos_c.unsqueeze(-1) * torch.sin(phase)).sum(1)
        sin_q = (sin_c.unsqueeze(-1) * torch.sin(phase)).sum(1)
        real = cos_k + sin_q
        imag = sin_k - cos_q
        sigma = self.log_sigma.exp().to(offsets.dtype).unsqueeze(1).clamp(0.05, 1.0)
        env = torch.exp(-u.pow(2) / (2 * sigma.pow(2)))
        real, imag = real * env * inside, imag * env * inside
        pair = torch.stack((real, imag), dim=1)                                       # (K, 2, n)
        # Rule 3: re-zero-mean after sampling (discretisation breaks the exact integral).
        count = inside.sum(-1, keepdim=True).clamp_min(1).unsqueeze(1)
        pair = (pair - pair.sum(-1, keepdim=True) / count) * inside.unsqueeze(1)
        # Rule 2: convolution is an INTEGRAL. dt = 1/r. Never L2-normalise here.
        gain = self.log_gain.exp().to(offsets.dtype).view(self.K, 1, 1)
        return pair * gain / float(rate_hz)

    # ------------------------------------------------------------------ masks
    def masks(self, rate_hz: float, duration_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(nyquist observability, resolution flag), each (B, K).

        `nyquist`: fraction of this kernel's harmonics representable at `rate_hz`.
        `resolution`: how much of the kernel's span the recording can actually supply.
        """
        spans = self.spans
        m = torch.arange(1, self.M + 1, device=spans.device, dtype=spans.dtype)
        live = (m.view(1, self.M) / spans.unsqueeze(1)) <= (self.nyquist_margin * rate_hz / 2.0)
        nyq = live.to(spans.dtype).mean(dim=1)                                  # (K,)
        res = (duration_s.unsqueeze(1) / spans.unsqueeze(0)).clamp(0.0, 1.0)    # (B, K)
        return nyq.unsqueeze(0).expand_as(res), res

    # ------------------------------------------------------------------ window reconstruction
    @staticmethod
    def contiguous_window(patches: torch.Tensor, patch_len: torch.Tensor,
                          patch_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B,P,S,C) padded patches -> (B, T, C) contiguous signal + (B,) valid lengths.

        Padding slots and heterogeneous rates make a naive `reshape(B, P*S, C)` wrong twice over: it
        interleaves a 20 Hz item's real samples with the zero padding sized for a 100 Hz item in the
        same batch. Index the VALID samples instead. (Same failure and same fix as
        `model/tokenizer/baseline_backbone.py`.)
        """
        B, P, S, C = patches.shape
        valid = patch_len.clamp_min(0)
        if patch_mask is not None:
            valid = valid * patch_mask.long()
        cumulative = torch.cat([valid.new_zeros((B, 1)), valid.cumsum(dim=1)], dim=1).float()
        total = cumulative[:, -1]
        width = int(total.max().item()) if total.numel() else 0
        width = max(width, 1)
        position = torch.arange(width, device=patches.device, dtype=torch.float32).view(1, -1)
        inside = position < total.unsqueeze(1)
        index = torch.minimum(position, (total - 1).clamp_min(0).unsqueeze(1))
        patch_of = (torch.searchsorted(cumulative, index.contiguous(), right=True) - 1).clamp(0, P - 1)
        offset = (index - cumulative.gather(1, patch_of)).long().clamp_min(0)
        offset = torch.minimum(offset, (patch_len.gather(1, patch_of) - 1).clamp_min(0))
        flat = patches.reshape(B, P * S, C)
        window = flat.gather(1, (patch_of * S + offset).unsqueeze(-1).expand(-1, -1, C))
        return window * inside.unsqueeze(-1).to(window.dtype), total

    # ------------------------------------------------------------------ the analysis stage
    def analyze(self, patches, sampling_rate_hz, patch_len_samples=None,
                source_rate_hz=None, patch_mask=None) -> dict:
        """Per-frame band magnitudes and the masks. Returns a dict; `project` turns it into tokens."""
        B, P, S, C = patches.shape
        device = patches.device
        rate = float(torch.as_tensor(sampling_rate_hz).reshape(-1)[0].item())
        if patch_len_samples is None:
            patch_len_samples = torch.full((B, P), S, dtype=torch.long, device=device)
        window, total = self.contiguous_window(patches, patch_len_samples, patch_mask)
        duration = total / max(rate, 1e-6)                                       # (B,) seconds
        n_frames = int(P * self.F)                                               # duration-only
        half = int(math.ceil(self.spans.max().item() * rate / 2.0)) + 2
        # reflection padding: a zero edge is a step discontinuity to a band-pass kernel and injects
        # broadband energy the recording does not contain.
        pad_width = min(half, max(window.shape[1] - 1, 0))
        signal = window.permute(0, 2, 1)                                         # (B, C, T)
        if pad_width > 0:
            signal = F.pad(signal, (pad_width, pad_width), mode="reflect")
        elif half > 0:
            signal = F.pad(signal, (half, half), mode="replicate")
            pad_width = half

        taps = 2 * half + 1
        offsets_base = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
        responses, edge = [], []
        for start in range(0, n_frames, CK_FRAME_CHUNK):
            stop = min(start + CK_FRAME_CHUNK, n_frames)
            idx = torch.arange(start, stop, device=device, dtype=torch.float32)
            t_f = (idx + 0.5) / self.F                                           # exact, seconds
            centre = torch.floor(t_f * rate)                                     # index anchor only
            # Rule 1 (the important one): offsets are EXACT; the kernel absorbs the sub-sample shift.
            offs = (centre.unsqueeze(1) + offsets_base.unsqueeze(0)) / rate - t_f.unsqueeze(1)
            gather_idx = (centre.unsqueeze(1) + offsets_base.unsqueeze(0)).long() + pad_width
            gather_idx = gather_idx.clamp(0, signal.shape[-1] - 1)
            seg = signal[:, :, gather_idx.reshape(-1)].reshape(B, C, stop - start, taps)
            with torch.autocast(device_type=device.type, enabled=False):
                kern = torch.stack([self.kernel_at(offs[j].float(), rate)
                                    for j in range(offs.shape[0])], dim=0)       # (f, K, 2, taps)
                z = torch.einsum("bcft,fkqt->bckfq", seg.float(), kern)          # (B,C,K,f,2)
            responses.append(z.pow(2).sum(-1).clamp_min(1e-12).sqrt())           # magnitude |z|
            frame_edge = ((t_f.unsqueeze(0) >= 0) & (t_f.unsqueeze(0) <= duration.unsqueeze(1)))
            edge.append(frame_edge.to(torch.float32))
        magnitude = torch.cat(responses, dim=3)                                  # (B, C, K, n_frames)
        edge_support = torch.cat(edge, dim=1)                                    # (B, n_frames)
        compressed = torch.log1p(magnitude)
        nyq, res = self.masks(rate, duration)
        return {"compressed": compressed, "nyquist": nyq, "resolution": res,
                "edge": edge_support, "window": window, "duration": duration,
                "n_patches": P, "rate": rate}

    # ------------------------------------------------------------------ normalisation stats
    def reset_norm_accumulator(self):
        self._acc_count.zero_(); self._acc_sum.zero_(); self._acc_sqsum.zero_()

    @torch.no_grad()
    def accumulate_norm_stats(self, patches, sampling_rate_hz, patch_len_samples=None,
                              patch_mask=None, channel_mask=None, source_rate_hz=None):
        out = self.analyze(patches, sampling_rate_hz, patch_len_samples,
                           source_rate_hz=source_rate_hz, patch_mask=patch_mask)
        value = out["compressed"]                                                # (B,C,K,f)
        if channel_mask is not None:
            value = value * channel_mask.to(value.dtype).view(value.shape[0], -1, 1, 1)
            weight = channel_mask.to(torch.float64).sum() * value.shape[-1]
        else:
            weight = torch.tensor(float(value.shape[0] * value.shape[1] * value.shape[-1]),
                                  dtype=torch.float64)
        flat = value.double().permute(2, 0, 1, 3).reshape(self.K, -1)
        self._acc_sum += flat.sum(dim=1)
        self._acc_sqsum += flat.pow(2).sum(dim=1)
        self._acc_count += torch.full_like(self._acc_count, float(flat.shape[1]))

    @torch.no_grad()
    def finalize_norm_stats(self, eps: float = 1e-5):
        count = self._acc_count.clamp_min(1.0)
        mu = self._acc_sum / count
        var = (self._acc_sqsum / count - mu.pow(2)).clamp_min(0.0)
        self.norm_mu.copy_(mu.float())
        self.norm_sd.copy_(var.sqrt().float().clamp_min(eps))
        self._norm_fitted.fill_(1.0)

    # ------------------------------------------------------------------ projection to tokens
    def project(self, analysis: dict) -> torch.Tensor:
        compressed = analysis["compressed"]                                      # (B,C,K,f)
        B, C, K, n_frames = compressed.shape
        P = analysis["n_patches"]
        if self.norm == "frozen":
            compressed = (compressed - self.norm_mu.view(1, 1, K, 1)) / self.norm_sd.view(1, 1, K, 1)
        x = compressed.reshape(B * C, K, n_frames)
        x = self.pw1(self.dw1(x)); x = F.gelu(self.gn1(x))
        x = self.pw2(self.dw2(x)); x = F.gelu(self.gn2(x))                       # (B*C, c2, f/2)
        c2, reduced = x.shape[1], x.shape[2]
        per_token = max(reduced // P, 1)
        x = x[:, :, :per_token * P].reshape(B, C, c2, P, per_token)
        # ORDERED flatten: a rising and a falling ramp must be different inputs. An order-invariant
        # pool (mean/max/std) cannot distinguish them, which would re-destroy the sub-second
        # structure this front end exists to keep.
        ordered = x.permute(0, 3, 1, 2, 4).reshape(B, P, C, c2 * per_token)
        if ordered.shape[-1] < self.in_dim - (2 * K + self.frames_per_token + 2):
            pad = self.in_dim - (2 * K + self.frames_per_token + 2) - ordered.shape[-1]
            ordered = F.pad(ordered, (0, pad))
        nyq = analysis["nyquist"].view(B, 1, 1, K).expand(B, P, C, K)
        res = analysis["resolution"].view(B, 1, 1, K).expand(B, P, C, K)
        edge = analysis["edge"]
        edge = edge[:, :per_token * P].reshape(B, P, per_token) if edge.shape[1] >= per_token * P \
            else F.pad(edge, (0, per_token * P - edge.shape[1])).reshape(B, P, per_token)
        edge = edge.unsqueeze(2).expand(B, P, C, per_token)
        if edge.shape[-1] < self.frames_per_token:
            edge = F.pad(edge, (0, self.frames_per_token - edge.shape[-1]))
        edge = edge[..., :self.frames_per_token]
        window = analysis["window"]                                              # (B,T,C)
        amplitude = window.abs().mean(dim=1).view(B, 1, C, 1).expand(B, P, C, 1)
        dc = window.mean(dim=1).view(B, 1, C, 1).expand(B, P, C, 1)
        features = torch.cat([ordered[..., :self.in_dim - (2 * K + self.frames_per_token + 2)],
                              nyq, res, edge, amplitude, dc], dim=-1)
        return self.proj(features)

    def forward(self, patches, sampling_rate_hz, patch_len_samples=None,
                source_rate_hz=None, patch_mask=None) -> torch.Tensor:
        return self.project(self.analyze(patches, sampling_rate_hz, patch_len_samples,
                                         source_rate_hz=source_rate_hz, patch_mask=patch_mask))
