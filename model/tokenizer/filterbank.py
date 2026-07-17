"""Physical-Hz constant-Q filterbank tokenizer (M1 port).

Ported verbatim from `legacy_code/model/feature_extractor.py` (the battle-tested V2
tokenizer, with its unit-test suite at `tests/test_filterbank.py`) into the
evidence-engine Pipeline A front end — see docs/design/EVIDENCE_ENGINE_BUILD_PLAN.md
(M1). Key hyperparameters and the physics that pins each value are collected below
so they are easy to find and hard to second-guess.

`learnable=True` is the constrained-learnable arm (SincNet-style: band centers move
inside (f_min, f_max), everything else fixed). Select frontends via
`model.tokenizer.scattering.build_frontend` — the fixed filterbank is the default
until an ablation earns a switch.
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

# ============================================================================
# PhysicalFilterbankTokenizer — default hyperparameters (justified)
# ============================================================================
# These are physically motivated, not tuned knobs. Change them in config.py for a
# run; change the DEFAULTS here only with a reason that supersedes the ones below.
#
# FB_F_MIN_HZ / FB_F_MAX_HZ — the analysis band, in physical Hz (constant, rate-
#   independent). Human activity energy lives ~0.3–15 Hz: gait cadence ~0.5–3 Hz,
#   harmonics + hand/limb motion up to ~10–12 Hz. 15 Hz is also <= the Nyquist of
#   the lowest native rate in the corpus (20 Hz -> 10 Hz Nyquist; bands above that
#   are simply Nyquist-masked per-sample, so 15 Hz costs nothing at low rates and
#   captures the extra content at 50/100 Hz). Below 0.3 Hz is quasi-DC (gravity/
#   tilt), which is handled separately by the signed DC feature, not a band.
FB_F_MIN_HZ = 0.3
FB_F_MAX_HZ = 15.0
#
# FB_N_BANDS — number of log-spaced constant-Q bands across [f_min, f_max].
#   Log spacing matches how HAR discriminability scales (fine at low freq where
#   cadence lives, coarse up high). 32 is generous; at Q=4 and ~1.5 s windows the
#   top bands overlap / are resolution-limited, so 16–24 is likely equivalent and
#   cheaper — but that is an ABLATION (measure before cutting), so the default
#   stays at the safe, higher value.
FB_N_BANDS = 32
#
# FB_Q — constant-Q quality factor (center / bandwidth). Q=4 gives ~1/4-octave
#   bands: selective enough to separate gait harmonics, wide enough that a band
#   still contains signal energy in a short (~1–1.5 s) window. Standard for
#   audio/vibration constant-Q analysis.
FB_Q = 4.0
#
# FB_DFT_SIZE (S) — the fixed zero-pad/rDFT length. MUST satisfy S >= max patch
#   length N = sampling_rate * patch_seconds over the whole corpus, because the
#   filterbank matrix is built for exactly S//2+1 bins (one fixed shape so any
#   patch length maps onto one shared filterbank; batched). Zero-padding beyond N
#   is pure frequency-domain interpolation — it does NOT change the band-energy
#   output (verified: S=256 vs 512 give band-cosine 1.000000 and identical Nyquist/
#   resolution masks at 20/25/50/100 Hz). So S only trades compute vs headroom.
#   Corpus worst case: 2.5 s (dsads) * 100 Hz = 250 samples; resample rounding adds
#   <1. 256 (smallest power of two >= 250) covers it with margin and is ~2x cheaper
#   than 512 in the tokenizer hot path (rDFT is ~S log S: measured 1.1 ms vs 2.6 ms
#   per batch). Overflow is a hard ValueError (never silent truncation), so if a
#   future patch exceeds this the run fails loudly — raise S then. Keep it a power
#   of two for FFT efficiency.
FB_DFT_SIZE = 256
#
# FB_NYQUIST_MARGIN — a band is "observable" only if center + 2*sigma <= margin *
#   (rate/2). 0.9 keeps a 10% guard below Nyquist so a band's Gaussian tail does
#   not straddle the aliasing edge.
FB_NYQUIST_MARGIN = 0.9
#
# FB_RESOLUTION_MIN_CYCLES — a band at f is "resolved" once ~this many cycles fit
#   in the window D = N/rate (a 0.3 Hz band needs ~3 s to see one cycle). Below
#   that the band is present-but-blurry and gets FLAGGED (resolution mask), not
#   zeroed. 1.0 cycle is the minimum to estimate a frequency's energy at all.
FB_RESOLUTION_MIN_CYCLES = 1.0
# ============================================================================


class PhysicalFilterbankTokenizer(nn.Module):
    """
    Physical-Hz constant-Q filterbank tokenizer (PHz-FB). Design rationale:
    docs/design/EVIDENCE_ENGINE.md §7 (tokenizer direction).

    Turns each native-rate, zero-padded patch of each channel into one d_model
    token, entirely in the *physical frequency* domain, so the representation is
    rate-invariant and anti-aliased by construction (no interpolation exists in the
    path). Pipeline, per patch / per channel:

        Hann window + DC removal
          -> native-rate zero-padded rDFT (size S); bin m -> physical Hz  phi[m] = m*r/S
          -> fixed constant-Q Gaussian filterbank (K bands, centers fixed in Hz) -> E_k
          -> log1p compression + frozen per-band standardization
          -> concat[ e_hat(K), nyquist_mask(K), (resolution_flag(K)), amplitude(1) ]
          -> shared Linear(-> d_model)

    Contract (drop-in for the old extractors):
        forward(patches, sampling_rate_hz, patch_len_samples=None)
          patches:            (B, P, S, C)   native-rate, zero-padded to S
          sampling_rate_hz:   scalar | (B,)  ONE rate per sample (see note)
          patch_len_samples:  scalar | (B,) | None   true N per sample, for the Hann
                                                      window / DC / masks (None -> S)
        returns tokens:       (B, P, C, d_model)

    Rate is bound at the *sample* level, not per channel — all channels within one
    sample share one rate. This matches the corpus (each device resamples all
    channels to a common grid). Genuinely mixed-rate channels within one sample
    would need a (B, C) rate/length signature; out of scope by design.
    """

    def __init__(
        self,
        d_model: int = 384,
        n_bands: int = FB_N_BANDS,
        f_min: float = FB_F_MIN_HZ,
        f_max: float = FB_F_MAX_HZ,
        Q: float = FB_Q,
        dft_size: int = FB_DFT_SIZE,
        nyquist_margin: float = FB_NYQUIST_MARGIN,
        learnable: bool = False,               # True -> Arm B (learnable Gaussian centers)
        use_amplitude: bool = True,
        use_dc: bool = True,                   # signed per-channel DC (gravity/tilt) feature
        use_resolution_mask: bool = True,      # low-freq mirror of the Nyquist mask
        resolution_min_cycles: float = FB_RESOLUTION_MIN_CYCLES,  # band "resolved" once ~this many cycles fit in D
        norm: str = "frozen",                  # 'frozen' | 'none' (per-band standardization)
    ):
        super().__init__()
        self.d_model = d_model
        self.n_bands = int(n_bands)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.Q = float(Q)
        self.S = int(dft_size)
        self.M = self.S // 2                    # rDFT returns M+1 bins
        self.nyquist_margin = float(nyquist_margin)
        self.use_amplitude = bool(use_amplitude)
        self.use_dc = bool(use_dc)
        self.use_resolution_mask = bool(use_resolution_mask)
        self.resolution_min_cycles = float(resolution_min_cycles)
        self.norm = norm
        self.learnable = bool(learnable)

        # Log-spaced physical-Hz band centers f_1..f_K
        k = torch.arange(self.n_bands, dtype=torch.float32)
        centers = self.f_min * (self.f_max / self.f_min) ** (k / (self.n_bands - 1))
        if self.learnable:
            # Arm B: unconstrained logits -> centers in (f_min, f_max) via a sigmoid map,
            # so every center is differentiable everywhere (no clamp wall to freeze at)
            # and cannot overflow. Init matches Arm A's log-spaced centers (frac=k/(K-1)).
            frac = (k / (self.n_bands - 1)).clamp(1e-4, 1 - 1e-4)
            self._center_logits = nn.Parameter(torch.logit(frac))
        else:
            self.register_buffer("centers", centers)

        # Frozen per-band standardization buffers (identity until calibrated via
        # fit_norm_stats / the accumulate+finalize API over the augmented (r,D) mix).
        self.register_buffer("norm_mu", torch.zeros(self.n_bands))
        self.register_buffer("norm_sd", torch.ones(self.n_bands))
        self.register_buffer("_norm_fitted", torch.zeros(1))
        # Running accumulators for streaming calibration (not persisted). float64 to
        # avoid catastrophic cancellation in the two-pass variance (band log-energies
        # can have large means relative to their variance).
        self.register_buffer("_acc_count", torch.zeros(self.n_bands, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sum", torch.zeros(self.n_bands, dtype=torch.float64), persistent=False)
        self.register_buffer("_acc_sqsum", torch.zeros(self.n_bands, dtype=torch.float64), persistent=False)

        # Frozen signed-DC standardization (scalar). DC = per-channel patch mean in
        # native units (the gravity/tilt component the band path removes in _band_energy).
        # It is SIGNED, so it does NOT go through log1p; it needs its own accumulator.
        # Stats are pooled over all channels+patches (gravity can land on any axis, so
        # one shared (mu,sd) keeps the feature axis-agnostic and unit-normalized —
        # standardizing away the m/s^2-vs-g device fingerprint while preserving the
        # relative direction/tilt across the 3 accel channels within a patch).
        self.register_buffer("dc_mu", torch.zeros(1))
        self.register_buffer("dc_sd", torch.ones(1))
        self.register_buffer("_dc_acc_count", torch.zeros(1, dtype=torch.float64), persistent=False)
        self.register_buffer("_dc_acc_sum", torch.zeros(1, dtype=torch.float64), persistent=False)
        self.register_buffer("_dc_acc_sqsum", torch.zeros(1, dtype=torch.float64), persistent=False)

        in_dim = self.n_bands + self.n_bands                 # e_hat + nyquist mask
        if self.use_resolution_mask:
            in_dim += self.n_bands                           # resolution flag
        if self.use_amplitude:
            in_dim += 1                                      # amplitude scalar
        if self.use_dc:
            in_dim += 1                                      # signed DC (gravity/tilt) scalar
        self.in_dim = in_dim
        self.proj = nn.Linear(in_dim, d_model)

    # ------------------------------------------------------------------ helpers
    def _band_centers(self) -> torch.Tensor:
        if self.learnable:
            # sigmoid(logits) in (0,1) -> centers in (f_min, f_max); nonzero gradient
            # everywhere (no clamp wall), and exp-overflow is impossible.
            frac = torch.sigmoid(self._center_logits)
            return self.f_min * (self.f_max / self.f_min) ** frac
        return self.centers

    def get_output_dim(self) -> int:
        return self.d_model

    def get_config(self) -> dict:
        """Hyperparameters needed to reconstruct this tokenizer (for save/load, M4)."""
        return {
            "n_bands": self.n_bands, "f_min": self.f_min, "f_max": self.f_max,
            "Q": self.Q, "dft_size": self.S, "nyquist_margin": self.nyquist_margin,
            "learnable": self.learnable, "use_amplitude": self.use_amplitude,
            "use_dc": self.use_dc,
            "use_resolution_mask": self.use_resolution_mask, "norm": self.norm,
        }

    def _prep_rate_len(self, sampling_rate_hz, patch_len_samples, B, device, dtype
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize (rate, N) inputs to (B,) float rate and (B,) long length."""
        if not torch.is_tensor(sampling_rate_hz):
            sampling_rate_hz = torch.as_tensor(sampling_rate_hz)
        r = sampling_rate_hz.to(device=device, dtype=dtype).reshape(-1)
        if r.numel() == 1:
            r = r.expand(B)
        assert r.numel() == B, f"sampling_rate_hz must be scalar or length B={B}, got {r.numel()}"

        if patch_len_samples is None:
            N = torch.full((B,), self.S, dtype=torch.long, device=device)
        else:
            if not torch.is_tensor(patch_len_samples):
                patch_len_samples = torch.as_tensor(patch_len_samples)
            N = patch_len_samples.to(device=device).reshape(-1).long()
            if N.numel() == 1:
                N = N.expand(B)
        assert N.numel() == B, f"patch_len_samples must be scalar or length B={B}"
        # Guardrail: native samples must fit the DFT window, else the zero-pad
        # silently becomes a truncation that destroys low-frequency resolution.
        n_max = int(N.max())
        if n_max > self.S:
            raise ValueError(
                f"patch_len_samples max {n_max} exceeds dft_size S={self.S}; raise "
                f"dft_size so that r*D <= S for every sample."
            )
        return r, N

    def _hann_and_valid(self, N, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-sample Hann window (B,S) placed in [0,N) and a (B,S) validity mask."""
        B = N.numel()
        idx = torch.arange(self.S, device=device).unsqueeze(0)     # (1, S)
        valid = (idx < N.unsqueeze(1)).to(dtype)                   # (B, S)
        Nf = N.to(dtype).clamp(min=2).unsqueeze(1)                 # (B, 1); guard N<2
        hann = 0.5 * (1.0 - torch.cos(2 * math.pi * idx / (Nf - 1.0)))
        window = hann * valid                                      # zero outside [0, N)
        return window, valid

    def _band_energy(self, patches, r, N):
        """
        Core DSP. (B,P,S,C) + per-sample (r,N) -> band energy E (B,P,C,K), the band
        centers / sigmas used for the observability masks, and the signed per-channel
        DC mean (B,P,C) that the band path removes (the gravity/tilt component).
        """
        B, P, S, C = patches.shape
        device, dtype = patches.device, patches.dtype
        window, valid = self._hann_and_valid(N, device, dtype)     # (B,S),(B,S)

        # DC removal over the *real* samples only, then apply the Hann window.
        vm = valid.view(B, 1, S, 1)
        Nf = N.to(dtype).clamp(min=1).view(B, 1, 1, 1)
        mean = (patches * vm).sum(dim=2, keepdim=True) / Nf        # (B,P,1,C)
        dc = mean.squeeze(2)                                        # (B,P,C) signed DC
        x_win = (patches - mean) * vm * window.view(B, 1, S, 1)    # (B,P,S,C)

        # Native-rate zero-padded rDFT over the time axis; keep power (drop phase).
        X = torch.fft.rfft(x_win, n=S, dim=2)                      # (B,P,M+1,C) complex
        power = X.real ** 2 + X.imag ** 2                          # (B,P,M+1,C)

        # Normalize by window energy so band energy is a power estimate independent of
        # N=r*D. By Parseval, sum_m|X|^2 = S*sum_n(w*x)^2 which scales with sum_n w^2 ~ N;
        # without this, E (and the amplitude scalar) would scale with r*D and silently
        # encode the sampling rate into a feature that is supposed to be rate-invariant.
        win_energy = (window ** 2).sum(dim=1).clamp(min=1e-8)      # (B,)
        power = power / win_energy.view(B, 1, 1, 1)

        # Physical-Hz constant-Q Gaussian filterbank. phi depends on r -> per sample.
        centers = self._band_centers().to(device=device, dtype=dtype)   # (K,)
        sigma = centers / (2.0 * self.Q)                                 # (K,)
        m = torch.arange(self.M + 1, device=device, dtype=dtype)         # (M+1,)
        phi = m.unsqueeze(0) * r.unsqueeze(1) / self.S                   # (B, M+1) Hz
        diff = phi.unsqueeze(1) - centers.view(1, -1, 1)                 # (B,K,M+1)
        H = torch.exp(-0.5 * (diff / sigma.view(1, -1, 1)) ** 2)         # (B,K,M+1)
        E = torch.einsum("bkm,bpmc->bpck", H, power)                     # (B,P,C,K)
        return E, centers, sigma, dc

    def _observability_masks(self, r, N, centers, sigma
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nyquist observability o (B,K) and low-freq resolution flag res (B,K)."""
        dtype = r.dtype
        nyq = self.nyquist_margin * (r * 0.5)                            # (B,)
        o = (centers.view(1, -1) + 2.0 * sigma.view(1, -1)
             <= nyq.view(-1, 1)).to(dtype)                              # (B,K)
        D = (N.to(dtype) / r).clamp(min=1e-6)                           # (B,) window seconds
        res = (centers.view(1, -1) * D.view(-1, 1)
               / self.resolution_min_cycles).clamp(0.0, 1.0)           # (B,K)
        return o, res

    def masks(self, sampling_rate_hz, patch_len_samples=None
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Public: (nyquist observability o, resolution flag res), each (B, K)."""
        device, dtype = self.norm_mu.device, self.norm_mu.dtype
        B = torch.as_tensor(sampling_rate_hz).reshape(-1).numel()
        r, N = self._prep_rate_len(sampling_rate_hz, patch_len_samples, B, device, dtype)
        centers = self._band_centers().to(device=device, dtype=dtype)
        sigma = centers / (2.0 * self.Q)
        return self._observability_masks(r, N, centers, sigma)

    # ---------------------------------------------------------------- calibration
    def reset_norm_accumulator(self):
        self._acc_count.zero_()
        self._acc_sum.zero_()
        self._acc_sqsum.zero_()
        self._dc_acc_count.zero_()
        self._dc_acc_sum.zero_()
        self._dc_acc_sqsum.zero_()

    @torch.no_grad()
    def accumulate_norm_stats(self, patches, sampling_rate_hz, patch_len_samples=None, patch_mask=None):
        """Fold one (augmented) batch into the running per-band log-energy stats.

        Only *observable* bands are folded in (Nyquist mask applied per band), so a band
        above a low-rate sample's Nyquist is not dragged toward zero by out-of-band
        filter-tail energy. This keeps the frozen mean equal to the observable-conditional
        mean, so the neutral 0 that forward() imputes for masked bands matches their mean.

        patch_mask: optional (B, P) bool — padded patches are excluded from the stats.
        """
        B, P = patches.shape[0], patches.shape[1]
        r, N = self._prep_rate_len(sampling_rate_hz, patch_len_samples, B,
                                   patches.device, patches.dtype)
        E, centers, sigma, dc = self._band_energy(patches, r, N)         # (B,P,C,K), dc (B,P,C)
        o, _ = self._observability_masks(r, N, centers, sigma)          # (B,K)
        e = torch.log1p(E).to(torch.float64)                           # (B,P,C,K)
        w = o.view(B, 1, 1, self.n_bands).expand_as(e).to(torch.float64)
        if patch_mask is not None:
            w = w * patch_mask.view(B, P, 1, 1).to(torch.float64)      # exclude padded patches
        e = e.reshape(-1, self.n_bands)
        w = w.reshape(-1, self.n_bands)
        self._acc_count += w.sum(dim=0)                                 # per-band counts
        self._acc_sum += (e * w).sum(dim=0)
        self._acc_sqsum += (e * e * w).sum(dim=0)

        # Signed DC (gravity/tilt) stats: pooled over all channels+patches (one scalar
        # mu/sd), so the feature is axis-agnostic and the m/s^2-vs-g scale is normalized
        # out. No log1p (dc is signed). Padded patches excluded via patch_mask.
        if self.use_dc:
            d = dc.to(torch.float64).reshape(B, P, -1)                 # (B,P,C)
            if patch_mask is not None:
                dw = patch_mask.view(B, P, 1).to(torch.float64).expand_as(d)
            else:
                dw = torch.ones_like(d)
            d = d.reshape(-1)
            dw = dw.reshape(-1)
            self._dc_acc_count += dw.sum()
            self._dc_acc_sum += (d * dw).sum()
            self._dc_acc_sqsum += (d * d * dw).sum()

    @torch.no_grad()
    def finalize_norm_stats(self, eps: float = 1e-5):
        """Set frozen mu/sd from the per-band accumulators. Call once after calibration.

        Bands never observed during calibration fall back to identity (mu=0, sd=1) so an
        unseen-but-later-observable band cannot blow up e_hat at inference.
        """
        seen = self._acc_count > 0
        safe_n = self._acc_count.clamp(min=1.0)
        mu = self._acc_sum / safe_n
        var = (self._acc_sqsum / safe_n) - mu * mu
        sd = var.clamp(min=eps).sqrt()
        mu = torch.where(seen, mu, torch.zeros_like(mu))
        sd = torch.where(seen, sd, torch.ones_like(sd))
        self.norm_mu.copy_(mu)
        self.norm_sd.copy_(sd)

        # Frozen signed-DC standardization (scalar). Falls back to identity (0,1) if
        # the DC feature is disabled or never accumulated.
        if self.use_dc and self._dc_acc_count.item() > 0:
            dc_n = self._dc_acc_count.clamp(min=1.0)
            dc_mu = self._dc_acc_sum / dc_n
            dc_var = (self._dc_acc_sqsum / dc_n) - dc_mu * dc_mu
            dc_sd = dc_var.clamp(min=eps).sqrt()
            self.dc_mu.copy_(dc_mu.to(self.dc_mu.dtype))
            self.dc_sd.copy_(dc_sd.to(self.dc_sd.dtype))
        else:
            self.dc_mu.zero_()
            self.dc_sd.fill_(1.0)
        self._norm_fitted.fill_(1.0)

    @torch.no_grad()
    def fit_norm_stats(self, patches, sampling_rate_hz, patch_len_samples=None, eps: float = 1e-5):
        """Convenience one-shot calibration over a single (large) batch."""
        self.reset_norm_accumulator()
        self.accumulate_norm_stats(patches, sampling_rate_hz, patch_len_samples)
        self.finalize_norm_stats(eps)

    # --------------------------------------------------------------------- forward
    def forward(self, patches, sampling_rate_hz, patch_len_samples=None) -> torch.Tensor:
        B, P, S, C = patches.shape
        assert S == self.S, (
            f"patch time dim {S} != dft_size {self.S}; zero-pad patches to S before the tokenizer"
        )
        device, dtype = patches.device, patches.dtype
        r, N = self._prep_rate_len(sampling_rate_hz, patch_len_samples, B, device, dtype)

        E, centers, sigma, dc = self._band_energy(patches, r, N)    # (B,P,C,K), dc (B,P,C)

        # Compression + frozen per-band standardization.
        e = torch.log1p(E)
        e_hat = (e - self.norm_mu) / self.norm_sd if self.norm == "frozen" else e

        # Amplitude scalar: total log-energy, preserves absolute magnitude.
        amp = torch.log1p(E.sum(dim=-1, keepdim=True))              # (B,P,C,1)

        # Signed DC (gravity/tilt) feature: the per-channel patch mean the band path
        # removed. Frozen-standardized (scalar mu/sd) so the m/s^2-vs-g device scale is
        # normalized out and only the relative cross-channel gravity direction survives.
        # This restores static-posture discrimination (stand/sit/lie differ only in DC).
        dc_feat = ((dc - self.dc_mu) / self.dc_sd).unsqueeze(-1) if self.norm == "frozen" \
            else dc.unsqueeze(-1)                                    # (B,P,C,1)

        # Nyquist observability mask (o) zeroes bands above native Nyquist (neutral,
        # since e_hat is standardized). Resolution flag (res) is the low-freq mirror:
        # a band at f_k needs ~resolution_min_cycles cycles within D=N/r to be resolved;
        # below that the value is present-but-blurry, so we *flag* it rather than zero it.
        o, res = self._observability_masks(r, N, centers, sigma)   # (B,K),(B,K)
        o_bpck = o.view(B, 1, 1, self.n_bands).expand(B, P, C, self.n_bands)
        e_hat = e_hat * o_bpck

        feats = [e_hat, o_bpck]
        if self.use_resolution_mask:
            feats.append(res.view(B, 1, 1, self.n_bands).expand(B, P, C, self.n_bands))

        if self.use_amplitude:
            feats.append(amp)

        if self.use_dc:
            feats.append(dc_feat)

        token_in = torch.cat(feats, dim=-1)                        # (B,P,C,in_dim)
        return self.proj(token_in)                                 # (B,P,C,d_model)
