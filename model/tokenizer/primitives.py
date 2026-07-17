"""Structured physical primitives (M1) — the feature families that survived the M0 probe.

The M0 robustness probe (training/tokenizer/outputs/m0_probe/REPORT.md) fixed this set:

  * grav_band_energy  — gravity-aligned per-band log energy, [vertical, horizontal-pooled]
                        per modality. Fully rotation-invariant (horiz pooling kills yaw);
                        the best cross-dataset transfer family in M0 (0.502 vs raw 0.461).
  * eigen_ratios      — per-band accel-triad covariance eigen-ratios (linearity/planarity/
                        isotropy). Exactly rotation- AND gain-invariant (C -> RCR^T keeps
                        eigenvalues; ratios cancel g^2). Grounding targets for A3.
  * coherence         — accel<->gyro magnitude coherence per band (rotation-invariant via
                        magnitudes). Undefined without a gyro -> validity-masked.
  * spectral_shape    — relative band-energy distribution + centroid + entropy of |acc|.
  * cadence           — octave-aware dominant periodicity (log2 Hz) + strength. Validity
                        needs BOTH periodicity strength AND a motion-energy floor (M0:
                        static windows autocorrelate on drift and fabricate cadences).
  * dc_tilt           — unit gravity direction in the sensor frame (the static posture
                        cue); validity = plausible gravity present.

`raw` per-channel band energy is EXCLUDED (M0: fragile under rotation; config-specific
shortcut). Output is a NAMED dict of (values, valid) — never a monolithic vector; fusion
is the learned encoder's job (M0: naive concat does not fuse).

Everything is computed over TEXT-identified channel groups (preprocess.find_triads) and
degrades gracefully: a missing triad yields valid=False entries, never a crash.

Batch contract: one sampling rate per call (batches are bucketed by (rate, patch_seconds)
upstream — EVIDENCE_ENGINE.md §5.2.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .preprocess import accel_gyro_triads, estimate_gravity, gravity_align

# ----------------------------------------------------------------------------------------------
# Constants (M0-probe values — see the report before second-guessing)
# ----------------------------------------------------------------------------------------------
# Coarse physical bands for the primitives (NOT the filterbank's 32 constant-Q bands —
# eigendecomposition per band is O(bands), and M0 showed these 4 active bands separate).
PRIMITIVE_BANDS: tuple[tuple[float, float], ...] = ((0.5, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 12.0))
ENERGY_BANDS: tuple[tuple[float, float], ...] = ((0.0, 0.5), *PRIMITIVE_BANDS, (12.0, 25.0))

CADENCE_LAG_S = (0.25, 2.0)      # autocorr peak search (0.5-4 Hz)
CADENCE_MIN_STRENGTH = 0.30      # periodicity floor
CADENCE_MIN_MOTION_G = 0.03      # dynamic |acc| std floor (M0: static windows fabricate cadences)
CADENCE_OCTAVE_TOLERANCE = 0.75  # prefer half the lag (double rate) if its autocorr >= this * peak

EPS = 1e-12


@dataclass
class Primitive:
    """One named primitive: values (B, ...) + a per-window validity mask (B,)."""

    values: torch.Tensor
    valid: torch.Tensor  # (B,) bool — False entries carry no gradient / no loss / no vote

    def masked(self, fill: float = 0.0) -> torch.Tensor:
        shape = (-1,) + (1,) * (self.values.dim() - 1)
        return torch.where(self.valid.view(shape), self.values,
                           torch.full_like(self.values, fill))


# ----------------------------------------------------------------------------------------------
# DSP helpers (batched, rFFT-based; physical Hz throughout)
# ----------------------------------------------------------------------------------------------
def _band_power(x: torch.Tensor, rate: float, lo: float, hi: float) -> torch.Tensor:
    """Mean power of x (B, T, C) in [lo, hi) Hz -> (B, C)."""
    T = x.shape[1]
    spec = torch.fft.rfft(x, dim=1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / rate).to(x.device)
    sel = ((freqs >= lo) & (freqs < hi)).view(1, -1, 1)
    return (spec.real**2 + spec.imag**2).mul(sel).sum(dim=1) / (T**2)


def _bandpass(x: torch.Tensor, rate: float, lo: float, hi: float) -> torch.Tensor:
    """FFT-mask bandpass of x (B, T, C)."""
    T = x.shape[1]
    spec = torch.fft.rfft(x, dim=1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / rate).to(x.device)
    keep = ((freqs >= lo) & (freqs < hi)).view(1, -1, 1)
    return torch.fft.irfft(spec * keep, n=T, dim=1)


def _demeaned_magnitude(x: torch.Tensor) -> torch.Tensor:
    """(B, T, 3) -> (B, T) demeaned Euclidean magnitude (rotation-invariant scalar signal)."""
    mag = x.norm(dim=2)
    return mag - mag.mean(dim=1, keepdim=True)


# ----------------------------------------------------------------------------------------------
# The primitive families
# ----------------------------------------------------------------------------------------------
def grav_band_energy(
    aligned: torch.Tensor,
    acc_idx: Sequence[int],
    gyro_idx: Optional[Sequence[int]],
    rate: float,
    aligned_ok: torch.Tensor,
) -> Primitive:
    """Gravity-aligned [vertical, horizontal-pooled] log band energy per modality.

    aligned: (B, T, C) ALREADY gravity-aligned windows. Output (B, n_bands * n_mod * 2).
    Horizontal pooling (x+y after align) is what kills the residual yaw freedom.
    """
    feats = []
    triads = [aligned[:, :, list(acc_idx)]]
    if gyro_idx is not None:
        triads.append(aligned[:, :, list(gyro_idx)])
    for lo, hi in ENERGY_BANDS:
        for tri in triads:
            p = _band_power(tri, rate, lo, hi)                       # (B, 3)
            vert = torch.log10(p[:, 2] + EPS)
            horiz = torch.log10(p[:, 0] + p[:, 1] + EPS)
            feats += [vert, horiz]
    return Primitive(values=torch.stack(feats, dim=1), valid=aligned_ok)


def eigen_ratios(acc: torch.Tensor, rate: float) -> Primitive:
    """Per-band accel covariance eigen-ratios -> (B, n_bands, 3) [linearity, planarity, isotropy].

    Rotation-invariant (C -> RCR^T) and gain-invariant (ratios cancel g^2) — exact, not
    approximate. Bands with ~no energy are per-entry NaN; window-valid if ANY band has energy.
    """
    B = acc.shape[0]
    out = acc.new_full((B, len(PRIMITIVE_BANDS), 3), float("nan"))
    for b, (lo, hi) in enumerate(PRIMITIVE_BANDS):
        band = _bandpass(acc, rate, lo, hi)                          # (B, T, 3)
        centered = band - band.mean(dim=1, keepdim=True)
        cov = torch.einsum("bti,btj->bij", centered, centered) / band.shape[1]
        lam = torch.linalg.eigvalsh(cov).flip(-1)                    # (B, 3) descending
        l1 = lam[:, 0]
        ok = l1 > 1e-10
        vals = torch.stack([
            (lam[:, 0] - lam[:, 1]) / l1.clamp(min=EPS),
            (lam[:, 1] - lam[:, 2]) / l1.clamp(min=EPS),
            lam[:, 2] / l1.clamp(min=EPS),
        ], dim=1)
        out[:, b] = torch.where(ok.unsqueeze(1), vals, torch.full_like(vals, float("nan")))
    valid = torch.isfinite(out).any(dim=(1, 2))
    return Primitive(values=out, valid=valid)


def coherence(
    acc: torch.Tensor,
    gyro: Optional[torch.Tensor],
    rate: float,
    seg_seconds: float = 2.0,
) -> Primitive:
    """Welch magnitude-squared coherence between |acc| and |gyro| per band -> (B, n_bands).

    Rotation-invariant via magnitudes. No gyro -> valid=False (zeros), never a crash.
    """
    B, T = acc.shape[0], acc.shape[1]
    n_bands = len(PRIMITIVE_BANDS)
    if gyro is None:
        return Primitive(values=acc.new_zeros(B, n_bands),
                         valid=torch.zeros(B, dtype=torch.bool, device=acc.device))

    a, g = _demeaned_magnitude(acc), _demeaned_magnitude(gyro)      # (B, T)
    nper = min(int(seg_seconds * rate), T)
    hop = max(1, nper // 2)
    window = torch.hann_window(nper, device=acc.device, dtype=acc.dtype)

    def segments(x):
        segs = x.unfold(1, nper, hop) * window                      # (B, n_seg, nper)
        return torch.fft.rfft(segs, dim=2)

    fa, fg = segments(a), segments(g)
    pxx = (fa.real**2 + fa.imag**2).mean(dim=1)                     # (B, F)
    pyy = (fg.real**2 + fg.imag**2).mean(dim=1)
    pxy = (fa * fg.conj()).mean(dim=1)
    cxy = pxy.abs().pow(2) / (pxx * pyy).clamp(min=EPS)             # (B, F)

    freqs = torch.fft.rfftfreq(nper, d=1.0 / rate).to(acc.device)
    out = []
    for lo, hi in PRIMITIVE_BANDS:
        sel = (freqs >= lo) & (freqs < hi)
        out.append(cxy[:, sel].mean(dim=1) if bool(sel.any())
                   else acc.new_full((B,), float("nan")))
    values = torch.stack(out, dim=1)
    valid = torch.isfinite(values).any(dim=1)
    return Primitive(values=values, valid=valid)


def spectral_shape(acc: torch.Tensor, rate: float) -> Primitive:
    """Relative band-energy distribution of |acc| + centroid + entropy -> (B, n_bands + 2).

    Gain-invariant (normalized) and rotation-invariant (magnitude signal).
    """
    mag = _demeaned_magnitude(acc).unsqueeze(2)                     # (B, T, 1)
    powers = torch.stack(
        [_band_power(mag, rate, lo, hi)[:, 0] for lo, hi in PRIMITIVE_BANDS], dim=1
    )                                                               # (B, n_bands)
    total = powers.sum(dim=1, keepdim=True)
    rel = powers / total.clamp(min=EPS)
    mids = torch.tensor([(lo + hi) / 2 for lo, hi in PRIMITIVE_BANDS],
                        device=acc.device, dtype=acc.dtype)
    centroid = (rel * mids).sum(dim=1, keepdim=True)
    entropy = -(rel * (rel + EPS).log()).sum(dim=1, keepdim=True)
    values = torch.cat([rel, centroid, entropy], dim=1)
    valid = total.squeeze(1) > EPS
    return Primitive(values=values, valid=valid)


def cadence(acc: torch.Tensor, rate: float) -> Primitive:
    """Octave-aware dominant periodicity -> (B, 2) [log2 cadence Hz, strength].

    Autocorrelation peak over lags CADENCE_LAG_S, then octave disambiguation: if HALF
    the peak lag (double the rate) autocorrelates at >= CADENCE_OCTAVE_TOLERANCE * peak,
    prefer it — this resolves the M0-observed stride-vs-step lock (walking halved to
    stride, running locked to step) toward a consistent STEP rate.

    Validity needs BOTH: periodicity strength >= CADENCE_MIN_STRENGTH AND dynamic motion
    std >= CADENCE_MIN_MOTION_G (M0: near-static windows autocorrelate on drift and
    fabricate cadences without the motion floor).
    """
    B, T = acc.shape[0], acc.shape[1]
    mag = _demeaned_magnitude(acc)                                  # (B, T)
    motion_ok = mag.std(dim=1) >= CADENCE_MIN_MOTION_G

    # Batched autocorrelation via FFT (zero-padded to 2T to avoid circular wrap).
    spec = torch.fft.rfft(mag, n=2 * T, dim=1)
    ac = torch.fft.irfft(spec.real**2 + spec.imag**2, n=2 * T, dim=1)[:, :T]
    ac = ac / ac[:, :1].clamp(min=EPS)                              # (B, T), ac[0] = 1

    lag_lo = max(1, int(CADENCE_LAG_S[0] * rate))
    lag_hi = min(int(CADENCE_LAG_S[1] * rate), T - 1)
    if lag_hi <= lag_lo:
        return Primitive(values=acc.new_zeros(B, 2),
                         valid=torch.zeros(B, dtype=torch.bool, device=acc.device))

    search = ac[:, lag_lo:lag_hi]
    peak_val, rel_idx = search.max(dim=1)
    peak_lag = rel_idx + lag_lo                                     # (B,)

    # Octave disambiguation: half lag = double frequency (stride -> step).
    half_lag = (peak_lag // 2).clamp(min=1)
    half_val = ac.gather(1, half_lag.unsqueeze(1)).squeeze(1)
    use_half = (half_lag >= lag_lo) & (half_val >= CADENCE_OCTAVE_TOLERANCE * peak_val)
    lag = torch.where(use_half, half_lag, peak_lag)
    strength = torch.where(use_half, half_val, peak_val)

    log2_hz = torch.log2(rate / lag.to(acc.dtype))
    valid = motion_ok & (strength >= CADENCE_MIN_STRENGTH)
    values = torch.stack([log2_hz, strength], dim=1)
    return Primitive(values=values, valid=valid)


def dc_tilt(acc: torch.Tensor, rate: float) -> Primitive:
    """Unit gravity direction in the sensor frame -> (B, 3); the static-posture cue.

    Pre-alignment by definition (post-align it is trivially ~[0,0,1]) — feed the RAW
    accel triad here. Validity = plausible gravity magnitude (estimate_gravity gate).
    """
    gravity, present = estimate_gravity(acc, rate)
    unit = gravity / gravity.norm(dim=1, keepdim=True).clamp(min=EPS)
    return Primitive(values=unit, valid=present)


# ----------------------------------------------------------------------------------------------
# The front door
# ----------------------------------------------------------------------------------------------
def compute_primitives(
    windows: torch.Tensor,
    channel_names: Sequence[str],
    sampling_rate_hz: float,
    align_gravity: bool = True,
) -> dict[str, Primitive]:
    """windows (B, T, C) + channel text + rate -> named dict of primitives.

    Channel groups are found by TEXT; anything missing degrades to valid=False.
    Rotation-invariant families are computed on the RAW frame (they don't need
    alignment — that exactness is the point); grav_band_energy and dc_tilt handle
    the gravity frame explicitly.
    """
    acc_idx, gyro_idx = accel_gyro_triads(channel_names)
    B = windows.shape[0]
    if acc_idx is None:
        # No accelerometer triad: every primitive is invalid (never a crash).
        false = torch.zeros(B, dtype=torch.bool, device=windows.device)
        zero = windows.new_zeros(B, 1)
        return {name: Primitive(values=zero, valid=false)
                for name in ("grav_band_energy", "eigen_ratios", "coherence",
                             "spectral_shape", "cadence", "dc_tilt")}

    acc = windows[:, :, list(acc_idx)]
    gyro = windows[:, :, list(gyro_idx)] if gyro_idx is not None else None

    aligned, _, aligned_ok = gravity_align(
        windows, channel_names, sampling_rate_hz, enabled=align_gravity
    )

    return {
        "grav_band_energy": grav_band_energy(aligned, acc_idx, gyro_idx,
                                             sampling_rate_hz, aligned_ok),
        "eigen_ratios": eigen_ratios(acc, sampling_rate_hz),
        "coherence": coherence(acc, gyro, sampling_rate_hz),
        "spectral_shape": spectral_shape(acc, sampling_rate_hz),
        "cadence": cadence(acc, sampling_rate_hz),
        "dc_tilt": dc_tilt(acc, sampling_rate_hz),
    }
