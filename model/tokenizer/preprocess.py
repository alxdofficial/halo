"""Time-domain preprocessing for the Pipeline A front end (M1).

Two parts:
  1. Patching utilities ported from `legacy_code/model/preprocessing.py`
     (create_patches / zero_pad_patches / preprocess_imu_data) — native-rate
     patches zero-padded to the filterbank DFT size, no interpolation.
  2. NEW: joint gravity canonicalization (`gravity_align`) — estimate the gravity
     direction from the low-pass accelerometer, rotate the whole IMU frame so
     "up" is canonical +z, applying the SAME rotation to every co-located triad
     (accel AND gyro — rotating one alone creates physically impossible frames).

Gravity-align yields a PARTIAL canonical frame: pitch/roll are fixed, yaw is not
(gravity says nothing about heading). Yaw-robustness comes from the invariant
primitives (eigen-ratios, horizontally-pooled energies — see primitives.py), not
from this alignment. That division of labor is the M0-probe result.

Units: the corpus accelerometer is unit-canonicalized to g (|gravity| ~= 1.0), so
all gravity thresholds here are in g (the M0 lesson — an m/s^2-scale threshold
silently disables alignment on this corpus).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch

# ----------------------------------------------------------------------------------------------
# Gravity canonicalization constants (g units — corpus accel is canonicalized to g)
# ----------------------------------------------------------------------------------------------
GRAVITY_LP_HZ = 0.5      # low-pass cutoff isolating the quasi-DC gravity component
GRAVITY_MIN_G = 0.5      # |lowpass acc| below this  => gravity absent (e.g. gravity-removed sets)
GRAVITY_MAX_G = 2.0      # |lowpass acc| above this  => not a plausible gravity estimate


# ================================================================================================
# Patching (ported)
# ================================================================================================
def create_patches(
    data: torch.Tensor,
    sampling_rate_hz: float,
    patch_size_sec: float,
    stride_sec: Optional[float] = None,
) -> torch.Tensor:
    """Split (T, C) data into (num_patches, patch_timesteps, C) fixed-duration patches."""
    if stride_sec is None:
        stride_sec = patch_size_sec

    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, dtype=torch.float32)
    elif data.device.type != "cpu":
        data = data.cpu()

    num_timesteps, _ = data.shape
    patch_timesteps = int(sampling_rate_hz * patch_size_sec)
    stride_timesteps = int(sampling_rate_hz * stride_sec)

    if patch_timesteps > num_timesteps:
        raise ValueError(
            f"Patch size ({patch_timesteps} timesteps) is larger than data length "
            f"({num_timesteps} timesteps). Reduce patch_size_sec or provide more data."
        )

    patches = data.t().unsqueeze(0)                                   # (1, C, T)
    patches = patches.unfold(2, patch_timesteps, stride_timesteps)    # (1, C, P, N)
    patches = patches.squeeze(0).permute(1, 2, 0)                     # (P, N, C)
    return patches.contiguous()


def zero_pad_patches(patches: torch.Tensor, target_size: int) -> torch.Tensor:
    """Zero-pad native-rate patches (P, N, C) to the DFT size S — no resampling."""
    num_patches, n, num_channels = patches.shape
    if n == target_size:
        return patches
    if n > target_size:
        raise ValueError(
            f"Native patch length ({n}) exceeds DFT size ({target_size}); "
            f"raise dft_size so that sampling_rate * patch_size_sec <= dft_size."
        )
    out = patches.new_zeros(num_patches, target_size, num_channels)
    out[:, :n, :] = patches
    return out


def preprocess_imu_data(
    data: torch.Tensor,
    sampling_rate_hz: float,
    patch_size_sec: float,
    stride_sec: Optional[float] = None,
    pad_to_size: int = None,
) -> Tuple[torch.Tensor, dict]:
    """Native-rate patches -> zero-pad to the filterbank DFT size S. No interpolation,
    no per-patch z-score (the filterbank does its own DC removal + amplitude handling)."""
    if pad_to_size is None:
        raise ValueError("preprocess_imu_data requires pad_to_size (the filterbank DFT size S)")

    patches = create_patches(data, sampling_rate_hz, patch_size_sec, stride_sec)
    original_patch_size = patches.shape[1]
    patches = zero_pad_patches(patches, target_size=pad_to_size)
    metadata = {
        "original_patch_size": original_patch_size,
        "patch_len_samples": original_patch_size,   # true N for the tokenizer
        "dft_size": pad_to_size,
        "sampling_rate_hz": sampling_rate_hz,
        "patch_size_sec": patch_size_sec,
        "num_channels": data.shape[1],
    }
    return patches, metadata


# ================================================================================================
# Channel-group identification (text-identified triads — nothing keyed by position)
# ================================================================================================
def find_triads(channel_names: Sequence[str]) -> dict[str, tuple[int, int, int]]:
    """Group channels into xyz triads by name: ``<prefix>_x|_y|_z`` -> {prefix: (ix, iy, iz)}.

    Only complete triads are returned; stray channels are ignored (graceful degradation
    for accel-only or exotic streams). This is the text-identified grouping the design
    requires — channel COUNT and ORDER never matter.
    """
    by_prefix: dict[str, dict[str, int]] = {}
    for index, name in enumerate(channel_names):
        lower = name.lower()
        for axis in ("x", "y", "z"):
            if lower.endswith(f"_{axis}"):
                by_prefix.setdefault(lower[: -2], {})[axis] = index
                break
    return {
        prefix: (axes["x"], axes["y"], axes["z"])
        for prefix, axes in by_prefix.items()
        if len(axes) == 3
    }


def accel_gyro_triads(
    channel_names: Sequence[str],
) -> tuple[Optional[tuple[int, int, int]], Optional[tuple[int, int, int]]]:
    """(accel triad, gyro triad) indices by channel text, either may be None."""
    triads = find_triads(channel_names)
    acc = next((idx for p, idx in triads.items() if "acc" in p), None)
    gyro = next((idx for p, idx in triads.items() if "gyr" in p), None)
    return acc, gyro


# ================================================================================================
# Gravity canonicalization (NEW — M1)
# ================================================================================================
def estimate_gravity(acc: torch.Tensor, sampling_rate_hz: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate the gravity vector from low-pass accel.

    acc: (B, T, 3) accelerometer in g units. Returns (gravity (B, 3), present (B,) bool).
    ``present`` is False when the low-pass magnitude is outside [GRAVITY_MIN_G,
    GRAVITY_MAX_G] — gravity-removed streams, free-fall, or corrupted units.
    """
    B, T, _ = acc.shape
    spec = torch.fft.rfft(acc, dim=1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / sampling_rate_hz).to(acc.device)
    spec = spec * (freqs < GRAVITY_LP_HZ).view(1, -1, 1)
    lowpass = torch.fft.irfft(spec, n=T, dim=1)
    gravity = lowpass.mean(dim=1)                                     # (B, 3)
    norm = gravity.norm(dim=1)
    present = (norm >= GRAVITY_MIN_G) & (norm <= GRAVITY_MAX_G)
    return gravity, present


def rotation_to_z(u: torch.Tensor) -> torch.Tensor:
    """Batch Rodrigues rotation matrices taking unit vectors u (B, 3) onto +z."""
    B = u.shape[0]
    ez = torch.tensor([0.0, 0.0, 1.0], device=u.device, dtype=u.dtype)
    v = torch.linalg.cross(u, ez.expand(B, 3))                        # (B, 3) axis * sin
    s = v.norm(dim=1)                                                 # sin(theta)
    c = u @ ez                                                        # cos(theta)
    k = torch.zeros(B, 3, 3, device=u.device, dtype=u.dtype)
    k[:, 0, 1], k[:, 0, 2] = -v[:, 2], v[:, 1]
    k[:, 1, 0], k[:, 1, 2] = v[:, 2], -v[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -v[:, 1], v[:, 0]
    eye = torch.eye(3, device=u.device, dtype=u.dtype).expand(B, 3, 3)
    factor = ((1.0 - c) / (s**2).clamp(min=1e-12)).view(B, 1, 1)
    r = eye + k + factor * (k @ k)
    # Degenerate cases: already aligned -> identity; anti-parallel -> 180deg about x.
    aligned = s < 1e-6
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=u.device, dtype=u.dtype))
    r[aligned & (c > 0)] = torch.eye(3, device=u.device, dtype=u.dtype)
    r[aligned & (c < 0)] = flip
    return r


def gravity_align(
    windows: torch.Tensor,
    channel_names: Sequence[str],
    sampling_rate_hz: float,
    enabled: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate each window so the estimated gravity ("up") is canonical +z.

    windows: (B, T, C); channels identified by TEXT (accel triad required for the
    estimate; every complete xyz triad — accel AND gyro — is rotated by the SAME R,
    joint and physical). Non-triad channels pass through untouched.

    Returns (aligned windows (B, T, C), R (B, 3, 3), aligned (B,) bool). Windows whose
    gravity estimate is absent/implausible are returned UNROTATED with aligned=False —
    the caller carries that flag (it is a validity input for downstream primitives).
    Yaw remains free (documented partial frame).
    """
    B = windows.shape[0]
    eye = torch.eye(3, device=windows.device, dtype=windows.dtype).expand(B, 3, 3).clone()
    if not enabled:
        return windows, eye, torch.zeros(B, dtype=torch.bool, device=windows.device)

    acc_idx, _ = accel_gyro_triads(channel_names)
    if acc_idx is None:
        return windows, eye, torch.zeros(B, dtype=torch.bool, device=windows.device)

    gravity, present = estimate_gravity(windows[:, :, list(acc_idx)], sampling_rate_hz)
    unit = gravity / gravity.norm(dim=1, keepdim=True).clamp(min=1e-12)
    r = rotation_to_z(unit)
    r[~present] = torch.eye(3, device=windows.device, dtype=windows.dtype)

    out = windows.clone()
    for triad in find_triads(channel_names).values():
        idx = list(triad)
        out[:, :, idx] = torch.einsum("bij,btj->bti", r, windows[:, :, idx])
    return out, r, present
