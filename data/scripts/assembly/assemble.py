"""Assemble a curated device stream into windowed HARMONISED / NON-HARMONISED grids.

This is the one place the data pipeline is tied together, in a fixed order:

    raw session frame
      → deployment_policy.curate_frame   device/channel selection + gravity reconstruction
      → accel_units (to g)               accelerometer unit → g; gyroscope is NEVER scaled
      → resample (optional)              anti-aliased polyphase to a target rate (60 Hz harmonised)
      → fixed-length windows             + per-window majority label
      → baseline_view.to_view            harmonised: fixed 6-ch [acc,gyro] pad+mask
                                         non_harmonised: native 3/6-ch

The two versions differ ONLY in the last step (`baseline_view`); everything upstream is shared, so a
sample is identical between them except for the channel layout. See docs/DATA_HETEROGENEITY.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from data.scripts.curate import accel_units
from data.scripts.assembly import baseline_view
from data.scripts.curate.deployment_policy import StreamSpec, curate_frame


@dataclass(frozen=True)
class Grid:
    """Windowed grid for one dataset stream under one channel alignment."""
    data: np.ndarray            # (N, T, W) float32 — accelerometer in g
    mask: np.ndarray            # (W,) bool — True = real channel, False = zero-pad
    channels: Tuple[str, ...]   # W canonical channel names
    labels: List                # (N,) per-window majority label (empty if no `activity` column)
    alignment: str              # "harmonised" | "non_harmonised"
    dataset: str
    rate_hz: float


def canonicalize_units(curated: pd.DataFrame, dataset: str, channels: Sequence[str]) -> np.ndarray:
    """(T, C) float32 of the curated channels with the ACCELEROMETER rescaled to g.

    Gyroscope columns are left untouched — `accel_units` only ever scales accelerometer axes.
    """
    arr = curated[list(channels)].to_numpy(dtype=np.float32).copy()
    scale = accel_units.accel_scale_factor(dataset)
    if scale != 1.0:
        acc_cols = [i for i, c in enumerate(channels) if accel_units.is_accel_channel(c)]
        arr[:, acc_cols] *= scale
    return arr


def resample_signal(arr: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    """Anti-aliased polyphase resample of a (T, C) signal from `src_hz` to `dst_hz`.

    Uses `scipy.signal.resample_poly` (unity-DC-gain FIR), so gravity-present accel keeps its ~1 g
    magnitude and downsampling introduces no aliasing. No-op when the rates match.
    """
    if src_hz == dst_hz:
        return np.asarray(arr, dtype=np.float32)
    s, d = int(round(src_hz)), int(round(dst_hz))
    g = gcd(s, d)
    return resample_poly(arr, up=d // g, down=s // g, axis=0).astype(np.float32)


def resample_labels(labels, n_out: int):
    """Nearest-neighbor resample of categorical per-sample labels to length `n_out`."""
    if labels is None:
        return None
    labels = np.asarray(labels)
    if len(labels) in (0, n_out):
        return labels
    idx = np.clip(np.round(np.linspace(0, len(labels) - 1, n_out)).astype(int), 0, len(labels) - 1)
    return labels[idx]


def _majority(labels: np.ndarray):
    vals, counts = np.unique(labels, return_counts=True)
    return vals[int(np.argmax(counts))]


def fixed_windows(arr: np.ndarray, window: int, stride: int, labels: Optional[np.ndarray] = None):
    """Fixed-length windows (non-overlapping when stride == window).

    Returns `(windows (N, window, C), win_labels)`. `win_labels` is a per-window majority vote when
    `labels` is given, else an empty list. (Converters emit whole-recording sessions; the grids use
    these simple fixed windows only — there is no activity-aware variable windowing.)
    """
    n = arr.shape[0]
    starts = list(range(0, n - window + 1, stride)) if n >= window else []
    if not starts:
        return np.empty((0, window, arr.shape[1]), np.float32), []
    windows = np.stack([arr[s:s + window] for s in starts]).astype(np.float32)
    if labels is None:
        return windows, []
    labels = np.asarray(labels)
    return windows, [_majority(labels[s:s + window]) for s in starts]


def assemble(raw: pd.DataFrame, dataset: str, spec: StreamSpec, *, alignment: str,
             window: int, rate_hz: float, resample_to: Optional[float] = None,
             stride: Optional[int] = None) -> Grid:
    """Run the full pipeline for one session frame and one channel alignment.

    `spec` is the deployment `StreamSpec` for the device stream
    (`deployment_policy.get_stream_spec(dataset, stream_id)`). `rate_hz` is the session's native rate;
    pass `resample_to` (e.g. 60 for the harmonised corpus) to anti-alias-resample before windowing.
    """
    if alignment not in ("harmonised", "non_harmonised"):
        raise ValueError(f"alignment must be 'harmonised' or 'non_harmonised', got {alignment!r}")

    curated, meta = curate_frame(raw, spec)
    arr = canonicalize_units(curated, dataset, meta.channels)               # (T, C) accel in g
    labels = curated["activity"].to_numpy() if "activity" in curated.columns else None

    rate = float(rate_hz)
    if resample_to is not None and resample_to != rate_hz:
        arr = resample_signal(arr, rate_hz, resample_to)
        labels = resample_labels(labels, arr.shape[0])
        rate = float(resample_to)

    windows, win_labels = fixed_windows(arr, window, stride or window, labels)

    if len(windows) == 0:
        _, out_channels, mask = baseline_view.to_view(
            np.zeros((0, len(meta.channels)), np.float32), meta.channels, alignment)
        data = np.zeros((0, window, len(out_channels)), np.float32)
        return Grid(data, mask, out_channels, [], alignment, dataset, rate)

    n, t, c = windows.shape
    flat, out_channels, mask = baseline_view.to_view(windows.reshape(n * t, c), meta.channels, alignment)
    data = flat.reshape(n, t, len(out_channels)).astype(np.float32)
    return Grid(data, mask, out_channels, win_labels, alignment, dataset, rate)
