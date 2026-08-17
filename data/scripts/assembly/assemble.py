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
from fractions import Fraction
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
    # Stable physical-event identity per window. Populated by stream_grid(), where the converted
    # session id and local window ordinal are available. It deliberately does not come from labels.
    event_ids: Optional[List[str]] = None
    # True signal length for each row before right-padding. Native HALO grids retain the final
    # partial context of a recording; fixed-layout baseline grids remain full-window only.
    lengths: Optional[np.ndarray] = None


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


#: Largest denominator allowed when expressing dst/src as an exact up/down ratio. `resample_poly`'s
#: filter length grows with the ratio, so this bounds the cost of an awkward rate. 1000 is ample for
#: everything in the corpus: 60/51.2 reduces to 75/64 exactly, and KneE-PAD's 60/148.148 lands
#: within 1e-7 of its true value.
MAX_RESAMPLE_DENOMINATOR = 1000


def resample_signal(arr: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    """Anti-aliased polyphase resample of a (T, C) signal from `src_hz` to `dst_hz`.

    Uses `scipy.signal.resample_poly` (unity-DC-gain FIR), so gravity-present accel keeps its ~1 g
    magnitude and downsampling introduces no aliasing. No-op when the rates match.

    The up/down ratio comes from `Fraction.limit_denominator`, not `round`. Rounding was wrong for
    every non-integer source rate in the corpus: FORTH-TRACE's 51.2 Hz became 51 and KneE-PAD's
    148.148 Hz became 148, which time-stretches the output by 0.39% and 0.10% respectively. Measured
    on the built grids, that turned FORTH-TRACE's 2,082 native windows into 2,083 harmonised ones —
    one window of signal that does not exist. The physical consequence is small (a 2.00 Hz cadence
    reads as 1.992 Hz, far inside any filterbank band) but there is no reason to accept it.
    """
    if src_hz == dst_hz:
        return np.asarray(arr, dtype=np.float32)
    ratio = Fraction(dst_hz / src_hz).limit_denominator(MAX_RESAMPLE_DENOMINATOR)
    return resample_poly(arr, up=ratio.numerator, down=ratio.denominator,
                         axis=0).astype(np.float32)


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


def fixed_windows(arr: np.ndarray, window: int, stride: int, labels: Optional[np.ndarray] = None,
                  *, include_partial: bool = False):
    """Fixed-length windows (non-overlapping when stride == window).

    Returns `(windows (N, window, C), win_labels, lengths)`. When ``include_partial`` is true, the
    final incomplete window is right-padded with zeros and its honest sample count is recorded in
    ``lengths``. `win_labels` is a per-window majority vote over real samples when labels are given.
    """
    n = arr.shape[0]
    starts = list(range(0, n - window + 1, stride)) if n >= window else []
    if include_partial and n > 0:
        next_start = starts[-1] + stride if starts else 0
        if next_start < n:
            starts.append(next_start)
    if not starts:
        return (np.empty((0, window, arr.shape[1]), np.float32), [],
                np.empty(0, dtype=np.int32))
    lengths = np.asarray([min(window, n - start) for start in starts], dtype=np.int32)
    windows = np.zeros((len(starts), window, arr.shape[1]), dtype=np.float32)
    for index, (start, length) in enumerate(zip(starts, lengths.tolist())):
        windows[index, :length] = arr[start:start + length]
    if labels is None:
        return windows, [], lengths
    labels = np.asarray(labels)
    return windows, [_majority(labels[s:s + length])
                     for s, length in zip(starts, lengths.tolist())], lengths


def assemble(raw: pd.DataFrame, dataset: str, spec: StreamSpec, *, alignment: str,
             window: int, rate_hz: float, resample_to: Optional[float] = None,
             stride: Optional[int] = None, include_partial: bool = False) -> Grid:
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

    windows, win_labels, lengths = fixed_windows(
        arr, window, stride or window, labels, include_partial=include_partial,
    )

    if len(windows) == 0:
        _, out_channels, mask = baseline_view.to_view(
            np.zeros((0, len(meta.channels)), np.float32), meta.channels, alignment)
        data = np.zeros((0, window, len(out_channels)), np.float32)
        return Grid(data, mask, out_channels, [], alignment, dataset, rate, lengths=lengths)

    n, t, c = windows.shape
    flat, out_channels, mask = baseline_view.to_view(windows.reshape(n * t, c), meta.channels, alignment)
    data = flat.reshape(n, t, len(out_channels)).astype(np.float32)
    return Grid(data, mask, out_channels, win_labels, alignment, dataset, rate, lengths=lengths)
