"""Shared corpus data-prep for CrossHAR (grid -> the 6-ch / 20 Hz / 120-sample
input contract both the SSL pretrain and the adapter consume).

CrossHAR's verified input contract (BASELINES.md): 6-channel acc+gyro, 20 Hz,
120 samples (= 6 s) per window. Our training grids are stored at their native
rate / window length / channel set, so this module maps a grid window to the
contract:

  * 6 channels in a fixed ``acc_{x,y,z}, gyro_{x,y,z}`` order; accelerometer-only
    train sets (wisdm, unimib_shar, capture24) get their gyro channels
    ZERO-FILLED (CrossHAR per-channel InstanceNorm maps a constant-zero channel
    to zero, so this is a benign "gyro absent" encoding, not fake signal).
  * resampled to 20 Hz (polyphase, anti-aliased) and center-cropped / wrap-padded
    to exactly 120 samples (wrap matches the periodic-padding convention and only
    the <6 s native windows ever need padding).

Values stay in native g here; the CrossHAR-specific normalization (per-channel
InstanceNorm, which discards gravity/DC) is applied downstream by the adapter and
by the upstream SSL pipeline, NOT here.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
from scipy.signal import resample_poly

from eval import data as eval_data

TARGET_HZ = 20
TARGET_LEN = 120          # 6 s @ 20 Hz
SIX_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")

# The 9 non-eval training datasets (the global-vocab train corpus; mirrors the
# harnet adapter's TRAIN_DATASETS). Eval datasets are never touched here.
TRAIN_DATASETS = [
    "uci_har", "hhar", "pamap2", "wisdm", "kuhar", "unimib_shar", "hapt",
    "mhealth", "capture24",
]


def load_grid(dataset: str, stream: str):
    """Read a grid directly (train datasets have no eval_labels.json, so the eval
    loader cannot open them). Returns (windows, labels, subjects, channels, rate)."""
    gdir = eval_data.DATASETS_DIR / dataset / "grids" / "non_harmonised" / stream
    windows = np.load(gdir / "data.npy")
    meta = json.loads((gdir / "meta.json").read_text())
    return (windows, list(meta["labels"]), list(map(str, meta["subjects"])),
            list(meta["channels"]), float(meta["rate_hz"]))


def to_six_channels(windows: np.ndarray, channels: List[str]) -> np.ndarray:
    """(N, T, C) grid -> (N, T, 6) in the fixed acc+gyro order; missing channels
    (e.g. gyro on accel-only sets) are zero-filled."""
    idx = {c: i for i, c in enumerate(channels)}
    out = np.zeros((windows.shape[0], windows.shape[1], len(SIX_CHANNELS)),
                   dtype=np.float32)
    for j, ch in enumerate(SIX_CHANNELS):
        if ch in idx:
            out[:, :, j] = windows[:, :, idx[ch]]
    return out


def resample_crop_pad(windows: np.ndarray, rate_hz: float) -> np.ndarray:
    """(N, T, 6) at `rate_hz` -> (N, 120, 6) at 20 Hz (resample + center-crop /
    wrap-pad), matching harnet's resampling convention at the CrossHAR contract."""
    frac = Fraction(int(round(TARGET_HZ)), int(round(rate_hz))).limit_denominator(1000)
    y = resample_poly(windows.astype(np.float64), frac.numerator, frac.denominator, axis=1)
    L = y.shape[1]
    if L > TARGET_LEN:
        off = (L - TARGET_LEN) // 2
        y = y[:, off:off + TARGET_LEN, :]
    elif L < TARGET_LEN:
        total = TARGET_LEN - L
        left = total // 2
        y = np.pad(y, ((0, 0), (left, total - left), (0, 0)), mode="wrap")
    return y.astype(np.float32)


def grid_to_contract(windows: np.ndarray, channels: List[str], rate_hz: float) -> np.ndarray:
    """Full grid-window -> CrossHAR contract input (N, 120, 6), native g."""
    return resample_crop_pad(to_six_channels(windows, channels), rate_hz)


def iter_train_streams(max_per_stream: int | None = None,
                       seed: int = 3431) -> Iterator[Tuple[str, str, np.ndarray, List[str], np.ndarray]]:
    """Yield ``(dataset, stream, X6, raw_labels, subjects)`` for every training
    stream, where X6 is (n, 120, 6) native-g contract input. `max_per_stream`
    subsamples each stream (used by the smoke pretrain / head-fit)."""
    rng = np.random.RandomState(seed)
    for ds in TRAIN_DATASETS:
        for stream in eval_data.list_streams(ds):
            windows, labels, subjects, channels, rate = load_grid(ds, stream)
            subjects = np.asarray(subjects)
            labels = np.asarray(labels, dtype=object)
            if max_per_stream is not None and len(windows) > max_per_stream:
                sel = rng.choice(len(windows), size=max_per_stream, replace=False)
                windows, labels, subjects = windows[sel], labels[sel], subjects[sel]
            x6 = grid_to_contract(windows, channels, rate)
            yield ds, stream, x6, list(labels), subjects


def build_pretrain_array(max_per_stream: int | None = None,
                         seed: int = 3431) -> np.ndarray:
    """Pool the training corpus into a single (N, 120, 6) array for SSL pretrain
    (labels are irrelevant to self-supervision)."""
    parts = [x6 for _, _, x6, _, _ in iter_train_streams(max_per_stream, seed)]
    data = np.concatenate(parts, axis=0)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(data))
    return data[perm]
