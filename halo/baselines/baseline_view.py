"""Harmonised vs non-harmonised model-input views over a deployment-curated stream.

`deployment_policy.curate_*` produces ONE physical phone/watch stream as a canonical-order frame
with 3 (accelerometer-only) or 6 (acc+gyro) channels. This module turns that curated array into the
two dataset VERSIONS every downstream model consumes — without ever inventing data:

  * **harmonised**     — the fixed 6-channel canonical layout
                         ``[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]``; an acc-only stream gets
                         ZERO-padded gyro slots plus a per-channel validity mask. One schema for the
                         whole corpus — the "you did the alignment preprocessing" view.
  * **non_harmonised** — the curated stream's NATIVE channel set (3 or 6), canonical order, all-valid
                         mask. Variable width across datasets — the "raw, unforced" view.

Design note (see docs/baselines/BASELINE_FAIRNESS_POLICY.md §2A): because `deployment_policy` already
canonicalises channel *order*, a FIXED-input baseline that pads the non-harmonised view up to its
width lands on the exact same tensor as the harmonised view — i.e. the two views **coincide for
fixed-input models**. The distinction is therefore meaningful only for channel-flexible consumers
(HALO's variable-count tokenizer, NormWear's channel-independent encoder), which can ingest the
native view directly. Padding is **zero-fill + mask only — never random or imputed fill.**
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from halo.data.deployment_policy import STANDARD_CHANNEL_ORDER

# The harmonised schema is exactly the deployment policy's canonical order (width 6).
HARMONISED_CHANNELS: Tuple[str, ...] = tuple(STANDARD_CHANNEL_ORDER)


def _canonical_subset(channels: Sequence[str]) -> Tuple[str, ...]:
    """The given channels reordered to canonical order (STANDARD_CHANNEL_ORDER), dupes/unknowns aside."""
    present = set(channels)
    return tuple(c for c in STANDARD_CHANNEL_ORDER if c in present)


def _place(data: np.ndarray, channels: Sequence[str], target: Sequence[str]):
    """Scatter native `channels` of `data` into the `target` canonical slots.

    Returns (out (T, len(target)) float32, mask (len(target),) bool). A target slot with no matching
    source channel stays exactly 0.0 and its mask entry is False (a genuine absence, never fabricated)."""
    idx = {name: k for k, name in enumerate(channels)}
    out = np.zeros((data.shape[0], len(target)), dtype=np.float32)
    mask = np.zeros(len(target), dtype=bool)
    for j, name in enumerate(target):
        k = idx.get(name)
        if k is not None:
            out[:, j] = data[:, k]
            mask[j] = True
    return out, mask


def to_view(
    data: np.ndarray,
    channels: Sequence[str],
    alignment: str,
    pad_to: int | None = None,
) -> Tuple[np.ndarray, Tuple[str, ...], np.ndarray]:
    """Build a model-input tensor for one channel-alignment regime.

    Args:
        data:      (T, C) curated stream (native channels, any order).
        channels:  the C canonical channel names for `data` (from STANDARD_CHANNEL_ORDER).
        alignment: ``"harmonised"`` -> fixed 6-ch canonical layout + zero-pad+mask;
                   ``"non_harmonised"`` -> the native canonical subset (variable width).
        pad_to:    only for ``non_harmonised`` feeding a FIXED-input model — pad the native view up to
                   this many canonical channels (must be 6, the canonical width). ``None`` keeps the
                   native width. (``non_harmonised`` with ``pad_to=6`` is byte-identical to
                   ``harmonised`` — that is the documented "collapse for fixed models".)

    Returns:
        (out (T, W) float32, out_channels (W,), mask (W,) bool) where mask[j] is True iff column j is a
        real curated channel (False = zero-padded absence).
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != len(channels):
        raise ValueError(f"data shape {data.shape} does not match {len(channels)} channels {tuple(channels)}")
    unknown = [c for c in channels if c not in HARMONISED_CHANNELS]
    if unknown:
        raise ValueError(f"channels {unknown} are not in the canonical schema {HARMONISED_CHANNELS}")

    if alignment == "harmonised":
        target = HARMONISED_CHANNELS
    elif alignment == "non_harmonised":
        if pad_to is None:
            target = _canonical_subset(channels)
        elif pad_to == len(HARMONISED_CHANNELS):
            target = HARMONISED_CHANNELS
        else:
            raise ValueError(f"pad_to must be None or {len(HARMONISED_CHANNELS)} (the canonical width), got {pad_to}")
    else:
        raise ValueError(f"alignment must be 'harmonised' or 'non_harmonised', got {alignment!r}")

    out, mask = _place(data, channels, target)
    return out, tuple(target), mask
