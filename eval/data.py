"""Loader for the new-repo grid format, feeding the ZS-XD evaluation.

Each dataset stores windowed grids under::

    data/datasets/<ds>/grids/{harmonised,non_harmonised}/<stream>/
        data.npy   float32 (N, T, C)   accelerometer (+gyro) in g
        mask.npy   bool    (C,)         per-channel validity (False = zero-pad)
        meta.json  {dataset, stream_id, alignment, rate_hz, channels[list],
                    labels[per-window list], subjects[per-window list]}

and a pre-registered candidate label vocabulary at
``data/datasets/<ds>/eval_labels.json`` (the ZS-XD target strings for that
dataset). The global ConSE training vocabulary lives at
``data/labels/global_labels.json``.

Unlike the legacy loader (which majority-voted raw per-timestep activity codes
through an ``idx_to_label`` map, with an offset bug), the grid meta already
carries a decoded per-window label string and subject id — so ground truth is
read directly, offset-free. Native (`non_harmonised`) is the default eval source
because baseline adapters resample per their own input contract; harmonised is
exposed via the `alignment` argument.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO / "data" / "datasets"
GLOBAL_LABELS_PATH = REPO / "data" / "labels" / "global_labels.json"

ALIGNMENTS = ("non_harmonised", "harmonised")


@dataclass
class EvalStream:
    """One dataset/stream grid, ready for model-agnostic scoring.

    Attributes:
        dataset:     dataset name (e.g. ``"motionsense"``).
        stream:      stream / placement id (e.g. ``"phone_front_pocket"``).
        alignment:   ``"non_harmonised"`` (native channels/rate) or ``"harmonised"``.
        windows:     (N, T, C) float32 sensor windows (accel + optional gyro), in g.
        gt:          per-window ground-truth label strings, length N (verbatim from
                     the grid meta — NOT yet filtered to `eval_labels`).
        subjects:    (N,) subject ids, one per window.
        channels:    the C channel names of `windows`, in grid order.
        rate_hz:     sampling rate of `windows`.
        mask:        (C,) bool per-channel validity (False = zero-padded absence).
        eval_labels: the dataset's pre-registered ZS-XD candidate label vocabulary.
    """
    dataset: str
    stream: str
    alignment: str
    windows: np.ndarray
    gt: List[str]
    subjects: np.ndarray
    channels: List[str]
    rate_hz: float
    mask: np.ndarray
    eval_labels: List[str]

    @property
    def n_windows(self) -> int:
        return self.windows.shape[0]


def _grid_dir(dataset: str, stream: str, alignment: str) -> Path:
    if alignment not in ALIGNMENTS:
        raise ValueError(f"alignment must be one of {ALIGNMENTS}, got {alignment!r}")
    return DATASETS_DIR / dataset / "grids" / alignment / stream


def list_streams(dataset: str, alignment: str = "non_harmonised") -> List[str]:
    """Stream ids available for a dataset under the given alignment."""
    root = DATASETS_DIR / dataset / "grids" / alignment
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def load_eval_labels(dataset: str) -> List[str]:
    """The dataset's pre-registered ZS-XD candidate label vocabulary."""
    path = DATASETS_DIR / dataset / "eval_labels.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No eval_labels.json for '{dataset}' at {path}. This dataset is not "
            "set up as a ZS-XD evaluation target."
        )
    return list(json.loads(path.read_text())["labels"])


def load_global_labels() -> List[str]:
    """The global ConSE training-label vocabulary (closed-vocab baselines)."""
    if not GLOBAL_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Global label vocabulary missing at {GLOBAL_LABELS_PATH}. Run "
            "`python -m data.scripts.labels.build_global_label_mapping`."
        )
    return list(json.loads(GLOBAL_LABELS_PATH.read_text())["labels"])


def load_eval_stream(
    dataset: str,
    stream: str,
    alignment: str = "non_harmonised",
) -> EvalStream:
    """Load one dataset/stream grid as an :class:`EvalStream`.

    Args:
        dataset:   dataset name under ``data/datasets/``.
        stream:    stream / placement id (see :func:`list_streams`).
        alignment: ``"non_harmonised"`` (default; native channels/rate — the eval
                   source, since adapters resample per baseline) or ``"harmonised"``.

    The returned `gt` / `subjects` are 1:1 with `windows` (length N) and verbatim
    from the grid — restrict `gt` to `eval_labels` at scoring time via
    :func:`eval.scoring.filter_ground_truth`.
    """
    gdir = _grid_dir(dataset, stream, alignment)
    if not gdir.exists():
        avail = list_streams(dataset, alignment)
        raise FileNotFoundError(
            f"No grid for {dataset}/{stream} ({alignment}) at {gdir}. "
            f"Available {alignment} streams: {avail}"
        )

    windows = np.load(gdir / "data.npy")
    mask = np.load(gdir / "mask.npy")
    meta = json.loads((gdir / "meta.json").read_text())

    gt = list(meta["labels"])
    subjects = np.asarray(meta["subjects"])
    channels = list(meta["channels"])

    # Structural invariants — fail loud rather than silently misalign scoring.
    n = windows.shape[0]
    if not (len(gt) == len(subjects) == n):
        raise ValueError(
            f"{dataset}/{stream}: meta labels ({len(gt)}) / subjects "
            f"({len(subjects)}) do not match window count ({n})."
        )
    if windows.shape[2] != len(channels):
        raise ValueError(
            f"{dataset}/{stream}: window channel dim ({windows.shape[2]}) != "
            f"len(channels) ({len(channels)})."
        )
    if mask.shape != (len(channels),):
        raise ValueError(
            f"{dataset}/{stream}: mask shape {mask.shape} != ({len(channels)},)."
        )

    return EvalStream(
        dataset=dataset,
        stream=stream,
        alignment=alignment,
        windows=windows,
        gt=gt,
        subjects=subjects,
        channels=channels,
        rate_hz=float(meta["rate_hz"]),
        mask=mask.astype(bool),
        eval_labels=load_eval_labels(dataset),
    )
