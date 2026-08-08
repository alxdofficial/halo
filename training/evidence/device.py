"""Device selection helpers for Phase-B commands."""

from __future__ import annotations

import torch


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit device request without silently changing execution hardware."""
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested!r} was requested, but PyTorch cannot initialize CUDA. "
            "Fix the driver/runtime or pass --device cpu explicitly."
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            f"MPS device {requested!r} was requested, but MPS is unavailable. "
            "Pass --device cpu explicitly to run on CPU."
        )
    return device
