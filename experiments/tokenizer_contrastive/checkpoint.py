"""Portable checkpoint helpers for the contrastive tokenizer experiment."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .config import ExperimentConfig
from .model import ContrastiveTokenizerEncoder, tokenizer_source_metadata


def save_checkpoint(
    path: Path,
    model: ContrastiveTokenizerEncoder,
    optimizer,
    config: ExperimentConfig,
    epoch: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format_version": 1,
        "config": config.to_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "tokenizer_source": tokenizer_source_metadata(config.legacy_root),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
    }, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    device: torch.device,
    legacy_root: str | Path | None = None,
):
    payload = torch.load(path, map_location=device, weights_only=False)
    config = ExperimentConfig.from_dict(payload["config"])
    if legacy_root is not None:
        config.legacy_root = str(legacy_root)
    current_source = tokenizer_source_metadata(config.legacy_root)
    recorded_source = payload.get("tokenizer_source")
    if recorded_source and current_source["sha256"] != recorded_source["sha256"]:
        raise RuntimeError(
            "Tokenizer source hash differs from the source used to create this checkpoint"
        )
    model = ContrastiveTokenizerEncoder(config).to(device)
    model.load_state_dict(payload["model_state"])
    return model, config, payload
