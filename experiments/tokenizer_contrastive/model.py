"""Minimal window encoder around the reference HALO physical-Hz tokenizer."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .config import ExperimentConfig


TOKENIZER_RELATIVE_PATH = Path("model/feature_extractor.py")


def load_tokenizer_class(legacy_root: str | Path):
    source = Path(legacy_root) / TOKENIZER_RELATIVE_PATH
    if not source.exists():
        raise FileNotFoundError(
            f"HALO V2 tokenizer source not found at {source}. "
            "Pass --legacy-root until the model is ported into this repository."
        )
    module_name = "halo_contrastive_reference_feature_extractor"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import tokenizer source {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PhysicalFilterbankTokenizer, source


def tokenizer_source_metadata(legacy_root: str | Path) -> dict[str, str]:
    source = Path(legacy_root) / TOKENIZER_RELATIVE_PATH
    return {
        "path": str(source.resolve()),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


class ContrastiveTokenizerEncoder(nn.Module):
    """Train the tokenizer projection with minimal masked window pooling.

    The physical-Hz filterbank centers remain fixed. The learned components are the
    tokenizer projection, modality embeddings, representation MLP, and disposable
    contrastive projection head.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        tokenizer_class, _ = load_tokenizer_class(config.legacy_root)
        self.patch_seconds = config.patch_seconds
        self.dft_size = config.dft_size
        self.tokenizer = tokenizer_class(
            d_model=config.token_dim,
            dft_size=config.dft_size,
            learnable=False,
            norm="none",
        )
        self.modality_embedding = nn.Embedding(2, config.token_dim)
        self.register_buffer(
            "modality_ids", torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        )
        self.token_norm = nn.LayerNorm(config.token_dim)
        self.representation_head = nn.Sequential(
            nn.Linear(6 * config.token_dim + 6, config.representation_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.representation_dim, config.representation_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(config.representation_dim, config.representation_dim),
            nn.GELU(),
            nn.Linear(config.representation_dim, config.projection_dim),
        )

    def _patchify(
        self, windows: torch.Tensor, rates: torch.Tensor, window_lengths: torch.Tensor
    ):
        batch, time, channels = windows.shape
        patch_lengths = torch.round(rates * self.patch_seconds).to(torch.long)
        if not torch.all(patch_lengths == patch_lengths[0]):
            raise ValueError("A batch must have one patch length; use harmonised grids")
        patch_length = int(patch_lengths[0].item())
        if patch_length > self.dft_size:
            raise ValueError(
                f"patch length {patch_length} exceeds dft_size={self.dft_size}"
            )
        patch_counts = torch.div(window_lengths, patch_length, rounding_mode="floor")
        patch_count = int(patch_counts.max().item())
        if patch_count < 1 or torch.any(patch_counts < 1):
            raise ValueError("Window is shorter than one tokenizer patch")
        cropped = windows[:, :patch_count * patch_length]
        patches = cropped.reshape(batch, patch_count, patch_length, channels)
        padded = windows.new_zeros(batch, patch_count, self.dft_size, channels)
        padded[:, :, :patch_length] = patches
        patch_mask = (
            torch.arange(patch_count, device=windows.device)[None, :]
            < patch_counts[:, None]
        )
        return padded, patch_lengths, patch_mask

    def encode(
        self,
        windows: torch.Tensor,
        rates: torch.Tensor,
        channel_mask: torch.Tensor,
        window_lengths: torch.Tensor,
    ) -> torch.Tensor:
        patches, patch_lengths, patch_mask = self._patchify(
            windows, rates, window_lengths
        )
        tokens = self.tokenizer(patches, rates, patch_lengths)
        if tokens.shape[2] != 6:
            raise ValueError(f"Expected six harmonised channels, got {tokens.shape[2]}")
        tokens = tokens + self.modality_embedding(self.modality_ids).view(1, 1, 6, -1)
        tokens = self.token_norm(tokens)
        patch_weights = patch_mask[:, :, None, None].to(tokens.dtype)
        patch_denominator = patch_weights.sum(dim=1).clamp_min(1.0)
        channel_tokens = (tokens * patch_weights).sum(dim=1) / patch_denominator
        channel_tokens = channel_tokens * channel_mask[:, :, None].to(tokens.dtype)
        # Preserve ordered axis tokens so invariance must be learned. Averaging xyz here
        # would make summed band energy invariant before the contrastive objective sees it.
        flattened = channel_tokens.reshape(len(windows), -1)
        representation_input = torch.cat(
            [flattened, channel_mask.to(tokens.dtype)], dim=-1
        )
        return self.representation_head(representation_input)

    def forward(
        self,
        windows: torch.Tensor,
        rates: torch.Tensor,
        channel_mask: torch.Tensor,
        window_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        representation = self.encode(windows, rates, channel_mask, window_lengths)
        projection = self.projector(representation)
        return {
            "representation": F.normalize(representation, dim=-1),
            "projection": F.normalize(projection, dim=-1),
        }
