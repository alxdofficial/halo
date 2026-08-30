"""Deterministic synthetic Task-1 episodes for mechanical smoke tests."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from applications.motion_monitoring.task1.episodes import (
    DetectionEpisode,
    EmbeddingSequence,
)


def _normalized(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values, dim=-1, eps=1e-8)


class SyntheticDetectionDataset(Dataset[DetectionEpisode]):
    """Generate independent references, positives, distractors, and absent queries."""

    def __init__(
        self,
        episode_count: int = 128,
        *,
        feature_dim: int = 16,
        query_patches: int = 120,
        reference_patches: int = 6,
        target_present_probability: float = 0.75,
        seed: int = 0,
    ) -> None:
        if episode_count <= 0 or feature_dim < 4:
            raise ValueError(
                "episode_count must be positive and feature_dim at least four"
            )
        if query_patches < 3 * reference_patches or reference_patches < 3:
            raise ValueError("query must fit targets, distractors, and background")
        if not 0 <= target_present_probability <= 1:
            raise ValueError("target_present_probability must be in [0, 1]")
        self.episode_count = episode_count
        self.feature_dim = feature_dim
        self.query_patches = query_patches
        self.reference_patches = reference_patches
        self.target_present_probability = target_present_probability
        self.seed = seed

    def __len__(self) -> int:
        return self.episode_count

    def _generator(self, index: int) -> torch.Generator:
        return torch.Generator().manual_seed(self.seed + index * 104729)

    def _action(self, generator: torch.Generator) -> torch.Tensor:
        time = torch.linspace(0, 1, self.reference_patches)
        frequencies = torch.arange(1, self.feature_dim + 1, dtype=torch.float32)
        phase = torch.rand(self.feature_dim, generator=generator) * (2 * math.pi)
        values = torch.sin(2 * math.pi * time[:, None] * frequencies[None] / 5 + phase)
        values += 0.35 * torch.cos(2 * math.pi * time[:, None] * frequencies[None] / 3)
        return _normalized(values)

    def __getitem__(self, index: int) -> DetectionEpisode:
        if not 0 <= index < len(self):
            raise IndexError(index)
        generator = self._generator(index)
        action = self._action(generator)
        reference = _normalized(
            action + 0.04 * torch.randn(action.shape, generator=generator)
        )
        query = _normalized(
            torch.randn(self.query_patches, self.feature_dim, generator=generator)
        )

        # A coherent non-target movement makes structured motion alone insufficient.
        distractor = self._action(generator)
        distractor_start = self.query_patches - self.reference_patches - 2
        query[distractor_start : distractor_start + self.reference_patches] = distractor

        targets: list[tuple[float, float]] = []
        if torch.rand((), generator=generator) < self.target_present_probability:
            duration = int(
                torch.randint(
                    max(3, self.reference_patches - 2),
                    self.reference_patches + 3,
                    (),
                    generator=generator,
                )
            )
            warped = (
                F.interpolate(
                    action.T.unsqueeze(0),
                    size=duration,
                    mode="linear",
                    align_corners=True,
                )
                .squeeze(0)
                .T
            )
            warped = _normalized(
                warped + 0.05 * torch.randn(warped.shape, generator=generator)
            )
            latest = distractor_start - duration - 2
            start = int(torch.randint(2, latest + 1, (), generator=generator))
            query[start : start + duration] = warped
            targets.append((float(start), float(start + duration)))

        intervals = torch.column_stack(
            [torch.arange(self.query_patches), torch.arange(1, self.query_patches + 1)]
        ).float()
        reference_intervals = torch.column_stack(
            [
                torch.arange(self.reference_patches),
                torch.arange(1, self.reference_patches + 1),
            ]
        ).float()
        return DetectionEpisode(
            reference=EmbeddingSequence(
                reference,
                reference_intervals,
                torch.ones(self.reference_patches, dtype=torch.bool),
            ),
            query=EmbeddingSequence(
                query,
                intervals,
                torch.ones(self.query_patches, dtype=torch.bool),
            ),
            targets_sec=torch.tensor(targets, dtype=torch.float32).reshape(-1, 2),
            metadata={"synthetic_index": index},
        )
