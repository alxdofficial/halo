"""Small encoder-agnostic metric for recurrent-motion affinity."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RecurrentMotionMetric(nn.Module):
    """Normalized projection followed by calibrated cosine affinity.

    Source labels never enter this module. The projection learns which encoder
    dimensions transfer as motion identity; the scalar calibration turns cosine
    similarity into a same-motion logit.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int | None = None,
        *,
        initial_temperature: float = 0.2,
        initial_cosine_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or (projection_dim is not None and projection_dim <= 0):
            raise ValueError("feature dimensions must be positive")
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        if not -1.0 <= initial_cosine_threshold <= 1.0:
            raise ValueError("initial cosine threshold must be in [-1, 1]")
        projection_dim = projection_dim or input_dim
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.projection = nn.Linear(input_dim, projection_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        initial_scale = torch.tensor(1.0 / initial_temperature)
        self.logit_scale = nn.Parameter(torch.log(torch.expm1(initial_scale)))
        self.logit_bias = nn.Parameter(-initial_scale * initial_cosine_threshold)

    def embed(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected input dimension {self.input_dim}, got {embeddings.shape[-1]}"
            )
        return F.normalize(self.projection(embeddings), dim=-1)

    def logits_from_projected(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        if (
            left.shape[-1] != self.projection_dim
            or right.shape[-1] != self.projection_dim
        ):
            raise ValueError("projected features have the wrong final dimension")
        cosine = (left * right).sum(dim=-1).clamp(-1.0, 1.0)
        scale = F.softplus(self.logit_scale) + 1e-4
        return cosine * scale + self.logit_bias

    def pair_logits(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.logits_from_projected(self.embed(left), self.embed(right))

    def pairwise_logits(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return projected embeddings and all symmetric pair logits."""

        projected = self.embed(embeddings)
        cosine = projected @ projected.transpose(-1, -2)
        scale = F.softplus(self.logit_scale) + 1e-4
        return projected, cosine.clamp(-1.0, 1.0) * scale + self.logit_bias
