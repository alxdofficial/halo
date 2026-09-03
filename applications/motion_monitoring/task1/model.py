"""Small differentiable matching head for Task 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from applications.motion_monitoring.task1.episodes import DetectionBatch
from applications.motion_monitoring.task1.matcher import (
    TemporalMatch,
    full_timeline_matches,
)


@dataclass(frozen=True)
class AlignmentOutput:
    endpoint_logits: torch.Tensor
    endpoint_valid: torch.Tensor
    reference_embeddings: torch.Tensor
    query_embeddings: torch.Tensor


def _soft_min(values: torch.Tensor, smoothing: float) -> torch.Tensor:
    return -smoothing * torch.logsumexp(-values / smoothing, dim=0)


def _bounded_soft_dtw_endpoints(
    reference: torch.Tensor,
    query: torch.Tensor,
    *,
    smoothing: float,
    warp_penalty: float,
) -> torch.Tensor:
    """Open-begin soft-DTW with no consecutive non-diagonal moves."""

    cost = 1.0 - torch.clamp(reference @ query.T, -1.0, 1.0)
    query_length = len(query)
    sentinel = cost.new_tensor(1e4)
    diagonal = [cost.new_zeros(()) for _ in range(query_length + 1)]
    vertical = [sentinel for _ in range(query_length + 1)]
    horizontal = [sentinel for _ in range(query_length + 1)]

    for reference_index in range(len(reference)):
        next_diagonal = [sentinel]
        next_vertical = [sentinel]
        next_horizontal = [sentinel]
        for query_index in range(1, query_length + 1):
            local = cost[reference_index, query_index - 1]
            next_diagonal.append(
                local
                + _soft_min(
                    torch.stack(
                        [
                            diagonal[query_index - 1],
                            vertical[query_index - 1],
                            horizontal[query_index - 1],
                        ]
                    ),
                    smoothing,
                )
            )
            next_vertical.append(
                local
                + warp_penalty
                + _soft_min(
                    torch.stack([diagonal[query_index], horizontal[query_index]]),
                    smoothing,
                )
            )
            next_horizontal.append(
                local
                + warp_penalty
                + _soft_min(
                    torch.stack([next_diagonal[-2], next_vertical[-2]]), smoothing
                )
            )
        diagonal, vertical, horizontal = (
            next_diagonal,
            next_vertical,
            next_horizontal,
        )
    endpoint_costs = torch.stack(
        [
            _soft_min(torch.stack(states), smoothing)
            for states in zip(diagonal[1:], vertical[1:], horizontal[1:])
        ]
    )
    return endpoint_costs / len(reference)


def _valid_runs(mask: torch.Tensor) -> list[tuple[int, int]]:
    padded = F.pad(mask.to(torch.int8), (1, 1))
    changes = torch.diff(padded)
    starts = torch.nonzero(changes == 1, as_tuple=False).flatten().tolist()
    ends = torch.nonzero(changes == -1, as_tuple=False).flatten().tolist()
    return list(zip(starts, ends))


class DifferentiableSubsequenceMatcher(nn.Module):
    """Learn a small metric while retaining subsequence alignment as the decoder."""

    def __init__(
        self,
        feature_dim: int,
        *,
        projection_dim: int | None = None,
        smoothing: float = 0.1,
        warp_penalty: float = 0.05,
        score_temperature: float = 0.2,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or (projection_dim is not None and projection_dim <= 0):
            raise ValueError("feature dimensions must be positive")
        if smoothing <= 0 or score_temperature <= 0 or warp_penalty < 0:
            raise ValueError(
                "alignment scales must be positive and warp penalty non-negative"
            )
        projection_dim = projection_dim or feature_dim
        self.projection = nn.Linear(feature_dim, projection_dim, bias=False)
        if projection_dim == feature_dim:
            nn.init.eye_(self.projection.weight)
        else:
            nn.init.orthogonal_(self.projection.weight)
        self.score_bias = nn.Parameter(torch.zeros(()))
        self.smoothing = float(smoothing)
        self.warp_penalty = float(warp_penalty)
        self.score_temperature = float(score_temperature)

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(embeddings), dim=-1, eps=1e-8)

    @torch.no_grad()
    def detect(
        self,
        reference: np.ndarray | torch.Tensor,
        query: np.ndarray | torch.Tensor,
        query_intervals_sec: np.ndarray,
        *,
        score_threshold: float,
        query_valid: np.ndarray | torch.Tensor | None = None,
        nms_iou: float = 0.3,
        max_detections: int | None = None,
    ) -> list[TemporalMatch]:
        """Project embeddings and match each quality-contiguous query run."""

        device = self.projection.weight.device
        dtype = self.projection.weight.dtype
        reference_tensor = torch.as_tensor(reference, dtype=dtype, device=device)
        query_tensor = torch.as_tensor(query, dtype=dtype, device=device)
        if (
            reference_tensor.ndim != 2
            or query_tensor.ndim != 2
            or reference_tensor.shape[1] != query_tensor.shape[1]
        ):
            raise ValueError(
                "reference and query must be [time, feature] with matching features"
            )
        projected_reference = self.project(reference_tensor).cpu().numpy()
        projected_query = self.project(query_tensor).cpu().numpy()
        intervals = np.asarray(query_intervals_sec, dtype=np.float64)
        if intervals.shape != (len(projected_query), 2):
            raise ValueError("query_intervals_sec must have shape [query_time, 2]")
        if not np.isfinite(intervals).all() or np.any(
            intervals[:, 1] <= intervals[:, 0]
        ):
            raise ValueError("query intervals must be finite with positive duration")
        if np.any(np.diff(intervals, axis=0) < 0):
            raise ValueError("query intervals must be ordered in physical time")
        if np.any(np.linalg.norm(projected_reference, axis=1) <= 1e-12):
            raise ValueError("projected reference patches must remain non-zero")
        valid = (
            np.ones(len(projected_query), dtype=np.bool_)
            if query_valid is None
            else np.asarray(torch.as_tensor(query_valid).detach().cpu(), dtype=np.bool_)
        )
        if valid.shape != (len(projected_query),):
            raise ValueError("query_valid must have shape [query_time]")
        if np.any(np.linalg.norm(projected_query[valid], axis=1) <= 1e-12):
            raise ValueError("valid projected query patches must remain non-zero")
        padded = np.pad(valid.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        matches: list[TemporalMatch] = []
        for start, end in zip(starts, ends, strict=True):
            if end - start < (len(projected_reference) + 1) // 2:
                continue
            for match in full_timeline_matches(
                projected_reference,
                projected_query[start:end],
                intervals[start:end],
                warp_penalty=self.warp_penalty,
                score_threshold=score_threshold,
                nms_iou=nms_iou,
            ):
                matches.append(
                    TemporalMatch(
                        start_patch=match.start_patch + int(start),
                        end_patch=match.end_patch + int(start),
                        start_sec=match.start_sec,
                        end_sec=match.end_sec,
                        score=match.score,
                        path_length=match.path_length,
                        duration_ratio=match.duration_ratio,
                    )
                )
        if max_detections is not None:
            if max_detections <= 0:
                raise ValueError("max_detections must be positive when provided")
            matches = sorted(matches, key=lambda item: item.score)[:max_detections]
        return sorted(matches, key=lambda item: item.start_sec)

    def forward(self, batch: DetectionBatch) -> AlignmentOutput:
        reference = self.project(batch.reference)
        query = self.project(batch.query)
        rows: list[torch.Tensor] = []
        valid_rows: list[torch.Tensor] = []
        for batch_index in range(len(reference)):
            reference_indices = torch.nonzero(
                batch.reference_valid[batch_index], as_tuple=False
            ).flatten()
            if len(reference_indices) > 1 and torch.any(
                reference_indices[1:] != reference_indices[:-1] + 1
            ):
                raise ValueError(
                    "reference validity must describe one contiguous execution"
                )
            valid_reference = reference[batch_index, reference_indices]
            chunks: list[torch.Tensor] = []
            valid_chunks: list[torch.Tensor] = []
            cursor = 0
            alignment_valid = batch.query_valid[batch_index] & batch.alignment_valid[batch_index]
            for start, end in _valid_runs(alignment_valid):
                if start > cursor:
                    chunks.append(query.new_full((start - cursor,), -1e4))
                    valid_chunks.append(
                        torch.zeros(
                            start - cursor, dtype=torch.bool, device=query.device
                        )
                    )
                costs = _bounded_soft_dtw_endpoints(
                    valid_reference,
                    query[batch_index, start:end],
                    smoothing=self.smoothing,
                    warp_penalty=self.warp_penalty,
                )
                chunks.append(self.score_bias - costs / self.score_temperature)
                minimum_query_patches = (len(valid_reference) + 1) // 2
                valid_chunks.append(
                    torch.arange(end - start, device=query.device)
                    >= minimum_query_patches - 1
                )
                cursor = end
            if cursor < query.shape[1]:
                chunks.append(query.new_full((query.shape[1] - cursor,), -1e4))
                valid_chunks.append(
                    torch.zeros(
                        query.shape[1] - cursor,
                        dtype=torch.bool,
                        device=query.device,
                    )
                )
            rows.append(torch.cat(chunks))
            valid_rows.append(torch.cat(valid_chunks))
        return AlignmentOutput(
            torch.stack(rows), torch.stack(valid_rows), reference, query
        )
