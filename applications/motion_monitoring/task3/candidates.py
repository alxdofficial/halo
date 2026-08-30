"""Physical-time multiscale candidate pooling and exact-event assignment."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .contracts import CandidateBatch, CandidateTargets, EventBatch


def _validate_timeline_inputs(
    patch_embeddings: torch.Tensor,
    patch_intervals_sec: torch.Tensor,
    patch_mask: torch.Tensor,
    durations_sec: Sequence[float],
    candidate_stride_sec: float | None,
) -> tuple[float, ...]:
    if patch_embeddings.ndim != 3:
        raise ValueError("patch_embeddings must have shape [batch, time, feature]")
    if not patch_embeddings.is_floating_point():
        raise TypeError("patch_embeddings must use a floating-point dtype")
    if patch_intervals_sec.shape != (*patch_embeddings.shape[:2], 2):
        raise ValueError("patch_intervals_sec must have shape [batch, time, 2]")
    if not patch_intervals_sec.is_floating_point():
        raise TypeError("patch_intervals_sec must use a floating-point dtype")
    if patch_mask.shape != patch_embeddings.shape[:2] or patch_mask.dtype != torch.bool:
        raise ValueError("patch_mask must be boolean with shape [batch, time]")
    if (
        patch_intervals_sec.device != patch_embeddings.device
        or patch_mask.device != patch_embeddings.device
    ):
        raise ValueError("patch embeddings, intervals, and mask must share one device")
    if not torch.isfinite(patch_embeddings[patch_mask]).all():
        raise ValueError("valid patch embeddings must be finite")
    if not torch.isfinite(patch_intervals_sec[patch_mask]).all():
        raise ValueError("valid patch intervals must be finite")
    if not torch.all(
        patch_intervals_sec[..., 1][patch_mask]
        > patch_intervals_sec[..., 0][patch_mask]
    ):
        raise ValueError("valid patch intervals must have positive duration")
    for batch_index in range(len(patch_embeddings)):
        valid_intervals = patch_intervals_sec[batch_index, patch_mask[batch_index]]
        if len(valid_intervals) > 1 and (
            torch.any(valid_intervals[1:, 0] < valid_intervals[:-1, 0])
            or torch.any(valid_intervals[1:, 1] < valid_intervals[:-1, 1])
        ):
            raise ValueError("valid patch intervals must be ordered in physical time")
    durations = tuple(float(value) for value in durations_sec)
    if not durations or any(
        not torch.isfinite(torch.tensor(value)) or value <= 0 for value in durations
    ):
        raise ValueError("durations_sec must contain positive finite values")
    if tuple(sorted(set(durations))) != durations:
        raise ValueError("durations_sec must be unique and strictly increasing")
    if candidate_stride_sec is not None and candidate_stride_sec <= 0:
        raise ValueError("candidate_stride_sec must be positive")
    return durations


def _candidate_spans(
    intervals: torch.Tensor,
    valid: torch.Tensor,
    durations_sec: tuple[float, ...],
    candidate_stride_sec: float | None,
) -> tuple[list[int], list[int], list[int]]:
    starts: list[int] = []
    ends: list[int] = []
    scales: list[int] = []
    valid_indices = valid.nonzero(as_tuple=False).flatten().tolist()
    if not valid_indices:
        return starts, ends, scales

    for scale_index, duration in enumerate(durations_sec):
        last_start = -float("inf")
        for start in valid_indices:
            start_time = float(intervals[start, 0].item())
            if (
                candidate_stride_sec is not None
                and start_time < last_start + candidate_stride_sec - 1e-9
            ):
                continue
            target_end = start_time + duration
            later_ends = intervals[start:, 1]
            end_offset = int(
                torch.searchsorted(later_ends.contiguous(), target_end).item()
            )
            end = start + end_offset + 1
            if (
                end > len(valid)
                or float(intervals[end - 1, 1].item()) + 1e-9 < target_end
            ):
                continue
            if not bool(valid[start:end].all()):
                continue
            # Do not bridge timestamp discontinuities hidden inside an otherwise valid mask.
            if end - start > 1:
                gaps = intervals[start + 1 : end, 0] - intervals[start : end - 1, 1]
                typical = torch.median(
                    intervals[start:end, 1] - intervals[start:end, 0]
                )
                if bool(torch.any(gaps > torch.clamp(typical * 0.5, min=1e-6))):
                    continue
            starts.append(start)
            ends.append(end)
            scales.append(scale_index)
            last_start = start_time
    return starts, ends, scales


def pool_multiscale_candidates(
    patch_embeddings: torch.Tensor,
    patch_intervals_sec: torch.Tensor,
    patch_mask: torch.Tensor,
    *,
    durations_sec: Sequence[float],
    candidate_stride_sec: float | None = None,
    normalize: bool = True,
) -> CandidateBatch:
    """Pool dense candidates over physical durations without re-encoding.

    Pooling is weighted by each patch's represented duration. Invalid patches,
    padded candidates, and spans crossing hard timestamp gaps are excluded.
    Gradients remain connected to every contributing patch embedding.
    """

    durations = _validate_timeline_inputs(
        patch_embeddings,
        patch_intervals_sec,
        patch_mask,
        durations_sec,
        candidate_stride_sec,
    )
    batch_size, _, feature_dim = patch_embeddings.shape
    spans = [
        _candidate_spans(
            patch_intervals_sec[index],
            patch_mask[index],
            durations,
            candidate_stride_sec,
        )
        for index in range(batch_size)
    ]
    max_candidates = max((len(item[0]) for item in spans), default=0)
    if max_candidates == 0:
        raise ValueError(
            "no valid candidates can be formed from the requested durations"
        )

    device = patch_embeddings.device
    embeddings = patch_embeddings.new_zeros((batch_size, max_candidates, feature_dim))
    candidate_mask = torch.zeros(
        (batch_size, max_candidates), dtype=torch.bool, device=device
    )
    # Physical clocks remain in the interval dtype. Unix timestamps collapse
    # one-second candidates when copied into float32.
    start_sec = patch_intervals_sec.new_zeros((batch_size, max_candidates))
    end_sec = patch_intervals_sec.new_zeros((batch_size, max_candidates))
    scale_index = torch.full(
        (batch_size, max_candidates), -1, dtype=torch.long, device=device
    )
    start_patch = torch.full_like(scale_index, -1)
    end_patch = torch.full_like(scale_index, -1)
    recording_id = (
        torch.arange(batch_size, device=device)[:, None]
        .expand(-1, max_candidates)
        .clone()
    )

    patch_duration = (
        patch_intervals_sec[..., 1] - patch_intervals_sec[..., 0]
    ).to(patch_embeddings.dtype)
    weighted = patch_embeddings * patch_duration.unsqueeze(-1)
    weighted_prefix = torch.cat(
        [weighted.new_zeros((batch_size, 1, feature_dim)), weighted.cumsum(dim=1)],
        dim=1,
    )
    duration_prefix = torch.cat(
        [patch_duration.new_zeros((batch_size, 1)), patch_duration.cumsum(dim=1)], dim=1
    )

    for batch_index, (starts, ends, scales) in enumerate(spans):
        count = len(starts)
        if not count:
            continue
        starts_tensor = torch.tensor(starts, dtype=torch.long, device=device)
        ends_tensor = torch.tensor(ends, dtype=torch.long, device=device)
        total_duration = (
            duration_prefix[batch_index, ends_tensor]
            - duration_prefix[batch_index, starts_tensor]
        )
        pooled = (
            weighted_prefix[batch_index, ends_tensor]
            - weighted_prefix[batch_index, starts_tensor]
        ) / total_duration.unsqueeze(-1).clamp_min(
            torch.finfo(patch_embeddings.dtype).eps
        )
        if normalize:
            pooled = F.normalize(pooled, dim=-1)
        embeddings[batch_index, :count] = pooled
        candidate_mask[batch_index, :count] = True
        start_sec[batch_index, :count] = patch_intervals_sec[
            batch_index, starts_tensor, 0
        ]
        end_sec[batch_index, :count] = patch_intervals_sec[
            batch_index, ends_tensor - 1, 1
        ]
        scale_index[batch_index, :count] = torch.tensor(
            scales, dtype=torch.long, device=device
        )
        start_patch[batch_index, :count] = starts_tensor
        end_patch[batch_index, :count] = ends_tensor

    return CandidateBatch(
        embeddings=embeddings,
        candidate_mask=candidate_mask,
        start_sec=start_sec,
        end_sec=end_sec,
        scale_index=scale_index,
        start_patch=start_patch,
        end_patch=end_patch,
        recording_id=recording_id,
    )


def assign_event_targets(
    candidates: CandidateBatch,
    events: EventBatch,
    *,
    positive_iou: float = 0.7,
    background_overlap_iou: float = 0.05,
) -> CandidateTargets:
    """Assign exact event identities to candidates using temporal IoU.

    Partial overlaps remain unassigned. A candidate is background only when its
    source annotation track is declared exhaustive and it has negligible overlap
    with every event. Background regions are never treated as one positive class.
    """

    if not 0 < positive_iou <= 1:
        raise ValueError("positive_iou must be in (0, 1]")
    if not 0 <= background_overlap_iou < positive_iou:
        raise ValueError("background_overlap_iou must be in [0, positive_iou)")
    if events.start_sec.shape[0] != candidates.embeddings.shape[0]:
        raise ValueError("events and candidates must have the same batch size")
    if events.start_sec.shape[1] == 0:
        invalid = torch.full_like(candidates.start_patch, -1)
        return CandidateTargets(
            label_id=invalid,
            instance_id=invalid.clone(),
            scope_id=invalid.clone(),
            assigned_mask=torch.zeros_like(candidates.candidate_mask),
            background_mask=candidates.candidate_mask & events.exhaustive.unsqueeze(1),
            best_iou=torch.zeros_like(candidates.start_sec),
        )

    candidate_start = candidates.start_sec.unsqueeze(-1)
    candidate_end = candidates.end_sec.unsqueeze(-1)
    event_start = events.start_sec.unsqueeze(1)
    event_end = events.end_sec.unsqueeze(1)
    intersection = (
        torch.minimum(candidate_end, event_end)
        - torch.maximum(candidate_start, event_start)
    ).clamp_min(0)
    union = torch.maximum(candidate_end, event_end) - torch.minimum(
        candidate_start, event_start
    )
    iou = intersection / union.clamp_min(torch.finfo(candidates.embeddings.dtype).eps)
    iou = iou.masked_fill(~events.event_mask.unsqueeze(1), -1.0)
    best_iou, best_event = iou.max(dim=-1)

    gathered_label = events.label_id.gather(1, best_event)
    gathered_instance = events.instance_id.gather(1, best_event)
    gathered_scope = events.scope_id.gather(1, best_event)
    assigned = candidates.candidate_mask & (best_iou >= positive_iou)
    background = (
        candidates.candidate_mask
        & events.exhaustive.unsqueeze(1)
        & (best_iou <= background_overlap_iou)
    )
    invalid = torch.full_like(gathered_label, -1)
    return CandidateTargets(
        label_id=torch.where(assigned, gathered_label, invalid),
        instance_id=torch.where(assigned, gathered_instance, invalid),
        scope_id=torch.where(assigned, gathered_scope, invalid),
        assigned_mask=assigned,
        background_mask=background,
        best_iou=best_iou.clamp_min(0),
    )
