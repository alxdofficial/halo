"""The two universal, label-free Phase-A objectives.

``JEPA`` masks physical-time intervals and channels in a student view, then predicts the
corresponding contextual tokens from a clean EMA teacher. ``VICReg`` aligns two independent
augmentations of every window while preserving per-dimension variance and reducing redundancy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------
# Defaults (M2 starting points — swept later, all in one place)
# ----------------------------------------------------------------------------------------------
MASK_RATIO_TIME = 0.5        # fraction of time steps masked for JEPA prediction
MASK_RATIO_CHANNEL = 0.25    # fraction of batches that get a whole-channel mask event
GYRO_DROP_BIAS = 0.7         # within channel-mask events, P(drop the whole gyro triad)
MIN_VISIBLE_TIME = 2         # never mask below this many visible time steps (floor on T)


# ================================================================================================
# JEPA — masked spatio-temporal latent prediction
# ================================================================================================
@dataclass
class MaskPlan:
    """Boolean mask over the (B, T, C) token grid; true tokens are hidden from the student."""

    token_mask: torch.Tensor          # (B, T, C) bool


def _ensure_learnable_target(mask: torch.Tensor, valid: torch.Tensor,
                             channel_mask: Optional[torch.Tensor], rnd) -> torch.Tensor:
    """Give otherwise-unsupervised short windows one cross-channel prediction target."""
    B, _, C = mask.shape
    real_channels = (channel_mask if channel_mask is not None else
                     torch.ones(B, C, dtype=torch.bool, device=mask.device))
    empty = ~mask.flatten(1).any(dim=1)
    eligible = empty & valid.any(dim=1) & (real_channels.sum(dim=1) >= 2)
    if not bool(eligible.any()):
        return mask
    scores = rnd(B, C).masked_fill(~real_channels, -1.0)
    channel = scores.argmax(dim=1)
    rows = torch.nonzero(eligible).squeeze(1)
    mask[rows, :, channel[rows]] = valid[rows]
    return mask


def make_mask_plan(
    B: int,
    T: int,
    C: int,
    gyro_channels: Optional[list[int]],
    generator: Optional[torch.Generator] = None,
    time_ratio: float = MASK_RATIO_TIME,
    channel_event_p: float = MASK_RATIO_CHANNEL,
    gyro_bias: float = GYRO_DROP_BIAS,
    device: torch.device = torch.device("cpu"),
    valid_patches: Optional[torch.Tensor] = None,   # (B, T) True = real patch
    channel_mask: Optional[torch.Tensor] = None,    # (B, C) True = real channel
) -> MaskPlan:
    """Structured spatio-temporal mask for the JEPA objective.

    - masking is a RATIO over the variable (T, C) grid, with a floor of MIN_VISIBLE_TIME
      visible steps (robust to the multi-scale patch_seconds axis: T varies per batch);
    - channel stream: whole-channel mask events, biased toward dropping the gyro triad;
    - temporal stream: one contiguous block drawn at a random temporal location.

    VALIDITY-AWARE (pass valid_patches + channel_mask): the temporal block lands only on
    REAL patches (per-sample `usable` count) and channel drops hit only REAL channels. If a short
    window has only one real patch, a fallback masks one real channel while leaving at least one
    other channel visible. When validity masks are omitted, every patch and channel is real.
    """
    rnd = lambda *s: torch.rand(*s, generator=generator, device=device)  # noqa: E731
    mask = torch.zeros(B, T, C, dtype=torch.bool, device=device)
    t_idx = torch.arange(T, device=device).unsqueeze(0)                  # (1, T)

    # per-sample number of maskable time steps: real patches if known, else all T
    usable = (valid_patches.sum(dim=1) if valid_patches is not None
              else torch.full((B,), T, device=device)).long()           # (B,)

    # --- temporal stream: contiguous block within [0, usable) per sample ---
    keep_vis = torch.clamp(usable - 1, min=1)                            # leave >=1 visible
    keep_vis = torch.minimum(keep_vis, torch.full_like(keep_vis, MIN_VISIBLE_TIME))
    max_block = torch.clamp(usable - keep_vis, min=0)                    # (B,) 0 if usable<=1
    block = torch.minimum(torch.clamp(torch.round(time_ratio * usable.float()).long(),
                                      min=1), max_block)                 # (B,) 0 where usable<=1
    max_start = torch.clamp(usable - block, min=0)
    start = (rnd(B) * (max_start + 1).float()).long()
    lo = start.unsqueeze(1)
    hi = (start + block).unsqueeze(1)
    random_block = (t_idx >= lo) & (t_idx < hi) & (block.unsqueeze(1) > 0)
    mask |= random_block.unsqueeze(2)

    # --- channel stream (per sample event): whole channels across all time ---
    event = rnd(B) < channel_event_p
    coin = rnd(B)
    if gyro_channels:
        gyro_real = (channel_mask[:, gyro_channels].all(dim=1) if channel_mask is not None
                     else torch.ones(B, dtype=torch.bool, device=device))
        drop_gyro = event & (coin < gyro_bias) & gyro_real       # only drop REAL gyro
        for c in gyro_channels:
            mask[drop_gyro, :, c] = True
        single = event & ~((coin < gyro_bias) & gyro_real)
    else:
        single = event
    # single drop picks a REAL channel (score absent channels out) so it never wastes
    scores = rnd(B, C)
    if channel_mask is not None:
        scores = scores.masked_fill(~channel_mask, -1.0)
    chan = scores.argmax(dim=1)
    rows = torch.nonzero(single).squeeze(1)
    mask[rows, :, chan[rows]] = True

    # never mask a non-real token (keeps the mask itself clean; loss also intersects)
    if valid_patches is not None:
        mask &= valid_patches.unsqueeze(2)
    if channel_mask is not None:
        mask &= channel_mask.unsqueeze(1)
    valid = (valid_patches if valid_patches is not None else
             torch.ones(B, T, dtype=torch.bool, device=device))
    mask = _ensure_learnable_target(mask, valid, channel_mask, rnd)

    return MaskPlan(token_mask=mask)


def make_per_resolution_mask_plan(
    resolution_ids: torch.Tensor,
    C: int,
    gyro_channels: Optional[list[int]],
    generator: Optional[torch.Generator] = None,
    time_ratio: float = MASK_RATIO_TIME,
    channel_event_p: float = MASK_RATIO_CHANNEL,
    gyro_bias: float = GYRO_DROP_BIAS,
    channel_mask: Optional[torch.Tensor] = None,
    valid_patches: Optional[torch.Tensor] = None,
) -> MaskPlan:
    """Mask ONE contiguous block per temporal resolution, drawn independently per grid.

    The student attends across resolutions. Inferring a hidden fine-grained token from a visible
    coarse summary of the same seconds (or the reverse) is treated as legitimate inference, so
    nothing here couples the two grids' masked spans.

    Three consequences, all deliberate:

    * ``time_ratio`` is realised directly in token counts, up to rounding in small grids.
    * A resolution may be masked in full because the other resolution supplies context.
    * Blocks stay CONTIGUOUS within each grid's own physical-time ordering. Scattered i.i.d.
      masking degenerates for quasi-stationary motion: an adjacent 0.5 s token is nearly a copy,
      so the task collapses to interpolate-from-neighbours.

    The only remaining floor is global: a window must keep at least one visible real token, or
    the student sees nothing but mask tokens and text conditioning.
    """
    device = resolution_ids.device
    B, T = resolution_ids.shape
    rnd = lambda *s: torch.rand(*s, generator=generator, device=device)  # noqa: E731
    valid = resolution_ids.ge(0)
    if valid_patches is not None:
        valid = valid & valid_patches

    temporal = torch.zeros(B, T, dtype=torch.bool, device=device)
    for group in (0, 1):
        group_valid = valid & resolution_ids.eq(group)
        n_group = group_valid.sum(dim=1)                                  # (B,)
        # Rank within this grid's own time order. The collate emits tokens sorted by physical
        # centre time, so a contiguous rank range IS a contiguous span of seconds.
        rank = group_valid.cumsum(dim=1) - 1                              # (B, T)
        block = torch.clamp(torch.round(float(time_ratio) * n_group.float()).long(), min=1)
        block = torch.minimum(block, n_group)                             # 0 where the grid is absent
        max_start = (n_group - block).clamp(min=0)
        start = (rnd(B) * (max_start + 1).float()).long().clamp(max=max_start)
        in_block = group_valid & (rank >= start.unsqueeze(1)) \
            & (rank < (start + block).unsqueeze(1)) & (block > 0).unsqueeze(1)
        temporal |= in_block

    mask = temporal.unsqueeze(2).expand(B, T, C).clone()

    # Whole-channel events are unchanged: they span every token of both grids.
    event = rnd(B) < channel_event_p
    coin = rnd(B)
    if gyro_channels:
        gyro_real = (channel_mask[:, gyro_channels].all(dim=1) if channel_mask is not None
                     else torch.ones(B, dtype=torch.bool, device=device))
        drop_gyro = event & (coin < gyro_bias) & gyro_real
        for c in gyro_channels:
            mask[drop_gyro, :, c] = True
        single = event & ~((coin < gyro_bias) & gyro_real)
    else:
        single = event
    scores = rnd(B, C)
    if channel_mask is not None:
        scores = scores.masked_fill(~channel_mask, -1.0)
    chan = scores.argmax(dim=1)
    rows = torch.nonzero(single).squeeze(1)
    if rows.numel():
        mask[rows, :, chan[rows]] = True

    mask &= valid.unsqueeze(2)
    if channel_mask is not None:
        mask &= channel_mask.unsqueeze(1)

    # Global floor. A fully masked RESOLUTION is fine here -- the other grid is the context -- but
    # a fully masked WINDOW leaves the student only mask tokens. Reveal that window's last real
    # patch rather than dropping its supervision entirely.
    observable = valid.unsqueeze(2)
    if channel_mask is not None:
        observable = observable & channel_mask.unsqueeze(1)
    blind = observable.flatten(1).any(dim=1) & ~(observable & ~mask).flatten(1).any(dim=1)
    if bool(blind.any()):
        rows = torch.nonzero(blind).squeeze(1)
        index = torch.arange(T, device=device).unsqueeze(0).expand(rows.numel(), T)
        last = index.masked_fill(~valid[rows], -1).amax(dim=1)
        mask[rows, last] = False
    return MaskPlan(token_mask=mask)


@dataclass
class VICRegOutput:
    """VICReg total plus its independently auditable components."""

    total: torch.Tensor
    invariance: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    min_std: torch.Tensor


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("covariance matrix must be square")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    *,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> VICRegOutput:
    """Variance-invariance-covariance regularization over aligned positive pairs.

    Unlike NT-Xent this loss has no negative-pair concept: row ``i`` in each view is the positive,
    while other rows only estimate per-feature variance/covariance. Inputs remain unnormalized because
    their per-dimension scale is part of the collapse-prevention objective. Computation is promoted to
    fp32 so AMP cannot underflow covariance statistics.

    This is the published MSE invariance term.
    """
    if z_a.shape != z_b.shape or z_a.ndim != 2:
        raise ValueError(
            f"VICReg expects matching (B,D) tensors, got {tuple(z_a.shape)} and {tuple(z_b.shape)}"
        )
    if z_a.shape[0] < 2:
        raise ValueError("VICReg needs at least two pairs to estimate variance/covariance")
    a, b = z_a.float(), z_b.float()
    inv = F.mse_loss(a, b)

    std_a = torch.sqrt(a.var(dim=0, unbiased=False) + eps)
    std_b = torch.sqrt(b.var(dim=0, unbiased=False) + eps)
    var = 0.5 * (
        F.relu(float(target_std) - std_a).mean()
        + F.relu(float(target_std) - std_b).mean()
    )

    a = a - a.mean(dim=0)
    b = b - b.mean(dim=0)
    denom = float(max(z_a.shape[0] - 1, 1))
    cov_a = (a.T @ a) / denom
    cov_b = (b.T @ b) / denom
    d = max(z_a.shape[1], 1)
    cov = (_off_diagonal(cov_a).pow(2).sum() + _off_diagonal(cov_b).pow(2).sum()) / d
    total = float(invariance_weight) * inv + float(variance_weight) * var \
        + float(covariance_weight) * cov
    return VICRegOutput(
        total=total,
        invariance=inv,
        variance=var,
        covariance=cov,
        min_std=torch.minimum(std_a.min(), std_b.min()),
    )


@torch.no_grad()
def pair_contrast(a: torch.Tensor, b: torch.Tensor, generator=None) -> dict[str, float]:
    """Positive-pair similarity MINUS the random-pair baseline, for any aligned-pair objective.

    A bare positive similarity is uninterpretable. cos(a_i, b_i) = 0.95 means the objective is
    working if a random pair sits at 0.1, and means the representation has collapsed into a
    narrow cone -- so the objective is measuring nothing -- if a random pair ALSO sits at 0.95.
    Both aligned-pair terms here (JEPA and augmentation agreement) have that failure mode; without
    this baseline, a
    collapsed run and a converged run were indistinguishable from the loss value alone.

    `margin` is the quantity to watch: it is the actual discriminative signal available to the
    objective, and it going to ~0 while the loss looks healthy is the collapse signature.
    """
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] < 2:
        return {}
    # Centre first because an un-centred cosine saturates once the representation's common mean
    # grows past its spread, so the margin decays toward 0 even when the objective is healthy. That
    # produced a false alarm on VICReg, whose MSE invariance loss is translation-invariant
    # (augmented pairs reached 1.5% of random-pair distance while this reported margin 0.073).
    centre = torch.cat([a, b]).detach().float().mean(0, keepdim=True)
    an = F.normalize(a.detach().float() - centre, dim=-1)
    bn = F.normalize(b.detach().float() - centre, dim=-1)
    pos = (an * bn).sum(-1).mean()
    # derangement, so no "random" pair is accidentally the true pair
    perm = (torch.arange(len(an), device=an.device) +
            1 + int(torch.randint(len(an) - 1, (1,), generator=generator))) % len(an)
    rnd = (an * bn[perm]).sum(-1).mean()
    values = torch.stack((pos, rnd, pos - rnd)).cpu().tolist()
    return dict(zip(("positive_similarity", "random_similarity", "margin"), values))


def masked_ema_latent_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    token_groups: Optional[torch.Tensor] = None,
    token_durations: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cosine prediction of stop-gradient EMA latents at valid masked token positions.

    NOTE the collapse mode this cannot see on its own: if the EMA teacher's latents all lie in
    a narrow cone, predicting them is trivial and this loss goes to ~0 while the representation
    carries nothing. data2vec normalises its targets precisely to prevent that; we do not, so
    the guard here is diagnostic -- `pair_contrast` on (prediction, target) at masked positions,
    logged as `jepa/margin`. Watch that, not this loss.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"EMA prediction/target shapes differ: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if mask.shape != prediction.shape[:-1]:
        raise ValueError(
            f"EMA mask must have shape {tuple(prediction.shape[:-1])}, got {tuple(mask.shape)}"
        )
    pred = F.normalize(prediction.float(), dim=-1)
    tgt = F.normalize(target.detach().float(), dim=-1)
    per_token = 1.0 - (pred * tgt).sum(dim=-1)
    masked = mask & torch.isfinite(per_token)

    if token_durations is not None and token_durations.shape != prediction.shape[:2]:
        raise ValueError(
            f"EMA token durations must have shape {tuple(prediction.shape[:2])}, "
            f"got {tuple(token_durations.shape)}"
        )

    def _reduce(selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep this reduction entirely on-device. The former bool(selected.any()) synchronized the
        # CUDA stream once per resolution in the middle of every training forward.
        weights = selected.to(per_token.dtype)
        if token_durations is not None:
            weights = weights * token_durations.to(per_token.dtype).unsqueeze(2).clamp(min=0.0)
        values = torch.where(selected, per_token, torch.zeros_like(per_token))
        denominator = weights.sum()
        reduced = (values * weights).sum() / denominator.clamp(min=1e-6)
        return reduced, denominator.gt(0).to(per_token.dtype)

    if token_groups is None:
        reduced, active = _reduce(masked)
        return reduced * active
    if token_groups.shape != prediction.shape[:2]:
        raise ValueError(
            f"EMA token groups must have shape {tuple(prediction.shape[:2])}, "
            f"got {tuple(token_groups.shape)}"
        )
    group_losses = []
    group_active = []
    for group in (0, 1):
        reduced, active = _reduce(masked & token_groups.eq(group).unsqueeze(2))
        group_losses.append(reduced)
        group_active.append(active)
    losses = torch.stack(group_losses)
    active = torch.stack(group_active)
    return (losses * active).sum() / active.sum().clamp(min=1.0)


@dataclass
class PhaseALossOutput:
    """Weighted two-objective loss used by the Phase-A trainer."""

    total: torch.Tensor
    terms: dict[str, torch.Tensor]


def phase_a_loss(
    jepa: torch.Tensor,
    vicreg_loss: torch.Tensor,
    *,
    jepa_weight: float = 1.0,
    vicreg_weight: float = 1.0,
) -> PhaseALossOutput:
    """Combine the fixed-weight JEPA and augmentation-VICReg objectives.

    Keeping exactly two named terms makes gradient telemetry and ablations unambiguous. Frontend
    adaptation regularization is model regularization and is added by the trainer, not represented
    as a third pretraining objective.
    """
    if jepa_weight < 0 or vicreg_weight <= 0:
        raise ValueError("jepa_weight must be nonnegative and vicreg_weight must be positive")
    terms = {
        "jepa": float(jepa_weight) * jepa,
        "vicreg": float(vicreg_weight) * vicreg_loss,
    }
    return PhaseALossOutput(total=terms["jepa"] + terms["vicreg"], terms=terms)
