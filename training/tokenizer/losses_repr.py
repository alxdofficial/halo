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
CAUSAL_FRACTION = 0.3        # fraction of batches using the causal/future (world-model) mask
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
    causal_p: float = CAUSAL_FRACTION,
    device: torch.device = torch.device("cpu"),
    valid_patches: Optional[torch.Tensor] = None,   # (B, T) True = real patch
    channel_mask: Optional[torch.Tensor] = None,    # (B, C) True = real channel
) -> MaskPlan:
    """Structured spatio-temporal mask for the JEPA objective.

    - masking is a RATIO over the variable (T, C) grid, with a floor of MIN_VISIBLE_TIME
      visible steps (robust to the multi-scale patch_seconds axis: T varies per batch);
    - channel stream: whole-channel mask events, biased toward dropping the gyro triad;
    - temporal stream: contiguous random block, or the causal variant (mask the tail =
      predict the future = the world-model objective).

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
    causal = rnd(B) < causal_p
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
    causal_block = (t_idx >= (usable - block).unsqueeze(1)) & (t_idx < usable.unsqueeze(1)) \
        & (block.unsqueeze(1) > 0)
    time_mask = torch.where(causal.unsqueeze(1), causal_block, random_block)  # (B, T)
    mask |= time_mask.unsqueeze(2)

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


def make_multiresolution_mask_plan(
    patch_starts: torch.Tensor,
    patch_ends: torch.Tensor,
    resolution_ids: torch.Tensor,
    C: int,
    gyro_channels: Optional[list[int]],
    generator: Optional[torch.Generator] = None,
    time_ratio: float = MASK_RATIO_TIME,
    channel_event_p: float = MASK_RATIO_CHANNEL,
    gyro_bias: float = GYRO_DROP_BIAS,
    causal_p: float = CAUSAL_FRACTION,
    channel_mask: Optional[torch.Tensor] = None,
    valid_patches: Optional[torch.Tensor] = None,
) -> MaskPlan:
    """Mask one physical interval and every scale token whose support overlaps it.

    This is the non-leaking counterpart of ``make_mask_plan`` for simultaneous temporal
    resolutions. A masked short patch cannot be reconstructed by reading an overlapping long
    patch, and channel events apply to every resolution of that channel.

    INTERVAL LENGTH vs MASKED FRACTION. Masking is by OVERLAP, so a token of support ``p`` is
    caught whenever the interval comes within ``p`` of it: the realised fraction is
    ``(L + p) / W``, not ``L / W``. Setting ``L = time_ratio * W`` therefore does NOT mask
    ``time_ratio`` of the tokens. Measured at the live patch sizes (0.5 s / 1.4 s in a 6 s
    window) it masked 0.558 of short tokens and 0.674 of long ones against a nominal 0.5.

    WHAT IS AND IS NOT GUARANTEED. A masked SHORT token cannot be read off a long one: any long
    token containing it also overlaps the interval, so it is masked too. The REVERSE does not
    hold and never did -- a masked LONG token can have a visible short token inside its support,
    because the short tokens near the far end of the long token's span may miss the interval.
    Measured at (0.5 s, 1.4 s) in a 6 s window: 25.7% of masked long tokens have a visible short
    token inside them. That is a pre-existing property of overlap masking, not a regression.

    The mask intentionally uses the uncompensated interval. A retired compensation arm reduced
    the number of masked long tokens but increased cross-resolution leakage from 25.7% to 40.1%.
    """
    device = patch_starts.device
    B, T = patch_starts.shape
    rnd = lambda *s: torch.rand(*s, generator=generator, device=device)  # noqa: E731
    valid = resolution_ids.ge(0)
    if valid_patches is not None:
        valid &= valid_patches
    observed_end = patch_ends.masked_fill(~valid, 0.0).amax(dim=1)
    interval_len = observed_end * float(time_ratio)
    causal = rnd(B) < causal_p
    random_start = rnd(B) * (observed_end - interval_len).clamp(min=0.0)
    interval_start = torch.where(causal, observed_end - interval_len, random_start)
    interval_end = interval_start + interval_len
    temporal = valid & (patch_starts < interval_end.unsqueeze(1)) \
        & (patch_ends > interval_start.unsqueeze(1))

    # The masked encoder isolates temporal attention within each resolution. Therefore every
    # A resolution supervised by temporal JEPA needs at least one visible token in that resolution;
    # a one-token long grid cannot infer its signal from the visible short grid. Remove only the
    # unlearnable resolution's temporal masks, preserving valid supervision in the other scale.
    for group in (0, 1):
        group_valid = valid & resolution_ids.eq(group)
        all_group_masked = group_valid.any(dim=1) \
            & ((temporal & group_valid).sum(dim=1) == group_valid.sum(dim=1))
        temporal &= ~(all_group_masked.unsqueeze(1) & group_valid)

    mask = temporal.unsqueeze(2).expand(B, T, C).clone()

    # Whole-channel events mirror the single-resolution objective, but naturally span
    # every token from both grids.
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
    mask = _ensure_learnable_target(mask, valid, channel_mask, rnd)
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
    return {"positive_similarity": float(pos),
            "random_similarity": float(rnd),
            "margin": float(pos - rnd)}


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

    def _reduce(selected: torch.Tensor) -> Optional[torch.Tensor]:
        if not bool(selected.any()):
            return None
        if token_durations is None:
            return per_token[selected].mean()
        if token_durations.shape != prediction.shape[:2]:
            raise ValueError(
                f"EMA token durations must have shape {tuple(prediction.shape[:2])}, "
                f"got {tuple(token_durations.shape)}"
            )
        weights = token_durations.to(per_token.dtype).unsqueeze(2).clamp(min=0.0)
        weights = weights * selected.to(per_token.dtype)
        values = torch.where(selected, per_token, torch.zeros_like(per_token))
        return (values * weights).sum() / weights.sum().clamp(min=1e-6)

    if token_groups is None:
        reduced = _reduce(masked)
        return reduced if reduced is not None else prediction.new_zeros(())
    if token_groups.shape != prediction.shape[:2]:
        raise ValueError(
            f"EMA token groups must have shape {tuple(prediction.shape[:2])}, "
            f"got {tuple(token_groups.shape)}"
        )
    group_losses = []
    for group in (0, 1):
        reduced = _reduce(masked & token_groups.eq(group).unsqueeze(2))
        if reduced is not None:
            group_losses.append(reduced)
    return (torch.stack(group_losses).mean() if group_losses
            else prediction.new_zeros(()))


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
