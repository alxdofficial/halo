"""Pipeline A Phase-1 losses — the ELITE 3 (M2; EVIDENCE_ENGINE.md §5.2.1–5.2.3).

Exactly three objectives, deliberately not a menu:

  A1  masked spatio-temporal latent prediction — mask whole channels (biased toward
      dropping the gyro triad = the real deployment shift) AND temporal blocks
      (random-block for representation; causal/future = the world-model variant whose
      prediction error later feeds abstention). Prediction in latent space.
  A2  window-level augmentation agreement. DEFAULT is self-supervised VICReg over two
      independent augmentations of the same window (no negatives and no labels). SimCLR/NT-Xent
      remains a historical control. The
      original label-SupCon (`supcon_config_conditional`: same-activity windows across
      DIFFERENT configs are positives, config as input not nuisance) is kept selectable
      for the do-no-harm ablation (a2_mode='supcon'). No CLIP sensor<->text term (label
      transfer is text->text in the evidence head).
  A3  physical-primitive grounding — regress the M1 primitives (cadence log2-Hz,
      eigen-ratios) from the representation. Targets are computed on the AUGMENTED view
      (or analytically transformed — never predict a clean-signal primitive from an
      augmented input) and validity-masked. Small weight: a grounding rail, not a driver.

Deferred to ablations (convergence risk lives there): equivariance operator, sparse
feature-space reconstruction, analysis-consistency as a separate loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------
# Defaults (M2 starting points — swept later, all in one place)
# ----------------------------------------------------------------------------------------------
MASK_RATIO_TIME = 0.5        # fraction of time steps masked (A1 spec: start ~50%, ablate up)
MASK_RATIO_CHANNEL = 0.25    # fraction of batches that get a whole-channel mask event
GYRO_DROP_BIAS = 0.7         # within channel-mask events, P(drop the whole gyro triad)
CAUSAL_FRACTION = 0.3        # fraction of batches using the causal/future (world-model) mask
MIN_VISIBLE_TIME = 2         # never mask below this many visible time steps (floor on T)
SUPCON_TEMPERATURE = 0.1
SIMCLR_TEMPERATURE = 0.1     # NT-Xent temperature for the self-supervised A2 (SimCLR default)
A3_WEIGHT = 0.1              # grounding rail, not a driver
# Per-primitive A3 weights. Eigen-ratios are valid on ~100% of windows in every corpus and
# carry CROSS-CHANNEL motion geometry that the A1 target (per-channel band energies +
# amplitude + DC) does not contain, so they are the substantive half at weight 1.0. Cadence
# is largely recoverable from the band energies and is valid on only 7-15% of windows in the
# free-living majority (nhanes 0.07, extrasensory 0.09, capture24 0.10 — measured), so it
# keeps the 0.5 it effectively had and is independently zeroable.
# A1 magnitude term. The band cosine lands ~O(0.1-2); Huber on the standardized DC and the
# log1p amplitude lands ~O(0.1-1), so 0.5 keeps shape as the dominant signal while making
# absolute energy learnable at all. Needs a sweep — it is the one number here with no
# measurement behind it yet.
A1_MAGNITUDE_WEIGHT = 0.5
A3_CADENCE_WEIGHT = 0.5
A3_EIGEN_WEIGHT = 1.0
HUBER_DELTA = 1.0


# ================================================================================================
# A1 — masked spatio-temporal latent prediction
# ================================================================================================
@dataclass
class MaskPlan:
    """Boolean masks over the (B, T, C) token grid. True = MASKED (hidden from encoder,
    predicted by the head). kind records the temporal variant for logging/ablation."""

    token_mask: torch.Tensor          # (B, T, C) bool
    kind: str = "random_block"        # 'random_block' | 'causal'


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
    """Structured spatio-temporal mask (A1 spec §5.2.1).

    - masking is a RATIO over the variable (T, C) grid, with a floor of MIN_VISIBLE_TIME
      visible steps (robust to the multi-scale patch_seconds axis: T varies per batch);
    - channel stream: whole-channel mask events, biased toward dropping the gyro triad;
    - temporal stream: contiguous random block, or the causal variant (mask the tail =
      predict the future = the world-model objective).

    VALIDITY-AWARE (pass valid_patches + channel_mask): the temporal block lands only on
    REAL patches (per-sample `usable` count) and channel drops hit only REAL channels, so
    A1 supervision (masked ∩ real) is non-empty for every window with >= 2 real patches.
    Without them (defaults None) the legacy blind grid mask is used — kept for unit tests.
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

    return MaskPlan(token_mask=mask, kind="mixed")


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

    # If every real token overlaps (possible for a very short recording), there is no
    # honest visible/masked split. Skip temporal A1 for that sample rather than exposing
    # the same raw interval through one resolution to predict the other.
    all_masked = valid.any(dim=1) & ((temporal & valid).sum(dim=1) == valid.sum(dim=1))
    temporal[all_masked] = False

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
    return MaskPlan(token_mask=mask, kind="mixed_multiresolution")


def _a1_per_token(
    predicted: torch.Tensor,
    target: torch.Tensor,
    feature_valid: Optional[torch.Tensor],
    n_bands: Optional[int],
    magnitude_weight: float,
    delta: float = HUBER_DELTA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token A1 loss (B,T,C) and the (B,T,C) mask of tokens that carry any signal.

    SHAPE-vs-MAGNITUDE SPLIT. The A1 target is not a learned latent — it is
    ``[band energies (K) | amplitude | DC]``, calibrated physical quantities. L2-normalising
    all of it (JEPA/BYOL hygiene, correct for learned latents that can drift in scale)
    discarded magnitude, which for inertial sensing IS much of the signal: sitting and
    walking share a broadly similar spectral SHAPE and differ by an order of magnitude in
    energy. It also largely cancelled the `amplitude` feature — added precisely to
    "preserve absolute magnitude" — by dividing it by the norm of the vector it sits in,
    and the DC feature that separates stand/sit/lie.

    That mattered more than a normalisation detail: A3's eigen-ratios are gain-invariant by
    construction and A2's VICReg is invariant to the magnitude augmentations, so with A1
    scale-blind too, NOTHING in the objective stack supervised absolute signal energy.

    So: cosine on the band sub-vector, where scale-invariance genuinely is the spectral
    shape we want predicted; Huber on the trailing amplitude/DC scalars, where the absolute
    value is the content. ``n_bands=None`` restores the all-cosine behaviour.
    """
    target = target.detach()
    if feature_valid is not None:
        fv = feature_valid.to(target.dtype)
        target = target * fv
        predicted = predicted * fv

    split = n_bands if (n_bands is not None and 0 < n_bands < target.shape[-1]) else None
    band_p, band_t = (predicted[..., :split], target[..., :split]) if split else (predicted, target)

    # A token whose every OBSERVABLE band was masked out has an all-zero band target:
    # F.normalize returns zeros, the cosine is 0 and the term is a constant 2.0 with no
    # gradient. Zero the shape term there rather than let it inflate the loss.
    has_signal = band_t.abs().sum(dim=-1) > 0                           # (B, T, C)
    cos_term = 2.0 - 2.0 * (F.normalize(band_p, dim=-1) * F.normalize(band_t, dim=-1)).sum(dim=-1)
    per_token = torch.where(has_signal, cos_term, torch.zeros_like(cos_term))

    keep = has_signal
    if split:
        # Magnitude features are always observable (feature_valid is 1 there), so a token
        # with no usable band still carries real magnitude supervision — keep it.
        mag = F.huber_loss(predicted[..., split:], target[..., split:],
                           delta=delta, reduction="none").mean(dim=-1)
        per_token = per_token + float(magnitude_weight) * mag
        keep = torch.ones_like(has_signal)
    return per_token, keep


def masked_latent_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
    feature_valid: Optional[torch.Tensor] = None,
    token_groups: Optional[torch.Tensor] = None,
    token_durations: Optional[torch.Tensor] = None,   # (B,T) represented seconds per patch (F1)
    n_bands: Optional[int] = None,
    magnitude_weight: float = A1_MAGNITUDE_WEIGHT,
) -> torch.Tensor:
    """A1: regression in LATENT space on masked tokens only.

    predicted: (B, T, C, D) head outputs; target: (B, T, C, D) the (stop-grad) latent
    targets for the same grid; token_mask: (B, T, C) True where masked.

    ``n_bands`` splits the feature axis into the band sub-vector (compared by cosine, so
    scale-invariant) and the trailing amplitude/DC scalars (compared by Huber, so absolute
    magnitude is supervised). See ``_a1_per_token``.

    feature_valid: optional broadcastable-to-(B,T,C,D) mask over the FEATURE dims. Bands
    above a sample's Nyquist are zero in the target and trivial to predict; zeroing those
    dims in BOTH pred and target excludes them so A1 only supervises OBSERVABLE signal.
    """
    per_token, has_signal = _a1_per_token(
        predicted, target, feature_valid, n_bands, magnitude_weight)
    masked = token_mask & torch.isfinite(per_token) & has_signal

    def _reduce(sel_mask):
        """Mean over selected tokens, optionally weighted by represented duration (F1): a partial
        tail patch contributes proportionally, not as a full-length patch. Uniform when absent."""
        if not bool(sel_mask.any()):
            return None
        if token_durations is None:
            return per_token[sel_mask].mean()
        w = token_durations.to(per_token.dtype).unsqueeze(2).clamp(min=0.0) * sel_mask.to(per_token.dtype)
        pt = torch.where(sel_mask, per_token, torch.zeros_like(per_token))   # zero (finite) elsewhere
        return (pt * w).sum() / w.sum().clamp(min=1e-6)

    if token_groups is None:
        out = _reduce(masked)
        return out if out is not None else predicted.new_zeros(())

    # Reduce within each resolution before reducing across resolutions. Otherwise a
    # 12-token short grid contributes three times the weight of a 4-token long grid.
    group_losses = []
    for group in (0, 1):
        out = _reduce(masked & token_groups.eq(group).unsqueeze(2))
        if out is not None:
            group_losses.append(out)
    return (torch.stack(group_losses).mean() if group_losses
            else predicted.new_zeros(()))


def masked_latent_per_window(predicted, target, token_mask, feature_valid=None,
                             n_bands=None,
                             magnitude_weight: float = A1_MAGNITUDE_WEIGHT) -> torch.Tensor:
    """Per-WINDOW A1 loss (B,) for telemetry — same cosine loss as `masked_latent_loss` but reduced
    per window instead of globally, so it can be grouped by data source. Windows with no masked token
    return NaN (caller filters). Pure diagnostic; not part of the training gradient."""
    per_token, has_signal = _a1_per_token(
        predicted, target, feature_valid, n_bands, magnitude_weight)
    m = (token_mask & torch.isfinite(per_token) & has_signal).to(per_token.dtype)
    denom = m.sum(dim=(1, 2))
    win = (per_token.nan_to_num(0.0) * m).sum(dim=(1, 2)) / denom.clamp(min=1)
    return torch.where(denom > 0, win, torch.full_like(win, float("nan")))


# ================================================================================================
# A2 — config-conditional supervised contrastive (window level)
# ================================================================================================
def supcon_config_conditional(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = SUPCON_TEMPERATURE,
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al.) over L2-normalized window embeddings.

    labels: (B,) integer activity ids. The CONFIG-CONDITIONAL part is architectural, not
    in this formula: same-activity windows from different placements/datasets are
    positives here precisely BECAUSE the encoder receives the config as input text —
    it can factor config out without being blind to it. Anchors with no positive in
    the batch are skipped (the batch sampler should make them rare).
    """
    z = F.normalize(embeddings, dim=1)
    B = z.shape[0]
    sim = (z @ z.t()) / temperature                                     # (B, B)
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(self_mask, float("-inf"))

    positives = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    has_pos = positives.any(dim=1)
    if not bool(has_pos.any()):
        return z.new_zeros(())

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # masked_fill, NOT multiply: the -inf self-diagonal times a False positive is 0*-inf=NaN
    pos_log_prob = log_prob.masked_fill(~positives, 0.0).sum(dim=1) \
        / positives.sum(dim=1).clamp(min=1)
    return -(pos_log_prob[has_pos]).mean()


def nt_xent(z_a: torch.Tensor, z_b: torch.Tensor,
            temperature: float = SIMCLR_TEMPERATURE) -> torch.Tensor:
    """Self-supervised SimCLR NT-Xent over two augmented views (Chen et al. 2020).

    z_a, z_b: (B, d_proj) projections of the SAME B windows under two INDEPENDENT
    augmentations. Both are L2-normalized. Stacking to (2B, d), the positive for anchor
    z_a[i] is z_b[i] (row i and row i+B) and symmetrically; the negatives are every other
    sample across BOTH views (2B-2 of them). The self-similarity diagonal is masked to
    -inf so it never enters the denominator. Returns the symmetric mean NT-Xent (the mean
    over all 2B anchors — each view serves as anchor once). No labels: this is the
    self-supervised replacement for the label-SupCon A2 term.
    """
    B = z_a.shape[0]
    z = torch.cat([F.normalize(z_a, dim=1), F.normalize(z_b, dim=1)], dim=0)   # (2B, d)
    sim = (z @ z.t()) / temperature                                            # (2B, 2B)
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    # positive of row i is its view-partner: i<B -> i+B ; i>=B -> i-B
    targets = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return F.cross_entropy(sim, targets)


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
    Every aligned-pair term here (A1's masked-latent prediction, A2's augmented views, A4's
    cross-placement pairs) has that failure mode and none of them logged the baseline, so a
    collapsed run and a converged run were indistinguishable from the loss value alone.

    `margin` is the quantity to watch: it is the actual discriminative signal available to the
    objective, and it going to ~0 while the loss looks healthy is the collapse signature.
    """
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] < 2:
        return {}
    an = F.normalize(a.detach().float(), dim=-1)
    bn = F.normalize(b.detach().float(), dim=-1)
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
) -> torch.Tensor:
    """Cosine prediction of stop-gradient EMA latents at valid masked token positions.

    NOTE the collapse mode this cannot see on its own: if the EMA teacher's latents all lie in
    a narrow cone, predicting them is trivial and this loss goes to ~0 while the representation
    carries nothing. data2vec normalises its targets precisely to prevent that; we do not, so
    the guard here is diagnostic -- `pair_contrast` on (prediction, target) at masked positions,
    logged as `a1/margin`. Watch that, not this loss.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"EMA prediction/target shapes differ: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if mask.shape != prediction.shape[:-1]:
        raise ValueError(
            f"EMA mask must have shape {tuple(prediction.shape[:-1])}, got {tuple(mask.shape)}"
        )
    if not bool(mask.any()):
        return prediction.new_zeros(())
    pred = F.normalize(prediction[mask].float(), dim=-1)
    tgt = F.normalize(target.detach()[mask].float(), dim=-1)
    return (1.0 - (pred * tgt).sum(dim=-1)).mean()


# ================================================================================================
# A3 — physical-primitive grounding (aug-aware targets, validity-masked)
# ================================================================================================
@dataclass
class GroundingTargets:
    """Targets computed on the FINAL augmented view.

    The collate calls `cadence(acc, rate)` / `eigen_ratios(acc, rate)` on the augmented
    signal at the augmented rate, so a time-warp's effect on physical cadence is MEASURED
    directly. (An earlier design cached a clean cadence and transformed it analytically by
    log2(alpha); that path was removed 2026-07-26 — it was never wired into training.)
    Rotation, gain and rate resampling leave every target here invariant, which is the
    primitive-selection criterion.
    """

    cadence_log2hz: torch.Tensor      # (B,)
    cadence_valid: torch.Tensor       # (B,) bool
    eigen_ratios: torch.Tensor        # (B, n_bands, 3)
    eigen_valid: torch.Tensor         # (B,) bool


def grounding_loss(
    cadence_pred: torch.Tensor,
    eigen_pred: torch.Tensor,
    targets: GroundingTargets,
    delta: float = HUBER_DELTA,
    cadence_weight: float = A3_CADENCE_WEIGHT,
    eigen_weight: float = A3_EIGEN_WEIGHT,
) -> torch.Tensor:
    """A3: Huber regression per primitive, masked by per-primitive validity.

    cadence_pred: (B,); eigen_pred: (B, n_bands, 3) — eigen predictions are softmax'd
    onto the simplex upstream or raw (targets live on the simplex; Huber on the three
    descriptors is the M2 baseline, a simplex-aware head is an ablation).
    Loss is 0 (not NaN) when nothing is valid — aggregation-safe.

    FIXED per-primitive weights, NOT a mean over whichever primitives happen to be valid.
    The previous `total / max(terms, 1)` coupled two unrelated quantities: eigen's effective
    weight was 1.0 when cadence was invalid and 0.5 when it was valid, so identical eigen
    supervision counted twice as much on capture24 (cadence valid ~10% of windows) as on
    hhar (~62%). Eigen is valid on essentially every window, so it keeps weight 1.0; cadence
    keeps the 0.5 it effectively had when present, and can now be zeroed independently on
    corpora where it is mostly invalid.
    """
    total = cadence_pred.new_zeros(())
    if cadence_weight and bool(targets.cadence_valid.any()):
        v = targets.cadence_valid
        total = total + float(cadence_weight) * F.huber_loss(
            cadence_pred[v], targets.cadence_log2hz[v], delta=delta)
    if eigen_weight and bool(targets.eigen_valid.any()):
        v = targets.eigen_valid
        pred, tgt = eigen_pred[v], targets.eigen_ratios[v]
        finite = torch.isfinite(tgt)
        if bool(finite.any()):
            total = total + float(eigen_weight) * F.huber_loss(
                pred[finite], tgt[finite], delta=delta)
    return total


# ================================================================================================
# The combined Phase-1 objective
# ================================================================================================
@dataclass
class EliteLossWeights:
    a1_physical: float = 1.0
    a2_supcon: float = 1.0
    a3_grounding: float = A3_WEIGHT   # rail, not driver


@dataclass
class EliteLossOutput:
    total: torch.Tensor
    parts: dict[str, float] = field(default_factory=dict)
    terms: dict[str, torch.Tensor] = field(default_factory=dict)


def elite3_loss(
    a1_pred: torch.Tensor,
    a1_target: torch.Tensor,
    a1_mask: torch.Tensor,
    a2_embeddings: torch.Tensor,
    a2_labels: torch.Tensor,
    a3_cadence_pred: torch.Tensor,
    a3_eigen_pred: torch.Tensor,
    a3_targets: GroundingTargets,
    weights: EliteLossWeights = EliteLossWeights(),
    a1_feature_valid: Optional[torch.Tensor] = None,
    a1_token_groups: Optional[torch.Tensor] = None,
    a1_token_durations: Optional[torch.Tensor] = None,
    a1_n_bands: Optional[int] = None,          # split point: bands (cosine) | amplitude+DC (Huber)
    a1_magnitude_weight: float = A1_MAGNITUDE_WEIGHT,
    a2_loss: Optional[torch.Tensor] = None,
    a2_key: str = "a2_supcon",
) -> EliteLossOutput:
    """Weighted sum of the elite 3. Exactly these; additions go through an ablation.

    A2 is pluggable: pass a precomputed ``a2_loss`` (e.g. the self-supervised
    ``nt_xent`` over two views, labelled by ``a2_key='a2_simclr'``) to use it directly and
    SKIP the label-based supcon; leave it None to compute the label-SupCon here (the
    ablation path). Either way it is scaled by ``weights.a2_supcon``.
    """
    l1 = masked_latent_loss(a1_pred, a1_target, a1_mask, feature_valid=a1_feature_valid,
                            n_bands=a1_n_bands, magnitude_weight=a1_magnitude_weight,
                            token_groups=a1_token_groups, token_durations=a1_token_durations)
    l2 = a2_loss if a2_loss is not None else supcon_config_conditional(a2_embeddings, a2_labels)
    l3 = grounding_loss(a3_cadence_pred, a3_eigen_pred, a3_targets)
    total = weights.a1_physical * l1 + weights.a2_supcon * l2 + weights.a3_grounding * l3
    return EliteLossOutput(
        total=total,
        parts={"a1_physical": float(l1.detach()), a2_key: float(l2.detach()),
               "a3_grounding": float(l3.detach())},
        terms={"a1_physical": weights.a1_physical * l1,
               a2_key: weights.a2_supcon * l2,
               "a3_grounding": weights.a3_grounding * l3},
    )
