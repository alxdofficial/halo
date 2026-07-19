"""Pipeline A Phase-1 losses — the ELITE 3 (M2; EVIDENCE_ENGINE.md §5.2.1–5.2.3).

Exactly three objectives, deliberately not a menu:

  A1  masked spatio-temporal latent prediction — mask whole channels (biased toward
      dropping the gyro triad = the real deployment shift) AND temporal blocks
      (random-block for representation; causal/future = the world-model variant whose
      prediction error later feeds abstention). Prediction in latent space.
  A2  config-conditional supervised contrastive — window-level sensor<->sensor SupCon;
      same-activity windows across DIFFERENT configs are positives (the config is an
      input, not a nuisance to blind-invariate away). No CLIP sensor<->text term (label
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
A3_WEIGHT = 0.1              # grounding rail, not a driver
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


def masked_latent_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
    feature_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """A1: regression in LATENT space on masked tokens only.

    predicted: (B, T, C, D) head outputs; target: (B, T, C, D) the (stop-grad) latent
    targets for the same grid; token_mask: (B, T, C) True where masked. Targets are
    L2-normalized per token (direction, not magnitude — standard JEPA/BYOL hygiene).

    feature_valid: optional broadcastable-to-(B,T,C,D) mask over the FEATURE dims. Bands
    above a sample's Nyquist are zero in the target and trivial to predict; zeroing those
    dims in BOTH pred and target before the cosine excludes them from the objective so A1
    only supervises OBSERVABLE signal (refinement to the signal-only-target fix).
    """
    target = target.detach()
    if feature_valid is not None:
        fv = feature_valid.to(target.dtype)
        target = target * fv
        predicted = predicted * fv
    target = F.normalize(target, dim=-1)
    predicted = F.normalize(predicted, dim=-1)
    per_token = 2.0 - 2.0 * (predicted * target).sum(dim=-1)            # cosine loss
    masked = token_mask & torch.isfinite(per_token)
    if not bool(masked.any()):
        return predicted.new_zeros(())
    return per_token[masked].mean()


def masked_latent_per_window(predicted, target, token_mask, feature_valid=None) -> torch.Tensor:
    """Per-WINDOW A1 loss (B,) for telemetry — same cosine loss as `masked_latent_loss` but reduced
    per window instead of globally, so it can be grouped by data source. Windows with no masked token
    return NaN (caller filters). Pure diagnostic; not part of the training gradient."""
    target = target.detach()
    if feature_valid is not None:
        fv = feature_valid.to(target.dtype)
        target = target * fv
        predicted = predicted * fv
    target = F.normalize(target, dim=-1)
    predicted = F.normalize(predicted, dim=-1)
    per_token = 2.0 - 2.0 * (predicted * target).sum(dim=-1)            # (B, T, C)
    m = (token_mask & torch.isfinite(per_token)).to(per_token.dtype)
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


# ================================================================================================
# A3 — physical-primitive grounding (aug-aware targets, validity-masked)
# ================================================================================================
@dataclass
class GroundingTargets:
    """Targets computed on the AUGMENTED view (or analytically transformed).

    cadence_log2hz: (B,) — if built from a cached CLEAN cadence under a time-warp with
    factor alpha, transform it: log2(alpha * hz) = log2(hz) + log2(alpha). Rotation /
    gain / rate leave every target here invariant (nothing to do) — that invariance is
    the primitive-selection criterion.
    """

    cadence_log2hz: torch.Tensor      # (B,)
    cadence_valid: torch.Tensor       # (B,) bool
    eigen_ratios: torch.Tensor        # (B, n_bands, 3)
    eigen_valid: torch.Tensor         # (B,) bool


def transform_cadence_for_timewarp(cadence_log2hz: torch.Tensor, alpha: float) -> torch.Tensor:
    """Analytic target transform under a uniform time-warp y(t)=x(alpha*t)."""
    return cadence_log2hz + torch.log2(torch.tensor(float(alpha)))


def grounding_loss(
    cadence_pred: torch.Tensor,
    eigen_pred: torch.Tensor,
    targets: GroundingTargets,
    delta: float = HUBER_DELTA,
) -> torch.Tensor:
    """A3: Huber regression per primitive, masked by per-primitive validity.

    cadence_pred: (B,); eigen_pred: (B, n_bands, 3) — eigen predictions are softmax'd
    onto the simplex upstream or raw (targets live on the simplex; Huber on the three
    descriptors is the M2 baseline, a simplex-aware head is an ablation).
    Loss is 0 (not NaN) when nothing is valid — aggregation-safe.
    """
    total, terms = cadence_pred.new_zeros(()), 0
    if bool(targets.cadence_valid.any()):
        v = targets.cadence_valid
        total = total + F.huber_loss(cadence_pred[v], targets.cadence_log2hz[v], delta=delta)
        terms += 1
    if bool(targets.eigen_valid.any()):
        v = targets.eigen_valid
        pred, tgt = eigen_pred[v], targets.eigen_ratios[v]
        finite = torch.isfinite(tgt)
        if bool(finite.any()):
            total = total + F.huber_loss(pred[finite], tgt[finite], delta=delta)
            terms += 1
    return total / max(terms, 1)


# ================================================================================================
# The combined Phase-1 objective
# ================================================================================================
@dataclass
class EliteLossWeights:
    a1_masked: float = 1.0
    a2_supcon: float = 1.0
    a3_grounding: float = A3_WEIGHT   # rail, not driver


@dataclass
class EliteLossOutput:
    total: torch.Tensor
    parts: dict[str, float] = field(default_factory=dict)


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
) -> EliteLossOutput:
    """Weighted sum of the elite 3. Exactly these; additions go through an ablation."""
    l1 = masked_latent_loss(a1_pred, a1_target, a1_mask, feature_valid=a1_feature_valid)
    l2 = supcon_config_conditional(a2_embeddings, a2_labels)
    l3 = grounding_loss(a3_cadence_pred, a3_eigen_pred, a3_targets)
    total = weights.a1_masked * l1 + weights.a2_supcon * l2 + weights.a3_grounding * l3
    return EliteLossOutput(
        total=total,
        parts={"a1_masked": float(l1.detach()), "a2_supcon": float(l2.detach()),
               "a3_grounding": float(l3.detach())},
    )
