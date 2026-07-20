"""Unit tests for the elite-3 Phase-1 losses (M2)."""

from __future__ import annotations

import math

import pytest
import torch

from training.tokenizer.losses_repr import (
    MIN_VISIBLE_TIME,
    EliteLossWeights,
    GroundingTargets,
    elite3_loss,
    grounding_loss,
    make_mask_plan,
    make_multiresolution_mask_plan,
    masked_latent_loss,
    supcon_config_conditional,
    transform_cadence_for_timewarp,
)

GYRO = [3, 4, 5]


def gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_multiresolution_mask_hides_every_overlapping_support():
    # Four 0.5-second tokens and two 1-second tokens over the same two seconds.
    starts = torch.tensor([[0.0, 0.0, 0.5, 1.0, 1.0, 1.5]])
    ends = torch.tensor([[0.5, 1.0, 1.0, 1.5, 2.0, 2.0]])
    groups = torch.tensor([[0, 1, 0, 0, 1, 0]])
    plan = make_multiresolution_mask_plan(
        starts, ends, groups, C=1, gyro_channels=None, generator=gen(4),
        channel_event_p=0.0, causal_p=1.0,
        valid_patches=torch.ones_like(groups, dtype=torch.bool),
        channel_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    masked = plan.token_mask[0, :, 0]
    masked_start = starts[0, masked].min()
    masked_end = ends[0, masked].max()
    overlap = (starts[0] < masked_end) & (ends[0] > masked_start)
    assert torch.equal(masked, overlap), "an overlapping scale token leaked the masked interval"


def test_masked_loss_weights_resolutions_equally():
    # Three short tokens have loss 0; one opposite long token has cosine loss 4.
    # Equal-scale reduction is 2.0, whereas a token-weighted reduction would be 1.0.
    target = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]]])
    pred = target.clone()
    pred[:, 3] = torch.tensor([-1.0, 0.0])
    mask = torch.ones(1, 4, 1, dtype=torch.bool)
    groups = torch.tensor([[0, 0, 0, 1]])
    loss = masked_latent_loss(pred, target, mask, token_groups=groups)
    assert torch.allclose(loss, torch.tensor(2.0), atol=1e-6)


# ------------------------------------------------------------------------- mask plan
def test_mask_ratio_and_floor():
    plan = make_mask_plan(B=64, T=10, C=6, gyro_channels=GYRO, generator=gen())
    mask = plan.token_mask
    assert mask.shape == (64, 10, 6)
    ratio = mask.float().mean()
    assert 0.3 < ratio < 0.8, f"overall mask ratio implausible: {ratio}"
    visible_t = (~mask).any(dim=2).sum(dim=1)
    assert (visible_t >= MIN_VISIBLE_TIME).all(), "visible-time floor violated"


def test_mask_adapts_to_small_T():
    """Multi-scale patch_seconds means T can be tiny — the floor must hold at T=3."""
    plan = make_mask_plan(B=32, T=3, C=6, gyro_channels=GYRO, generator=gen(1))
    visible_t = (~plan.token_mask).any(dim=2).sum(dim=1)
    assert (visible_t >= MIN_VISIBLE_TIME).all()


def test_gyro_triad_dropped_jointly():
    """Modality drops take the WHOLE triad; a single-channel drop may hit one gyro axis
    (dead-channel event), so the joint-unity property is: >=2 gyro channels fully
    dropped implies all 3 (only the triad event can drop more than one)."""
    plan = make_mask_plan(B=512, T=8, C=6, gyro_channels=GYRO, generator=gen(2))
    full_channel = plan.token_mask.all(dim=1)            # (B, C) masked at EVERY t
    gyro_count = full_channel[:, GYRO].sum(dim=1)
    assert (gyro_count == 3).sum() > 0, "gyro triad drops never happened at B=512"
    assert not bool((gyro_count == 2).any()), "2-of-3 gyro drop: triad event not joint"


def test_validity_aware_mask_guarantees_supervision():
    """With valid_patches + channel_mask, every window with >=2 real patches must get
    at least one masked REAL token (the A1 zero-supervision fix), and no non-real token
    is ever masked."""
    B, T, C = 256, 6, 6
    g = gen(5)
    # random per-window real-patch counts (1..6) and accel-only vs full-IMU
    usable = torch.randint(1, T + 1, (B,), generator=g)
    valid_patches = torch.arange(T).unsqueeze(0) < usable.unsqueeze(1)      # (B,T) prefix
    channel_mask = torch.ones(B, C, dtype=torch.bool)
    accel_only = torch.rand(B, generator=g) < 0.6
    channel_mask[accel_only, 3:] = False                                    # drop gyro
    plan = make_mask_plan(B, T, C, GYRO, generator=g,
                          valid_patches=valid_patches, channel_mask=channel_mask)
    real = valid_patches.unsqueeze(2) & channel_mask.unsqueeze(1)
    # no non-real token masked
    assert not bool((plan.token_mask & ~real).any())
    # every window with >=2 real patches has >=1 masked real token
    sup = (plan.token_mask & real).flatten(1).sum(1)
    assert (sup[usable >= 2] >= 1).all(), "zero A1 supervision on a >=2-patch window"


def test_causal_variant_masks_the_tail():
    plan = make_mask_plan(B=256, T=10, C=4, gyro_channels=None,
                          generator=gen(3), causal_p=1.0, channel_event_p=0.0)
    mask = plan.token_mask.any(dim=2)                    # (B, T)
    # causal: masked steps must be a suffix
    first_masked = mask.float().argmax(dim=1)
    for b in range(0, 256, 37):
        row = mask[b]
        if row.any():
            assert bool(row[int(first_masked[b]):].all()), "causal mask must be a suffix"


# ------------------------------------------------------------------- masked latent
def test_masked_latent_loss_zero_for_perfect_prediction():
    x = torch.randn(4, 6, 3, 16)
    mask = torch.rand(4, 6, 3) < 0.5
    assert masked_latent_loss(x, x.clone(), mask).abs() < 1e-6


def test_masked_latent_loss_only_counts_masked_tokens():
    target = torch.randn(2, 5, 3, 8)
    pred = target.clone()
    pred[:, 0] = -target[:, 0]                            # corrupt t=0 only
    mask = torch.zeros(2, 5, 3, dtype=torch.bool)
    mask[:, 1:] = True                                    # t=0 NOT masked -> loss ignores it
    assert masked_latent_loss(pred, target, mask).abs() < 1e-6
    mask[:, 0] = True                                     # now it counts
    assert masked_latent_loss(pred, target, mask) > 0.1


def test_masked_latent_loss_empty_mask_is_zero_and_finite():
    loss = masked_latent_loss(torch.randn(2, 4, 3, 8), torch.randn(2, 4, 3, 8),
                              torch.zeros(2, 4, 3, dtype=torch.bool))
    assert loss == 0.0 and torch.isfinite(loss)


# --------------------------------------------------------------------------- supcon
def test_supcon_prefers_clustered_embeddings():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    clustered = torch.stack([
        torch.tensor([1.0, 0.0]), torch.tensor([0.99, 0.1]),
        torch.tensor([0.0, 1.0]), torch.tensor([0.1, 0.99]),
        torch.tensor([-1.0, 0.0]), torch.tensor([-0.99, 0.1]),
    ])
    scrambled = clustered[torch.tensor([0, 2, 4, 1, 3, 5])]
    assert supcon_config_conditional(clustered, labels) < \
        supcon_config_conditional(scrambled, labels)


def test_supcon_no_positive_anchors_is_zero():
    labels = torch.tensor([0, 1, 2, 3])
    loss = supcon_config_conditional(torch.randn(4, 8), labels)
    assert loss == 0.0


def test_supcon_gradient_flows():
    z = torch.randn(8, 16, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    supcon_config_conditional(z, labels).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


# ------------------------------------------------------------------------ grounding
def _targets(b: int = 6) -> GroundingTargets:
    return GroundingTargets(
        cadence_log2hz=torch.ones(b),
        cadence_valid=torch.tensor([True, True, False, True, False, True]),
        eigen_ratios=torch.rand(b, 4, 3),
        eigen_valid=torch.ones(b, dtype=torch.bool),
    )


def test_grounding_masks_invalid_cadence():
    t = _targets()
    pred_cad = torch.ones(6)
    pred_cad[2] = 999.0                                   # invalid slot: must not matter
    pred_cad[4] = -999.0
    loss = grounding_loss(pred_cad, t.eigen_ratios.clone(), t)
    assert loss < 1e-6


def test_grounding_all_invalid_is_zero():
    t = _targets()
    t.cadence_valid[:] = False
    t.eigen_valid[:] = False
    loss = grounding_loss(torch.randn(6), torch.randn(6, 4, 3), t)
    assert loss == 0.0 and torch.isfinite(loss)


def test_grounding_nan_target_entries_are_skipped():
    t = _targets()
    t.eigen_ratios[:, 2, :] = float("nan")               # one empty band
    loss = grounding_loss(t.cadence_log2hz.clone(), t.eigen_ratios.nan_to_num(0.3), t)
    assert torch.isfinite(loss)


def test_timewarp_cadence_transform():
    base = torch.tensor([1.0])                            # 2 Hz
    warped = transform_cadence_for_timewarp(base, alpha=2.0)
    assert torch.allclose(warped, torch.tensor([2.0]))    # 4 Hz -> log2 = 2
    assert torch.allclose(
        transform_cadence_for_timewarp(base, 1.0), base
    )


# ------------------------------------------------------------------------- combined
def test_elite3_combined_finite_and_weighted():
    B, T, C, D = 6, 8, 6, 16
    t = _targets()
    out = elite3_loss(
        a1_pred=torch.randn(B, T, C, D),
        a1_target=torch.randn(B, T, C, D),
        a1_mask=torch.rand(B, T, C) < 0.5,
        a2_embeddings=torch.randn(B, D),
        a2_labels=torch.tensor([0, 0, 1, 1, 2, 2]),
        a3_cadence_pred=torch.randn(B),
        a3_eigen_pred=torch.rand(B, 4, 3),
        a3_targets=t,
        weights=EliteLossWeights(a1_masked=1.0, a2_supcon=1.0, a3_grounding=0.1),
    )
    assert torch.isfinite(out.total)
    assert set(out.parts) == {"a1_masked", "a2_supcon", "a3_grounding"}
    expected = out.parts["a1_masked"] + out.parts["a2_supcon"] + 0.1 * out.parts["a3_grounding"]
    assert math.isclose(float(out.total), expected, rel_tol=1e-5)
