"""Unit tests for the consolidated JEPA + relation Phase-A losses."""

from __future__ import annotations

import pytest
import torch

from training.tokenizer.losses_repr import (
    MIN_VISIBLE_TIME,
    make_mask_plan,
    make_multiresolution_mask_plan,
    masked_ema_latent_loss,
    phase_a_loss,
    relation_loss,
    vicreg,
)
from training.tokenizer.pretrain import (
    update_ema_encoder,
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


def test_multiresolution_mask_drops_unlearnable_fully_masked_resolution():
    """Temporal attention is isolated by resolution, so a one-token scale cannot be its own context."""
    starts = torch.tensor([[0.0, 0.4, 0.8, 0.0]])
    ends = torch.tensor([[0.4, 0.8, 1.0, 1.0]])
    groups = torch.tensor([[0, 0, 0, 1]])
    plan = make_multiresolution_mask_plan(
        starts, ends, groups, C=1, gyro_channels=None, generator=gen(4),
        channel_event_p=0.0, causal_p=1.0, time_ratio=0.5,
        valid_patches=torch.ones_like(groups, dtype=torch.bool),
        channel_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    masked = plan.token_mask[0, :, 0]
    assert bool(masked[:3].any()) and bool((~masked[:3]).any())
    assert not bool(masked[3]), "one-token isolated resolution received an impossible JEPA target"


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
    at least one masked REAL token (the JEPA zero-supervision fix), and no non-real token
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
    assert (sup[usable >= 2] >= 1).all(), "zero JEPA supervision on a >=2-patch window"


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


# --------------------------------------------------------------------------- VICReg / EMA targets
def test_vicreg_prefers_aligned_pairs_without_negative_mining():
    torch.manual_seed(4)
    z = torch.randn(32, 16)
    aligned = vicreg(z, z + 0.01 * torch.randn_like(z))
    shuffled = vicreg(z, z.roll(1, dims=0))
    assert aligned.total < shuffled.total
    assert torch.isfinite(aligned.total)


def test_vicreg_penalizes_collapsed_features():
    diverse = torch.randn(64, 16)
    collapsed = torch.zeros_like(diverse)
    good = vicreg(diverse, diverse.clone())
    bad = vicreg(collapsed, collapsed.clone())
    assert bad.variance > good.variance
    assert bad.min_std < 0.02


def test_vicreg_gradient_flows_to_both_views():
    a = torch.randn(16, 8, requires_grad=True)
    b = torch.randn(16, 8, requires_grad=True)
    vicreg(a, b).total.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert b.grad is not None and torch.isfinite(b.grad).all()


def test_relation_uses_every_augmented_pair_and_sparse_placement_pairs():
    torch.manual_seed(7)
    z_a = torch.randn(16, 8, requires_grad=True)
    z_b = (z_a.detach() + 0.05 * torch.randn(16, 8)).requires_grad_(True)
    left, right = torch.tensor([0, 2]), torch.tensor([1, 3])
    out = relation_loss(z_a, z_b, left, right, cross_placement_weight=0.2)
    assert out.placement_pairs == 2
    assert out.total > out.augmentation.total
    assert out.cross_placement_weighted > 0
    out.total.backward()
    assert z_a.grad is not None and z_b.grad is not None


def test_relation_accepts_one_verified_pair():
    z = torch.eye(4)
    out = relation_loss(z, z.clone(), torch.tensor([0]), torch.tensor([1]))
    assert out.placement_pairs == 1
    assert torch.isfinite(out.total) and out.cross_placement > 0


def test_relation_without_placement_is_universal_vicreg():
    z_a, z_b = torch.randn(8, 4), torch.randn(8, 4)
    relation = relation_loss(z_a, z_b)
    baseline = vicreg(z_a, z_b)
    assert relation.placement_pairs == 0
    assert torch.allclose(relation.total, baseline.total)


def test_masked_ema_latent_is_stop_gradient_and_masked():
    pred = torch.randn(2, 3, 2, 8, requires_grad=True)
    target = pred.detach().clone().requires_grad_(True)
    target.data[:, 0] *= -1
    mask = torch.zeros(2, 3, 2, dtype=torch.bool)
    mask[:, 1:] = True
    loss = masked_ema_latent_loss(pred, target, mask)
    assert loss < 1e-6
    loss.backward()
    assert pred.grad is not None
    assert target.grad is None


def test_masked_ema_latent_weights_resolutions_equally():
    target = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]]])
    pred = target.clone()
    pred[:, 3] = torch.tensor([-1.0, 0.0])
    mask = torch.ones(1, 4, 1, dtype=torch.bool)
    groups = torch.tensor([[0, 0, 0, 1]])
    loss = masked_ema_latent_loss(pred, target, mask, token_groups=groups)
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)


def test_masked_ema_latent_duration_weights_partial_tail():
    target = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
    pred = target.clone()
    pred[:, 1] = torch.tensor([-1.0, 0.0])
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    durations = torch.tensor([[1.0, 0.1]])
    loss = masked_ema_latent_loss(pred, target, mask, token_durations=durations)
    assert torch.allclose(loss, torch.tensor(2.0 / 11.0), atol=1e-6)


def test_phase_a_loss_has_exactly_two_weighted_terms():
    jepa = torch.tensor(2.0)
    relation = torch.tensor(3.0)
    out = phase_a_loss(jepa, relation, jepa_weight=0.5, relation_weight=2.0)
    assert set(out.terms) == {"jepa", "relation"}
    assert out.total == pytest.approx(7.0)


def test_ema_teacher_updates_without_gradients_and_roundtrips_state():
    student = torch.nn.Linear(3, 2, bias=False)
    teacher = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        student.weight.fill_(2.0)
        teacher.weight.zero_()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    update_ema_encoder(student, teacher, decay=0.5)
    assert torch.allclose(teacher.weight, torch.ones_like(teacher.weight))
    assert not teacher.weight.requires_grad

    restored = torch.nn.Linear(3, 2, bias=False)
    restored.load_state_dict(teacher.state_dict())
    assert torch.equal(restored.weight, teacher.weight)


def test_cross_placement_cosine_survives_a_large_common_mean():
    """The cross-placement term must keep discriminating when the projector's mean drifts.

    Cosine is invariant to a global SCALE but not to a common MEAN offset, and nothing in VICReg
    constrains that mean (variance and covariance are both computed mean-centred). A 3,000-step
    run measured ||batch mean|| drifting 2.0 -> 39.6 while the spread stayed ~11, which drove the
    uncentred cross-placement margin from 0.244 to 0.003. Centring makes the term invariant to
    both, so the loss must be (near) unchanged by adding a constant vector to every row.
    """
    torch.manual_seed(0)
    n, d = 64, 128
    spread = torch.randn(n, d)
    partner = spread + 0.3 * torch.randn(n, d)
    z = torch.cat([spread, partner])
    left = torch.arange(n)
    right = torch.arange(n) + n

    def cross_of(offset):
        shifted = z + offset
        out = relation_loss(shifted, shifted.roll(1, 0), left, right,
                            cross_placement_weight=0.1)
        return float(out.cross_placement)

    base = cross_of(torch.zeros(d))
    shifted = cross_of(torch.full((d,), 40.0 / d ** 0.5))   # ||mean|| = 40, the measured drift
    assert abs(shifted - base) < 0.02 * max(base, 1e-6) + 1e-4, (base, shifted)
    # and it must still be far from the no-information value of 1.0
    assert base < 0.5, base
