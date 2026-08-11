"""Gate tests for the sensor-granularity design of record.

These assert the properties the design DEPENDS on, not merely that shapes line up:

  * a dead axis must be distinguishable from an axis that read zero (mhealth's degenerate gyro);
  * the fold must honour the existing `sensor_id` contract, where an accel-only stream maps ALL six
    channel slots to sensor 0 and relies on `channel_mask`;
  * whole-sensor JEPA masking must NEVER cross a placement boundary — the constraint that separates
    the well-posed same-site objective from the ill-posed cross-placement one deleted 2026-08-06;
  * a sensor must never have its signal AND its descriptor hidden at once (nothing left to
    reconstruct from);
  * descriptor scoring must be retrieval-based, and must not turn duplicate descriptions into each
    other's negatives.
"""

from __future__ import annotations

import torch

from model.tokenizer.sensor_tokens import (
    ConditioningProjection, DescriptorHead, SensorFold, descriptor_retrieval_loss,
)
from training.tokenizer.losses_repr import make_sensor_mask_plan


D = 32
SEED = 20260811


# ----------------------------------------------------------------------------- SensorFold
def _fold_inputs(B=2, P=4, C=6):
    tokens = torch.randn(B, P, C, D)
    sensor_id = torch.tensor([[0, 0, 0, 1, 1, 1]] * B)
    channel_mask = torch.ones(B, C, dtype=torch.bool)
    return tokens, sensor_id, channel_mask


def test_fold_produces_one_token_per_sensor():
    tokens, sid, cm = _fold_inputs()
    fold = SensorFold(D)
    out, mask = fold(tokens, sid, cm)
    assert out.shape == (2, 4, 2, D)
    assert mask.all()


def test_absent_axis_is_distinguishable_from_zero_valued_axis():
    """The validity indicator, not just zeroing — otherwise a dead sensor reads as a resting one."""
    tokens, sid, cm = _fold_inputs()
    fold = SensorFold(D).eval()
    cm_dead = cm.clone()
    cm_dead[0, 4] = False                       # gyro-y absent
    tokens_zero = tokens.clone()
    tokens_zero[0, :, 4] = 0.0                  # gyro-y present but reading zero
    with torch.no_grad():
        absent, _ = fold(tokens, sid, cm_dead)
        zeroed, _ = fold(tokens_zero, sid, cm)
    assert not torch.allclose(absent[0, :, 1], zeroed[0, :, 1])


def test_accel_only_stream_honours_the_existing_sensor_id_contract():
    """`stream_sensor_texts` maps ALL six slots to sensor 0 when gyro is absent.

    Absent channels keep a valid index and rely on `channel_mask`. The fold must count only LIVE
    channels; counting all six would make sensor 0 own six channels and raise.
    """
    tokens, _, cm = _fold_inputs()
    sid_all_zero = torch.zeros(2, 6, dtype=torch.long)
    cm_accel = cm.clone()
    cm_accel[:, 3:] = False
    out, mask = SensorFold(D)(tokens, sid_all_zero, cm_accel, n_sensors=1)
    assert out.shape == (2, 4, 1, D)
    assert mask.all()


def test_fold_rejects_a_sensor_with_too_many_live_channels():
    tokens, _, cm = _fold_inputs()
    sid = torch.zeros(2, 6, dtype=torch.long)       # 6 LIVE channels on one sensor
    try:
        SensorFold(D)(tokens, sid, cm, n_sensors=1)
    except ValueError as exc:
        assert "more than 3 live channels" in str(exc)
    else:                                            # pragma: no cover
        raise AssertionError("expected a ValueError for a 6-live-channel sensor")


def test_absent_sensor_reports_absent():
    tokens, sid, cm = _fold_inputs()
    cm2 = cm.clone()
    cm2[0, 3:] = False
    _, mask = SensorFold(D)(tokens, sid, cm2)
    assert mask[0].tolist() == [True, False]
    assert mask[1].tolist() == [True, True]


# ------------------------------------------------------------------- ConditioningProjection
def test_conditioning_is_do_no_harm_at_init():
    """Negative gate bias => the conditioner barely moves the token before training."""
    torch.manual_seed(SEED)
    proj = ConditioningProjection(9, D).eval()
    tokens = torch.randn(2, 4, 2, D)
    with torch.no_grad():
        out = proj(tokens, torch.randn(2, 2, 9), torch.ones(2, 2, dtype=torch.bool))
    assert (out - tokens).abs().max() < 0.5


def test_conditioning_is_suppressed_for_absent_sensors():
    torch.manual_seed(SEED)
    proj = ConditioningProjection(9, D).eval()
    tokens = torch.randn(2, 4, 2, D)
    valid = torch.tensor([[True, False], [True, True]])
    with torch.no_grad():
        out = proj(tokens, torch.randn(2, 2, 9), valid)
    assert torch.allclose(out[0, :, 1], tokens[0, :, 1])       # absent sensor untouched


# ------------------------------------------------------------------------ descriptor scoring
def test_descriptor_loss_collapses_duplicate_descriptions():
    """Two rows sharing a sensor description must not become each other's negatives."""
    torch.nn.functional.normalize
    target = torch.nn.functional.normalize(torch.randn(1, 384), dim=-1).repeat(4, 1)
    pred = torch.nn.functional.normalize(torch.randn(4, 384), dim=-1)
    loss, acc = descriptor_retrieval_loss(pred, target)
    assert float(loss) == 0.0 and float(acc) == 1.0


def test_descriptor_head_emits_unit_vectors_in_frozen_text_space():
    head = DescriptorHead(D).eval()
    with torch.no_grad():
        out = head(torch.randn(2, 3, D))
    assert out.shape == (2, 3, 384)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2, 3), atol=1e-5)


# ------------------------------------------------------------------------- sensor mask plan
def _plan(placement, present=None, B=256, T=6, S=4, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    present = present if present is not None else torch.ones(B, S, dtype=torch.bool)
    return make_sensor_mask_plan(B, T, S, generator=g, sensor_present=present,
                                 sensor_placement=placement.expand(B, S).contiguous())


def test_whole_sensor_masking_never_crosses_a_placement():
    """THE constraint. A sensor alone at its placement has no co-tenant to be predicted from."""
    plan = _plan(torch.tensor([[0, 1, 2, 3]]))          # every sensor alone
    assert int(plan.token_mask.all(dim=1).sum()) == 0


def test_whole_sensor_masking_only_hits_co_tenants():
    plan = _plan(torch.tensor([[0, 0, 1, 2]]))          # only sensors 0,1 share a placement
    per_sensor = plan.token_mask.all(dim=1).sum(dim=0)
    assert per_sensor[2] == 0 and per_sensor[3] == 0
    assert per_sensor[0] > 0 and per_sensor[1] > 0


def test_signal_and_descriptor_are_never_both_hidden():
    plan = _plan(torch.tensor([[0, 0, 1, 1]]))
    assert int((plan.token_mask.all(dim=1) & plan.descriptor_mask).sum()) == 0


def test_absent_sensors_are_never_masked():
    present = torch.ones(256, 4, dtype=torch.bool)
    present[:, 3] = False
    plan = _plan(torch.tensor([[0, 0, 1, 1]]), present=present)
    assert int(plan.token_mask[:, :, 3].sum()) == 0
    assert int(plan.descriptor_mask[:, 3].sum()) == 0


def test_mask_events_fire_at_roughly_the_configured_rate():
    plan = _plan(torch.tensor([[0, 0, 1, 1]]), B=2048)
    sensor_rate = float(plan.token_mask.all(dim=1).any(dim=1).float().mean())
    descriptor_rate = float(plan.descriptor_mask.any(dim=1).float().mean())
    assert 0.15 < sensor_rate < 0.35
    assert 0.15 < descriptor_rate < 0.35


# --------------------------------------------------------------------------- encoder wiring
def test_sensor_encoder_forward_shapes_and_masking():
    from model.tokenizer.encoder import SetTokenizerEncoder

    B, P, C, S_PAD, N_TRUE = 2, 5, 6, 256, 64
    enc = SetTokenizerEncoder(d_model=64, num_layers=2, num_heads=4, dim_feedforward=128,
                              dropout=0.0, token_granularity="sensor", sensor_bias_dim=9).eval()
    patches = torch.zeros(B, P, S_PAD, C)
    patches[:, :, :N_TRUE] = torch.randn(B, P, N_TRUE, C)
    sensor_texts = [["a phone accelerometer on the front pocket; includes gravity",
                     "a phone gyroscope on the front pocket"]] * B
    role = [["x", "y", "z", "x", "y", "z"]] * B
    sid = torch.tensor([[0, 0, 0, 1, 1, 1]] * B)
    pos = torch.arange(P).float().unsqueeze(0).expand(B, P).contiguous()
    cm = torch.ones(B, C, dtype=torch.bool)
    bias = torch.randn(B, 2, 9)

    with torch.no_grad():
        out = enc(patches, 50.0, N_TRUE, role, pos, channel_mask=cm, sensor_texts=sensor_texts,
                  sensor_id=sid, sensor_bias=bias)
    assert out["tokens"].shape == (B, P, 2, 64)          # per SENSOR, not per channel
    assert out["descriptor"].shape == (B, 2, 384)
    assert out["descriptor_pred"].shape == (B, 2, 384)
    assert out["sensor_present"].all()

    token_mask = torch.zeros(B, P, 2, dtype=torch.bool)
    token_mask[:, :, 1] = True
    with torch.no_grad():
        masked = enc(patches, 50.0, N_TRUE, role, pos, channel_mask=cm,
                     sensor_texts=sensor_texts, sensor_id=sid, sensor_bias=bias,
                     token_mask=token_mask,
                     descriptor_mask=torch.tensor([[True, False]] * B))
    assert not torch.allclose(out["pooled"], masked["pooled"])


def test_sensor_encoder_handles_accel_only_streams():
    from model.tokenizer.encoder import SetTokenizerEncoder

    B, P, C, S_PAD, N_TRUE = 2, 5, 6, 256, 64
    enc = SetTokenizerEncoder(d_model=64, num_layers=2, num_heads=4, dim_feedforward=128,
                              dropout=0.0, token_granularity="sensor", sensor_bias_dim=9).eval()
    patches = torch.zeros(B, P, S_PAD, C)
    patches[:, :, :N_TRUE] = torch.randn(B, P, N_TRUE, C)
    cm = torch.ones(B, C, dtype=torch.bool)
    cm[:, 3:] = False
    pos = torch.arange(P).float().unsqueeze(0).expand(B, P).contiguous()
    with torch.no_grad():
        out = enc(patches, 50.0, N_TRUE, [["x", "y", "z", "x", "y", "z"]] * B, pos,
                  channel_mask=cm,
                  sensor_texts=[["a watch accelerometer on the wrist; includes gravity"]] * B,
                  sensor_id=torch.zeros(B, C, dtype=torch.long),
                  sensor_bias=torch.randn(B, 1, 9))
    assert out["tokens"].shape == (B, P, 1, 64)
    assert out["sensor_present"].all()
