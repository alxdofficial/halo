"""Tests for factored text conditioning (docs/design/TEXT_CONDITIONING.md).

Accel and gyro are modelled as two distinct modality-level SENSORS, so the tokenizer's identity
text factors into:
  * per-CHANNEL role text — AXIS only ("x"/"y"/"z"), corpus-constant; and
  * per-SENSOR identity text — TWO strings [accel, gyro], each carrying device + modality + placement
    (+ the gravity convention on the ACCEL sensor only), broadcast to that sensor's channels via
    ``sensor_id`` = [0,0,0,1,1,1] (accel triad -> sensor 0, gyro triad -> sensor 1).

These tests pin the properties that make the change safe and honest: axis lives only in the role and
device/modality/placement/gravity only in the sensor (no double-injection); the modality identity is
broadcast by ``sensor_id`` so accel vs gyro channels get distinct identities; the gated residual is
do-no-harm / ablatable at init; and the legacy per-channel path is unchanged.
"""

import torch

from model.tokenizer.channel_text import FactoredChannelTextFusion, _TextPooler
from model.tokenizer.encoder import SetTokenizerEncoder
from training.tokenizer.pretrain_data import CHANNELS, stream_sensor_texts


# ----------------------------------------------------------------------------- data-side factoring
def test_role_text_is_axis_only_and_sensor_id_groups_by_modality():
    """(a) role = pure axis (no modality/placement); (d) sensor_id groups accel vs gyro."""
    role, sensor, sensor_id = stream_sensor_texts("wisdm", "watch_wrist")
    assert role == ["x", "y", "z", "x", "y", "z"]        # axis ONLY
    assert len(role) == len(CHANNELS)
    assert len(sensor) == 2                               # two modality sensors
    assert sensor_id == [0, 0, 0, 1, 1, 1]               # accel triad -> 0, gyro triad -> 1
    for r in role:
        for banned in ("accelerometer", "gyroscope", "axis", "wrist", "watch", "gravity"):
            assert banned not in r, f"{banned!r} leaked into role {r!r}"


def test_each_sensor_string_carries_device_modality_and_placement():
    """(b) device + modality + placement all live in each sensor string; axis never does."""
    _, (accel, gyro), _ = stream_sensor_texts("wisdm", "watch_wrist")
    for s in (accel, gyro):
        assert "watch" in s and "wrist" in s             # device + placement present
        assert "-axis" not in s and s not in ("x", "y", "z")   # axis absent
    assert "accelerometer" in accel and "gyroscope" not in accel   # accel sensor is the accel modality
    assert "gyroscope" in gyro and "accelerometer" not in gyro     # gyro sensor is the gyro modality


def test_gravity_convention_is_on_accel_sensor_only():
    """(c) the gravity clause rides on the accel sensor; the gyro sensor never mentions gravity."""
    # gravity-present stream
    _, (accel, gyro), _ = stream_sensor_texts("wisdm", "watch_wrist")
    assert "gravity" in accel and "includes gravity" in accel
    assert "gravity" not in gyro
    # gravity-removed stream: still ONLY the accel sensor carries the clause
    _, (accel_r, gyro_r), _ = stream_sensor_texts("kuhar", "phone_waist")
    assert "gravity removed" in accel_r
    assert "gravity" not in gyro_r


def test_accel_only_stream_still_returns_two_sensors_and_canonical_ids():
    """Acc-only streams still factor into 2 sensors + [0,0,0,1,1,1]; the gyro entry is unused
    (those slots are channel_mask-absent) but keeps sensor_id a valid index for all six slots."""
    for ds, st in (("capture24", "watch_wrist"), ("unimib_shar", "phone_pocket")):
        role, sensor, sensor_id = stream_sensor_texts(ds, st)
        assert role == ["x", "y", "z", "x", "y", "z"]
        assert len(sensor) == 2
        assert sensor_id == [0, 0, 0, 1, 1, 1]
        assert "accelerometer" in sensor[0] and "gyroscope" in sensor[1]


def test_role_text_is_constant_across_streams_sensor_text_is_not():
    r1, s1, _ = stream_sensor_texts("wisdm", "watch_wrist")
    r2, s2, _ = stream_sensor_texts("wisdm", "phone_pocket")
    assert r1 == r2, "role text must be corpus-constant (axis only)"
    assert s1 != s2, "sensor text must vary with device/placement"


# ------------------------------------------------------------------------------------ the pooler
def test_text_pooler_handles_all_masked_row_without_nan():
    torch.manual_seed(0)
    pool = _TextPooler(d_model=32, text_dim=16).eval()
    toks = torch.randn(3, 5, 16)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[1] = False                                # a padding row
    with torch.no_grad():
        out = pool(toks, mask)
    assert out.shape == (3, 32) and torch.isfinite(out).all()


# --------------------------------------------------------------------------------- the fusion
def _fusion_inputs(B=2, P=4, C=6, d=32, text_dim=16, S=5, n_sensors=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        sensor_tokens=torch.randn(B, P, C, d, generator=g),
        role_tokens=torch.randn(B, C, S, text_dim, generator=g),
        role_mask=torch.ones(B, C, S, dtype=torch.bool),
        sensor_text_tokens=torch.randn(B, n_sensors, S, text_dim, generator=g),
        sensor_text_mask=torch.ones(B, n_sensors, S, dtype=torch.bool),
        sensor_id=torch.zeros(B, C, dtype=torch.long),
    )


def test_gate_bias_makes_identity_negligible_at_init():
    """Do-no-harm / ablatable: a very negative gate bias => fusion ≈ identity map on sensor tokens."""
    torch.manual_seed(0)
    fusion = FactoredChannelTextFusion(d_model=32, text_dim=16, gate_bias_init=-30.0).eval()
    x = _fusion_inputs()
    with torch.no_grad():
        out = fusion(**x)
    assert torch.allclose(out, x["sensor_tokens"], atol=1e-4), "identity should be ~off at very negative gate bias"


def test_sensor_identity_is_broadcast_by_sensor_id():
    """Two channels in the SAME sensor get the SAME sensor contribution; different sensors differ."""
    torch.manual_seed(1)
    fusion = FactoredChannelTextFusion(d_model=16, text_dim=8, gate_bias_init=0.0).eval()
    B, P, C, d, S = 1, 1, 4, 16, 3
    # zero the role path so we isolate the sensor contribution, and zero sensor tokens so the
    # output IS the gated sensor identity.
    role_tokens = torch.zeros(B, C, S, 8)
    role_mask = torch.ones(B, C, S, dtype=torch.bool)
    sensor_text_tokens = torch.randn(B, 2, S, 8)     # two DISTINCT sensors
    sensor_text_mask = torch.ones(B, 2, S, dtype=torch.bool)
    sensor_id = torch.tensor([[0, 0, 1, 1]])         # ch0,1 -> sensor0 ; ch2,3 -> sensor1
    sensor_tokens = torch.zeros(B, P, C, d)
    with torch.no_grad():
        out = fusion(sensor_tokens, role_tokens, role_mask,
                     sensor_text_tokens, sensor_text_mask, sensor_id)[0, 0]   # (C, d)
    assert torch.allclose(out[0], out[1], atol=1e-6), "same-sensor channels must match"
    assert torch.allclose(out[2], out[3], atol=1e-6), "same-sensor channels must match"
    assert not torch.allclose(out[0], out[2], atol=1e-4), "different sensors must differ"


def test_modality_sensor_identity_broadcasts_accel_vs_gyro():
    """(e) With 2 distinct sensor texts + sensor_id [0,0,0,1,1,1], the fusion gives the accel identity
    to channels 0-2 and the (different) gyro identity to channels 3-5 — the accel/gyro modality split."""
    torch.manual_seed(3)
    fusion = FactoredChannelTextFusion(d_model=16, text_dim=8, gate_bias_init=0.0).eval()
    B, P, C, d, S = 1, 1, 6, 16, 3
    # zero the role path + sensor tokens so the output IS the gated sensor identity per channel.
    role_tokens = torch.zeros(B, C, S, 8)
    role_mask = torch.ones(B, C, S, dtype=torch.bool)
    sensor_text_tokens = torch.randn(B, 2, S, 8)         # sensor 0 = accel, sensor 1 = gyro (distinct)
    sensor_text_mask = torch.ones(B, 2, S, dtype=torch.bool)
    sensor_id = torch.tensor([[0, 0, 0, 1, 1, 1]])       # the canonical modality grouping
    sensor_tokens = torch.zeros(B, P, C, d)
    with torch.no_grad():
        out = fusion(sensor_tokens, role_tokens, role_mask,
                     sensor_text_tokens, sensor_text_mask, sensor_id)[0, 0]   # (C, d)
    for i in (1, 2):
        assert torch.allclose(out[0], out[i], atol=1e-6), "accel channels (0-2) must share one identity"
    for i in (4, 5):
        assert torch.allclose(out[3], out[i], atol=1e-6), "gyro channels (3-5) must share one identity"
    assert not torch.allclose(out[0], out[3], atol=1e-4), "accel vs gyro identities must differ"


def test_identity_is_role_plus_sensor_sum():
    """identity = role + sensor: with the gate forced to 1, output - sensor_tokens = role_pool + sensor_pool."""
    torch.manual_seed(2)
    fusion = FactoredChannelTextFusion(d_model=16, text_dim=8, gate_bias_init=0.0).eval()
    x = _fusion_inputs(d=16, text_dim=8)
    B, P, C, d = x["sensor_tokens"].shape
    # Force gate == 1 everywhere so the residual is exactly the identity vector.
    with torch.no_grad():
        fusion.gate_sensor.weight.zero_()
        fusion.gate_identity.weight.zero_()
        fusion.gate_identity.bias.fill_(30.0)        # sigmoid(30) ~ 1
        out = fusion(**x)
        role = fusion.pool(x["role_tokens"].reshape(B * C, -1, 8),
                           x["role_mask"].reshape(B * C, -1)).reshape(B, C, d)
        sens = fusion.pool(x["sensor_text_tokens"].reshape(B, -1, 8),
                           x["sensor_text_mask"].reshape(B, -1)).reshape(B, 1, d)
        expected_identity = role + sens              # sensor_id all 0 -> broadcast sensor 0
        got_identity = (out - x["sensor_tokens"]).mean(dim=1)   # constant over patches
    assert torch.allclose(got_identity, expected_identity, atol=1e-4)


# --------------------------------------------------------------------------- encoder integration
def test_encoder_factored_forward_runs_with_two_modality_sensors_and_legacy_is_unchanged():
    """Both paths run; the factored path threads the two modality sensors + [0,0,0,1,1,1] end-to-end;
    and constructing 'per_channel' still builds the legacy fusion."""
    from model.tokenizer.channel_text import ChannelTextFusion
    legacy = SetTokenizerEncoder(d_model=32, num_layers=1, num_heads=4, dim_feedforward=64)
    assert isinstance(legacy.fusion, ChannelTextFusion)

    enc = SetTokenizerEncoder(d_model=32, num_layers=1, num_heads=4, dim_feedforward=64,
                              text_conditioning="factored")
    assert isinstance(enc.fusion, FactoredChannelTextFusion)

    B, P, C, S = 2, 3, 6, enc.filterbank.S
    patches = torch.randn(B, P, S, C)
    role_texts = [["x", "y", "z", "x", "y", "z"] for _ in range(B)]
    sensor_texts = [
        ["a watch accelerometer on the wrist; includes gravity",
         "a watch gyroscope on the wrist"]
        for _ in range(B)
    ]
    sensor_id = torch.tensor([[0, 0, 0, 1, 1, 1]] * B)
    positions = torch.arange(P).float().unsqueeze(0).repeat(B, 1)
    out = enc(patches, 50.0, S, role_texts, positions,
              channel_mask=torch.ones(B, C, dtype=torch.bool),
              patch_padding_mask=torch.ones(B, P, dtype=torch.bool),
              sensor_texts=sensor_texts, sensor_id=sensor_id)
    assert out["pooled"].shape == (B, 32) and torch.isfinite(out["pooled"]).all()


def test_encoder_embedding_changes_when_gyro_sensor_text_changes():
    """The gyro modality sensor is operational: changing ONLY the gyro sensor string moves the
    pooled embedding (it is broadcast to channels 3-5 via sensor_id)."""
    torch.manual_seed(8)
    enc = SetTokenizerEncoder(d_model=24, num_layers=1, num_heads=4, dim_feedforward=48,
                              dropout=0.0, text_conditioning="factored").eval()
    B, P, C, S = 1, 2, 6, enc.filterbank.S
    patches = torch.randn(B, P, S, C)
    roles = [["x", "y", "z", "x", "y", "z"]]
    positions = torch.arange(P).float().unsqueeze(0)
    kwargs = dict(
        channel_mask=torch.ones(B, C, dtype=torch.bool),
        patch_padding_mask=torch.ones(B, P, dtype=torch.bool),
        sensor_id=torch.tensor([[0, 0, 0, 1, 1, 1]]),
    )
    accel = "a watch accelerometer on the wrist; includes gravity"
    with torch.no_grad():
        wrist = enc(patches, 50.0, S, roles, positions,
                    sensor_texts=[[accel, "a watch gyroscope on the wrist"]], **kwargs)["pooled"]
        pocket = enc(patches, 50.0, S, roles, positions,
                     sensor_texts=[[accel, "a phone gyroscope in the right pocket"]], **kwargs)["pooled"]
    assert not torch.allclose(wrist, pocket, atol=1e-6), "gyro sensor text has no operational effect"


def test_encoder_factored_requires_sensor_inputs():
    enc = SetTokenizerEncoder(d_model=16, num_layers=1, num_heads=2, dim_feedforward=32,
                              text_conditioning="factored")
    B, P, C, S = 1, 2, 6, enc.filterbank.S
    patches = torch.randn(B, P, S, C)
    positions = torch.zeros(B, P)
    role_texts = [["x", "y", "z", "x", "y", "z"]]
    try:
        enc(patches, 50.0, S, role_texts, positions)     # missing sensor_texts/sensor_id
        raise AssertionError("expected ValueError for missing factored inputs")
    except ValueError:
        pass


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
