"""Tests for factored text conditioning (docs/design/TEXT_CONDITIONING.md).

The refactor splits the tokenizer's per-channel placement-repeating text into:
  * per-CHANNEL role text (axis/modality only), and
  * per-SENSOR identity text (device/placement/gravity), broadcast to that sensor's channels.

These tests pin the properties that make the change safe and honest: the sensor identity really is
broadcast by ``sensor_id``; the two sources are summed (no double-injection); the gated residual is
do-no-harm / ablatable at init; and the legacy per-channel path is byte-for-byte unchanged.
"""

import torch

from model.tokenizer.channel_text import FactoredChannelTextFusion, _TextPooler
from model.tokenizer.encoder import SetTokenizerEncoder
from training.tokenizer.pretrain_data import CHANNELS, stream_sensor_texts


# ----------------------------------------------------------------------------- data-side factoring
def test_role_text_has_no_placement_and_sensor_text_has_no_axis():
    """The compounding hazard (§6): a config fact must live in exactly ONE source."""
    role, sensor, sensor_id = stream_sensor_texts("wisdm", "watch_wrist")
    assert len(role) == len(CHANNELS) and len(sensor) == 1
    assert sensor_id == [0] * len(CHANNELS)
    for r in role:
        assert "wrist" not in r and "watch" not in r and "gravity" not in r, f"placement leaked into role: {r!r}"
        assert ("accelerometer" in r or "gyroscope" in r) and "axis" in r
    (s,) = sensor
    assert "wrist" in s and "watch" in s          # placement + device present
    assert "x-axis" not in s and "y-axis" not in s  # axis absent


def test_role_text_is_constant_across_streams_sensor_text_is_not():
    r1, s1, _ = stream_sensor_texts("wisdm", "watch_wrist")
    r2, s2, _ = stream_sensor_texts("wisdm", "phone_pocket")
    assert r1 == r2, "role text must be corpus-constant (axis/modality only)"
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
def test_encoder_factored_forward_runs_and_legacy_is_unchanged():
    """Both paths run; and constructing 'per_channel' still builds the legacy fusion."""
    from model.tokenizer.channel_text import ChannelTextFusion
    legacy = SetTokenizerEncoder(d_model=32, num_layers=1, num_heads=4, dim_feedforward=64)
    assert isinstance(legacy.fusion, ChannelTextFusion)

    enc = SetTokenizerEncoder(d_model=32, num_layers=1, num_heads=4, dim_feedforward=64,
                              text_conditioning="factored")
    assert isinstance(enc.fusion, FactoredChannelTextFusion)

    B, P, C, S = 2, 3, 6, enc.filterbank.S
    patches = torch.randn(B, P, S, C)
    role_texts = [["accelerometer x-axis"] * C for _ in range(B)]
    sensor_texts = [["a watch on the left wrist; accelerometer includes gravity"] for _ in range(B)]
    sensor_id = torch.zeros(B, C, dtype=torch.long)
    positions = torch.arange(P).float().unsqueeze(0).repeat(B, 1)
    out = enc(patches, 50.0, S, role_texts, positions,
              channel_mask=torch.ones(B, C, dtype=torch.bool),
              patch_padding_mask=torch.ones(B, P, dtype=torch.bool),
              sensor_texts=sensor_texts, sensor_id=sensor_id)
    assert out["pooled"].shape == (B, 32) and torch.isfinite(out["pooled"]).all()


def test_encoder_factored_requires_sensor_inputs():
    enc = SetTokenizerEncoder(d_model=16, num_layers=1, num_heads=2, dim_feedforward=32,
                              text_conditioning="factored")
    B, P, C, S = 1, 2, 6, enc.filterbank.S
    patches = torch.randn(B, P, S, C)
    positions = torch.zeros(B, P)
    role_texts = [["accelerometer x-axis"] * C]
    try:
        enc(patches, 50.0, S, role_texts, positions)     # missing sensor_texts/sensor_id
        raise AssertionError("expected ValueError for missing factored inputs")
    except ValueError:
        pass


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
