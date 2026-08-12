"""Correctness and checkpoint compatibility for transformer execution optimizations."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F

from model.tokenizer.transformer import CrossChannelSelfAttention, TemporalSelfAttention


def _legacy_qkv_state(module: torch.nn.Module) -> OrderedDict:
    state = OrderedDict((name, value.clone()) for name, value in module.state_dict().items())
    weight = state.pop("qkv_proj.weight")
    bias = state.pop("qkv_proj.bias")
    for name, value in zip(("q", "k", "v"), weight.chunk(3, dim=0)):
        state[f"{name}_proj.weight"] = value.clone()
    for name, value in zip(("q", "k", "v"), bias.chunk(3, dim=0)):
        state[f"{name}_proj.bias"] = value.clone()
    return state


def test_temporal_attention_strictly_loads_legacy_qkv_checkpoint():
    torch.manual_seed(4)
    original = TemporalSelfAttention(
        32, num_heads=4, dropout=0.0, use_rope=True,
        rope_min_period=0.4, rope_max_period=600.0,
    ).eval()
    restored = TemporalSelfAttention(
        32, num_heads=4, dropout=0.0, use_rope=True,
        rope_min_period=0.4, rope_max_period=600.0,
    ).eval()
    restored.load_state_dict(_legacy_qkv_state(original), strict=True)
    x = torch.randn(6, 11, 32)
    positions = torch.linspace(0.2, 5.8, 11).expand(6, -1)
    valid = torch.ones(6, 11, dtype=torch.bool)
    valid[:, -2:] = False
    with torch.no_grad():
        expected = original(x, key_padding_mask=valid, positions=positions)
        actual = restored(x, key_padding_mask=valid, positions=positions)
    assert torch.equal(expected, actual)


def test_cross_sensor_attention_strictly_loads_legacy_qkv_checkpoint():
    torch.manual_seed(5)
    original = CrossChannelSelfAttention(32, num_heads=4, dropout=0.0).eval()
    restored = CrossChannelSelfAttention(32, num_heads=4, dropout=0.0).eval()
    restored.load_state_dict(_legacy_qkv_state(original), strict=True)
    x = torch.randn(20, 2, 32)
    valid = torch.tensor([[True, True], [True, False]]).repeat(10, 1)
    with torch.no_grad():
        expected = original(x, channel_mask=valid)
        actual = restored(x, channel_mask=valid)
    assert torch.equal(expected, actual)


def test_fused_projection_preserves_qkv_parameter_count():
    attention = TemporalSelfAttention(64, num_heads=8)
    qkv_parameters = attention.qkv_proj.weight.numel() + attention.qkv_proj.bias.numel()
    assert qkv_parameters == 3 * (64 * 64 + 64)


def test_single_sensor_cross_attention_matches_full_sdpa_at_eval():
    torch.manual_seed(17)
    module = CrossChannelSelfAttention(32, num_heads=4, dropout=0.1).eval()
    x = torch.randn(13, 1, 32)
    valid = torch.ones(13, 1, dtype=torch.bool)

    actual = module(x, channel_mask=valid)
    qkv = module.qkv_proj(x).view(13, 1, 3, 4, 8)
    q, k, value = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
    expected = F.scaled_dot_product_attention(
        q, k, value,
        attn_mask=valid.unsqueeze(1).unsqueeze(2),
        dropout_p=0.0,
        scale=module.scale,
    )
    expected = module.out_proj(expected.transpose(1, 2).reshape(13, 1, 32))

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_two_sensor_direct_cross_attention_matches_sdpa_at_eval():
    torch.manual_seed(23)
    module = CrossChannelSelfAttention(32, num_heads=4, dropout=0.1).eval()
    x = torch.randn(17, 2, 32)
    valid = torch.tensor([[True, True], [True, False]]).repeat(9, 1)[:17]

    actual = module(x, channel_mask=valid)
    qkv = module.qkv_proj(x).view(17, 2, 3, 4, 8)
    q, k, value = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
    expected = F.scaled_dot_product_attention(
        q, k, value,
        attn_mask=valid.unsqueeze(1).unsqueeze(2),
        dropout_p=0.0,
        scale=module.scale,
    )
    expected = module.out_proj(expected.transpose(1, 2).reshape(17, 2, 32))

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
