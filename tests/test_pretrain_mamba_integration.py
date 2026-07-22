"""Integration tests: the mamba frontend is reachable from the training harness (audit #2).

The prior unit tests only built SetTokenizerEncoder(frontend="mamba") directly; the audits noted
nothing exercised PipelineAModel, the loop's frontend touchpoints, calibration, or a checkpoint
round-trip. These do.
"""

from dataclasses import asdict

import torch

from model.tokenizer.mamba_frontend import SelectiveSSMChannelTokenizer
from training.tokenizer.eval_transfer import build_encoder
from training.tokenizer.pretrain import PipelineAModel, PretrainConfig


def _cfg(frontend):
    return PretrainConfig(d_model=32, num_layers=1, num_heads=2, dim_feedforward=64,
                          frontend=frontend, mamba_d_state=8, multiresolution=False)


def test_pipeline_model_builds_mamba_not_a_silent_filterbank():
    """The regression that started this: frontend='mamba' must actually build the SSM tokenizer."""
    model = PipelineAModel(_cfg("mamba"), a1_target_dim=34)
    assert isinstance(model.encoder.filterbank, SelectiveSSMChannelTokenizer)
    # and the fixed arm is unchanged
    from model.tokenizer.filterbank import PhysicalFilterbankTokenizer
    assert isinstance(PipelineAModel(_cfg("fixed"), a1_target_dim=34).encoder.filterbank,
                      PhysicalFilterbankTokenizer)


def test_loop_frontend_touchpoints_exist_for_mamba():
    """Every attribute/method the training loop calls on model.encoder.filterbank must work."""
    fe = PipelineAModel(_cfg("mamba"), a1_target_dim=34).encoder.filterbank
    assert fe.learnable is True                                   # logging gate (pretrain.py ~647)
    reg = fe.adaptation_regularization()                          # loss term (pretrain.py ~619)
    assert torch.isfinite(reg) and reg.requires_grad
    summ = fe.adaptation_summary()                               # logging (pretrain.py ~648)
    assert "frontend/delta_mult_baseline_mean" in summ


def test_loop_calibration_path_works_for_mamba():
    """The frontend-agnostic calibration the loop now runs (accumulate/finalize on the frontend)."""
    fe = PipelineAModel(_cfg("mamba"), a1_target_dim=34).encoder.filterbank
    fe.reset_norm_accumulator()
    fe.accumulate_norm_stats(torch.randn(4, 3, 256, 6), 50.0, torch.full((4, 3), 200),
                             patch_mask=torch.ones(4, 3, dtype=torch.bool),
                             channel_mask=torch.ones(4, 6, dtype=torch.bool))
    fe.finalize_norm_stats()
    assert fe._norm_fitted.item() == 1.0 and fe.norm_sd.numel() == 2   # per-modality


def test_checkpoint_roundtrip_reconstructs_mamba():
    """A mamba checkpoint must reconstruct as a mamba encoder (was: always a filterbank)."""
    model = PipelineAModel(_cfg("mamba"), a1_target_dim=34)
    ckpt = {"config": asdict(_cfg("mamba")), "encoder": model.encoder.state_dict()}
    enc = build_encoder(ckpt, torch.device("cpu"))
    assert isinstance(enc.filterbank, SelectiveSSMChannelTokenizer)
    # state actually loaded (params equal)
    a = dict(model.encoder.named_parameters()); b = dict(enc.named_parameters())
    assert torch.allclose(a["filterbank.A_log"], b["filterbank.A_log"])


def test_mamba_config_is_stamped_in_checkpoint():
    """d_state/d_conv/scan_chunk must survive in the checkpoint for reproducibility (audit #8)."""
    c = asdict(_cfg("mamba"))
    for k in ("mamba_d_state", "mamba_d_conv", "mamba_scan_chunk", "frontend"):
        assert k in c


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
