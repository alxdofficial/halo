"""Per-patch export must preserve the historical pooled encoder API."""

import numpy as np
import torch

from training.tokenizer.eval_transfer import encode_dataset, encode_dataset_detailed


class _DummyEncoder(torch.nn.Module):
    def __init__(self, multiresolution=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.use_duration_embedding = multiresolution
        self.eval_resolution_pair = (0.5, 1.0)
        self.min_resolution_ratio = 1.75
        self.text_conditioning = "per_channel"

    def forward(self, patches, sampling_rate_hz, patch_len_samples, channel_texts, positions,
                patch_padding_mask=None, **kwargs):
        scalar = patches.mean(dim=(2, 3)) * self.scale
        per_patch = scalar.unsqueeze(-1).repeat(1, 1, 4)
        mask = patch_padding_mask.to(per_patch.dtype)
        pooled = (per_patch * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return {"pooled": pooled, "per_patch": per_patch}


def test_detailed_export_keeps_pooled_result_and_excludes_padding():
    data = np.arange(3 * 120 * 6, dtype=np.float32).reshape(3, 120, 6)
    enc = _DummyEncoder(multiresolution=False)
    detailed = encode_dataset_detailed(
        enc, data, [f"channel {i}" for i in range(6)], torch.device("cpu"), 50.0,
        channel_mask=[True] * 6,
    )
    legacy = encode_dataset(
        enc, data, [f"channel {i}" for i in range(6)], torch.device("cpu"), 50.0,
        channel_mask=[True] * 6,
    )
    assert torch.equal(detailed["pooled"], legacy)
    assert len(detailed["patch_Z"]) == 9  # two full + one end-anchored tail patch per window
    assert torch.bincount(detailed["patch_window"]).tolist() == [3, 3, 3]
    assert (detailed["patch_duration"] > 0).all()
    assert detailed["patch_resolution"].eq(0).all()


def test_multiresolution_export_retains_physical_metadata():
    data = np.ones((2, 120, 6), dtype=np.float32)
    detailed = encode_dataset_detailed(
        _DummyEncoder(multiresolution=True), data, ["x"] * 6, torch.device("cpu"), 50.0,
        channel_mask=[True] * 6,
    )
    assert set(detailed["patch_resolution"].tolist()) == {0, 1}
    assert torch.all(detailed["patch_time"] >= 0)
    assert torch.all(detailed["patch_duration"] > 0)
    for window in range(2):
        rows = detailed["patch_window"].eq(window)
        assert rows.sum() == 8  # five short-grid + three long-grid patches


def test_detailed_export_can_retain_a_live_autograd_graph():
    data = np.ones((2, 120, 6), dtype=np.float32)
    encoder = _DummyEncoder(multiresolution=True)
    detailed = encode_dataset_detailed(
        encoder, data, ["x"] * 6, torch.device("cpu"), 50.0,
        channel_mask=[True] * 6, requires_grad=True,
    )
    detailed["patch_Z"].square().mean().backward()
    assert detailed["patch_Z"].requires_grad
    assert encoder.scale.grad is not None
    assert float(encoder.scale.grad.abs()) > 0
