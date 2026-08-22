"""The encoder-swap harness must be a faithful stand-in, or the comparison measures the harness.

Each test here pins one property that, if broken, would silently make a third-party backbone look
worse than it is: rate-correct resampling, an identical row population, matched row scale, real
freezing, and sensor isolation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model.tokenizer.baseline_backbone import BaselineRowEncoder


def _batch(n=8, patches=6, samples=64, channels=6, rate=50.0, slots=2):
    torch.manual_seed(0)
    sensor_id = torch.zeros(n, channels, dtype=torch.long)
    sensor_id[:, 3:] = 1
    channel_mask = torch.ones(n, channels, dtype=torch.bool)
    if slots == 1:
        channel_mask[:, 3:] = False
    return {
        "patches": torch.randn(n, patches, samples, channels),
        "rates": torch.full((n,), rate),
        "patch_len": torch.full((n, patches), samples, dtype=torch.long),
        "positions": torch.arange(patches, dtype=torch.float32).repeat(n, 1),
        "patch_durations": torch.ones(n, patches),
        "channel_mask": channel_mask,
        "patch_padding_mask": torch.ones(n, patches, dtype=torch.bool),
        "sensor_id": sensor_id,
        "role_texts": [["x", "y", "z"] * 2 for _ in range(n)],
        "sensor_texts": [["a wrist accelerometer"][:1] * (1 if slots == 1 else 2)
                         for _ in range(n)],
        "source_rates": torch.full((n,), rate),
    }


def _run(encoder, batch):
    return encoder(
        batch["patches"], batch["rates"], batch["patch_len"], batch["role_texts"],
        batch["positions"], patch_durations=batch["patch_durations"],
        channel_mask=batch["channel_mask"], patch_padding_mask=batch["patch_padding_mask"],
        sensor_texts=batch["sensor_texts"], sensor_id=batch["sensor_id"],
        source_rate_hz=batch["source_rates"],
    )


@pytest.fixture(scope="module")
def harnet():
    return BaselineRowEncoder("harnet", d_model=128, freeze=True, device=torch.device("cpu"))


def test_rejects_an_unknown_backbone():
    with pytest.raises(ValueError):
        BaselineRowEncoder("resnet50")


def test_rows_carry_the_engine_contract(harnet):
    batch = _batch()
    out = _run(harnet, batch)
    assert out["retrieval_tokens"].shape == (8, 6, 2, 128)
    assert out["sensor_present"].shape == (8, 2)
    assert out["descriptor"] is not None                 # live_sensor_rows indexes this
    assert torch.isfinite(out["retrieval_tokens"]).all()


def test_row_scale_matches_our_trunks_contract(harnet):
    """Our trunk emits rows of norm sqrt(d); an arm with a different scale tests a different
    numerical regime in the mixer rather than its own representation."""
    out = _run(harnet, _batch())
    present = out["sensor_present"]
    rows = out["retrieval_tokens"][:, 0][present]
    norms = rows.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, 128 ** 0.5), rtol=2e-3)


def test_resampling_follows_the_sampling_RATE_not_the_array_length(harnet):
    """The same waveform recorded at 25 Hz and at 50 Hz must reach the backbone alike.

    Resampling by array length instead of by rate time-compresses the signal -- an artefact that
    would handicap every third-party backbone and stay invisible in the loss.
    """
    def sine_batch(rate, samples):
        batch = _batch(n=2, samples=samples, rate=rate, slots=1)
        t = torch.arange(6 * samples, dtype=torch.float32) / rate      # seconds
        wave = torch.stack([torch.sin(2 * np.pi * f * t) for f in (1.0, 2.5, 4.0)], dim=-1)
        patched = wave.reshape(1, 6, samples, 3).repeat(2, 1, 1, 1)
        batch["patches"] = torch.zeros(2, 6, samples, 6)
        batch["patches"][:, :, :, :3] = patched
        return batch

    seen = []
    original = harnet._features

    def spy(window, joint):
        seen.append(window.clone())
        return original(window, joint)

    harnet._features = spy
    try:
        _run(harnet, sine_batch(25.0, 32))
        _run(harnet, sine_batch(50.0, 64))
    finally:
        harnet._features = original

    slow, fast = seen[0], seen[1]
    assert slow.shape == fast.shape == (2, 150, 3)
    # Both describe the same seconds of the same sinusoids resampled to 30 Hz. They are not
    # bit-identical -- linear interpolation from 25 Hz costs more on the 4 Hz component than from
    # 50 Hz -- but they must be the same waveform, not one a time-scaled copy of the other.
    # residual is pure interpolation error and scales with frequency: 0.008 at 1 Hz, 0.12 at 4 Hz
    assert torch.allclose(slow, fast, atol=0.15)
    a = (slow - slow.mean()).flatten()
    b = (fast - fast.mean()).flatten()
    correlation = float(a @ b / (a.norm() * b.norm()))
    assert correlation > 0.999, correlation
    # a length-based resample would time-compress the 25 Hz batch by 2x: assert that it does NOT
    shifted = torch.nn.functional.interpolate(
        slow[:, :75].permute(0, 2, 1), size=150, mode="linear", align_corners=True
    ).permute(0, 2, 1)
    assert torch.norm(slow - fast) < torch.norm(shifted - fast)


def test_a_frozen_backbone_stays_frozen_through_encoder_train():
    """`nn.Module.train()` recurses; without an override the frozen arm's BatchNorm statistics
    would drift on our corpus and the arm would not be frozen at all."""
    frozen = BaselineRowEncoder("harnet", freeze=True, device=torch.device("cpu"))
    frozen.train()
    norms = [m for m in frozen.net.modules()
             if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    assert norms and all(not m.training for m in norms)
    assert not any(p.requires_grad for p in frozen.net.parameters())


def test_a_fine_tuned_backbone_actually_receives_gradient():
    """Both loaders hand back gradient-disabled models, so 'fine-tuned' must switch them on."""
    live = BaselineRowEncoder("harnet", freeze=False, device=torch.device("cpu"))
    live.train()
    _run(live, _batch(n=4))["retrieval_tokens"].pow(2).mean().backward()
    grads = [p.grad for p in live.net.parameters() if p.grad is not None]
    assert grads and torch.stack([g.norm() for g in grads]).norm() > 0


def test_an_accel_row_ignores_the_gyroscope(harnet):
    """Rows are the retrieval unit; one sensor's row must not depend on a co-resident sensor."""
    batch = _batch()
    before = _run(harnet, batch)["retrieval_tokens"][:, :, 0]
    batch["patches"][:, :, :, 3:] = torch.randn_like(batch["patches"][:, :, :, 3:])
    after = _run(harnet, batch)["retrieval_tokens"][:, :, 0]
    assert torch.equal(before, after)


def test_padding_and_mixed_rates_do_not_inject_zeros(harnet):
    """A short window must resample from its own valid samples, never from padding."""
    batch = _batch(n=4)
    batch["patch_len"][:, 3:] = 0
    batch["patch_padding_mask"][:, 3:] = False
    batch["patches"][:, 3:] = 12345.0                    # poison the padded region
    out = _run(harnet, batch)
    assert torch.isfinite(out["retrieval_tokens"]).all()
    assert out["retrieval_tokens"].abs().max() < 1e3


def test_a_single_sample_window_still_produces_a_row(harnet):
    """~0.1% of corpus windows (wisdm, capture24, opportunity) carry exactly one sample.

    Our encoder emits rows for them, so refusing them would shrink the baseline arms' row
    population and stop the comparison being matched -- and it crashed all four arms at
    validation before this was handled.
    """
    batch = _batch(n=4)
    batch["patch_len"][:] = 0
    batch["patch_len"][:, 0] = 1
    batch["patch_padding_mask"][:] = False
    batch["patch_padding_mask"][:, 0] = True
    out = _run(harnet, batch)
    assert torch.isfinite(out["retrieval_tokens"]).all()
    assert out["sensor_present"].any()
