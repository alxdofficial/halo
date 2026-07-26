import numpy as np
import pytest
import torch

from eval.data import EvalStream, load_global_labels
from training.diagnostics.baseline_heterogeneity import (
    _subset_indices,
    transform_stream,
)
from training.diagnostics.zeroshot_difficulty import summarize_label_novelty
from training.evidence.bank_guard import (
    assert_artifact_matches_bank,
    assert_bank_current,
    bank_fingerprint,
    vocab_fingerprint,
)


def _stream(rate=50.0):
    t = np.arange(300) / rate
    acc = np.stack([
        np.sin(2 * np.pi * t),
        np.cos(2 * np.pi * t),
        np.ones_like(t),
    ], axis=-1)
    gyro = np.stack([
        0.2 * np.sin(4 * np.pi * t),
        0.2 * np.cos(4 * np.pi * t),
        np.zeros_like(t),
    ], axis=-1)
    window = np.concatenate([acc, gyro], axis=-1).astype(np.float32)
    return EvalStream(
        dataset="motionsense",
        stream="phone_front_pocket",
        alignment="non_harmonised",
        windows=np.stack([window, window]),
        gt=["walking", "sitting"],
        subjects=np.asarray(["s1", "s2"]),
        channels=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        rate_hz=rate,
        mask=np.ones(6, dtype=bool),
        eval_labels=["walking", "sitting"],
        gravity_state="present",
        channel_descriptions=["channel"] * 6,
    )


def test_subset_indices_is_deterministic_and_covers_every_label():
    labels = np.asarray(["a"] * 20 + ["b"] * 4 + ["c"] * 2)
    first = _subset_indices(labels, 9, seed=7)
    second = _subset_indices(labels, 9, seed=7)
    assert np.array_equal(first, second)
    assert set(labels[first]) == {"a", "b", "c"}
    with pytest.raises(ValueError):
        _subset_indices(labels, 2, seed=7)


def test_baseline_transforms_keep_metadata_and_physics_consistent():
    stream = _stream()
    rate = transform_stream(stream, "rate", 20.0, seed=1)
    assert rate.rate_hz == pytest.approx(20.0)
    assert rate.windows.shape[1] == 120

    channel = transform_stream(stream, "channel", 20.0, seed=1)
    assert channel.mask.tolist() == [True, True, True, False, False, False]
    assert np.count_nonzero(channel.windows[..., 3:]) == 0

    rotated = transform_stream(stream, "orientation", 20.0, seed=1)
    np.testing.assert_allclose(
        np.linalg.norm(rotated.windows[..., :3], axis=-1),
        np.linalg.norm(stream.windows[..., :3], axis=-1),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.linalg.norm(rotated.windows[..., 3:], axis=-1),
        np.linalg.norm(stream.windows[..., 3:], axis=-1),
        atol=1e-5,
    )

    gravity = transform_stream(stream, "gravity", 20.0, seed=1)
    assert gravity.gravity_state == "removed"
    assert all("gravity removed" in text for text in gravity.channel_descriptions[:3])


def test_novelty_summary_reports_macro_and_micro_accuracy():
    rows = [
        {"novel": False, "zs_acc": 1.0, "n_query": 2},
        {"novel": False, "zs_acc": 0.0, "n_query": 6},
        {"novel": True, "zs_acc": 0.5, "n_query": 4},
    ]
    result = summarize_label_novelty(rows)
    assert result["seen"]["macro_label_accuracy"] == 0.5
    assert result["seen"]["micro_query_accuracy"] == 0.25
    assert result["novel"]["micro_query_accuracy"] == 0.5


def _current_bank():
    vocab = load_global_labels()
    bank = {
        "vocab": vocab,
        "vocab_fp": vocab_fingerprint(vocab),
        "backbone": {"fingerprint": "abc"},
        "corpus": {
            "datasets": ["d"],
            "streams": {"d/s": 4},
            "n_encoded_windows": 4,
            "phase_a_corpus_fp": "corpus",
        },
        "max_per_stream": 5,
        "max_per_label": 3,
        "d_model": 2,
        "embed_probe": torch.tensor([1.0, 2.0]),
    }
    bank["bank_fp"] = bank_fingerprint(bank)
    return bank


def test_bank_and_learned_artifact_are_bound_to_exact_bank():
    bank = _current_bank()
    assert_bank_current(bank, context="test")
    artifact = {
        "vocab": list(bank["vocab"]),
        "bank_fp": bank["bank_fp"],
    }
    assert_artifact_matches_bank(
        artifact, bank, context="test", artifact_name="decoder"
    )

    stale = dict(artifact)
    stale["bank_fp"] = "old"
    with pytest.raises(SystemExit):
        assert_artifact_matches_bank(
            stale, bank, context="test", artifact_name="decoder"
        )

    mutated = dict(bank)
    mutated["max_per_label"] = 99
    with pytest.raises(SystemExit):
        assert_bank_current(mutated, context="test")
