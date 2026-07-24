"""Regression tests for Phase-B episodic retrieval and variable label budgets."""

import numpy as np
import pytest
import torch

from training.evidence.train_decoder import (
    estimate_density_threshold,
    retrieve,
    run_episode,
    sample_label_set,
)


def test_retrieve_caps_topk_to_eligible_memory_without_infinities():
    zq = torch.tensor([[1.0, 0.0]])
    Z = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    allowed = torch.tensor([[False, True, True]])
    idx, weights, vals = retrieve(zq, Z, allowed, k=48, tau=0.1)
    assert idx.shape == weights.shape == vals.shape == (1, 2)
    assert set(idx[0].tolist()) == {1, 2}
    assert torch.isfinite(vals).all() and torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(1), torch.ones(1))


def test_retrieve_fails_when_a_query_has_no_eligible_memory():
    with pytest.raises(ValueError, match="no eligible"):
        retrieve(
            torch.eye(2),
            torch.eye(2),
            torch.tensor([[True, False], [False, False]]),
            k=1,
            tau=0.1,
        )


class _CaptureDecoder:
    def __init__(self):
        self.zev = None

    def __call__(self, *, zq, zev, ev_label_text, w_retr, cand_text, return_aux=False):
        self.zev = zev.detach().clone()
        logits = torch.zeros(len(zq), len(cand_text))
        if return_aux:
            aux = {
                "evidence": torch.zeros_like(logits),
                "pool_weights": w_retr,
                "delta": torch.zeros(*zev.shape[:2], cand_text.shape[-1]),
                "delta_norm": 0.0,
            }
            return logits, aux
        return logits


def test_episode_retrieval_is_restricted_to_explicit_training_memory():
    # Row 4 is the nearest possible match but represents held-out validation memory. Only rows 2/3
    # are in the training pool, and H removes rows 0/1 by label.
    Z = torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [0.3, 0.7], [0.2, 0.8], [1.0, 0.0],
    ])
    y = torch.arange(5)
    subj = torch.arange(5)
    qi = torch.tensor([0])
    H = torch.tensor([0, 1])
    text = torch.eye(5)
    memory_mask = torch.tensor([False, False, True, True, False])
    dec = _CaptureDecoder()
    run_episode(
        dec, Z, y, subj, qi, H, text, text, k=48, tau=0.1,
        memory_mask=memory_mask,
    )
    selected = {tuple(row.tolist()) for row in dec.zev[0]}
    assert selected == {tuple(Z[2].tolist()), tuple(Z[3].tolist())}


def test_variable_label_budget_spans_range_is_deterministic_and_reserves_memory():
    present = torch.arange(10)
    rng1 = np.random.default_rng(17)
    draws1 = [
        sample_label_set(present, 3, 6, rng1, reserve_labels=1).tolist()
        for _ in range(100)
    ]
    sizes = {len(draw) for draw in draws1}
    assert sizes == {3, 4, 5, 6}
    assert all(len(draw) <= len(present) - 1 for draw in draws1)

    rng2 = np.random.default_rng(17)
    draws2 = [
        sample_label_set(present, 3, 6, rng2, reserve_labels=1).tolist()
        for _ in range(100)
    ]
    assert draws1 == draws2


def test_density_threshold_calibration_is_finite_and_deterministic():
    gen = torch.Generator().manual_seed(4)
    Z = torch.nn.functional.normalize(torch.randn(30, 8, generator=gen), dim=-1)
    y = torch.arange(30) % 6
    subj = torch.arange(30)
    train_q = torch.arange(30)
    present = torch.unique(y)
    memory_mask = torch.ones(30, dtype=torch.bool)
    kwargs = dict(
        Z=Z, y=y, subj=subj, train_q=train_q, train_present=present,
        memory_mask=memory_mask, min_labels=2, max_labels=4, k=5,
        seed=9, n_episodes=2, queries_per_episode=5,
    )
    a = estimate_density_threshold(**kwargs)
    b = estimate_density_threshold(**kwargs)
    assert np.isfinite(a) and -1.0 <= a <= 1.0
    assert a == b
