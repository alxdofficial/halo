"""Unit tests for the tokenizer-quality metric suite — synthetic data with known answers."""

import numpy as np

from eval.tokenizer_metrics import (
    alignment, cross_config_retrieval, decodability, effective_rank, knn_purity,
    linear_probe_ba, uniformity,
)


def _clustered(n_per=100, n_classes=4, d=16, spread=0.05, seed=0):
    """Tight, well-separated Gaussian clusters — a 'good' representation."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_classes, d) * 3.0
    Z, y = [], []
    for c in range(n_classes):
        Z.append(centers[c] + rng.randn(n_per, d) * spread)
        y.append(np.full(n_per, c))
    return np.concatenate(Z), np.concatenate(y)


def test_knn_purity_high_for_separated_low_for_random():
    Z, y = _clustered(spread=0.02)
    assert knn_purity(Z, y) > 0.95
    rng = np.random.RandomState(1)
    assert knn_purity(rng.randn(*Z.shape), y) < 0.5      # random features -> chance-ish


def test_effective_rank_collapse_vs_full():
    rng = np.random.RandomState(0)
    full = rng.randn(500, 16)
    collapsed = np.outer(rng.randn(500), rng.randn(16))   # rank-1
    assert effective_rank(full) > 8
    assert effective_rank(collapsed) < 1.5


def test_alignment_uniformity_ordering():
    tight, y = _clustered(spread=0.02)
    loose, _ = _clustered(spread=0.5)
    assert alignment(tight, y) < alignment(loose, y)      # tighter classes -> lower alignment


def test_decodability_detects_leaked_config():
    """If a config axis is linearly encoded, decodability is high; if orthogonal noise, ~chance."""
    rng = np.random.RandomState(0)
    d = 16
    cfg = rng.randint(0, 3, 600)
    # config baked into the first dim
    Z_leak = rng.randn(600, d); Z_leak[:, 0] += cfg * 5.0
    assert decodability(Z_leak, cfg) > 0.9
    Z_clean = rng.randn(600, d)                            # config independent of features
    assert decodability(Z_clean, cfg) < 0.55


def test_cross_config_retrieval_same_activity_across_config():
    """Same 'activity' (label) present under two 'configs'; a config-invariant rep retrieves across."""
    rng = np.random.RandomState(0)
    d = 16
    centers = rng.randn(3, d) * 3.0                        # 3 activities
    Z, y, cfg = [], [], []
    for c in range(2):                                    # 2 configs, small config-specific shift
        shift = rng.randn(d) * 0.05
        for a in range(3):
            Z.append(centers[a] + shift + rng.randn(40, d) * 0.05)
            y.append(np.full(40, a)); cfg.append(np.full(40, c))
    Z, y, cfg = np.concatenate(Z), np.concatenate(y), np.concatenate(cfg)
    assert cross_config_retrieval(Z, y, cfg) > 0.9        # activity dominates the small config shift


def test_linear_probe_ba_separable():
    Z, y = _clustered(spread=0.1)
    from eval.tokenizer_metrics import _stratified_split
    assert linear_probe_ba(*_stratified_split(Z, y)) > 0.95


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))


def test_transfer_probe_trains_and_tests_on_DIFFERENT_sets():
    """Regression for the mislabelled-transfer bug: transfer must NOT be an in-distribution split.
    Train and test are disjoint sets; a probe that ignores the test set would score differently."""
    from eval.tokenizer_metrics import transfer_probe_ba
    rng = np.random.RandomState(0); d = 16
    centers = rng.randn(4, d) * 3.0
    def make(shift):
        Z, y = [], []
        for a in range(4):
            Z.append(centers[a] + shift + rng.randn(60, d) * 0.1); y.append(np.full(60, a))
        return np.concatenate(Z), np.concatenate(y)
    trZ, trY = make(0.0)
    teZ, teY = make(rng.randn(d) * 0.1)          # same label structure, small domain shift -> transfers
    ba, n = transfer_probe_ba(trZ, trY, teZ, teY)
    assert n == 4 and ba > 0.9, f"transfer should work across a small shift (ba={ba}, n={n})"
    # if the test domain has a DIFFERENT label->feature mapping, transfer must drop
    teZ_scrambled = teZ.copy()
    teY_scrambled = (teY + 1) % 4                 # labels permuted vs features
    ba2, _ = transfer_probe_ba(trZ, trY, teZ_scrambled, teY_scrambled)
    assert ba2 < ba, "scrambled-label test domain must transfer worse"


def test_knn_purity_macro_downweights_common_classes():
    """Macro purity must differ from micro when class sizes are imbalanced."""
    rng = np.random.RandomState(0); d = 16
    # one huge tight class + two tiny noisy ones
    Z = np.concatenate([rng.randn(400, d) * 0.02 + 5,
                        rng.randn(20, d) * 2.0, rng.randn(20, d) * 2.0 + 1])
    y = np.concatenate([np.zeros(400), np.ones(20), np.full(20, 2)])
    micro = knn_purity(Z, y, macro=False); macro = knn_purity(Z, y, macro=True)
    assert micro > macro, f"micro {micro} should exceed macro {macro} when a clean class dominates"
