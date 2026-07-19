"""Unit + smoke tests for the model-agnostic ZS-XD scoring core.

The scoring maths is exercised on synthetic data (no model, no SBERT download
needed for the pure-numpy paths). A single smoke test loads the REAL motionsense
grid on disk to check the new-format loader end to end. ConSE tests that need
SBERT are skipped if `sentence_transformers` cannot load its model offline.
"""

import numpy as np
import pytest

from eval import data as eval_data
from eval import scoring


# =============================================================================
# Ground-truth filtering (offset-free, candidate-vocab restriction)
# =============================================================================

def test_filter_ground_truth_drops_out_of_vocab():
    gt = ["walking", "climbingup", "sitting", "running"]
    subj = ["a", "a", "b", "b"]
    kept_gt, kept_subj, keep_idx = scoring.filter_ground_truth(gt, subj, ["walking", "sitting"])
    assert kept_gt == ["walking", "sitting"]
    assert list(keep_idx) == [0, 2]
    assert list(kept_subj) == ["a", "b"]


# =============================================================================
# Subject-disjoint split
# =============================================================================

def test_subject_disjoint_split_never_shares_a_subject():
    # 20 subjects, 5 windows each; string ids like the real grids.
    subjects = np.array([f"s{i}" for i in range(20) for _ in range(5)])
    tr, va, te = scoring.subject_disjoint_split(subjects, seed=0)
    s_tr = set(subjects[tr]); s_va = set(subjects[va]); s_te = set(subjects[te])
    assert s_tr and s_va and s_te                      # every split non-empty
    assert s_tr.isdisjoint(s_va)
    assert s_tr.isdisjoint(s_te)
    assert s_va.isdisjoint(s_te)
    # every window accounted for exactly once
    assert len(tr) + len(va) + len(te) == len(subjects)
    assert s_tr | s_va | s_te == set(subjects)


def test_subject_disjoint_split_requires_three_subjects():
    subjects = np.array(["a", "a", "b", "b"])
    with pytest.raises(ValueError):
        scoring.subject_disjoint_split(subjects)


def test_subject_disjoint_split_gives_test_at_least_two_when_possible():
    subjects = np.repeat([f"s{i}" for i in range(15)], 3)
    tr, va, te = scoring.subject_disjoint_split(subjects, seed=1)
    assert len(set(subjects[te])) >= 2   # floored val -> remainder to test


# =============================================================================
# Balanced subsample
# =============================================================================

def test_balanced_subsample_is_roughly_balanced_and_sized():
    # class A: 100 windows, class B: 10 windows.
    names = ["A"] * 100 + ["B"] * 10
    idx = np.arange(len(names))
    picked, counts = scoring.balanced_subsample_indices(idx, names, rate=0.5, return_counts=True)
    assert len(picked) == sum(counts.values())
    # ~50% of 110 = 55 budget; B is scarce (<=10) so it is taken (near-)fully,
    # and the balanced fill keeps B from being swamped by A.
    assert counts["B"] >= 10 or counts["B"] >= counts["A"] // 2
    assert set(picked).issubset(set(idx))


# =============================================================================
# Metrics: macro-F1 on a known confusion
# =============================================================================

def test_macro_f1_on_known_confusion():
    # 2 classes, perfectly predicted -> macro-F1 = 100.
    gt = ["walk", "walk", "sit", "sit"]
    perfect = scoring.classification_metrics(gt, gt)
    assert perfect["f1_macro"] == pytest.approx(100.0)
    assert perfect["accuracy"] == pytest.approx(100.0)

    # Everything predicted "walk": walk P=2/4=.5 R=1 F1=2/3; sit F1=0.
    all_walk = ["walk"] * 4
    m = scoring.classification_metrics(gt, all_walk)
    # macro over {walk, sit}: (66.66.. + 0)/2
    assert m["f1_macro"] == pytest.approx(100.0 * (2 / 3) / 2, rel=1e-6)
    assert m["balanced_accuracy"] == pytest.approx(50.0)   # recall over GT classes
    assert m["n_gt_classes"] == 2


def test_macro_f1_charges_false_positive_into_unseen_candidate():
    # Predicting a class with zero GT windows must be penalized (union class set).
    gt = ["walk", "walk", "walk", "walk"]
    pred = ["walk", "walk", "walk", "run"]  # 'run' has no GT window -> FP
    m = scoring.classification_metrics(gt, pred)
    assert m["n_scored_classes"] == 2           # {walk, run}
    assert m["f1_macro"] < 100.0                 # not silently exempt


# =============================================================================
# Soft pooling of per-patch scores
# =============================================================================

def test_soft_pool_prefers_consistent_class():
    # 3 patches, 2 labels; label 0 consistently higher -> pooled argmax = 0.
    patch_sims = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    mask = np.array([True, True, True])
    scores = scoring.soft_pool_patch_scores(patch_sims, mask)
    assert int(np.argmax(scores)) == 0
    assert scores.shape == (2,)


def test_segment_predictions_respects_mask():
    # Masked-out patch would flip the vote; soft pooling must ignore it.
    patch_sims = np.array([[[0.9, 0.1], [0.0, 5.0]]])  # (1 seg, 2 patches, 2 labels)
    masks = np.array([[True, False]])
    preds = scoring.segment_predictions(patch_sims, masks, ["a", "b"])
    assert preds == ["a"]


# =============================================================================
# Subject-stratified bootstrap CI
# =============================================================================

def test_bootstrap_ci_returns_lo_hi_bracketing_point():
    rng = np.random.RandomState(0)
    subjects = np.repeat([f"s{i}" for i in range(8)], 20)
    gt = rng.choice(["a", "b", "c"], size=len(subjects)).tolist()
    # imperfect predictions so the metric has spread
    pred = [g if rng.rand() < 0.7 else rng.choice(["a", "b", "c"]) for g in gt]
    point = scoring.classification_metrics(gt, pred)["f1_macro"]
    ci = scoring.subject_bootstrap_ci(gt, pred, subjects, B=200, seed=0)
    assert ci["ci_degenerate"] is False
    lo, hi = ci["f1_macro_ci_lo"], ci["f1_macro_ci_hi"]
    assert lo <= hi
    assert lo - 5 <= point <= hi + 5   # point estimate near the interval


def test_bootstrap_ci_degenerate_with_one_subject():
    gt = ["a", "b", "a", "b"]
    pred = ["a", "b", "b", "a"]
    subjects = np.array(["only"] * 4)
    ci = scoring.subject_bootstrap_ci(gt, pred, subjects)
    assert ci["ci_degenerate"] is True
    assert np.isnan(ci["f1_macro_ci_lo"]) and np.isnan(ci["f1_macro_ci_hi"])


def test_groupkfold_ci_brackets_point_deterministic_and_degenerate():
    """#7: leave-one-subject-out jackknife CI for small cohorts — brackets the point, is a valid
    deterministic interval, and stays flagged-degenerate for a single-subject cohort."""
    rng = np.random.RandomState(0)
    subjects = np.repeat([f"s{i}" for i in range(10)], 30)     # small cohort
    gt = rng.choice(["a", "b", "c"], size=len(subjects)).tolist()
    pred = [g if rng.rand() < 0.7 else rng.choice(["a", "b", "c"]) for g in gt]
    point = scoring.classification_metrics(gt, pred)["f1_macro"]
    ci = scoring.subject_groupkfold_ci(gt, pred, subjects)
    assert ci["ci_degenerate"] is False and ci["ci_method"] == "subject_groupkfold_loso"
    assert ci["n_folds"] == 10
    lo, hi = ci["f1_macro_ci_lo"], ci["f1_macro_ci_hi"]
    assert 0.0 <= lo <= point <= hi <= 100.0                    # brackets the pooled point
    assert ci == scoring.subject_groupkfold_ci(gt, pred, subjects)   # deterministic
    deg = scoring.subject_groupkfold_ci(["a"] * 6, ["a"] * 6, np.array(["only"] * 6))
    assert deg["ci_degenerate"] is True and np.isnan(deg["f1_macro_ci_lo"])


def test_fit_temperature_calibrates_overconfident_head():
    """#82: temperature scaling on a held-out set returns T>1 for over-confident logits, lowers NLL,
    and preserves argmax; degenerate input returns T=1."""
    import torch
    rng = np.random.RandomState(0)
    n, C = 300, 4
    y = rng.randint(0, C, size=n)
    logits = rng.normal(0, 0.5, size=(n, C))
    for i in range(n):                                   # 70% accurate but big margins -> over-confident
        logits[i, y[i] if rng.rand() < 0.7 else rng.randint(C)] += 8.0
    T = scoring.fit_temperature(logits, y)
    lg = torch.tensor(logits, dtype=torch.float32); yt = torch.tensor(y)
    nll1 = torch.nn.functional.cross_entropy(lg, yt).item()
    nllT = torch.nn.functional.cross_entropy(lg / T, yt).item()
    assert T > 1.0 and nllT <= nll1 + 1e-6
    assert (logits.argmax(1) == (logits / T).argmax(1)).all()   # argmax unchanged
    assert scoring.fit_temperature(np.zeros((0, C)), np.array([])) == 1.0   # degenerate -> 1.0


# =============================================================================
# ConSE bridge
# =============================================================================

def test_conse_embeddings_convex_combination_shape_and_norm():
    # 3 training labels with orthonormal-ish embeddings; probs pick label 0.
    train_embs = np.eye(3)
    probs = np.array([[0.8, 0.15, 0.05], [0.1, 0.1, 0.8]])
    v = scoring.conse_embeddings(probs, train_embs, top_T=3)
    assert v.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, rtol=1e-6)
    assert int(np.argmax(v[0])) == 0   # dominant training class dominates
    assert int(np.argmax(v[1])) == 2


def test_conse_embeddings_rejects_mismatched_vocab():
    with pytest.raises(ValueError):
        scoring.conse_embeddings(np.ones((2, 4)), np.eye(3))


def _sbert_available() -> bool:
    try:
        scoring.get_sbert_encoder()(["walking"])
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _sbert_available(), reason="SBERT model unavailable offline")
def test_conse_predict_maps_into_target_vocab():
    train_vocab = ["running", "sitting down", "standing still"]
    target = ["jogging", "sitting", "standing"]
    # Confident on 'running' -> should bridge to the semantically nearest target.
    probs = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])
    preds, info = scoring.conse_predict(probs, train_vocab, target)
    assert len(preds) == 2
    assert all(p in target for p in preds)
    assert set(info["predicted_classes"]).issubset(set(target))
    assert 0.0 <= info["reachability_lb"] <= 1.0


# =============================================================================
# Real-data smoke test: new-format grid loader
# =============================================================================

def test_load_eval_stream_motionsense_real_grid():
    s = eval_data.load_eval_stream("motionsense", "phone_front_pocket")
    n = s.n_windows
    assert s.windows.ndim == 3 and s.windows.shape[2] == 6       # (N, T, 6)
    assert 200 <= s.windows.shape[1] <= 400                       # ~6 s window
    assert len(s.gt) == n and len(s.subjects) == n               # 1:1 with windows
    assert s.mask.shape == (6,)
    assert s.channels[:3] == ["acc_x", "acc_y", "acc_z"]
    # motionsense candidate vocabulary (pre-registered eval_labels.json)
    assert s.eval_labels == [
        "jogging", "sitting", "standing", "walking",
        "walking_downstairs", "walking_upstairs",
    ]
    # every window's GT is within the candidate vocab for motionsense
    assert set(s.gt).issubset(set(s.eval_labels))


def test_load_gt_aligns_with_stream(monkeypatch=None):
    from baselines import base
    eval_labels, gt, subjects, keep_idx = base.load_gt("motionsense", "phone_front_pocket")
    s = eval_data.load_eval_stream("motionsense", "phone_front_pocket")
    assert eval_labels == s.eval_labels
    assert len(gt) == len(subjects) == len(keep_idx)
    assert set(gt).issubset(set(eval_labels))
    # keep_idx indexes back into the full window set
    assert keep_idx.max() < s.n_windows
