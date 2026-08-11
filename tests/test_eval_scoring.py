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


def test_filter_ground_truth_aligns_canonical_grid_labels_to_native_candidates():
    gt = ["walking_upstairs", "walking_downstairs", "walking"]
    candidates = ["climbingup", "climbingdown", "walking"]
    kept_gt, _, keep_idx = scoring.filter_ground_truth(gt, ["a", "a", "a"], candidates)
    assert kept_gt == candidates
    assert list(keep_idx) == [0, 1, 2]


def test_ground_truth_alignment_rejects_ambiguous_candidate_synonyms():
    with pytest.raises(ValueError, match="collapse"):
        scoring.align_ground_truth_labels(
            ["walking_upstairs"], ["ascending_stairs", "climbingup"]
        )


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


def test_paired_subject_bootstrap_difference_uses_matched_subjects():
    gt = ["a", "b", "a", "b", "a", "b", "a", "b"]
    learned = ["a", "b", "a", "b", "a", "b", "a", "a"]
    control = ["a", "a", "a", "a", "b", "b", "b", "b"]
    subjects = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    result = scoring.paired_subject_bootstrap_difference(
        gt, learned, control, subjects, B=200, seed=4
    )
    assert result["f1_macro_difference"] > 0
    assert result["ci_method"] == "paired_subject_bootstrap"
    assert result["n_subjects"] == 4


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


# =============================================================================
# Enrollment leakage unit: label blocks vs continuous recordings
# =============================================================================

def test_execution_ids_group_label_blocks_onto_their_recording():
    """A converter emits one session per contiguous label block, so several sessions come out of
    ONE continuous capture. Grouping them back is what stops a k-curve from measuring adjacency.

    MM-Fit is the sharp case: its three sets of an exercise are cut from a single per-workout
    recording, so the block ids report three "executions" per (workout, exercise) where there is
    one capture. Measured 2026-08-11, before the fix: median 3, max 4.
    """
    s = eval_data.load_eval_stream("mmfit", "left_wrist", alignment="native")
    assert s.execution_granularity == "recording"
    assert s.block_ids is not None and len(s.block_ids) == s.n_windows

    def per_subject_label(ids):
        groups = {}
        for subject, label, value in zip(s.subjects, s.gt, ids):
            groups.setdefault((str(subject), str(label)), set()).add(value)
        return sorted(len(v) for v in groups.values())

    blocks, executions = per_subject_label(s.block_ids), per_subject_label(s.execution_ids)
    assert max(blocks) > 1, "expected mmfit to carry several label blocks per (workout, exercise)"
    assert set(executions) == {1}, "one workout is one capture, so k must collapse to 1"
    # The grouping is a coarsening, never a re-partition: blocks nest inside recordings.
    nesting = {}
    for block, execution in zip(s.block_ids, s.execution_ids):
        assert nesting.setdefault(block, execution) == execution


def test_execution_ids_are_untouched_without_a_recording_map():
    """monipar declares no recordings.json: one session already IS one weekly visit, so its seven
    executions per (subject, exercise) are real and must survive unchanged."""
    s = eval_data.load_eval_stream("monipar", "watch_wrist", alignment="native")
    assert s.execution_granularity == "block"
    assert list(s.execution_ids) == list(s.block_ids)


# =============================================================================
# Quality screen — training and evaluation must refuse the same windows
# =============================================================================

def test_quality_screen_drops_the_windows_training_refuses():
    """`scan_duplicates`/`scan_implausible` were applied by the trainer and the memory-bank build
    but not by the evaluator, so a window too corrupt to train on was still scored. motionsense is
    a Phase-B development stream and carries 34 flagged byte-identical duplicates."""
    from data.scripts.scan_duplicates import load as load_duplicates

    flagged = load_duplicates("native", require=True).get("motionsense/phone_front_pocket", set())
    assert flagged, "expected motionsense to carry cached duplicate windows"

    screened = eval_data.load_eval_stream("motionsense", "phone_front_pocket", alignment="native")
    raw = eval_data.load_eval_stream("motionsense", "phone_front_pocket", alignment="native",
                                     apply_quality_screen=False)
    assert screened.quality_screen == "applied"
    assert raw.quality_screen == "not requested" and raw.n_quality_excluded == 0
    assert screened.n_windows == raw.n_windows - screened.n_quality_excluded
    assert screened.n_quality_excluded >= len(flagged)
    # Every parallel array is filtered together, not just the signal.
    for field in ("gt", "subjects", "event_ids", "execution_ids", "block_ids"):
        assert len(getattr(screened, field)) == screened.n_windows


def test_quality_screen_reports_when_the_cache_cannot_cover_the_alignment():
    """The caches are built one alignment at a time. A silent empty screen is indistinguishable
    from a clean stream, so an uncovered alignment has to say so rather than return nothing."""
    s = eval_data.load_eval_stream("motionsense", "phone_front_pocket",
                                   alignment="non_harmonised")
    assert s.quality_screen.startswith("unavailable:")
    assert s.n_quality_excluded == 0


# =============================================================================
# Per-stream candidate vocabulary
# =============================================================================

def test_candidate_vocabulary_follows_the_stream_protocol():
    """PHYTMO records six upper-limb exercises on the arm units and fourteen lower-limb ones on the
    shin/thigh units. Scoring an arm stream against the dataset-wide 20 charges it for labels its
    acquisition configuration never records."""
    dataset_wide = eval_data.load_eval_labels("phytmo")
    arm = eval_data.load_eval_labels("phytmo", "left_arm")
    shin = eval_data.load_eval_labels("phytmo", "left_shin")
    assert len(dataset_wide) == 20 and len(arm) == 6 and len(shin) == 14
    assert set(arm).isdisjoint(shin) and set(arm) | set(shin) == set(dataset_wide)
    # The loader hands the stream its own vocabulary, and the ground truth stays inside it.
    stream = eval_data.load_eval_stream("phytmo", "left_arm", alignment="native")
    assert stream.eval_labels == arm
    assert set(stream.gt).issubset(set(arm))


def test_candidate_vocabulary_defaults_to_the_dataset_when_undeclared():
    """kneepad deliberately declares no per-stream subset: its reduced coverage is a consequence of
    the 6 s window, not of the protocol, so the window length must not redefine the task."""
    assert eval_data.load_eval_labels("kneepad", "left_hamstrings") == \
        eval_data.load_eval_labels("kneepad")
