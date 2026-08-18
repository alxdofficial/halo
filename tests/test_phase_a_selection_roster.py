"""Phase-A checkpoint-selection roster: subsampling, leakage guard, posture canary.

These cover the protocol properties the 2026-08-17 regression exposed. That run selected its "best"
checkpoint on an internal probe while held-out transfer decayed, and its damage concentrated in
gravity-defined static postures that an aggregate score barely moved on.
"""

from __future__ import annotations

import numpy as np
import pytest

from training.tokenizer.eval_transfer import (
    PHASE_A_SELECTION_DATASETS,
    PHASE_A_SELECTION_STREAMS,
    POSTURE_LABELS,
    SELECTION_MAX_WINDOWS_PER_STREAM,
    _selection_subsample,
    assert_selection_roster_is_untrained,
)


def test_selection_roster_is_disjoint_from_training_corpus():
    """The roster must never intersect either named Phase-A training recipe."""
    from data.scripts.curate.deployment_policy import (
        EXPANDED_PHASE_A_TRAIN_DATASETS,
        CORPUS_MATCHED_TRAIN_DATASETS,
    )

    for corpus in (EXPANDED_PHASE_A_TRAIN_DATASETS, CORPUS_MATCHED_TRAIN_DATASETS):
        assert not set(PHASE_A_SELECTION_DATASETS) & set(corpus)


def test_selection_roster_is_disjoint_from_the_phase_b_test_roster():
    """Selecting on a Phase-B test dataset would silently burn the readout."""
    from data.scripts.curate.deployment_policy import (
        ESTABLISHED_EVAL_DATASETS,
        NEW_HELDOUT_EVAL_DATASETS,
    )

    test_roster = set(ESTABLISHED_EVAL_DATASETS + NEW_HELDOUT_EVAL_DATASETS) - {
        "motionsense", "realworld", "shoaib",      # the declared development cohorts
    }
    assert not set(PHASE_A_SELECTION_DATASETS) & test_roster


def test_guard_rejects_a_trained_selection_source():
    overlap = [PHASE_A_SELECTION_DATASETS[0], "capture24"]
    with pytest.raises(ValueError, match="must be held out"):
        assert_selection_roster_is_untrained(overlap)
    assert_selection_roster_is_untrained(["capture24", "wisdm"])       # no overlap -> silent


def test_roster_covers_more_than_phone_placements():
    """A phone-only roster is close to blind to the wrist damage it exists to catch."""
    placements = {stream for _, stream, _ in PHASE_A_SELECTION_STREAMS}
    assert any("wrist" in placement for placement in placements)


def test_subsample_returns_every_row_when_under_cap():
    labels = np.array(["a", "b"] * 10)
    subjects = np.array([1, 2] * 10)
    take = _selection_subsample(labels, subjects, 100)
    assert take.tolist() == list(range(20))


def test_subsample_is_deterministic_sorted_and_capped():
    rng = np.random.default_rng(0)
    labels = rng.choice(["a", "b", "c"], size=5_000)
    subjects = rng.integers(0, 20, size=5_000)
    first = _selection_subsample(labels, subjects, 500)
    second = _selection_subsample(labels, subjects, 500)
    assert len(first) == 500
    assert np.array_equal(first, second)                    # fixed seed -> comparable across steps
    assert np.array_equal(first, np.sort(first))            # sorted -> safe as a grid index
    assert len(set(first.tolist())) == len(first)           # no duplicated windows


def test_subsample_keeps_rare_classes_scoreable():
    """Round-robin, not proportional: a rare posture class must survive the cap.

    A proportional draw would leave `lying` with ~5 of 500 rows here, which cannot support a
    subject-disjoint canary. The canary is the point, so the rare stratum is protected.
    """
    labels = np.array(["walking"] * 4_900 + ["lying"] * 100)
    subjects = np.concatenate([np.arange(4_900) % 10, np.arange(100) % 10])
    take = _selection_subsample(labels, subjects, 500)
    kept = labels[take]
    assert (kept == "lying").sum() >= 100                   # every rare row survives
    assert len(set(subjects[take][kept == "lying"].tolist())) >= 5   # across several subjects


def test_posture_labels_are_static_and_gravity_defined():
    """Guard the canary's meaning: a periodic class here would dilute what it measures."""
    assert set(POSTURE_LABELS) == {"lying", "sitting", "standing"}


def test_selection_cap_is_small_enough_for_in_run_use():
    assert SELECTION_MAX_WINDOWS_PER_STREAM <= 10_000
