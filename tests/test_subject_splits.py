"""Tests for the shared subject-split manifest (REMEDIATION_PLAN Phase 1.1 — H7/H8).

The manifest exists to guarantee two properties that the old per-model
`subject_disjoint_split(S, seed)` did NOT have:

  H8 — **models must agree on folds.** Previously harnet's gravity exclusions changed the subject
       universe before the shuffle, so 16.5% of shared subjects landed in different folds than
       HALO's, meaning the two models were selected and calibrated on different data.
  H7 — **every dataset must be represented in validation.** Shuffling the aggregate pool left
       HHAR, MHEALTH and PAMAP2 with zero val subjects.
"""

import numpy as np
import pytest

from eval.splits import _split_one_dataset, build_manifest, load_manifest, split_indices


def test_folds_are_identical_when_a_model_drops_a_dataset():
    """H8: excluding a stream must not move any OTHER model's subjects."""
    full = np.array([f"capture24:P{i:03d}" for i in range(1, 40)]
                    + [f"kuhar:{i}" for i in range(1, 30)])
    reduced = np.array([s for s in full if not s.startswith("kuhar:")])  # gravity-removed, excluded

    def fold_of(ids):
        tr, va, te = split_indices(ids)
        out = {}
        for idx, name in ((tr, "train"), (va, "val"), (te, "test")):
            for i in idx:
                out[ids[i]] = name
        return out

    a, b = fold_of(full), fold_of(reduced)
    shared = set(a) & set(b)
    assert shared, "test setup: expected overlapping subjects"
    disagree = {s for s in shared if a[s] != b[s]}
    assert not disagree, f"{len(disagree)} shared subjects changed fold when a dataset was dropped"


def test_every_sufficiently_large_dataset_has_val_and_test_subjects():
    """H7: stratification — no dataset with >=3 subjects may be absent from val or test."""
    m = load_manifest()
    for ds, counts in m["per_dataset"].items():
        total = sum(counts.values())
        if total >= 3:
            assert counts["val"] >= 1, f"{ds} has {total} subjects but ZERO val subjects"
            assert counts["test"] >= 1, f"{ds} has {total} subjects but ZERO test subjects"


def test_folds_are_subject_disjoint():
    m = load_manifest()
    by_fold = {"train": set(), "val": set(), "test": set()}
    for subj, fold in m["assignment"].items():
        by_fold[fold].add(subj)
    assert not (by_fold["train"] & by_fold["val"])
    assert not (by_fold["train"] & by_fold["test"])
    assert not (by_fold["val"] & by_fold["test"])


def test_split_is_deterministic():
    subs = [f"s{i}" for i in range(20)]
    assert _split_one_dataset(subs, 7) == _split_one_dataset(subs, 7)
    assert _split_one_dataset(subs, 7) != _split_one_dataset(subs, 8)


def test_tiny_cohort_goes_entirely_to_train():
    """A 1-2 subject cohort cannot be split 3 ways; it must not produce a 1-subject val fold
    (the NFI-FARED pathology: 1 val subject carrying 1 label)."""
    assert set(_split_one_dataset(["a", "b"], 0).values()) == {"train"}


def test_unknown_subjects_default_to_train_never_val_or_test():
    """An unseen subject must never silently land in val/test — that would leak."""
    tr, va, te = split_indices(np.array(["nonexistent_dataset:zzz"]))
    assert len(tr) == 1 and len(va) == 0 and len(te) == 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
