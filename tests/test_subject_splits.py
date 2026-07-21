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
    """H8: excluding a stream must not move any OTHER model's subjects.

    Uses REAL manifest ids. The previous version of this test used fabricated ones
    (``kuhar:1``..``kuhar:29``; the real ids are ``kuhar:s1001``...), so 29 of its 68 fixture
    entries missed the manifest entirely and silently exercised the train-fallback instead of
    the property under test.
    """
    m = load_manifest()
    cap = [k for k in m["assignment"] if k.startswith("capture24:")][:40]
    kuh = [k for k in m["assignment"] if k.startswith("kuhar:")][:30]
    assert cap and kuh, "test setup: expected capture24 and kuhar subjects in the manifest"
    full = np.array(cap + kuh)
    reduced = np.array(cap)                      # a model that cannot consume kuhar

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


def test_every_adapter_subject_id_hits_the_manifest():
    """THE property that actually matters, and that had no test.

    Each adapter builds its own ``f"{dataset}:{subject}"`` array. If that string does not match a
    manifest key, ``split_indices`` used to route it silently to TRAIN — so a model's val/test
    folds would be drawn from a different population than everyone else's, with no error. That is
    exactly how harnet's ``hapt`` subjects (absent from the manifest) got fabricated folds.
    """
    from data.scripts.eda.grid_io import discover_grids
    from training.tokenizer.pretrain_data import TRAIN_DATASETS
    from eval.splits import EXTRA_MANIFEST_DATASETS

    a = load_manifest()["assignment"]
    wanted = set(TRAIN_DATASETS) | set(EXTRA_MANIFEST_DATASETS)
    missing = set()
    for ref in discover_grids("native"):
        if ref.dataset not in wanted:
            continue
        for s in ref.subjects:
            if f"{ref.dataset}:{s}" not in a:
                missing.add(f"{ref.dataset}:{s}")
    assert not missing, (f"{len(missing)} constructed subject ids are absent from the manifest, "
                         f"e.g. {sorted(missing)[:5]}")


def test_aliased_cohorts_share_a_fold():
    """hapt and uci_har are the same 30 people; both must land on the same side of the split.

    harnet's DEFAULT corpus contains hapt while HALO's does not, so before this was enforced,
    hapt subjects sat in harnet's head-fit TRAIN fold while their uci_har twins were its
    epoch-selection and temperature-calibration folds.
    """
    from eval.splits import ALIASED_COHORTS, _canonical_key

    a = load_manifest()["assignment"]
    for alias_ds in ALIASED_COHORTS:
        members = [k for k in a if k.startswith(f"{alias_ds}:")]
        assert members, f"{alias_ds} is declared aliased but absent from the manifest"
        for key in members:
            canon = _canonical_key(alias_ds, key.split(":", 1)[1])
            assert canon in a, f"{key} aliases {canon}, which is not in the manifest"
            assert a[key] == a[canon], (f"{key} is in fold {a[key]} but its twin {canon} is in "
                                        f"{a[canon]} — the same person is on both sides")


def test_unknown_subject_raises_instead_of_silently_becoming_train():
    """The silent-fallback bug: a miss used to degrade to an empty val fold with NO error."""
    with pytest.raises(KeyError, match="absent from the split manifest"):
        split_indices(np.array(["no_such_dataset:zzz"]))


def test_empty_val_or_test_fold_raises():
    """An all-train id set must raise: nan val-metric -> last epoch kept -> calibration on nothing."""
    a = load_manifest()["assignment"]
    train_only = np.array([k for k, v in a.items() if v == "train"][:20])
    with pytest.raises(ValueError, match="empty fold"):
        split_indices(train_only)


def test_every_sufficiently_large_dataset_has_val_and_test_subjects():
    """H7: stratification — no dataset with >=3 subjects may be absent from val or test."""
    m = load_manifest()
    # Recount from `assignment`, NOT from the `per_dataset` summary — asserting a summary against
    # itself passes even when the summary and the assignment have diverged.
    counted: dict = {}
    for subj, fold in m["assignment"].items():
        ds = subj.split(":", 1)[0]
        counted.setdefault(ds, {"train": 0, "val": 0, "test": 0})[fold] += 1
    for ds, counts in counted.items():
        total = sum(counts.values())
        if total >= 3:
            assert counts["val"] >= 1, f"{ds} has {total} subjects but ZERO val subjects"
            assert counts["test"] >= 1, f"{ds} has {total} subjects but ZERO test subjects"
        assert counts == m["per_dataset"][ds], f"{ds}: summary {m['per_dataset'][ds]} != {counts}"


def test_folds_are_subject_disjoint_at_the_PERSON_level():
    """Disjointness over dict keys is trivially true; the real risk is one PERSON under two keys.

    Checks the canonical identity (after alias resolution), which is where a genuine leak lives.
    """
    from eval.splits import _canonical_key

    m = load_manifest()
    by_fold: dict = {"train": set(), "val": set(), "test": set()}
    for subj, fold in m["assignment"].items():
        ds, s = subj.split(":", 1)
        by_fold[fold].add(_canonical_key(ds, s))
    assert not (by_fold["train"] & by_fold["val"]), by_fold["train"] & by_fold["val"]
    assert not (by_fold["train"] & by_fold["test"]), by_fold["train"] & by_fold["test"]
    assert not (by_fold["val"] & by_fold["test"]), by_fold["val"] & by_fold["test"]


def test_split_is_deterministic():
    subs = [f"s{i}" for i in range(20)]
    assert _split_one_dataset(subs, 7) == _split_one_dataset(subs, 7)
    assert _split_one_dataset(subs, 7) != _split_one_dataset(subs, 8)


def test_tiny_cohort_goes_entirely_to_train():
    """A 1-2 subject cohort cannot be split 3 ways; it must not produce a 1-subject val fold
    (the NFI-FARED pathology: 1 val subject carrying 1 label)."""
    assert set(_split_one_dataset(["a", "b"], 0).values()) == {"train"}


def test_allow_unknown_routes_to_train_never_val_or_test():
    """With allow_unknown=True the lenient path is still safe: unknowns go to TRAIN only."""
    a = load_manifest()["assignment"]
    known_val = next(k for k, v in a.items() if v == "val")
    known_test = next(k for k, v in a.items() if v == "test")
    ids = np.array(["nonexistent_dataset:zzz", known_val, known_test])
    tr, va, te = split_indices(ids, allow_unknown=True)
    assert list(tr) == [0] and list(va) == [1] and list(te) == [2]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
