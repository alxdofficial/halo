"""`ActiveSupportUnits` must reproduce the per-(label, subject) `torch.unique` scans it replaced.

The tables removed one GPU synchronization per (candidate label, subject) pair from the Phase-B
episode sampler. That is only a legitimate optimisation if it is *exactly* equivalent, including the
order in which the shared RNG is consumed — a single extra or reordered draw silently changes every
episode that follows. `_reference_prepare` below is the pre-optimisation implementation, kept
verbatim as the oracle.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from training.evidence.train_patch_decoder import (
    ActiveSupportUnits,
    active_support_units,
    prepare_support_feasible_query_pool,
    support_feasible_labels,
)


def _reference_prepare(pool, index_rows, bank, labels, *, support_count, episode_type, rng):
    """The original nested-loop implementation, unchanged, as the equivalence oracle."""
    if support_count == 0:
        return labels, pool
    device = pool.device
    rows = index_rows.detach().cpu().long()
    patch = bank["patch"]
    active_y = torch.as_tensor(patch["y"])[rows].long().to(device)
    active_subj = torch.as_tensor(patch["subj"])[rows].long().to(device)
    active_window = torch.as_tensor(patch["window"])[rows].long().to(device)
    active_event = torch.as_tensor(patch["event"])[rows].long().to(device)
    active_verified = torch.as_tensor(patch["event_verified"])[rows].bool().to(device)
    unit_offset = int(torch.as_tensor(patch["window"]).max()) + 1
    active_unit = torch.where(active_verified, active_event + unit_offset, active_window)

    window_y = torch.as_tensor(bank["y"], device=device, dtype=torch.long)
    window_subj = torch.as_tensor(bank["subj"], device=device, dtype=torch.long)
    window_event = torch.as_tensor(bank["event"], device=device, dtype=torch.long)
    window_verified = torch.as_tensor(bank["event_verified"], device=device, dtype=torch.bool)
    pool_unit = torch.where(window_verified[pool], window_event[pool] + unit_offset, pool)

    feasible_labels, query_parts = [], []
    for label in labels.tolist():
        label_pool_mask = window_y[pool].eq(label)
        label_pool = pool[label_pool_mask]
        if not len(label_pool):
            continue
        active_label = active_y.eq(label)
        if episode_type == "cross_subject_few_support":
            viable_subjects = []
            for subject in torch.unique(window_subj[label_pool]).tolist():
                available = active_label & active_subj.ne(subject)
                if torch.unique(active_unit[available]).numel() >= support_count:
                    viable_subjects.append(int(subject))
            if not viable_subjects:
                continue
            query_subject = int(rng.choice(np.asarray(viable_subjects)))
            query_part = label_pool[window_subj[label_pool].eq(query_subject)]
        elif episode_type == "same_subject_enrollment":
            viable = []
            label_pool_units = pool_unit[label_pool_mask]
            for subject in torch.unique(window_subj[label_pool]).tolist():
                subject_mask = window_subj[label_pool].eq(subject)
                subject_pool = label_pool[subject_mask]
                available = active_label & active_subj.eq(subject)
                units = torch.unique(active_unit[available])
                if len(units) < support_count:
                    continue
                reserved = torch.as_tensor(
                    rng.choice(units.detach().cpu().numpy(), size=support_count, replace=False),
                    device=device, dtype=torch.long,
                )
                query_part = subject_pool[
                    ~torch.isin(label_pool_units[subject_mask], reserved)
                ]
                if len(query_part):
                    viable.append((int(subject), query_part))
            if not viable:
                continue
            _, query_part = viable[int(rng.integers(len(viable)))]
        else:
            units = torch.unique(active_unit[active_label])
            if len(units) < support_count:
                continue
            reserved = torch.as_tensor(
                rng.choice(units.detach().cpu().numpy(), size=support_count, replace=False),
                device=device, dtype=torch.long,
            )
            query_part = label_pool[~torch.isin(pool_unit[label_pool_mask], reserved)]
        if len(query_part):
            feasible_labels.append(label)
            query_parts.append(query_part)

    if not feasible_labels:
        return labels[:0], pool[:0]
    return (
        torch.tensor(feasible_labels, device=device, dtype=torch.long),
        torch.cat(query_parts),
    )


def _random_bank(seed, *, n_windows=140, n_labels=6, n_subjects=7):
    """A bank with the awkward structure the tables have to survive.

    Deliberately includes unverified windows (unit == window id), verified windows sharing an event
    across several windows, and events shared *between subjects* — the last is what separates a
    correct "distinct units excluding this subject" count from the naive per-subject subtraction.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_labels, n_windows)
    subj = rng.integers(0, n_subjects, n_windows)
    event = rng.integers(0, max(2, n_windows // 4), n_windows)
    verified = rng.random(n_windows) < 0.6
    return {
        "y": torch.as_tensor(y, dtype=torch.long),
        "subj": torch.as_tensor(subj, dtype=torch.long),
        "event": torch.as_tensor(event, dtype=torch.long),
        "event_verified": torch.as_tensor(verified, dtype=torch.bool),
        "patch": {
            "y": torch.as_tensor(y, dtype=torch.long),
            "subj": torch.as_tensor(subj, dtype=torch.long),
            "window": torch.arange(n_windows, dtype=torch.long),
            "event": torch.as_tensor(event, dtype=torch.long),
            "event_verified": torch.as_tensor(verified, dtype=torch.bool),
        },
    }


@pytest.mark.parametrize("episode_type", [
    "ordinary_few_support", "cross_subject_few_support",
    "same_subject_enrollment", "semantic_zero_support",
])
@pytest.mark.parametrize("support_count", [1, 2, 4, 8])
def test_optimised_sampler_matches_the_original_scan_exactly(episode_type, support_count):
    for seed in range(12):
        bank = _random_bank(seed)
        n = len(bank["y"])
        pool = torch.arange(n)
        index_rows = torch.arange(n)
        labels = torch.arange(6)

        want_labels, want_pool = _reference_prepare(
            pool, index_rows, bank, labels, support_count=support_count,
            episode_type=episode_type, rng=np.random.default_rng(1000 + seed),
        )
        got_labels, got_pool = prepare_support_feasible_query_pool(
            pool, index_rows, bank, labels, support_count=support_count,
            episode_type=episode_type, rng=np.random.default_rng(1000 + seed),
        )
        assert got_labels.tolist() == want_labels.tolist(), (episode_type, support_count, seed)
        assert got_pool.tolist() == want_pool.tolist(), (episode_type, support_count, seed)


def test_feasibility_predicate_accepts_disjoint_validation_query_units():
    """Held-out query units need not be members of the active support index."""
    bank = {
        "y": torch.tensor([0, 0, 0]),
        "subj": torch.tensor([0, 0, 1]),
        "event": torch.tensor([0, 1, 2]),
        "event_verified": torch.zeros(3, dtype=torch.bool),
        "patch": {
            "y": torch.tensor([0, 0]),
            "subj": torch.tensor([0, 0]),
            "window": torch.tensor([0, 1]),
            "event": torch.tensor([0, 1]),
            "event_verified": torch.zeros(2, dtype=torch.bool),
        },
    }
    got = support_feasible_labels(
        torch.tensor([2]), torch.tensor([0, 1]), bank, torch.tensor([0]),
        support_count=2, episode_type="ordinary_few_support",
    )
    assert got.tolist() == [0]


def test_tables_reproduce_each_torch_unique_they_replaced():
    """Check the three lookups directly, not just their effect on the sampler."""
    bank = _random_bank(7)
    rows = torch.arange(len(bank["y"]))
    patch = bank["patch"]
    active_y = torch.as_tensor(patch["y"])[rows].long()
    active_subj = torch.as_tensor(patch["subj"])[rows].long()
    offset = int(torch.as_tensor(patch["window"]).max()) + 1
    active_unit = torch.where(
        torch.as_tensor(patch["event_verified"])[rows].bool(),
        torch.as_tensor(patch["event"])[rows].long() + offset,
        torch.as_tensor(patch["window"])[rows].long(),
    )
    tables = ActiveSupportUnits(active_y, active_subj, active_unit)

    checked_excluding = 0
    for label in active_y.unique().tolist():
        is_label = active_y.eq(label)
        assert tables.for_label(label).tolist() == active_unit[is_label].unique().tolist()
        for subject in active_subj.unique().tolist():
            both = is_label & active_subj.eq(subject)
            assert (
                tables.for_label_subject(label, subject).tolist()
                == active_unit[both].unique().tolist()
            )
            other = is_label & active_subj.ne(subject)
            assert tables.count_excluding_subject(label, subject) == (
                active_unit[other].unique().numel()
            )
            checked_excluding += 1
    assert checked_excluding > 0


def test_cached_tables_follow_the_active_index_instead_of_going_stale():
    """The active index is refreshed mid-run; a cache that missed that would corrupt feasibility."""
    bank = _random_bank(11)
    n = len(bank["y"])
    first = torch.arange(0, n // 2)
    second = torch.arange(n // 2, n)

    def fresh(rows):
        patch = bank["patch"]
        offset = int(torch.as_tensor(patch["window"]).max()) + 1
        return ActiveSupportUnits(
            torch.as_tensor(patch["y"])[rows].long(),
            torch.as_tensor(patch["subj"])[rows].long(),
            torch.where(
                torch.as_tensor(patch["event_verified"])[rows].bool(),
                torch.as_tensor(patch["event"])[rows].long() + offset,
                torch.as_tensor(patch["window"])[rows].long(),
            ),
        )

    # Interleave the two indices so a single-entry or key-blind cache would hand back the wrong one.
    for rows in (first, second, first, second, first):
        offset, cached = active_support_units(bank, rows)
        reference = fresh(rows)
        assert offset == int(torch.as_tensor(bank["patch"]["window"]).max()) + 1
        labels = torch.as_tensor(bank["patch"]["y"])[rows].unique().tolist()
        subjects = torch.as_tensor(bank["patch"]["subj"])[rows].unique().tolist()
        for label in labels:
            assert cached.for_label(label).tolist() == reference.for_label(label).tolist()
            for subject in subjects:
                assert (
                    cached.for_label_subject(label, subject).tolist()
                    == reference.for_label_subject(label, subject).tolist()
                )
                assert (
                    cached.count_excluding_subject(label, subject)
                    == reference.count_excluding_subject(label, subject)
                )


def test_cache_reuses_the_same_tables_for_an_unchanged_index():
    bank = _random_bank(12)
    rows = torch.arange(len(bank["y"]))
    _, first = active_support_units(bank, rows)
    _, again = active_support_units(bank, rows.clone())
    assert again is first, "an unchanged active index must not rebuild the tables"


def test_tables_handle_an_empty_active_index():
    empty = torch.empty(0, dtype=torch.long)
    tables = ActiveSupportUnits(empty, empty, empty)
    assert tables.for_label(0).tolist() == []
    assert tables.for_label_subject(0, 0).tolist() == []
    assert tables.count_excluding_subject(0, 0) == 0
