"""Tests for Task-3 event-anchored sampling and the a-priori operating point."""

from __future__ import annotations

import random

import numpy as np
import pytest

from applications.motion_monitoring.task3.sampling import (
    EventIndex,
    EventInstance,
    build_event_index,
    crop_around,
    sample_batch_instances,
    split_identities,
)


def _instance(unit, label, recording, subject="s1", start=0.0, end=2.0, session="sess"):
    return EventInstance(
        unit_index=unit,
        dataset="ds",
        annotation_kind="gesture",
        stream_id="imu",
        label=label,
        start_sec=start,
        end_sec=end,
        subject_id=subject,
        session_id=session,
        recording_id=recording,
    )


def test_a_batch_always_contains_two_independent_executions_of_one_identity() -> None:
    """The defect this replaces: a random crop per recording almost never held two.

    Measured expected occurrences of a given identity inside one 120 s crop were
    0.34 for Opportunity and 1.0 for CrossFit clips, so under the
    different-instance rule the loss saw no positive pairs at all.
    """

    index = EventIndex(
        [
            _instance(0, "open_fridge", "r1", "s1"),
            _instance(1, "open_fridge", "r2", "s2"),
            _instance(2, "close_door", "r3", "s1"),
            _instance(3, "drink", "r4", "s3"),
        ]
    )
    for seed in range(12):
        drawn = sample_batch_instances(
            index, batch_size=4, positives=2, rng=random.Random(seed)
        )
        positives = [item for item, role in drawn if role == "positive"]
        assert len(drawn) == 4
        assert len(positives) >= 2
        # Two executions of one identity, from different recordings.
        assert len({item.label for item in positives}) == 1
        assert len({item.recording_id for item in positives[:2]}) == 2


def test_sampling_refuses_an_index_with_no_recurring_identity() -> None:
    index = EventIndex([_instance(0, "a", "r1"), _instance(1, "b", "r2")])
    with pytest.raises(ValueError, match="two independent executions"):
        sample_batch_instances(index, batch_size=2, positives=2, rng=random.Random(0))


def test_crop_contains_its_event_and_places_it_at_a_varying_offset() -> None:
    """A centred event would let the matcher infer the boundary from the window."""

    instance = _instance(0, "a", "r1", start=50.0, end=53.0)
    offsets = set()
    for seed in range(24):
        start, end = crop_around(
            instance,
            crop_seconds=20.0,
            timeline_start=0.0,
            timeline_end=300.0,
            rng=random.Random(seed),
        )
        assert end - start == pytest.approx(20.0)
        assert start <= instance.start_sec and instance.end_sec <= end
        offsets.add(round(instance.start_sec - start, 3))
    assert len(offsets) > 5, "event position inside the crop must vary"


def test_crop_falls_back_to_the_whole_timeline_when_it_is_shorter() -> None:
    instance = _instance(0, "a", "r1", start=1.0, end=3.0)
    assert crop_around(
        instance, crop_seconds=120.0, timeline_start=0.0, timeline_end=10.0,
        rng=random.Random(0),
    ) == (0.0, 10.0)


def test_identity_split_holds_out_whole_identities() -> None:
    """Task 3 claims transfer to identities never seen, so the hold-out is by identity."""

    index = EventIndex(
        [_instance(i, f"g{i//2}", f"r{i}") for i in range(12)]
    )
    fit, held = split_identities(index, holdout_fraction=0.25, seed=5)
    assert set(fit.identities) & set(held.identities) == set()
    assert len(held.identities) >= 1 and len(fit.identities) >= 1
    assert len(fit) + len(held) == len(index)
    with pytest.raises(ValueError):
        split_identities(index, holdout_fraction=0.0, seed=5)


def test_operating_point_never_exceeds_its_false_edge_budget() -> None:
    from applications.motion_monitoring.task3.train_full import fix_operating_point

    scores = np.array([3.0, 2.5, 2.0, 0.5, 0.2, -1.0, -2.0, -3.0])
    targets = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    for budget in (0.05, 0.2, 0.4, 0.9):
        result = fix_operating_point(scores, targets, false_edge_rate=budget)
        assert result["holdout_false_edge_rate"] <= budget + 1e-9
    # Ties must resolve upwards rather than silently admitting extra edges.
    tied = fix_operating_point(
        np.ones(4), np.array([1, 0, 0, 0]), false_edge_rate=0.3
    )
    assert tied["holdout_false_edge_rate"] == 0.0


def test_event_index_skips_background_and_clipped_events() -> None:
    class _Event:
        def __init__(self, label, kind, start, end, metadata=None):
            self.label, self.annotation_kind = label, kind
            self.start_sec, self.end_sec = start, end
            self.metadata = metadata or {}

    class _Recording:
        subject_id, recording_id, session_id = "s1", "r1", "sess"
        events = [
            _Event("real", "gesture", 0.0, 2.0),
            _Event("NULL", "gesture", 2.0, 4.0),
            _Event("clipped", "gesture", 4.0, 6.0, {"clipped_by_recording_crop": True}),
            _Event("other_kind", "activity", 6.0, 8.0),
        ]

    class _Unit:
        dataset, cache_index, stream_id = "ds", 0, "imu"
        annotation_kind = "gesture"
        background_labels = ("NULL",)

    index = build_event_index([_Unit()], {"ds": [_Recording()]})
    assert [item.label for item in index.instances(("ds", "gesture", "imu", "real"))] == ["real"]
    assert len(index) == 1
