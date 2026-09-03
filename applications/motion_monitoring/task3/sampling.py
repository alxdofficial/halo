"""Event-anchored batch sampling for Task 3 (design doc section 2.3).

Section 2.3 specifies that positives are *independent executions with the same
identity*. The original implementation instead cropped a random window from each
recording and formed pairs from whatever happened to land inside it. Measured
expected occurrences of a given identity inside one 120 s crop: Opportunity 0.34,
CrossFit 1.0, OpenPack 1.22, OCA 2.77. Under the ``different_instance`` rule that
yields almost no positive pairs, so the loss collapses onto negatives.

This module indexes source events first and then chooses crops around them, so a
batch is guaranteed to contain at least one identity with two independent
executions. Crops keep the deployment property that matters: the window is much
longer than the event and the event sits at a random offset inside it, so the
matcher never receives a boundary for free and must still find it on the
candidate grid.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import random
from typing import Any


@dataclass(frozen=True)
class EventInstance:
    """One bounded execution, located inside a specific manifest unit."""

    unit_index: int
    dataset: str
    annotation_kind: str
    stream_id: str
    label: str
    start_sec: float
    end_sec: float
    subject_id: str
    session_id: str
    recording_id: str

    @property
    def scope(self) -> tuple[str, str, str]:
        # Placement and channels are fixed by (dataset, stream), so this is the
        # compatibility scope without needing the representation loaded.
        return (self.dataset, self.annotation_kind, self.stream_id)

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return (*self.scope, self.label)

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


class EventIndex:
    """Index of every usable source event in a Task-3 manifest."""

    def __init__(self, instances: Sequence[EventInstance]) -> None:
        self._instances = tuple(instances)
        self._by_identity: dict[tuple[str, str, str, str], list[EventInstance]] = defaultdict(list)
        self._by_scope: dict[tuple[str, str, str], list[EventInstance]] = defaultdict(list)
        for item in self._instances:
            self._by_identity[item.identity].append(item)
            self._by_scope[item.scope].append(item)

    def __len__(self) -> int:
        return len(self._instances)

    @property
    def identities(self) -> tuple[tuple[str, str, str, str], ...]:
        return tuple(sorted(self._by_identity))

    def recurring_identities(self, *, minimum: int = 2) -> list[tuple[str, str, str, str]]:
        """Identities with enough independent executions to form a positive pair.

        Independence is by recording: two candidates carved from one execution are
        the same instance, and two executions inside one recording are weaker
        positives than two from different recordings, so recordings are counted.
        """

        eligible = []
        for identity, items in sorted(self._by_identity.items()):
            if len({item.recording_id for item in items}) >= minimum or len(items) >= minimum:
                eligible.append(identity)
        return eligible

    def instances(self, identity: tuple[str, str, str, str]) -> list[EventInstance]:
        return list(self._by_identity[identity])

    def scope_instances(self, scope: tuple[str, str, str]) -> list[EventInstance]:
        return list(self._by_scope[scope])

    def summary(self) -> dict[str, Any]:
        per_identity = {k: len(v) for k, v in self._by_identity.items()}
        return {
            "events": len(self._instances),
            "identities": len(self._by_identity),
            "scopes": len(self._by_scope),
            "recurring_identities": len(self.recurring_identities()),
            "median_instances_per_identity": (
                sorted(per_identity.values())[len(per_identity) // 2] if per_identity else 0
            ),
        }


def build_event_index(
    units: Sequence[Any],
    recording_caches,
    *,
    background_by_dataset: dict[str, frozenset[str]] | None = None,
) -> EventIndex:
    """Read every manifest unit once and record its bounded executions."""

    background_by_dataset = background_by_dataset or {}
    instances: list[EventInstance] = []
    for unit_index, unit in enumerate(units):
        recording = recording_caches[unit.dataset][unit.cache_index]
        background = frozenset(unit.background_labels) | background_by_dataset.get(
            unit.dataset, frozenset()
        )
        for event in recording.events:
            if event.annotation_kind != unit.annotation_kind:
                continue
            if event.label in background:
                continue
            if bool(event.metadata.get("clipped_by_recording_crop", False)):
                continue
            if event.end_sec <= event.start_sec:
                continue
            instances.append(
                EventInstance(
                    unit_index=unit_index,
                    dataset=unit.dataset,
                    annotation_kind=unit.annotation_kind,
                    stream_id=unit.stream_id,
                    label=event.label,
                    start_sec=float(event.start_sec),
                    end_sec=float(event.end_sec),
                    subject_id=str(recording.subject_id),
                    session_id=str(recording.session_id),
                    recording_id=str(recording.recording_id),
                )
            )
    return EventIndex(instances)


def crop_around(
    instance: EventInstance,
    *,
    crop_seconds: float,
    timeline_start: float,
    timeline_end: float,
    rng: random.Random,
) -> tuple[float, float]:
    """A window of ``crop_seconds`` containing the event at a random offset.

    The jitter is what keeps the boundary unknown: an event pinned to the centre
    of every crop would let the matcher infer position from the window instead of
    from the signal.
    """

    if crop_seconds <= 0:
        raise ValueError("crop_seconds must be positive")
    available = timeline_end - timeline_start
    if available <= crop_seconds:
        return timeline_start, timeline_end
    duration = instance.end_sec - instance.start_sec
    if duration >= crop_seconds:
        # A single execution longer than the crop: keep its leading part rather
        # than silently widening the window.
        start = instance.start_sec
    else:
        slack = crop_seconds - duration
        offset = rng.uniform(0.0, slack)
        start = instance.start_sec - offset
    start = min(max(start, timeline_start), timeline_end - crop_seconds)
    return start, start + crop_seconds


def sample_batch_instances(
    index: EventIndex,
    *,
    batch_size: int,
    positives: int,
    rng: random.Random,
) -> list[tuple[EventInstance, str]]:
    """Choose one identity to recur in this batch, then fill with same-scope negatives.

    Returns ``(instance, role)`` pairs. The caller turns each into a crop, because
    only it knows the representation bounds of the unit the instance lives in.
    """

    if batch_size < 2:
        raise ValueError("a Task-3 batch needs at least two timelines")
    if not 2 <= positives <= batch_size:
        raise ValueError("positives must be between two and the batch size")
    eligible = index.recurring_identities(minimum=2)
    if not eligible:
        raise ValueError(
            "no identity has two independent executions; the index cannot form a positive pair"
        )
    identity = eligible[rng.randrange(len(eligible))]
    members = index.instances(identity)

    # Prefer executions from different recordings: two candidates carved from one
    # execution are the same instance, and two executions inside one recording are
    # a weaker positive than two from different recordings.
    by_recording: dict[str, list[EventInstance]] = defaultdict(list)
    for item in members:
        by_recording[item.recording_id].append(item)
    recordings = sorted(by_recording)
    rng.shuffle(recordings)
    chosen: list[tuple[EventInstance, str]] = []
    for recording_id in recordings:
        if len(chosen) >= positives:
            break
        options = by_recording[recording_id]
        chosen.append((options[rng.randrange(len(options))], "positive"))
    while len(chosen) < positives:
        chosen.append((members[rng.randrange(len(members))], "positive"))

    # Negatives share the scope, so the comparison is like-for-like, and differ in
    # identity. Same subject and session where the scope offers it (section 2.3).
    pool = [item for item in index.scope_instances(identity[:3]) if item.label != identity[3]]
    anchor_subject = chosen[0][0].subject_id
    pool.sort(key=lambda item: (item.subject_id != anchor_subject, rng.random()))
    for item in pool[: batch_size - len(chosen)]:
        chosen.append((item, "negative"))
    # If the scope cannot supply enough distinct identities, add further executions
    # of the anchor rather than silently shrinking the batch.
    while len(chosen) < batch_size:
        chosen.append((members[rng.randrange(len(members))], "positive"))
    return chosen


def split_identities(
    index: EventIndex, *, holdout_fraction: float, seed: int
) -> tuple[EventIndex, EventIndex]:
    """Split an index by identity, so the hold-out contains motions never fitted.

    Task 3's deployment claim is that the matching rule transfers to identities
    whose labels were never seen, so the operating point must be fixed on held-out
    *identities* rather than held-out windows of the same ones.
    """

    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must lie in (0, 1)")
    identities = list(index.identities)
    if len(identities) < 2:
        raise ValueError("splitting by identity needs at least two identities")
    rng = random.Random(seed)
    rng.shuffle(identities)
    count = max(1, min(len(identities) - 1, round(len(identities) * holdout_fraction)))
    held = set(identities[:count])
    fit_rows = [item for item in index._instances if item.identity not in held]
    holdout_rows = [item for item in index._instances if item.identity in held]
    return EventIndex(fit_rows), EventIndex(holdout_rows)
