"""Task-2 batch construction under the compatibility contract (doc sections 3-4).

An episode draws a personal reference set and one query. A batch is assembled so
that every comparison inside it is like-for-like:

1. no batch crosses a ``SensorCompatibilityKey`` (device family, placement,
   channel set, gravity state). Native rate is deliberately not part of the key.
2. no batch crosses a source dataset, so "which dataset" can never stand in for
   "which person".
3. anchor and positive share (dataset, subject, task, key) and differ in
   execution; ``different_day`` and ``remount`` positives are over-represented
   relative to their raw frequency because those pairs teach the between-day floor.
4. an other-subject negative shares (dataset, task, key) and differs in subject.
5. a modified negative is a same-person execution carrying a declared physical
   modification; nuisance transforms ride on positives *and* negatives so that
   "was transformed" is uninformative.
6. a query is never a member of its own reference set, and no transformed
   descendant of a query appears there.
7. the channel set is part of the compatibility key, so a batch is automatically
   all-six-axis or all-acceleration-only. Producing an acceleration-only training
   analogue for MoniPar is a *corpus* decision, not a batch one: the reduced
   signal must be encoded as such (``channel_views`` in
   ``data/adapters/task2_modified_v1.py``) and cannot be projected back out of
   embeddings that were computed from six channels.

Splitting by subject and session happens before any transform is generated; this
module refuses to build a batch whose members violate the rules rather than
silently repairing them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import random
import math
from typing import Any

import numpy as np
import torch

from applications.motion_monitoring.data.compatibility import SensorCompatibilityKey
from .contracts import BoundedExecution, ExecutionEpisode


PositiveRelation = str
RELATIONS = ("same_session", "different_session", "different_day")
SIX_AXIS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
ACCELERATION_ONLY = ("acc_x", "acc_y", "acc_z")


@dataclass(frozen=True)
class ExecutionRecord:
    """One bounded execution with the provenance the batch rules need.

    ``origin_execution_id`` is the physical execution this came from. A declared
    modification of an execution shares its origin, which is how a query is kept
    from ever being a transformed descendant of its own reference.
    ``source_dataset`` is where the movement was recorded, not where the variant
    is stored, so a pre-materialised variant groups with its clean siblings.
    """

    execution: BoundedExecution
    key: SensorCompatibilityKey
    day: str | None = None
    origin_execution_id: str | None = None
    variant: str = "clean"
    modification_kind: str | None = None
    severity: float = 0.0
    nuisance_kind: str | None = None
    source_dataset: str | None = None
    source_subject_id: str | None = None

    def __post_init__(self) -> None:
        if self.variant not in {"clean", "nuisance", "modified"}:
            raise ValueError(f"unknown variant {self.variant!r}")
        if (self.variant == "modified") != bool(self.modification_kind):
            raise ValueError("a modified record declares a modification kind, others do not")
        if self.variant == "modified" and not 0.0 < self.severity <= 1.0:
            raise ValueError("a modified record needs a severity in (0, 1]")

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.dataset, self.execution.subject_id, self.execution.task_id)

    @property
    def dataset(self) -> str:
        return self.source_dataset or self.execution.dataset

    @property
    def root_id(self) -> str:
        return self.origin_execution_id or self.execution.execution_id

    @property
    def accepted(self) -> bool:
        """Whether this record may serve as a reference or an accepted query."""

        return self.variant in {"clean", "nuisance"}


def _digest(*parts: object) -> int:
    text = ":".join(str(part) for part in parts)
    return int(sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def relation_between(anchor: ExecutionRecord, other: ExecutionRecord) -> PositiveRelation:
    if anchor.day is not None and other.day is not None and anchor.day != other.day:
        return "different_day"
    if anchor.execution.session_id != other.execution.session_id:
        return "different_session"
    return "same_session"


@dataclass(frozen=True)
class EpisodePlan:
    """What a batch actually contains, for audit and telemetry."""

    dataset: str
    key: SensorCompatibilityKey
    reference_count: int
    positives: int
    modified: int
    other_subject: int
    relations: dict[str, int]
    channels: tuple[str, ...] = SIX_AXIS


class Task2BatchBuilder:
    """Draw compatibility-clean contrastive batches from a pool of executions."""

    def __init__(
        self,
        records: Sequence[ExecutionRecord],
        *,
        reference_count: int = 3,
        positives_per_episode: int = 2,
        modified_per_episode: int = 2,
        other_subject_per_episode: int = 1,
        different_day_boost: float = 3.0,
        dataset_weights: Mapping[str, float] | None = None,
        seed: int = 20260902,
    ) -> None:
        if reference_count < 1:
            raise ValueError("a reference set needs at least one execution")
        if positives_per_episode < 1:
            raise ValueError("an episode needs at least one accepted query")
        if modified_per_episode + other_subject_per_episode < 1:
            raise ValueError("an episode needs at least one negative")
        if different_day_boost < 1.0:
            raise ValueError("different-day boost must be at least 1")
        self.dataset_weights = dict(dataset_weights or {})
        if any(value < 0 for value in self.dataset_weights.values()):
            raise ValueError("dataset weights must be non-negative")
        self.reference_count = reference_count
        self.positives_per_episode = positives_per_episode
        self.modified_per_episode = modified_per_episode
        self.other_subject_per_episode = other_subject_per_episode
        self.different_day_boost = different_day_boost
        self.seed = seed
        self._records = tuple(records)
        self._by_group: dict[tuple[str, SensorCompatibilityKey, str, str], list[ExecutionRecord]] = (
            defaultdict(list)
        )
        self._by_task: dict[tuple[str, SensorCompatibilityKey, str], list[ExecutionRecord]] = (
            defaultdict(list)
        )
        for record in self._records:
            dataset, subject, task = record.identity
            self._by_group[(dataset, record.key, subject, task)].append(record)
            self._by_task[(dataset, record.key, task)].append(record)
        self._eligible = [
            group
            for group, items in self._by_group.items()
            if len({item.root_id for item in items if item.accepted})
            >= reference_count + positives_per_episode
            and len({item.root_id for item in items if item.variant == "modified"})
            >= modified_per_episode
            and len(
                {
                    item.root_id
                    for item in self._by_task[(group[0], group[1], group[3])]
                    if item.execution.subject_id != group[2] and item.accepted
                }
            )
            >= other_subject_per_episode
        ]
        self._eligible.sort(key=lambda group: (group[0], str(group[1]), group[2], group[3]))

    @property
    def eligible_groups(self) -> tuple[tuple[str, SensorCompatibilityKey, str, str], ...]:
        return tuple(self._eligible)

    def _weighted_positive_order(
        self, anchor: ExecutionRecord, candidates: Sequence[ExecutionRecord], rng: random.Random
    ) -> list[ExecutionRecord]:
        scored = []
        for candidate in candidates:
            relation = relation_between(anchor, candidate)
            weight = self.different_day_boost if relation == "different_day" else 1.0
            # Exponential-race sampling without replacement: larger weights have
            # smaller expected keys and are selected earlier.
            noise = max(rng.random(), 1e-12)
            scored.append((-math.log(noise) / weight, candidate))
        scored.sort(key=lambda item: item[0])
        return [candidate for _, candidate in scored]

    def build_episode(
        self,
        group: tuple[str, SensorCompatibilityKey, str, str],
        *,
        rng: random.Random,
    ) -> tuple[list[ExecutionEpisode], EpisodePlan]:
        """Assemble every episode sharing one reference set, by selection only.

        Variants are pre-materialised and encoded once (see
        ``data/adapters/task2_modified_v1.py``), so this method never transforms a
        signal. It chooses which already-existing execution plays which role.
        """

        dataset, key, subject, task = group
        view = tuple(key.channels)
        members = [item for item in self._by_group[group] if item.accepted]
        rng.shuffle(members)
        def unique_roots(items: Sequence[ExecutionRecord]) -> list[ExecutionRecord]:
            selected: list[ExecutionRecord] = []
            roots: set[str] = set()
            for item in items:
                if item.root_id in roots:
                    continue
                roots.add(item.root_id)
                selected.append(item)
            return selected

        clean = unique_roots([item for item in members if item.variant == "clean"])
        references = (clean or unique_roots(members))[: self.reference_count]
        if len(references) < self.reference_count:
            raise ValueError(f"group {group} cannot supply a full reference set")
        reference_roots = {item.root_id for item in references}
        anchor = references[0]
        candidates = unique_roots(
            [item for item in members if item.root_id not in reference_roots]
        )
        positives = self._weighted_positive_order(anchor, candidates, rng)[
            : self.positives_per_episode
        ]
        if len(positives) < self.positives_per_episode:
            raise ValueError(f"group {group} cannot supply enough accepted queries")

        reference_executions = tuple(item.execution for item in references)
        episodes: list[ExecutionEpisode] = []
        relations: dict[str, int] = defaultdict(int)
        for positive in positives:
            episodes.append(
                ExecutionEpisode(
                    accepted_references=reference_executions,
                    query=positive.execution,
                    episode_kind="accepted_query",
                    nuisance_kind=positive.nuisance_kind,
                )
            )
            relations[relation_between(anchor, positive)] += 1

        modified = unique_roots(
            [
                item
                for item in self._by_group[group]
                if item.variant == "modified" and item.root_id not in reference_roots
            ]
        )
        rng.shuffle(modified)
        for item in modified[: self.modified_per_episode]:
            episodes.append(
                ExecutionEpisode(
                    accepted_references=reference_executions,
                    query=item.execution,
                    episode_kind="modified_query",
                    severity=item.severity,
                    modification_kind=item.modification_kind,
                )
            )

        others = unique_roots(
            [
                item
                for item in self._by_task[(dataset, key, task)]
                if item.execution.subject_id != subject and item.accepted
            ]
        )
        rng.shuffle(others)
        selected_others = others[: self.other_subject_per_episode]
        for item in selected_others:
            episodes.append(
                ExecutionEpisode(
                    accepted_references=reference_executions,
                    query=item.execution,
                    episode_kind="other_subject_query",
                )
            )

        plan = EpisodePlan(
            dataset=dataset,
            key=key,
            reference_count=len(references),
            positives=len(positives),
            modified=min(len(modified), self.modified_per_episode),
            other_subject=len(selected_others),
            relations=dict(relations),
            channels=view,
        )
        return episodes, plan

    def build_batch(
        self,
        *,
        groups: int = 2,
        seed: int | None = None,
    ) -> tuple[list[ExecutionEpisode], list[EpisodePlan]]:
        """Draw ``groups`` reference sets from ONE dataset and compatibility key."""

        if groups < 1:
            raise ValueError("a batch needs at least one reference set")
        if not self._eligible:
            raise ValueError("no group has enough executions for the declared episode shape")
        rng = random.Random(self.seed if seed is None else seed)
        # Select the source first, then its compatibility key. Dataset weights are
        # source-level weights: a source must not become more likely merely because
        # it exposes more channel views. Only configurations that can supply the
        # requested number of independent reference sets are eligible.
        configuration_counts = Counter(group[:2] for group in self._eligible)
        configurations = sorted(
            {
                configuration
                for configuration, count in configuration_counts.items()
                if count >= groups
            },
            key=lambda item: (item[0], str(item[1])),
        )
        if not configurations:
            raise ValueError(
                f"no source/configuration can supply {groups} independent reference sets"
            )
        sources = sorted({dataset for dataset, _ in configurations})
        if self.dataset_weights:
            weights = [max(self.dataset_weights.get(dataset, 0.0), 0.0) for dataset in sources]
            if not any(weights):
                raise ValueError("every eligible source has weight zero")
        else:
            weights = [
                float(
                    sum(
                        configuration_counts[item]
                        for item in configurations
                        if item[0] == dataset
                    )
                )
                for dataset in sources
            ]
        dataset = rng.choices(sources, weights=weights, k=1)[0]
        source_configurations = [item for item in configurations if item[0] == dataset]
        _, key = rng.choice(source_configurations)
        pool = [group for group in self._eligible if group[0] == dataset and group[1] == key]
        rng.shuffle(pool)
        episodes: list[ExecutionEpisode] = []
        plans: list[EpisodePlan] = []
        for group in pool[:groups]:
            group_episodes, plan = self.build_episode(group, rng=rng)
            episodes.extend(group_episodes)
            plans.append(plan)
        if len(plans) != groups:
            raise RuntimeError("Task-2 batch assembly returned fewer groups than requested")
        validate_batch(episodes)
        return episodes, plans


def relation_summary(plans: Sequence[EpisodePlan]) -> dict[str, Any]:
    """Aggregate what a run's batches actually contained, for the run record.

    ``same_session_share`` is the number to watch: a ruler trained almost entirely
    on within-session positives has never been asked to tolerate a remount, and
    that shows up at evaluation as a nuisance false-alarm rate on genuine
    between-week repeats.
    """

    relations: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    channels: Counter[tuple[str, ...]] = Counter()
    for plan in plans:
        relations.update(plan.relations)
        datasets[plan.dataset] += plan.positives
        channels[tuple(plan.channels)] += 1
    total = sum(relations.values())
    return {
        "positive_relations": dict(relations),
        "same_session_share": (relations["same_session"] / total) if total else float("nan"),
        "different_day_share": (relations["different_day"] / total) if total else float("nan"),
        "positives_by_dataset": dict(datasets),
        "batches_by_channel_view": {"+".join(view): count for view, count in channels.items()},
        "reference_sets": len(plans),
    }


def validate_batch(episodes: Sequence[ExecutionEpisode]) -> None:
    """Assert the compatibility contract on an assembled batch."""

    if not episodes:
        raise ValueError("an empty batch cannot satisfy the Task-2 contract")
    datasets = {episode.query.dataset for episode in episodes}
    if len(datasets) != 1:
        raise ValueError(f"a Task-2 batch must not cross source datasets: {sorted(datasets)}")
    keys = {
        execution.sensor_config
        for episode in episodes
        for execution in (*episode.accepted_references, episode.query, *episode.personal_context)
    }
    if len(keys) != 1:
        raise ValueError("a Task-2 batch must not cross sensor compatibility keys")
    for episode in episodes:
        reference_ids = {item.execution_id for item in episode.accepted_references}
        if episode.query.execution_id in reference_ids:
            raise ValueError("a query may not appear in its own reference set")
    if not any(episode.is_positive for episode in episodes):
        raise ValueError("a Task-2 batch needs at least one accepted query")
    if not any(episode.is_negative for episode in episodes):
        raise ValueError("a Task-2 batch needs at least one negative query")
