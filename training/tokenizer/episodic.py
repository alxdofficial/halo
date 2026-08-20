"""Episode sampling, row materialization, and memory-bank construction for Phase B.

WHY THIS FILE EXISTS
--------------------
The current trainer is ``training/tokenizer/pretrain_episodic.py`` and the current model is
``model/evidence/engine.py``. This module owns the data contract they share: independent episode
plans, subject/stream-disjoint support, fixed-size stratified banks, grouped encoder batches, and
conversion of live encoder output into ``SensorRows``.

The compact engine performs learned retrieval, hard top-k selection, evidence attention, and voting.
The closed-form scoring helpers retained near the middle of this file are historical evaluation
controls only; the current trainer does not call them.

WHAT AN EPISODE CONTAINS
------------------------
    C candidate labels
    Q query windows per candidate      the things to be classified
    k support windows per ENROLLED candidate   the "enrolled" evidence (k=0 .. max_support)
    N background windows from labels OUTSIDE the candidate set, sized so support + background is
      a fixed bank capacity

Background windows are not padding. They are the semantic bridge at k=0, when no candidate has
enrolled evidence. The current bank is stratified across streams and labels and excludes every
candidate label, matching the intended novel-label deployment condition.

SUPPORT AND QUERY NEVER SHARE A SUBJECT
---------------------------------------
Support windows for a candidate are drawn from subjects disjoint from that candidate's query
subject. Sampling both from one subject makes enrollment a same-person retrieval task, which is the
subject-leakage trap already recorded against this corpus (hapt == uci_har, same 30 people).

RELATION TO THE EXISTING CODE
-----------------------------
The corpus index, dataset, and base collate are reused unchanged. The collate is wrapped to carry the
gravity convention required by retrieval telemetry. See ``docs/design/COMPACT_EVIDENCE_ENGINE.md``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

from training.evidence.admissible_retrieval import (
    SensorRows, compatibility_mask, rank_scores, soft_vote_all, vote,
)
from training.tokenizer.pretrain_data import _pad_sensor_rows

# Deployment's vote temperature (admissible_retrieval.soft_vote_all default). Named here so the
# trainer logs one number rather than relying on a default it does not state.
VOTE_TEMPERATURE = 0.07


# ==================================================================================================
# Episode plans
# ==================================================================================================
@dataclass(frozen=True)
class EpisodeSpec:
    """Shape of the episodes drawn each step. All counts are per episode."""

    candidate_counts: tuple[int, ...] = (2, 4, 8, 16)
    queries_per_candidate: int = 2
    max_support: int = 4
    background_windows: int = 16
    #: Fraction of positive-support episodes that replace canonical names with neutral episode-local
    #: aliases. Every candidate is enrolled in those episodes because an unseen arbitrary name with
    #: no support is information-theoretically unanswerable.
    alias_episode_fraction: float = 0.5
    #: 'subject' — support comes from other people, possibly on the SAME device.
    #: 'stream'  — support additionally comes from a different acquisition stream, which is the
    #:             regime HALO claims (unseen acquisition configurations). Measured on this corpus,
    #:             'subject' leaves a +42.4 point stream-identity lift for the correct answer.
    disjointness: str = "subject"
    #: Draw EVERY candidate's queries from one acquisition stream. Combined with
    #: disjointness='stream' this drives the provenance lift to exactly zero: no support row shares
    #: any candidate's query stream, so device identity cannot favour one candidate over another.
    #: It is also what deployment looks like — one query arrives from one device.
    shared_query_stream: bool = False

    def validate(self) -> None:
        if not self.candidate_counts or min(self.candidate_counts) < 2:
            raise ValueError("an episode needs at least two candidate labels")
        if self.queries_per_candidate < 1:
            raise ValueError("queries_per_candidate must be positive")
        if self.max_support < 0:
            raise ValueError("max_support cannot be negative")
        if self.background_windows < 0:
            raise ValueError("background_windows cannot be negative")
        if not 0.0 <= self.alias_episode_fraction <= 1.0:
            raise ValueError("alias_episode_fraction must be in [0,1]")
        if self.disjointness not in {"subject", "stream"}:
            raise ValueError("disjointness must be 'subject' or 'stream'")
        if self.shared_query_stream and self.disjointness != "stream":
            raise ValueError(
                "shared_query_stream is only meaningful with disjointness='stream'; on its own it "
                "puts every candidate's support on the query's own device"
            )


@dataclass(frozen=True)
class EpisodePlan:
    """One episode, resolved to dataset positions.

    The flat batch layout is FIXED — queries, then support, then background — so the trainer can
    slice an encoded batch back into roles positionally. ``expected_labels`` lets it assert the
    DataLoader delivered the windows this plan asked for; a silent desync between plan and batch
    would mislabel every row without raising.
    """

    candidates: tuple[int, ...]           # label ids, one per candidate slot
    query_positions: tuple[int, ...]      # dataset positions, grouped by candidate slot
    query_slot: tuple[int, ...]           # candidate slot each query belongs to
    support_positions: tuple[int, ...]
    support_slot: tuple[int, ...]         # candidate slot each support window is bound to
    background_positions: tuple[int, ...]
    support_k: int                        # requested k for this episode (0 = zero-shot episode)
    enrolled_slots: tuple[int, ...]       # candidate slots that actually received support
    label_mode: str = "coherent"          # coherent | random_alias

    @property
    def n_query(self) -> int:
        return len(self.query_positions)

    @property
    def n_support(self) -> int:
        return len(self.support_positions)

    @property
    def n_background(self) -> int:
        return len(self.background_positions)

    def flat_positions(self) -> list[int]:
        return list(self.query_positions + self.support_positions + self.background_positions)

    def expected_labels(self) -> list[int]:
        """Label id per flat batch row for queries and support (background is unconstrained)."""
        return ([self.candidates[s] for s in self.query_slot]
                + [self.candidates[s] for s in self.support_slot])


def matched_support_variants(
    plan: EpisodePlan, support_grid: tuple[int, ...],
) -> list[EpisodePlan]:
    """Replay one base episode at nested support counts without changing its task.

    Candidate labels, queries, background rows, and the enrolled candidate subset stay fixed.  For
    each enrolled candidate, k support rows are a prefix of the maximum-k draw, so k=1 is literally a
    subset of k=2 and k=4.  This makes a validation k-curve an intervention on support quantity rather
    than a comparison between unrelated random episodes.
    """
    if not support_grid:
        raise ValueError("support_grid cannot be empty")
    if min(support_grid) < 0 or max(support_grid) > plan.support_k:
        raise ValueError(
            f"support grid {support_grid} must lie between zero and base k={plan.support_k}"
        )
    by_slot: dict[int, list[int]] = {}
    for position, slot in zip(plan.support_positions, plan.support_slot):
        by_slot.setdefault(int(slot), []).append(int(position))

    variants = []
    for support_k in support_grid:
        positions: list[int] = []
        slots: list[int] = []
        enrolled: list[int] = []
        for slot in plan.enrolled_slots:
            selected = by_slot.get(int(slot), [])[:support_k]
            if selected:
                positions.extend(selected)
                slots.extend([int(slot)] * len(selected))
                enrolled.append(int(slot))
        variants.append(dataclasses.replace(
            plan,
            support_positions=tuple(positions),
            support_slot=tuple(slots),
            support_k=int(support_k),
            enrolled_slots=tuple(enrolled),
            label_mode=("coherent" if support_k == 0 else plan.label_mode),
        ))
    return variants


def label_window_table(
    keys, subject_ids: np.ndarray,
) -> dict[int, dict[int, np.ndarray]]:
    """label id -> subject id -> dataset positions carrying that (label, subject).

    Built once. Episode sampling is then pure integer work on small arrays, which is what keeps a
    step from spending its budget in Python.
    """
    if len(subject_ids) != len(keys):
        raise ValueError("subject_ids must align 1:1 with keys")
    table: dict[int, dict[int, list[int]]] = {}
    for position, key in enumerate(keys):
        table.setdefault(int(key.label_id), {}).setdefault(
            int(subject_ids[position]), []
        ).append(position)
    return {
        label: {subject: np.asarray(rows, dtype=np.int64)
                for subject, rows in by_subject.items()}
        for label, by_subject in table.items()
    }


def stream_label_table(
    keys, subject_ids: np.ndarray, stream_ids: np.ndarray,
) -> dict[int, dict[int, dict[int, np.ndarray]]]:
    """stream -> label -> subject -> positions. Backs ``shared_query_stream`` episode assembly."""
    if not (len(subject_ids) == len(stream_ids) == len(keys)):
        raise ValueError("subject_ids and stream_ids must align 1:1 with keys")
    nested: dict[int, dict[int, dict[int, list[int]]]] = {}
    for position, key in enumerate(keys):
        nested.setdefault(int(stream_ids[position]), {}).setdefault(
            int(key.label_id), {}
        ).setdefault(int(subject_ids[position]), []).append(position)
    return {
        stream: {label: {subject: np.asarray(rows, dtype=np.int64)
                         for subject, rows in by_subject.items()}
                 for label, by_subject in by_label.items()}
        for stream, by_label in nested.items()
    }


def eligible_labels(
    table: dict[int, dict[int, np.ndarray]],
    spec: EpisodeSpec,
    stream_ids: np.ndarray | None = None,
    execution_ids: np.ndarray | None = None,
) -> list[int]:
    """Labels that can supply a query subject and (when k>0) a disjoint support subject.

    Under ``disjointness='stream'`` the label must additionally span at least two acquisition
    streams, or its support can only ever come from the query's own device.
    """
    out = []
    for label, by_subject in table.items():
        query_capable = [
            subject for subject, rows in by_subject.items()
            if len(np.unique(execution_ids[rows])) >= spec.queries_per_candidate
        ] if execution_ids is not None else [
            subject for subject, rows in by_subject.items()
            if len(rows) >= spec.queries_per_candidate
        ]
        if not query_capable:
            continue
        if spec.max_support > 0 and len(by_subject) < 2:
            continue                       # no subject left over to enroll from
        if spec.max_support > 0 and spec.disjointness == "stream":
            if stream_ids is None:
                raise ValueError("disjointness='stream' requires stream_ids")
            positions = np.concatenate(list(by_subject.values()))
            if len(np.unique(stream_ids[positions])) < 2:
                continue                   # single-device label: no cross-device support exists
        out.append(int(label))
    return sorted(out)


def provenance_lift(
    plans: list["EpisodePlan"], stream_ids: np.ndarray,
) -> dict[str, float]:
    """How much acquisition-stream identity alone reveals the answer.

    A support row that shares its BOUND candidate's query stream more often than it shares a
    DISTRACTOR's lets "retrieve from my own device" stand in for recognising the activity. The
    provenance probe already reads dataset identity out of these features at BA 0.70, so the
    shortcut is available whenever the lift is positive. Reported at startup: a training arm whose
    episodes carry a large lift is measuring device matching, not adaptation.
    """
    matched, distractor = [], []
    for plan in plans:
        query_streams: dict[int, set[int]] = {}
        for position, slot in zip(plan.query_positions, plan.query_slot):
            query_streams.setdefault(slot, set()).add(int(stream_ids[position]))
        for position, slot in zip(plan.support_positions, plan.support_slot):
            stream = int(stream_ids[position])
            matched.append(stream in query_streams.get(slot, set()))
            for other, streams in query_streams.items():
                if other != slot:
                    distractor.append(stream in streams)
    if not matched:
        return {"matched": 0.0, "distractor": 0.0, "lift": 0.0, "support_rows": 0}
    return {
        "matched": float(np.mean(matched)),
        "distractor": float(np.mean(distractor)) if distractor else 0.0,
        "lift": float(np.mean(matched)) - (float(np.mean(distractor)) if distractor else 0.0),
        "support_rows": len(matched),
    }


def build_episode_plans(
    table: dict[int, dict[int, np.ndarray]],
    pool: list[int],
    *,
    n_episodes: int,
    spec: EpisodeSpec,
    seed: int,
    support_schedule: tuple[int, ...] | None = None,
    stream_ids: np.ndarray | None = None,
    query_table: dict[int, dict[int, np.ndarray]] | None = None,
    execution_ids: np.ndarray | None = None,
) -> list[EpisodePlan]:
    """Materialise every episode up front.

    Precomputing rather than streaming is deliberate: the batch sampler and the training loop must
    agree on which episode they are looking at, and a shared RNG advanced from two places is the
    classic way for them to silently disagree. A list is indexed by both, so they cannot.

    ``support_schedule`` pins k round-robin instead of drawing it. VALIDATION MUST USE IT: with a
    random draw the k cells a run reports depend on the seed and the episode count, so two arms are
    not comparable and k=0 — the zero-shot canary this design exists to protect — can be absent
    entirely. Training leaves it None; a random k there is the intended curriculum.
    """
    spec.validate()
    if spec.disjointness == "stream" and spec.max_support > 0 and stream_ids is None:
        raise ValueError("disjointness='stream' requires stream_ids")
    required = max(spec.candidate_counts) + (1 if spec.background_windows else 0)
    if len(pool) < required:
        # The +1 is not slack. Background rows are drawn from labels OUTSIDE the candidate set, so a
        # pool exactly the size of the candidate set yields none — and a k=0 episode with no
        # background has NO memory at all, making its target unreachable by any rule.
        raise ValueError(
            f"episodes need {required} eligible labels (max candidates "
            f"{max(spec.candidate_counts)}"
            f"{' + 1 reserved for background' if spec.background_windows else ''}) "
            f"but only {len(pool)} are available; lower --candidate-counts or widen --datasets"
        )
    rng = np.random.default_rng(seed)
    pool_array = np.asarray(pool, dtype=np.int64)
    plans: list[EpisodePlan] = []
    for _ in range(n_episodes):
        n_candidates = int(rng.choice(spec.candidate_counts))
        candidates = rng.choice(pool_array, n_candidates, replace=False).tolist()
        support_k = (int(rng.integers(0, spec.max_support + 1)) if support_schedule is None
                     else int(support_schedule[len(plans) % len(support_schedule)]))
        if support_k > spec.max_support:
            raise ValueError(
                f"support_schedule asks for k={support_k} but max_support is {spec.max_support}"
            )
        # Enrollment is PARTIAL by construction. With a uniform count every episode is either
        # "nothing enrolled" or "everything enrolled", and neither teaches the model to weigh
        # enrolled evidence against text evidence within one decision.
        label_mode = (
            "random_alias"
            if support_k > 0 and rng.random() < spec.alias_episode_fraction else "coherent"
        )
        enrolled = (
            set(range(n_candidates))
            if label_mode == "random_alias" else (
                set() if support_k == 0 else set(
                    rng.choice(n_candidates, int(rng.integers(1, n_candidates + 1)),
                               replace=False).tolist()
                )
            )
        )

        query_positions: list[int] = []
        query_slot: list[int] = []
        support_positions: list[int] = []
        support_slot: list[int] = []
        actually_enrolled: list[int] = []
        for slot, label in enumerate(candidates):
            by_subject = (query_table[int(label)] if query_table is not None
                          else table[int(label)])
            capable = sorted(
                subject for subject, rows in by_subject.items()
                if (len(np.unique(execution_ids[rows])) if execution_ids is not None else len(rows))
                >= spec.queries_per_candidate
            )
            query_subject = int(rng.choice(capable))
            rows = by_subject[query_subject]
            if execution_ids is None:
                picked = rng.choice(rows, spec.queries_per_candidate, replace=False)
            else:
                executions = np.unique(execution_ids[rows])
                selected_executions = rng.choice(
                    executions, spec.queries_per_candidate, replace=False,
                )
                picked = np.asarray([
                    rng.choice(rows[execution_ids[rows] == execution])
                    for execution in selected_executions
                ], dtype=np.int64)
            query_positions.extend(int(v) for v in picked)
            query_slot.extend([slot] * spec.queries_per_candidate)

            if slot not in enrolled:
                continue
            # Support is drawn from the label's FULL window set, not the (possibly stream-restricted)
            # query view — otherwise a shared-query-stream episode could only ever enroll on-device.
            other = [rows for subject, rows in table[int(label)].items()
                     if subject != query_subject]
            if not other:
                continue                   # eligibility guarantees this only when max_support == 0
            available = np.concatenate(other)
            if spec.disjointness == "stream":
                # Support must also come off a DIFFERENT device. Otherwise "which candidate's
                # evidence was recorded on my hardware" answers the episode without recognising
                # anything, and that is exactly the cue the cross-configuration claim forbids.
                query_streams = np.unique(stream_ids[np.asarray(picked)])
                available = available[~np.isin(stream_ids[available], query_streams)]
                if not len(available):
                    continue               # nothing enrollable off-device for this candidate
            available_units = (
                np.unique(execution_ids[available]) if execution_ids is not None else available
            )
            take = min(support_k, len(available_units))
            if take == 0:
                continue
            chosen_units = rng.choice(available_units, take, replace=False)
            chosen = (
                np.asarray([rng.choice(available[execution_ids[available] == execution])
                            for execution in chosen_units], dtype=np.int64)
                if execution_ids is not None else chosen_units
            )
            support_positions.extend(int(v) for v in chosen)
            support_slot.extend([slot] * take)
            actually_enrolled.append(slot)

        background_positions: list[int] = []
        if spec.background_windows:
            outside = np.setdiff1d(pool_array, np.asarray(candidates, dtype=np.int64))
            if len(outside):
                # Sample the LABEL first, then a window. Sampling windows directly would let the
                # corpus's window-count imbalance (capture24 dominates) decide what the text-vote
                # bridge is made of.
                for label in rng.choice(outside, spec.background_windows, replace=True).tolist():
                    by_subject = table[int(label)]
                    subject = int(rng.choice(sorted(by_subject)))
                    rows = by_subject[subject]
                    background_positions.append(int(rng.choice(rows)))

        if not support_positions and not background_positions:
            # No enrolled evidence and no corpus rows: every candidate scores identically and the
            # loss is a constant. Silently training on such episodes would dilute every real
            # gradient with noise-free zeros.
            raise ValueError(
                "constructed an episode with no memory rows at all (no support, no background); "
                "raise --background-windows or --max-support"
            )
        if label_mode == "random_alias" and len(actually_enrolled) != n_candidates:
            # Cross-stream availability can fail for the particular query stream even when a label
            # is globally eligible. Keep the episode, but do not present an unanswerable alias.
            label_mode = "coherent"
        plans.append(EpisodePlan(
            candidates=tuple(int(v) for v in candidates),
            query_positions=tuple(query_positions),
            query_slot=tuple(query_slot),
            support_positions=tuple(support_positions),
            support_slot=tuple(support_slot),
            background_positions=tuple(background_positions),
            support_k=support_k,
            enrolled_slots=tuple(sorted(actually_enrolled)),
            label_mode=label_mode,
        ))
    return plans


def build_shared_stream_plans(
    table: dict[int, dict[int, np.ndarray]],
    stream_table: dict[int, dict[int, dict[int, np.ndarray]]],
    pool: list[int],
    *,
    n_episodes: int,
    spec: EpisodeSpec,
    seed: int,
    stream_ids: np.ndarray,
    support_schedule: tuple[int, ...] | None = None,
    execution_ids: np.ndarray | None = None,
) -> list[EpisodePlan]:
    """Episodes whose every query comes from ONE acquisition stream.

    This is the configuration in which acquisition provenance carries zero information: with all
    queries on stream S and (``disjointness='stream'``) no support drawn from S, a support row is
    off-device for every candidate equally. ``provenance_lift`` over these plans is exactly 0.0.

    It also mirrors deployment — a query arrives from one device, and the enrolled examples may have
    been recorded on another. Composed from ``build_episode_plans`` one episode at a time rather
    than reimplemented, so plan structure, subject disjointness and the guards stay identical.
    """
    spec.validate()
    if not spec.shared_query_stream:
        raise ValueError("build_shared_stream_plans requires spec.shared_query_stream")
    rng = np.random.default_rng(seed)
    allowed = set(pool)
    # A stream can host an episode only if it carries enough of the pool's labels with enough
    # windows from a single subject to fill the queries.
    usable: dict[int, list[int]] = {}
    for stream, by_label in stream_table.items():
        labels = [
            label for label, by_subject in by_label.items()
            if label in allowed and any(
                (len(np.unique(execution_ids[rows])) if execution_ids is not None else len(rows))
                >= spec.queries_per_candidate for rows in by_subject.values()
            )
        ]
        # Same reservation as build_episode_plans: background comes from a label outside the
        # candidate set, so the stream must carry at least one more label than the episode uses.
        if len(labels) >= min(spec.candidate_counts) + (1 if spec.background_windows else 0):
            usable[int(stream)] = sorted(labels)
    if not usable:
        raise ValueError(
            f"no acquisition stream carries {min(spec.candidate_counts)} eligible labels with "
            f"{spec.queries_per_candidate} windows from one subject; shared-query-stream episodes "
            "are not constructible on this corpus subset"
        )
    streams = sorted(usable)
    plans: list[EpisodePlan] = []
    for episode in range(n_episodes):
        stream = int(rng.choice(streams))
        labels = usable[stream]
        ceiling = len(labels) - (1 if spec.background_windows else 0)
        counts = tuple(c for c in spec.candidate_counts if c <= ceiling)
        episode_spec = dataclasses.replace(spec, candidate_counts=counts)
        schedule = None if support_schedule is None else (
            support_schedule[len(plans) % len(support_schedule)],
        )
        plans.extend(build_episode_plans(
            table, labels, n_episodes=1, spec=episode_spec,
            seed=int(rng.integers(0, 2**31 - 1)), support_schedule=schedule,
            stream_ids=stream_ids, query_table=stream_table[stream],
            execution_ids=execution_ids,
        ))
    return plans


class EpisodicBatchSampler(Sampler[list[int]]):
    """Yields one episode's flat dataset positions per batch. Plans are fixed at construction."""

    def __init__(self, plans: list[EpisodePlan]):
        self.plans = plans

    def __len__(self) -> int:
        return len(self.plans)

    def __iter__(self):
        for plan in self.plans:
            yield plan.flat_positions()


class GroupedEpisodicBatchSampler(Sampler[list[int]]):
    """Concatenate independent episodes for one shared encoder forward.

    Episode semantics do not cross this boundary: each plan is still scored separately with its own
    candidates, aliases, support, and background. Only the expensive sensor encoder sees the union.
    """

    def __init__(self, plans: list[EpisodePlan], episodes_per_batch: int):
        if episodes_per_batch < 1:
            raise ValueError("episodes_per_batch must be positive")
        if len(plans) % episodes_per_batch:
            raise ValueError("plan count must be divisible by episodes_per_batch")
        self.plans = plans
        self.episodes_per_batch = int(episodes_per_batch)

    def __len__(self) -> int:
        return len(self.plans) // self.episodes_per_batch

    def __iter__(self):
        for start in range(0, len(self.plans), self.episodes_per_batch):
            positions: list[int] = []
            for plan in self.plans[start:start + self.episodes_per_batch]:
                positions.extend(plan.flat_positions())
            yield positions


class EpisodicCollate:
    """Wrap a Phase-A collate and carry the two fields the Phase-A objective never needed.

    A wrapper rather than an edit to ``MultiScaleCollate``: the retrieval rule needs each window's
    gravity convention (to refuse gravity-removed vs gravity-present accelerometer comparisons) and
    optionally its sensor_bias, neither of which the SSL objective reads. Adding them here keeps the
    shared collate byte-identical for every existing caller.
    """

    def __init__(self, base):
        self.base = base

    def __call__(self, batch: list[dict]) -> dict:
        out = self.base(batch)
        out["gravity_state"] = [item.get("gravity_state") for item in batch]
        if "sensor_bias" in batch[0]:
            out["sensor_bias"] = _pad_sensor_rows(batch, "sensor_bias")
        return out


# ==================================================================================================
# Learned query head
# ==================================================================================================
# ==================================================================================================
# Live retrieval rows
# ==================================================================================================
def sensor_modality_codes(
    sensor_id: torch.Tensor, channel_mask: torch.Tensor, n_sensors: int,
) -> torch.Tensor:
    """(B, n_sensors) 0=accel, 1=gyro, -1=slot carries no live channel.

    Derived from the SAME two tensors the encoder folds with, rather than re-deriving from the
    stream name: augmentation can drop a modality, and a code that disagreed with the tokens would
    silently authorise physically meaningless comparisons that ``compatibility_mask`` exists to
    forbid. Mirrors ``build_memory``'s ``modalities_present`` ordering.
    """
    if sensor_id.shape != channel_mask.shape:
        raise ValueError("sensor_id and channel_mask must both be (B, 6)")
    device = sensor_id.device
    codes = torch.full((sensor_id.shape[0], n_sensors), -1, dtype=torch.long, device=device)
    for modality, axes in ((0, slice(0, 3)), (1, slice(3, 6))):
        ids = sensor_id[:, axes].clone()
        ids[~channel_mask[:, axes]] = -1
        for column in range(ids.shape[1]):
            live = ids[:, column] >= 0
            if not bool(live.any()):
                continue
            rows = torch.nonzero(live, as_tuple=True)[0]
            slots = ids[rows, column]
            previous = codes[rows, slots]
            if bool(((previous >= 0) & (previous != modality)).any()):
                raise ValueError(
                    "a sensor slot carries both accelerometer and gyroscope channels; "
                    "sensor_id and channel_mask disagree about the sensor layout"
                )
            codes[rows, slots] = modality
    return codes


def gravity_codes(modality: torch.Tensor, gravity_state: list) -> torch.Tensor:
    """(B, n_sensors) 1 = gravity-removed ACCELEROMETER, else 0 — build_memory's convention.

    The gyroscope has no gravity component, so its code is always 0 and it is never split across the
    convention. Read from the per-window state (augmentation can remove gravity) rather than the
    stream's declared state.
    """
    if len(gravity_state) != modality.shape[0]:
        raise ValueError("gravity_state must have one entry per batch row")
    removed = torch.tensor(
        [state == "removed" for state in gravity_state],
        dtype=torch.bool, device=modality.device,
    ).unsqueeze(1)
    return (removed & modality.eq(0)).long()


@dataclass(frozen=True)
class LiveRows:
    """Retrieval rows produced by one forward pass, plus the window each row came from."""

    rows: SensorRows
    window: torch.Tensor          # (R,) batch row that produced this retrieval row


def live_sensor_rows(
    out: dict,
    batch: dict,
    *,
    labels: torch.Tensor,
    enrolled_candidate: torch.Tensor,
    select: torch.Tensor | None = None,
    bias_weight: float = 0.0,
) -> LiveRows:
    """Flatten one encoded batch into ``SensorRows``, the deployment retrieval unit.

    A retrieval row is one (patch, sensor) pair, exactly as ``build_memory`` stores them: rows are
    taken from ``retrieval_tokens`` (sensor-isolated temporal attention, before descriptor
    conditioning and before cross-sensor mixing), so an accelerometer row does not depend on whether
    a gyroscope happened to coexist.

    ``select`` restricts the flattening to a subset of batch rows (queries, support, background),
    which is how one forward pass serves all three roles.
    """
    tokens = out.get("retrieval_tokens")
    if tokens is None:
        raise KeyError(
            "the encoder returned no retrieval_tokens; episodic training requires a "
            "use_sensor_isolated_retrieval encoder (the rows Phase B deploys)"
        )
    present = out["sensor_present"]                                  # (B, N)
    patch_pad = batch["patch_padding_mask"].to(tokens.device)        # (B, P)
    B, P, N, d = tokens.shape
    valid = patch_pad.unsqueeze(2) & present.unsqueeze(1)            # (B, P, N)

    sensor_id = batch["sensor_id"].to(tokens.device)
    channel_mask = batch["channel_mask"].to(tokens.device)
    modality = sensor_modality_codes(sensor_id, channel_mask, N)     # (B, N)
    gravity = gravity_codes(modality, batch["gravity_state"])        # (B, N)
    # A slot the encoder calls present must have a modality. If it does not, the layout tensors and
    # the fold disagree and every downstream compatibility decision for that row is unfounded.
    if bool((present & modality.lt(0)).any()):
        raise ValueError("encoder reports a present sensor slot with no live accel/gyro channel")

    if select is not None:
        keep = torch.zeros(B, dtype=torch.bool, device=tokens.device)
        keep[select.to(tokens.device)] = True
        valid = valid & keep.view(B, 1, 1)

    window_i, _patch_i, sensor_i = torch.nonzero(valid, as_tuple=True)
    feature = tokens[valid]                                          # (R, d)

    descriptor = out["descriptor"]                                   # (B, N, 384)
    row_descriptor = F.normalize(descriptor[window_i, sensor_i].float(), dim=-1)

    if bias_weight and "sensor_bias" not in batch:
        raise ValueError(
            "rank_scores was asked to blend sensor_bias but the batch carries none; "
            "either supply sensor_bias or leave the blend at zero"
        )
    row_bias = (batch["sensor_bias"].to(tokens.device)[window_i, sensor_i]
                if "sensor_bias" in batch and batch["sensor_bias"] is not None
                else feature.new_zeros((len(feature), 1)))

    labels = labels.to(tokens.device)
    enrolled_candidate = enrolled_candidate.to(tokens.device)
    dataset_ids = torch.zeros(B, dtype=torch.long, device=tokens.device)
    return LiveRows(
        rows=SensorRows(
            feature=feature,
            descriptor=row_descriptor,
            bias=row_bias,
            modality=modality[window_i, sensor_i],
            gravity=gravity[window_i, sensor_i],
            label=labels[window_i],
            dataset=dataset_ids[window_i],
            enrolled_candidate=enrolled_candidate[window_i],
            # Same quantity the mixer's co-membership channel reads at deployment, so the training
            # and inference definitions of "same recording" cannot drift apart.
            source_window=window_i,
        ),
        window=window_i,
    )


def select_rows(live: "LiveRows", index: torch.Tensor) -> "LiveRows":
    """A sub-view of a row set, with every parallel field kept in step.

    Written as one function rather than inline indexing because ``SensorRows`` has eight parallel
    fields and a hand-rolled subset that misses one silently mislabels or misbinds every row it
    touches — the class of bug that has cost this subsystem the most.
    """
    index = index.to(live.rows.feature.device)
    rows = live.rows
    # Enumerated from the dataclass rather than hand-listed. The hand-listed version silently
    # dropped `source_window` the moment that field was added — which is the exact failure this
    # function's docstring warns about, and the reason its test enumerates fields too.
    return LiveRows(
        rows=replace(rows, **{
            name: None if getattr(rows, name) is None else getattr(rows, name)[index]
            for name in rows.__dataclass_fields__
        }),
        window=live.window[index],
    )


def mixer_pool_index(scores: torch.Tensor, pool: int | None) -> torch.Tensor:
    """The evidence rows the mixer is allowed to see: the ones deployment would have retrieved.

    WHY THE MIXER IS POOLED WHEN THE VOTE IS NOT
    --------------------------------------------
    The deployed bank holds millions of rows, so at inference the mixer can only ever attend over
    the top-k that retrieval already selected. Training it over the ~520 rows of a whole episode
    would show it a token set of a different size and composition than it will ever see again, which
    is a distribution shift in its input, not merely in its output.

    The vote stays over every row, so the encoder still receives gradient through the cosine from
    all of them. Only the learned CORRECTION is restricted, and rows outside the pool simply take
    the deployed weighting unchanged. Ranking by max-over-queries is the right selector because a
    row is worth mixing if ANY query in the episode would have retrieved it.
    """
    M = scores.shape[1]
    if pool is None or pool >= M:
        return torch.arange(M, device=scores.device)
    per_row = scores.masked_fill(~torch.isfinite(scores), float("-inf")).max(dim=0).values
    return per_row.topk(pool).indices.sort().values


# ==================================================================================================
# The loss: deployment's rule, differentiated
# ==================================================================================================
def episode_logits(
    query: LiveRows,
    memory: LiveRows,
    candidate_text: torch.Tensor,        # (C, 384) L2-normalised
    label_text: torch.Tensor,            # (V, 384) L2-normalised
    *,
    n_query_windows: int,
    admissibility: torch.Tensor | None = None,
    temperature: float = VOTE_TEMPERATURE,
    bias_weight: float = 0.0,
    return_stats: bool = False,
    top_k: int | None = None,
):
    """(n_query_windows, C) nonnegative vote mass — deployment's rule, end to end differentiable.

    ``soft_vote_all`` rather than ``vote``: they share the identical adjusted score, but the top-k
    truncation deployment applies gives no gradient to unselected rows. Training over the whole
    bounded working set is the same ordering rule with credit reaching every row that participated.
    """
    scores = rank_scores(
        query.rows.feature, query.rows.bias, memory.rows,
        compatibility_mask(query.rows.modality, query.rows.gravity, memory.rows),
        bias_weight=bias_weight,
    )
    if admissibility is None:
        # Gate disabled: every comparison the HARD compatibility filter already allowed is fully
        # admissible. This is the honest identity element used by the explicit `--no-gate` ablation;
        # the default joint arm supplies a learned tensor.
        admissibility = scores.new_ones((scores.shape[0], scores.shape[1], candidate_text.shape[0]))
    if top_k is None:
        voted = soft_vote_all(
            scores, memory.rows, candidate_text, label_text, admissibility,
            temperature=temperature, return_stats=return_stats,
        )
    else:
        if return_stats:
            raise ValueError("retrieval statistics are available only for the soft training rule")
        voted = vote(
            scores, memory.rows, candidate_text, label_text, admissibility,
            top_k=top_k, temperature=temperature,
        )
    per_row, stats = voted if return_stats else (voted, None)
    merged = per_row.new_zeros((n_query_windows, per_row.shape[1]))
    merged = merged.index_add(0, query.window, per_row)
    return (merged, stats) if return_stats else merged


def mixed_vote(
    log_weights: torch.Tensor,          # (Q, M, C) or (Q, M) per-pair log-weight
    memory: "LiveRows",
    candidate_text: torch.Tensor,       # (C, 384)
    label_text: torch.Tensor,           # (V, 384)
    *,
    top_k: int | None = None,
    return_stats: bool = False,
    text_vote_matrix: torch.Tensor | None = None,
    allow_corpus_text_vote: bool = True,
):
    """(Q, C) vote mass from LEARNED evidence weights, with the semantic step left fixed.

    Identical to ``soft_vote_all`` (``top_k=None``) and to ``vote`` (``top_k=k``) except for where
    the weight comes from: there it is ``exp(cos/tau) * admissibility``, here it is whatever the
    mixer emits. Everything after — the binding mask, per-candidate top-k selection, the enrolled
    identity vote, the rectified label-text vote — is unchanged, so the readout still has NO
    per-candidate parameters and an unseen candidate is scored by the same operation as a seen one.

    Feeding ``scores/tau + log(admissibility)`` in reproduces those two functions exactly; that
    parity is asserted in the tests and is what makes the mixer arm attributable. A 2-D input is
    broadcast across candidates, which is the shape they imply with a neutral gate.

    ``top_k`` exists so validation can score the MIXER under the deployment rule. Without it the
    checkpoint-selection metric would silently fall back to the un-mixed path and be blind to the
    module being trained.
    """
    rows = memory.rows
    if log_weights.dim() == 2:
        log_weights = log_weights.unsqueeze(-1).expand(-1, -1, candidate_text.shape[0])
    Q, M, C = log_weights.shape
    if C != candidate_text.shape[0]:
        raise ValueError(
            f"log_weights carries {C} candidates but candidate_text has "
            f"{candidate_text.shape[0]}; the mixer and the vote disagree on the candidate set"
        )
    device = log_weights.device
    candidate_index = torch.arange(C, device=device).view(1, C, 1)
    bound = rows.enrolled_candidate.to(device).view(1, 1, M)
    row_label = rows.label.to(device)
    # (M, C) rectified label-text cosine; the ConSE bridge, and the only mechanism live at k=0.
    # `text_vote_matrix` replaces the FROZEN SBERT cosine with a context-refined one. It is the
    # only way the semantic geometry itself becomes learnable — every other learned quantity here
    # reweights evidence AROUND a fixed geometry. Rows with no label still vote nothing.
    row_candidate_cos = (
        F.relu(label_text[row_label.clamp_min(0)] @ candidate_text.T)
        if text_vote_matrix is None else F.relu(text_vote_matrix)
    )
    if not allow_corpus_text_vote:
        row_candidate_cos = torch.zeros_like(row_candidate_cos)
    if row_candidate_cos.dim() == 2:
        row_candidate_cos = row_candidate_cos * row_label.ge(0).unsqueeze(1).to(log_weights.dtype)
    elif row_candidate_cos.shape == (Q, M, C):
        row_candidate_cos = row_candidate_cos * row_label.ge(0).view(1, M, 1).to(log_weights.dtype)
    else:
        raise ValueError(
            f"text_vote_matrix must be {(M, C)} or {(Q, M, C)}, got "
            f"{tuple(row_candidate_cos.shape)}"
        )

    per_candidate = log_weights.permute(0, 2, 1)                        # (Q, C, M)
    allowed = bound.lt(0) | bound.eq(candidate_index)
    per_candidate = per_candidate.masked_fill(~allowed, float("-inf"))

    if top_k is None:
        selected, index = per_candidate, None
    else:
        selected, index = per_candidate.topk(min(top_k, M), dim=2)      # (Q, C, k)
    finite = torch.isfinite(selected)
    stable = selected.masked_fill(~finite, float("-inf")).flatten(1).max(1).values
    stable = torch.where(torch.isfinite(stable), stable, torch.zeros_like(stable))
    weights = torch.nan_to_num(
        torch.exp(selected - stable[:, None, None]) * finite.to(log_weights.dtype), nan=0.0
    )

    cosine_by_query = (
        row_candidate_cos.T.unsqueeze(0).expand(Q, C, M)
        if row_candidate_cos.dim() == 2 else row_candidate_cos.permute(0, 2, 1)
    )
    if index is None:
        sel_bound, sel_cos = bound.expand(Q, C, M), cosine_by_query
    else:
        sel_bound = bound.expand(Q, C, M).gather(2, index)
        sel_cos = cosine_by_query.gather(2, index)
    enrolled_vote = (sel_bound.ge(0) & sel_bound.eq(candidate_index)).to(log_weights.dtype)
    text_vote = sel_cos * sel_bound.lt(0).to(log_weights.dtype)
    result = (weights * (enrolled_vote + text_vote)).sum(dim=2)
    if not return_stats:
        return result
    flat = weights.flatten(1)
    share = flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(share * share.clamp_min(1e-12).log()).sum(dim=1)
    valid = finite.flatten(1).sum(dim=1).clamp_min(1)
    total = (weights * (enrolled_vote + text_vote)).sum().clamp_min(1e-12)
    # Same keys as ``soft_vote_all`` so the mixer arm's retrieval diagnostics line up column for
    # column with every arm already logged.
    stats = {
        "retrieval/normalized_entropy": (entropy / valid.clamp_min(2).float().log()).mean().detach(),
        "retrieval/effective_rows": entropy.exp().mean().detach(),
        "retrieval/available_rows": valid.float().mean().detach(),
        "retrieval/max_weight": share.max(dim=1).values.mean().detach(),
        "retrieval/enrolled_vote_mass_share": ((weights * enrolled_vote).sum() / total).detach(),
        "retrieval/text_vote_mass_share": ((weights * text_vote).sum() / total).detach(),
    }
    return result, stats


def episode_identity_ids(
    plan: "EpisodePlan",
    memory: "LiveRows",
    query: "LiveRows",
    n_slots: int,
    n_groups: int,
    generator: np.random.Generator,
    window_offset: int = 0,
) -> dict[str, torch.Tensor]:
    """Coreference slots and source-window groups for the mixer's identity channels.

    Slot ids are drawn fresh per episode so a slot can only ever mean "these tokens refer to the same
    thing" — never a stable label id the model could memorise across episodes. Slot 0 is reserved for
    unbound evidence. Groups are source-window co-membership and are independent of slots, because a
    row shares a slot with the candidate it is bound to and a group with its own recording.

    Both vocabularies are checked rather than wrapped. A modulo here would silently alias two
    different windows onto one group id, which reads downstream as "these came from the same
    recording" — a false identity claim that no test on the mixer itself could catch.
    """
    device = memory.rows.feature.device
    n_candidates = len(plan.candidates)
    if n_candidates > n_slots - 1:
        raise ValueError(
            f"episode has {n_candidates} candidates but only {n_slots - 1} non-null slots; "
            "raise EvidenceMixerConfig.n_slots"
        )
    perm = generator.permutation(n_slots - 1)[:n_candidates] + 1
    candidate_slot = torch.as_tensor(perm, dtype=torch.long, device=device)
    bound = memory.rows.enrolled_candidate.to(device)
    if bool(bound.ge(n_candidates).any()):
        raise ValueError(
            f"memory binds a row to candidate {int(bound.max())} but the plan lists only "
            f"{n_candidates}; the plan and the delivered batch are out of step"
        )
    evidence_slot = torch.where(bound.ge(0), candidate_slot[bound.clamp_min(0)],
                                torch.zeros_like(bound))
    evidence_group = memory.window.to(device) - int(window_offset)
    query_group = query.window.to(device) - int(window_offset)
    if bool((evidence_group < 0).any()) or bool((query_group < 0).any()):
        raise ValueError("window_offset exceeds an episode row index")
    highest = int(max(evidence_group.max().item(), query_group.max().item()))
    if highest >= n_groups:
        raise ValueError(
            f"episode window id {highest} exceeds the {n_groups}-entry group vocabulary; "
            "raise EvidenceMixerConfig.n_groups"
        )
    return {
        "candidate_slot": candidate_slot,
        "evidence_slot": evidence_slot,
        "evidence_group": evidence_group,
        "query_group": query_group,
    }


def retrieval_alignment_loss(
    scores: torch.Tensor,          # (Q, M) retrieval scores
    query_label: torch.Tensor,     # (Q,) canonical vocabulary id of each query row
    memory_label: torch.Tensor,    # (M,) canonical vocabulary id of each bank row
    label_text: torch.Tensor,      # (V, 384) L2-normalised frozen label text
    temperature: float = 0.1,
) -> torch.Tensor:
    """Teach retrieval that similar SIGNAL should mean semantically similar LABEL.

    WHY THIS EXISTS
    ---------------
    Measured on the first end-to-end run: retrieval already finds rows whose labels are nearer the
    query's true label than the bank average (+0.049, p=4e-15), and that weak preference is where
    ALL of the model's accuracy comes from — uniform weights over the retrieved set score 0.365
    against 0.271 for ignoring the query entirely. But it is weak, and within the retrieved set the
    score carries almost nothing (corr 0.028 with label relevance), because retrieval's only
    training signal is the vote gradient, diluted across a 57-effective-row average.

    This supplies the missing direct signal. The target is the frozen text similarity between the
    query's label and each bank row's label — a quantity the model never sees at inference, used
    here exactly as the episodic target is: supervision on TRAINING concepts only. What it teaches
    is a mapping from signal to semantic neighbourhood, which is the mechanism that has to
    generalize to unseen labels, not a lookup table of the labels it was trained on.

    It also, for free, teaches support retrieval: an enrolled row carries its candidate's canonical
    label, so it is the most relevant row in the bank by construction.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    with torch.no_grad():
        relevance = label_text[query_label] @ label_text[memory_label].t()
        target = torch.softmax(relevance / temperature, dim=1)
    return -(target * torch.log_softmax(scores.float(), dim=1)).sum(dim=1).mean()


def episode_cross_entropy(votes: torch.Tensor, target_slot: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over candidate slots, matching the Stage-2 gate trainer exactly.

    ``soft_vote_all`` returns nonnegative MASS, not logits, so the log is taken before the softmax
    inside cross_entropy — the same ``(votes + 1e-8).log()`` the admissibility trainer uses. Scoring
    the raw mass instead would silently change the objective's curvature.
    """
    return F.cross_entropy((votes + 1e-8).log(), target_slot)


def anchor_penalty(live: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Mean cosine distance from the frozen starting encoder's rows. Retention, not regression.

    The measured failure mode of episodic training on this system is that the optimiser suppresses
    the label-text channel and zero-shot falls below chance. This term prices drift away from the
    representation that scored 47.5 untrained, so the episodic objective must *earn* any change it
    makes. Zero weight is a legitimate arm — it measures the unmitigated effect.
    """
    return (1.0 - F.cosine_similarity(live.float(), anchor.float().detach(), dim=-1)).mean()


def macro_f1(truth: list[int], prediction: list[int]) -> float:
    scores = []
    for label in sorted(set(truth)):
        tp = sum(t == label and p == label for t, p in zip(truth, prediction))
        fp = sum(t != label and p == label for t, p in zip(truth, prediction))
        fn = sum(t == label and p != label for t, p in zip(truth, prediction))
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(scores)) if scores else 0.0


def episode_row_roles(
    plan: EpisodePlan, row_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flat-batch index tensors for the three roles, in the layout ``flat_positions`` writes."""
    n_q, n_s, n_b = plan.n_query, plan.n_support, plan.n_background
    query = torch.arange(row_offset, row_offset + n_q, dtype=torch.long)
    support = torch.arange(row_offset + n_q, row_offset + n_q + n_s, dtype=torch.long)
    background = torch.arange(
        row_offset + n_q + n_s, row_offset + n_q + n_s + n_b, dtype=torch.long,
    )
    return query, support, background


def episode_binding(plan: EpisodePlan, batch_size: int, row_offset: int = 0) -> torch.Tensor:
    """(B,) candidate slot each batch row is BOUND to; -1 for query and background rows.

    Binding is what makes a row 'enrolled evidence for candidate c' in ``soft_vote_all``: bound rows
    vote their candidate by identity and are barred from voting for any other. Background rows are
    unbound and reach a candidate only through label text — the k=0 path.
    """
    binding = torch.full((batch_size,), -1, dtype=torch.long)
    for offset, slot in enumerate(plan.support_slot):
        binding[row_offset + plan.n_query + offset] = int(slot)
    return binding


# ==================================================================================================
# The episodic memory bank
# ==================================================================================================
@dataclass(frozen=True)
class BankSpec:
    """How the memory bank is drawn. The same construction serves training and deployment.

    WHY A FIXED, SMALL, STRATIFIED BANK
    -----------------------------------
    Three properties, each load-bearing.

    FIXED SIZE. The bank is ``n_windows`` recordings whether the episode enrolled nothing or
    enrolled four examples of every candidate; support displaces background rather than adding to
    it. So retrieval always chooses from the same-size pool, its scores always mean the same thing,
    and the step cost does not depend on k.

    SMALL. 512 recordings, not the whole corpus. A bank the model can attend over is a bank whose
    rows all receive gradient, and it is the same object at deployment — where the claim is that a
    few hundred evenly-drawn training recordings are enough context, not that the user must ship a
    corpus.

    STRATIFIED, NOT UNIFORM OVER WINDOWS. Sampling windows directly lets window COUNT decide what
    the bank is made of, and this corpus is dominated by one free-living dataset. Half the draws go
    stream-first (uniform over acquisition configurations, then over the labels that stream
    carries) and half go label-first (uniform over labels, then over the streams that carry them).
    Neither margin can be made exactly uniform while the other is, so the bank splits the
    difference explicitly rather than picking one and calling it even.

    EXCLUDING THE ANSWER
    --------------------
    ``exclude_candidate_labels`` keeps rows carrying the episode's own candidate labels out of the
    bank. That is not a handicap, it is the deployment condition: the bank is drawn from TRAINING
    labels and the query carries an unseen one, so at deployment no bank row can already be named
    the answer. Training against a bank that contains the answer would teach label lookup inside a
    closed vocabulary — the failure mode this whole design exists to avoid — and would make the
    k=0 cell measure something that will not exist when it matters.
    """

    n_windows: int = 512
    stream_first_fraction: float = 0.5
    exclude_candidate_labels: bool = True

    def validate(self) -> None:
        if self.n_windows <= 0:
            raise ValueError("a bank needs at least one window")
        if not 0.0 <= self.stream_first_fraction <= 1.0:
            raise ValueError("stream_first_fraction must be in [0,1]")


def bank_index(
    stream_table: dict[int, dict[int, dict[int, np.ndarray]]],
) -> dict[str, object]:
    """Precompute the draws a bank needs, once, so sampling is integer work on small arrays."""
    streams = sorted(stream_table)
    labels_by_stream = {stream: sorted(stream_table[stream]) for stream in streams}
    streams_by_label: dict[int, list[int]] = {}
    for stream, labels in labels_by_stream.items():
        for label in labels:
            streams_by_label.setdefault(label, []).append(stream)
    return {
        "streams": np.asarray(streams, dtype=np.int64),
        "labels_by_stream": {s: np.asarray(v, dtype=np.int64)
                             for s, v in labels_by_stream.items()},
        "labels": np.asarray(sorted(streams_by_label), dtype=np.int64),
        "streams_by_label": {l: np.asarray(v, dtype=np.int64)
                             for l, v in streams_by_label.items()},
        "table": stream_table,
        # Subject keys per (stream, label), sorted once. The sampler draws subject-first tens of
        # thousands of times per training run; sorting inside the draw made that O(subjects log
        # subjects) PER WINDOW and was measured at a quarter of bank-construction time.
        "subjects_by_cell": {
            (stream, label): np.asarray(sorted(by_subject), dtype=np.int64)
            for stream, by_label in stream_table.items()
            for label, by_subject in by_label.items()
        },
    }


def sample_bank_positions(
    index: dict[str, object],
    *,
    spec: BankSpec,
    n_windows: int,
    exclude_labels: Sequence[int],
    rng: np.random.Generator,
) -> list[int]:
    """``n_windows`` dataset positions, stratified per :class:`BankSpec`.

    Draws are with replacement across cells but de-duplicated at the end and topped up, so a bank
    never carries the same window twice — a duplicate row is a vote counted twice.
    """
    if n_windows <= 0:
        return []
    table = index["table"]
    blocked = set(int(label) for label in exclude_labels) if spec.exclude_candidate_labels else set()
    # The blocked set is CONSTANT for the whole bank, so every exclusion is resolved here, once.
    # The previous version ran np.isin inside each draw — 46% of bank-construction time for
    # exactly the same distribution.
    if blocked:
        blocked_array = np.fromiter(blocked, dtype=np.int64)
        allowed_by_stream = {
            stream: options[~np.isin(options, blocked_array)]
            for stream, options in index["labels_by_stream"].items()
        }
        allowed_labels = index["labels"][~np.isin(index["labels"], blocked_array)]
    else:
        allowed_by_stream = index["labels_by_stream"]
        allowed_labels = index["labels"]
    streams = index["streams"]
    subjects_by_cell = index["subjects_by_cell"]

    def pick(options: np.ndarray) -> int:
        # rng.integers-and-index is a uniform draw with a fraction of rng.choice's overhead.
        return int(options[rng.integers(len(options))])

    def draw_window(stream: int, label: int) -> int:
        by_subject = table[stream][label]
        subject = pick(subjects_by_cell[(stream, label)])
        rows = by_subject[subject]
        return int(rows[rng.integers(len(rows))])

    def draw_stream_first() -> int | None:
        stream = pick(streams)
        options = allowed_by_stream[stream]
        if not len(options):
            return None
        return draw_window(stream, pick(options))

    def draw_label_first() -> int | None:
        if not len(allowed_labels):
            return None
        label = pick(allowed_labels)
        return draw_window(pick(index["streams_by_label"][label]), label)

    chosen: list[int] = []
    seen: set[int] = set()
    # Bounded: a bank of a few hundred rows over a corpus of hundreds of thousands of windows
    # collides rarely, but a degenerate corpus must not spin here.
    for attempt in range(n_windows * 8):
        if len(chosen) >= n_windows:
            break
        stream_first = rng.random() < spec.stream_first_fraction
        position = draw_stream_first() if stream_first else draw_label_first()
        if position is None or position in seen:
            continue
        seen.add(position)
        chosen.append(position)

    # Collisions are uncommon on the production corpus but common in small controlled corpora. Top
    # up from the exact eligible population rather than silently returning a smaller bank. This path
    # is deliberately a fallback: the primary draws retain the requested stream/label stratification.
    if len(chosen) < n_windows:
        eligible: list[int] = []
        for stream, by_label in table.items():
            for label, by_subject in by_label.items():
                if int(label) in blocked:
                    continue
                for rows in by_subject.values():
                    eligible.extend(int(value) for value in rows if int(value) not in seen)
        need = n_windows - len(chosen)
        if len(eligible) >= need:
            extra = rng.choice(np.asarray(eligible, dtype=np.int64), size=need, replace=False)
            chosen.extend(int(value) for value in extra)
    if len(chosen) != n_windows:
        raise ValueError(
            f"requested {n_windows} distinct bank windows but only drew {len(chosen)}; every "
            "remaining label is excluded or the eligible corpus is too small"
        )
    return chosen




@dataclass(frozen=True)
class BankPlan:
    """One step's bank: the background it drew, and which labels it must not contain."""

    positions: tuple[int, ...]
    excluded_labels: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.positions)


def build_bank_plan(
    index: dict[str, object],
    *,
    spec: BankSpec,
    support_windows: int,
    exclude_labels: Sequence[int],
    rng: np.random.Generator,
) -> BankPlan:
    """Background rows for one step, sized so background + support is exactly ``spec.n_windows``."""
    spec.validate()
    support_windows = int(support_windows)
    if support_windows < 0 or support_windows > spec.n_windows:
        raise ValueError(
            f"support_windows={support_windows} must lie in [0, bank size {spec.n_windows}]"
        )
    background = spec.n_windows - support_windows
    positions = sample_bank_positions(
        index, spec=spec, n_windows=background,
        exclude_labels=exclude_labels, rng=rng,
    )
    return BankPlan(positions=tuple(positions),
                    excluded_labels=tuple(sorted(set(int(v) for v in exclude_labels))))


def bank_composition(
    positions: Sequence[int], stream_ids: np.ndarray, labels: np.ndarray,
) -> dict[str, float]:
    """How even the bank actually came out. Reported, not assumed.

    A stratified sampler can still be lopsided when the corpus is: a stream carrying one label
    contributes that label every time it is drawn. Normalised entropy of 1.0 is perfectly even
    coverage; the number falling over a run means the bank is quietly narrowing.
    """
    def normalised_entropy(values: np.ndarray) -> float:
        _, counts = np.unique(values, return_counts=True)
        if len(counts) < 2:
            return 0.0
        p = counts / counts.sum()
        return float(-(p * np.log(p)).sum() / np.log(len(counts)))

    picked = np.asarray(list(positions), dtype=np.int64)
    return {
        "bank/windows": float(len(picked)),
        "bank/distinct_streams": float(len(np.unique(stream_ids[picked]))),
        "bank/distinct_labels": float(len(np.unique(labels[picked]))),
        "bank/stream_entropy": normalised_entropy(stream_ids[picked]),
        "bank/label_entropy": normalised_entropy(labels[picked]),
    }
