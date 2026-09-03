"""Support-set sampling — the training curriculum, which is the paper's contribution.

WHAT AN EPISODE IS
------------------
One query recording, a candidate label roster, and K labelled *support* recordings the comparator
may compare the query against. Everything the model learns about "how to compare" comes from how
these are drawn, so this module is the method rather than plumbing around it.

THE FOUR RULES
--------------
1. **Compatibility.** In ``"compatible"`` mode every support recording shares the query's exact
   acquisition key. This is a deployment filter, not a learned quantity, and no novelty is claimed
   for it — you would not offer smartwatch examples to a pocket-phone query in a real product.
2. **Never the query, never its subject, never its execution.** Support is subject-disjoint, and
   the execution (one continuous physical capture) is the leakage unit — the same unit
   ``eval/data.py`` uses, derived the same way, because two label blocks seconds apart are not
   independent examples.
3. **Verbatim labels.** No canonicalisation beyond what the corpus already applied, no synonym
   merging, no deduplication. Two candidates may carry near-identical text; the readout handles
   that by giving them near-identical votes, which is the right answer.
4. **Ground truth present with probability p.** Above it the episode is few-shot; below it the
   answer is absent from the support set and the episode trains the zero-shot path. The two are
   interleaved in one stream, never trained as separate arms.

WHAT HAPPENS WHEN THE POOL IS TOO SMALL
---------------------------------------
K shrinks for that episode and the shrink is recorded in telemetry. Support is never padded with
incompatible rows (that would silently violate rule 1) and the episode is never skipped (that would
quietly bias the corpus toward whichever configurations happen to be well populated). If shrinking
is common, that is a finding about the corpus to report, not a bug to hide.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

from data.scripts.labels.canonical_labels import canonicalize
from data.scripts.curate.compatibility import (
    AcquisitionKey,
    are_compatible,
    is_near_miss,
    stream_key,
)
from data.scripts.eda.grid_io import discover_grids

REPO = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO / "data" / "datasets"

SamplingMode = Literal["compatible", "near_miss", "unfiltered"]

#: A-priori constants (design doc §3). They may be varied deliberately as an experiment; they are
#: never tuned against evaluation data, because there is no development split.
DEFAULT_SUPPORT = 32
DEFAULT_P_GT_PRESENT = 0.5
DEFAULT_LABEL_SUBSET = (2, 8)


def _recording_map(dataset: str) -> dict:
    """``{event_id_without_ordinal: recording_id}``, composed exactly as ``eval/data.py`` does.

    Duplicated deliberately rather than imported: ``eval.data`` pulls in the whole evaluation
    stack, and the two paths must agree on the leakage unit even if one is refactored. The
    agreement is asserted in ``tests/test_compare_sampling.py``.
    """
    recordings_path = DATASETS_DIR / dataset / "recordings.json"
    if not recordings_path.exists():
        return {}
    recordings = json.loads(recordings_path.read_text())
    events_path = DATASETS_DIR / dataset / "events.json"
    events = json.loads(events_path.read_text()) if events_path.exists() else {}
    composed: dict = {}
    for session, recording in recordings.items():
        key = f"{dataset}:{events.get(session, session)}"
        previous = composed.setdefault(key, recording)
        if previous != recording:
            raise ValueError(
                f"{dataset}: sessions sharing physical event {key!r} disagree on their recording"
            )
    return composed


def _execution_ids(dataset: str, event_ids: Sequence[str]) -> np.ndarray:
    """The leakage unit for each window: one continuous physical capture."""
    blocks = np.asarray([
        value.rsplit(":", 1)[0]
        if ":" in str(value) and str(value).rsplit(":", 1)[1].isdigit() else value
        for value in event_ids
    ], dtype=object)
    recordings = _recording_map(dataset)
    if not recordings:
        return blocks
    return np.asarray([recordings.get(block, block) for block in blocks], dtype=object)


@dataclass(frozen=True)
class Recording:
    """One window of the corpus, addressed the way an episode needs it."""

    stream_index: int
    window_index: int
    dataset: str
    stream: str
    label: str
    subject: str
    execution: str


@dataclass
class SupportCorpus:
    """Windows grouped by acquisition key, then by label, for O(1) episode draws."""

    recordings: list[Recording]
    keys: list[AcquisitionKey]                      # per stream index
    stream_names: list[tuple[str, str]]             # per stream index
    by_key: dict[AcquisitionKey, list[int]] = field(default_factory=dict)
    by_key_label: dict[tuple[AcquisitionKey, str], list[int]] = field(default_factory=dict)
    near_miss_keys: dict[AcquisitionKey, list[AcquisitionKey]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.recordings)

    def key_of(self, recording: Recording) -> AcquisitionKey:
        return self.keys[recording.stream_index]

    def summary(self) -> dict[str, object]:
        pools = {key: len(rows) for key, rows in self.by_key.items()}
        return {
            "windows": len(self.recordings),
            "streams": len(self.stream_names),
            "keys": len(pools),
            "largest_pool": max(pools.values()) if pools else 0,
            "median_pool": int(np.median(list(pools.values()))) if pools else 0,
            "keys_with_near_miss": sum(1 for v in self.near_miss_keys.values() if v),
        }


def build_support_corpus(
    datasets: Sequence[str],
    *,
    alignment: str = "native",
    max_per_stream: int | None = None,
    seed: int = 0,
    exclude_labels: Iterable[str] = ("unlabeled",),
) -> SupportCorpus:
    """Index the training grids into the structure the sampler draws from.

    Reads grid metadata only — never ``data.npy`` — so building the index is cheap and the encoder
    stays responsible for loading signal.
    """

    rng = np.random.default_rng(seed)
    banned = {str(label).lower() for label in exclude_labels}
    wanted = set(datasets)

    recordings: list[Recording] = []
    keys: list[AcquisitionKey] = []
    stream_names: list[tuple[str, str]] = []

    for ref in discover_grids(alignment):
        if ref.dataset not in wanted or ref.n_windows == 0:
            continue
        try:
            key = stream_key(ref.dataset, ref.stream)
        except KeyError:
            # A stream with no curated spec cannot be placed in a configuration, so it cannot
            # legitimately enter anyone's support set. Skip loudly rather than guess.
            print(f"[support-corpus] no acquisition key for {ref.dataset}/{ref.stream}; skipped")
            continue
        stream_index = len(stream_names)
        stream_names.append((ref.dataset, ref.stream))
        keys.append(key)

        executions = _execution_ids(ref.dataset, ref.event_ids)
        chosen = np.arange(ref.n_windows)
        if max_per_stream is not None and ref.n_windows > max_per_stream:
            chosen = np.sort(rng.choice(ref.n_windows, size=max_per_stream, replace=False))
        for window in chosen:
            label = canonicalize(ref.labels[int(window)])
            if str(label).lower() in banned:
                continue
            recordings.append(Recording(
                stream_index=stream_index,
                window_index=int(window),
                dataset=ref.dataset,
                stream=ref.stream,
                label=label,
                subject=str(ref.subjects[int(window)]),
                execution=str(executions[int(window)]),
            ))

    corpus = SupportCorpus(recordings=recordings, keys=keys, stream_names=stream_names)
    for index, recording in enumerate(recordings):
        key = keys[recording.stream_index]
        corpus.by_key.setdefault(key, []).append(index)
        corpus.by_key_label.setdefault((key, recording.label), []).append(index)
    distinct = list(corpus.by_key)
    for key in distinct:
        corpus.near_miss_keys[key] = [
            other for other in distinct if is_near_miss(key, other)
        ]
    return corpus


@dataclass(frozen=True)
class Episode:
    """One training episode. Indices address ``SupportCorpus.recordings``."""

    query: int
    support: tuple[int, ...]
    support_candidate: tuple[int, ...]   # candidate slot each support row is bound to
    candidates: tuple[str, ...]          # verbatim label strings
    gt_slot: int | None                  # index into candidates, or None for a zero-shot episode
    mode: SamplingMode
    requested_support: int
    shrunk: bool

    @property
    def is_zero_shot(self) -> bool:
        return self.gt_slot is None


def _pool_for(corpus: SupportCorpus, key: AcquisitionKey, mode: SamplingMode) -> list[int]:
    if mode == "compatible":
        return corpus.by_key.get(key, [])
    if mode == "near_miss":
        rows: list[int] = []
        for other in corpus.near_miss_keys.get(key, []):
            rows.extend(corpus.by_key.get(other, []))
        return rows
    if mode == "unfiltered":
        return list(range(len(corpus.recordings)))
    raise ValueError(f"unknown sampling mode {mode!r}")


def draw_episode(
    corpus: SupportCorpus,
    rng: np.random.Generator,
    *,
    support_size: int = DEFAULT_SUPPORT,
    p_gt_present: float = DEFAULT_P_GT_PRESENT,
    label_subset: tuple[int, int] = DEFAULT_LABEL_SUBSET,
    mode: SamplingMode = "compatible",
    query_index: int | None = None,
) -> Episode | None:
    """Draw one episode, or ``None`` when the query admits no usable support at all.

    ``None`` is returned only when the pool cannot supply a single admissible row — an isolated
    configuration. The caller counts these; they are a corpus finding, not an error to swallow.
    """

    if not 0.0 <= p_gt_present <= 1.0:
        raise ValueError("p_gt_present must be in [0, 1]")
    low, high = label_subset
    if low < 2 or high < low:
        raise ValueError("label_subset must be (low >= 2, high >= low)")

    query_index = int(rng.integers(len(corpus))) if query_index is None else int(query_index)
    query = corpus.recordings[query_index]
    key = corpus.key_of(query)

    admissible = [
        index for index in _pool_for(corpus, key, mode)
        if corpus.recordings[index].subject != query.subject
        and corpus.recordings[index].execution != query.execution
        and index != query_index
    ]
    if not admissible:
        return None

    by_label: dict[str, list[int]] = defaultdict(list)
    for index in admissible:
        by_label[corpus.recordings[index].label].append(index)

    available = sorted(by_label)
    gt_available = query.label in by_label
    want_gt = bool(rng.random() < p_gt_present) and gt_available

    others = [label for label in available if label != query.label]
    n_labels = int(rng.integers(low, high + 1))
    n_others = max(0, n_labels - (1 if want_gt else 0))
    if n_others > len(others):
        n_others = len(others)
    picked = list(rng.choice(others, size=n_others, replace=False)) if n_others else []
    if want_gt:
        picked.append(query.label)
    if len(picked) < 2:
        # A single candidate is not a decision. Widen only within the same admissible pool.
        extra = [label for label in available if label not in picked]
        if not extra:
            return None
        picked.append(str(rng.choice(extra)))
    rng.shuffle(picked)
    candidates = tuple(str(label) for label in picked)
    gt_slot = candidates.index(query.label) if want_gt else None

    # Round-robin across candidates so support is label-balanced to within one row. An episode that
    # over-represents one candidate teaches a prior over candidates, which is exactly the closed-
    # vocabulary habit this design is trying to avoid.
    remaining = {label: list(by_label[label]) for label in candidates}
    for rows in remaining.values():
        rng.shuffle(rows)
    support: list[int] = []
    support_candidate: list[int] = []
    while len(support) < support_size:
        progressed = False
        for slot, label in enumerate(candidates):
            if len(support) >= support_size:
                break
            rows = remaining[label]
            if not rows:
                continue
            support.append(rows.pop())
            support_candidate.append(slot)
            progressed = True
        if not progressed:
            break

    if not support:
        return None
    return Episode(
        query=query_index,
        support=tuple(support),
        support_candidate=tuple(support_candidate),
        candidates=candidates,
        gt_slot=gt_slot,
        mode=mode,
        requested_support=int(support_size),
        shrunk=len(support) < support_size,
    )


def draw_batch(
    corpus: SupportCorpus,
    rng: np.random.Generator,
    *,
    batch_size: int,
    max_attempts_per_episode: int = 32,
    **kwargs,
) -> tuple[list[Episode], dict[str, float]]:
    """Draw ``batch_size`` episodes plus the telemetry that makes the draw auditable."""

    episodes: list[Episode] = []
    attempts = 0
    unusable = 0
    while len(episodes) < batch_size and attempts < batch_size * max_attempts_per_episode:
        attempts += 1
        episode = draw_episode(corpus, rng, **kwargs)
        if episode is None:
            unusable += 1
            continue
        episodes.append(episode)
    if not episodes:
        raise RuntimeError(
            "no usable episode in "
            f"{attempts} attempts — the corpus has no admissible support under mode "
            f"{kwargs.get('mode', 'compatible')!r}"
        )
    zero_shot = sum(1 for episode in episodes if episode.is_zero_shot)
    telemetry = {
        "sampler/realised_gt_rate": 1.0 - zero_shot / len(episodes),
        "sampler/shrunk_episode_fraction": sum(e.shrunk for e in episodes) / len(episodes),
        "sampler/mean_support_size": float(np.mean([len(e.support) for e in episodes])),
        "sampler/mean_candidate_count": float(np.mean([len(e.candidates) for e in episodes])),
        "sampler/unusable_query_fraction": unusable / max(attempts, 1),
    }
    return episodes, telemetry
