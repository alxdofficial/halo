from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from training.evidence.admissibility_gate import AdmissibilityGate
from training.evidence.admissible_retrieval import admissibility_from_unique
from training.evidence.train_admissibility_gate import (
    ArchiveIndex, _admissibility_from_row_values, _draw_episode, _merge_query_windows,
)


def _bank() -> dict:
    # Three labels, five executions each, and two adjacent windows per execution.
    labels, events = [], []
    for label in range(3):
        for event in range(5):
            labels.extend([label, label])
            events.extend([100 * label + event, 100 * label + event])
    n = len(labels)
    return {
        "y": torch.tensor(labels),
        "event": torch.tensor(events),
        "subj": torch.zeros(n, dtype=torch.long),
        "cfg": torch.zeros(n, dtype=torch.long),
        "cfg_names": {0: "study/watch"},
        "sensor": {
            "window": torch.arange(n).repeat_interleave(2),
        },
    }


def test_episode_support_and_query_are_distinct_recorded_executions():
    index = ArchiveIndex.from_bank(_bank())
    episode = _draw_episode(
        [0, 1, 2], index, candidate_count=3, max_support=2,
        rng=np.random.default_rng(7), fixed_support=2, full_enrollment=True,
    )
    for position, query in enumerate(episode.queries):
        support = [window for window, owner in zip(
            episode.support_windows, episode.support_positions
        ) if owner == position]
        executions = [index.execution_by_window[query]] + [
            index.execution_by_window[window] for window in support
        ]
        assert len(executions) == len(set(executions)) == 3


def test_sensor_row_lookup_keeps_requested_window_order():
    index = ArchiveIndex.from_bank(_bank())
    rows = index.sensor_rows([3, 1])
    assert rows.tolist() == [6, 7, 2, 3]


def test_execution_identity_includes_subject_when_event_ids_repeat():
    bank = _bank()
    bank["event"][:4] = 7
    bank["subj"][:2] = 0
    bank["subj"][2:4] = 1
    index = ArchiveIndex.from_bank(bank)
    assert index.execution_by_window[0] == index.execution_by_window[1]
    assert index.execution_by_window[0] != index.execution_by_window[2]


def test_partial_enrollment_never_degenerates_to_full_enrollment():
    index = ArchiveIndex.from_bank(_bank())
    episode = _draw_episode(
        [0, 1, 2], index, candidate_count=3, max_support=2,
        rng=np.random.default_rng(11), fixed_support=1, partial_enrollment=True,
    )
    assert 0 < episode.enrolled_count < len(episode.candidates)


def test_cached_evidence_admissibility_is_exact_and_keeps_gradients():
    torch.manual_seed(17)
    gate = AdmissibilityGate(rank=3)
    unique = F.normalize(torch.randn(5, 384), dim=-1)
    descriptor_id = torch.tensor([0, 2, 2, 4, 1, 0])
    candidate = F.normalize(torch.randn(4, 384), dim=-1)
    query = F.normalize(torch.randn(2, 384), dim=-1)
    full = admissibility_from_unique(gate, unique, descriptor_id, candidate, query)
    row = gate(unique, candidate).index_select(0, descriptor_id)
    cached = _admissibility_from_row_values(gate, row, query, candidate)
    assert torch.allclose(cached, full, atol=1e-7, rtol=1e-6)
    cached.sum().backward()
    assert gate.sensor_proj.weight.grad is not None
    assert gate.concept_proj.weight.grad is not None


def test_batched_query_merge_matches_individual_window_sums():
    per_sensor = torch.tensor([
        [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],
    ])
    owner = torch.tensor([0, 0, 1, 2, 2])
    merged = _merge_query_windows(per_sensor, owner, 3)
    expected = torch.stack((per_sensor[:2].sum(0), per_sensor[2], per_sensor[3:].sum(0)))
    assert torch.equal(merged, expected)
