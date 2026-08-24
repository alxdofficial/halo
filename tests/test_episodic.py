"""Native end-to-end episodic training: episode construction, row layout, and gradient flow.

These cover the properties that make the episodic arm's numbers mean anything. Each one guards a
failure that is silent at runtime: a support window sharing its query's subject turns enrollment
into same-person retrieval; a plan/batch desync mislabels every row; a sensor slot with the wrong
modality code authorises physically meaningless comparisons the compatibility filter exists to
forbid; and an episode with no memory rows contributes a constant loss.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from training.evidence.admissibility_gate import AdmissibilityGate
from training.evidence.admissible_retrieval import admissibility_from_gate
from training.tokenizer.pretrain_episodic import support_count_grid
from training.tokenizer.episodic import (
    EpisodePlan,
    EpisodeSpec,
    GroupedEpisodicBatchSampler,
    EpisodicCollate,
    build_episode_plans,
    eligible_labels,
    episode_binding,
    episode_cross_entropy,
    episode_logits,
    episode_row_roles,
    gravity_codes,
    label_window_table,
    live_recording_rows,
    live_sensor_rows,
    matched_support_variants,
    sensor_modality_codes,
    select_rows,
)


class _Key:
    """Minimal stand-in for pretrain_data.WindowKey (only label_id is read by the table builder)."""

    def __init__(self, label_id: int):
        self.label_id = label_id


def _synthetic_table(n_labels: int = 8, n_subjects: int = 4, per_cell: int = 6):
    """label -> subject -> positions, with every label present for every subject."""
    keys, subjects = [], []
    for label in range(n_labels):
        for subject in range(n_subjects):
            for _ in range(per_cell):
                keys.append(_Key(label))
                subjects.append(subject)
    return label_window_table(keys, np.asarray(subjects, dtype=np.int64)), keys, subjects


def _spec(**kwargs) -> EpisodeSpec:
    base = dict(candidate_counts=(2, 4), queries_per_candidate=2, max_support=2,
                background_windows=6)
    base.update(kwargs)
    return EpisodeSpec(**base)


# --------------------------------------------------------------------------------------------
# Episode construction
# --------------------------------------------------------------------------------------------
def test_support_and_query_never_share_a_subject():
    """Enrollment must be a cross-subject task, or it measures the subject-leakage trap instead."""
    table, keys, subjects = _synthetic_table()
    subjects = np.asarray(subjects)
    spec = _spec()
    plans = build_episode_plans(table, eligible_labels(table, spec), n_episodes=40,
                                spec=spec, seed=3)
    checked = 0
    for plan in plans:
        for slot in range(len(plan.candidates)):
            query_subjects = {
                int(subjects[position])
                for position, s in zip(plan.query_positions, plan.query_slot) if s == slot
            }
            support_subjects = {
                int(subjects[position])
                for position, s in zip(plan.support_positions, plan.support_slot) if s == slot
            }
            assert not (query_subjects & support_subjects)
            checked += bool(support_subjects)
    assert checked > 0, "no episode enrolled anything — the assertion proved nothing"


def test_background_labels_are_outside_the_candidate_set():
    """Background rows are the ConSE bridge; a candidate's own window there would leak the answer."""
    table, keys, _ = _synthetic_table()
    spec = _spec()
    plans = build_episode_plans(table, eligible_labels(table, spec), n_episodes=25,
                                spec=spec, seed=11)
    for plan in plans:
        candidates = set(plan.candidates)
        assert plan.n_background == spec.background_windows
        for position in plan.background_positions:
            assert keys[position].label_id not in candidates


def test_support_schedule_pins_k_for_validation():
    """Validation must report the same k cells every run, k=0 always among them."""
    table, _, _ = _synthetic_table()
    spec = _spec()
    schedule = (0, 1, 2)
    plans = build_episode_plans(table, eligible_labels(table, spec), n_episodes=12,
                                spec=spec, seed=5, support_schedule=schedule)
    assert [p.support_k for p in plans] == list(schedule) * 4
    zero_shot = [p for p in plans if p.support_k == 0]
    assert zero_shot and all(p.n_support == 0 for p in zero_shot)
    # A k=0 episode still has memory: the background rows are its only path to a prediction.
    assert all(p.n_background > 0 for p in zero_shot)


def test_specialized_phase_b_support_schedules_do_not_mix_regimes():
    assert support_count_grid(16, "zero-shot") == (0,)
    assert support_count_grid(16, "enrollment") == (1, 2, 4, 8, 16)
    assert support_count_grid(16, "unified") == (0, 1, 2, 4, 8, 16)
    with pytest.raises(ValueError, match="requires max_support"):
        support_count_grid(0, "enrollment")


def test_candidate_schedule_groups_shapes_without_sharing_episode_identity():
    from training.tokenizer.pretrain_episodic import grouped_candidate_schedule

    schedule = grouped_candidate_schedule(steps=12, episodes_per_step=4,
                                          candidate_counts=(8, 16, 32, 64), seed=7)
    grouped = schedule.reshape(12, 4)
    assert all(len(set(row.tolist())) == 1 for row in grouped)
    assert {value: int(np.sum(schedule == value)) for value in (8, 16, 32, 64)} == {
        8: 12, 16: 12, 32: 12, 64: 12,
    }


def test_requested_support_count_is_not_silently_truncated():
    table, _, _ = _synthetic_table(per_cell=2)
    spec = _spec(max_support=8)
    plans = build_episode_plans(
        table, eligible_labels(table, spec), n_episodes=8, spec=spec, seed=13,
        support_schedule=(8,),
    )
    assert all(plan.support_k == 8 for plan in plans)
    assert all(plan.n_support == 0 for plan in plans)
    assert all(not plan.enrolled_slots for plan in plans)


def test_matched_k_curve_changes_only_nested_support():
    table, _, _ = _synthetic_table()
    spec = _spec(max_support=2)
    base = build_episode_plans(
        table, eligible_labels(table, spec), n_episodes=1, spec=spec, seed=17,
        support_schedule=(2,),
    )[0]
    variants = matched_support_variants(base, (0, 1, 2))
    assert [plan.support_k for plan in variants] == [0, 1, 2]
    assert all(plan.candidates == base.candidates for plan in variants)
    assert all(plan.query_positions == base.query_positions for plan in variants)
    assert all(plan.background_positions == base.background_positions for plan in variants)
    one, two = variants[1], variants[2]
    for slot in base.enrolled_slots:
        one_rows = [p for p, s in zip(one.support_positions, one.support_slot) if s == slot]
        two_rows = [p for p, s in zip(two.support_positions, two.support_slot) if s == slot]
        assert one_rows == two_rows[:1]


def test_random_alias_episodes_enroll_every_candidate_and_never_create_alias_k0():
    table, _, _ = _synthetic_table()
    spec = _spec(alias_episode_fraction=1.0)
    plans = build_episode_plans(
        table, eligible_labels(table, spec), n_episodes=12, spec=spec, seed=71,
        support_schedule=(2,),
    )
    assert all(plan.label_mode == "random_alias" for plan in plans)
    assert all(set(plan.enrolled_slots) == set(range(len(plan.candidates))) for plan in plans)
    variants = matched_support_variants(plans[0], (0, 1, 2))
    assert variants[0].label_mode == "coherent"
    assert all(plan.label_mode == "random_alias" for plan in variants[1:])


def test_queries_and_support_are_distinct_physical_executions():
    table, _, _ = _synthetic_table(per_cell=8)
    # Two adjacent windows belong to the same physical execution. The episode must choose execution
    # identities first rather than satisfying q=2 or k=2 with two crops of one recording.
    n_rows = sum(len(rows) for by_subject in table.values() for rows in by_subject.values())
    execution_ids = np.arange(n_rows, dtype=np.int64) // 2
    spec = _spec()
    pool = eligible_labels(table, spec, execution_ids=execution_ids)
    plans = build_episode_plans(
        table, pool, n_episodes=20, spec=spec, seed=19, support_schedule=(2,),
        execution_ids=execution_ids,
    )
    for plan in plans:
        for slot in range(len(plan.candidates)):
            query = [p for p, s in zip(plan.query_positions, plan.query_slot) if s == slot]
            support = [p for p, s in zip(plan.support_positions, plan.support_slot) if s == slot]
            assert len(np.unique(execution_ids[query])) == len(query)
            assert len(np.unique(execution_ids[support])) == len(support)


def test_plans_are_deterministic_for_a_seed():
    table, _, _ = _synthetic_table()
    spec = _spec()
    pool = eligible_labels(table, spec)
    a = build_episode_plans(table, pool, n_episodes=8, spec=spec, seed=42)
    b = build_episode_plans(table, pool, n_episodes=8, spec=spec, seed=42)
    assert a == b


def test_large_candidate_roster_does_not_multiply_query_encoder_work():
    """C is the decision roster; only the configured active labels need query recordings."""
    table, _, _ = _synthetic_table(n_labels=70, n_subjects=3, per_cell=6)
    spec = _spec(
        candidate_counts=(64,), query_labels_per_episode=4, queries_per_candidate=3,
        max_support=0, background_windows=1,
    )
    # Simulate one label-poor acquisition stream: it can provide four active truths even though the
    # episode asks the classifier to reject sixty additional global-label distractors.
    query_pool = [0, 1, 2, 3]
    query_table = {label: table[label] for label in query_pool}
    plan = build_episode_plans(
        table, list(range(70)), n_episodes=1, spec=spec, seed=23,
        query_pool=query_pool, query_table=query_table,
    )[0]
    assert len(plan.candidates) == 64
    assert len(set(plan.query_slot)) == 4
    assert plan.n_query == 4 * 3
    assert {plan.candidates[slot] for slot in plan.query_slot}.issubset(query_pool)


def test_large_roster_partial_enrollment_respects_fixed_memory_capacity():
    """Large C and k remain legal by capping enrolled labels, never by truncating k."""
    table, _, _ = _synthetic_table(n_labels=70, n_subjects=4, per_cell=20)
    spec = _spec(
        candidate_counts=(64,), query_labels_per_episode=4, queries_per_candidate=2,
        max_support=16, background_windows=1, max_memory_windows=512,
        alias_episode_fraction=0.0,
    )
    plans = build_episode_plans(
        table, list(range(70)), n_episodes=20, spec=spec, seed=31,
        support_schedule=(16,),
    )
    assert all(plan.n_support <= 512 for plan in plans)
    assert all(len(plan.enrolled_slots) <= 32 for plan in plans)
    for plan in plans:
        counts = [plan.support_slot.count(slot) for slot in plan.enrolled_slots]
        assert counts and set(counts) == {16}
        assert set(plan.enrolled_slots) & set(plan.query_slot)
        assert set(plan.enrolled_slots) - set(plan.query_slot)


def test_random_alias_rejects_a_roster_that_cannot_be_fully_enrolled():
    table, _, _ = _synthetic_table(n_labels=70, n_subjects=4, per_cell=20)
    spec = _spec(
        candidate_counts=(64,), query_labels_per_episode=4, max_support=16,
        background_windows=1, max_memory_windows=512, alias_episode_fraction=1.0,
    )
    with pytest.raises(ValueError, match="every candidate"):
        build_episode_plans(
            table, list(range(70)), n_episodes=1, spec=spec, seed=37,
            support_schedule=(16,),
        )


def test_pool_must_reserve_a_label_for_background():
    """A pool exactly the size of the candidate set yields no background — and no k=0 path."""
    table, _, _ = _synthetic_table(n_labels=4)
    spec = _spec(candidate_counts=(4,), background_windows=6)
    with pytest.raises(ValueError, match="reserved for background"):
        build_episode_plans(table, eligible_labels(table, spec), n_episodes=1, spec=spec, seed=0)


def test_labels_without_a_second_subject_are_ineligible_when_k_may_be_positive():
    table, _, _ = _synthetic_table(n_labels=3, n_subjects=1)
    assert eligible_labels(table, _spec(max_support=2)) == []
    assert eligible_labels(table, _spec(max_support=0)) == [0, 1, 2]


def test_flat_layout_matches_roles_and_binding():
    """Roles are decoded positionally, so the layout contract has to be exact."""
    table, keys, _ = _synthetic_table()
    spec = _spec()
    plan = build_episode_plans(table, eligible_labels(table, spec), n_episodes=1,
                               spec=spec, seed=1, support_schedule=(2,))[0]
    flat = plan.flat_positions()
    assert len(flat) == plan.n_query + plan.n_support + plan.n_background
    query, support, background = episode_row_roles(plan)
    assert flat[:plan.n_query] == list(plan.query_positions)
    assert [flat[i] for i in support.tolist()] == list(plan.support_positions)
    assert [flat[i] for i in background.tolist()] == list(plan.background_positions)

    binding = episode_binding(plan, len(flat))
    assert bool((binding[query] == -1).all()), "queries must never be bound to a candidate"
    assert bool((binding[background] == -1).all()), "background rows vote only through label text"
    assert binding[support].tolist() == list(plan.support_slot)

    expected = plan.expected_labels()
    assert [keys[p].label_id for p in flat[:len(expected)]] == expected


def test_grouped_sampler_concatenates_plans_without_merging_episode_semantics():
    table, _, _ = _synthetic_table()
    spec = _spec()
    plans = build_episode_plans(
        table, eligible_labels(table, spec), n_episodes=4, spec=spec, seed=29,
    )
    batches = list(GroupedEpisodicBatchSampler(plans, episodes_per_batch=2))
    assert batches == [
        plans[0].flat_positions() + plans[1].flat_positions(),
        plans[2].flat_positions() + plans[3].flat_positions(),
    ]
    offset = len(plans[0].flat_positions())
    query, support, background = episode_row_roles(plans[1], row_offset=offset)
    binding = episode_binding(plans[1], len(batches[0]), row_offset=offset)
    assert query.min().item() == offset
    assert binding[support].tolist() == list(plans[1].support_slot)
    assert bool((binding[query] == -1).all())
    assert bool((binding[background] == -1).all())


# --------------------------------------------------------------------------------------------
# Row metadata
# --------------------------------------------------------------------------------------------
def test_sensor_modality_codes_for_both_and_single_modality_streams():
    sensor_id = torch.tensor([[0, 0, 0, 1, 1, 1],      # accel + gyro
                              [0, 0, 0, 0, 0, 0]])     # accel only (gyro channels dead)
    channel_mask = torch.tensor([[True] * 6,
                                 [True, True, True, False, False, False]])
    codes = sensor_modality_codes(sensor_id, channel_mask, n_sensors=2)
    assert codes[0].tolist() == [0, 1]
    assert codes[1].tolist() == [0, -1], "an absent modality must not claim a slot"


def test_sensor_modality_codes_reject_a_slot_carrying_both_modalities():
    sensor_id = torch.zeros(1, 6, dtype=torch.long)     # every channel folded into slot 0
    channel_mask = torch.ones(1, 6, dtype=torch.bool)
    with pytest.raises(ValueError, match="both accelerometer and gyroscope"):
        sensor_modality_codes(sensor_id, channel_mask, n_sensors=1)


def test_gravity_code_never_marks_a_gyroscope():
    """Only accelerometers carry a gravity convention; splitting gyros on it would be an artifact."""
    modality = torch.tensor([[0, 1], [0, 1]])
    codes = gravity_codes(modality, ["removed", "present"])
    assert codes.tolist() == [[1, 0], [0, 0]]


def test_partial_axis_triad_still_counts_as_one_sensor():
    """A two-axis accelerometer is an accelerometer — SensorFold handles the dead axis."""
    sensor_id = torch.tensor([[0, 0, 0, 1, 1, 1]])
    channel_mask = torch.tensor([[True, False, True, False, False, False]])
    assert sensor_modality_codes(sensor_id, channel_mask, n_sensors=2)[0].tolist() == [0, -1]


# --------------------------------------------------------------------------------------------
# Live rows + loss
# --------------------------------------------------------------------------------------------
def _fake_encoder_output(B=4, P=3, N=2, d=8, requires_grad=False):
    tokens = torch.randn(B, P, N, d, requires_grad=requires_grad)
    pooled = tokens.mean(dim=(1, 2))
    return {
        "retrieval_tokens": tokens,
        "pooled": pooled,
        "sensor_present": torch.ones(B, N, dtype=torch.bool),
        "descriptor": torch.randn(B, N, 384),
    }, {
        "patch_padding_mask": torch.ones(B, P, dtype=torch.bool),
        "sensor_id": torch.tensor([[0, 0, 0, 1, 1, 1]] * B),
        "channel_mask": torch.ones(B, 6, dtype=torch.bool),
        "gravity_state": ["present"] * B,
    }


def test_live_rows_count_valid_patch_sensor_pairs_only():
    out, batch = _fake_encoder_output(B=4, P=3, N=2)
    batch["patch_padding_mask"][2, 2] = False          # one padded patch
    out["sensor_present"][3, 1] = False                # one absent sensor
    live = live_sensor_rows(
        out, batch, labels=torch.zeros(4, dtype=torch.long),
        enrolled_candidate=torch.full((4,), -1),
    )
    expected = 4 * 3 * 2 - 2 - 3                        # padded patch (2 sensors), absent sensor (3 patches)
    assert live.rows.feature.shape[0] == expected
    assert live.window.max() < 4


def test_window_granularity_emits_one_row_per_present_sensor():
    out, batch = _fake_encoder_output(B=4, P=1, N=2)
    out["retrieval_window_rows"] = True
    # The source batch can still carry six patch masks; window retrieval must not duplicate rows.
    batch["patch_padding_mask"] = torch.ones(4, 6, dtype=torch.bool)
    out["sensor_present"][3, 1] = False
    live = live_sensor_rows(
        out, batch, labels=torch.zeros(4, dtype=torch.long),
        enrolled_candidate=torch.full((4,), -1),
    )
    assert live.rows.feature.shape[0] == 4 * 2 - 1


def test_live_rows_require_a_sensor_isolated_encoder():
    out, batch = _fake_encoder_output()
    out.pop("retrieval_tokens")
    with pytest.raises(KeyError, match="retrieval_tokens"):
        live_sensor_rows(out, batch, labels=torch.zeros(4, dtype=torch.long),
                         enrolled_candidate=torch.full((4,), -1))


def test_selection_restricts_rows_to_the_requested_batch_positions():
    out, batch = _fake_encoder_output(B=4, P=3, N=2)
    live = live_sensor_rows(
        out, batch, labels=torch.zeros(4, dtype=torch.long),
        enrolled_candidate=torch.full((4,), -1), select=torch.tensor([1, 2]),
    )
    assert sorted(set(live.window.tolist())) == [1, 2]
    assert live.rows.feature.shape[0] == 2 * 3 * 2


def test_shared_live_batch_is_exactly_the_per_episode_row_materialization():
    from training.tokenizer.pretrain_episodic import prepare_live_batch

    out, batch = _fake_encoder_output(B=8, P=2, N=2)
    plans = [
        EpisodePlan((0, 1), (10, 11), (0, 1), (12,), (0,), (13,), 1, (0,)),
        EpisodePlan((2, 3), (20, 21), (0, 1), (22,), (1,), (23,), 1, (1,)),
    ]
    batch["labels"] = torch.tensor([0, 1, 0, 4, 2, 3, 3, 5])
    shared, labels = prepare_live_batch(out, batch, plans, np.asarray([0, 4]), torch.device("cpu"))

    for plan, offset in zip(plans, (0, 4)):
        query_i, support_i, background_i = episode_row_roles(plan, row_offset=offset)
        binding = episode_binding(plan, len(labels), row_offset=offset)
        expected_query = live_recording_rows(
            out, batch, labels=labels, enrolled_candidate=binding, select=query_i,
        )
        expected_memory = live_recording_rows(
            out, batch, labels=labels, enrolled_candidate=binding,
            select=torch.cat([support_i, background_i]),
        )
        query_end = offset + plan.n_query
        episode_end = offset + len(plan.flat_positions())
        actual_query = select_rows(shared, torch.nonzero(
            shared.window.ge(offset) & shared.window.lt(query_end), as_tuple=True,
        )[0])
        actual_memory = select_rows(shared, torch.nonzero(
            shared.window.ge(query_end) & shared.window.lt(episode_end), as_tuple=True,
        )[0])
        for actual, expected in ((actual_query, expected_query), (actual_memory, expected_memory)):
            assert torch.equal(actual.window, expected.window)
            for name in actual.rows.__dataclass_fields__:
                assert torch.equal(getattr(actual.rows, name), getattr(expected.rows, name))


def test_enrolled_evidence_wins_when_it_matches_the_query():
    """An enrolled row identical to the query must carry its bound candidate. The identity vote."""
    torch.manual_seed(0)
    d, C = 8, 2
    signal = torch.randn(1, 1, 1, d)
    out = {
        "retrieval_tokens": torch.cat([signal, signal, torch.randn(1, 1, 1, d)]),
        "sensor_present": torch.ones(3, 1, dtype=torch.bool),
        "descriptor": torch.randn(3, 1, 384),
    }
    batch = {
        "patch_padding_mask": torch.ones(3, 1, dtype=torch.bool),
        "sensor_id": torch.tensor([[0, 0, 0, 0, 0, 0]] * 3),
        "channel_mask": torch.tensor([[True, True, True, False, False, False]] * 3),
        "gravity_state": ["present"] * 3,
    }
    labels = torch.tensor([0, 0, 1])
    binding = torch.tensor([-1, 0, -1])                # row 1 enrolled for candidate 0
    query = live_sensor_rows(out, batch, labels=labels, enrolled_candidate=binding,
                             select=torch.tensor([0]))
    memory = live_sensor_rows(out, batch, labels=labels, enrolled_candidate=binding,
                              select=torch.tensor([1, 2]))
    label_text = torch.nn.functional.normalize(torch.randn(2, 384), dim=-1)
    votes = episode_logits(query, memory, label_text[:C], label_text, n_query_windows=3)
    assert int(votes[0].argmax()) == 0


def test_zero_shot_episode_predicts_through_label_text_alone():
    """With nothing enrolled, the candidate whose TEXT matches the nearest corpus row must win."""
    torch.manual_seed(0)
    d = 8
    signal = torch.randn(1, 1, 1, d)
    out = {
        "retrieval_tokens": torch.cat([signal, signal, torch.randn(1, 1, 1, d) * 5]),
        "sensor_present": torch.ones(3, 1, dtype=torch.bool),
        "descriptor": torch.randn(3, 1, 384),
    }
    batch = {
        "patch_padding_mask": torch.ones(3, 1, dtype=torch.bool),
        "sensor_id": torch.zeros(3, 6, dtype=torch.long),
        "channel_mask": torch.tensor([[True, True, True, False, False, False]] * 3),
        "gravity_state": ["present"] * 3,
    }
    # Corpus row 1 carries vocabulary label 2; its text is aligned with candidate 0's text.
    labels = torch.tensor([0, 2, 3])
    binding = torch.full((3,), -1)
    query = live_sensor_rows(out, batch, labels=labels, enrolled_candidate=binding,
                             select=torch.tensor([0]))
    memory = live_sensor_rows(out, batch, labels=labels, enrolled_candidate=binding,
                              select=torch.tensor([1, 2]))
    label_text = torch.zeros(4, 384)
    label_text[0, 0] = label_text[2, 0] = 1.0          # candidate 0 == corpus label 2
    label_text[1, 1] = label_text[3, 1] = 1.0
    votes = episode_logits(query, memory, label_text[:2], label_text, n_query_windows=3)
    assert int(votes[0].argmax()) == 0
    assert float(votes[0, 0]) > 0.0, "the k=0 text-vote path produced no evidence at all"


def test_loss_is_differentiable_end_to_end_to_the_encoder_output():
    """The whole point: gradient must reach the representation, not stop at the scoring rule."""
    out, batch = _fake_encoder_output(B=6, P=2, N=2, d=8, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1, 2, 3])
    binding = torch.tensor([-1, -1, 0, 1, -1, -1])
    query = live_sensor_rows(out, batch, labels=labels, enrolled_candidate=binding,
                             select=torch.tensor([0, 1]))
    memory = live_sensor_rows(out, batch, labels=labels, enrolled_candidate=binding,
                              select=torch.tensor([2, 3, 4, 5]))
    label_text = torch.nn.functional.normalize(torch.randn(4, 384), dim=-1)
    votes = episode_logits(query, memory, label_text[:2], label_text, n_query_windows=6)
    loss = episode_cross_entropy(votes[:2], torch.tensor([0, 1]))
    loss.backward()
    grad = out["retrieval_tokens"].grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.norm()) > 0
    # Support rows must receive gradient too, or enrollment is never learned.
    assert float(grad[2:4].norm()) > 0
    # ...and so must background rows, which carry the zero-shot text-vote path.
    assert float(grad[4:6].norm()) > 0


def test_joint_admissibility_gate_receives_finite_gradient():
    out, batch = _fake_encoder_output(B=6, P=2, N=2, d=8, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1, 2, 3])
    binding = torch.tensor([-1, -1, 0, 1, -1, -1])
    query = live_sensor_rows(out, batch, labels=labels, enrolled_candidate=binding,
                             select=torch.tensor([0, 1]))
    memory = live_sensor_rows(out, batch, labels=labels, enrolled_candidate=binding,
                              select=torch.tensor([2, 3, 4, 5]))
    label_text = torch.nn.functional.normalize(torch.randn(4, 384), dim=-1)
    gate = AdmissibilityGate(rank=8)
    admissibility = admissibility_from_gate(gate, memory.rows, label_text[:2],
                                            query.rows.descriptor)
    votes = episode_logits(
        query, memory, label_text[:2], label_text, n_query_windows=6,
        admissibility=admissibility,
    )
    episode_cross_entropy(votes[:2], torch.tensor([0, 1])).backward()
    gradients = [parameter.grad for parameter in gate.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.norm()) for gradient in gradients) > 0


def test_hard_deployment_vote_uses_same_inputs_and_shape_as_soft_rule():
    out, batch = _fake_encoder_output(B=6, P=2, N=2, d=8)
    labels = torch.tensor([0, 1, 0, 1, 2, 3])
    binding = torch.tensor([-1, -1, 0, 1, -1, -1])
    query = live_sensor_rows(out, batch, labels=labels, enrolled_candidate=binding,
                             select=torch.tensor([0, 1]))
    memory = live_sensor_rows(out, batch, labels=labels, enrolled_candidate=binding,
                              select=torch.tensor([2, 3, 4, 5]))
    label_text = torch.nn.functional.normalize(torch.randn(4, 384), dim=-1)
    soft = episode_logits(query, memory, label_text[:2], label_text, n_query_windows=6)
    hard = episode_logits(
        query, memory, label_text[:2], label_text, n_query_windows=6, top_k=64,
    )
    assert soft.shape == hard.shape == (6, 2)
    assert torch.isfinite(soft).all() and torch.isfinite(hard).all()


def test_episodic_collate_adds_gravity_without_touching_the_base_output():
    """The wrapper must be strictly additive: every existing Phase-A key survives unchanged."""
    def base(batch):
        return {"labels": torch.tensor([item["label_id"] for item in batch]), "marker": "kept"}

    items = [{"label_id": 3, "gravity_state": "removed"},
             {"label_id": 4, "gravity_state": "present"}]
    out = EpisodicCollate(base)(items)
    assert out["marker"] == "kept"
    assert out["labels"].tolist() == [3, 4]
    assert out["gravity_state"] == ["removed", "present"]


# --------------------------------------------------------------------------------------------
# Acquisition-provenance control
# --------------------------------------------------------------------------------------------
def _streamed_table(n_labels=6, n_streams=3, n_subjects=3, per_cell=4):
    """Every label recorded on every stream by every subject, so both modes are constructible."""
    keys, subjects, streams = [], [], []
    for label in range(n_labels):
        for stream in range(n_streams):
            for subject in range(n_subjects):
                for _ in range(per_cell):
                    keys.append(_Key(label))
                    # Subjects are global: a person is not shared across streams here, which is the
                    # harder case (subject disjointness alone cannot imply stream disjointness).
                    subjects.append(stream * n_subjects + subject)
                    streams.append(stream)
    return (keys, np.asarray(subjects, dtype=np.int64), np.asarray(streams, dtype=np.int64))


def test_stream_disjoint_support_never_comes_from_the_query_stream():
    from training.tokenizer.episodic import provenance_lift

    keys, subjects, streams = _streamed_table()
    table = label_window_table(keys, subjects)
    spec = _spec(disjointness="stream", candidate_counts=(2, 4))
    plans = build_episode_plans(table, eligible_labels(table, spec, streams), n_episodes=30,
                                spec=spec, seed=2, support_schedule=(1, 2),
                                stream_ids=streams)
    assert provenance_lift(plans, streams)["matched"] == 0.0


def test_shared_query_stream_drives_the_provenance_lift_to_zero():
    """The configuration in which device identity cannot favour any candidate."""
    from training.tokenizer.episodic import (
        build_shared_stream_plans, provenance_lift, stream_label_table,
    )

    keys, subjects, streams = _streamed_table()
    table = label_window_table(keys, subjects)
    stream_table = stream_label_table(keys, subjects, streams)
    spec = _spec(disjointness="stream", shared_query_stream=True, candidate_counts=(2, 4))
    plans = build_shared_stream_plans(
        table, stream_table, eligible_labels(table, spec, streams), n_episodes=30,
        spec=spec, seed=4, stream_ids=streams, support_schedule=(1, 2),
    )
    lift = provenance_lift(plans, streams)
    assert lift["support_rows"] > 0
    assert lift["matched"] == 0.0 and lift["distractor"] == 0.0 and lift["lift"] == 0.0
    for plan in plans:                       # every query in an episode shares one stream
        assert len({int(streams[p]) for p in plan.query_positions}) == 1


def test_single_stream_labels_are_ineligible_for_stream_disjoint_support():
    keys, subjects, streams = _streamed_table(n_streams=1)
    table = label_window_table(keys, subjects)
    assert eligible_labels(table, _spec(disjointness="stream"), streams) == []
    assert eligible_labels(table, _spec(disjointness="subject"), streams) != []


def test_shared_query_stream_requires_stream_disjoint_support():
    with pytest.raises(ValueError, match="only meaningful with disjointness='stream'"):
        _spec(disjointness="subject", shared_query_stream=True).validate()


def test_checkpoint_selection_follows_the_active_label_regime():
    from training.tokenizer.pretrain_episodic import select_validation_metric

    coherent_run = select_validation_metric(
        {"coherent_mean_macro_f1": 0.6, "alias_mean_macro_f1": 0.2},
        include_aliases=False,
    )
    assert coherent_run["selection_metric"] == "coherent_curve_mean"
    assert coherent_run["selection_score"] == 0.6

    alias_run = select_validation_metric(
        {"coherent_mean_macro_f1": 0.6, "alias_mean_macro_f1": 0.2},
        include_aliases=True,
    )
    assert alias_run["selection_metric"].startswith("mean(")
    assert alias_run["selection_score"] == 0.4


def test_replace_counts_preserves_every_spec_field():
    """REGRESSION: a positional rebuild silently reset `disjointness`, making the fix inert.

    Every test still passed while the stream-disjoint mode did nothing at all. Field-by-field, so
    adding a field to EpisodeSpec without carrying it through fails here rather than in a run.
    """
    import dataclasses

    from training.tokenizer.pretrain_episodic import replace_counts

    spec = EpisodeSpec(candidate_counts=(2, 4, 64), queries_per_candidate=3, max_support=2,
                       background_windows=7, disjointness="stream", shared_query_stream=True)
    narrowed = replace_counts(spec, pool_size=8)
    assert narrowed.candidate_counts == (2, 4)
    for field in dataclasses.fields(EpisodeSpec):
        if field.name == "candidate_counts":
            continue
        assert getattr(narrowed, field.name) == getattr(spec, field.name), field.name


def test_shared_stream_episodes_keep_the_requested_candidate_distribution():
    """Draw the candidate count first, then a stream that can supply it.

    Choosing the stream first silently truncated each episode to whatever that stream carried, and
    streams are label-poor. Measured on the real corpus that turned a uniform draw over
    {2,4,8,16} into 45/30/17/8 percent: nearly half the episodes were binary and chance accuracy
    was 32.8% instead of 22.2%, so every reported number sat against an inflated floor.
    """
    import collections

    import numpy as np

    from training.tokenizer.episodic import EpisodeSpec, build_shared_stream_plans

    # 12 streams, deliberately label-poor: only two of them carry enough labels for C=8.
    rich, poor = list(range(10)), list(range(3))
    stream_table, table = {}, {}
    position = 0
    for stream in range(12):
        labels = rich if stream < 2 else poor
        stream_table[stream] = {}
        for label in labels:
            rows = np.arange(position, position + 6)
            position += 6
            stream_table[stream][label] = {stream * 100: rows}
            table.setdefault(label, {})[stream * 100] = rows
    stream_ids = np.zeros(position, dtype=np.int64)
    for stream, by_label in stream_table.items():
        for rows in by_label.values():
            for value in rows.values() if isinstance(rows, dict) else [rows]:
                stream_ids[value] = stream

    spec = EpisodeSpec(candidate_counts=(2, 8), queries_per_candidate=2, max_support=0,
                       background_windows=1, alias_episode_fraction=0.0,
                       disjointness="stream", shared_query_stream=True)
    plans = build_shared_stream_plans(
        table, stream_table, sorted(set(rich)), n_episodes=200, spec=spec, seed=0,
        stream_ids=stream_ids,
    )
    counts = collections.Counter(len(plan.candidates) for plan in plans)
    assert set(counts) == {2, 8}
    # Both counts must be well represented even though only 2 of 12 streams can host C=8.
    assert counts[8] / len(plans) > 0.35, counts
    assert counts[2] / len(plans) > 0.35, counts


# --- alias curriculum (2026-08-22) -------------------------------------------------------------
def test_alias_curriculum_is_none_when_disabled():
    from training.tokenizer.pretrain_episodic import alias_curriculum
    assert alias_curriculum(100, 4, 0.5, 0, 0) is None


def test_alias_curriculum_holds_then_ramps_then_saturates():
    from training.tokenizer.pretrain_episodic import alias_curriculum
    schedule = alias_curriculum(4 * 40, 4, 0.5, 10, 10)   # 4 ep/step, hold 10, ramp 10 -> 0.5
    assert schedule is not None
    step = np.arange(4 * 40) // 4
    assert schedule[step < 10].max() == 0.0                       # warmup: no alias at all
    assert schedule[step == 15][0] == pytest.approx(0.25)         # halfway up the ramp
    assert schedule[step >= 20].min() == pytest.approx(0.5)       # saturated at the target
    assert schedule.max() <= 0.5


def test_alias_schedule_controls_label_mode_per_episode():
    """All-zero schedule => no alias episodes; all-one => alias whenever the episode has support."""
    table, _keys, _subjects = _synthetic_table()
    spec = _spec(alias_episode_fraction=0.5)
    pool = eligible_labels(table, spec)
    n = 64
    zero = build_episode_plans(table, pool, n_episodes=n, spec=spec, seed=7,
                               alias_schedule=np.zeros(n))
    one = build_episode_plans(table, pool, n_episodes=n, spec=spec, seed=7,
                              alias_schedule=np.ones(n))
    assert all(p.label_mode == "coherent" for p in zero)
    aliased = [p for p in one if p.support_positions]
    assert aliased and all(p.label_mode == "random_alias" for p in aliased)


def test_alias_schedule_none_matches_flat_fraction():
    """The default path is unchanged by the schedule parameter existing."""
    table, _keys, _subjects = _synthetic_table()
    spec = _spec(alias_episode_fraction=0.5)
    pool = eligible_labels(table, spec)
    a = build_episode_plans(table, pool, n_episodes=32, spec=spec, seed=11)
    b = build_episode_plans(table, pool, n_episodes=32, spec=spec, seed=11, alias_schedule=None)
    assert [p.label_mode for p in a] == [p.label_mode for p in b]
