"""Structural and gradient guarantees of the recording-level compact evidence engine."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from model.blocks import AttentionSpec, ScaledSum, SetAttentionStack
from model.evidence.engine import (
    EngineConfig, EvidenceEngine, batched_vote, enrolled_1nn_logits, vote,
)
from model.evidence.evidence_mixer import EvidenceMixerConfig
from model.evidence.evidence_reranker import EvidenceRerankerConfig
from model.evidence.retrieval_scorer import PairScorer, PairScorerConfig
from model.evidence.rows import SensorRows
from model.tokenizer.transformer import TemporalTrunk

TEXT_DIM = 384
_LEARNED = PairScorerConfig(learned=True)


def _rows(n, *, d=64, enrolled=None, vocab=40, seed=0, windows_per_row=2):
    generator = torch.Generator().manual_seed(seed)
    return SensorRows(
        feature=torch.randn(n, d, generator=generator),
        descriptor=F.normalize(torch.randn(n, TEXT_DIM, generator=generator), dim=-1),
        bias=torch.zeros(n, 14),
        modality=torch.randint(0, 2, (n,), generator=generator),
        gravity=torch.randint(0, 2, (n,), generator=generator),
        label=torch.randint(0, vocab, (n,), generator=generator),
        dataset=torch.zeros(n, dtype=torch.long),
        enrolled_candidate=torch.full((n,), -1) if enrolled is None else enrolled,
        source_window=torch.arange(n) // windows_per_row,
    )


def _engine(*, d=64, top_k=8, vote_scope="bank", mixing="attention"):
    spec = AttentionSpec(d_model=d, n_heads=4, ffn_mult=2, dropout=0.0)
    return EvidenceEngine(None, EngineConfig(
        spec=spec, top_k=top_k, vote_scope=vote_scope, mixing=mixing,
        mixer=EvidenceMixerConfig(n_groups=max(16, top_k + 2)),
    )).eval()


def _reranker_engine(*, d=64, top_k=8, vote_scope="bank"):
    spec = AttentionSpec(d_model=d, n_heads=4, ffn_mult=2, dropout=0.0)
    return EvidenceEngine(None, EngineConfig(
        spec=spec, top_k=top_k, vote_scope=vote_scope, mixing="rerank",
        reranker=EvidenceRerankerConfig(n_groups=max(16, top_k + 2)),
    )).eval()


def _episode(n_memory=64, n_query=6, n_cand=5, d=64, vocab=40, seed=0):
    generator = torch.Generator().manual_seed(seed + 100)
    enrolled = torch.full((n_memory,), -1)
    enrolled[:4] = torch.tensor([0, 0, 1, 2])
    query = _rows(n_query, d=d, vocab=vocab, seed=seed)
    memory = _rows(n_memory, d=d, enrolled=enrolled, vocab=vocab, seed=seed + 1)
    label_text = F.normalize(torch.randn(vocab, TEXT_DIM, generator=generator), dim=-1)
    candidate_text = F.normalize(torch.randn(n_cand, TEXT_DIM, generator=generator), dim=-1)
    return query, memory, candidate_text, label_text


def _permute_rows(rows: SensorRows, order: torch.Tensor) -> SensorRows:
    return SensorRows(**{
        name: None if getattr(rows, name) is None else getattr(rows, name)[order]
        for name in rows.__dataclass_fields__
    })


# -------------------------------------------------------------------------------- attention and trunk
def test_parameters_per_block_matches_a_real_block():
    spec = AttentionSpec(d_model=128, n_heads=4, ffn_mult=2, dropout=0.0)
    stack = SetAttentionStack(spec, n_layers=1)
    measured = sum(p.numel() for p in stack.parameters()) - 2 * spec.d_model
    assert measured == spec.parameters_per_block()


def test_set_attention_is_permutation_equivariant_and_masks_padding():
    spec = AttentionSpec(d_model=32, n_heads=4, ffn_mult=2, dropout=0.0)
    stack = SetAttentionStack(spec, n_layers=2).eval()
    x = torch.randn(2, 9, 32)
    order = torch.randperm(9)
    assert torch.allclose(stack(x)[:, order], stack(x[:, order]), atol=1e-5)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[:, -2:] = False
    disturbed = x.clone(); disturbed[:, -2:] += 20
    assert torch.allclose(
        stack(x, key_padding_mask=mask)[:, :-2],
        stack(disturbed, key_padding_mask=mask)[:, :-2], atol=1e-5,
    )


def test_scaled_sum_controls_identity_magnitude():
    mixed = ScaledSum(4, init=[1.0, 0.25, 0.25, 0.25])
    assert torch.allclose(mixed.log_gain.exp(), torch.tensor([1.0, 0.25, 0.25, 0.25]))


def test_temporal_trunk_encodes_each_sensor_in_isolation():
    spec = AttentionSpec(d_model=32, n_heads=4, ffn_mult=2, dropout=0.0)
    trunk = TemporalTrunk(spec, num_layers=2).eval()
    x = torch.randn(2, 5, 3, 32)
    positions = torch.arange(5.0).view(1, 5).expand(2, 5)
    baseline = trunk(x, positions=positions)
    x[:, :, 2] += 5
    assert torch.allclose(baseline[:, :, 0], trunk(x, positions=positions)[:, :, 0,], atol=1e-5)


# -------------------------------------------------------------------------------- retrieval and vote
def test_pair_scorer_starts_at_cosine_rule():
    spec = AttentionSpec(d_model=32, n_heads=4)
    scorer = PairScorer(spec, _LEARNED).eval()
    qf, mf = torch.randn(7, 32), torch.randn(19, 32)
    qd = F.normalize(torch.randn(7, TEXT_DIM), dim=-1)
    md = F.normalize(torch.randn(19, TEXT_DIM), dim=-1)
    expected = F.normalize(qf, dim=-1) @ F.normalize(mf, dim=-1).T / 0.07
    assert (scorer(qf, qd, mf, md) - expected).abs().max() < 0.05


def test_joint_vote_is_a_probability_distribution_and_enrollment_is_one_hot():
    candidate = torch.eye(3, TEXT_DIM)
    labels = candidate.clone()
    scores = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    bound = torch.tensor([0, 1, 2])
    mass = vote(scores, labels, candidate, bound)
    assert torch.allclose(mass.sum(1), torch.ones(1))
    assert mass.argmax(1).item() == 2
    F.cross_entropy(mass.clamp_min(1e-8).log(), torch.tensor([2])).backward()
    assert scores.grad is not None and float(scores.grad.abs().sum()) > 0


def test_corpus_text_affinity_is_normalized_across_candidates():
    candidate = F.normalize(torch.randn(4, TEXT_DIM), dim=-1)
    labels = F.normalize(torch.randn(7, TEXT_DIM), dim=-1)
    mass = vote(torch.zeros(2, 7), labels, candidate, torch.full((7,), -1))
    assert torch.isfinite(mass).all()
    assert torch.allclose(mass.sum(1), torch.ones(2), atol=1e-6)


def test_batched_vote_matches_independent_episode_votes_with_padding():
    candidate = F.normalize(torch.randn(2, 5, TEXT_DIM), dim=-1)
    labels = F.normalize(torch.randn(2, 7, TEXT_DIM), dim=-1)
    scores = torch.randn(2, 4, 7)
    bound = torch.tensor([
        [0, 1, -1, -1, -1, -1, -1],
        [2, -1, 4, -1, -1, -1, -1],
    ])
    mask = torch.tensor([
        [True, True, True, True, True, False, False],
        [True, True, True, True, True, True, True],
    ])
    actual = batched_vote(scores, labels, candidate, bound, mask)
    expected = torch.stack([
        vote(scores[e, :, mask[e]], labels[e, mask[e]], candidate[e], bound[e, mask[e]])
        for e in range(2)
    ])
    assert torch.allclose(actual, expected, atol=1e-6)


def test_enrolled_1nn_ignores_corpus_rows_and_marks_missing_candidates():
    scores = torch.tensor([
        [100.0, 1.0, 3.0, 2.0],
        [100.0, 4.0, 2.0, 7.0],
    ])
    # The highest-scoring row is corpus-only. Candidate 0 has two enrolled rows, candidate 1 one,
    # and candidate 2 none.
    logits, available = enrolled_1nn_logits(scores, torch.tensor([-1, 0, 1, 0]), 3)
    assert torch.equal(available, torch.tensor([[True, True, False], [True, True, False]]))
    assert torch.equal(logits[:, :2], torch.tensor([[2.0, 3.0], [7.0, 2.0]]))
    assert torch.isfinite(logits).all()


def test_enrolled_1nn_is_finite_but_unavailable_at_k_zero():
    logits, available = enrolled_1nn_logits(
        torch.randn(3, 5), torch.full((5,), -1), 4,
    )
    assert torch.equal(logits, torch.zeros_like(logits))
    assert not bool(available.any())


# -------------------------------------------------------------------------------- recording-level engine
def test_step_zero_is_exactly_the_retrieval_baseline():
    engine = _engine()
    out = engine(*_episode())
    assert torch.equal(out["logits"], out["base_logits"])
    assert torch.count_nonzero(out["residual_logits"]) == 0


def test_reranker_starts_near_baseline_and_is_bounded():
    engine = _reranker_engine()
    out = engine(*_episode())
    assert float(out["residual_logits"].detach().abs().max()) < 0.01
    assert (float(out["score_correction"].detach().abs().max())
            <= engine.reranker.cfg.max_correction)


def test_reranker_all_parameters_receive_gradient_on_first_step():
    engine = _reranker_engine().train()
    query, memory, candidate, labels = _episode()
    out = engine(query, memory, candidate, labels)
    F.cross_entropy(out["logits"], torch.zeros(len(out["logits"]), dtype=torch.long)).backward()
    missing = [name for name, parameter in engine.reranker.named_parameters()
               if parameter.grad is None or not float(parameter.grad.abs().sum())]
    assert missing == []


def _vectorized_episodes():
    episodes = [_episode(n_memory=47, n_query=7, seed=101),
                _episode(n_memory=64, n_query=9, seed=202)]
    queries, memories, candidates = [], [], []
    label_text = episodes[0][3]
    for index, (query, memory, candidate, _labels) in enumerate(episodes):
        # Both episodes contain four recordings but different numbers of sensor rows, exercising
        # query and memory padding without changing the semantic output shape.
        query = SensorRows(**{
            **{name: getattr(query, name) for name in query.__dataclass_fields__},
            "source_window": torch.arange(len(query.feature)) % 4,
        })
        queries.append(query)
        memories.append(memory)
        candidates.append(candidate)
    return queries, memories, torch.stack(candidates), label_text


def test_vectorized_episodes_match_sequential_logits_and_controls():
    engine = _reranker_engine()
    queries, memories, candidates, labels = _vectorized_episodes()
    seeds = [31, 47]
    expected = [
        engine(query, memory, candidate, labels,
               generator=torch.Generator().manual_seed(seed))
        for query, memory, candidate, seed in zip(queries, memories, candidates, seeds)
    ]
    actual = engine.forward_many(
        queries, memories, candidates, labels,
        generators=[torch.Generator().manual_seed(seed) for seed in seeds],
    )
    for reference, batched in zip(expected, actual):
        for field in ("logits", "base_logits", "base_mass", "residual_logits",
                      "score_correction", "scores", "recording_scores",
                      "enrolled_1nn_logits"):
            assert torch.allclose(reference[field], batched[field], atol=2e-5), field
        assert torch.equal(reference["selected"], batched["selected"])
        assert torch.equal(reference["enrolled_1nn_available"],
                           batched["enrolled_1nn_available"])


def test_vectorized_episodes_match_sequential_parameter_gradients():
    sequential = _reranker_engine().train()
    vectorized = copy.deepcopy(sequential).train()
    queries, memories, candidates, labels = _vectorized_episodes()

    def trainable_copy(rows):
        return SensorRows(**{
            name: (getattr(rows, name).detach().clone().requires_grad_(True)
                   if name == "feature" else getattr(rows, name).detach().clone())
            for name in rows.__dataclass_fields__
        })

    sequential_queries = [trainable_copy(rows) for rows in queries]
    sequential_memories = [trainable_copy(rows) for rows in memories]
    vectorized_queries = [trainable_copy(rows) for rows in queries]
    vectorized_memories = [trainable_copy(rows) for rows in memories]
    targets = [torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4])]
    seeds = [53, 59]
    sequential_loss = torch.stack([
        F.cross_entropy(
            sequential(query, memory, candidate, labels,
                       generator=torch.Generator().manual_seed(seed))["logits"], target,
        )
        for query, memory, candidate, target, seed
        in zip(sequential_queries, sequential_memories, candidates, targets, seeds)
    ]).mean()
    sequential_loss.backward()
    outputs = vectorized.forward_many(
        vectorized_queries, vectorized_memories, candidates, labels,
        generators=[torch.Generator().manual_seed(seed) for seed in seeds],
    )
    torch.stack([
        F.cross_entropy(output["logits"], target)
        for output, target in zip(outputs, targets)
    ]).mean().backward()
    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        sequential.named_parameters(), vectorized.named_parameters(), strict=True,
    ):
        assert name_a == name_b
        assert parameter_a.grad is not None and parameter_b.grad is not None
        assert torch.allclose(parameter_a.grad, parameter_b.grad, atol=2e-5), name_a
    for sequential_rows, vectorized_rows in zip(
        sequential_queries + sequential_memories,
        vectorized_queries + vectorized_memories,
    ):
        assert sequential_rows.feature.grad is not None
        assert vectorized_rows.feature.grad is not None
        assert torch.allclose(
            sequential_rows.feature.grad, vectorized_rows.feature.grad, atol=2e-5,
        )


def test_engine_returns_one_output_per_recording():
    engine = _engine()
    query, memory, candidate, labels = _episode(n_query=7)
    out = engine(query, memory, candidate, labels)
    assert out["query_window"].tolist() == [0, 1, 2, 3]
    assert out["logits"].shape == (4, len(candidate))
    assert torch.allclose(out["base_mass"].sum(1), torch.ones(4), atol=1e-6)


def test_global_shortlist_uses_any_query_row_in_the_recording():
    engine = _engine(d=4, top_k=1, mixing="off")
    query = _rows(2, d=4, windows_per_row=2)
    memory = _rows(3, d=4, windows_per_row=1)
    with torch.no_grad():
        query.feature.copy_(torch.tensor([[1., 0, 0, 0], [0., 1, 0, 0]]))
        memory.feature.copy_(torch.tensor([[0.8, 0.6, 0, 0], [0., 1., 0, 0], [0., 0, 1, 0]]))
        query.modality.zero_(); memory.modality.zero_()
        query.gravity.zero_(); memory.gravity.zero_()
    candidate = F.normalize(torch.randn(3, TEXT_DIM), dim=-1)
    labels = F.normalize(torch.randn(40, TEXT_DIM), dim=-1)
    out = engine(query, memory, candidate, labels)
    assert out["selected"].item() == 1


def test_memory_row_order_does_not_change_the_result():
    engine = _engine()
    query, memory, candidate, labels = _episode()
    generator = torch.Generator().manual_seed(3)
    direct = engine(query, memory, candidate, labels, generator=generator)["logits"]
    order = torch.randperm(len(memory.feature))
    shuffled = engine(
        query, _permute_rows(memory, order), candidate, labels,
        generator=torch.Generator().manual_seed(3),
    )["logits"]
    assert torch.allclose(direct, shuffled, atol=1e-5)


def test_candidate_permutation_only_permutes_outputs():
    engine = _engine()
    query, memory, candidate, labels = _episode()
    order = torch.randperm(len(candidate))
    inverse = torch.empty_like(order); inverse[order] = torch.arange(len(order))
    rebound = memory.enrolled_candidate.clone()
    live = rebound.ge(0); rebound[live] = inverse[rebound[live]]
    memory2 = SensorRows(**{
        **{name: getattr(memory, name) for name in memory.__dataclass_fields__},
        "enrolled_candidate": rebound,
    })
    direct = engine(
        query, memory, candidate, labels, generator=torch.Generator().manual_seed(7),
    )["logits"][:, order]
    permuted = engine(
        query, memory2, candidate[order], labels, generator=torch.Generator().manual_seed(7),
    )["logits"]
    assert torch.allclose(direct, permuted, atol=1e-5)


def test_reranker_is_structurally_candidate_permutation_equivariant():
    engine = _reranker_engine()
    query, memory, candidate, labels = _episode()
    order = torch.randperm(len(candidate))
    inverse = torch.empty_like(order); inverse[order] = torch.arange(len(order))
    rebound = memory.enrolled_candidate.clone()
    live = rebound.ge(0); rebound[live] = inverse[rebound[live]]
    memory2 = SensorRows(**{
        **{name: getattr(memory, name) for name in memory.__dataclass_fields__},
        "enrolled_candidate": rebound,
    })
    direct = engine(
        query, memory, candidate, labels, generator=torch.Generator().manual_seed(7),
    )["logits"][:, order]
    permuted = engine(
        query, memory2, candidate[order], labels, generator=torch.Generator().manual_seed(7),
    )["logits"]
    assert torch.allclose(direct, permuted, atol=1e-5)


def test_query_recordings_do_not_leak_into_each_other():
    engine = _engine()
    query, memory, candidate, labels = _episode()
    with torch.no_grad():
        engine.mixer.residual_head.weight.normal_(std=0.1)
    baseline = engine(
        query, memory, candidate, labels, generator=torch.Generator().manual_seed(11),
    )["logits"]
    changed = SensorRows(**{
        **{name: getattr(query, name) for name in query.__dataclass_fields__},
        "feature": query.feature.clone(),
    })
    changed.feature[changed.source_window == 0] += 10
    altered = engine(
        changed, memory, candidate, labels, generator=torch.Generator().manual_seed(11),
    )["logits"]
    assert not torch.allclose(baseline[0], altered[0])
    assert torch.allclose(baseline[1:], altered[1:], atol=1e-5)


def test_all_query_rows_can_affect_the_recording_residual():
    engine = _engine()
    query, memory, candidate, labels = _episode()
    with torch.no_grad():
        engine.mixer.residual_head.weight.normal_(std=0.1)
    baseline = engine(query, memory, candidate, labels)["residual_logits"]
    changed = SensorRows(**{
        **{name: getattr(query, name) for name in query.__dataclass_fields__},
        "feature": query.feature.clone(),
    })
    changed.feature[1] += 5                         # second row, same recording as row zero
    altered = engine(changed, memory, candidate, labels)["residual_logits"]
    assert not torch.allclose(baseline[0], altered[0])


def test_zero_initialized_residual_wakes_the_whole_mixer_by_step_two():
    engine = _engine().train()
    query, memory, candidate, labels = _episode()
    optimizer = torch.optim.SGD(engine.mixer.parameters(), lr=0.1)
    target = torch.zeros(3, dtype=torch.long)

    out = engine(query, memory, candidate, labels)
    F.cross_entropy(out["logits"], target).backward()
    assert float(engine.mixer.residual_head.weight.grad.abs().sum()) > 0
    dormant = [name for name, p in engine.mixer.named_parameters()
               if not name.startswith("residual_head") and
               (p.grad is None or not float(p.grad.abs().sum()))]
    assert dormant                              # expected one-step delay, not hidden
    optimizer.step(); optimizer.zero_grad(set_to_none=True)

    out = engine(query, memory, candidate, labels)
    F.cross_entropy(out["logits"], target).backward()
    missing = [name for name, p in engine.mixer.named_parameters()
               if p.grad is None or not float(p.grad.abs().sum())]
    assert missing == []


def test_full_bank_vote_reaches_more_memory_rows_than_topk():
    coverage = {}
    for scope in ("topk", "bank"):
        engine = _engine(vote_scope=scope).train()
        query, memory, candidate, labels = _episode()
        memory.feature.requires_grad_(True)
        out = engine(query, memory, candidate, labels)
        F.cross_entropy(out["logits"], torch.zeros(3, dtype=torch.long)).backward()
        coverage[scope] = int((memory.feature.grad.abs().sum(1) > 0).sum())
    assert coverage["bank"] == len(memory.feature)
    assert coverage["topk"] < coverage["bank"]


def test_identity_channels_preserve_binding_and_recording_groups():
    engine = _engine(top_k=8)
    _, memory, candidate, _ = _episode(n_memory=16)
    selected = torch.arange(8).view(1, 8)
    candidate_slot, evidence_slot, groups, bound = engine._identity_channels(
        memory, selected, len(candidate), torch.Generator().manual_seed(2),
    )
    live = bound[0].ge(0)
    assert torch.equal(evidence_slot[0, live], candidate_slot[bound[0, live]])
    assert groups[0, 0] == groups[0, 1]
    assert groups[0, 1] != groups[0, 2]


def test_engine_requires_recording_provenance_and_checks_group_capacity():
    engine = _engine()
    query, memory, candidate, labels = _episode()
    blind = SensorRows(**{
        **{name: getattr(memory, name) for name in memory.__dataclass_fields__},
        "source_window": None,
    })
    with pytest.raises(ValueError, match="source_window"):
        engine(query, blind, candidate, labels)
    with pytest.raises(ValueError, match="co-membership groups"):
        EngineConfig(
            spec=AttentionSpec(d_model=16, n_heads=4), top_k=16,
            mixer=EvidenceMixerConfig(n_groups=4), mixing="attention",
        )


def test_fixed_retrieval_with_mixing_off_has_no_parameters():
    engine = _engine(mixing="off")
    assert sum(p.numel() for p in engine.parameters() if p.requires_grad) == 0
    out = engine(*_episode())
    assert torch.equal(out["logits"], out["base_logits"])


def test_parameter_report_totals_parts():
    report = _engine().parameter_report()
    assert report["TOTAL"] == report["scorer"] + report["mixer"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mixed_precision_and_cpu_generator_are_finite():
    engine = _engine().cuda().train()
    query, memory, candidate, labels = _episode()
    query, memory = query.to("cuda"), memory.to("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = engine(
            query, memory, candidate.cuda(), labels.cuda(),
            generator=torch.Generator().manual_seed(1),
        )
    assert out["logits"].dtype == torch.float32
    assert torch.isfinite(out["logits"]).all()
