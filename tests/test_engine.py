from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from model.blocks import AttentionSpec
from model.evidence.engine import EngineConfig, EvidenceEngine
from model.evidence.evidence_reranker import EvidenceRerankerConfig
from model.evidence.rows import SensorRows


def _rows(n: int, d: int = 16, text: int = 384, *, labels=None, bound=None, seed=0):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % 6 if labels is None else torch.as_tensor(labels)
    bound = torch.full((n,), -1) if bound is None else torch.as_tensor(bound)
    return SensorRows(
        feature=torch.randn(n, d, generator=generator),
        descriptor=F.normalize(torch.randn(n, text, generator=generator), dim=-1),
        bias=torch.zeros(n, 1), modality=torch.zeros(n, dtype=torch.long),
        gravity=torch.zeros(n, dtype=torch.long), label=labels.long(),
        dataset=torch.zeros(n, dtype=torch.long), enrolled_candidate=bound.long(),
        source_window=torch.arange(n),
    )


def _engine(d=16, temperature=0.1):
    return EvidenceEngine(None, EngineConfig(
        spec=AttentionSpec(d_model=d, n_heads=4), surrogate_temperature=temperature,
        reranker=EvidenceRerankerConfig(
            n_interaction_heads=4, interaction_dim=4, hidden_dim=16,
            correction_gain_init=0.01, max_correction=0.5,
        ),
    ))


def _texts(v=8, c=4, seed=2):
    generator = torch.Generator().manual_seed(seed)
    labels = F.normalize(torch.randn(v, 384, generator=generator), dim=-1)
    return labels, labels[:c].clone()


def test_zero_correction_is_exact_raw_corrected_nearest():
    engine = _engine()
    with torch.no_grad():
        engine.reranker.head_out.weight.zero_()
    query = _rows(3, seed=3)
    memory = _rows(12, labels=torch.arange(12) % 4, seed=4)
    label_text, candidate = _texts()
    result = engine(query, memory, candidate, label_text)
    assert torch.equal(result["logits"], result["base_logits"])
    assert torch.equal(result["logits"].argmax(1), result["base_logits"].argmax(1))


def test_forward_value_is_hard_nearest_not_soft_vote():
    engine = _engine(temperature=0.5)
    query = _rows(2, seed=5)
    memory = _rows(9, labels=torch.arange(9) % 4, seed=6)
    label_text, candidate = _texts()
    result = engine(query, memory, candidate, label_text)
    compatibility = engine._compatibility(
        memory.label.unsqueeze(0), memory.enrolled_candidate.unsqueeze(0),
        torch.ones(1, len(memory.feature), dtype=torch.bool), candidate.unsqueeze(0), label_text,
    )[0]
    expected = (result["scores"].unsqueeze(-1) + compatibility).max(dim=1).values
    assert torch.allclose(result["logits"], expected)


def test_many_weak_rows_cannot_outvote_one_strong_enrolled_match():
    engine = _engine()
    with torch.no_grad():
        engine.reranker.head_out.weight.zero_()
    d = 16
    query = _rows(1, d=d, seed=7)
    strong = F.normalize(query.feature.clone(), dim=-1)
    weak = F.normalize(torch.randn(100, d, generator=torch.Generator().manual_seed(8)), dim=-1)
    weak = weak - (weak @ strong.T).clamp_min(0) * strong
    memory = _rows(101, d=d, labels=torch.ones(101, dtype=torch.long), seed=9)
    memory = SensorRows(**{**memory.__dict__,
        "feature": torch.cat((strong, weak)),
        "enrolled_candidate": torch.tensor([0] + [-1] * 100),
        "label": torch.tensor([0] + [1] * 100),
    })
    label_text, candidate = _texts(c=2)
    result = engine(query, memory, candidate, label_text)
    assert result["logits"].argmax(1).item() == 0


def test_enrolled_binding_overrides_arbitrary_candidate_text():
    engine = _engine()
    with torch.no_grad():
        engine.reranker.head_out.weight.zero_()
    query = _rows(1, seed=10)
    memory = _rows(2, labels=[0, 0], bound=[1, 0], seed=11)
    memory.feature[0].copy_(query.feature[0])
    memory.feature[1].copy_(-query.feature[0])
    label_text, _ = _texts()
    arbitrary = F.normalize(torch.randn(2, 384), dim=-1)
    result = engine(query, memory, arbitrary, label_text)
    assert result["logits"].argmax(1).item() == 1


def test_memory_permutation_invariance():
    engine = _engine().eval()
    query = _rows(3, seed=12)
    memory = _rows(15, labels=torch.arange(15) % 4, seed=13)
    label_text, candidate = _texts()
    expected = engine(query, memory, candidate, label_text)["logits"]
    order = torch.randperm(15)
    permuted = SensorRows(**{
        name: None if getattr(memory, name) is None else getattr(memory, name)[order]
        for name in memory.__dataclass_fields__
    })
    actual = engine(query, permuted, candidate, label_text)["logits"]
    assert torch.allclose(actual, expected, atol=1e-6)


def test_candidate_permutation_equivariance():
    engine = _engine().eval()
    query = _rows(2, seed=14)
    memory = _rows(12, labels=torch.arange(12) % 4, seed=15)
    label_text, candidate = _texts()
    expected = engine(query, memory, candidate, label_text)["logits"]
    order = torch.tensor([2, 0, 3, 1])
    actual = engine(query, memory, candidate[order], label_text)["logits"]
    assert torch.allclose(actual, expected[:, order], atol=1e-6)


def test_all_reranker_modules_receive_gradient_on_first_step():
    engine = _engine().train()
    query = _rows(4, seed=16)
    memory = _rows(24, labels=torch.arange(24) % 4, seed=17)
    query.feature.requires_grad_(); memory.feature.requires_grad_()
    query.descriptor.requires_grad_(); memory.descriptor.requires_grad_()
    label_text, candidate = _texts()
    loss = F.cross_entropy(engine(query, memory, candidate, label_text)["logits"],
                           torch.tensor([0, 1, 2, 3]))
    loss.backward()
    missing = [name for name, parameter in engine.reranker.named_parameters()
               if parameter.grad is None or not torch.isfinite(parameter.grad).all()]
    assert not missing
    assert query.feature.grad is not None and query.feature.grad.abs().sum() > 0
    assert memory.feature.grad is not None and memory.feature.grad.abs().sum() > 0
    assert query.descriptor.grad is not None and query.descriptor.grad.abs().sum() > 0
    assert memory.descriptor.grad is not None and memory.descriptor.grad.abs().sum() > 0
    assert bool(memory.feature.grad.norm(dim=1).gt(0).all())


def test_acquisition_descriptions_affect_corrections():
    engine = _engine().eval()
    query = _rows(2, seed=18)
    memory = _rows(10, labels=torch.arange(10) % 4, seed=19)
    label_text, candidate = _texts()
    before = engine(query, memory, candidate, label_text)["score_correction"]
    changed = copy.deepcopy(memory)
    changed = SensorRows(**{**changed.__dict__, "descriptor": -changed.descriptor})
    after = engine(query, changed, candidate, label_text)["score_correction"]
    assert not torch.allclose(after, before)


def test_vectorized_many_matches_sequential():
    engine = _engine().eval()
    queries = [_rows(3, seed=20), _rows(2, seed=21)]
    memories = [_rows(11, labels=torch.arange(11) % 4, seed=22),
                _rows(15, labels=torch.arange(15) % 4, seed=23)]
    label_text, candidate = _texts()
    sequential = [engine(q, m, candidate, label_text)["logits"]
                  for q, m in zip(queries, memories)]
    batched = engine.forward_many(
        queries, memories, torch.stack((candidate, candidate)), label_text,
    )
    for expected, actual in zip(sequential, batched):
        assert torch.allclose(actual["logits"], expected, atol=1e-6)


def test_duplicate_recording_rows_fail_loudly():
    engine = _engine()
    query = _rows(2)
    query = SensorRows(**{**query.__dict__, "source_window": torch.tensor([0, 0])})
    memory = _rows(4)
    label_text, candidate = _texts()
    with pytest.raises(ValueError, match="exactly one pooled row"):
        engine(query, memory, candidate, label_text)


def test_correction_is_bounded():
    engine = _engine()
    query = _rows(4, seed=24)
    memory = _rows(20, seed=25)
    label_text, candidate = _texts()
    correction = engine(query, memory, candidate, label_text)["score_correction"]
    assert float(correction.detach().abs().max()) <= engine.reranker.cfg.max_correction
