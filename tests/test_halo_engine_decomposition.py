"""Parity checks for the HALO evidence-engine decomposition diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from eval.run_halo_engine_decomposition import (
    _support_hard_logits,
    _topk_base_logits,
    _uniform_corpus_logits,
)
from model.blocks import AttentionSpec
from model.evidence.engine import EngineConfig, EvidenceEngine
from model.evidence.rows import SensorRows


def _rows(n: int, *, d: int, seed: int, bound: torch.Tensor | None = None) -> SensorRows:
    generator = torch.Generator().manual_seed(seed)
    return SensorRows(
        feature=torch.randn(n, d, generator=generator),
        descriptor=F.normalize(torch.randn(n, 384, generator=generator), dim=-1),
        bias=torch.zeros(n, 14),
        modality=torch.zeros(n, dtype=torch.long),
        gravity=torch.zeros(n, dtype=torch.long),
        label=torch.arange(n) % 6,
        dataset=torch.zeros(n, dtype=torch.long),
        enrolled_candidate=(torch.full((n,), -1, dtype=torch.long) if bound is None else bound),
        source_window=torch.arange(n) // 2,
    )


def test_topk_decomposition_is_exactly_the_engine_topk_base():
    spec = AttentionSpec(d_model=8, n_heads=2, dropout=0.0)
    engine = EvidenceEngine(None, EngineConfig(
        spec=spec, top_k=3, mixing="off", vote_scope="bank",
    )).eval()
    query = _rows(5, d=8, seed=1)
    memory = _rows(9, d=8, seed=2, bound=torch.tensor([0, 1, -1, -1, -1, -1, -1, -1, -1]))
    generator = torch.Generator().manual_seed(3)
    candidate = F.normalize(torch.randn(3, 384, generator=generator), dim=-1)
    labels = F.normalize(torch.randn(6, 384, generator=generator), dim=-1)
    bank = engine(query, memory, candidate, labels)
    expected = engine(query, memory, candidate, labels, vote_scope="topk")["base_logits"]
    actual = _topk_base_logits(bank, memory, candidate, labels)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_support_hard_and_uniform_corpus_return_recording_logits():
    support = _rows(3, d=4, seed=4, bound=torch.tensor([0, 1, 2]))
    result = {
        "scores": torch.tensor([
            [3.0, 1.0, 0.0],
            [2.0, 4.0, 0.0],
            [0.0, 1.0, 5.0],
        ]),
        "query_inverse": torch.tensor([0, 0, 1]),
    }
    hard = _support_hard_logits(result, support, 3)
    assert hard.argmax(1).tolist() == [0, 2]

    memory = _rows(3, d=4, seed=5, bound=torch.tensor([0, -1, 2]))
    uniform = _uniform_corpus_logits(result, memory, 3)
    assert uniform.shape == (2, 3)
    assert torch.isfinite(uniform).all()
    assert torch.allclose(uniform.exp().sum(1), torch.ones(2), atol=1e-6)
