"""Parity checks for the current recording-level reranker diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from eval.run_halo_engine_decomposition import METHODS, _raw_and_reranked
from model.blocks import AttentionSpec
from model.evidence.engine import EngineConfig, EvidenceEngine
from model.evidence.rows import SensorRows


def _rows(n: int, *, d: int, seed: int, bound: torch.Tensor | None = None) -> SensorRows:
    generator = torch.Generator().manual_seed(seed)
    return SensorRows(
        feature=torch.randn(n, d, generator=generator),
        descriptor=F.normalize(torch.randn(n, 384, generator=generator), dim=-1),
        bias=torch.zeros(n, 1),
        modality=torch.zeros(n, dtype=torch.long),
        gravity=torch.zeros(n, dtype=torch.long),
        label=torch.arange(n) % 6,
        dataset=torch.zeros(n, dtype=torch.long),
        enrolled_candidate=(torch.full((n,), -1, dtype=torch.long) if bound is None else bound),
        source_window=torch.arange(n),
    )


def test_decomposition_uses_exact_engine_raw_and_reranked_logits():
    spec = AttentionSpec(d_model=8, n_heads=2, dropout=0.0)
    engine = EvidenceEngine(None, EngineConfig(spec=spec)).eval()
    query = _rows(5, d=8, seed=1)
    memory = _rows(9, d=8, seed=2, bound=torch.tensor([0, 1, -1, -1, -1, -1, -1, -1, -1]))
    generator = torch.Generator().manual_seed(3)
    candidate = F.normalize(torch.randn(3, 384, generator=generator), dim=-1)
    labels = F.normalize(torch.randn(6, 384, generator=generator), dim=-1)
    result = engine(query, memory, candidate, labels)
    raw, reranked = _raw_and_reranked(result)
    assert torch.equal(raw, result["base_logits"])
    assert torch.equal(reranked, result["logits"])


def test_decomposition_contains_only_nearest_neighbor_arms():
    assert METHODS == (
        "support_raw_1nn",
        "support_reranked_1nn",
        "corpus_raw_1nn",
        "corpus_reranked_1nn",
        "full_raw_1nn",
        "full_reranked_1nn",
    )
