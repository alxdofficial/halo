"""Recording-level HALO evidence engine: cosine retrieval, residual reranking, corrected 1-NN.

One query row and one memory row represent one six-second model window.  Every memory row receives a
small learned correction conditioned on the query and memory signal/configuration pair.  Candidate
scores are maxima over corrected, label-compatible neighbors.  A smooth-max surrogate supplies
gradients to all rows during training; the forward value is the exact corrected-nearest score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import AttentionSpec
from .evidence_reranker import EvidenceReranker, EvidenceRerankerConfig
from .rows import SensorRows


@dataclass(frozen=True)
class EngineConfig:
    spec: AttentionSpec = field(default_factory=AttentionSpec)
    trunk_layers: int = 3
    semantic_scale: float = 1.0
    surrogate_temperature: float = 0.05
    telemetry_neighbors: int = 8
    reranker: EvidenceRerankerConfig = field(default_factory=EvidenceRerankerConfig)

    def __post_init__(self) -> None:
        if self.semantic_scale <= 0.0:
            raise ValueError("semantic_scale must be positive")
        if self.surrogate_temperature <= 0.0:
            raise ValueError("surrogate_temperature must be positive")
        if self.telemetry_neighbors < 1:
            raise ValueError("telemetry_neighbors must be positive")


def _pad_first(tensors: list[torch.Tensor], fill: float | int = 0) -> torch.Tensor:
    if not tensors:
        raise ValueError("cannot pad an empty tensor list")
    width = max(tensor.shape[0] for tensor in tensors)
    out = tensors[0].new_full((len(tensors), width, *tensors[0].shape[1:]), fill)
    for index, tensor in enumerate(tensors):
        out[index, :tensor.shape[0]] = tensor
    return out


def enrolled_1nn_logits(
    scores: torch.Tensor,                 # (Q,M)
    bound_candidate: torch.Tensor,        # (M,), -1 = corpus
    n_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact support-only 1-NN reference, with an explicit candidate availability mask."""
    Q, M = scores.shape
    bound = bound_candidate.to(scores.device)
    if bound.shape != (M,) or bool(bound.ge(n_candidates).any()):
        raise ValueError("enrollment bindings disagree with memory/candidate dimensions")
    enrolled = bound.ge(0)
    slot = bound.clamp_min(0).unsqueeze(0).expand(Q, -1)
    logits = scores.new_full((Q, n_candidates), float("-inf"))
    source = scores.masked_fill(~enrolled.unsqueeze(0), float("-inf"))
    logits.scatter_reduce_(1, slot, source, reduce="amax", include_self=True)
    available = (
        F.one_hot(bound.clamp_min(0), n_candidates).bool()
        & enrolled.unsqueeze(1)
    ).any(dim=0).unsqueeze(0).expand(Q, -1)
    logits = torch.where(available, logits, torch.full_like(logits, -1e4))
    logits = torch.where(available.any(dim=1, keepdim=True), logits, torch.zeros_like(logits))
    return logits, available


class EvidenceEngine(nn.Module):
    """Residual all-memory reranking with a corrected-nearest readout."""

    def __init__(self, encoder: nn.Module | None, cfg: EngineConfig | None = None):
        super().__init__()
        self.cfg = cfg or EngineConfig()
        self.encoder = encoder
        self.reranker = EvidenceReranker(self.cfg.spec, self.cfg.reranker)

    @staticmethod
    def _validate_recordings(rows: SensorRows, name: str) -> None:
        if rows.source_window is None:
            raise ValueError(f"{name} rows require source_window identity")
        if len(torch.unique(rows.source_window)) != len(rows.feature):
            raise ValueError(
                f"{name} contains multiple rows for one recording; the active engine accepts "
                "exactly one pooled row per six-second window"
            )
        if rows.feature.dim() != 2 or rows.descriptor.dim() != 2:
            raise ValueError(f"{name} features and descriptors must be matrices")
        if rows.feature.shape[0] != rows.descriptor.shape[0]:
            raise ValueError(f"{name} feature and descriptor row counts disagree")

    def _compatibility(
        self,
        memory_label: torch.Tensor,          # (E,M)
        memory_bound: torch.Tensor,          # (E,M)
        memory_mask: torch.Tensor,           # (E,M)
        candidate_text: torch.Tensor,        # (E,C,Z)
        label_text: torch.Tensor,            # (V,Z)
    ) -> torch.Tensor:
        """Candidate-specific score offsets, never an evidence vote.

        Corpus rows pay a penalty according to the semantic distance between their canonical label
        and each candidate.  Enrolled rows are bound exactly to their episode-local candidate.  The
        offsets are added before taking a nearest-neighbor maximum; rows never distribute normalized
        probability mass across candidates or across the memory bank.
        """
        E, M = memory_label.shape
        C = candidate_text.shape[1]
        unbound = memory_bound.lt(0) & memory_mask
        if bool((unbound & memory_label.lt(0)).any()):
            raise ValueError("an unbound corpus recording has no canonical label")
        if bool((memory_bound >= C).any()):
            raise ValueError("an enrolled recording is bound outside the candidate set")
        canonical = F.normalize(label_text[memory_label.clamp_min(0)].float(), dim=-1)
        candidate = F.normalize(candidate_text.float(), dim=-1)
        semantic = torch.bmm(canonical, candidate.transpose(1, 2))
        corpus_offset = self.cfg.semantic_scale * (semantic - 1.0)
        enrolled_offset = corpus_offset.new_full((E, M, C), float("-inf"))
        enrolled_offset.scatter_(2, memory_bound.clamp_min(0).unsqueeze(-1), 0.0)
        offset = torch.where(memory_bound.unsqueeze(-1).ge(0), enrolled_offset, corpus_offset)
        return offset.masked_fill(~memory_mask.unsqueeze(-1), float("-inf"))

    def _candidate_nearest(
        self,
        score: torch.Tensor,                 # (E,Q,M)
        compatibility: torch.Tensor,         # (E,M,C)
        memory_mask: torch.Tensor,           # (E,M)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact nearest forward value with an all-row smooth backward surrogate."""
        joint = score.unsqueeze(-1) + compatibility.unsqueeze(1)       # (E,Q,M,C)
        joint = joint.masked_fill(~memory_mask[:, None, :, None], float("-inf"))
        available = torch.isfinite(joint).any(dim=2)
        hard, winner = joint.max(dim=2)
        tau = self.cfg.surrogate_temperature
        safe_joint = torch.where(
            available.unsqueeze(2), joint, torch.zeros_like(joint),
        )
        soft = tau * torch.logsumexp(safe_joint / tau, dim=2)
        disabled = torch.full_like(hard, -1e4)
        hard = torch.where(available, hard, disabled)
        soft = torch.where(available, soft, disabled)
        # Hard-forward / soft-backward: deployment and training make the same nearest decision, while
        # every finite row receives a gradient proportional to its smooth-max responsibility.
        logits = soft + (hard - soft).detach()
        weight = torch.softmax(safe_joint.float() / tau, dim=2)
        weight = torch.where(available.unsqueeze(2), weight, torch.zeros_like(weight))
        entropy = -(weight * weight.clamp_min(1e-12).log()).sum(dim=2)
        return logits, winner, entropy, weight

    def _forward_batched(
        self,
        query_feature: torch.Tensor,
        query_descriptor: torch.Tensor,
        memory_feature: torch.Tensor,
        memory_descriptor: torch.Tensor,
        memory_label: torch.Tensor,
        memory_bound: torch.Tensor,
        memory_mask: torch.Tensor,
        candidate_text: torch.Tensor,
        label_text: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        reranked = self.reranker.forward_batched(
            query_feature, query_descriptor, memory_feature, memory_descriptor,
            memory_bound.ge(0) & memory_mask,
        )
        compatibility = self._compatibility(
            memory_label, memory_bound, memory_mask, candidate_text, label_text,
        )
        base_logits, base_winner, base_entropy, _ = self._candidate_nearest(
            reranked["base_score"], compatibility, memory_mask,
        )
        logits, winner, entropy, responsibility = self._candidate_nearest(
            reranked["score"], compatibility, memory_mask,
        )
        return {
            **reranked,
            "logits": logits,
            "base_logits": base_logits,
            "winner": winner,
            "base_winner": base_winner,
            "neighbor_entropy": entropy,
            "base_neighbor_entropy": base_entropy,
            "neighbor_responsibility": responsibility,
        }

    def forward_many(
        self,
        queries: list[SensorRows],
        memories: list[SensorRows],
        candidate_text: torch.Tensor,       # (E,C,Z)
        label_text: torch.Tensor,
        *,
        generators=None,
        collect_stats: bool = False,
    ) -> list[dict]:
        del generators  # the active model contains no random identity channels
        E = len(queries)
        if E < 1 or len(memories) != E:
            raise ValueError("queries and memories must contain the same nonzero episode count")
        if candidate_text.dim() != 3 or candidate_text.shape[0] != E:
            raise ValueError("candidate_text must be (episode,candidate,text_dim)")
        for rows in queries:
            self._validate_recordings(rows, "query")
        for rows in memories:
            self._validate_recordings(rows, "memory")

        q_lengths = [len(rows.feature) for rows in queries]
        m_lengths = [len(rows.feature) for rows in memories]
        q_feature = _pad_first([rows.feature for rows in queries])
        q_descriptor = _pad_first([rows.descriptor for rows in queries])
        m_feature = _pad_first([rows.feature for rows in memories])
        m_descriptor = _pad_first([rows.descriptor for rows in memories])
        m_label = _pad_first([rows.label for rows in memories])
        m_bound = _pad_first([rows.enrolled_candidate for rows in memories], fill=-1)
        device = q_feature.device
        q_mask = torch.arange(q_feature.shape[1], device=device)[None, :] < torch.tensor(
            q_lengths, device=device,
        )[:, None]
        m_mask = torch.arange(m_feature.shape[1], device=device)[None, :] < torch.tensor(
            m_lengths, device=device,
        )[:, None]
        batched = self._forward_batched(
            q_feature, q_descriptor, m_feature, m_descriptor, m_label, m_bound, m_mask,
            candidate_text, label_text,
        )

        results = []
        C = candidate_text.shape[1]
        for episode, (query, memory, Q, M) in enumerate(zip(queries, memories, q_lengths, m_lengths)):
            base_score = batched["base_score"][episode, :Q, :M]
            corrected_score = batched["score"][episode, :Q, :M]
            one_nn, available = enrolled_1nn_logits(
                base_score, memory.enrolled_candidate, C,
            )
            k = min(self.cfg.telemetry_neighbors, M)
            selected = corrected_score.topk(k, dim=1).indices
            result = {
                "logits": batched["logits"][episode, :Q],
                "base_logits": batched["base_logits"][episode, :Q],
                "base_mass": torch.softmax(batched["base_logits"][episode, :Q], dim=-1),
                "residual_logits": (
                    batched["logits"][episode, :Q] - batched["base_logits"][episode, :Q]
                ),
                "score_correction": batched["score_correction"][episode, :Q, :M],
                "scores": corrected_score,
                "base_scores": base_score,
                "recording_scores": corrected_score,
                "enrolled_1nn_logits": one_nn,
                "enrolled_1nn_available": available,
                "selected": selected,
                "winner": batched["winner"][episode, :Q, :C],
                "query_window": query.source_window,
                "query_inverse": torch.arange(Q, device=device),
            }
            if collect_stats:
                result["stats"] = self._stats(
                    memory, base_score, corrected_score,
                    batched["score_correction"][episode, :Q, :M],
                    batched["descriptor_cosine"][episode, :Q, :M],
                    batched["neighbor_entropy"][episode, :Q],
                )
                result["stats"]["engine/mean_abs_logit_change"] = float(
                    result["residual_logits"].detach().abs().mean()
                )
            results.append(result)
        return results

    def forward(
        self,
        query: SensorRows,
        memory: SensorRows,
        candidate_text: torch.Tensor,
        label_text: torch.Tensor,
        *,
        generator=None,
        collect_stats: bool = False,
        **_unused,
    ) -> dict:
        return self.forward_many(
            [query], [memory], candidate_text.unsqueeze(0), label_text,
            generators=[generator], collect_stats=collect_stats,
        )[0]

    def _stats(
        self,
        memory: SensorRows,
        base_score: torch.Tensor,
        corrected_score: torch.Tensor,
        correction: torch.Tensor,
        descriptor_cosine: torch.Tensor,
        neighbor_entropy: torch.Tensor,
    ) -> dict[str, float]:
        with torch.no_grad():
            enrolled = memory.enrolled_candidate.ge(0).unsqueeze(0).expand_as(correction)

            def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
                return float(value[mask].mean()) if bool(mask.any()) else 0.0

            top_before = base_score.argmax(dim=1)
            top_after = corrected_score.argmax(dim=1)
            stats = self.reranker.telemetry()
            stats.update({
                "retrieval/memory_recordings": float(base_score.shape[1]),
                "retrieval/bank_coverage": 1.0,
                "retrieval/base_score_std": float(base_score.std()),
                "retrieval/acquisition_cosine_mean": float(descriptor_cosine.mean()),
                "retrieval/top_recording_changed": float(top_before.ne(top_after).float().mean()),
                "reranker/correction_mean": float(correction.mean()),
                "reranker/correction_std": float(correction.std()),
                "reranker/correction_abs_mean": float(correction.abs().mean()),
                "reranker/correction_abs_max": float(correction.abs().max()),
                "reranker/enrolled_correction_mean": masked_mean(correction, enrolled),
                "reranker/corpus_correction_mean": masked_mean(correction, ~enrolled),
                "reranker/effective_neighbors": float(neighbor_entropy.exp().mean()),
            })
            return stats

    def parameter_report(self) -> dict[str, int]:
        report: dict[str, int] = {}
        if self.encoder is not None:
            for name, module in self.encoder.named_children():
                if name == "text_encoder":
                    continue
                count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if count:
                    report[f"encoder.{name}"] = count
        report["reranker"] = sum(p.numel() for p in self.reranker.parameters() if p.requires_grad)
        report["TOTAL"] = sum(report.values())
        return report

    def frozen_text_parameters(self) -> int:
        text_encoder = getattr(self.encoder, "text_encoder", None)
        if text_encoder is None:
            return 0
        if getattr(text_encoder, "_model", None) is None:
            text_encoder._init_model()
        return sum(p.numel() for p in text_encoder._model.parameters())
