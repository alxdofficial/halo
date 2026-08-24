"""Compact HALO evidence engine: retrieve, rerank evidence rows, and vote.

The closed-form path is a jointly normalized retrieval vote. The learned path is a recording-level
scalar correction to each shortlisted evidence row. It begins close to the retrieval baseline but
with live gradients through the complete reranker on the first optimizer step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import AttentionSpec
from .evidence_mixer import UNBOUND_SLOT, EvidenceMixer, EvidenceMixerConfig
from .evidence_reranker import EvidenceReranker, EvidenceRerankerConfig
from .retrieval_scorer import PairScorer, PairScorerConfig, physics_violation_rate
from .rows import SensorRows, evidence_label_tokens


@dataclass(frozen=True)
class EngineConfig:
    spec: AttentionSpec = field(default_factory=AttentionSpec)
    trunk_layers: int = 3
    top_k: int = 64
    scorer: PairScorerConfig = field(default_factory=PairScorerConfig)
    mixer: EvidenceMixerConfig = field(default_factory=EvidenceMixerConfig)
    reranker: EvidenceRerankerConfig = field(default_factory=EvidenceRerankerConfig)
    mixing: str = "rerank"                      # rerank | attention (historical) | off
    #: Full-bank voting is the training default: every row participates and receives credit. The
    #: learned reranker remains bounded to top_k globally retrieved rows per recording.
    vote_scope: str = "bank"                    # topk | bank

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.mixing not in ("rerank", "attention", "off"):
            raise ValueError("mixing must be 'rerank', 'attention', or 'off'")
        if self.vote_scope not in ("topk", "bank"):
            raise ValueError("vote_scope must be 'topk' or 'bank'")
        groups = self.reranker.n_groups if self.mixing == "rerank" else self.mixer.n_groups
        if self.mixing != "off" and groups < self.top_k + 2:
            raise ValueError(
                f"top_k={self.top_k} requires at least {self.top_k + 2} co-membership groups"
            )


def vote(
    log_weight: torch.Tensor,        # (Q, M)
    label_vector: torch.Tensor,      # (M, Z) or (Q, M, Z)
    candidate_text: torch.Tensor,    # (C, Z)
    bound_candidate: torch.Tensor,   # (M,) or (Q, M), -1 = corpus row
) -> torch.Tensor:
    """A jointly normalized candidate distribution for each query row.

    Retrieval allocates one finite unit of mass over memory rows. Each corpus row distributes its
    mass over candidates according to normalized positive label-text affinity; an enrolled row gives
    all of its mass to the candidate name assigned at enrollment. Consequently every output row sums
    to one and candidates compete for the same evidence.
    """
    if log_weight.dim() != 2:
        raise ValueError(f"log_weight must be (Q, M), got {tuple(log_weight.shape)}")
    Q, M = log_weight.shape
    C = candidate_text.shape[0]
    with torch.autocast(device_type=log_weight.device.type, enabled=False):
        scores = log_weight.float()
        labels = label_vector.float()
        bound = bound_candidate.to(scores.device)
        if bool(bound.ge(C).any()):
            raise ValueError("an enrolled row is bound outside the candidate set")

        # Full-bank voting normally shares one memory across every query row. Its label-to-candidate
        # geometry is therefore (M, C), not (Q, M, C). Computing it once avoids repeating the same
        # 384-d text dot products for every patch while preserving gradients through retrieval.
        shared_memory = labels.dim() == 2 and bound.dim() == 1
        if shared_memory:
            if labels.shape[0] != M or bound.shape != (M,):
                raise ValueError("memory labels/bindings disagree with retrieval width")
            affinity = (labels @ candidate_text.float().T).clamp_min(0.0)
        else:
            if labels.dim() == 2:
                labels = labels.unsqueeze(0).expand(Q, M, -1)
            if bound.dim() == 1:
                bound = bound.unsqueeze(0).expand(Q, M)
            if labels.shape[:2] != (Q, M) or bound.shape != (Q, M):
                raise ValueError(
                    "label vectors/bindings and retrieval weights disagree on query/memory shape"
                )
            affinity = torch.einsum(
                "qmz,cz->qmc", labels, candidate_text.float(),
            ).clamp_min(0.0)
        total = affinity.sum(dim=-1, keepdim=True)
        corpus_vote = torch.where(
            total.gt(1e-8), affinity / total.clamp_min(1e-8),
            torch.full_like(affinity, 1.0 / C),
        )
        enrolled_vote = F.one_hot(bound.clamp_min(0), C).to(corpus_vote.dtype)
        row_vote = torch.where(bound.unsqueeze(-1).ge(0), enrolled_vote, corpus_vote)
        row_weight = torch.softmax(scores, dim=1)
        mass = row_weight @ row_vote if shared_memory else torch.einsum(
            "qm,qmc->qc", row_weight, row_vote,
        )
        return mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-8)


def enrolled_1nn_logits(
    recording_scores: torch.Tensor,       # (W, M)
    bound_candidate: torch.Tensor,        # (M,), -1 = ordinary corpus row
    n_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest enrolled-row score per candidate, plus the candidates that have support.

    This is a reference rule, not part of HALO's prediction path. Missing candidates receive a
    finite sentinel so cross-entropy remains numerically defined, while ``available`` tells callers
    which query targets admit a real enrolled-1NN comparison. A recording with no enrollment gets
    uniform zero logits and an all-false availability mask.
    """
    if recording_scores.dim() != 2:
        raise ValueError("recording_scores must be (recording, memory)")
    W, M = recording_scores.shape
    bound = bound_candidate.to(recording_scores.device)
    if bound.shape != (M,):
        raise ValueError("enrollment bindings and memory width disagree")
    if bool(bound.ge(n_candidates).any()):
        raise ValueError("an enrolled row is bound outside the candidate set")
    logits = recording_scores.new_full((W, n_candidates), float("-inf"))
    enrolled = bound.ge(0)
    slot = bound.clamp_min(0).unsqueeze(0).expand(W, -1)
    source = recording_scores.masked_fill(~enrolled.unsqueeze(0), float("-inf"))
    logits.scatter_reduce_(1, slot, source, reduce="amax", include_self=True)
    candidate_has_support = (
        F.one_hot(bound.clamp_min(0), n_candidates).bool()
        & enrolled.unsqueeze(1)
    ).any(dim=0)
    available = candidate_has_support.unsqueeze(0).expand(W, -1)
    # Keep the reference finite even in k=0 episodes. Consumers must still use ``available`` and
    # never interpret these fallback values as actual nearest-neighbour evidence.
    logits = torch.where(available, logits, torch.full_like(logits, -1e4))
    logits = torch.where(available.any(dim=1, keepdim=True), logits, torch.zeros_like(logits))
    return logits, available


def _pad_first(tensors: list[torch.Tensor], fill: float | int = 0) -> torch.Tensor:
    """Stack row tensors after padding only their leading dimension."""
    if not tensors:
        raise ValueError("cannot pad an empty tensor list")
    width = max(tensor.shape[0] for tensor in tensors)
    shape = (len(tensors), width, *tensors[0].shape[1:])
    out = tensors[0].new_full(shape, fill)
    for index, tensor in enumerate(tensors):
        out[index, :tensor.shape[0]] = tensor
    return out


def batched_vote(
    log_weight: torch.Tensor,       # (E, Q, M)
    label_vector: torch.Tensor,     # (E, M, Z)
    candidate_text: torch.Tensor,   # (E, C, Z)
    bound_candidate: torch.Tensor,  # (E, M)
    memory_mask: torch.Tensor,      # (E, M)
) -> torch.Tensor:
    """The fixed evidence vote for a batch of semantically independent episodes."""
    if log_weight.dim() != 3:
        raise ValueError("batched log weights must be (episode, query, memory)")
    E, _Q, M = log_weight.shape
    if label_vector.shape[:2] != (E, M) or bound_candidate.shape != (E, M):
        raise ValueError("batched evidence labels/bindings disagree with retrieval scores")
    if memory_mask.shape != (E, M) or candidate_text.shape[0] != E:
        raise ValueError("batched episode masks/candidates disagree with retrieval scores")
    C = candidate_text.shape[1]
    with torch.autocast(device_type=log_weight.device.type, enabled=False):
        scores = log_weight.float().masked_fill(~memory_mask[:, None, :], float("-inf"))
        affinity = torch.bmm(
            label_vector.float(), candidate_text.float().transpose(1, 2),
        ).clamp_min(0.0)
        total = affinity.sum(dim=-1, keepdim=True)
        corpus_vote = torch.where(
            total.gt(1e-8), affinity / total.clamp_min(1e-8),
            torch.full_like(affinity, 1.0 / C),
        )
        bound = bound_candidate.to(scores.device)
        enrolled_vote = F.one_hot(bound.clamp_min(0), C).to(corpus_vote.dtype)
        row_vote = torch.where(bound.unsqueeze(-1).ge(0), enrolled_vote, corpus_vote)
        row_weight = torch.softmax(scores, dim=-1)
        mass = torch.bmm(row_weight, row_vote)
        return mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class EvidenceEngine(nn.Module):
    """Retrieval plus one recording-level evidence set and a candidate residual."""

    def __init__(self, encoder: nn.Module | None, cfg: EngineConfig | None = None):
        super().__init__()
        self.cfg = cfg or EngineConfig()
        self.encoder = encoder
        self.scorer = PairScorer(self.cfg.spec, self.cfg.scorer)
        self.mixer = (EvidenceMixer(self.cfg.spec, self.cfg.mixer)
                      if self.cfg.mixing == "attention" else None)
        self.reranker = (EvidenceReranker(self.cfg.spec, self.cfg.reranker)
                         if self.cfg.mixing == "rerank" else None)

    @staticmethod
    def _recording_layout(query: SensorRows) -> tuple[torch.Tensor, torch.Tensor]:
        if query.source_window is None:
            raise ValueError("recording-level mixing requires query source_window identity")
        window, inverse = torch.unique(
            query.source_window.to(query.feature.device), sorted=True, return_inverse=True,
        )
        if not len(window):
            raise ValueError("an evidence-engine query cannot be empty")
        return window, inverse

    @staticmethod
    def _pack_query(
        query: SensorRows, inverse: torch.Tensor, n_recordings: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = torch.bincount(inverse, minlength=n_recordings)
        width = int(counts.max())
        feature = query.feature.new_zeros((n_recordings, width, query.feature.shape[-1]))
        descriptor = query.descriptor.new_zeros(
            (n_recordings, width, query.descriptor.shape[-1]),
        )
        mask = torch.zeros((n_recordings, width), dtype=torch.bool, device=query.feature.device)
        order = torch.argsort(inverse, stable=True)
        group = inverse[order]
        offsets = torch.cumsum(counts, dim=0) - counts
        within_group = torch.arange(len(inverse), device=inverse.device) - torch.repeat_interleave(
            offsets, counts, output_size=len(inverse),
        )
        feature[group, within_group] = query.feature[order]
        descriptor[group, within_group] = query.descriptor[order]
        mask[group, within_group] = True
        return feature, descriptor, mask

    @staticmethod
    def _recording_scores(
        scores: torch.Tensor, inverse: torch.Tensor, n_recordings: int,
    ) -> torch.Tensor:
        # A memory row belongs in the recording shortlist when any patch/sensor row retrieves it.
        out = scores.new_full((n_recordings, scores.shape[1]), float("-inf"))
        index = inverse.unsqueeze(1).expand_as(scores)
        return out.scatter_reduce(0, index, scores, reduce="amax", include_self=True)

    def _identity_channels(
        self,
        memory: SensorRows,
        selected: torch.Tensor,          # (W, K)
        n_candidates: int,
        generator: torch.Generator | None,
    ):
        device = selected.device
        reasoner = self.reranker if self.reranker is not None else self.mixer
        if reasoner is None:
            raise RuntimeError("identity channels requested while evidence reasoning is disabled")
        n_groups = reasoner.cfg.n_groups
        perm_device = generator.device if generator is not None else device
        if memory.source_window is None:
            raise ValueError("the evidence reasoner requires memory source_window identity")
        windows = memory.source_window.to(device)[selected]
        ordered, order = windows.sort(dim=1)
        starts = torch.ones_like(ordered, dtype=torch.bool)
        starts[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
        dense = torch.empty_like(ordered).scatter_(1, order, starts.cumsum(dim=1) - 1)
        # Each shortlist has at most K distinct recordings, and EngineConfig already proves that
        # n_groups - 2 >= configured top_k >= K. Avoid a device synchronization to re-check it on
        # every episode.
        group_perm = (
            torch.randperm(n_groups - 2, device=perm_device, generator=generator).to(device) + 2
        )
        evidence_group = group_perm[dense]
        bound = memory.enrolled_candidate.to(device)[selected]
        if self.reranker is not None:
            return None, None, evidence_group, bound
        n_slots = reasoner.cfg.n_slots
        if n_candidates >= n_slots:
            raise ValueError(
                f"episode has {n_candidates} candidates but the historical mixer has "
                f"{n_slots} slots"
            )
        candidate_slot = (
            torch.randperm(n_slots - 1, device=perm_device, generator=generator).to(device) + 1
        )[:n_candidates]
        evidence_slot = torch.where(
            bound.ge(0), candidate_slot[bound.clamp_min(0)],
            torch.full_like(bound, UNBOUND_SLOT),
        )
        return candidate_slot, evidence_slot, evidence_group, bound

    def forward(
        self,
        query: SensorRows,
        memory: SensorRows,
        candidate_text: torch.Tensor,
        label_text: torch.Tensor,
        *,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
        collect_stats: bool = False,
        vote_scope: str | None = None,
    ) -> dict:
        """Return one candidate-logit vector per unique query recording."""
        scores = self.scorer(
            query.feature, query.descriptor, memory.feature, memory.descriptor,
        )
        query_window, inverse = self._recording_layout(query)
        n_recordings = len(query_window)
        recording_scores = self._recording_scores(scores, inverse, n_recordings)
        with torch.no_grad():
            one_nn_logits, one_nn_available = enrolled_1nn_logits(
                recording_scores, memory.enrolled_candidate, candidate_text.shape[0],
            )
        k = min(self.cfg.top_k if top_k is None else int(top_k), len(memory.feature))
        selected = PairScorer.select(recording_scores, k)
        row_label_text = evidence_label_tokens(memory, candidate_text, label_text)
        scope = self.cfg.vote_scope if vote_scope is None else vote_scope
        if scope not in ("topk", "bank"):
            raise ValueError("vote_scope must be 'topk' or 'bank'")

        if scope == "bank":
            row_mass = vote(scores, row_label_text, candidate_text, memory.enrolled_candidate)
        else:
            row_selected = selected[inverse]
            row_mass = vote(
                scores.gather(1, row_selected), row_label_text[row_selected], candidate_text,
                memory.enrolled_candidate.to(scores.device)[row_selected],
            )
        base_mass = row_mass.new_zeros((n_recordings, candidate_text.shape[0]))
        base_mass.index_add_(0, inverse, row_mass)
        counts = torch.bincount(inverse, minlength=n_recordings).to(base_mass.dtype).unsqueeze(1)
        base_mass = base_mass / counts.clamp_min(1)
        base_logits = base_mass.clamp_min(1e-8).log()

        residual = torch.zeros_like(base_logits)
        score_correction = recording_scores.new_zeros(selected.shape)
        mixed = None
        if self.reranker is not None or self.mixer is not None:
            candidate_slot, evidence_slot, evidence_group, bound = self._identity_channels(
                memory, selected, candidate_text.shape[0], generator,
            )
            query_feature, query_descriptor, query_mask = self._pack_query(
                query, inverse, n_recordings,
            )
            reasoner = self.reranker if self.reranker is not None else self.mixer
            reasoner_kwargs = dict(
                retrieval_score=recording_scores.gather(1, selected),
                candidate_text=candidate_text,
                query_feature=query_feature,
                query_descriptor=query_descriptor,
                query_mask=query_mask,
                evidence_feature=memory.feature[selected],
                evidence_descriptor=memory.descriptor[selected],
                evidence_label_text=row_label_text[selected],
                evidence_group=evidence_group,
            )
            if self.mixer is not None:
                reasoner_kwargs.update(
                    candidate_slot=candidate_slot, evidence_slot=evidence_slot,
                )
            mixed = reasoner(**reasoner_kwargs)
            if self.reranker is not None:
                score_correction = mixed["score_correction"].float()
                row_selected = selected[inverse]
                row_correction = score_correction[inverse]
                if scope == "bank":
                    reranked_scores = scores.scatter_add(1, row_selected, row_correction)
                    reranked_row_mass = vote(
                        reranked_scores, row_label_text, candidate_text,
                        memory.enrolled_candidate,
                    )
                else:
                    reranked_row_mass = vote(
                        scores.gather(1, row_selected) + row_correction,
                        row_label_text[row_selected], candidate_text,
                        memory.enrolled_candidate.to(scores.device)[row_selected],
                    )
                reranked_mass = reranked_row_mass.new_zeros(base_mass.shape)
                reranked_mass.index_add_(0, inverse, reranked_row_mass)
                reranked_mass = reranked_mass / counts.clamp_min(1)
                logits = reranked_mass.clamp_min(1e-8).log()
                residual = logits - base_logits
            else:
                residual = mixed["residual_logits"].float()
                logits = base_logits + residual
        else:
            logits = base_logits
        result = {
            "logits": logits,
            "base_logits": base_logits,
            "base_mass": base_mass,
            "residual_logits": residual,
            "score_correction": score_correction,
            "scores": scores,
            "recording_scores": recording_scores,
            "enrolled_1nn_logits": one_nn_logits,
            "enrolled_1nn_available": one_nn_available,
            "selected": selected,
            "query_window": query_window,
            "query_inverse": inverse,
        }
        if collect_stats:
            result["stats"] = self._stats(
                query, memory, scores, selected, inverse, mixed, residual, score_correction,
            )
        return result

    def forward_many(
        self,
        queries: list[SensorRows],
        memories: list[SensorRows],
        candidate_text: torch.Tensor,       # (E, C, text_dim), same C within one step
        label_text: torch.Tensor,
        *,
        generators: list[torch.Generator | None] | None = None,
        collect_stats: bool = False,
    ) -> list[dict[str, torch.Tensor | dict[str, float]]]:
        """Vectorized active path for independent, same-C episodes.

        Padding exists only at the episode batch boundary. Masks keep every memory, candidate set,
        recording, vote, and loss independent; this method is mathematically the same operation as
        calling :meth:`forward` once per episode. Historical learned-scorer and candidate-residual
        mixer arms intentionally retain the sequential reference path.
        """
        E = len(queries)
        if E < 1 or len(memories) != E:
            raise ValueError("queries and memories must contain the same nonzero episode count")
        if candidate_text.dim() != 3 or candidate_text.shape[0] != E:
            raise ValueError("candidate_text must be (episode, candidate, text_dim)")
        if self.cfg.scorer.learned or self.reranker is None or self.mixer is not None:
            raise ValueError("forward_many supports the active fixed-cosine scalar-reranker path")
        if self.cfg.vote_scope != "bank":
            raise ValueError("forward_many requires the active full-bank vote")
        if generators is None:
            generators = [None] * E
        if len(generators) != E:
            raise ValueError("one optional random generator is required per episode")
        C = candidate_text.shape[1]
        if any(row.source_window is None for row in queries + memories):
            raise ValueError("vectorized evidence reasoning requires source-window identity")

        q_lengths = [len(row.feature) for row in queries]
        m_lengths = [len(row.feature) for row in memories]
        if min(m_lengths) < self.cfg.top_k:
            raise ValueError(
                "vectorized active episodes require every memory to contain the configured top-k"
            )
        q_feature = _pad_first([row.feature for row in queries])
        q_descriptor = _pad_first([row.descriptor for row in queries])
        m_feature = _pad_first([row.feature for row in memories])
        m_descriptor = _pad_first([row.descriptor for row in memories])
        m_label = _pad_first([row.label for row in memories])
        m_bound = _pad_first([row.enrolled_candidate for row in memories], fill=-1)
        device = q_feature.device
        q_mask = torch.arange(q_feature.shape[1], device=device)[None, :] < torch.tensor(
            q_lengths, device=device,
        )[:, None]
        m_mask = torch.arange(m_feature.shape[1], device=device)[None, :] < torch.tensor(
            m_lengths, device=device,
        )[:, None]

        with torch.autocast(device_type=device.type, enabled=False):
            scores = self.scorer.fixed_gain.float() * torch.bmm(
                F.normalize(q_feature.float(), dim=-1),
                F.normalize(m_feature.float(), dim=-1).transpose(1, 2),
            )
            scores = scores.masked_fill(~m_mask[:, None, :], float("-inf"))

        windows, inverses = zip(*(self._recording_layout(query) for query in queries))
        recording_counts = {len(window) for window in windows}
        if len(recording_counts) != 1:
            raise ValueError("vectorized episodes must carry the same number of query recordings")
        W = len(windows[0])
        inverse = _pad_first(list(inverses))
        recording_scores = scores.new_full((E, W, scores.shape[2]), float("-inf"))
        scatter_scores = scores.masked_fill(~q_mask[:, :, None], float("-inf"))
        recording_scores.scatter_reduce_(
            1, inverse.unsqueeze(-1).expand_as(scores), scatter_scores,
            reduce="amax", include_self=True,
        )
        k = self.cfg.top_k
        selected = recording_scores.topk(k, dim=-1).indices

        with torch.no_grad():
            one_nn = recording_scores.new_full((E, W, C), float("-inf"))
            slot = m_bound.clamp_min(0)[:, None, :].expand(E, W, -1)
            source = recording_scores.masked_fill(
                ~(m_bound.ge(0) & m_mask)[:, None, :], float("-inf"),
            )
            one_nn.scatter_reduce_(2, slot, source, reduce="amax", include_self=True)
            candidate_has_support = (
                F.one_hot(m_bound.clamp_min(0), C).bool()
                & (m_bound.ge(0) & m_mask).unsqueeze(-1)
            ).any(dim=1)
            one_nn_available = candidate_has_support[:, None, :].expand(E, W, C)
            one_nn = torch.where(one_nn_available, one_nn, torch.full_like(one_nn, -1e4))
            one_nn = torch.where(
                one_nn_available.any(dim=-1, keepdim=True), one_nn, torch.zeros_like(one_nn),
            )

        canonical = label_text[m_label.clamp_min(0)]
        enrolled = candidate_text.gather(
            1, m_bound.clamp_min(0).unsqueeze(-1).expand(-1, -1, candidate_text.shape[-1]),
        )
        row_label_text = torch.where(m_bound.unsqueeze(-1).ge(0), enrolled, canonical)
        base_row_mass = batched_vote(
            scores, row_label_text, candidate_text, m_bound, m_mask,
        )
        base_mass = scores.new_zeros((E, W, C))
        base_mass.scatter_add_(
            1, inverse.unsqueeze(-1).expand(-1, -1, C),
            base_row_mass * q_mask.unsqueeze(-1),
        )
        counts = scores.new_zeros((E, W, 1))
        counts.scatter_add_(1, inverse.unsqueeze(-1), q_mask.unsqueeze(-1).to(scores.dtype))
        base_mass = base_mass / counts.clamp_min(1)
        base_logits = base_mass.clamp_min(1e-8).log()

        packed_query = [self._pack_query(query, inv, W) for query, inv in zip(queries, inverses)]
        query_width = max(value[0].shape[1] for value in packed_query)

        def pad_recordings(values: list[torch.Tensor], fill=0):
            result = values[0].new_full((E, W, query_width, *values[0].shape[2:]), fill)
            for episode, value in enumerate(values):
                result[episode, :, :value.shape[1]] = value
            return result

        packed_feature = pad_recordings([value[0] for value in packed_query])
        packed_descriptor = pad_recordings([value[1] for value in packed_query])
        packed_mask = pad_recordings([value[2] for value in packed_query], fill=False)
        batch_index = torch.arange(E, device=device)[:, None, None]
        selected_feature = m_feature[batch_index, selected]
        selected_descriptor = m_descriptor[batch_index, selected]
        selected_label_text = row_label_text[batch_index, selected]
        groups = []
        for memory, chosen, generator in zip(memories, selected, generators):
            groups.append(self._identity_channels(memory, chosen, C, generator)[2])
        evidence_group = torch.stack(groups)

        mixed = self.reranker(
            retrieval_score=recording_scores.gather(2, selected).reshape(E * W, k),
            candidate_text=candidate_text[:, None].expand(-1, W, -1, -1).reshape(
                E * W, C, candidate_text.shape[-1],
            ),
            query_feature=packed_feature.reshape(E * W, query_width, -1),
            query_descriptor=packed_descriptor.reshape(E * W, query_width, -1),
            query_mask=packed_mask.reshape(E * W, query_width),
            evidence_feature=selected_feature.reshape(E * W, k, -1),
            evidence_descriptor=selected_descriptor.reshape(E * W, k, -1),
            evidence_label_text=selected_label_text.reshape(E * W, k, -1),
            evidence_group=evidence_group.reshape(E * W, k),
        )
        correction = mixed["score_correction"].reshape(E, W, k)
        row_selected = selected.gather(
            1, inverse.unsqueeze(-1).expand(-1, -1, k),
        )
        row_correction = correction.gather(
            1, inverse.unsqueeze(-1).expand(-1, -1, k),
        ) * q_mask.unsqueeze(-1)
        reranked_scores = scores.scatter_add(2, row_selected, row_correction)
        reranked_row_mass = batched_vote(
            reranked_scores, row_label_text, candidate_text, m_bound, m_mask,
        )
        reranked_mass = scores.new_zeros((E, W, C))
        reranked_mass.scatter_add_(
            1, inverse.unsqueeze(-1).expand(-1, -1, C),
            reranked_row_mass * q_mask.unsqueeze(-1),
        )
        reranked_mass = reranked_mass / counts.clamp_min(1)
        logits = reranked_mass.clamp_min(1e-8).log()
        residual = logits - base_logits

        results = []
        for episode in range(E):
            result = {
                "logits": logits[episode],
                "base_logits": base_logits[episode],
                "base_mass": base_mass[episode],
                "residual_logits": residual[episode],
                "score_correction": correction[episode],
                "scores": scores[episode, :q_lengths[episode], :m_lengths[episode]],
                "recording_scores": recording_scores[episode, :, :m_lengths[episode]],
                "enrolled_1nn_logits": one_nn[episode],
                "enrolled_1nn_available": one_nn_available[episode],
                "selected": selected[episode],
                "query_window": windows[episode],
                "query_inverse": inverses[episode],
            }
            if collect_stats:
                result["stats"] = self._stats(
                    queries[episode], memories[episode], result["scores"], result["selected"],
                    inverses[episode], None, result["residual_logits"],
                    result["score_correction"],
                )
            results.append(result)
        return results

    def _stats(self, query, memory, scores, selected, inverse, mixed, residual,
               score_correction):
        with torch.no_grad():
            reasoner = self.reranker if self.reranker is not None else self.mixer
            stats = dict(reasoner.telemetry()) if reasoner is not None else {}
            if self.cfg.scorer.learned:
                stats["retrieval/base_gain"] = float(self.scorer.base_gain)
                stats["retrieval/residual_gain"] = float(self.scorer.residual_gain)
            else:
                stats["retrieval/fixed_gain"] = float(self.scorer.fixed_gain)
            bound = memory.enrolled_candidate.to(selected.device)[selected]
            stats["retrieval/enrolled_share"] = float(bound.ge(0).float().mean())
            reached = torch.zeros(len(memory.feature), dtype=torch.bool, device=selected.device)
            reached[selected.reshape(-1)] = True
            stats["retrieval/bank_coverage"] = float(reached.float().mean())
            finite_scores = scores[scores.isfinite()]
            stats["retrieval/score_spread"] = (
                float(finite_scores.std()) if len(finite_scores) > 1 else 0.0
            )
            stats["engine/mean_abs_logit_change"] = float(residual.abs().mean())
            if self.reranker is not None:
                recording_scores = self._recording_scores(scores, inverse, selected.shape[0])
                selected_scores = recording_scores.gather(1, selected)
                base_weight = torch.softmax(
                    selected_scores, dim=1,
                )
                reranked_weight = torch.softmax(
                    selected_scores + score_correction, dim=1,
                )
                base_entropy = -(base_weight * base_weight.clamp_min(1e-12).log()).sum(dim=1)
                reranked_entropy = -(
                    reranked_weight * reranked_weight.clamp_min(1e-12).log()
                ).sum(dim=1)
                centered_score = selected_scores - selected_scores.mean(dim=1, keepdim=True)
                centered_correction = score_correction - score_correction.mean(
                    dim=1, keepdim=True,
                )
                correction_score_cosine = F.cosine_similarity(
                    centered_score, centered_correction, dim=1, eps=1e-8,
                )
                enrolled = bound.ge(0)

                def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
                    return (float(value[mask].mean()) if bool(mask.any()) else 0.0)

                stats.update({
                    "reranker/correction_mean": float(score_correction.mean()),
                    "reranker/correction_std": float(score_correction.std()),
                    "reranker/correction_abs_mean": float(score_correction.abs().mean()),
                    "reranker/correction_abs_max": float(score_correction.abs().max()),
                    "reranker/correction_score_cosine": float(
                        correction_score_cosine.mean()
                    ),
                    "reranker/enrolled_correction_mean": masked_mean(
                        score_correction, enrolled,
                    ),
                    "reranker/corpus_correction_mean": masked_mean(
                        score_correction, ~enrolled,
                    ),
                    "reranker/weight_shift_l1": float(
                        (reranked_weight - base_weight).abs().sum(dim=1).mean()
                    ),
                    "reranker/evidence_entropy_before": float(base_entropy.mean()),
                    "reranker/evidence_entropy_after": float(reranked_entropy.mean()),
                    "reranker/effective_rows_before": float(base_entropy.exp().mean()),
                    "reranker/effective_rows_after": float(reranked_entropy.exp().mean()),
                    "reranker/enrolled_weight_before": float(
                        (base_weight * enrolled).sum(dim=1).mean()
                    ),
                    "reranker/enrolled_weight_after": float(
                        (reranked_weight * enrolled).sum(dim=1).mean()
                    ),
                })
            expanded = selected[inverse]
            stats.update(physics_violation_rate(
                expanded, query.modality, query.gravity, memory.modality, memory.gravity,
            ))
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
        report["scorer"] = sum(p.numel() for p in self.scorer.parameters() if p.requires_grad)
        report["mixer"] = (sum(p.numel() for p in self.mixer.parameters() if p.requires_grad)
                           if self.mixer is not None else 0)
        report["reranker"] = (
            sum(p.numel() for p in self.reranker.parameters() if p.requires_grad)
            if self.reranker is not None else 0
        )
        report["TOTAL"] = sum(report.values())
        return report

    def frozen_text_parameters(self) -> int:
        text_encoder = getattr(self.encoder, "text_encoder", None)
        if text_encoder is None:
            return 0
        if getattr(text_encoder, "_model", None) is None:
            text_encoder._init_model()
        return sum(p.numel() for p in text_encoder._model.parameters())
