"""The whole model, as one object: encode, retrieve, mix, vote.

    query window ─┐
                  ├─ SetTokenizerEncoder ──> feature + description per sensor row
    memory bank ──┘
                            │
                            ├─ PairScorer(q_feat, q_desc, m_feat, m_desc) ──> (Q, M) scores
                            │        top-k per query
                            ▼
                       EvidenceMixer ── self-attention over [candidates | query | retrieved rows]
                            │
                            ├─ log evidence weights (Q, k, C)
                            └─ label vectors, refined in text space (Q, k, 384)
                            ▼
                  vote: softmax over rows, cosine against candidate names ──> (Q, C) logits

WHY THE PIECES ARE ONE MODULE
-----------------------------
Every capacity question used to have three answers, because the encoder, the retrieval rule and the
readout were separate objects configured separately. ``EvidenceEngine`` holds all three, they share
one :class:`~model.blocks.AttentionSpec`, and :meth:`parameter_report` prints what the model costs
in one table. Growing the model is one number.

WHAT HAS NO PARAMETERS, AND WHY THAT MATTERS
--------------------------------------------
The map from evidence to a candidate score is a cosine between text vectors. There are no
per-candidate weights anywhere in this file. An unseen candidate is therefore scored by exactly the
same operation as a seen one — closed-vocabulary collapse is structurally impossible rather than
merely discouraged, which is the property that a learned decoder with a candidate-indexed output
layer cannot have and that this design exists to keep.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import AttentionSpec
from .evidence_mixer import UNBOUND_SLOT, EvidenceMixer, EvidenceMixerConfig
from .retrieval_scorer import PairScorer, PairScorerConfig, physics_violation_rate
from .rows import SensorRows, evidence_label_tokens


@dataclass(frozen=True)
class EngineConfig:
    """Everything that decides the model's size and its retrieval depth."""

    spec: AttentionSpec = field(default_factory=AttentionSpec)
    trunk_layers: int = 3
    #: Rows handed to the mixer per query. Identical in training and deployment — the previous
    #: design trained on the whole bank and deployed on a top-k slice, so the objective and the
    #: deployed rule were different functions.
    #:
    #: 64 rather than 32 because selection is hard: a row outside the top-k receives no gradient on
    #: that step, so k is the width of the live path back to the encoder as much as it is a
    #: retrieval depth. The cost is close to free — the stack is launch-bound, and the mixer's
    #: sequence grows from 106 to 202 tokens, which is still short.
    top_k: int = 64
    scorer: PairScorerConfig = field(default_factory=PairScorerConfig)
    mixer: EvidenceMixerConfig = field(default_factory=EvidenceMixerConfig)
    #: "off" removes the mixer entirely: the evidence weight is the retrieval score alone and the
    #: label vectors are the frozen row text. With scorer.learned=False as well, the whole model
    #: reduces to the closed-form cosine-retrieval ConSE rule and nothing is learned downstream of
    #: the encoder. The ablation ladder walks this back one stage at a time.
    mixing: str = "attention"                   # attention | off

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.mixing not in ("attention", "off"):
            raise ValueError("mixing must be 'attention' or 'off'")
        if self.mixing == "off":
            return                              # no mixer, so no group vocabulary to check
        # Every retrieved row can be its own recording. Group 0 means "no recording" (candidate
        # labels), group 1 is the query recording, and evidence groups start at 2.
        # Checking here means a bad combination fails at construction rather than on the first
        # forward, halfway into a run.
        if self.mixer.n_groups < self.top_k + 2:
            raise ValueError(
                f"top_k={self.top_k} rows can span {self.top_k} recordings but the mixer has "
                f"{self.mixer.n_groups} co-membership groups and reserves one neutral group plus "
                "one query group; set EvidenceMixerConfig.n_groups >= top_k + 2"
            )

    def with_readout(self, readout: str) -> "EngineConfig":
        return replace(self, mixer=replace(self.mixer, readout=readout))


def vote(
    log_weight: torch.Tensor,        # (Q, k, C)
    label_vector: torch.Tensor,      # (Q, k, text_dim) L2-normalised
    candidate_text: torch.Tensor,    # (C, text_dim) L2-normalised
    bound_candidate: torch.Tensor,   # (Q, k) candidate slot a row is enrolled to, -1 = corpus row
) -> torch.Tensor:
    """(Q, C) candidate logits.

    Every row votes through the rectified cosine between its label vector and the candidate's name.
    For a corpus row that is the ConSE bridge, and the only mechanism available with nothing
    enrolled. For an ENROLLED row the label vector IS its bound candidate's text, so the cosine is
    exactly 1 and this reproduces the identity vote it used to get from a special case.

    That special case is gone deliberately. It was numerically redundant, and being a constant it
    was unreachable by the "semantic" readout — support strength would have had no learned control
    at all under that arm. One path for every row also means one thing to reason about.

    An enrolled row is masked out of every candidate but its own. Letting it compete for a
    candidate it cannot support would spend softmax mass on guaranteed-zero evidence.
    """
    # Evidence normalization and the text-space cosine are numerically sensitive reductions, and
    # their output feeds log-mass cross entropy. They are cheap relative to attention and remain an
    # explicit FP32 island under mixed-precision training.
    with torch.autocast(device_type=log_weight.device.type, enabled=False):
        log_weight = log_weight.float()
        n_cand = candidate_text.shape[0]
        index = torch.arange(n_cand, device=log_weight.device).view(1, 1, n_cand)
        bound = bound_candidate.unsqueeze(-1)
        allowed = bound.lt(0) | bound.eq(index)

        masked = log_weight.masked_fill(~allowed, float("-inf"))
        # Include a fixed zero-logit, zero-vote null row. Without it, a candidate with one allowed
        # enrolled row receives softmax mass 1 regardless of whether retrieval scored that row well.
        # If every candidate has one such row, every logit becomes exactly 1 and the classification
        # loss has zero gradient with respect to retrieval. The null row keeps absolute score
        # calibrated and makes the no-evidence case finite without a special branch.
        null = log_weight.new_zeros((log_weight.shape[0], 1, n_cand))
        weight = torch.softmax(torch.cat([masked, null], dim=1), dim=1)[:, :-1]

        evidence = torch.einsum(
            "qkz,cz->qkc", label_vector.float(), candidate_text.float(),
        ).clamp_min(0.0)
        return (weight * evidence).sum(dim=1)


class EvidenceEngine(nn.Module):
    """Retrieval, mixing and voting over an episodic memory bank."""

    def __init__(self, encoder: nn.Module | None, cfg: EngineConfig | None = None):
        super().__init__()
        self.cfg = cfg or EngineConfig()
        self.encoder = encoder
        self.scorer = PairScorer(self.cfg.spec, self.cfg.scorer)
        self.mixer = (EvidenceMixer(self.cfg.spec, self.cfg.mixer)
                      if self.cfg.mixing == "attention" else None)

    # ---------------------------------------------------------------- identity channels
    def _identity_channels(
        self,
        memory: SensorRows,
        selected: torch.Tensor,          # (Q, k)
        n_candidates: int,
        generator: torch.Generator | None,
    ):
        """Per-episode randomised coreference slots and co-membership groups.

        Both are permuted every episode. A stable id would let the model memorise "slot 7 means
        walking", turning a relational channel into a closed vocabulary — the failure this design
        is built to avoid.
        """
        device = selected.device
        n_slots, n_groups = self.mixer.cfg.n_slots, self.mixer.cfg.n_groups
        if n_candidates >= n_slots:
            raise ValueError(
                f"episode has {n_candidates} candidates but the mixer has {n_slots} slots "
                "(slot 0 is reserved for unbound rows); raise EvidenceMixerConfig.n_slots"
            )
        # torch.randperm requires the generator and the output to share a device, and callers
        # legitimately pass a CPU generator with CUDA rows — torch.Generator() is CPU by default,
        # and seeding one is how deployment gets reproducible slot assignments. Draw on the
        # generator's device, then move.
        perm_device = generator.device if generator is not None else device
        slot_perm = (torch.randperm(n_slots - 1, device=perm_device, generator=generator)
                     .to(device) + 1)
        candidate_slot = slot_perm[:n_candidates]

        if memory.source_window is None:
            raise ValueError(
                "the mixer needs source_window to tell co-membership from coincidence; without it "
                "every row would look like its own recording"
            )
        # Group ids are compacted WITHIN each query's retrieved set, not across the bank. Two rows
        # only ever need distinguishable ids when they appear in the same sequence, so the id pool
        # has to cover k rows, not the whole bank — which is the difference between a 64-entry and
        # a 512-entry embedding table for exactly the same expressiveness.
        window = memory.source_window.to(device)[selected]                    # (Q, k)
        ordered, order = window.sort(dim=1)
        starts_run = torch.ones_like(ordered, dtype=torch.bool)
        starts_run[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
        dense = torch.empty_like(ordered).scatter_(
            1, order, starts_run.cumsum(dim=1) - 1,
        )
        distinct = int(dense.max()) + 1
        if distinct > n_groups - 2:
            raise ValueError(
                f"a query retrieved {distinct} distinct recordings but the mixer has "
                f"{n_groups} groups (group 0 is neutral and group 1 is the query recording); raise "
                "EvidenceMixerConfig.n_groups"
            )
        group_perm = (torch.randperm(n_groups - 2, device=perm_device, generator=generator)
                      .to(device) + 2)
        evidence_group = group_perm[dense]

        bound = memory.enrolled_candidate.to(device)[selected]                # (Q, k)
        evidence_slot = torch.where(
            bound.ge(0), candidate_slot[bound.clamp_min(0)],
            torch.full_like(bound, UNBOUND_SLOT),
        )
        return candidate_slot, evidence_slot, evidence_group, bound

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        query: SensorRows,
        memory: SensorRows,
        candidate_text: torch.Tensor,     # (C, 384) L2-normalised, episode-local names
        label_text: torch.Tensor,         # (V, 384) L2-normalised vocabulary
        *,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
        collect_stats: bool = False,
    ) -> dict:
        """(Q, C) logits per query sensor row, plus the diagnostics that make the run readable."""
        scores = self.scorer(
            query.feature, query.descriptor, memory.feature, memory.descriptor,
        )
        k = self.cfg.top_k if top_k is None else int(top_k)
        selected = PairScorer.select(scores, k)                               # (Q, k)
        picked_score = scores.gather(1, selected)

        row_label_text = evidence_label_tokens(memory, candidate_text, label_text)
        if self.mixer is None:
            bound = memory.enrolled_candidate.to(selected.device)[selected]
            mixed = {
                "log_weight": picked_score.unsqueeze(-1).expand(-1, -1, candidate_text.shape[0]),
                "label_vector": row_label_text[selected],
            }
            logits = vote(mixed["log_weight"], mixed["label_vector"], candidate_text, bound)
            result = {"logits": logits, "scores": scores, "selected": selected,
                      "log_weight": mixed["log_weight"]}
            if collect_stats:
                result["stats"] = self._stats(query, memory, scores, selected, bound, mixed)
            return result

        candidate_slot, evidence_slot, evidence_group, bound = self._identity_channels(
            memory, selected, candidate_text.shape[0], generator,
        )
        mixed = self.mixer(
            retrieval_score=picked_score,
            candidate_text=candidate_text,
            query_feature=query.feature,
            query_descriptor=query.descriptor,
            evidence_feature=memory.feature[selected],
            evidence_descriptor=memory.descriptor[selected],
            evidence_label_text=row_label_text[selected],
            candidate_slot=candidate_slot,
            evidence_slot=evidence_slot,
            evidence_group=evidence_group,
        )
        logits = vote(mixed["log_weight"], mixed["label_vector"], candidate_text, bound)

        result = {"logits": logits, "scores": scores, "selected": selected,
                  "log_weight": mixed["log_weight"]}
        if collect_stats:
            result["stats"] = self._stats(query, memory, scores, selected, bound, mixed)
        return result

    def _stats(self, query, memory, scores, selected, bound, mixed) -> dict[str, float]:
        with torch.no_grad():
            stats = dict(self.mixer.telemetry()) if self.mixer is not None else {}
            if self.cfg.scorer.learned:
                stats["retrieval/base_gain"] = float(self.scorer.base_gain)
                stats["retrieval/residual_gain"] = float(self.scorer.residual_gain)
            else:
                stats["retrieval/fixed_gain"] = float(self.scorer.fixed_gain)
            stats["retrieval/enrolled_share"] = float(bound.ge(0).float().mean())
            reached = torch.zeros(len(memory.feature), dtype=torch.bool, device=selected.device)
            reached[selected.reshape(-1)] = True
            # What fraction of the bank any query looked at. A collapsing retriever drives this
            # down, and a bank whose rows are never reachable is a bank that is partly wasted.
            stats["retrieval/bank_coverage"] = float(reached.float().mean())
            stats["retrieval/score_spread"] = float(scores.std())
            stats["mixer/mean_abs_correction"] = float(
                (mixed["log_weight"] - scores.gather(1, selected).unsqueeze(-1)).abs().mean()
            )
            stats.update(physics_violation_rate(
                selected, query.modality, query.gravity, memory.modality, memory.gravity,
            ))
        return stats

    # ---------------------------------------------------------------- budgeting
    def parameter_report(self) -> dict[str, int]:
        """Learnable parameters by part. The frozen text encoder is reported separately."""
        report: dict[str, int] = {}
        if self.encoder is not None:
            for name, module in self.encoder.named_children():
                if name == "text_encoder":
                    continue
                count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if count:
                    report[f"encoder.{name}"] = count
            loose = sum(p.numel() for n, p in self.encoder.named_parameters()
                        if p.requires_grad and "." not in n)
            if loose:
                report["encoder.<direct>"] = loose
        report["scorer"] = sum(p.numel() for p in self.scorer.parameters() if p.requires_grad)
        report["mixer"] = (sum(p.numel() for p in self.mixer.parameters() if p.requires_grad)
                           if self.mixer is not None else 0)
        report["TOTAL"] = sum(v for k, v in report.items() if k != "TOTAL")
        return report

    def frozen_text_parameters(self) -> int:
        """The shared, never-trained text tower.

        Reported separately from :meth:`parameter_report` because it is not capacity this model
        learned: it is the same off-the-shelf sentence encoder any text-conditioned baseline
        carries, it is identical across every HALO size, and it holds no task information. The
        module loads lazily, so this forces it rather than reporting zero for a tower that is
        obviously there.
        """
        text_encoder = getattr(self.encoder, "text_encoder", None)
        if text_encoder is None:
            return 0
        if getattr(text_encoder, "_model", None) is None:
            text_encoder._init_model()
        return sum(p.numel() for p in text_encoder._model.parameters())
