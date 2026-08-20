"""Self-attention over one query's retrieved evidence, and the readout it feeds.

WHERE THIS SITS
---------------
Retrieval (:mod:`model.evidence.retrieval_scorer`) has already scored every memory row against
every query and kept the top ``k`` for each. This module gives that shortlist context: the query it
was retrieved for, the descriptions of the acquisition configurations involved, the labels the
retrieved recordings carry, and the candidate names the episode is deciding between. Everything
attends to everything, once, in a sequence that belongs to a single query.

    per query:  [ CAND_1..CAND_C ][ QRY QDESC ][ EV_1..EV_k ][ EDESC_1..EDESC_k ][ ELBL_1..ELBL_k ]

ONE SEQUENCE PER QUERY, NOT ONE PER EPISODE
--------------------------------------------
The previous mixer built a single sequence containing every query in the episode alongside the
whole memory bank — roughly 1.8k tokens — which meant one query's refinement could depend on
another query's evidence. Episodes are a training construct; at deployment a query arrives alone.
So a representation that used its episode-mates was reading something that will not be there.

Batching over queries instead makes the sequence ``C + 2 + 3k`` — about 114 tokens at C=16, k=32 —
and it is *cheaper*: 125 sequences of 114 tokens is roughly half the attention pair-ops of one
sequence of 1.8k, because cost is quadratic in length and only linear in batch.

ROLES AND THE TWO IDENTITY CHANNELS
------------------------------------
Attention has no idea what a vector *is* unless told. Six roles, and two independent identity
channels that must not be collapsed into one:

``slot``   COREFERENCE — "these tokens refer to the same concept". An enrolled evidence row shares
           a slot with the candidate it is bound to.
``group``  CO-MEMBERSHIP — "these tokens came from the same recording". The two sensors of one
           window share a group, which is where cross-sensor fusion now happens: the trunk encodes
           each sensor in isolation, and the mixer relates them here, next to their labels.

Slot ids are randomised per episode by the caller, so a slot can only ever mean "same referent",
never a stable label identity to memorise.

TWO READOUTS, AND THEY ARE ALTERNATIVES
---------------------------------------
Mixing is the same either way. What differs is what attention is allowed to emit, and the two are
genuinely different models rather than a feature and its extension.

``readout="weights"`` emits a scalar per (query, row, candidate), added to the retrieval score in
log-weight units. Two low-rank forms, both routed THROUGH an evidence row:

    evidence_candidate_form         evidence x candidate
                                    "does this row speak to this candidate"
    query_evidence_candidate_form   query x evidence x candidate
                                    "for THIS query, is this row informative about THIS candidate"

There is deliberately no query-x-candidate form. That would be a learned path from the query
straight to a candidate score with no evidence in between — a closed-vocabulary classifier bolted
onto a retrieval model, and the exact channel through which an earlier decoder collapsed below the
chance floor.

``readout="semantic"`` emits no scalar at all. It displaces each retrieved row's label vector
inside the frozen text space, so context can say "this row's 'walking' is, here, a treadmill sort
of walking", and the evidence weight is the retrieval score alone. This is strictly more expressive
per row — a 384-d displacement against a scalar — and it subsumes reweighting, because a vector
pushed equidistant from every candidate contributes equally to all of them.

It also changes the shape of the computation, which is the reason to measure rather than assume.
With the weight shared across candidates the readout collapses:

    logit_c = sum_m w_m <lhat_qm, t_c> = < sum_m w_m lhat_qm , t_c > = <v_q, t_c>

The episode becomes ONE vector scored against every candidate, which is a CLIP-shaped model. It
stays evidence-grounded only while ``semantic_gain`` is small, because ``v_q`` then lives in the
convex hull of vectors near actually-retrieved labels. The gain is learned, so that is a soft
constraint, and it is logged for exactly that reason.

Neither readout has per-candidate parameters. An unseen candidate is scored by the same operation
as a seen one under both.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import AttentionSpec, ScaledSum, SetAttentionStack

(ROLE_CANDIDATE, ROLE_QUERY, ROLE_QUERY_DESC,
 ROLE_EVIDENCE, ROLE_EVIDENCE_DESC, ROLE_EVIDENCE_LABEL) = range(6)
N_ROLES = 6
UNBOUND_SLOT = 0
NO_RECORDING_GROUP = 0
QUERY_RECORDING_GROUP = 1


@dataclass(frozen=True)
class EvidenceMixerConfig:
    text_dim: int = 384
    n_layers: int = 2
    n_slots: int = 64           # > candidates in one episode; ids randomised per episode
    #: > distinct recordings among ONE query's retrieved rows, which is bounded by top_k. Sized
    #: against the default top_k=64 with headroom; EngineConfig checks the pair at construction.
    n_groups: int = 96
    form_rank: int = 32         # rank of each low-rank interaction form ("weights" readout)
    #: THE ONE DESIGN CHOICE. Both readouts consume the same mixed evidence; they differ in what
    #: attention is allowed to emit. See the module docstring.
    #:
    #: "weights"  — per-(row, candidate) scalar corrections to the evidence weight. The readout
    #:              cosine stays at the text encoder's own geometry.
    #: "semantic" — attention displaces each retrieved row's label vector inside text space, and
    #:              emits no scalar at all. Weights are then the retrieval score alone.
    readout: str = "weights"                    # weights | semantic
    #: How far from pure retrieval the "weights" readout starts, in log-weight units. The forms
    #: emit O(1) at standard init, so this IS roughly the initial mean |adjustment| in nats. A
    #: mixer earning its keep drives it up; one that never leaves init is inert — that is a result.
    correction_gain_init: float = 0.02
    #: Displacement as a fraction of a unit-norm text vector, for the "semantic" readout.
    semantic_gain_init: float = 0.05
    #: Scale of the standardised retrieval score as an additive attention bias.
    score_bias_init: float = 1.0
    #: Initial gain on the role/slot/group identity channels, relative to content at 1.0. See
    #: :class:`~model.blocks.ScaledSum` — at 1.0 the content is only a quarter of every token.
    identity_gain_init: float = 1.0


class _LowRankForm(nn.Module):
    """``<A x, B y> / sqrt(rank)`` — one bilinear interaction at a chosen rank."""

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.left = nn.Linear(d_model, rank)
        self.right = nn.Linear(d_model, rank)
        self.scale = rank ** -0.5

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """(Q,k,d) x (Q,C,d) -> (Q,k,C)."""
        return torch.einsum("qkr,qcr->qkc", self.left(left), self.right(right)) * self.scale


class EvidenceMixer(nn.Module):
    """Context-mixes one query's retrieved evidence, then reads out evidence weights."""

    def __init__(self, spec: AttentionSpec, cfg: EvidenceMixerConfig | None = None):
        super().__init__()
        self.spec = spec
        self.cfg = cfg or EvidenceMixerConfig()
        if self.cfg.readout not in ("weights", "semantic"):
            raise ValueError("readout must be 'weights' or 'semantic'")
        d = spec.d_model

        self.proj_text = nn.Linear(self.cfg.text_dim, d)
        self.proj_signal = nn.Linear(d, d)
        self.role_emb = nn.Embedding(N_ROLES, d)
        self.slot_emb = nn.Embedding(self.cfg.n_slots, d)
        self.group_emb = nn.Embedding(self.cfg.n_groups, d)
        # Content + role + slot + group, each normalised to a direction with its own learned gain.
        self.compose = ScaledSum(4, init=[1.0] + [self.cfg.identity_gain_init] * 3)
        self.stack = SetAttentionStack(spec, self.cfg.n_layers)

        self.score_bias_gain = nn.Parameter(torch.tensor(float(self.cfg.score_bias_init)))
        # Exactly one readout is built. Constructing both and using one would put parameters with
        # no path to the loss into every checkpoint, which is the defect this design keeps finding.
        if self.cfg.readout == "weights":
            # "Does this row speak to this candidate" — row-level, query-independent.
            self.evidence_candidate_form = _LowRankForm(d, self.cfg.form_rank)
            # "For THIS query, is this row informative about THIS candidate" — the trilinear
            # interaction, built by gating the evidence token with the query token.
            self.query_evidence_candidate_form = _LowRankForm(d, self.cfg.form_rank)
            self.correction_gain = nn.Parameter(torch.tensor(float(self.cfg.correction_gain_init)))
        else:
            self.semantic_head = nn.Linear(d, self.cfg.text_dim)
            self.semantic_gain = nn.Parameter(torch.tensor(float(self.cfg.semantic_gain_init)))

    # ---------------------------------------------------------------- token assembly
    def _token(self, content, role: int, slot, group):
        role_vec = self.role_emb(torch.full(content.shape[:-1], role,
                                            dtype=torch.long, device=content.device))
        slot_vec = self.slot_emb(slot)
        group_vec = self.group_emb(group)
        return self.compose(content, role_vec, slot_vec, group_vec)

    def _encode(
        self,
        *,
        candidate_text: torch.Tensor,      # (Q, C, text_dim)
        query_feature: torch.Tensor,       # (Q, d)
        query_descriptor: torch.Tensor,    # (Q, text_dim)
        evidence_feature: torch.Tensor,    # (Q, k, d)
        evidence_descriptor: torch.Tensor, # (Q, k, text_dim)
        evidence_label_text: torch.Tensor, # (Q, k, text_dim)
        candidate_slot: torch.Tensor,      # (Q, C)
        evidence_slot: torch.Tensor,       # (Q, k)
        evidence_group: torch.Tensor,      # (Q, k)
        retrieval_score: torch.Tensor,     # (Q, k)
    ):
        n_query, n_cand = candidate_text.shape[0], candidate_text.shape[1]
        n_ev = evidence_feature.shape[1]
        device = query_feature.device
        no_group = torch.full(
            (n_query,), NO_RECORDING_GROUP, dtype=torch.long, device=device,
        )
        query_group = torch.full(
            (n_query,), QUERY_RECORDING_GROUP, dtype=torch.long, device=device,
        )
        unbound = torch.full((n_query,), UNBOUND_SLOT, dtype=torch.long, device=device)

        cand = self._token(
            self.proj_text(candidate_text), ROLE_CANDIDATE, candidate_slot,
            no_group.unsqueeze(1).expand(n_query, n_cand),
        )
        qry = self._token(self.proj_signal(query_feature), ROLE_QUERY, unbound, query_group)
        qdesc = self._token(
            self.proj_text(query_descriptor), ROLE_QUERY_DESC, unbound, query_group,
        )
        ev = self._token(self.proj_signal(evidence_feature), ROLE_EVIDENCE,
                         evidence_slot, evidence_group)
        edesc = self._token(self.proj_text(evidence_descriptor), ROLE_EVIDENCE_DESC,
                            evidence_slot, evidence_group)
        elbl = self._token(self.proj_text(evidence_label_text), ROLE_EVIDENCE_LABEL,
                           evidence_slot, evidence_group)

        tokens = torch.cat(
            [cand, qry.unsqueeze(1), qdesc.unsqueeze(1), ev, edesc, elbl], dim=1,
        )
        # The retrieval score reaches attention as an additive COLUMN bias on the evidence tokens,
        # standardised per query so a score range of tens of nats cannot saturate the softmax. It
        # is a bias rather than an input feature so no later layer can launder it into a path that
        # reaches a candidate without passing through the row it describes.
        centred = retrieval_score - retrieval_score.mean(dim=1, keepdim=True)
        centred = centred / centred.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        bias = torch.zeros(n_query, tokens.shape[1], device=device, dtype=tokens.dtype)
        bias[:, n_cand + 2:] = self.score_bias_gain * centred.repeat(1, 3)

        hidden = self.stack(tokens, attn_bias=bias.unsqueeze(1))
        offset = n_cand + 2
        return {
            "candidate": hidden[:, :n_cand],
            "query": hidden[:, n_cand],
            "evidence": hidden[:, offset:offset + n_ev],
            "label": hidden[:, offset + 2 * n_ev:offset + 3 * n_ev],
        }

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        *,
        retrieval_score: torch.Tensor,        # (Q, k)
        candidate_text: torch.Tensor,         # (C, text_dim) L2-normalised
        query_feature: torch.Tensor,          # (Q, d)
        query_descriptor: torch.Tensor,       # (Q, text_dim)
        evidence_feature: torch.Tensor,       # (Q, k, d)
        evidence_descriptor: torch.Tensor,    # (Q, k, text_dim)
        evidence_label_text: torch.Tensor,    # (Q, k, text_dim) L2-normalised
        candidate_slot: torch.Tensor,         # (C,) episode-randomised
        evidence_slot: torch.Tensor,          # (Q, k)
        evidence_group: torch.Tensor,         # (Q, k)
    ) -> dict[str, torch.Tensor]:
        """Returns per-candidate log evidence weights and the label vectors the vote will use.

        ``log_weight`` (Q, k, C) is ``retrieval score + correction_gain * relational correction``.
        ``label_vector`` (Q, k, text_dim) is the retrieved label text, refined if configured.
        """
        n_query = query_feature.shape[0]
        cand_batch = candidate_text.unsqueeze(0).expand(n_query, -1, -1)
        slots = candidate_slot.unsqueeze(0).expand(n_query, -1)
        hidden = self._encode(
            candidate_text=cand_batch, query_feature=query_feature,
            query_descriptor=query_descriptor, evidence_feature=evidence_feature,
            evidence_descriptor=evidence_descriptor, evidence_label_text=evidence_label_text,
            candidate_slot=slots, evidence_slot=evidence_slot, evidence_group=evidence_group,
            retrieval_score=retrieval_score,
        )
        if self.cfg.readout == "weights":
            correction = (
                self.evidence_candidate_form(hidden["evidence"], hidden["candidate"])
                + self.query_evidence_candidate_form(
                    hidden["evidence"] * hidden["query"].unsqueeze(1), hidden["candidate"],
                )
            )
            # Retrieval is O(10) log units while the residual starts at O(0.01). A BF16 addition
            # can round the entire learned correction away even though backward still reports a
            # gradient. The attention stack remains mixed precision; only its scalar readout sum
            # is promoted.
            with torch.autocast(device_type=retrieval_score.device.type, enabled=False):
                log_weight = (retrieval_score.float().unsqueeze(-1)
                              + self.correction_gain.float() * correction.float())
            return {
                "log_weight": log_weight,
                "label_vector": evidence_label_text,
            }
        # "semantic": the weight is the retrieval score alone, shared across candidates, and every
        # per-candidate distinction is carried by where attention moves the row's label vector.
        direction = F.normalize(self.semantic_head(hidden["label"]), dim=-1)
        with torch.autocast(device_type=retrieval_score.device.type, enabled=False):
            refined = F.normalize(
                evidence_label_text.float() + self.semantic_gain.float() * direction.float(),
                dim=-1,
            )
        return {
            "log_weight": retrieval_score.float().unsqueeze(-1).expand(
                -1, -1, candidate_text.shape[0],
            ),
            "label_vector": refined,
        }

    def telemetry(self) -> dict[str, float]:
        """The scalars that say whether this module is doing anything."""
        stats = {"mixer/score_bias_gain": float(self.score_bias_gain.detach())}
        if self.cfg.readout == "weights":
            stats["mixer/correction_gain"] = float(self.correction_gain.detach())
        else:
            stats["mixer/semantic_gain"] = float(self.semantic_gain.detach())
        return stats
