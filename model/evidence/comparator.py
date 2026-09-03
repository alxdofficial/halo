"""Support-only comparator: recognise a query by comparing it to K labelled recordings.

WHAT THIS REPLACES
------------------
The Phase-B engine retrieved a top-k slice from a large frozen corpus bank and mixed it. Three
measurements retired that design:

* retrieval ranked by *acquisition configuration*, not activity — same-activity rows from another
  device sat at the 39th percentile, near chance;
* a learned retrieval stage was worth exactly nothing over plain cosine (+0.0000 paired);
* the readout, not the retrieval, carried ~80% of all learning.

So the retrieval stage is gone. The support set is handed to the model — chosen by an explicit
compatibility filter that is a deployment consideration, not a learned quantity — and the
comparator attends over all of it. K is small enough that there is nothing to select.

THE READOUT
-----------
For candidate ``c`` the score is a weighted vote over support recordings::

    score(c) = sum_e  w(query, e, c) * vote(e, c)

``w`` is the comparator's attention weight for that (query, support, candidate) triple. ``vote`` is
1 when support recording ``e`` is *enrolled* against candidate ``c`` (its label is that candidate),
and otherwise the rectified cosine between the support recording's own label text and the
candidate's text. That second branch is what makes an unseen candidate scorable at all, and it is
why labels can stay verbatim: two candidates with near-identical text simply receive near-identical
votes, which is correct rather than a problem to deduplicate away.

There are no per-candidate parameters anywhere. An unseen candidate is scored by the same operation
as a seen one, and permuting candidates permutes the logits.

IDENTITY AT INITIALISATION
--------------------------
``residual_head`` is zero-initialised, so at step 0 the comparator's logits are *exactly* the
closed-form vote over the same support set. That closed form is the untrained floor every result is
quoted against, and the step-0 control depends on the equality being exact rather than approximate.
``tests/test_comparator.py`` asserts it to 1e-6.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import AttentionSpec, ScaledSum, SetAttentionStack

# Token roles. Every token carries exactly one, added as a direction rather than concatenated.
(ROLE_CANDIDATE, ROLE_QUERY, ROLE_QUERY_DESC,
 ROLE_SUPPORT, ROLE_SUPPORT_DESC, ROLE_SUPPORT_LABEL) = range(6)
N_ROLES = 6

#: Slot 0 means "not bound to any candidate". Query rows always use it.
UNBOUND_SLOT = 0


@dataclass(frozen=True)
class ComparatorConfig:
    """Shape and initialisation of the comparator.

    ``n_slots`` bounds how many candidates one episode may carry. The slot embedding is an
    episode-randomised coreference tag, not a label identity: it lets attention notice that a
    support row and a candidate refer to the same thing without ever learning what that thing is.
    """

    text_dim: int = 384
    n_layers: int = 2
    n_slots: int = 64
    #: Content leads at initialisation. Role and coreference stay visible without jointly
    #: outweighing the signal they are supposed to annotate.
    identity_gain_init: float = 0.25


class SupportComparator(nn.Module):
    """Attention over [candidates | query rows | support rows], then a corrected vote."""

    def __init__(self, spec: AttentionSpec, cfg: ComparatorConfig | None = None):
        super().__init__()
        self.spec = spec
        self.cfg = cfg or ComparatorConfig()
        d = spec.d_model
        self.proj_text = nn.Linear(self.cfg.text_dim, d)
        self.proj_signal = nn.Linear(d, d)
        self.role_emb = nn.Embedding(N_ROLES, d)
        self.slot_emb = nn.Embedding(self.cfg.n_slots, d)
        self.compose = ScaledSum(3, init=[1.0, self.cfg.identity_gain_init,
                                          self.cfg.identity_gain_init])
        self.stack = SetAttentionStack(spec, self.cfg.n_layers)
        # One shared scalar head over candidate tokens: permutation-equivariant, and it works on a
        # candidate the model has never seen. Zero init makes the whole module exactly the
        # closed-form vote at step 0. No bias: a shared constant added to every candidate logit
        # cancels in the softmax and would be permanently unidentifiable.
        self.residual_head = nn.Linear(d, 1, bias=False)
        nn.init.zeros_(self.residual_head.weight)

    # ------------------------------------------------------------------ tokens
    def _token(self, content: torch.Tensor, role: int, slot: torch.Tensor) -> torch.Tensor:
        role_vec = self.role_emb(torch.full(
            content.shape[:-1], role, dtype=torch.long, device=content.device,
        ))
        return self.compose(content, role_vec, self.slot_emb(slot))

    def forward(
        self,
        *,
        candidate_text: torch.Tensor,        # (B, C, Z)  frozen text of each candidate label
        query_feature: torch.Tensor,         # (B, Q, d)  encoder rows for the query recording
        query_descriptor: torch.Tensor,      # (B, Q, Z)  the query's sensor text
        query_mask: torch.Tensor,            # (B, Q)     True = a real row
        support_feature: torch.Tensor,       # (B, K, d)  one pooled row per support recording
        support_descriptor: torch.Tensor,    # (B, K, Z)  each support recording's sensor text
        support_label_text: torch.Tensor,    # (B, K, Z)  each support recording's verbatim label
        support_mask: torch.Tensor,          # (B, K)     True = a real support row
        candidate_slot: torch.Tensor,        # (B, C)     episode-randomised coreference tags
        support_slot: torch.Tensor,          # (B, K)     tag of the candidate a row is bound to
    ) -> torch.Tensor:
        """Return ``(B, C)`` residual logits, zero at initialisation."""

        B, C, _ = candidate_text.shape
        Q = query_feature.shape[1]
        device = query_feature.device

        query_slot = torch.full((B, Q), UNBOUND_SLOT, dtype=torch.long, device=device)

        candidate = self._token(self.proj_text(candidate_text), ROLE_CANDIDATE, candidate_slot)
        query = self._token(self.proj_signal(query_feature), ROLE_QUERY, query_slot)
        query_desc = self._token(self.proj_text(query_descriptor), ROLE_QUERY_DESC, query_slot)
        support = self._token(self.proj_signal(support_feature), ROLE_SUPPORT, support_slot)
        support_desc = self._token(
            self.proj_text(support_descriptor), ROLE_SUPPORT_DESC, support_slot,
        )
        support_label = self._token(
            self.proj_text(support_label_text), ROLE_SUPPORT_LABEL, support_slot,
        )

        tokens = torch.cat(
            [candidate, query, query_desc, support, support_desc, support_label], dim=1,
        )
        valid = torch.cat([
            torch.ones((B, C), dtype=torch.bool, device=device),
            query_mask, query_mask,
            support_mask, support_mask, support_mask,
        ], dim=1)

        hidden = self.stack(tokens, key_padding_mask=valid)
        with torch.autocast(device_type=device.type, enabled=False):
            return self.residual_head(hidden[:, :C].float()).squeeze(-1)

    def telemetry(self) -> dict[str, float]:
        gains = self.compose.log_gain.detach().exp()
        return {
            "comparator/residual_head_norm": float(self.residual_head.weight.detach().norm()),
            "comparator/content_gain": float(gains[0]),
            "comparator/identity_gain_mean": float(gains[1:].mean()),
        }


def support_vote(
    *,
    candidate_text: torch.Tensor,        # (B, C, Z)  L2-normalised
    support_label_text: torch.Tensor,    # (B, K, Z)  L2-normalised
    support_bound: torch.Tensor,         # (B, K)     candidate index, -1 = not a candidate's label
    support_mask: torch.Tensor,          # (B, K)
    weights: torch.Tensor,               # (B, K, C)  non-negative
) -> torch.Tensor:
    """The closed-form readout: ``sum_e weight(e,c) * vote(e,c)``.

    An enrolled support row votes 1 for the candidate it is bound to and nothing for the others.
    Any other row votes the rectified cosine between its own label text and each candidate's text.
    Neither branch has a per-candidate parameter, so the rule is identical for a candidate the
    model has never seen.
    """

    B, C, _ = candidate_text.shape
    K = support_label_text.shape[1]
    device = candidate_text.device

    index = torch.arange(C, device=device).view(1, 1, C)
    bound = support_bound.to(device).view(B, K, 1)

    enrolled_vote = (bound.ge(0) & bound.eq(index)).to(weights.dtype)
    semantic = F.relu(torch.bmm(
        F.normalize(support_label_text.float(), dim=-1),
        F.normalize(candidate_text.float(), dim=-1).transpose(1, 2),
    )).to(weights.dtype)
    # A row bound to a candidate speaks only through the identity vote; letting it also vote
    # semantically would count the same evidence twice, with the duplicate landing on whichever
    # other candidates happen to share vocabulary with its label.
    text_vote = semantic * bound.lt(0).to(weights.dtype)

    vote = enrolled_vote + text_vote
    masked = weights * support_mask.unsqueeze(-1).to(weights.dtype)
    return (masked * vote).sum(dim=1)


def comparator_logits(
    comparator: SupportComparator | None,
    *,
    candidate_text: torch.Tensor,
    query_feature: torch.Tensor,
    query_descriptor: torch.Tensor,
    query_mask: torch.Tensor,
    support_feature: torch.Tensor,
    support_descriptor: torch.Tensor,
    support_label_text: torch.Tensor,
    support_bound: torch.Tensor,
    support_mask: torch.Tensor,
    candidate_slot: torch.Tensor | None = None,
    temperature: float = 0.07,
    vote_scale: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Score every candidate for every episode in the batch.

    ``comparator=None`` is the untrained floor: cosine similarity between the query and each
    support recording supplies the weights, and nothing is learned anywhere. With a comparator
    whose ``residual_head`` is still zero the two paths agree exactly, which is what makes the
    step-0 control a control rather than an approximation.
    """

    B, C, _ = candidate_text.shape
    K = support_feature.shape[1]

    # Pool the query's rows into one direction, then score it against every support recording.
    valid_query = query_mask.unsqueeze(-1).to(query_feature.dtype)
    pooled = (query_feature * valid_query).sum(dim=1) / valid_query.sum(dim=1).clamp_min(1e-6)
    similarity = torch.bmm(
        F.normalize(pooled.float(), dim=-1).unsqueeze(1),
        F.normalize(support_feature.float(), dim=-1).transpose(1, 2),
    ).squeeze(1)                                                   # (B, K)
    similarity = similarity.masked_fill(~support_mask, float("-inf"))

    # Softmax over support rows, shared across candidates. With every row masked out (K = 0 or an
    # all-empty support set) the vote is zero and the logits fall back to the text path alone.
    empty = ~support_mask.any(dim=1, keepdim=True)
    safe = torch.where(support_mask, similarity, torch.zeros_like(similarity))
    weights = torch.softmax(safe / temperature, dim=1)
    weights = torch.where(support_mask, weights, torch.zeros_like(weights))
    weights = torch.where(empty, torch.zeros_like(weights), weights)
    weights = weights.unsqueeze(-1).expand(B, K, C)

    base = vote_scale * support_vote(
        candidate_text=candidate_text,
        support_label_text=support_label_text,
        support_bound=support_bound,
        support_mask=support_mask,
        weights=weights,
    )

    residual = base.new_zeros(base.shape)
    if comparator is not None:
        if candidate_slot is None:
            raise ValueError("a comparator needs episode-randomised candidate slots")
        support_slot = torch.where(
            support_bound.ge(0),
            torch.gather(candidate_slot, 1, support_bound.clamp_min(0)),
            torch.full_like(support_bound, UNBOUND_SLOT),
        )
        residual = comparator(
            candidate_text=candidate_text,
            query_feature=query_feature,
            query_descriptor=query_descriptor,
            query_mask=query_mask,
            support_feature=support_feature,
            support_descriptor=support_descriptor,
            support_label_text=support_label_text,
            support_mask=support_mask,
            candidate_slot=candidate_slot,
            support_slot=support_slot,
        )

    return {"logits": base + residual, "base_logits": base, "residual": residual,
            "support_weight": weights[..., 0]}
