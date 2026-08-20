"""Learned retrieval: one scalar per (query row, memory row) pair.

    score = f( query feature, query description, memory feature, memory description )

WHAT THIS REPLACES
------------------
Retrieval used to be two hand-written pieces bolted together:

1. a HARD compatibility filter that set the score to ``-inf`` unless modality matched and, for
   accelerometers, the gravity convention matched;
2. a cosine between L2-normalised feature vectors, divided by a fixed temperature.

Both encode real facts. A gyroscope's angular rate is not commensurable with an accelerometer's
proper acceleration, and a gravity-removed stream has |DC| about 0 against about 1 g, so comparing
across that convention compares an artifact of preprocessing. The problem was never that the rules
were wrong — it was that they were *stipulated*, so they could not improve, could not trade off
against each other, and could not be checked against what the model actually needs.

They are learnable now because the information they used is already in the inputs. The sensor
description string carries modality and gravity convention verbatim — "a watch accelerometer on the
left wrist; gravity removed; recorded alongside a gyroscope" — so a function of the two description
vectors can express both rules, and can express them for acquisition configurations that were never
in the corpus, through language rather than through a metadata table. That is the whole HALO claim
applied to its own retrieval step.

Because the constraint is now learned rather than enforced, whether it *was* learned becomes a
measurement. :func:`physics_violation_rate` reports how often the selected top-k crosses a modality
or gravity boundary; a model that has understood its descriptions drives it toward zero on its own.

HOW IT STARTS
-------------
``base_gain`` multiplies the raw feature cosine and is initialised to ``1/0.07``, the deployed
temperature; ``residual_gain`` multiplies the learned head and is initialised to 0.02. So at step 0
the score is the old cosine rule to within 2%, and any later difference is attributable to training
rather than to a different starting point. Neither scalar is frozen: ``base_gain`` may go to zero
and hand the whole job to the learned head. Both are logged.

The hard ``-inf`` filter is gone entirely. Nothing here can make a row unreachable, which matters
because an unreachable row is one no gradient can ever recover.

SHAPE OF THE LEARNED PART
-------------------------
``H`` factored bilinear heads over the concatenated ``[feature ; projected description]`` of each
side, then a small MLP over the resulting ``H`` scalars plus the two raw cosines. The heads are the
cheap part — a projection per side and one matmul, not an MLP over ``Q x M`` concatenated pairs,
which at 125 queries against a 512-row bank would be 64k forward passes through a 1280-wide MLP.
The nonlinearity that a bilinear form lacks is supplied once, at the end, over ``H+2`` numbers.

Query and memory get SEPARATE projections (``query_proj`` / ``memory_proj``). Retrieval is not
symmetric: "is this row useful evidence for that query" is a different question from its converse.
The patch/text fusion, by contrast, is JOINT — see ``_joint_heads``: one linear map over the
concatenated pair, never a signal score and a text score computed apart and summed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import AttentionSpec

#: The vote temperature the hand-written rule used. Only an initialisation now.
DEPLOYED_TEMPERATURE = 0.07


@dataclass(frozen=True)
class PairScorerConfig:
    text_dim: int = 384
    n_interaction_heads: int = 8
    interaction_dim: int = 16
    pair_head_hidden: int = 32
    base_gain_init: float = 1.0 / DEPLOYED_TEMPERATURE
    residual_gain_init: float = 0.02
    #: False replaces the whole learned head with the plain feature cosine at a FIXED temperature
    #: and registers no parameters at all. This is the stand-in the ablation ladder substitutes
    #: when isolating which stage of the model is failing to learn.
    learned: bool = True


class PairScorer(nn.Module):
    """(Q, M) retrieval scores, and the top-k selection built on them."""

    def __init__(self, spec: AttentionSpec, cfg: PairScorerConfig | None = None):
        super().__init__()
        self.spec = spec
        self.cfg = cfg or PairScorerConfig()
        d = spec.d_model
        width = self.cfg.n_interaction_heads * self.cfg.interaction_dim
        if not self.cfg.learned:
            # A buffer, not a Parameter: the fixed stand-in must contribute no gradient anywhere,
            # so that a ladder rung which "disables retrieval learning" really does.
            self.register_buffer("fixed_gain", torch.tensor(float(self.cfg.base_gain_init)))
            return

        # One shared description projection: a description means the same thing on either side of
        # the comparison, so learning it twice would only let the two copies drift apart.
        self.desc_proj = nn.Linear(self.cfg.text_dim, d)
        self.query_proj = nn.Linear(2 * d, width)
        self.memory_proj = nn.Linear(2 * d, width)
        # Written as two Linears rather than a Sequential so the pair head can be run at a chosen
        # precision without mutating the module — see `_pair_head`. The head consumes the joint
        # interaction heads PLUS the two raw cosines as extra features: still one mechanism, one
        # set of weights — the cosines are inputs to it, not parallel scorers beside it.
        self.pair_head_in = nn.Linear(self.cfg.n_interaction_heads + 2, self.cfg.pair_head_hidden)
        # No bias on the output. A constant added to every pair's score is invisible: it cancels
        # in the softmax the vote takes over rows, and again in the standardisation the mixer
        # applies before using the score as an attention bias. It would be a parameter that can
        # never change an output, which is the definition of dead capacity.
        self.pair_head_out = nn.Linear(self.cfg.pair_head_hidden, 1, bias=False)
        self.base_gain = nn.Parameter(torch.tensor(float(self.cfg.base_gain_init)))
        self.residual_gain = nn.Parameter(torch.tensor(float(self.cfg.residual_gain_init)))

    def _joint_heads(self, feature: torch.Tensor, descriptor: torch.Tensor, proj: nn.Linear):
        """One row -> H interaction heads, from patch vector and description text TOGETHER.

        This is the fusion point, and it is deliberately ONE mechanism rather than a signal
        scorer and a text scorer summed: the concatenated ``[patch ; projected description]``
        goes through a single linear map, so every head is a learned function of both at once —
        a head can express "same placement AND similar dynamics", which neither input could
        alone, and which a sum of separate scores structurally cannot.
        """
        if descriptor.shape[-1] != self.cfg.text_dim:
            raise ValueError(
                f"descriptor must be {self.cfg.text_dim}-d, got {descriptor.shape[-1]}"
            )
        joint = torch.cat([feature, self.desc_proj(descriptor)], dim=-1)
        heads = proj(joint).view(
            len(feature), self.cfg.n_interaction_heads, self.cfg.interaction_dim,
        )
        return heads

    def _pair_head(self, stacked: torch.Tensor) -> torch.Tensor:
        """(Q, M, H+2) -> (Q, M), at reduced precision where that is free.

        WHY THIS IS THE PART WORTH OPTIMISING
        -------------------------------------
        Everything else here is a matmul over a few hundred thousand elements. This is an MLP
        evaluated at EVERY pair: at 104 queries against a 4,900-row bank it writes, reads and
        writes back a (104, 4900, 32) hidden activation — about 65 MB of traffic for 45 MFLOP of
        arithmetic. It is bandwidth-bound, and it was 71% of the scorer's forward pass.

        Running just this block in bfloat16 makes it 8.6x faster (0.68 ms -> 0.08 ms measured on a
        4090). The precision argument is bounded rather than hopeful: the block's output is
        multiplied by ``residual_gain`` and added to a score whose standard deviation is order 1.
        Measured on random features, the worst-case score error is 3e-5 at the initial gain of
        0.02 and 5e-3 even at a gain of 5, against which top-64 selection agrees with the float32
        path on 99.95% of rows. The base cosine and the final sum stay in float32, so the reduced
        precision only ever touches the learned correction.

        CPU stays in float32. The gain there is not established, and keeping one path exact makes
        the test suite a reference rather than a second approximation.
        """
        compute = (torch.bfloat16
                   if stacked.is_cuda and stacked.dtype == torch.float32
                   else stacked.dtype)
        hidden = F.linear(stacked.to(compute), self.pair_head_in.weight.to(compute),
                          self.pair_head_in.bias.to(compute))
        out = F.linear(F.gelu(hidden), self.pair_head_out.weight.to(compute))
        return out.squeeze(-1).to(stacked.dtype)

    def forward(
        self,
        query_feature: torch.Tensor,       # (Q, d)
        query_descriptor: torch.Tensor,    # (Q, text_dim)
        memory_feature: torch.Tensor,      # (M, d)
        memory_descriptor: torch.Tensor,   # (M, text_dim)
    ) -> torch.Tensor:
        """(Q, M) scores in log-weight units — the same units the vote consumes.

        Fully batched: two side projections, one batched matmul over the interaction heads, two
        cosine matmuls, one pointwise MLP. There is no Python loop over queries, rows or heads, so
        the whole (Q, M) grid is one wavefront of GPU work.
        """
        if not self.cfg.learned:
            return self.fixed_gain * (F.normalize(query_feature.float(), dim=-1)
                                      @ F.normalize(memory_feature.float(), dim=-1).t())
        query_heads = self._joint_heads(query_feature, query_descriptor, self.query_proj)
        memory_heads = self._joint_heads(memory_feature, memory_descriptor, self.memory_proj)
        # (H, Q, dh) x (H, dh, M) -> (H, Q, M): one batched matmul over the interaction heads,
        # written as bmm rather than einsum so the dispatch is not left to a parser.
        interaction = torch.bmm(
            query_heads.transpose(0, 1), memory_heads.permute(1, 2, 0),
        ).permute(1, 2, 0) / (self.cfg.interaction_dim ** 0.5)

        # Retrieval scores are log weights. Keep the base cosine and residual sum in FP32: at a
        # score near 10, BF16's unit in the last place is about 0.06, larger than the intended
        # 0.02-scale learned residual. Computing that sum in BF16 gives the residual a gradient but
        # removes its forward effect, which is a silent credit-assignment failure.
        with torch.autocast(device_type=query_feature.device.type, enabled=False):
            feature_cos = (F.normalize(query_feature.float(), dim=-1)
                           @ F.normalize(memory_feature.float(), dim=-1).t())
            desc_cos = (F.normalize(query_descriptor.float(), dim=-1)
                        @ F.normalize(memory_descriptor.float(), dim=-1).t())
        stacked = torch.cat(
            [feature_cos.unsqueeze(-1), desc_cos.unsqueeze(-1), interaction.float()], dim=-1,
        )
        correction = self._pair_head(stacked)
        with torch.autocast(device_type=query_feature.device.type, enabled=False):
            return (self.base_gain.float() * feature_cos
                    + self.residual_gain.float() * correction.float())

    @staticmethod
    def select(scores: torch.Tensor, top_k: int) -> torch.Tensor:
        """(Q, k) indices of the highest-scoring memory rows for each query.

        Selection is hard, and deliberately identical in training and deployment. The previous
        design voted over every row while training and truncated to top-k while deploying, so the
        thing being optimised was not the thing being run. The cost is that unselected rows receive
        no gradient on this step; the bank is resampled every episode, which is what returns them
        to circulation.
        """
        k = min(int(top_k), scores.shape[1])
        if k <= 0:
            raise ValueError("top_k must be positive")
        return scores.topk(k, dim=1).indices


def physics_violation_rate(
    selected: torch.Tensor,            # (Q, k) memory indices
    query_modality: torch.Tensor,      # (Q,)
    query_gravity: torch.Tensor,       # (Q,)
    memory_modality: torch.Tensor,     # (M,)
    memory_gravity: torch.Tensor,      # (M,)
) -> dict[str, float]:
    """How often learned retrieval selects a row the retired hard filter would have excluded.

    This is a diagnostic, never a loss and never a mask. Falling toward zero is evidence that the
    descriptions carry the constraint; staying high with good accuracy would be evidence that the
    constraint mattered less than it was assumed to. Either outcome is a result.
    """
    modality = memory_modality[selected]
    gravity = memory_gravity[selected]
    modality_ok = modality == query_modality.unsqueeze(1)
    is_accel = modality == 0
    gravity_ok = (gravity == query_gravity.unsqueeze(1)) | ~is_accel
    return {
        "retrieval/modality_violation": float((~modality_ok).float().mean()),
        "retrieval/gravity_violation": float((modality_ok & ~gravity_ok).float().mean()),
    }
