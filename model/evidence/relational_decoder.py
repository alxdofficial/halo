"""Relational evidence decoder — one attention stack over names, query and evidence.

Implements `docs/design/PHASE_B_TRAINING_INTENT.md` §6. It replaced a vote/pool decoder whose
measured failure was structural rather than a matter of tuning: its pooling
weight carried no candidate axis, so the *only* quantity separating candidate c from candidate c'
was `cos(evidence label text, candidate text)` — a kernel whose off-diagonal mean on the corpus
vocabulary is 0.273 and whose within-candidate-set maximum is 0.849, leaving a 0.151 margin to
decide on. Here the candidate score is read off a token that has attended over the evidence, so
discrimination no longer has to survive a fixed text kernel.

Token layout (B rows, C candidates, L background labels, P query patches, M evidence rows):

    CAND_c    proj_text(name_c)  + role_CAND   + slot(c)
    LBL_l     proj_text(name_l)  + role_LBL    + slot(C+l)
    QRY_p     f(z_q)             + role_QUERY  + time(p) + group(window)
    EV_m      g(z_m)             + role_EV/SUPPORT + slot(ref(m)) + time(m) + group(window)
                                 + same-config / same-subject / same-sensor as the query

    logit_c = MLP(h_CAND_c)

Every additive ingredient is normalized to a direction and multiplied by one learned positive
component scale before these sums. The seven scales start at one. This is load-bearing: normalizing
only the completed token would still let default ``nn.Embedding`` vectors (norm about sqrt(d))
drown projections of unit sensor/text vectors (norm below one).

The retrieval score is **not** a token feature. It enters as a log-relative prior on the attention
logits of the evidence keys:

    bias_m = log_softmax(s_m / tau) + log(number of valid evidence rows)

The uniform prior is therefore zero, adding a common offset to every selected cosine changes
nothing, and only within-roster retrieval preferences affect attention. This keeps the retriever's
units compatible with the ordinary attention logits instead of giving every evidence key the large
positive common mode of an absolute cosine. It also keeps the useful structural gradient semantics:
`dL/ds_m < 0` means that increasing this row relative to the other selected rows would reduce loss.
An earlier revision projected the score through `nn.Linear(1 + 2*n_score_freqs, d)` and added it to
the evidence token; there the relationship between score and usefulness was whatever the network
happened to learn, it could decay to zero and silently stop training the retriever, and it cost a
module, a Fourier-feature helper and a constant. It is also the only differentiable route from the
candidate loss back to the retriever: selection is a hard top-k over frozen memory vectors.

**Names appear once.** An evidence row carries signal plus a *pointer*; its label name lives only
on the label token it points at. Acquisition values are already inside `z` because the Phase-A
encoder ingests them, so only relations to the query are re-encoded here.

**Coreference is exact, never similarity.** Binding an enrolled row to its candidate is a fact, so
it is a shared discrete embedding. Expressing it as a similarity between the row's label text and
the candidate's text would move the 0.467 common mode out of a cosine and into an attention logit
and reproduce the original failure. Slot ids are randomised per episode by the caller, so a slot
can only ever mean "these tokens refer to the same thing". Source-window co-membership is encoded
the same way, as a shared group embedding, which is what lets this replace the old pairwise
same-window/same-sensor attention biases.

**The logit is the readout, and nothing else.** An earlier revision added a closed-form `base_c`
(prototype similarity, falling back to a label-text vote) that the readout corrected as a
zero-initialised residual, on the argument that it gave a non-random floor and made the learned
contribution directly measurable. Measurement said otherwise: on the real bank the prototype branch
was reachable for only 15.5% of decisions because it was gated on the retriever happening to select
a candidate's enrolled rows, so ~85% of base scores were the very label-text vote this decoder
exists to replace — and the two branches sat 6.4 logits apart (602:1 odds), which meant the base was
deciding by *which branch a candidate landed on* rather than by any comparison. The instrument had
become the model. There is now one path to a logit: attend, then read out.

Enrollment is still fully available to the model — declared support rows carry `ROLE_SUPPORT` and a
coreference slot binding them to their candidate, so the stack can compare the query against them
directly. It simply has to learn that comparison instead of being handed a closed-form version of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

ROLE_CANDIDATE = 0
ROLE_LABEL = 1
ROLE_QUERY = 2
ROLE_EVIDENCE = 3
ROLE_SUPPORT = 4
N_ROLES = 5

# Slot 0 is the unbound slot: padding, and any evidence row whose referent is not in the name set.
UNBOUND_SLOT = 0
# Group 0 is the unbound group, for padded tokens with no source window.
UNBOUND_GROUP = 0


@dataclass
class RelationalDecoderConfig:
    d_model: int = 256          # matches the encoder, so patch vectors need no up-projection
    text_dim: int = 384         # frozen SBERT
    n_layers: int = 3
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    n_time_freqs: int = 16
    time_max_sec: float = 30.0
    # LayerScale was 1e-4 so the stack was identity at init and the logits equalled the closed-form
    # base exactly. With no base term that rationale is gone, and 1e-4 would instead mean the
    # candidate tokens start almost blind to the evidence — every attention contribution enters
    # scaled by 1e-4 per layer, so a retrieved row moves a candidate logit by ~1e-8 at step 0. It is
    # still a stabiliser, just not an identity switch.
    layerscale_init: float = 0.1
    # Coreference vocabularies. Ids are randomised per episode, so these only have to be at least
    # as large as the largest (candidates + distinct background labels) and source-window count a
    # single episode can present.
    n_slots: int = 160
    n_groups: int = 128
    readout_hidden_mult: int = 2
    readout_init_std: float = 0.02
    # Divides the retrieval score before it is normalized into a relative evidence-key log prior.
    # Mirrors `training.evidence.policy.RETRIEVAL_TEMPERATURE`, which is the temperature the same
    # scores are read at everywhere else; duplicated rather than imported so the model package does
    # not depend on the training package.
    score_temperature: float = 0.07


def relative_evidence_attention_bias(
    score: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Convert selected cosine scores to a shift-invariant log prior over valid evidence.

    ``log_softmax + log(N)`` is the log likelihood ratio against a uniform evidence prior. Equal
    scores produce exactly zero bias, while useful relative score differences retain their order
    and gradient. Invalid rows return zero here and are hidden by the caller's padding mask.
    """
    if score.shape != mask.shape:
        raise ValueError(
            f"retrieval score and mask must have the same shape, got "
            f"{tuple(score.shape)} and {tuple(mask.shape)}"
        )
    if temperature <= 0:
        raise ValueError("score temperature must be positive")
    if not bool(mask.any(1).all()):
        raise ValueError("every episode row must contain at least one valid evidence item")
    scaled = score / temperature
    masked = scaled.masked_fill(~mask, float("-inf"))
    log_normalizer = torch.logsumexp(masked, dim=1, keepdim=True)
    count = mask.sum(1, keepdim=True).to(score.dtype)
    bias = scaled - log_normalizer + count.log()
    return bias.masked_fill(~mask, 0.0)


def fourier_time(t: torch.Tensor, n_freqs: int, time_max: float) -> torch.Tensor:
    """(...,) window-relative seconds -> (..., 2*n_freqs) sin/cos features.

    Frequencies are geometrically spaced so both whole-window and patch-scale position resolve. A
    pooled (whole-window) token passes t=0 and gets a constant vector, which is correct: it has no
    position within the window.
    """
    freqs = torch.exp(torch.linspace(math.log(1.0 / time_max), math.log(time_max),
                                     n_freqs, device=t.device))
    ang = t.unsqueeze(-1) * freqs * (2 * math.pi)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class _Block(nn.Module):
    """Pre-LN transformer block with LayerScale.

    `key_padding_mask` is a **float** additive mask, not a bool one: it carries `-inf` on padded
    keys and the retrieval-score bias on evidence keys. `nn.MultiheadAttention` adds a float
    key-padding mask straight onto the attention logits and backpropagates through it, which is what
    lets the candidate loss reach the retriever.
    """

    def __init__(self, cfg: RelationalDecoderConfig):
        super().__init__()
        d = cfg.d_model
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ls1 = nn.Parameter(cfg.layerscale_init * torch.ones(d))
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.ffn_mult * d), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn_mult * d, d),
        )
        self.ls2 = nn.Parameter(cfg.layerscale_init * torch.ones(d))

    def forward(self, x, key_padding_mask, *, need_weights: bool = False):
        h = self.ln1(x)
        a, weights = self.attn(
            h, h, h, key_padding_mask=key_padding_mask,
            need_weights=need_weights, average_attn_weights=need_weights,
        )
        x = x + self.ls1 * a
        return x + self.ls2 * self.ffn(self.ln2(x)), weights


class RelationalEvidenceDecoder(nn.Module):
    _COMPONENT_NAMES = ("signal", "text", "role", "slot", "time", "group", "relation")

    def __init__(self, cfg: RelationalDecoderConfig | None = None, **kw):
        super().__init__()
        cfg = cfg or RelationalDecoderConfig(**kw)
        self.cfg = cfg
        d = cfg.d_model

        self.proj_query = nn.Linear(cfg.d_model, d)      # f(z_q)
        self.proj_evidence = nn.Linear(cfg.d_model, d)   # g(z_m)  — deliberately not shared with f
        self.proj_text = nn.Linear(cfg.text_dim, d)      # frozen SBERT name -> token
        self.proj_time = nn.Linear(2 * cfg.n_time_freqs, d)
        self.role_emb = nn.Embedding(N_ROLES, d)
        self.slot_emb = nn.Embedding(cfg.n_slots, d)
        self.group_emb = nn.Embedding(cfg.n_groups, d)
        # Relations to the query. Absolute acquisition values already shaped the frozen Phase-A
        # vector; what Phase B cannot recover from z is how the row stands relative to this query.
        self.same_config_emb = nn.Embedding(2, d)
        self.same_subject_emb = nn.Embedding(2, d)
        self.same_sensor_emb = nn.Embedding(2, d)
        # Every token is an additive composition of independently parameterized information
        # sources. PyTorch embeddings otherwise start with norm ~= sqrt(d), while a projection of
        # an L2-normalized content vector starts with norm < 1. LayerNorm after the sum cannot undo
        # that ratio: it normalizes the mixture, not its ingredients. Normalize each ingredient to
        # a direction and give it one positive, auditable learned scale. All paths therefore begin
        # equally usable without preventing the model from learning their relative importance.
        self.component_log_scale = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(())) for name in self._COMPONENT_NAMES
        })
        self.in_ln = nn.LayerNorm(d)

        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_layers))
        self.final_ln = nn.LayerNorm(d)
        self.readout = nn.Sequential(
            nn.Linear(d, cfg.readout_hidden_mult * d), nn.GELU(),
            nn.Linear(cfg.readout_hidden_mult * d, 1),
        )
        # Deliberately NOT zero-initialised. A zero readout was correct while the logit was a
        # residual over a closed-form base — it made the model start exactly at that base. With the
        # base removed the readout *is* the prediction, so zeroing it would emit identical logits
        # for every candidate at step 0, making the model a constant predictor and starving the
        # attention stack of gradient on the first step.
        nn.init.normal_(self.readout[-1].weight, std=cfg.readout_init_std)
        nn.init.zeros_(self.readout[-1].bias)

    def _component(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Return one unit-direction token ingredient with a learned positive scale."""
        return F.normalize(value, dim=-1) * self.component_log_scale[name].exp()

    def component_scales(self) -> dict[str, torch.Tensor]:
        """Expose the seven relative input scales for telemetry and numerical tests."""
        return {name: value.exp() for name, value in self.component_log_scale.items()}

    # ------------------------------------------------------------------ #
    def forward(
        self,
        *,
        cand_text,            # (B, C, text) or (C, text)
        label_text,           # (B, L, text)   background label names present in the pool
        label_mask,           # (B, L) bool
        slot_ids,             # (B, C+L+1) long  randomised slot ids; last entry is UNBOUND
        zq,                   # (B, P, d)
        q_mask,               # (B, P) bool
        zev,                  # (B, M, d)
        ev_mask,              # (B, M) bool
        ev_slot,              # (B, M) long    index into slot_ids (C+L == unbound)
        ev_support_mask,      # (B, M) bool    declared support rather than retrieved background
        ev_score=None,        # (B, M) float   differentiable retrieval similarity for this row
        score_temperature=None, # scalar training override; deployment uses cfg.score_temperature
        q_time=None,          # (B, P)  window-relative seconds
        ev_time=None,         # (B, M)
        q_group=None,         # (B, P) long    source-window group ids (randomised per episode)
        ev_group=None,        # (B, M) long
        ev_same_config=None,  # (B, M) bool    same acquisition configuration as the query
        ev_same_subject=None, # (B, M) bool
        ev_same_sensor=None,  # (B, M) bool
        return_aux=False,
        return_attention=False,
    ):
        B, M, d = zev.shape
        device = zev.device
        P = zq.shape[1]
        if cand_text.dim() == 2:
            cand_text = cand_text.unsqueeze(0).expand(B, -1, -1)
        # Phase B consumes representation directions. Enforce the contract at the model boundary so
        # an alternate caller cannot turn embedding norm into a dataset/candidate shortcut.
        cand_text = F.normalize(cand_text, dim=-1)
        label_text = F.normalize(label_text, dim=-1)
        zq = F.normalize(zq, dim=-1)
        zev = F.normalize(zev, dim=-1)
        C, L = cand_text.shape[1], label_text.shape[1]
        if slot_ids.shape != (B, C + L + 1):
            raise ValueError(
                f"slot_ids must have shape {(B, C + L + 1)} (candidates, labels, unbound), "
                f"got {tuple(slot_ids.shape)}"
            )
        if int(slot_ids.max()) >= self.cfg.n_slots:
            raise ValueError(
                f"slot id {int(slot_ids.max())} exceeds the slot vocabulary "
                f"({self.cfg.n_slots}); raise RelationalDecoderConfig.n_slots"
            )
        if int(ev_slot.max()) > C + L or int(ev_slot.min()) < 0:
            raise ValueError("ev_slot must index slot_ids, i.e. lie in [0, C+L]")

        def optional(value, shape, dtype):
            if value is None:
                return torch.zeros(shape, dtype=dtype, device=device)
            if tuple(value.shape) != shape:
                raise ValueError(f"expected shape {shape}, got {tuple(value.shape)}")
            return value.to(device=device, dtype=dtype)

        q_time = optional(q_time, (B, P), torch.float32)
        ev_time = optional(ev_time, (B, M), torch.float32)
        q_group = optional(q_group, (B, P), torch.long)
        ev_group = optional(ev_group, (B, M), torch.long)
        for name, group in (("q_group", q_group), ("ev_group", ev_group)):
            if int(group.max()) >= self.cfg.n_groups:
                raise ValueError(
                    f"{name} id {int(group.max())} exceeds the group vocabulary "
                    f"({self.cfg.n_groups}); raise RelationalDecoderConfig.n_groups"
                )

        def role_component(role_id: int, shape: tuple[int, int]) -> torch.Tensor:
            ids = torch.full(shape, role_id, device=device, dtype=torch.long)
            return self._component("role", self.role_emb(ids))

        def slot_component(ids: torch.Tensor) -> torch.Tensor:
            return self._component("slot", self.slot_emb(ids))

        def group_component(ids: torch.Tensor) -> torch.Tensor:
            return self._component("group", self.group_emb(ids))

        def time_component(value: torch.Tensor) -> torch.Tensor:
            encoded = fourier_time(value, self.cfg.n_time_freqs, self.cfg.time_max_sec)
            return self._component("time", self.proj_time(encoded))

        # --- name tokens: the only place a label's text enters the model ---
        cand_tok = (self._component("text", self.proj_text(cand_text))
                    + role_component(ROLE_CANDIDATE, (B, C))
                    + slot_component(slot_ids[:, :C]))
        label_tok = (self._component("text", self.proj_text(label_text))
                     + role_component(ROLE_LABEL, (B, L))
                     + slot_component(slot_ids[:, C:C + L])) if L else \
            torch.zeros(B, 0, d, device=device, dtype=cand_tok.dtype)

        # --- query patches ---
        q_tok = (self._component("signal", self.proj_query(zq))
                 + role_component(ROLE_QUERY, (B, P))
                 + time_component(q_time)
                 + group_component(q_group))

        # --- evidence rows: signal plus pointers, never names ---
        ev_role = torch.full((B, M), ROLE_EVIDENCE, device=device, dtype=torch.long)
        ev_role = ev_role.masked_fill(ev_support_mask, ROLE_SUPPORT)
        relation = (
            F.normalize(
                self.same_config_emb(optional(ev_same_config, (B, M), torch.bool).long()),
                dim=-1,
            )
            + F.normalize(
                self.same_subject_emb(optional(ev_same_subject, (B, M), torch.bool).long()),
                dim=-1,
            )
            + F.normalize(
                self.same_sensor_emb(optional(ev_same_sensor, (B, M), torch.bool).long()),
                dim=-1,
            )
        )
        ev_tok = (self._component("signal", self.proj_evidence(zev))
                  + self._component("role", self.role_emb(ev_role))
                  + slot_component(torch.gather(slot_ids, 1, ev_slot))
                  + time_component(ev_time)
                  + group_component(ev_group)
                  + self._component("relation", relation))
        if ev_score is not None and tuple(ev_score.shape) != (B, M):
            raise ValueError(f"ev_score must have shape {(B, M)}, got {tuple(ev_score.shape)}")
        score_temperature = (
            self.cfg.score_temperature
            if score_temperature is None else float(score_temperature)
        )
        if score_temperature <= 0:
            raise ValueError("score_temperature must be positive")

        x = self.in_ln(torch.cat([cand_tok, label_tok, q_tok, ev_tok], dim=1))
        # Candidates are always valid keys, so no row can be fully padded and the softmax is safe.
        padded = torch.cat([
            torch.zeros(B, C, dtype=torch.bool, device=device),
            ~label_mask if L else torch.zeros(B, 0, dtype=torch.bool, device=device),
            ~q_mask, ~ev_mask,
        ], dim=1)
        # One float additive key mask carries both jobs: -inf hides padding, and a normalized
        # relative retrieval prior biases evidence rows against one another. Names and query patches
        # get the uniform-prior value zero. This is the retriever's entire gradient path — see the
        # module docstring.
        ev_bias = (
            relative_evidence_attention_bias(
                ev_score.to(x.dtype), ev_mask, score_temperature
            ) if ev_score is not None
            else torch.zeros(B, M, dtype=x.dtype, device=device)
        )
        key_bias = torch.cat([
            torch.zeros(B, C + L + P, dtype=x.dtype, device=device), ev_bias,
        ], dim=1).masked_fill(padded, float("-inf"))

        attention = None
        for block in self.blocks:
            x, weights = block(x, key_padding_mask=key_bias, need_weights=return_attention)
            if weights is not None:
                attention = weights                       # last block's head-averaged weights

        x = self.final_ln(x)

        logits = self.readout(x[:, :C, :]).squeeze(-1)                       # (B, C)
        if return_aux or return_attention:
            aux = {
                "logit_abs_mean": float(logits.detach().abs().mean()),
                "logit_spread": float(
                    (logits.detach().max(1).values - logits.detach().min(1).values).mean()
                ),
                "candidate_state": x[:, :C, :],
                "component_scales": {
                    name: float(value.detach())
                    for name, value in self.component_scales().items()
                },
            }
            if ev_score is not None and return_attention:
                score_detached = ev_score.detach().float()
                bias_detached = ev_bias.detach().float()
                valid_scores = score_detached[ev_mask]
                valid_bias = bias_detached[ev_mask]
                row_count = ev_mask.sum(1).clamp_min(1).float()
                row_mean = (
                    score_detached.masked_fill(~ev_mask, 0.0).sum(1) / row_count
                )
                row_variance = (
                    (score_detached - row_mean.unsqueeze(1)).square()
                    .masked_fill(~ev_mask, 0.0).sum(1) / row_count
                )
                aux.update({
                    "retrieval_score_mean": float(valid_scores.mean()),
                    "retrieval_score_within_row_std": float(row_variance.sqrt().mean()),
                    "retrieval_attention_bias_abs_mean": float(valid_bias.abs().mean()),
                    "retrieval_attention_bias_abs_max": float(valid_bias.abs().max()),
                })
            if attention is not None:
                # The explicit per-row vote is gone; this is what replaces it as the account of
                # "which stored examples decided this". Rows are candidates, columns evidence.
                aux["candidate_evidence_attention"] = attention[:, :C, C + L + P:]
                aux["candidate_label_attention"] = attention[:, :C, C:C + L]
                aux["candidate_query_attention"] = attention[:, :C, C + L:C + L + P]
                aux["candidate_candidate_attention"] = attention[:, :C, :C]
                candidate_attention = attention[:, :C]
                # MultiheadAttention reports post-dropout weights during training. Renormalize the
                # surviving mass before treating it as a probability distribution; otherwise role
                # masses need not sum to one and normalized entropy can exceed one.
                candidate_attention_detached = candidate_attention.detach()
                candidate_attention_detached = (
                    candidate_attention_detached
                    / candidate_attention_detached.sum(-1, keepdim=True).clamp_min(1e-12)
                )
                valid_keys = (~padded).sum(1).clamp_min(2).float()
                attention_entropy = -(
                    candidate_attention_detached
                    * candidate_attention_detached.clamp_min(1e-12).log()
                ).sum(-1) / valid_keys.log().unsqueeze(1)
                aux.update({
                    "candidate_attention_normalized_entropy": float(
                        attention_entropy.mean()
                    ),
                    "candidate_to_candidate_attention_mass": float(
                        candidate_attention_detached[:, :, :C].sum(-1).mean()
                    ),
                    "candidate_to_label_attention_mass": float(
                        candidate_attention_detached[:, :, C:C + L].sum(-1).mean()
                    ) if L else 0.0,
                    "candidate_to_query_attention_mass": float(
                        candidate_attention_detached[:, :, C + L:C + L + P].sum(-1).mean()
                    ),
                    "candidate_to_evidence_attention_mass": float(
                        candidate_attention_detached[:, :, C + L + P:].sum(-1).mean()
                    ),
                })
            return logits, aux
        return logits

    def param_groups(self, weight_decay: float = 0.01):
        """Split parameters for weight-decay grouping (exclude LN/bias/LayerScale/embeddings)."""
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim <= 1 or "ls1" in name or "ls2" in name or "emb" in name:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]


# --------------------------------------------------------------------------- #
# Input assembly. These live beside the decoder because they define what a slot
# means; the guarantee that slot ids are exact, per-episode random, and globally
# meaningless is only true if the model's inputs are built exactly this way.
# --------------------------------------------------------------------------- #

@dataclass
class CoreferenceSlots:
    label_ids: torch.Tensor    # (B, L) vocabulary id of each background-label token
    label_mask: torch.Tensor   # (B, L) True where the label token is real
    ev_slot: torch.Tensor      # (B, M) index into slot_ids; C+L addresses the unbound slot
    slot_ids: torch.Tensor     # (B, C+L+1) randomised slot embedding ids


def _row_permutation(n: int, n_slots: int, generator: torch.Generator | None) -> torch.Tensor:
    """`n` distinct slot ids drawn from [1, n_slots); id 0 is reserved for the unbound slot."""
    if n > n_slots - 1:
        raise ValueError(
            f"episode needs {n} coreference slots but the vocabulary holds {n_slots - 1}; "
            "raise RelationalDecoderConfig.n_slots"
        )
    return torch.randperm(n_slots - 1, generator=generator)[:n] + 1


def build_coreference_slots(
    ev_label: torch.Tensor,
    ev_mask: torch.Tensor,
    ev_support_mask: torch.Tensor,
    ev_support_candidate: torch.Tensor,
    n_candidates: int,
    *,
    n_slots: int,
    generator: torch.Generator | None = None,
) -> CoreferenceSlots:
    """Bind evidence rows to the name tokens they refer to.

    A declared support row points at its candidate; a retrieved background row points at a label
    token created for its stored label. Ids are re-drawn for every episode so the model can read
    "these tokens refer to the same thing" and nothing else — a fixed id would let it learn that,
    say, slot 3 is usually walking, which is exactly the shortcut this design exists to remove.
    """
    if ev_label.shape != ev_mask.shape or ev_label.shape != ev_support_mask.shape:
        raise ValueError("evidence label/mask/support shapes disagree")
    device = ev_label.device
    B, M = ev_label.shape
    background = ev_mask & ~ev_support_mask
    labels_cpu = ev_label.detach().cpu()
    background_cpu = background.detach().cpu()

    declared = (ev_mask & ev_support_mask).detach().cpu()
    candidate_cpu = ev_support_candidate.detach().cpu()
    if bool(declared.any()):
        bound = candidate_cpu[declared]
        if bool(((bound < 0) | (bound >= n_candidates)).any()):
            raise ValueError(
                "a row marked as declared support has no valid candidate position to bind to; "
                "coreference would be silently wrong"
            )

    per_row = [torch.unique(labels_cpu[b][background_cpu[b]]) for b in range(B)]
    L = max((len(labels) for labels in per_row), default=0)

    label_ids = torch.zeros(B, L, dtype=torch.long)
    label_mask = torch.zeros(B, L, dtype=torch.bool)
    ev_slot = torch.full((B, M), n_candidates + L, dtype=torch.long)
    slot_ids = torch.zeros(B, n_candidates + L + 1, dtype=torch.long)
    for b, labels in enumerate(per_row):
        count = len(labels)
        if count:
            label_ids[b, :count] = labels
            label_mask[b, :count] = True
            rows = background_cpu[b]
            # searchsorted is exact here: `labels` is the sorted unique set of these very values.
            ev_slot[b][rows] = n_candidates + torch.searchsorted(labels, labels_cpu[b][rows])
        if bool(declared[b].any()):
            ev_slot[b][declared[b]] = candidate_cpu[b][declared[b]]
        slot_ids[b, :n_candidates + L] = _row_permutation(n_candidates + L, n_slots, generator)
        slot_ids[b, n_candidates + L] = UNBOUND_SLOT

    return CoreferenceSlots(
        label_ids=label_ids.to(device), label_mask=label_mask.to(device),
        ev_slot=ev_slot.to(device), slot_ids=slot_ids.to(device),
    )


def build_window_groups(
    q_window: torch.Tensor,
    q_mask: torch.Tensor,
    ev_window: torch.Tensor,
    ev_mask: torch.Tensor,
    *,
    n_groups: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomised per-episode ids marking which tokens came from the same source window.

    This is the same coreference device applied to provenance, and it is what replaces the old
    decoder's pairwise same-window attention bias: co-membership becomes a shared embedding rather
    than a learned scalar on every token pair.
    """
    device = q_window.device
    B = q_window.shape[0]
    windows = torch.cat([q_window.detach().cpu(), ev_window.detach().cpu()], dim=1)
    valid = torch.cat([q_mask.detach().cpu(), ev_mask.detach().cpu()], dim=1)
    group = torch.full(windows.shape, UNBOUND_GROUP, dtype=torch.long)
    for b in range(B):
        present = torch.unique(windows[b][valid[b]])
        if not len(present):
            continue
        ids = _row_permutation(len(present), n_groups, generator)
        rows = valid[b]
        group[b][rows] = ids[torch.searchsorted(present, windows[b][rows])]
    q_group = group[:, :q_window.shape[1]].to(device)
    ev_group = group[:, q_window.shape[1]:].to(device)
    return q_group, ev_group


def label_text_votes(ev_label_text: torch.Tensor, cand_text: torch.Tensor) -> torch.Tensor:
    """(B,K,C) rectified cosine between each evidence row's label text and each candidate's.

    This kernel used to *be* the classifier, which is why it failed: on the corpus vocabulary its
    off-diagonal mean is 0.273 and 99.6% of pairs survive the rectifier. It is not on the forward
    path. It survives only as the ingredient of `retrieval_vote_base` — the untrained control every
    Phase-B result is quoted against — and as a confidence statistic.
    """
    if cand_text.dim() == 2:
        cand_text = cand_text.unsqueeze(0).expand(ev_label_text.shape[0], -1, -1)
    return torch.relu(torch.einsum(
        "bkt,bct->bkc", F.normalize(ev_label_text, dim=-1), F.normalize(cand_text, dim=-1)
    ))


def retrieval_vote_base(
    ev_label_text: torch.Tensor,
    cand_text: torch.Tensor,
    w_retr: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """The Stage-2 base term: the untrained retrieval + text-ensemble vote, in closed form.

    This is a **reporting control only** — the untrained retrieval + text-ensemble mechanism that
    every Phase-B result is quoted against. It is not on the forward path: the decoder's logit comes
    from the readout alone. Keep it computed beside each result so a trained model always has to
    beat the mechanism it replaced, on the same episode.
    """
    return scale * torch.einsum("bk,bkc->bc", w_retr, label_text_votes(ev_label_text, cand_text))
