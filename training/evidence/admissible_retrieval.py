"""Phase-B Stage 1 — closed-form admissibility-gated retrieval.

THE DESIGN OF RECORD'S PREDICTION RULE (docs/design/DESIGN_OF_RECORD.md). Not a fallback: across
v19 and v20 the closed form beat every learned readout (65.4 vs 45.5 at k=8, tying prototype and
ridge), so it is the champion and this promotes it to the design.

    1. COMPATIBILITY FILTER (hard)   accel<->accel, gyro<->gyro, gravity state. Makes the comparison
                                      meaningful AT ALL. Reads only the row's modality and gravity
                                      codes, so it is label-independent and already works unchanged
                                      on a vocabulary it has never seen.
    2. ADMISSIBILITY (soft)          score query and evidence sensor text against the candidate.
    3. RANK                          feature cosine plus log-admissibility, per candidate.
    4. VOTE                          enrolled rows vote their bound candidate by identity; corpus
                                      rows vote through label text (the k=0 path).
    5. MERGE ACROSS THE QUERY'S SENSORS   sum votes; per-sensor weights at most.

WHY THE TWO FILTERS ARE SEPARATE, AND WHY IT MATTERS
----------------------------------------------------
Hard filter = "is this comparison interpretable" (an accelerometer and a gyroscope measure different
physical quantities; their cosine is meaningless). Admissibility = "is this evidence relevant to THIS
concept" (a pocket phone is strong evidence about gait and none at all about what the arms are
doing, however similar the signal).

Collapsing them loses the contribution. In particular admissibility must never become a fixed
placement filter: wrist evidence genuinely bears on ambulation queries and genuinely does not bear on
arm gestures, so the weight is per-(row, CANDIDATE) and a query from an unusual placement still
retrieves whatever it can legitimately bear on.

WHAT REPLACED THE MEASURED-TABLE LOOKUP
---------------------------------------
Admissibility used to be a dictionary read keyed by the literal `"<dataset>/<stream>"` string and an
exact label match, with a neutral default on a miss. On a novel vocabulary every entry defaulted, the
gate became a uniform multiplier, and it provably could not change the argmax — no behaviour at all
in exactly the case the design exists for. `AdmissibilityGate` makes it a function of text instead.
The measured table survives as the gate's warm start, never as a runtime lookup. Held-out folds are
the independent check; the cells used for fitting are not validation data.

WHY THE BIAS BLEND SHIPS WITH A GUARD
-------------------------------------
``sensor_bias`` is close to a dataset fingerprint in this corpus (one device model per dataset), so
an additive bias-similarity term pulls toward same-dataset retrieval — inflating every number while
looking like the mechanism working, exactly the shape of the hapt==uci_har trap.
``retrieval_provenance`` measures that pull directly and must be reported beside any result that
enables the blend.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Additive weight on sensor_bias similarity when ranking. DEFAULT ZERO: the provenance probe reads
# dataset identity out of the pooled feature at BA 0.702, so a bias-similarity term pulls toward
# same-dataset retrieval, inflating every number while looking like the mechanism working. Enable it
# only alongside a reported `retrieval_provenance` with the blend on and off.
BIAS_BLEND_WEIGHT = 0.0

@dataclass(frozen=True)
class SensorRows:
    """A bank slice, keyed per patch per sensor. All tensors share a leading row dimension R."""

    feature: torch.Tensor          # (R, d)   signal embedding
    descriptor: torch.Tensor       # (R, 384) frozen SBERT of the sensor text
    bias: torch.Tensor             # (R, F)   sensor_bias
    modality: torch.Tensor         # (R,)     0=accel, 1=gyro
    gravity: torch.Tensor          # (R,)     0=present, 1=removed
    label: torch.Tensor            # (R,)     vocab index, -1 = unlabelled corpus row
    dataset: torch.Tensor          # (R,)     provenance only, never scored
    enrolled_candidate: torch.Tensor  # (R,)  candidate slot this row is bound to, -1 = corpus row

    def __post_init__(self) -> None:
        R = self.feature.shape[0]
        for name in ("descriptor", "bias", "modality", "gravity", "label", "dataset",
                     "enrolled_candidate"):
            got = getattr(self, name).shape[0]
            if got != R:
                raise ValueError(f"{name} has {got} rows, expected {R}")

    def to(self, device) -> "SensorRows":
        return SensorRows(**{
            name: getattr(self, name).to(device)
            for name in self.__dataclass_fields__
        })


def compatibility_mask(
    query_modality: torch.Tensor,     # (Q,)
    query_gravity: torch.Tensor,      # (Q,)
    rows: SensorRows,
) -> torch.Tensor:
    """(Q, R) True where a comparison is physically interpretable.

    Modality must match: an accelerometer measures proper acceleration and a gyroscope angular rate,
    so their embeddings are not commensurable regardless of what the encoder learned. Gravity state
    must match for accelerometers: a gravity-removed stream has |DC| ~ 0 against ~1 g, which the
    signed DC feature reads directly — comparing across the convention compares an artifact of
    preprocessing, which is the exact defect found in kuhar and uci_har.
    """
    same_modality = query_modality.unsqueeze(1) == rows.modality.unsqueeze(0)
    same_gravity = query_gravity.unsqueeze(1) == rows.gravity.unsqueeze(0)
    is_accel = (rows.modality.unsqueeze(0) == 0)
    return same_modality & (same_gravity | ~is_accel)


def rank_scores(
    query_feature: torch.Tensor,      # (Q, d)
    query_bias: torch.Tensor,         # (Q, F)
    rows: SensorRows,
    compatible: torch.Tensor,         # (Q, R)
    bias_weight: float = BIAS_BLEND_WEIGHT,
) -> torch.Tensor:
    """(Q, R) similarity. Feature cosine ranks; sensor_bias nudges. Incompatible rows are -inf."""
    qf = F.normalize(query_feature, dim=-1)
    rf = F.normalize(rows.feature, dim=-1)
    score = qf @ rf.t()
    if bias_weight:
        qb = F.normalize(query_bias, dim=-1)
        rb = F.normalize(rows.bias, dim=-1)
        score = score + bias_weight * (qb @ rb.t())
    return score.masked_fill(~compatible, float("-inf"))


def vote(
    scores: torch.Tensor,             # (Q, R) rank scores, -inf where incompatible
    rows: SensorRows,
    candidate_text: torch.Tensor,     # (C, 384) L2-normalised candidate label text
    label_text: torch.Tensor,         # (V, 384) L2-normalised vocab label text
    admissibility: torch.Tensor,      # (Q, R, C) in [0,1] from AdmissibilityGate
    top_k: int,
    temperature: float = 0.07,
    allow_corpus_text_vote: bool = True,
) -> torch.Tensor:
    """(Q, C) candidate logits, selecting the top-k pool SEPARATELY FOR EACH CANDIDATE.

    WHY PER-CANDIDATE SELECTION
    ---------------------------
    Admissibility is a per-(row, candidate) quantity, so a single shared top-k chosen by feature
    similarity would let similarity decide the pool and leave the gate only able to reweight what
    survived. If all k nearest rows come from a configuration that cannot witness the candidate, a
    post-hoc gate can shrink their votes but cannot reach past them to evidence that could. Ranking
    per candidate lets a low-admissibility row lose its slot to a more useful row, which makes
    admissibility affect retrieval rather than merely break similarity ties.

    Enrolled rows vote their BOUND candidate by identity — the alias arm established that this is
    what carries the prediction at k>=1, and that real label names contribute nothing there. Corpus
    rows vote through rectified label-text cosine, which is the ConSE bridge and the only mechanism
    available at k=0.
    """
    Q, R = scores.shape
    C = candidate_text.shape[0]
    if admissibility.shape != (Q, R, C):
        raise ValueError(f"admissibility must be {(Q, R, C)}, got {tuple(admissibility.shape)}")
    k = min(top_k, R)

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    gate = admissibility.clamp(0.0, 1.0)
    candidate_index = torch.arange(C, device=scores.device).view(1, C, 1)
    bound_by_row = rows.enrolled_candidate.to(scores.device).view(1, 1, R)
    binding_allowed = bound_by_row.lt(0) | bound_by_row.eq(candidate_index)
    # This is exactly the score used by `soft_vote_all`. Deployment truncates it to top-k;
    # training keeps every compatible row. A learned gate is never a hard exclusion.
    per_candidate = scores.unsqueeze(1).expand(Q, C, R) / temperature + (
        gate.permute(0, 2, 1).clamp_min(1e-8).log()
    )
    # An enrolled row has one explicit episode-local label binding. It cannot vote for another
    # candidate, so allowing it into that candidate's top-k would spend a slot on guaranteed zero
    # evidence and could crowd out a valid corpus or support row.
    per_candidate = per_candidate.masked_fill(~binding_allowed, float("-inf"))
    top_scores, top_idx = per_candidate.topk(k, dim=2)                     # (Q,C,k)
    finite = torch.isfinite(top_scores)
    # Convert adjusted scores to nonnegative evidence mass with one query-wide stabilizer. A
    # per-candidate softmax would cancel query-side admissibility; a joint softmax would preserve the
    # argmax but compress the candidate logits toward 1/C before cross-entropy.
    stable_max = top_scores.masked_fill(~finite, float("-inf")).flatten(1).max(1).values
    stable_max = torch.where(torch.isfinite(stable_max), stable_max, torch.zeros_like(stable_max))
    weights = torch.exp(top_scores - stable_max[:, None, None]) * finite.to(top_scores.dtype)
    # A candidate with no physically compatible/binding-eligible row has only -inf scores.
    weights = torch.nan_to_num(weights, nan=0.0) * finite.to(weights.dtype)

    bound = rows.enrolled_candidate[top_idx]                               # (Q,C,k)
    enrolled_vote = ((bound >= 0) & (bound == candidate_index)).to(scores.dtype)

    row_label = rows.label[top_idx]                                        # (Q,C,k)
    # Rectified cosine between the row's own label text and THIS candidate's text. Contracted
    # directly rather than via a full (Q,C,k,C) matmul whose off-diagonal candidates are discarded.
    if allow_corpus_text_vote:
        text_vote = F.relu(
            (label_text[row_label.clamp_min(0)] * candidate_text.view(1, C, 1, -1)).sum(dim=-1))
        text_vote = text_vote * ((bound < 0) & (row_label >= 0)).to(scores.dtype)
    else:
        text_vote = torch.zeros_like(enrolled_vote)

    return (weights * (enrolled_vote + text_vote)).sum(dim=2)              # (Q,C)


def soft_vote_all(
    scores: torch.Tensor,             # (Q, R), -inf where physically incompatible
    rows: SensorRows,
    candidate_text: torch.Tensor,     # (C, 384)
    label_text: torch.Tensor,         # (V, 384)
    admissibility: torch.Tensor,      # (Q, R, C)
    temperature: float = 0.07,
    allow_corpus_text_vote: bool = True,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable Stage-2 vote over the complete bounded working set.

    Deployment uses top-k on the same adjusted score. During gate refinement that operation gives no
    credit to unselected rows. Here every compatible row participates with weight proportional to
    ``exp(feature_cosine / temperature) * admissibility``. This preserves the Stage-1 ordering rule
    while giving the candidate loss a gradient to every gate value in the active working set.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    Q, R = scores.shape
    C = candidate_text.shape[0]
    if admissibility.shape != (Q, R, C):
        raise ValueError(f"admissibility must be {(Q, R, C)}, got {tuple(admissibility.shape)}")

    gate = admissibility.clamp(1e-8, 1.0)
    candidate_index = torch.arange(C, device=scores.device).view(1, C, 1)
    bound = rows.enrolled_candidate.to(scores.device).view(1, 1, R)
    binding_allowed = bound.lt(0) | bound.eq(candidate_index)
    retrieval_logits = scores.unsqueeze(1) / temperature + gate.permute(0, 2, 1).log()
    retrieval_logits = retrieval_logits.masked_fill(~binding_allowed, float("-inf"))
    finite = torch.isfinite(retrieval_logits)
    stable_max = retrieval_logits.masked_fill(~finite, float("-inf")).flatten(1).max(1).values
    stable_max = torch.where(torch.isfinite(stable_max), stable_max, torch.zeros_like(stable_max))
    weights = torch.exp(retrieval_logits - stable_max[:, None, None]) * finite.to(scores.dtype)
    weights = torch.nan_to_num(weights, nan=0.0)

    bound_by_candidate = rows.enrolled_candidate.view(1, 1, R).to(scores.device)
    enrolled_vote = (
        bound_by_candidate.ge(0) & bound_by_candidate.eq(candidate_index)
    ).to(scores.dtype)
    if allow_corpus_text_vote:
        row_label = rows.label.to(scores.device)
        text_vote = F.relu(
            (label_text[row_label.clamp_min(0)].unsqueeze(0)
             * candidate_text.unsqueeze(1)).sum(dim=-1)
        ).unsqueeze(0)  # (1,C,R)
        text_vote = text_vote * (
            bound_by_candidate.lt(0) & row_label.view(1, 1, R).ge(0)
        ).to(scores.dtype)
    else:
        text_vote = torch.zeros_like(enrolled_vote)
    result = (weights * (enrolled_vote + text_vote)).sum(dim=2)
    if not return_stats:
        return result
    flat_weights = weights.flatten(1)
    normalized_weights = flat_weights / flat_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(normalized_weights * normalized_weights.clamp_min(1e-12).log()).sum(dim=1)
    valid_count = finite.flatten(1).sum(dim=1).clamp_min(1)
    normalized = entropy / valid_count.clamp_min(2).float().log()
    return result, {
        "retrieval/normalized_entropy": normalized.mean().detach(),
        "retrieval/effective_rows": entropy.exp().mean().detach(),
        "retrieval/available_rows": valid_count.float().mean().detach(),
        "retrieval/max_weight": normalized_weights.max(dim=1).values.mean().detach(),
    }


def merge_sensors(
    per_sensor_logits: torch.Tensor,  # (S, C)
    sensor_weight: torch.Tensor | None = None,   # (S,)
) -> torch.Tensor:
    """(C,) — cross-sensor fusion, closed form.

    This is where a phone and a watch on the same body combine. It is a vote sum rather than an
    attention operation on purpose: the two sensors observe DIFFERENT things, and constraint 1 of
    the design says nothing may require inferring one placement's signal from another's. Summing
    independently-derived votes requires no such inference.
    """
    if sensor_weight is None:
        return per_sensor_logits.sum(dim=0)
    return (per_sensor_logits * sensor_weight.unsqueeze(-1)).sum(dim=0)


def retrieval_provenance(
    scores: torch.Tensor,             # (Q, R)
    rows: SensorRows,
    query_dataset: torch.Tensor,      # (Q,)
    top_k: int,
) -> dict[str, float]:
    """THE GUARD on the bias blend: how often does retrieval land in the query's own dataset?

    Run it with the blend on and off. If enabling ``sensor_bias`` raises this materially, the term
    is matching provenance rather than channel physics, and every downstream number is inflated.
    Report it beside any result that enables the blend — it is not optional colour.
    """
    k = min(top_k, scores.shape[1])
    _, idx = scores.topk(k, dim=1)
    same = (rows.dataset[idx] == query_dataset.unsqueeze(1))
    n_datasets = int(rows.dataset.unique().numel())
    return {
        "same_dataset_fraction": float(same.float().mean()),
        "chance": 1.0 / max(n_datasets, 1),
        "n_datasets": n_datasets,
        "top_k": k,
    }


def admissibility_from_unique(
    gate,                             # AdmissibilityGate
    unique_descriptor: torch.Tensor,  # (U, 384) the DISTINCT sensor descriptions in the bank
    row_descriptor_id: torch.Tensor,  # (R,) which distinct description each row carries
    candidate_text: torch.Tensor,     # (C, 384)
    query_descriptor: torch.Tensor,   # (Q, 384)
    semantic_labels: bool = True,
) -> torch.Tensor:
    """(Q,R,C) query-and-row observability, with exact descriptor de-duplication.

    A bank holds ~10^5 rows drawn from ~40 distinct sensor descriptions, so evaluating the gate per
    row would repeat the same projection thousands of times. The values are identical by
    construction — the gate reads only the text — so this is an exact speedup, not an approximation.
    """
    Q, R, C = len(query_descriptor), len(row_descriptor_id), len(candidate_text)
    if not semantic_labels:
        return torch.ones(Q, R, C, device=query_descriptor.device)
    row_gate = gate(unique_descriptor, candidate_text).index_select(0, row_descriptor_id)  # (R,C)
    query_gate = gate(query_descriptor, candidate_text)                    # (Q,C)
    # Both recordings must be able to witness the candidate. The geometric mean is a soft logical
    # AND that gives gradients to both factors while preserving the scale on which the single-sensor
    # gate was calibrated. A raw product would square typical values and overstate its penalty.
    # Compute the geometric mean in log space. This is mathematically identical to sqrt(q*r), but
    # avoids an infinite/undefined derivative if sigmoid outputs underflow near zero.
    return torch.exp(0.5 * (
        query_gate.unsqueeze(1).clamp_min(1e-8).log()
        + row_gate.unsqueeze(0).clamp_min(1e-8).log()
    ))


def admissibility_from_gate(
    gate,                             # AdmissibilityGate
    rows: SensorRows,
    candidate_text: torch.Tensor,     # (C, 384) candidate concept text
    query_descriptor: torch.Tensor,   # (Q, 384) query sensor descriptions
    semantic_labels: bool = True,
) -> torch.Tensor:
    """(S, R, C) admissibility from the learned gate, replacing the measured-table lookup.

    The gate reads each BANK ROW's own frozen sensor descriptor against each candidate's text, so an
    unseen configuration and an unseen label both resolve — where the string-keyed table returned its
    neutral default and left the gate a uniform multiplier that could not move the argmax.

    Both the query and evidence descriptions are scored. This prevents a wrist support from making
    an arm concept observable to a pocket query, while still allowing cross-placement evidence when
    both placements can witness the candidate.
    """
    if not semantic_labels:
        return torch.ones(
            len(query_descriptor), len(rows.feature), len(candidate_text),
            device=query_descriptor.device,
        )
    row_gate = gate(rows.descriptor, candidate_text)
    query_gate = gate(query_descriptor, candidate_text)
    return torch.exp(0.5 * (
        query_gate.unsqueeze(1).clamp_min(1e-8).log()
        + row_gate.unsqueeze(0).clamp_min(1e-8).log()
    ))


def predict(
    query_feature: torch.Tensor,      # (S, d) one row per the QUERY's sensors
    query_bias: torch.Tensor,         # (S, F)
    query_modality: torch.Tensor,     # (S,)
    query_gravity: torch.Tensor,      # (S,)
    rows: SensorRows,
    candidate_text: torch.Tensor,     # (C, 384)
    label_text: torch.Tensor,         # (V, 384)
    admissibility: torch.Tensor | None = None,
    query_descriptor: torch.Tensor | None = None,  # (S, 384), required with ``gate``
    top_k: int = 64,
    bias_weight: float = BIAS_BLEND_WEIGHT,
    temperature: float = 0.07,
    sensor_weight: torch.Tensor | None = None,
    gate=None,
    semantic_labels: bool = True,
) -> torch.Tensor:
    """(C,) candidate logits for one query window.

    Hard compatibility removes physically meaningless comparisons. Feature similarity and soft
    admissibility then jointly allocate evidence mass before top-k truncation.
    """
    compatible = compatibility_mask(query_modality, query_gravity, rows)
    scores = rank_scores(query_feature, query_bias, rows, compatible, bias_weight=bias_weight)
    if admissibility is None:
        if gate is None:
            raise ValueError("pass either an admissibility tensor or a gate")
        if query_descriptor is None:
            raise ValueError("query_descriptor is required when computing admissibility from a gate")
        admissibility = admissibility_from_gate(
            gate, rows, candidate_text, query_descriptor, semantic_labels=semantic_labels
        )
    per_sensor = vote(scores, rows, candidate_text, label_text, admissibility,
                      top_k=top_k, temperature=temperature,
                      allow_corpus_text_vote=semantic_labels)
    return merge_sensors(per_sensor, sensor_weight)
