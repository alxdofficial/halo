"""Phase-B Stage 1 — closed-form admissibility-gated retrieval.

THE DESIGN OF RECORD'S PREDICTION RULE (docs/design/DESIGN_OF_RECORD.md). Not a fallback: across
v19 and v20 the closed form beat every learned readout (65.4 vs 45.5 at k=8, tying prototype and
ridge), so it is the champion and this promotes it to the design.

    1. COMPATIBILITY FILTER (hard)   accel<->accel, gyro<->gyro, gravity state, units.
                                      Makes the comparison meaningful AT ALL.
    2. RANK                          feature cosine within the admissible pool, plus an additive
                                      sensor_bias blend.
    3. ADMISSIBILITY GATE (soft)     placement- AND concept-dependent. Down-weights evidence that
                                      cannot bear on a candidate.
    4. VOTE                          enrolled rows vote their bound candidate by identity; corpus
                                      rows vote through label text (the k=0 path).
    5. MERGE ACROSS THE QUERY'S SENSORS   sum votes; per-sensor weights at most.

WHY THE TWO FILTERS ARE SEPARATE, AND WHY IT MATTERS
----------------------------------------------------
Hard filter = "is this comparison interpretable" (an accelerometer and a gyroscope measure different
physical quantities; their cosine is meaningless). Soft gate = "is this evidence relevant to THIS
concept" (a pocket phone is strong evidence about gait and none at all about what the arms are
doing, however similar the signal).

Collapsing them loses the contribution. In particular the gate must NEVER hard-filter placement:
wrist evidence genuinely bears on ambulation queries and genuinely does not bear on arm gestures,
and a query from an unusual placement would retrieve nothing at all if placement were a hard filter.

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

# Additive weight on sensor_bias similarity when ranking. Small by construction: the signal decides
# WHICH rows are close, the bias only nudges among rows already close. Stage 2 may learn it.
BIAS_BLEND_WEIGHT = 0.15

# Floor on the admissibility gate. A gated-out row is down-weighted, never deleted — a hard zero
# would make the gate a filter, and the k=0 path depends on weak corpus evidence still counting.
GATE_FLOOR = 0.05


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


def admissibility(
    resolvability: torch.Tensor,      # (Q, R, C) in [0,1]: can this row bear on this candidate?
    floor: float = GATE_FLOOR,
) -> torch.Tensor:
    """Soft per-(query, row, candidate) weight.

    ``resolvability`` is the (source config, target config, candidate concept) function the whole
    design turns on — "a pocket phone cannot witness an arm gesture". Stage 1 supplies it from a
    measured table; Stage 2 learns a correction over it. Floored rather than zeroed so a gated-out
    row is down-weighted, not deleted.
    """
    return floor + (1.0 - floor) * resolvability.clamp(0.0, 1.0)


def vote(
    scores: torch.Tensor,             # (Q, R) rank scores, -inf where incompatible
    rows: SensorRows,
    candidate_text: torch.Tensor,     # (C, 384) L2-normalised candidate label text
    label_text: torch.Tensor,         # (V, 384) L2-normalised vocab label text
    resolvability: torch.Tensor,      # (Q, R, C)
    top_k: int,
    temperature: float = 0.07,
) -> torch.Tensor:
    """(Q, C) candidate logits from the top-k admissible rows.

    Enrolled rows vote their BOUND candidate by identity — the alias arm established that this is
    what carries the prediction at k>=1, and that real label names contribute nothing there. Corpus
    rows vote through rectified label-text cosine, which is the ConSE bridge and the only mechanism
    available at k=0.
    """
    Q, R = scores.shape
    C = candidate_text.shape[0]
    k = min(top_k, R)
    top_scores, top_idx = scores.topk(k, dim=1)                            # (Q,k)
    finite = torch.isfinite(top_scores)
    weights = torch.softmax(top_scores.masked_fill(~finite, float("-inf")) / temperature, dim=1)
    weights = torch.nan_to_num(weights, nan=0.0)

    gate = admissibility(torch.gather(
        resolvability, 1, top_idx.unsqueeze(-1).expand(Q, k, C)))           # (Q,k,C)

    bound = rows.enrolled_candidate[top_idx]                               # (Q,k)
    is_enrolled = bound >= 0
    enrolled_vote = F.one_hot(bound.clamp_min(0), num_classes=C).to(scores.dtype)
    enrolled_vote = enrolled_vote * is_enrolled.unsqueeze(-1).to(scores.dtype)

    row_label = rows.label[top_idx]                                        # (Q,k)
    text_vote = F.relu(label_text[row_label.clamp_min(0)] @ candidate_text.t())
    text_vote = text_vote * (~is_enrolled & (row_label >= 0)).unsqueeze(-1).to(scores.dtype)

    per_row = (enrolled_vote + text_vote) * gate
    return (weights.unsqueeze(-1) * per_row).sum(dim=1)


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


def predict(
    query_feature: torch.Tensor,      # (S, d) one row per the QUERY's sensors
    query_bias: torch.Tensor,         # (S, F)
    query_modality: torch.Tensor,     # (S,)
    query_gravity: torch.Tensor,      # (S,)
    rows: SensorRows,
    candidate_text: torch.Tensor,     # (C, 384)
    label_text: torch.Tensor,         # (V, 384)
    resolvability: torch.Tensor,      # (S, R, C)
    top_k: int = 64,
    bias_weight: float = BIAS_BLEND_WEIGHT,
    temperature: float = 0.07,
    sensor_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """(C,) candidate logits for one query window. Every step closed form; no learned parameters."""
    compatible = compatibility_mask(query_modality, query_gravity, rows)
    scores = rank_scores(query_feature, query_bias, rows, compatible, bias_weight=bias_weight)
    per_sensor = vote(scores, rows, candidate_text, label_text, resolvability,
                      top_k=top_k, temperature=temperature)
    return merge_sensors(per_sensor, sensor_weight)
