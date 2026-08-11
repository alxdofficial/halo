"""Gate tests for Phase-B Stage 1 closed-form admissibility-gated retrieval.

Asserts the behaviours the contribution claims, not shapes:
  * accelerometer evidence never scores against a gyroscope query, and vice versa;
  * gravity-removed accelerometry never scores against gravity-present accelerometry;
  * the admissibility gate DOWN-WEIGHTS but never DELETES (placement must stay soft);
  * an enrolled row votes its bound candidate by identity, indifferent to the label string --
    the property the alias arm measured;
  * cross-sensor merge is additive, so a second sensor can only add evidence;
  * the provenance guard actually detects a same-dataset pull.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from training.evidence.admissible_retrieval import (
    GATE_FLOOR, SensorRows, admissibility, compatibility_mask, merge_sensors, predict,
    rank_scores, retrieval_provenance, vote,
)

D, TEXT, C_CAND, V = 16, 384, 3, 5
SEED = 20260811


def _rows(R=12, seed=SEED) -> SensorRows:
    g = torch.Generator().manual_seed(seed)
    return SensorRows(
        feature=torch.randn(R, D, generator=g),
        descriptor=F.normalize(torch.randn(R, TEXT, generator=g), dim=-1),
        bias=torch.randn(R, 9, generator=g),
        modality=torch.tensor([0, 0, 0, 1, 1, 1] * (R // 6)),
        gravity=torch.tensor([0, 0, 1, 0, 0, 1] * (R // 6)),
        label=torch.arange(R) % V,
        dataset=torch.arange(R) % 3,
        enrolled_candidate=torch.full((R,), -1, dtype=torch.long),
    )


def _texts(seed=SEED):
    g = torch.Generator().manual_seed(seed + 1)
    return (F.normalize(torch.randn(C_CAND, TEXT, generator=g), dim=-1),
            F.normalize(torch.randn(V, TEXT, generator=g), dim=-1))


# ------------------------------------------------------------------- compatibility (hard)
def test_modality_never_crosses():
    rows = _rows()
    comp = compatibility_mask(torch.tensor([0]), torch.tensor([0]), rows)
    assert not comp[0, rows.modality == 1].any(), "accel query matched a gyro row"
    comp = compatibility_mask(torch.tensor([1]), torch.tensor([0]), rows)
    assert not comp[0, rows.modality == 0].any(), "gyro query matched an accel row"


def test_gravity_convention_never_crosses_for_accelerometers():
    rows = _rows()
    comp = compatibility_mask(torch.tensor([0]), torch.tensor([0]), rows)
    bad = (rows.modality == 0) & (rows.gravity == 1)
    assert not comp[0, bad].any(), "gravity-present query matched gravity-removed accel"


def test_gyroscopes_ignore_gravity_state():
    """A gyroscope has no gravity component, so the convention is not a comparability axis."""
    rows = _rows()
    comp = compatibility_mask(torch.tensor([1]), torch.tensor([0]), rows)
    assert comp[0, rows.modality == 1].all()


def test_incompatible_rows_are_unrankable():
    rows = _rows()
    comp = compatibility_mask(torch.tensor([0]), torch.tensor([0]), rows)
    scores = rank_scores(torch.randn(1, D), torch.randn(1, 9), rows, comp)
    assert torch.isinf(scores[~comp]).all() and (scores[~comp] < 0).all()
    assert torch.isfinite(scores[comp]).all()


# ------------------------------------------------------------------------- gate (soft)
def test_gate_downweights_but_never_deletes():
    """Placement must stay SOFT: a hard zero would make an unusual-placement query retrieve nothing."""
    gate = admissibility(torch.zeros(2, 4, C_CAND))
    assert torch.allclose(gate, torch.full_like(gate, GATE_FLOOR))
    assert float(gate.min()) > 0.0


def test_gate_is_identity_at_full_resolvability():
    gate = admissibility(torch.ones(2, 4, C_CAND))
    assert torch.allclose(gate, torch.ones_like(gate))


def test_gate_changes_the_prediction():
    rows = _rows()
    cand, lab = _texts()
    comp = compatibility_mask(torch.tensor([0]), torch.tensor([0]), rows)
    scores = rank_scores(torch.randn(1, D), torch.randn(1, 9), rows, comp)
    R = rows.feature.shape[0]
    open_gate = torch.ones(1, R, C_CAND)
    shut = open_gate.clone()
    shut[:, :, 0] = 0.0                       # candidate 0 unobservable by every row
    a = vote(scores, rows, cand, lab, open_gate, top_k=6)
    b = vote(scores, rows, cand, lab, shut, top_k=6)
    assert b[0, 0] < a[0, 0], "gating a candidate must reduce its logit"


# ------------------------------------------------------------------------------ voting
def test_enrolled_rows_vote_by_identity_not_by_label_text():
    """The alias-arm property: at k>=1 the enrolled binding carries it, the name is irrelevant."""
    rows = _rows()
    enrolled = rows.enrolled_candidate.clone()
    enrolled[:3] = 1                                  # first three rows bound to candidate 1
    rows = SensorRows(rows.feature, rows.descriptor, rows.bias, rows.modality, rows.gravity,
                      rows.label, rows.dataset, enrolled)
    cand, lab = _texts()
    comp = compatibility_mask(torch.tensor([0]), torch.tensor([0]), rows)
    scores = rank_scores(rows.feature[:1], rows.bias[:1], rows, comp)
    gate = torch.ones(1, rows.feature.shape[0], C_CAND)
    base = vote(scores, rows, cand, lab, gate, top_k=4)
    # Scramble the candidate NAMES; enrolled votes must be unchanged.
    g = torch.Generator().manual_seed(999)
    aliased = F.normalize(torch.randn(C_CAND, TEXT, generator=g), dim=-1)
    alias = vote(scores, rows, aliased, lab, gate, top_k=4)
    assert torch.argmax(base) == torch.argmax(alias) == 1


def test_corpus_rows_vote_through_label_text():
    """With nothing enrolled, the only mechanism is the ConSE bridge -- the k=0 path."""
    rows = _rows()
    cand, lab = _texts()
    comp = compatibility_mask(torch.tensor([0]), torch.tensor([0]), rows)
    scores = rank_scores(torch.randn(1, D), torch.randn(1, 9), rows, comp)
    gate = torch.ones(1, rows.feature.shape[0], C_CAND)
    out = vote(scores, rows, cand, lab, gate, top_k=6)
    assert out.shape == (1, C_CAND)
    assert torch.isfinite(out).all()
    assert float(out.abs().sum()) > 0.0


# ------------------------------------------------------------------------ sensor merge
def test_merge_is_additive_so_a_second_sensor_only_adds_evidence():
    one = torch.tensor([[1.0, 0.0, 0.0]])
    two = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert torch.allclose(merge_sensors(one), torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(merge_sensors(two), torch.tensor([1.0, 2.0, 0.0]))


def test_merge_respects_per_sensor_weights():
    logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = merge_sensors(logits, torch.tensor([1.0, 0.0]))
    assert torch.allclose(out, torch.tensor([1.0, 0.0, 0.0]))


# --------------------------------------------------------------------- provenance guard
def test_provenance_guard_detects_a_same_dataset_pull():
    rows = _rows(R=12)
    # Craft scores that rank the query's own dataset first.
    scores = torch.where(rows.dataset.unsqueeze(0) == 0,
                         torch.ones(1, 12), torch.zeros(1, 12))
    pulled = retrieval_provenance(scores, rows, torch.tensor([0]), top_k=4)
    assert pulled["same_dataset_fraction"] == 1.0
    neutral = retrieval_provenance(torch.zeros(1, 12), rows, torch.tensor([0]), top_k=12)
    assert neutral["same_dataset_fraction"] < 0.5


# ------------------------------------------------------------------------- end to end
def test_predict_runs_over_multiple_query_sensors():
    rows = _rows()
    cand, lab = _texts()
    R = rows.feature.shape[0]
    out = predict(
        query_feature=torch.randn(2, D),
        query_bias=torch.randn(2, 9),
        query_modality=torch.tensor([0, 1]),
        query_gravity=torch.tensor([0, 0]),
        rows=rows, candidate_text=cand, label_text=lab,
        resolvability=torch.ones(2, R, C_CAND), top_k=6,
    )
    assert out.shape == (C_CAND,)
    assert torch.isfinite(out).all()


def test_predict_is_deterministic():
    rows = _rows()
    cand, lab = _texts()
    R = rows.feature.shape[0]
    kw = dict(query_feature=torch.randn(1, D), query_bias=torch.randn(1, 9),
              query_modality=torch.tensor([0]), query_gravity=torch.tensor([0]),
              rows=rows, candidate_text=cand, label_text=lab,
              resolvability=torch.ones(1, R, C_CAND), top_k=6)
    assert torch.allclose(predict(**kw), predict(**kw))
