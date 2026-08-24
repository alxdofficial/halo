"""Gate tests for Phase-B Stage 1 closed-form admissibility-gated retrieval.

Asserts the behaviours the contribution claims, not shapes:
  * accelerometer evidence never scores against a gyroscope query, and vice versa;
  * gravity-removed accelerometry never scores against gravity-present accelerometry;
  * training and inference use the same continuous admissibility-adjusted score;
  * an enrolled row votes its bound candidate by identity, indifferent to the label string --
    the property the alias arm measured;
  * cross-sensor merge is additive, so a second sensor can only add evidence;
  * the provenance guard actually detects a same-dataset pull.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from training.evidence.admissible_retrieval import (
    SensorRows, compatibility_mask, evidence_label_tokens, merge_sensors, predict,
    rank_scores, retrieval_provenance, soft_vote_all, vote,
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


def test_enrolled_evidence_uses_episode_local_candidate_text():
    rows = _rows()
    candidate, labels = _texts()
    binding = rows.enrolled_candidate.clone()
    binding[:2] = torch.tensor([2, 0])
    rows = SensorRows(**{**rows.__dict__, "label": torch.tensor([-1, -1, *rows.label[2:].tolist()]),
                         "enrolled_candidate": binding})
    tokens = evidence_label_tokens(rows, candidate, labels)
    assert torch.equal(tokens[0], candidate[2])
    assert torch.equal(tokens[1], candidate[0])
    assert torch.equal(tokens[2], labels[rows.label[2]])


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
def test_soft_admissibility_changes_selection_before_top_k():
    """Admissibility must affect selection, rather than only reweight selected rows.

    Row 0 is the closest by feature but has negligible admissibility for candidate 0. The adjusted
    score lets another row occupy the slot without introducing a discontinuous threshold.
    """
    rows = _rows()
    cand, lab = _texts()
    R = rows.feature.shape[0]
    comp = compatibility_mask(torch.tensor([0]), torch.tensor([0]), rows)
    scores = rank_scores(torch.randn(1, D), torch.randn(1, 9), rows, comp)
    scores[0, 0] = 10.0                                   # row 0 is now the nearest by far

    gate = torch.ones(1, R, C_CAND)
    gate[0, 0, 0] = 1e-8                                  # ...but weak for candidate 0
    out = vote(scores, rows, cand, lab, gate, top_k=3)
    assert torch.isfinite(out).all()
    # Candidate 0 still gets evidence from admissible rows rather than a slot wasted on row 0.
    assert float(out[0, 0]) > 0.0


def test_uniformly_tiny_admissibility_remains_soft_and_finite():
    """A learned score may strongly discourage evidence but must not delete all gradient paths."""
    rows = _rows()
    cand, lab = _texts()
    R = rows.feature.shape[0]
    comp = compatibility_mask(torch.tensor([0]), torch.tensor([0]), rows)
    scores = rank_scores(torch.randn(1, D), torch.randn(1, 9), rows, comp)
    gate = torch.ones(1, R, C_CAND)
    gate[0, :, 0] = 1e-8
    out = vote(scores, rows, cand, lab, gate, top_k=6)
    assert torch.isfinite(out).all()
    assert float(out[0, 0]) >= 0.0


def test_admissibility_has_no_hidden_threshold_discontinuity():
    rows = _rows()
    cand, lab = _texts()
    R = rows.feature.shape[0]
    comp = compatibility_mask(torch.tensor([0]), torch.tensor([0]), rows)
    scores = rank_scores(torch.randn(1, D), torch.randn(1, 9), rows, comp)
    just_below = torch.full((1, R, C_CAND), 0.149)
    just_above = torch.full((1, R, C_CAND), 0.151)
    below = vote(scores, rows, cand, lab, just_below, top_k=6)
    above = vote(scores, rows, cand, lab, just_above, top_k=6)
    assert torch.allclose(below, above, atol=1e-6)


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


def test_support_bound_to_another_candidate_cannot_crowd_out_a_voting_row():
    rows = _rows()
    bound = rows.enrolled_candidate.clone()
    bound[0] = 1
    labels = rows.label.clone()
    labels[6] = 0
    rows = SensorRows(
        rows.feature, rows.descriptor, rows.bias, rows.modality, rows.gravity,
        labels, rows.dataset, bound,
    )
    candidate, label_text = _texts()
    candidate = candidate.clone()
    candidate[0] = label_text[0]
    scores = torch.zeros(1, len(rows.feature))
    scores[0, 0] = 10.0       # nearest row, but explicitly bound to candidate 1
    scores[0, 6] = 5.0        # next row can vote for candidate 0 through label text
    out = vote(
        scores, rows, candidate, label_text,
        torch.ones(1, len(rows.feature), C_CAND), top_k=1,
    )
    assert float(out[0, 0]) > 0.0


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


def test_full_soft_training_vote_gives_low_ranked_rows_credit():
    rows = _rows(R=12)
    _, label_text = _texts()
    candidate = label_text[:C_CAND].clone()
    scores = torch.linspace(0.0, 0.3, 12).unsqueeze(0)
    admissibility = torch.full((1, 12, C_CAND), 0.5, requires_grad=True)
    logits = soft_vote_all(
        scores, rows, candidate, label_text, admissibility, temperature=0.2,
    )
    (-logits.clamp_min(1e-8).log()[0, 0]).backward()
    assert admissibility.grad is not None
    # Row zero would be outside top-1, but the all-row training distribution still credits it.
    assert float(admissibility.grad[0, 0].abs().sum()) > 0.0


def test_soft_training_and_topk_inference_are_identical_when_topk_keeps_every_row():
    rows = _rows(R=12)
    candidate, label_text = _texts()
    scores = torch.randn(2, 12)
    admissibility = torch.sigmoid(torch.randn(2, 12, C_CAND))
    soft = soft_vote_all(scores, rows, candidate, label_text, admissibility, temperature=0.2)
    truncated = vote(
        scores, rows, candidate, label_text, admissibility, top_k=12, temperature=0.2,
    )
    assert torch.allclose(soft, truncated, atol=1e-6)


def test_full_soft_vote_rectifies_complete_cosine():
    base = _rows(R=12)
    rows = SensorRows(
        base.feature[:2], base.descriptor[:2], base.bias[:2], base.modality[:2],
        base.gravity[:2], torch.tensor([0, 1]), base.dataset[:2],
        torch.full((2,), -1, dtype=torch.long),
    )
    label_text = F.normalize(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]), dim=-1)
    candidate_text = F.normalize(torch.tensor([[1.0, 1.0]]), dim=-1)
    out = soft_vote_all(
        torch.zeros(1, 2), rows, candidate_text, label_text,
        torch.ones(1, 2, 1), temperature=1.0,
    )
    # Both complete cosines are zero. ReLU on individual coordinate products would be positive and
    # would make the training vote disagree with deployment.
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)


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
        admissibility=torch.ones(2, R, C_CAND), top_k=6,
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
              admissibility=torch.ones(1, R, C_CAND), top_k=6)
    assert torch.allclose(predict(**kw), predict(**kw))


# --------------------------------------------------------------------------------------------
# DEPLOYMENT MIXER BRANCH — the path that lets the trained model be scored on the eval harness
# --------------------------------------------------------------------------------------------
def _mixer_rows(R=40, C=4, V=9, d=16, seed=0):
    import torch.nn.functional as F
    torch.manual_seed(seed)
    bound = torch.full((R,), -1, dtype=torch.long)
    bound[: R // 4] = torch.randint(0, C, (R // 4,))
    return SensorRows(
        feature=F.normalize(torch.randn(R, d), dim=-1),
        descriptor=F.normalize(torch.randn(R, 384), dim=-1),
        bias=torch.zeros(R, 1),
        modality=torch.zeros(R, dtype=torch.long),
        gravity=torch.zeros(R, dtype=torch.long),
        label=torch.randint(0, V, (R,)),
        dataset=torch.zeros(R, dtype=torch.long),
        enrolled_candidate=bound,
        source_window=torch.arange(R) // 4,     # 4 rows per recording, as a real bank has
    )


def test_predict_routes_through_the_engine_when_given_one():
    """The deployment entry point must run the trained forward pass, not a re-derivation of it.

    There is deliberately no "starts at the closed-form rule" assertion here any more: the engine
    replaces the hand-written retrieval rule rather than correcting it, so agreement at init would
    be a coincidence to preserve rather than a property to check. What must hold is that the engine
    is what produced the numbers.
    """
    import torch.nn.functional as F
    from model.blocks import AttentionSpec
    from model.evidence.engine import EngineConfig, EvidenceEngine

    torch.manual_seed(0)
    S, R, C, V, d = 3, 40, 4, 9, 16
    rows = _mixer_rows(R, C, V, d)
    engine = EvidenceEngine(None, EngineConfig(
        spec=AttentionSpec(d_model=d, n_heads=4), top_k=8,
    )).eval()
    query_feature = F.normalize(torch.randn(S, d), dim=-1)
    query_descriptor = F.normalize(torch.randn(S, 384), dim=-1)
    candidate_text = F.normalize(torch.randn(C, 384), dim=-1)
    label_text = F.normalize(torch.randn(V, 384), dim=-1)
    common = dict(
        query_feature=query_feature, query_bias=torch.zeros(S, 1),
        query_modality=torch.zeros(S, dtype=torch.long),
        query_gravity=torch.zeros(S, dtype=torch.long),
        rows=rows, candidate_text=candidate_text, label_text=label_text,
        admissibility=torch.ones(S, R, C), query_descriptor=query_descriptor, top_k=8,
    )
    with torch.no_grad():
        through_predict = predict(**common, engine=engine,
                                  generator=torch.Generator().manual_seed(0))
        direct = engine(
            SensorRows(feature=query_feature, descriptor=query_descriptor,
                       bias=torch.zeros(S, 1),
                       modality=torch.zeros(S, dtype=torch.long),
                       gravity=torch.zeros(S, dtype=torch.long),
                       label=torch.full((S,), -1), dataset=torch.zeros(S, dtype=torch.long),
                       enrolled_candidate=torch.full((S,), -1),
                       source_window=torch.zeros(S, dtype=torch.long)),
            rows, candidate_text, label_text, top_k=8,
            generator=torch.Generator().manual_seed(0),
        )["logits"]
    assert torch.allclose(through_predict, merge_sensors(direct, None), atol=1e-6)


def test_predict_refuses_a_bank_without_source_window_provenance():
    """Without it every row looks like its own recording, and the co-membership channel that
    relates an accelerometer to the gyroscope beside it silently says nothing."""
    import torch.nn.functional as F
    from dataclasses import replace
    from model.blocks import AttentionSpec
    from model.evidence.engine import EngineConfig, EvidenceEngine

    torch.manual_seed(0)
    S, R, C, V, d = 2, 24, 3, 7, 16
    rows = replace(_mixer_rows(R, C, V, d), source_window=None)
    engine = EvidenceEngine(None, EngineConfig(
        spec=AttentionSpec(d_model=d, n_heads=4), top_k=4,
    )).eval()
    with pytest.raises(ValueError, match="source-window provenance"):
        predict(
            query_feature=F.normalize(torch.randn(S, d), dim=-1), query_bias=torch.zeros(S, 1),
            query_modality=torch.zeros(S, dtype=torch.long),
            query_gravity=torch.zeros(S, dtype=torch.long),
            rows=rows, candidate_text=F.normalize(torch.randn(C, 384), dim=-1),
            label_text=F.normalize(torch.randn(V, 384), dim=-1),
            admissibility=torch.ones(S, R, C),
            query_descriptor=F.normalize(torch.randn(S, 384), dim=-1), engine=engine,
        )


def test_a_top_k_wider_than_the_group_vocabulary_is_refused_at_construction():
    """Wrapping would alias two distinct recordings onto one co-membership id — a wrong answer that
    looks like a working one. The check sits in the config so it fires before a run starts rather
    than on the first forward, halfway in."""
    from model.blocks import AttentionSpec
    from model.evidence.engine import EngineConfig
    from model.evidence.evidence_mixer import EvidenceMixerConfig

    with pytest.raises(ValueError, match="co-membership groups"):
        EngineConfig(spec=AttentionSpec(d_model=16, n_heads=4), top_k=16,
                     mixing="attention", mixer=EvidenceMixerConfig(n_groups=4))
