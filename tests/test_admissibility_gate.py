"""Gate tests for the learned admissibility gate.

These assert the properties the design DEPENDS on, not merely that shapes line up:

  * rank 1 remains a valid ablation: signed sensor and concept latents can express an inversion;
  * a rank-2 gate CAN reproduce an inversion, which is the whole reason for the bilinear form;
  * an unfamiliar configuration is blended back toward neutral rather than extrapolated confidently
    — the one thing the old lookup did better, and the reason abstention is not optional;
  * the warm start actually freezes the projections, so it fits 17 parameters and not 3000;
  * collapse is observable, because "the gate learned nothing" must be a reported result.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from training.evidence.admissibility_gate import (
    DEFAULT_RANK,
    AdmissibilityGate, DEFAULT_NOVELTY_FLOOR, TableObservations, _pca_projection, fit_gate,
    latent_measurement_correlation, observations_from_table, predict_cells,
)

TEXT = 384
SEED = 20260812


def _emb(n: int, seed: int = SEED) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, TEXT, generator=g), dim=-1)


# ------------------------------------------------------------------------------ rank
def test_default_rank_is_eight():
    assert DEFAULT_RANK == 8
    assert AdmissibilityGate().rank == 8


def test_only_nonpositive_rank_is_refused():
    try:
        AdmissibilityGate(rank=0)
    except ValueError:
        pass
    else:
        raise AssertionError("rank zero must be refused")
    assert AdmissibilityGate(rank=1).rank == 1


def test_rank_two_reproduces_an_inversion():
    """THE PROPERTY THE BILINEAR FORM EXISTS FOR.

    Two configurations, two concepts, perfectly inverted: config A resolves concept 0 and not 1,
    config B the reverse. An additive score f(config)+g(concept) gives each config one number and must
    rank the concepts identically for both, so it cannot fit this at all. A rank-2 product can.
    """
    sensors = _emb(2, seed=1)
    concepts = _emb(2, seed=2)
    s_rows = torch.stack([sensors[0], sensors[0], sensors[1], sensors[1]])
    c_rows = torch.stack([concepts[0], concepts[1], concepts[0], concepts[1]])
    target = torch.tensor([0.9, 0.1, 0.1, 0.9])

    gate = fit_gate(s_rows, c_rows, target, rank=2, mode="full", steps=1500, weight_decay=0.0)
    pred = predict_cells(gate, s_rows, c_rows, abstain=False)
    assert pred[0] > pred[1], "config A must prefer concept 0"
    assert pred[3] > pred[2], "config B must prefer concept 1 — the inversion"


# ------------------------------------------------------------------------- forward shape
def test_forward_is_a_bounded_row_by_candidate_matrix():
    gate = AdmissibilityGate(rank=4)
    w = gate(_emb(7, seed=3), _emb(5, seed=4))
    assert w.shape == (7, 5)
    w = w.detach()
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0


def test_identity_coupling_is_a_dot_product_between_the_latents():
    """With LAMBDA = I the gate reads 'is the sensor where the motion is'."""
    gate = AdmissibilityGate(rank=4)
    with torch.no_grad():
        gate.coupling.copy_(torch.eye(4))
        gate.offset.zero_()
    s, c = _emb(3, seed=5), _emb(2, seed=6)
    u, v = gate.latents(s, c)
    assert torch.allclose(gate(s, c), torch.sigmoid(u @ v.t()), atol=1e-6)


# --------------------------------------------------------------------------- warm start
def test_pca_warm_start_freezes_the_projections():
    """The warm start must fit r*r + 1 parameters, not 3000 — that is why LAMBDA is a separate object."""
    s, c = _emb(20, seed=7), _emb(20, seed=8)
    target = torch.rand(20)
    before = _pca_projection(torch.unique(s, dim=0), 4).clone()
    gate = fit_gate(s, c, target, rank=4, mode="pca", steps=50)
    assert not gate.sensor_proj.weight.requires_grad
    assert not gate.concept_proj.weight.requires_grad
    assert torch.allclose(gate.sensor_proj.weight, before, atol=1e-6), "PCA rows must not move"
    trainable = sum(p.numel() for p in gate.parameters() if p.requires_grad)
    assert trainable == 4 * 4 + 1, trainable


def test_full_mode_frees_the_projections():
    s, c = _emb(20, seed=9), _emb(20, seed=10)
    gate = fit_gate(s, c, torch.rand(20), rank=4, mode="full", steps=50)
    assert gate.sensor_proj.weight.requires_grad
    assert sum(p.numel() for p in gate.parameters() if p.requires_grad) > 3000


def test_pca_projection_is_deterministic_and_orthonormal():
    x = _emb(30, seed=11)
    a, b = _pca_projection(x, 4), _pca_projection(x, 4)
    assert torch.allclose(a, b)
    assert torch.allclose(a @ a.t(), torch.eye(4), atol=1e-5)


def test_pca_projection_rejects_a_rank_it_cannot_support():
    try:
        _pca_projection(_emb(3, seed=12), 8)
    except ValueError:
        return
    raise AssertionError("rank above the sample count must fail loudly, not silently truncate")


def test_neutral_is_the_measured_median_not_a_hard_half():
    """0.5 would treat an unmeasured pair as MORE resolvable than a typical measured one."""
    s, c = _emb(40, seed=13), _emb(40, seed=14)
    target = torch.cat([torch.full((30,), 0.2), torch.full((10,), 0.9)])
    gate = fit_gate(s, c, target, rank=2, mode="pca", steps=10)
    assert abs(float(gate.neutral) - float(target.median())) < 1e-6
    assert float(gate.neutral) < 0.5


# --------------------------------------------------------------------------- abstention
def test_a_familiar_configuration_is_not_damped():
    s, c = _emb(12, seed=15), _emb(12, seed=16)
    gate = fit_gate(s, c, torch.rand(12), rank=2, mode="pca", steps=10)
    with torch.no_grad():
        raw = torch.sigmoid(gate.sensor_proj(s) @ gate.coupling @ gate.concept_proj(c).t()
                            + gate.offset)
    assert torch.allclose(gate(s, c), raw, atol=1e-5), "a seen sensor must pass through unchanged"


def test_an_unfamiliar_configuration_is_pulled_toward_neutral():
    """A linear map extrapolates confidently into regions it has never seen; that must be damped."""
    s, c = _emb(12, seed=17), _emb(12, seed=18)
    gate = fit_gate(s, c, torch.rand(12), rank=2, mode="pca", steps=10)
    gate.sensor_novelty_floor.fill_(0.99)
    c = c[:3]
    # Orthogonalise a probe against every known sensor so its max cosine is ~0.
    stranger = torch.randn(1, TEXT, generator=torch.Generator().manual_seed(19))
    known = gate.known_sensors
    stranger = stranger - (stranger @ known.t()) @ known / max(float(known.shape[0]), 1.0)
    stranger = F.normalize(stranger, dim=-1)
    familiarity = float((stranger @ F.normalize(known, dim=-1).t()).max())
    assert familiarity < DEFAULT_NOVELTY_FLOOR, familiarity
    out = gate(stranger, c)
    assert torch.allclose(out, torch.full_like(out, float(gate.neutral)), atol=1e-5)


def test_stage_two_support_extension_makes_new_training_text_familiar():
    s, c = _emb(12, seed=22), _emb(12, seed=23)
    gate = fit_gate(s, c, torch.rand(12), rank=2, mode="pca", steps=10)
    new_sensor = _emb(1, seed=24)
    new_concept = _emb(1, seed=25)
    before_sensors = len(gate.known_sensors)
    before_concepts = len(gate.known_concepts)
    gate.extend_known_support(new_sensor, new_concept)
    assert len(gate.known_sensors) == before_sensors + 1
    assert len(gate.known_concepts) == before_concepts + 1
    familiarity = gate._familiarity(
        new_sensor, gate.known_sensors, gate.sensor_novelty_floor
    )
    assert float(familiarity) > 0.99


# ---------------------------------------------------------------------------- telemetry
def test_collapse_is_visible_in_the_spread():
    """A gate that flattened to a constant is a RESULT and must not be silently absorbed."""
    flat = AdmissibilityGate.spread(torch.full((4, 6), 0.42))
    assert flat["gate/std"] < 1e-6, flat
    varied = AdmissibilityGate.spread(torch.linspace(0, 1, 24).reshape(4, 6))
    assert varied["gate/std"] > 0.1
    assert varied["gate/log_penalty_mean"] > 0.0


# --------------------------------------------------------------------- table plumbing
def test_observations_skip_streams_with_no_sensor_description():
    """Never guess a description: a stream we cannot describe is dropped, not invented."""
    table = {"schema_version": 2, "per_sensor": {
        "a/x::accel": {"labels": {"walking": 0.8, "sitting": 0.2}},
        "b/y::accel": {"labels": {"walking": 0.5}},
    }}
    obs = observations_from_table(
        table, {"a/x::accel": "a watch accelerometer on the wrist"}
    )
    assert len(obs) == 2
    assert set(obs.stream_key) == {"a/x::accel"}


def test_within_concept_correlation_ignores_per_concept_difficulty():
    """The load-bearing statistic: it removes 'some activities are easy' and keeps only the
    configuration-dependent part, which is the only part the contribution claims."""
    obs = TableObservations(
        sensor_text=["s0", "s1", "s0", "s1"],
        concept_text=["c0", "c0", "c1", "c1"],
        value=np.array([0.9, 0.1, 0.1, 0.9], dtype=np.float32),
        stream_key=["a/0", "a/1", "a/0", "a/1"],
        concept=["c0", "c0", "c1", "c1"],
    )
    sensors, concepts = _emb(2, seed=20), _emb(2, seed=21)
    s_rows = torch.stack([sensors[0], sensors[1], sensors[0], sensors[1]])
    c_rows = torch.stack([concepts[0], concepts[0], concepts[1], concepts[1]])
    gate = fit_gate(s_rows, c_rows, torch.from_numpy(obs.value), rank=2, mode="full",
                    steps=1500, weight_decay=0.0)
    report = latent_measurement_correlation(gate, obs, s_rows, c_rows)
    assert report["n_concepts_with_multiple_configs"] == 2
    assert report["within_concept_pearson_r"] > 0.5, report
