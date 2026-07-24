"""Tests for the density-gated evidential (Dirichlet) loss (model/evidence/edl.py) — Phase-B M5/D1.

The load-bearing property is **prediction identity**: the density-gated Dirichlet reparametrization
    alpha = 1 + g * beta * e          (g, beta > 0 per row)
is a strictly-increasing per-row transform of the decoder's evidence ``e``, so ``argmax_c alpha_c ==
argmax_c e_c`` and the transfer bAcc (the selection metric) is byte-for-byte unchanged from the CE
path. Also checks: the KL-to-uniform term is a genuine (>= 0) divergence, the EDL loss is finite,
gradients reach all three gate scalars, and the configured positive scale/slope are represented
exactly (not accidentally passed through another nonlinear transform).
"""

import math

import numpy as np
import pytest
import torch

from model.evidence.edl import (
    DensityGate,
    acc_at_coverage,
    aurc,
    edl_loss,
    kl_dirichlet_to_uniform,
)


def _rand_inputs(B=16, C=12, k=48, seed=0, vlo=0.2, vhi=0.6):
    g = torch.Generator().manual_seed(seed)
    e = torch.rand(B, C, generator=g)                         # non-negative per-candidate evidence
    vals = vlo + (vhi - vlo) * torch.rand(B, k, generator=g)  # raw top-k cosines (descending not required)
    target = torch.randint(0, C, (B,), generator=g)
    return e, vals, target


def test_prediction_identity_argmax_alpha_equals_argmax_e():
    """(i) argmax(alpha) == argmax(e) for every row — the do-no-harm guarantee on transfer bAcc."""
    gate = DensityGate()
    e, vals, _ = _rand_inputs(seed=1)
    alpha, gval = gate.alpha(e, vals)
    assert (gval > 0).all() and (gate.beta() > 0)              # strictly positive scaling per row
    assert torch.equal(alpha.argmax(1), e.argmax(1)), "gate broke the argmax — not do-no-harm"

    # ... and it is not an artefact of the init: perturb the gate into its sensitive region.
    gate2 = DensityGate(gate_delta=0.4, log_gamma=math.log(2.0), log_beta=math.log(5.0))
    e2, vals2, _ = _rand_inputs(seed=2, vlo=0.0, vhi=1.0)
    alpha2, g2 = gate2.alpha(e2, vals2)
    assert (g2 > 0).all()
    assert torch.equal(alpha2.argmax(1), e2.argmax(1))


def test_kl_to_uniform_is_nonnegative():
    """(ii) KL( Dir(alpha_tilde) || Dir(1) ) >= 0 (== 0 iff alpha_tilde all-ones)."""
    gate = DensityGate()
    e, vals, target = _rand_inputs(seed=3)
    alpha, _ = gate.alpha(e, vals)
    B, C = alpha.shape
    alpha_tilde = alpha.clone()
    alpha_tilde[torch.arange(B), target] = 1.0                # remove correct-class evidence
    kl = kl_dirichlet_to_uniform(alpha_tilde, C).detach()
    assert kl.shape == (B,)
    assert float(kl.min()) >= -1e-5, f"KL went negative: {float(kl.min()):.3e}"
    # all-ones alpha_tilde -> exactly zero divergence
    assert abs(float(kl_dirichlet_to_uniform(torch.ones(1, C), C))) < 1e-5


def test_edl_loss_is_finite():
    """(iii) L_edl (and its data/kl parts) is finite for random inputs across the anneal range."""
    gate = DensityGate()
    e, vals, target = _rand_inputs(seed=4)
    alpha, _ = gate.alpha(e, vals)
    for lam in (0.0, 0.1, 1.0):
        L_edl, L_data, L_kl = edl_loss(alpha, target, lam)
        assert torch.isfinite(L_edl) and torch.isfinite(L_data) and torch.isfinite(L_kl)
    assert float(L_kl.detach()) >= -1e-5


def test_gradients_flow_to_all_three_gate_scalars():
    """(iv) log_gamma, delta, log_beta all receive finite, non-zero gradient.

    Uses an UN-saturated gate configuration (delta inside the cosine range) so the sigmoid is in its
    sensitive region — at the do-no-harm init the gate is deliberately saturated (g~1) and log_gamma/
    delta gradients vanish, which is correct behaviour but would make a non-zero assertion vacuous.
    """
    gate = DensityGate(gate_delta=0.4, log_gamma=math.log(2.0), log_beta=math.log(10.0))
    e, vals, target = _rand_inputs(seed=5, vlo=0.0, vhi=1.0)
    alpha, _ = gate.alpha(e, vals)
    L_edl, _, _ = edl_loss(alpha, target, lambda_kl=0.1)
    L_edl.backward()
    for name, p in (("log_gamma", gate.log_gamma), ("delta", gate.delta), ("log_beta", gate.log_beta)):
        assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs()) > 0, \
            f"gate scalar {name} got no/zero/NaN gradient — it would never train"


def test_gate_initialization_uses_requested_scale_and_is_responsive():
    """(v) exp(log parameter) gives the requested 10 and delta=0.3 is the sigmoid midpoint."""
    gate = DensityGate()
    assert float(gate.gamma().detach()) == pytest.approx(10.0, rel=1e-6)
    assert float(gate.beta().detach()) == pytest.approx(10.0, rel=1e-6)
    lo = gate.gate(torch.full((2, 8), -0.2)).detach()
    mid = gate.gate(torch.full((2, 8), 0.3)).detach()
    hi = gate.gate(torch.full((2, 8), 0.8)).detach()
    assert torch.all(lo < 0.01)
    assert torch.allclose(mid, torch.full_like(mid, 0.5), atol=1e-6)
    assert torch.all(hi > 0.99)


def test_gate_shrinks_with_lower_density():
    """The gate must be MONOTONE in density: sparser support (lower cosines) -> smaller g (abstain)."""
    gate = DensityGate()
    lo = gate.gate(torch.full((4, 48), -0.5))
    hi = gate.gate(torch.full((4, 48), 0.5))
    assert (hi > lo).all(), "gate is not increasing in retrieval density"


def test_uncertainty_ranks_errors_metrics_sane():
    """u=C/S calibration helpers: AURC in [0,1], acc@cov in [0,1], and a monotone sanity check."""
    # construct a case where uncertainty perfectly ranks errors: high u -> wrong
    u = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    correct = np.array([True, True, True, False, False])
    a = aurc(u, correct)
    assert 0.0 <= a <= 1.0
    # keeping the 3 most-confident (cov=0.6) should be 100% correct here
    assert acc_at_coverage(u, correct, cov=0.6) == 1.0
    # abstaining on the uncertain tail cannot lower accuracy vs full coverage
    assert acc_at_coverage(u, correct, cov=0.6) >= acc_at_coverage(u, correct, cov=1.0)


def test_invalid_evidence_and_coverage_fail_loudly():
    gate = DensityGate()
    with pytest.raises(ValueError, match="non-negative"):
        gate.alpha(torch.tensor([[0.1, -0.1]]), torch.ones(1, 4))
    with pytest.raises(ValueError, match="coverage"):
        acc_at_coverage(np.array([0.1]), np.array([True]), cov=0.0)


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
