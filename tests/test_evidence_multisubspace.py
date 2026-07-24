"""Tests for D2 multi-subspace ("different ways of being similar") evidence re-weighting.

The multi-subspace head learns K low-dim projections P_m and mixes their per-evidence cosines with
a learned convex weighting; the result psi (B,k) is folded into the decoder's POOLING as an
identity-at-init gated residual  a_logit += gamma_ms * psi  (gamma_ms zero-init).

The SACRED property is identity-at-init: with gamma_ms == 0 a decoder built with n_subspaces > 0
must reproduce the n_subspaces == 0 decoder (== the untrained retrieval mechanism) EXACTLY. The
head only earns influence as gamma_ms lifts off zero. These tests pin that, plus: the residual is
live once gamma_ms != 0, gradients reach P_m / omega / gamma_ms, and psi has the right shape and
permutes with the evidence axis (it is a per-evidence score, not order-dependent).
"""

import torch
import torch.nn.functional as F

from model.evidence.decoder import DecoderConfig, EvidenceDecoder
from model.evidence.multisubspace import MultiSubspaceHead

_D2_PARAMS = {"gamma_ms", "ms_head.proj.weight", "ms_head.omega_logits"}


def _inputs(B=4, k=6, C=5, d=256, text=384, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    zq = torch.randn(B, d, generator=g)
    zev = torch.randn(B, k, d, generator=g)
    ev_label_text = F.normalize(torch.randn(B, k, text, generator=g), dim=-1)
    cand_text = F.normalize(torch.randn(C, text, generator=g), dim=-1)
    raw = torch.rand(B, k, generator=g)
    w_retr = raw / raw.sum(1, keepdim=True)
    return dict(zq=zq.to(device), zev=zev.to(device), ev_label_text=ev_label_text.to(device),
                w_retr=w_retr.to(device), cand_text=cand_text.to(device))


def test_n_subspaces_zero_builds_no_head_and_default_path_unchanged():
    """Default (n_subspaces=0): no head, no gamma_ms param, no gamma_ms in aux -> byte-identical."""
    dec = EvidenceDecoder(DecoderConfig(d_model=256, n_subspaces=0))
    assert dec.ms_head is None
    names = dict(dec.named_parameters())
    assert not (_D2_PARAMS & set(names)), "n_subspaces=0 must add NO D2 params to the state_dict"
    x = _inputs()
    with torch.no_grad():
        _, aux = dec(**x, return_aux=True)
    assert "gamma_ms" not in aux, "default path must not carry gamma_ms (keeps the reg a no-op)"


def test_identity_at_init_matches_n_subspaces_zero():
    """(i) gamma_ms=0 => n_subspaces=4 decoder == n_subspaces=0 decoder EXACTLY on the shared weights.

    Construct both, copy the shared (non-D2) state_dict into the multi-subspace decoder, and confirm
    logits / evidence / pool_weights are allclose at 1e-6 — the head adds only the zero-gated term.
    """
    torch.manual_seed(0)
    dec0 = EvidenceDecoder(DecoderConfig(d_model=256, n_subspaces=0)).eval()
    dec4 = EvidenceDecoder(DecoderConfig(d_model=256, n_subspaces=4, subspace_dim=64)).eval()
    # Copy shared weights; the D2 params are the ONLY thing dec4 has that dec0 does not.
    missing, unexpected = dec4.load_state_dict(dec0.state_dict(), strict=False)
    assert set(missing) == _D2_PARAMS, f"unexpected shared-weight mismatch: {missing}"
    assert unexpected == [], f"dec0 carried non-shared keys: {unexpected}"
    assert float(dec4.gamma_ms.detach()) == 0.0, "gamma_ms must be zero-init"

    x = _inputs(seed=7)
    with torch.no_grad():
        l0, a0 = dec0(**x, return_aux=True)
        l4, a4 = dec4(**x, return_aux=True)
    assert torch.allclose(l4, l0, atol=1e-6), f"logits differ at init: {float((l4 - l0).abs().max()):.2e}"
    assert torch.allclose(a4["evidence"], a0["evidence"], atol=1e-6)
    assert torch.allclose(a4["pool_weights"], a0["pool_weights"], atol=1e-6)
    assert float(a4["gamma_ms"]) == 0.0


def test_nonzero_gamma_ms_makes_the_residual_live():
    """(ii) Once gamma_ms != 0 the multi-subspace term actually changes the pooling and the output."""
    torch.manual_seed(1)
    dec = EvidenceDecoder(DecoderConfig(d_model=256, n_subspaces=4, subspace_dim=64)).eval()
    x = _inputs(seed=1)
    with torch.no_grad():
        base, ab = dec(**x, return_aux=True)
        dec.gamma_ms.data.fill_(0.8)
        moved, am = dec(**x, return_aux=True)
    assert not torch.allclose(base, moved, atol=1e-5), "gamma_ms != 0 left the output unchanged"
    assert not torch.allclose(ab["pool_weights"], am["pool_weights"], atol=1e-5)


def test_gamma_ms_gets_gradient_at_init():
    """(iii-a) gamma_ms is the gate-opener: it must receive gradient at EXACT init (like pool_phi),
    or the residual could never lift off zero. (P_m / omega are gated by gamma_ms=0 at init, so they
    correctly get no grad until the gate opens — checked separately below.)"""
    torch.manual_seed(2)
    dec = EvidenceDecoder(DecoderConfig(d_model=256, n_subspaces=4, subspace_dim=64)).train()
    x = _inputs(seed=2)
    F.cross_entropy(dec(**x), torch.zeros(4, dtype=torch.long)).backward()
    g = dec.gamma_ms.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs()) > 0, \
        "gamma_ms got no gradient at init — the D2 gate could never open"


def test_gradients_reach_subspaces_once_gate_open():
    """(iii-b) With gamma_ms off zero, credit must reach every learned D2 component: the K
    projections P_m (proj.weight), the omega logits, and gamma_ms itself."""
    torch.manual_seed(3)
    dec = EvidenceDecoder(DecoderConfig(d_model=256, n_subspaces=4, subspace_dim=64)).train()
    dec.gamma_ms.data.fill_(0.5)                    # open the gate so P_m / omega get credit
    x = _inputs(seed=3)
    F.cross_entropy(dec(**x), torch.zeros(4, dtype=torch.long)).backward()
    checks = {"ms_head.proj (P_m)": dec.ms_head.proj.weight.grad,
              "omega_logits": dec.ms_head.omega_logits.grad,
              "gamma_ms": dec.gamma_ms.grad}
    for name, g in checks.items():
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, \
            f"no/zero/NaN gradient for {name} — it would never train"


def test_psi_shape_and_permutation_consistency():
    """(iv) psi is (B,k), bounded (convex mix of cosines), and permutes WITH the evidence axis."""
    torch.manual_seed(4)
    B, k, d = 4, 6, 256
    head = MultiSubspaceHead(d, n_subspaces=4, subspace_dim=64)
    zq = torch.randn(B, d)
    zev = torch.randn(B, k, d)
    psi = head(zq, zev)
    assert psi.shape == (B, k)
    assert float(psi.detach().abs().max()) <= 1.0 + 1e-5, "psi must be a convex mix of cosines (in [-1,1])"
    perm = torch.randperm(k)
    psi_perm = head(zq, zev[:, perm])
    assert torch.allclose(psi[:, perm], psi_perm, atol=1e-6), \
        "psi is not permutation-consistent over the evidence axis"


def test_head_is_identity_at_init_within_decoder_param_groups():
    """The D2 params must be captured by param_groups (so the optimizer trains them): the scalar
    gamma_ms and omega_logits go no-decay, the projection weight goes to the decay group."""
    dec = EvidenceDecoder(DecoderConfig(d_model=256, n_subspaces=4, subspace_dim=64))
    groups = dec.param_groups(weight_decay=0.01)
    decay = {id(p) for p in groups[0]["params"]}
    no_decay = {id(p) for p in groups[1]["params"]}
    assert id(dec.gamma_ms) in no_decay, "gamma_ms must be no-decay"
    assert id(dec.ms_head.omega_logits) in no_decay, "omega_logits (1-d) must be no-decay"
    assert id(dec.ms_head.proj.weight) in decay, "the projection weight should get weight decay"


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
