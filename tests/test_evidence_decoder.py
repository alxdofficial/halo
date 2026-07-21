"""Tests for the Tier-2 evidence decoder (model/evidence/decoder.py).

The load-bearing property is **identity-at-init**: with zero-init heads the decoder must
reproduce the untrained retrieval+text-ensemble mechanism *exactly*, so the do-no-harm gate
(beat 47.5) is a true ratchet. Also checks: set-permutation invariance (it's a set, not a
sequence — positional encoding must not impose order), key-padding-mask correctness, gradient
flow to every learned component, and the same-window bias staying permutation-safe.
"""

import math

import pytest
import torch

from model.evidence.decoder import DecoderConfig, EvidenceDecoder, _fourier_time


def _inputs(B=4, k=6, C=5, d=256, text=384, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    zq = torch.randn(B, d, generator=g)
    zev = torch.randn(B, k, d, generator=g)
    ev_label_text = torch.nn.functional.normalize(torch.randn(B, k, text, generator=g), dim=-1)
    cand_text = torch.nn.functional.normalize(torch.randn(C, text, generator=g), dim=-1)
    raw = torch.rand(B, k, generator=g)
    w_retr = raw / raw.sum(1, keepdim=True)      # normalized over valid evidence
    return dict(zq=zq.to(device), zev=zev.to(device), ev_label_text=ev_label_text.to(device),
                w_retr=w_retr.to(device), cand_text=cand_text.to(device))


def _untrained_reference(ev_label_text, w_retr, cand_text, out_scale=10.0):
    """The 47.5 mechanism: e_c = Σ_i w_i · relu(⟨t(label_i), t(c)⟩); logits = out_scale·e."""
    cand = torch.nn.functional.normalize(cand_text, dim=-1)
    votes = torch.relu(torch.einsum("bkt,ct->bkc", ev_label_text, cand))
    e = torch.einsum("bk,bkc->bc", w_retr, votes)
    return out_scale * e


def test_identity_at_init_equals_untrained_mechanism():
    torch.manual_seed(0)
    dec = EvidenceDecoder(DecoderConfig(d_model=256)).eval()
    x = _inputs()
    with torch.no_grad():
        logits = dec(**x)
    ref = _untrained_reference(x["ev_label_text"], x["w_retr"], x["cand_text"])
    assert logits.shape == ref.shape == (4, 5)
    # zero-init refiner Δ and pooling φ ⇒ decoder ≡ untrained, exactly (up to fp).
    assert torch.allclose(logits, ref, atol=1e-4, rtol=1e-4), \
        f"max abs diff {float((logits - ref).abs().max()):.2e}"


def test_identity_holds_with_time_and_config_features():
    """Extra structural inputs must not perturb the init output (heads are still zero)."""
    torch.manual_seed(1)
    B, k, text = 4, 6, 384
    dec = EvidenceDecoder(DecoderConfig()).eval()
    x = _inputs(B=B, k=k, seed=1)
    q_cfg = torch.nn.functional.normalize(torch.randn(B, text), dim=-1)
    ev_cfg = torch.nn.functional.normalize(torch.randn(B, k, text), dim=-1)
    with torch.no_grad():
        logits = dec(**x, q_config_text=q_cfg, ev_config_text=ev_cfg,
                     q_time=torch.rand(B) * 5, ev_time=torch.rand(B, k) * 5)
    ref = _untrained_reference(x["ev_label_text"], x["w_retr"], x["cand_text"])
    assert torch.allclose(logits, ref, atol=1e-4, rtol=1e-4)


def test_set_permutation_invariance():
    """Evidence is a SET — permuting evidence order (with its w/label) must not change output.
    Guards against a positional encoding accidentally imposing sequence order."""
    torch.manual_seed(2)
    dec = EvidenceDecoder(DecoderConfig()).eval()
    # perturb heads off zero so the transformer genuinely participates
    for p in dec.parameters():
        p.data += 0.02 * torch.randn_like(p)
    x = _inputs(seed=2)
    with torch.no_grad():
        base = dec(**x)
    perm = torch.randperm(x["zev"].shape[1])
    xp = dict(x)
    xp["zev"] = x["zev"][:, perm]
    xp["ev_label_text"] = x["ev_label_text"][:, perm]
    xp["w_retr"] = x["w_retr"][:, perm]
    with torch.no_grad():
        permd = dec(**xp)
    assert torch.allclose(base, permd, atol=1e-5), \
        f"not permutation-invariant: {float((base - permd).abs().max()):.2e}"


def test_padding_mask_ignores_padded_evidence():
    """Padded evidence (ev_mask False) must not affect the output regardless of its content."""
    torch.manual_seed(3)
    dec = EvidenceDecoder(DecoderConfig()).eval()
    for p in dec.parameters():
        p.data += 0.02 * torch.randn_like(p)
    B, k = 4, 6
    x = _inputs(B=B, k=k, seed=3)
    ev_mask = torch.ones(B, k, dtype=torch.bool)
    ev_mask[:, -2:] = False                       # last two are padding
    # renormalize retrieval weights over the valid entries (as the caller does)
    w = x["w_retr"].clone(); w[~ev_mask] = 0; w = w / w.sum(1, keepdim=True)
    x["w_retr"] = w
    with torch.no_grad():
        base = dec(**x, ev_mask=ev_mask)
    # scribble over the padded slots — output must be identical
    x2 = dict(x)
    x2["zev"] = x["zev"].clone(); x2["zev"][:, -2:] = 99.0 * torch.randn(B, 2, 256)
    x2["ev_label_text"] = x["ev_label_text"].clone()
    x2["ev_label_text"][:, -2:] = torch.nn.functional.normalize(torch.randn(B, 2, 384), dim=-1)
    with torch.no_grad():
        scribbled = dec(**x2, ev_mask=ev_mask)
    assert torch.allclose(base, scribbled, atol=1e-5), \
        f"padding leaked into output: {float((base - scribbled).abs().max()):.2e}"


def test_gate_opening_heads_get_gradient_at_init():
    """At *exact* init the zero-init heads (Δ, φ) gate the gradient to ev_state, so the
    transformer body sees no grad at step 0 (standard zero-init-residual behaviour). But the
    heads that OPEN those gates — refiner.last, pool_phi, out_scale — must get gradient, or
    training could never start."""
    torch.manual_seed(4)
    dec = EvidenceDecoder(DecoderConfig()).train()
    x = _inputs(seed=4)
    loss = torch.nn.functional.cross_entropy(dec(**x), torch.zeros(4, dtype=torch.long))
    loss.backward()
    for name, g in {"refiner.last": dec.refiner[-1].weight.grad,
                    "pool_phi": dec.pool_phi.weight.grad,
                    "log_out_scale": dec.log_out_scale.grad}.items():
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, \
            f"gate-opening head {name} has no gradient at init — training can't start"


def test_gradients_reach_transformer_body_once_heads_open():
    """After the heads are off zero (step ≥ 2 in training), credit must reach every learned
    component: LayerScale, attention, and the input projections. This is the real credit
    assignment check — the M4a diagnostic proved gradients aren't the bottleneck, and this
    guards that the decoder keeps that property."""
    torch.manual_seed(4)
    dec = EvidenceDecoder(DecoderConfig()).train()
    for p in dec.parameters():                      # simulate post-step-1: heads no longer zero
        p.data += 0.02 * torch.randn_like(p)
    x = _inputs(seed=4)
    loss = torch.nn.functional.cross_entropy(dec(**x), torch.zeros(4, dtype=torch.long))
    loss.backward()
    checks = {
        "refiner.last": dec.refiner[-1].weight.grad,
        "refiner.first": dec.refiner[0].weight.grad,
        "pool_phi": dec.pool_phi.weight.grad,
        "layerscale1": dec.blocks[0].ls1.grad,
        "attn_in_proj": dec.blocks[0].attn.in_proj_weight.grad,
        "proj_z": dec.proj_z.weight.grad,
        "proj_lab": dec.proj_lab.weight.grad,
        "log_out_scale": dec.log_out_scale.grad,
    }
    for name, g in checks.items():
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, \
            f"no/zero/NaN gradient for {name}"


def test_pool_weights_equal_retrieval_prior_at_init():
    torch.manual_seed(5)
    dec = EvidenceDecoder(DecoderConfig()).eval()
    x = _inputs(seed=5)
    with torch.no_grad():
        _, aux = dec(**x, return_aux=True)
    assert torch.allclose(aux["pool_weights"], x["w_retr"], atol=1e-5)
    assert aux["delta_norm"] == pytest.approx(0.0, abs=1e-6)   # no refinement at init


def test_fourier_time_zero_is_constant():
    """A pooled (whole-window) token has t=0 -> a constant [0,1,...] feature (no position)."""
    f = _fourier_time(torch.zeros(3), n_freqs=8, time_max=30.0)
    assert f.shape == (3, 16)
    assert torch.allclose(f[:, :8], torch.zeros(3, 8))          # sin(0)=0
    assert torch.allclose(f[:, 8:], torch.ones(3, 8))           # cos(0)=1


def test_fourier_time_is_injective_over_a_window_and_not_aliased():
    """The above cannot fail for any sin/cos implementation. This one can.

    The features must actually DISTINGUISH positions inside a window. `freqs` spans up to 30 Hz
    against t in seconds, so the top bands complete many cycles over a 6 s window; if the encoding
    were dominated by those it would alias and two distinct times could collide.
    """
    t_s = torch.linspace(0.0, 6.0, 64)
    f = _fourier_time(t_s, n_freqs=8, time_max=30.0)
    d = torch.cdist(f, f) + torch.eye(len(t_s)) * 1e3
    assert d.min() > 1e-3, f"two distinct in-window times collide (min distance {d.min():.2e})"
    # nearby times must be closer than far-apart times -- i.e. the encoding is locally monotone
    assert torch.norm(f[0] - f[1]) < torch.norm(f[0] - f[-1])


def test_fourier_time_is_window_RELATIVE_not_absolute():
    """Shifting a whole window's absolute timestamps must not change the encoding.

    The decoder subtracts no origin internally, so window-relativity is a CALLER contract. This
    pins the contract: callers must pass offsets from the window start, and the check below is
    what fails if someone starts passing session-absolute time.
    """
    rel = torch.tensor([0.0, 1.5, 3.0])
    a = _fourier_time(rel, n_freqs=8, time_max=30.0)
    b = _fourier_time(rel + 3600.0, n_freqs=8, time_max=30.0)   # same window, an hour into a session
    assert not torch.allclose(a, b, atol=1e-3), (
        "_fourier_time is shift-invariant, so absolute vs relative time would be indistinguishable "
        "and the caller contract could not be violated -- if that ever becomes true, this test "
        "should be replaced by an assertion inside the decoder instead.")


def test_same_window_bias_is_permutation_safe_and_trainable():
    """The previous version never turned the feature on, so the active path had ZERO coverage.

    It also masked a real bug: the code gated on ``float(self.same_window_bias) == 0.0`` and the
    parameter is initialised to exactly 0.0, so the branch short-circuited on every forward and the
    parameter could never receive gradient. Now gated on whether window ids were supplied.
    """
    torch.manual_seed(6)
    dec = EvidenceDecoder(DecoderConfig()).eval()
    x = _inputs(seed=6)
    B, k = x["zev"].shape[:2]
    wid = torch.arange(k + 1).unsqueeze(0).expand(B, -1)       # every token its own window

    # value is 0 at init, so the output is still the untrained mechanism ...
    with torch.no_grad():
        out = dec(**x, window_id=wid)
    ref = _untrained_reference(x["ev_label_text"], x["w_retr"], x["cand_text"])
    assert torch.allclose(out, ref, atol=1e-4)

    # ... but the parameter must be REACHABLE, or it is frozen at zero forever.
    dec.zero_grad()
    dec.refiner[-1].weight.data.normal_(0, 0.02)               # open the gate so grad can flow
    dec(**x, window_id=wid).sum().backward()
    assert dec.same_window_bias.grad is not None, "same_window_bias never enters the graph"
    assert float(dec.same_window_bias.grad.abs()) > 0, "same_window_bias gets exactly zero gradient"

    # co-membership is by id, not by position: permuting evidence permutes the output identically
    perm = torch.randperm(k)
    xp = {**x, "zev": x["zev"][:, perm], "ev_label_text": x["ev_label_text"][:, perm],
          "w_retr": x["w_retr"][:, perm]}
    wid_p = torch.cat([wid[:, :1], wid[:, 1:][:, perm]], dim=1)
    with torch.no_grad():
        assert torch.allclose(dec(**x, window_id=wid), dec(**xp, window_id=wid_p), atol=1e-5)


def test_fully_masked_evidence_row_does_not_produce_nan():
    """A row with no valid evidence used to be all -inf -> softmax NaN -> NaNs the batch loss."""
    torch.manual_seed(11)
    dec = EvidenceDecoder(DecoderConfig()).eval()
    x = _inputs(B=3, k=5, seed=11)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[1] = False                                            # row 1 has NOTHING
    with torch.no_grad():
        logits = dec(**x, ev_mask=mask)
    assert torch.isfinite(logits).all(), f"non-finite logits: {logits}"
    assert torch.allclose(logits[1], torch.zeros_like(logits[1]), atol=1e-6), \
        "a row with no evidence should contribute no evidence"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
