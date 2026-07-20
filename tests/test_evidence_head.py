"""Unit tests for the Phase-B retrieval evidence head (model/evidence/head.py)."""

from __future__ import annotations

import torch

from model.evidence.head import EvidenceHead


def _toy(B=4, N=20, L=5, d=16, proj=8, seed=0):
    torch.manual_seed(seed)
    head = EvidenceHead(d_model=d, proj=proj)
    z_q = torch.randn(B, d, requires_grad=True)
    Z = torch.randn(N, d)
    mem_y = torch.randint(0, L, (N,))
    label_text = torch.randn(L, 384)
    return head, z_q, Z, mem_y, label_text


def test_evidence_shapes_and_nonneg():
    head, z_q, Z, mem_y, label_text = _toy()
    g_mem = head.project_query(Z)
    t_lab = head.project_text(label_text)
    gq = head.project_query(z_q)
    mask = torch.ones(z_q.shape[0], Z.shape[0], dtype=torch.bool)
    e = head.evidence(gq, g_mem, mem_y, cand_proj=t_lab, t_labels=t_lab, retrieval_mask=mask)
    assert e.shape == (z_q.shape[0], t_lab.shape[0])
    assert torch.isfinite(e).all()
    assert (e >= 0).all(), "evidence must be non-negative (relu kernel · softmax weights)"


def test_masked_softmax_no_nan_and_finite_grad():
    """The tau-gradient NaN regression: masking must not route -inf through /tau."""
    head, z_q, Z, mem_y, label_text = _toy()
    g_mem = head.project_query(Z)
    t_lab = head.project_text(label_text)
    gq = head.project_query(z_q)
    # mask out MOST neighbors for every query (leave a few) — the stress case
    mask = torch.zeros(z_q.shape[0], Z.shape[0], dtype=torch.bool)
    mask[:, :3] = True
    e = head.evidence(gq, g_mem, mem_y, cand_proj=t_lab, t_labels=t_lab, retrieval_mask=mask)
    head.logits(e).sum().backward()
    assert torch.isfinite(head.log_tau.grad).all(), "tau grad NaN (masked -inf through /tau)"
    assert torch.isfinite(z_q.grad).all()
    for p in head.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()


def test_retrieval_weights_respect_mask():
    head, z_q, Z, mem_y, label_text = _toy()
    g_mem = head.project_query(Z)
    t_lab = head.project_text(label_text)
    gq = head.project_query(z_q)
    mask = torch.ones(z_q.shape[0], Z.shape[0], dtype=torch.bool)
    mask[:, 5:] = False
    _, w, _ = head.evidence(gq, g_mem, mem_y, cand_proj=t_lab, t_labels=t_lab,
                            retrieval_mask=mask, return_weights=True)
    assert torch.allclose(w[:, 5:], torch.zeros_like(w[:, 5:])), "masked neighbors got weight"
    assert torch.allclose(w.sum(1), torch.ones(w.shape[0]), atol=1e-5)


def test_label_paraphrase_pool_is_diverse_not_noop():
    """Regression: augment_label('',...) fell into a generic fallback (~4 template wraps, no
    synonyms) making label-aug a near no-op. The merged pool must carry real lexical diversity."""
    from training.evidence.train_head import _global_label_paraphrases
    syn, templates = _global_label_paraphrases()
    assert len(templates) >= 5
    assert len(syn) >= 20, "per-dataset synonym tables not merged"
    assert "walking" in syn and len(syn["walking"]) >= 3, "label-aug no-op regression"


def test_tau_clamped_positive():
    head, *_ = _toy()
    with torch.no_grad():
        head.log_tau.fill_(-100.0)
    assert head.tau.item() >= 1e-3
    with torch.no_grad():
        head.log_tau.fill_(100.0)
    assert head.tau.item() <= 1.0
