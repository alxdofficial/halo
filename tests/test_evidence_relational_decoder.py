"""Tests for the relational evidence decoder (model/evidence/relational_decoder.py).

Three properties are load-bearing and each has a failure mode that would silently produce a
plausible-looking but meaningless model:

1. **The logit is the readout.** There is no closed-form base term any more; a candidate's score
   comes only from a token that attended over the evidence. An untrained decoder must therefore
   still *separate* candidates (a constant predictor is not a usable control arm), and gradient
   must reach every component including the retrieval-score projection — that projection is the
   only differentiable route from the loss back to the retriever.
2. **Coreference is exact and positionless.** Rows referring to the same concept must share a slot
   id, ids must be re-drawn per episode, and nothing about a slot may be stable across episodes.
   If ids leaked across episodes the model could learn "slot 3 is usually walking", which is the
   shortcut this whole design exists to remove.
3. **Padding is inert.** Evidence and label tokens are masked, not absent; if the mask were wrong,
   padded rows would leak into every attention softmax and the batch composition would change the
   answer.
"""

import pytest
import torch

from model.evidence.relational_decoder import (
    RelationalDecoderConfig,
    RelationalEvidenceDecoder,
    UNBOUND_SLOT,
    build_coreference_slots,
    build_window_groups,
    relative_evidence_attention_bias,
    retrieval_vote_base,
)


def _episode(B=3, C=4, L=3, P=2, M=7, d=32, text=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    ev_support_mask = torch.zeros(B, M, dtype=torch.bool)
    ev_support_mask[:, :2] = True                       # first two rows are declared support
    ev_support_candidate = torch.zeros(B, M, dtype=torch.long)
    ev_support_candidate[:, 0] = 0
    ev_support_candidate[:, 1] = 1
    ev_mask = torch.ones(B, M, dtype=torch.bool)
    slot_ids = torch.stack([torch.arange(1, C + L + 2) for _ in range(B)])
    slot_ids[:, -1] = UNBOUND_SLOT
    ev_slot = torch.zeros(B, M, dtype=torch.long)
    ev_slot[:, 0] = 0
    ev_slot[:, 1] = 1
    ev_slot[:, 2:] = C                                   # background rows -> first label token
    return dict(
        cand_text=torch.randn(B, C, text, generator=g),
        label_text=torch.randn(B, L, text, generator=g),
        label_mask=torch.ones(B, L, dtype=torch.bool),
        slot_ids=slot_ids,
        zq=torch.randn(B, P, d, generator=g),
        q_mask=torch.ones(B, P, dtype=torch.bool),
        zev=torch.randn(B, M, d, generator=g),
        ev_mask=ev_mask,
        ev_slot=ev_slot,
        ev_support_mask=ev_support_mask,
        ev_score=torch.rand(B, M, generator=g),
        q_time=torch.rand(B, P, generator=g),
        ev_time=torch.rand(B, M, generator=g),
        q_group=torch.ones(B, P, dtype=torch.long),
        ev_group=torch.arange(M).repeat(B, 1) % 5 + 1,
        ev_same_config=torch.zeros(B, M, dtype=torch.bool),
        ev_same_subject=torch.zeros(B, M, dtype=torch.bool),
        ev_same_sensor=torch.zeros(B, M, dtype=torch.bool),
    ), ev_support_candidate


def _decoder(d=32, text=16, seed=0):
    torch.manual_seed(seed)
    return RelationalEvidenceDecoder(
        RelationalDecoderConfig(d_model=d, text_dim=text, n_layers=2, n_heads=2)
    ).eval()


# --------------------------------------------------------------------------- #
# 1. identity at init
# --------------------------------------------------------------------------- #

def test_untrained_readout_still_separates_candidates():
    """A zero readout would emit identical logits for every candidate — a constant predictor, and
    a useless step-0 control arm. Without a base term to fall back on, the init must discriminate."""
    decoder = _decoder()
    inputs, _ = _episode()
    with torch.no_grad():
        logits, aux = decoder(**inputs, return_aux=True)
    assert logits.shape == (3, 4)
    assert aux["logit_spread"] > 1e-4
    assert torch.isfinite(logits).all()


def test_relative_retrieval_score_bias_changes_the_prediction():
    """Relative score differences control evidence influence and train the hard-path retriever."""
    decoder = _decoder()
    inputs, _ = _episode()
    uniform = torch.zeros_like(inputs["ev_score"])
    preferred = uniform.clone()
    preferred[:, 0] = 1.0
    with torch.no_grad():
        equal = decoder(**{**inputs, "ev_score": uniform})
        changed = decoder(**{**inputs, "ev_score": preferred})
    assert not torch.allclose(equal, changed, atol=1e-6)


def test_training_score_temperature_softens_the_same_selected_roster():
    decoder = _decoder()
    inputs, _ = _episode()
    with torch.no_grad():
        deployment = decoder(**inputs, score_temperature=0.07)
        exploratory = decoder(**inputs, score_temperature=0.20)
    assert not torch.allclose(deployment, exploratory, atol=1e-7)
    with pytest.raises(ValueError, match="positive"):
        decoder(**inputs, score_temperature=0.0)


def test_retrieval_bias_is_shift_invariant_and_uniform_scores_are_neutral():
    decoder = _decoder()
    inputs, _ = _episode()
    shifted = inputs["ev_score"] + 7.5
    with torch.no_grad():
        reference = decoder(**inputs)
        result = decoder(**{**inputs, "ev_score": shifted})
    torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)

    mask = torch.tensor([[True, True, False], [True, True, True]])
    score = torch.tensor([[0.8, 0.8, 100.0], [0.4, 0.4, 0.4]])
    bias = relative_evidence_attention_bias(score, mask, temperature=0.07)
    torch.testing.assert_close(bias, torch.zeros_like(bias), atol=1e-6, rtol=0.0)


def test_retrieval_bias_preserves_valid_score_order_and_has_unit_mean_prior():
    score = torch.tensor([[0.7, 0.9, 0.8, -50.0]])
    mask = torch.tensor([[True, True, True, False]])
    bias = relative_evidence_attention_bias(score, mask, temperature=0.07)
    assert bias[0, 1] > bias[0, 2] > bias[0, 0]
    torch.testing.assert_close(bias[0, mask[0]].exp().mean(), torch.tensor(1.0))
    assert bias[0, 3] == 0


def test_gradient_reaches_the_retrieval_score():
    decoder = _decoder()
    inputs, _ = _episode()
    score = inputs["ev_score"].clone().requires_grad_(True)
    decoder(**{**inputs, "ev_score": score}).sum().backward()
    assert score.grad is not None and float(score.grad.abs().sum()) > 0


def test_masked_evidence_is_ignored_however_it_is_scored():
    """The score bias shares one float mask with padding, so a bug there could unmask dead rows or
    poison the softmax with -inf * finite. Padded rows must not move the logits at all."""
    decoder = _decoder().eval()
    inputs, _ = _episode()
    mask = inputs["ev_mask"].clone()
    mask[:, -2:] = False
    score = torch.rand_like(inputs["ev_score"])
    loud_score, loud_z = score.clone(), inputs["zev"].clone()
    loud_score[:, -2:] = 50.0
    loud_z[:, -2:] = 99.0
    with torch.no_grad():
        quiet = decoder(**{**inputs, "ev_mask": mask, "ev_score": score})
        loud = decoder(**{**inputs, "ev_mask": mask, "zev": loud_z, "ev_score": loud_score})
    assert torch.equal(quiet, loud)
    assert bool(torch.isfinite(loud).all())


def test_retrieval_vote_base_matches_the_reported_identity_control():
    """`retrieval_vote_base` is no longer on the forward path, but it is still the untrained
    retrieval+text mechanism every Phase-B result is quoted against, so its formula must not drift."""
    g = torch.Generator().manual_seed(3)
    B, k, C, text = 4, 6, 5, 16
    ev_label_text = torch.randn(B, k, text, generator=g)
    cand_text = torch.randn(C, text, generator=g)
    raw = torch.rand(B, k, generator=g)
    w_retr = raw / raw.sum(1, keepdim=True)

    base = retrieval_vote_base(ev_label_text, cand_text, w_retr, scale=10.0)
    reference = 10.0 * torch.einsum("bk,bkc->bc", w_retr, torch.relu(torch.einsum(
        "bkt,ct->bkc",
        torch.nn.functional.normalize(ev_label_text, dim=-1),
        torch.nn.functional.normalize(cand_text, dim=-1),
    )))
    torch.testing.assert_close(base, reference)


# --------------------------------------------------------------------------- #
# 2. coreference
# --------------------------------------------------------------------------- #

def test_support_rows_bind_to_their_candidate_and_background_rows_to_their_label():
    ev_label = torch.tensor([[9, 9, 40, 41, 40, 7]])
    ev_mask = torch.ones(1, 6, dtype=torch.bool)
    ev_support = torch.tensor([[True, True, False, False, False, False]])
    ev_candidate = torch.tensor([[0, 2, -1, -1, -1, -1]])

    slots = build_coreference_slots(
        ev_label, ev_mask, ev_support, ev_candidate, n_candidates=3,
        n_slots=64, generator=torch.Generator().manual_seed(0),
    )
    assert slots.ev_slot[0, 0].item() == 0        # support -> candidate 0
    assert slots.ev_slot[0, 1].item() == 2        # support -> candidate 2
    # Background labels 7, 40, 41 become three label tokens; the two rows labelled 40 must agree.
    assert slots.label_ids[0].tolist() == [7, 40, 41]
    assert slots.ev_slot[0, 2].item() == slots.ev_slot[0, 4].item()
    assert slots.ev_slot[0, 3].item() != slots.ev_slot[0, 2].item()
    assert bool(slots.label_mask.all())


def test_slot_ids_are_redrawn_per_episode_but_keep_the_binding_structure():
    ev_label = torch.tensor([[5, 5, 6, 6]])
    ev_mask = torch.ones(1, 4, dtype=torch.bool)
    ev_support = torch.zeros(1, 4, dtype=torch.bool)
    ev_candidate = torch.full((1, 4), -1)

    draws = [
        build_coreference_slots(
            ev_label, ev_mask, ev_support, ev_candidate, n_candidates=2,
            n_slots=64, generator=torch.Generator().manual_seed(seed),
        ) for seed in (0, 1, 2, 3)
    ]
    # The binding is invariant: rows 0,1 always co-refer, and so do rows 2,3.
    for slots in draws:
        assert slots.ev_slot[0, 0] == slots.ev_slot[0, 1]
        assert slots.ev_slot[0, 2] == slots.ev_slot[0, 3]
        assert slots.ev_slot[0, 0] != slots.ev_slot[0, 2]
    # The identities are not: a fixed id would let the model memorise slot -> concept.
    assert len({tuple(slots.slot_ids[0].tolist()) for slots in draws}) > 1


def test_invalid_evidence_rows_point_at_the_unbound_slot():
    ev_label = torch.tensor([[5, 6, 7]])
    ev_mask = torch.tensor([[True, False, True]])
    ev_support = torch.zeros(1, 3, dtype=torch.bool)
    slots = build_coreference_slots(
        ev_label, ev_mask, ev_support, torch.full((1, 3), -1), n_candidates=2,
        n_slots=64, generator=torch.Generator().manual_seed(0),
    )
    n_labels = slots.label_ids.shape[1]
    assert slots.label_ids[0].tolist() == [5, 7]          # the masked row contributes no token
    assert slots.ev_slot[0, 1].item() == 2 + n_labels     # the unbound index
    assert slots.slot_ids[0, -1].item() == UNBOUND_SLOT


def test_support_row_without_a_candidate_position_is_rejected():
    """A -1 binding would otherwise be written straight into the slot table and index garbage."""
    with pytest.raises(ValueError, match="valid candidate position"):
        build_coreference_slots(
            torch.tensor([[5, 6]]), torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([[True, False]]), torch.tensor([[-1, -1]]),
            n_candidates=2, n_slots=64, generator=torch.Generator().manual_seed(0),
        )


def test_slot_vocabulary_overflow_is_an_error_not_a_collision():
    with pytest.raises(ValueError, match="n_slots"):
        build_coreference_slots(
            torch.arange(30).view(1, 30), torch.ones(1, 30, dtype=torch.bool),
            torch.zeros(1, 30, dtype=torch.bool), torch.full((1, 30), -1),
            n_candidates=4, n_slots=8, generator=torch.Generator().manual_seed(0),
        )


def test_window_groups_are_shared_within_a_window_and_distinct_across_them():
    q_window = torch.tensor([[100, 100]])
    ev_window = torch.tensor([[200, 200, 301, 999]])
    q_group, ev_group = build_window_groups(
        q_window, torch.ones(1, 2, dtype=torch.bool),
        ev_window, torch.tensor([[True, True, True, False]]),
        n_groups=64, generator=torch.Generator().manual_seed(0),
    )
    assert q_group[0, 0] == q_group[0, 1]
    assert ev_group[0, 0] == ev_group[0, 1]
    assert ev_group[0, 2] != ev_group[0, 0]
    assert ev_group[0, 0] != q_group[0, 0]
    assert ev_group[0, 3] == 0                            # masked row -> unbound group


# --------------------------------------------------------------------------- #
# 3. masking and structure
# --------------------------------------------------------------------------- #

def test_padded_evidence_and_labels_do_not_change_the_answer():
    decoder = _decoder()
    torch.nn.init.normal_(decoder.readout[-1].weight, std=0.1)
    inputs, _ = _episode()
    with torch.no_grad():
        reference = decoder(**inputs)

    B, M = inputs["ev_mask"].shape
    L = inputs["label_mask"].shape[1]
    C = inputs["cand_text"].shape[1]
    padded = dict(inputs)
    padded["zev"] = torch.cat([inputs["zev"], torch.randn(B, 3, inputs["zev"].shape[-1])], dim=1)
    padded["ev_mask"] = torch.cat([inputs["ev_mask"], torch.zeros(B, 3, dtype=torch.bool)], dim=1)
    padded["ev_support_mask"] = torch.cat(
        [inputs["ev_support_mask"], torch.zeros(B, 3, dtype=torch.bool)], dim=1)
    padded["label_text"] = torch.cat(
        [inputs["label_text"], torch.randn(B, 2, inputs["label_text"].shape[-1])], dim=1)
    padded["label_mask"] = torch.cat(
        [inputs["label_mask"], torch.zeros(B, 2, dtype=torch.bool)], dim=1)
    # Two more label tokens shift the unbound index and widen the slot table.
    padded["slot_ids"] = torch.cat(
        [inputs["slot_ids"][:, :C + L], torch.full((B, 2), 60), inputs["slot_ids"][:, -1:]], dim=1)
    padded["ev_slot"] = torch.cat([inputs["ev_slot"], torch.full((B, 3), C + L + 2)], dim=1)
    for field in ("ev_time", "ev_score", "ev_group", "ev_same_config", "ev_same_subject",
                  "ev_same_sensor"):
        pad = torch.zeros(B, 3, dtype=inputs[field].dtype)
        padded[field] = torch.cat([inputs[field], pad], dim=1)

    with torch.no_grad():
        widened = decoder(**padded)
    torch.testing.assert_close(widened, reference, rtol=1e-5, atol=1e-5)


def test_evidence_order_does_not_change_the_answer():
    """Evidence is a set. Any order dependence would be a positional artefact, not a finding."""
    decoder = _decoder()
    torch.nn.init.normal_(decoder.readout[-1].weight, std=0.1)
    inputs, _ = _episode()
    with torch.no_grad():
        reference = decoder(**inputs)

    order = torch.randperm(inputs["zev"].shape[1], generator=torch.Generator().manual_seed(5))
    shuffled = dict(inputs)
    for field in ("zev", "ev_mask", "ev_slot", "ev_support_mask", "ev_time", "ev_score",
                  "ev_group", "ev_same_config", "ev_same_subject", "ev_same_sensor"):
        shuffled[field] = inputs[field][:, order]
    with torch.no_grad():
        result = decoder(**shuffled)
    torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)


def test_embedding_norms_cannot_become_a_candidate_or_dataset_shortcut():
    decoder = _decoder()
    inputs, _ = _episode()
    with torch.no_grad():
        reference = decoder(**inputs)
        rescaled = decoder(**{
            **inputs,
            "cand_text": inputs["cand_text"] * 9.0,
            "label_text": inputs["label_text"] * 0.2,
            "zq": inputs["zq"] * 4.0,
            "zev": inputs["zev"] * 0.5,
        })
    torch.testing.assert_close(rescaled, reference, rtol=1e-5, atol=1e-5)


def test_additive_token_components_start_at_comparable_norms():
    """LayerNorm after a sum cannot rescue content drowned by a large structural embedding."""
    decoder = _decoder()
    g = torch.Generator().manual_seed(31)
    values = {
        "signal": decoder.proj_query(torch.randn(12, 32, generator=g)),
        "text": decoder.proj_text(torch.randn(12, 16, generator=g)),
        "role": decoder.role_emb(torch.arange(12) % 5),
        "slot": decoder.slot_emb(torch.arange(1, 13)),
        "time": decoder.proj_time(torch.randn(12, 32, generator=g)),
        "group": decoder.group_emb(torch.arange(1, 13)),
        "relation": decoder.same_config_emb(torch.arange(12) % 2),
    }
    norms = {
        name: decoder._component(name, value).norm(dim=-1)
        for name, value in values.items()
    }
    for name, value in norms.items():
        torch.testing.assert_close(value, torch.ones_like(value), atol=1e-6, rtol=1e-6,
                                   msg=lambda message: f"{name}: {message}")


def test_component_scales_are_positive_auditable_and_trainable():
    decoder = _decoder()
    inputs, _ = _episode()
    decoder.train()
    logits, aux = decoder(**inputs, return_aux=True)
    logits.square().mean().backward()
    assert set(aux["component_scales"]) == set(decoder._COMPONENT_NAMES)
    assert all(value == pytest.approx(1.0) for value in aux["component_scales"].values())
    for name, parameter in decoder.component_log_scale.items():
        assert parameter.grad is not None, f"no gradient reached component scale {name}"
        assert torch.isfinite(parameter.grad)


def test_every_learned_component_receives_gradient():
    decoder = _decoder()
    torch.nn.init.normal_(decoder.readout[-1].weight, std=0.1)
    inputs, _ = _episode()
    decoder.train()
    decoder(**inputs).sum().backward()
    missing = [name for name, parameter in decoder.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_candidate_to_evidence_attention_is_readable():
    """The explicit per-row vote is gone, so candidate->evidence attention is the only remaining
    account of which stored examples decided an answer. If it is not extractable and correctly
    aligned, "evidence engine" stops being a claim the paper can make."""
    decoder = _decoder()
    inputs, _ = _episode()
    B, M = inputs["ev_mask"].shape
    C = inputs["cand_text"].shape[1]
    with torch.no_grad():
        logits, aux = decoder(**inputs, return_attention=True)

    attention = aux["candidate_evidence_attention"]
    assert attention.shape == (B, C, M)
    assert (attention >= 0).all()
    # Every candidate's full attention row (labels + query + evidence + candidates) sums to 1, so
    # the evidence slice must be a proper sub-mass, not a renormalised or misaligned view.
    assert (attention.sum(-1) <= 1.0 + 1e-5).all()
    assert aux["candidate_label_attention"].shape == (B, C, inputs["label_mask"].shape[1])
    role_mass = sum(aux[name] for name in (
        "candidate_to_candidate_attention_mass",
        "candidate_to_label_attention_mass",
        "candidate_to_query_attention_mass",
        "candidate_to_evidence_attention_mass",
    ))
    assert role_mass == pytest.approx(1.0, abs=1e-5)
    assert 0.0 <= aux["candidate_attention_normalized_entropy"] <= 1.0 + 1e-5
    assert torch.isfinite(logits).all()


def test_attention_assigns_no_weight_to_padded_evidence():
    decoder = _decoder()
    inputs, _ = _episode()
    inputs["ev_mask"] = inputs["ev_mask"].clone()
    inputs["ev_mask"][:, -2:] = False
    with torch.no_grad():
        _, aux = decoder(**inputs, return_attention=True)
    assert float(aux["candidate_evidence_attention"][:, :, -2:].abs().max()) == 0.0


def test_mismatched_slot_table_is_rejected():
    decoder = _decoder()
    inputs, _ = _episode()
    inputs["slot_ids"] = inputs["slot_ids"][:, :-1]
    with pytest.raises(ValueError, match="slot_ids must have shape"):
        decoder(**inputs)


def test_retrieval_vote_base_ignores_zero_weight_padding():
    """The untrained control is quoted against every Phase-B result, so padded evidence rows must
    not perturb it. Rows beyond a query's roster carry zero pooling weight; whatever text sits in
    those slots has to be inert.
    """
    g = torch.Generator().manual_seed(11)
    B, k, C, text = 3, 5, 4, 16
    ev_text = torch.randn(B, k, text, generator=g)
    cand_text = torch.randn(C, text, generator=g)
    raw = torch.rand(B, k, generator=g)
    weights = raw / raw.sum(1, keepdim=True)

    base = retrieval_vote_base(ev_text, cand_text, weights, scale=10.0)

    pad = 3
    padded_text = torch.cat([ev_text, torch.randn(B, pad, text, generator=g) * 50], dim=1)
    padded_weights = torch.cat([weights, torch.zeros(B, pad)], dim=1)
    padded = retrieval_vote_base(padded_text, cand_text, padded_weights, scale=10.0)

    torch.testing.assert_close(base, padded)
