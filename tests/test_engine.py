"""Structural guarantees of the compact evidence engine.

These are not accuracy tests. Each one pins a property the design depends on and that a plausible
refactor could silently remove: that sets are read as sets, that a sensor is encoded in isolation,
that retrieval starts where the hand-written rule ended, that no candidate has parameters of its
own, and that every learnable part is actually reachable by a gradient.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from model.blocks import AttentionSpec, ScaledSum, SetAttentionStack
from model.evidence.engine import EngineConfig, EvidenceEngine, vote
from model.evidence.evidence_mixer import EvidenceMixerConfig
from model.evidence.retrieval_scorer import PairScorer
from model.evidence.rows import SensorRows
from model.tokenizer.transformer import TemporalTrunk
from training.tokenizer.episodic import (
    BankSpec, bank_composition, bank_index, build_bank_plan,
)

TEXT_DIM = 384


def _rows(n, *, d=64, enrolled=None, vocab=40, seed=0, windows_per_row=2):
    generator = torch.Generator().manual_seed(seed)
    return SensorRows(
        feature=torch.randn(n, d, generator=generator),
        descriptor=F.normalize(torch.randn(n, TEXT_DIM, generator=generator), dim=-1),
        bias=torch.zeros(n, 14),
        modality=torch.randint(0, 2, (n,), generator=generator),
        gravity=torch.randint(0, 2, (n,), generator=generator),
        label=torch.randint(0, vocab, (n,), generator=generator),
        dataset=torch.zeros(n, dtype=torch.long),
        enrolled_candidate=torch.full((n,), -1) if enrolled is None else enrolled,
        source_window=torch.arange(n) // windows_per_row,
    )


def _engine(*, d=64, top_k=8, readout="weights"):
    spec = AttentionSpec(d_model=d, n_heads=4, ffn_mult=2, dropout=0.0)
    cfg = EngineConfig(spec=spec, top_k=top_k,
                       mixer=EvidenceMixerConfig(readout=readout))
    return EvidenceEngine(None, cfg).eval()


def _episode(n_memory=64, n_query=6, n_cand=5, d=64, vocab=40, seed=0):
    generator = torch.Generator().manual_seed(seed + 100)
    enrolled = torch.full((n_memory,), -1)
    enrolled[:4] = torch.tensor([0, 0, 1, 2])
    query = _rows(n_query, d=d, vocab=vocab, seed=seed)
    memory = _rows(n_memory, d=d, enrolled=enrolled, vocab=vocab, seed=seed + 1)
    label_text = F.normalize(torch.randn(vocab, TEXT_DIM, generator=generator), dim=-1)
    candidate_text = F.normalize(torch.randn(n_cand, TEXT_DIM, generator=generator), dim=-1)
    return query, memory, candidate_text, label_text


# ------------------------------------------------------------------ the shared attention spec
def test_parameters_per_block_matches_a_real_block():
    """The budget formula is used to size the model before it exists; it must not drift."""
    spec = AttentionSpec(d_model=128, n_heads=4, ffn_mult=2, dropout=0.0)
    stack = SetAttentionStack(spec, n_layers=1)
    measured = sum(p.numel() for p in stack.parameters()) - 2 * spec.d_model   # minus out_norm
    assert measured == spec.parameters_per_block()


def test_attention_spec_rejects_indivisible_width():
    with pytest.raises(ValueError):
        AttentionSpec(d_model=100, n_heads=8)


def test_set_attention_is_permutation_equivariant():
    """Order in these sequences is an assembly artifact. Reading it would let the model use a
    positional cue that carries no information and differs between training and deployment."""
    spec = AttentionSpec(d_model=32, n_heads=4, ffn_mult=2, dropout=0.0)
    stack = SetAttentionStack(spec, n_layers=2).eval()
    x = torch.randn(2, 9, 32)
    order = torch.randperm(9)
    direct = stack(x)[:, order]
    permuted = stack(x[:, order])
    assert torch.allclose(direct, permuted, atol=1e-5)


def test_padded_tokens_cannot_influence_real_ones():
    spec = AttentionSpec(d_model=32, n_heads=4, ffn_mult=2, dropout=0.0)
    stack = SetAttentionStack(spec, n_layers=2).eval()
    x = torch.randn(1, 6, 32)
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, 4:] = False
    baseline = stack(x, key_padding_mask=mask)
    disturbed = x.clone()
    disturbed[0, 4:] += 10.0
    assert torch.allclose(baseline[0, :4], stack(disturbed, key_padding_mask=mask)[0, :4], atol=1e-5)
    assert torch.isfinite(baseline).all()


def test_scaled_sum_is_blind_to_the_magnitude_of_its_ingredients():
    """Adding a raw embedding (norm about sqrt(d)) to a projection of a unit-norm text vector lets
    the embedding drown the content. Every additive channel must enter as a direction."""
    mixed = ScaledSum(3)
    terms = [torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)]
    rescaled = [term * scale for term, scale in zip(terms, (1.0, 20.0, 0.05))]
    assert torch.allclose(mixed(*terms), mixed(*rescaled), atol=1e-6)
    assert torch.allclose(mixed.log_gain, torch.zeros(3))       # every channel starts at gain 1
    plain_sum = sum(F.normalize(term, dim=-1) for term in terms)
    assert torch.allclose(mixed(*terms), plain_sum, atol=1e-6)


# ------------------------------------------------------------------ the trunk
def test_temporal_trunk_encodes_each_sensor_in_isolation():
    """The cross-configuration claim needs a wrist accelerometer to encode identically whether or
    not a gyroscope happened to sit beside it."""
    spec = AttentionSpec(d_model=32, n_heads=4, ffn_mult=2, dropout=0.0)
    trunk = TemporalTrunk(spec, num_layers=2).eval()
    x = torch.randn(2, 5, 3, 32)
    positions = torch.arange(5.0).view(1, 5).expand(2, 5)
    baseline = trunk(x, positions=positions)
    disturbed = x.clone()
    disturbed[:, :, 2] += 5.0
    assert torch.allclose(baseline[:, :, 0], trunk(disturbed, positions=positions)[:, :, 0],
                          atol=1e-5)


# ------------------------------------------------------------------ retrieval
def test_pair_scorer_starts_at_the_retired_cosine_rule():
    """Any later difference must be attributable to training, not to a changed starting point."""
    spec = AttentionSpec(d_model=32, n_heads=4)
    scorer = PairScorer(spec).eval()
    qf, mf = torch.randn(7, 32), torch.randn(19, 32)
    qd = F.normalize(torch.randn(7, TEXT_DIM), dim=-1)
    md = F.normalize(torch.randn(19, TEXT_DIM), dim=-1)
    cosine = F.normalize(qf, dim=-1) @ F.normalize(mf, dim=-1).t() / 0.07
    assert (scorer(qf, qd, mf, md) - cosine).abs().max() < 0.05


def test_pair_scorer_reads_all_four_of_its_arguments():
    """A scorer that ignored the descriptions could not learn the physics the hard filter used to
    stipulate, and the removal of that filter would be a straight loss."""
    spec = AttentionSpec(d_model=32, n_heads=4)
    scorer = PairScorer(spec).eval()
    with torch.no_grad():                       # let the learned head matter, not just its init
        scorer.residual_gain.fill_(1.0)
    args = [torch.randn(5, 32), F.normalize(torch.randn(5, TEXT_DIM), dim=-1),
            torch.randn(9, 32), F.normalize(torch.randn(9, TEXT_DIM), dim=-1)]
    baseline = scorer(*args)
    for position in range(4):
        perturbed = list(args)
        perturbed[position] = perturbed[position] + 0.5
        assert not torch.allclose(baseline, scorer(*perturbed), atol=1e-4), position


def test_selection_is_the_top_k_of_the_score():
    scores = torch.randn(4, 30)
    picked = PairScorer.select(scores, 5)
    assert picked.shape == (4, 5)
    assert torch.allclose(scores.gather(1, picked).sort(descending=True).values,
                          scores.topk(5, dim=1).values)


# ------------------------------------------------------------------ the vote
def test_an_enrolled_row_is_invisible_to_every_candidate_but_its_own():
    """Not merely down-weighted — absent. It takes no softmax mass from candidate 1 at all, which
    is checked by deleting it and requiring candidate 1's score to be unchanged."""
    torch.manual_seed(0)
    candidate_text = F.normalize(torch.randn(2, TEXT_DIM), dim=-1)
    label_vector = F.normalize(torch.randn(1, 3, TEXT_DIM), dim=-1)
    log_weight = torch.randn(1, 3, 2)
    bound = torch.tensor([[0, -1, -1]])

    with_enrolled = vote(log_weight, label_vector, candidate_text, bound)
    without = vote(log_weight[:, 1:], label_vector[:, 1:], candidate_text,
                   torch.tensor([[-1, -1]]))
    assert with_enrolled.shape == (1, 2)
    assert pytest.approx(float(without[0, 1]), abs=1e-6) == float(with_enrolled[0, 1])
    assert not float(without[0, 0]) == pytest.approx(float(with_enrolled[0, 0]), abs=1e-6)


def test_an_enrolled_row_votes_exactly_one_for_the_candidate_it_is_bound_to():
    """The identity special case was deleted from `vote`. This is why that was safe: an enrolled
    row's label vector IS its candidate's text, so the ordinary cosine path returns exactly 1.

    It also has to be true, because a constant is unreachable by the "semantic" readout — under
    that arm, support strength would otherwise have no learned control at all.
    """
    torch.manual_seed(0)
    from model.evidence.rows import evidence_label_tokens

    candidate_text = F.normalize(torch.randn(4, TEXT_DIM), dim=-1)
    label_text = F.normalize(torch.randn(40, TEXT_DIM), dim=-1)     # matches _episode's vocab
    _, memory, _, _ = _episode(n_memory=16)
    row_text = evidence_label_tokens(memory, candidate_text, label_text)
    enrolled = memory.enrolled_candidate >= 0
    assert bool(enrolled.any())
    cosine = (row_text[enrolled] * candidate_text[memory.enrolled_candidate[enrolled]]).sum(-1)
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-6)


def test_a_candidate_with_no_admissible_evidence_scores_zero_not_nan():
    log_weight = torch.zeros(1, 2, 3)
    label_vector = F.normalize(torch.ones(1, 2, TEXT_DIM), dim=-1)
    candidate_text = F.normalize(torch.ones(3, TEXT_DIM), dim=-1)
    bound = torch.tensor([[0, 1]])              # every row is bound; candidate 2 has nothing
    logits = vote(log_weight, label_vector, candidate_text, bound)
    assert torch.isfinite(logits).all()
    assert float(logits[0, 2]) == 0.0


def test_one_enrolled_row_per_candidate_keeps_retrieval_learnable():
    """Without the null row each candidate's only allowed row gets softmax mass exactly one.

    The output then ties at one and retrieval receives zero gradient, which is the common k=1
    enrollment case rather than an edge case.
    """
    candidate_text = F.normalize(torch.randn(3, TEXT_DIM), dim=-1)
    label_vector = candidate_text.unsqueeze(0)
    log_weight = torch.tensor([[[0.2, -1.0, -1.0],
                                [-1.0, 0.4, -1.0],
                                [-1.0, -1.0, 0.8]]], requires_grad=True)
    bound = torch.tensor([[0, 1, 2]])
    logits = vote(log_weight, label_vector, candidate_text, bound)
    F.cross_entropy(logits, torch.tensor([2])).backward()
    assert logits.unique().numel() > 1
    assert log_weight.grad is not None and float(log_weight.grad.norm()) > 0.0


# ------------------------------------------------------------------ the engine
@pytest.mark.parametrize("readout", ["weights", "semantic"])
def test_every_learnable_parameter_receives_gradient(readout):
    """A parameter with no path to the loss is capacity that inflates the model's stated size and
    can never do anything. This suite has caught exactly that twice."""
    engine = _engine(readout=readout).train()
    query, memory, candidate_text, label_text = _episode()
    out = engine(query, memory, candidate_text, label_text)
    F.cross_entropy(out["logits"] / 0.1, torch.zeros(len(out["logits"]), dtype=torch.long)).backward()
    missing = [name for name, p in engine.named_parameters()
               if p.requires_grad and (p.grad is None or not float(p.grad.abs().sum()))]
    assert missing == []


@pytest.mark.parametrize("readout", ["weights", "semantic"])
def test_candidates_are_scored_by_their_text_not_their_index(readout):
    """The property that makes an unseen label scorable.

    Permuting the candidate set — their names, their coreference slots, and the bindings of the
    rows enrolled to them — permutes the mixer's output and changes nothing else. A per-candidate
    parameter anywhere, or any read of candidate POSITION, would break this. It is checked on the
    mixer rather than through the engine because the engine draws slot ids from candidate index,
    so an engine-level permutation deliberately changes the slot assignment too.
    """
    torch.manual_seed(0)
    engine = _engine(readout=readout)
    mixer = engine.mixer
    n_query, n_ev, n_cand, d = 4, 8, 5, 64
    with torch.no_grad():                        # make the learned readout matter, not just init
        if readout == "weights":
            mixer.correction_gain.fill_(1.0)
        else:
            mixer.semantic_gain.fill_(0.5)
    args = dict(
        retrieval_score=torch.randn(n_query, n_ev),
        candidate_text=F.normalize(torch.randn(n_cand, TEXT_DIM), dim=-1),
        query_feature=torch.randn(n_query, d),
        query_descriptor=F.normalize(torch.randn(n_query, TEXT_DIM), dim=-1),
        evidence_feature=torch.randn(n_query, n_ev, d),
        evidence_descriptor=F.normalize(torch.randn(n_query, n_ev, TEXT_DIM), dim=-1),
        evidence_label_text=F.normalize(torch.randn(n_query, n_ev, TEXT_DIM), dim=-1),
        candidate_slot=torch.tensor([5, 11, 2, 30, 7]),
        evidence_slot=torch.tensor([[5, 0, 11, 0, 0, 2, 0, 0]] * n_query),
        evidence_group=torch.tensor([[1, 1, 2, 3, 3, 4, 5, 6]] * n_query),
    )
    straight = mixer(**args)

    order = torch.tensor([3, 0, 4, 1, 2])
    remap = torch.empty_like(order)
    remap[order] = torch.arange(len(order))
    slot_of = {int(args["candidate_slot"][i]): i for i in range(n_cand)}
    permuted = dict(args)
    permuted["candidate_text"] = args["candidate_text"][order]
    permuted["candidate_slot"] = args["candidate_slot"][order]
    # An enrolled row's slot follows the candidate it is bound to, so it is unchanged by a
    # relabelling of candidate POSITIONS: the slot ids themselves already carry the coreference.
    shuffled = mixer(**permuted)
    assert torch.allclose(straight["log_weight"][:, :, order], shuffled["log_weight"], atol=1e-5)
    assert torch.allclose(straight["label_vector"], shuffled["label_vector"], atol=1e-5)
    assert slot_of                                # the slot map is what makes the above meaningful


def test_evidence_rows_are_read_as_a_set():
    """Retrieved rows arrive in score order. Nothing may depend on that order beyond the score,
    which enters explicitly as an attention bias."""
    engine = _engine()
    query, memory, candidate_text, label_text = _episode()
    scores = engine.scorer(query.feature, query.descriptor, memory.feature, memory.descriptor)
    selected = PairScorer.select(scores, engine.cfg.top_k)
    reordered = torch.arange(len(memory.feature))
    swap = selected[0, :2].tolist()
    reordered[swap[0]], reordered[swap[1]] = swap[1], swap[0]
    shuffled = SensorRows(**{
        name: (None if getattr(memory, name) is None else getattr(memory, name)[reordered])
        for name in memory.__dataclass_fields__
    })
    generator = torch.Generator().manual_seed(3)
    straight = engine(query, memory, candidate_text, label_text,
                      generator=torch.Generator().manual_seed(3))["logits"]
    swapped = engine(query, shuffled, candidate_text, label_text,
                     generator=torch.Generator().manual_seed(3))["logits"]
    assert torch.allclose(straight, swapped, atol=1e-5)


def test_top_k_is_the_same_size_in_training_and_evaluation():
    """The previous design voted over the whole bank while training and a top-k slice while
    deploying, so the objective was not the deployed rule."""
    engine = _engine(top_k=8)
    query, memory, candidate_text, label_text = _episode()
    engine.train()
    assert engine(query, memory, candidate_text, label_text)["log_weight"].shape[1] == 8
    engine.eval()
    assert engine(query, memory, candidate_text, label_text)["log_weight"].shape[1] == 8


def test_single_retrieved_row_is_finite():
    """Per-query score standardization must define variance zero at k=1."""
    engine = _engine(top_k=1)
    query, memory, candidate_text, label_text = _episode()
    result = engine(query, memory, candidate_text, label_text, collect_stats=True)
    assert torch.isfinite(result["logits"]).all()
    assert torch.isfinite(result["log_weight"]).all()


def test_engine_refuses_a_bank_without_recording_provenance():
    engine = _engine()
    query, memory, candidate_text, label_text = _episode()
    blind = SensorRows(**{
        **{name: getattr(memory, name) for name in memory.__dataclass_fields__},
        "source_window": None,
    })
    with pytest.raises(ValueError, match="source_window"):
        engine(query, blind, candidate_text, label_text)


def test_parameter_report_totals_the_parts():
    engine = _engine()
    report = engine.parameter_report()
    assert report["TOTAL"] == report["scorer"] + report["mixer"]
    assert report["TOTAL"] == sum(p.numel() for p in engine.parameters() if p.requires_grad)


# ------------------------------------------------------------------ the bank
def _lopsided_corpus():
    """Six streams; one carries a hundred times the windows of the rest."""
    rng = np.random.default_rng(0)
    table, stream_of, label_of, position = {}, [], [], 0
    for stream in range(6):
        labels = range(12) if stream == 0 else rng.choice(12, 4, replace=False)
        table[stream] = {}
        for label in labels:
            count = 4000 if stream == 0 else 40
            table[stream][int(label)] = {0: np.arange(position, position + count)}
            stream_of += [stream] * count
            label_of += [int(label)] * count
            position += count
    return table, np.asarray(stream_of), np.asarray(label_of)


def test_bank_covers_streams_that_window_counts_would_bury():
    table, stream_of, label_of = _lopsided_corpus()
    rng = np.random.default_rng(1)
    plan = build_bank_plan(bank_index(table), spec=BankSpec(n_windows=256),
                           support_windows=0, exclude_labels=(), rng=rng)
    stratified = bank_composition(plan.positions, stream_of, label_of)
    uniform = bank_composition(rng.choice(len(stream_of), 256, replace=False), stream_of, label_of)
    assert stratified["bank/distinct_streams"] == 6
    assert stratified["bank/stream_entropy"] > 0.85
    assert uniform["bank/stream_entropy"] < 0.2


def test_bank_holds_its_size_as_support_grows():
    table, _, _ = _lopsided_corpus()
    index = bank_index(table)
    for support in (0, 16, 64):
        plan = build_bank_plan(index, spec=BankSpec(n_windows=256), support_windows=support,
                               exclude_labels=(), rng=np.random.default_rng(2))
        assert len(plan) + support == 256


def test_bank_refuses_impossible_support_or_distinct_size():
    table = {0: {0: {0: np.arange(3)}}}
    index = bank_index(table)
    with pytest.raises(ValueError, match="must lie"):
        build_bank_plan(index, spec=BankSpec(n_windows=2), support_windows=3,
                        exclude_labels=(), rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="requested 4 distinct"):
        build_bank_plan(index, spec=BankSpec(n_windows=4), support_windows=0,
                        exclude_labels=(), rng=np.random.default_rng(0))


def test_bank_never_contains_the_episode_answer():
    """At deployment the bank holds training labels and the query carries an unseen one. Training
    against a bank that already contains the answer teaches lookup in a closed vocabulary."""
    table, _, label_of = _lopsided_corpus()
    plan = build_bank_plan(bank_index(table), spec=BankSpec(n_windows=256), support_windows=0,
                           exclude_labels=(2, 5, 9), rng=np.random.default_rng(3))
    assert not set(label_of[list(plan.positions)]) & {2, 5, 9}


def test_bank_rows_are_distinct():
    table, _, _ = _lopsided_corpus()
    plan = build_bank_plan(bank_index(table), spec=BankSpec(n_windows=256), support_windows=0,
                           exclude_labels=(), rng=np.random.default_rng(4))
    assert len(set(plan.positions)) == len(plan.positions)


# ------------------------------------------------------------------ identity channels
def test_evidence_inherits_the_slot_of_the_candidate_it_is_bound_to():
    """Coreference is the whole point of the slot channel: an enrolled row and the candidate it
    was enrolled for must be marked as referring to the same thing."""
    engine = _engine(top_k=32)
    query, memory, candidate_text, label_text = _episode(n_memory=32)
    selected = torch.arange(32).unsqueeze(0).expand(len(query.feature), 32)
    candidate_slot, evidence_slot, _, bound = engine._identity_channels(
        memory, selected, len(candidate_text), torch.Generator().manual_seed(0),
    )
    enrolled = bound[0] >= 0
    assert enrolled.any()
    assert torch.equal(evidence_slot[0][enrolled], candidate_slot[bound[0][enrolled]])
    assert (evidence_slot[0][~enrolled] == 0).all()          # UNBOUND_SLOT


def test_rows_from_one_recording_share_a_group_and_rows_from_two_do_not():
    """Co-membership is how the mixer relates the accelerometer and gyroscope of one window now
    that the trunk encodes each sensor alone."""
    engine = _engine(top_k=8)
    query, memory, candidate_text, label_text = _episode(n_memory=32)
    selected = torch.arange(8).unsqueeze(0)                  # rows 0..7 -> windows 0,0,1,1,2,2,3,3
    _, _, group, _ = engine._identity_channels(
        memory, selected, len(candidate_text), torch.Generator().manual_seed(0),
    )
    assert int(group[0, 0]) == int(group[0, 1])
    assert int(group[0, 0]) != int(group[0, 2])
    assert len(set(group[0].tolist())) == 4


def test_identity_ids_are_redrawn_every_episode():
    """A stable id would let the model memorise 'slot 7 means walking', which turns a relational
    channel back into the closed vocabulary this design exists to avoid."""
    engine = _engine()
    _, memory, candidate_text, _ = _episode(n_memory=32)
    selected = torch.arange(8).unsqueeze(0)
    first = engine._identity_channels(memory, selected, len(candidate_text),
                             torch.Generator().manual_seed(1))[0]
    second = engine._identity_channels(memory, selected, len(candidate_text),
                              torch.Generator().manual_seed(2))[0]
    assert not torch.equal(first, second)


def test_identity_vocabularies_are_checked_rather_than_silently_wrapped():
    """Wrapping would alias two distinct referents onto one id — a wrong answer that looks fine."""
    engine = _engine()
    _, memory, _, _ = _episode(n_memory=64)
    too_many = F.normalize(torch.randn(engine.mixer.cfg.n_slots + 1, TEXT_DIM), dim=-1)
    with pytest.raises(ValueError, match="slots"):
        engine._identity_channels(memory, torch.arange(8).unsqueeze(0), len(too_many), None)

    narrow = _engine()
    narrow.mixer.cfg = type(narrow.mixer.cfg)(**{
        **{f: getattr(narrow.mixer.cfg, f) for f in narrow.mixer.cfg.__dataclass_fields__},
        "n_groups": 3,
    })
    with pytest.raises(ValueError, match="groups"):
        narrow._identity_channels(memory, torch.arange(8).unsqueeze(0), 4, None)


# ------------------------------------------------------------------ the readout
def test_a_zero_gain_readout_leaves_the_retrieval_score_untouched():
    """``correction_gain`` is logged every step precisely so 'the mixer did nothing' is visible rather
    than something to discover after a full run."""
    engine = _engine()
    with torch.no_grad():
        engine.mixer.correction_gain.fill_(0.0)
    query, memory, candidate_text, label_text = _episode()
    out = engine(query, memory, candidate_text, label_text,
                 generator=torch.Generator().manual_seed(0))
    picked = out["scores"].gather(1, out["selected"]).unsqueeze(-1)
    assert torch.allclose(out["log_weight"], picked.expand_as(out["log_weight"]), atol=1e-6)


def test_the_correction_is_live_at_the_default_gain():
    engine = _engine()
    query, memory, candidate_text, label_text = _episode()
    out = engine(query, memory, candidate_text, label_text,
                 generator=torch.Generator().manual_seed(0))
    picked = out["scores"].gather(1, out["selected"]).unsqueeze(-1)
    assert float((out["log_weight"] - picked).detach().abs().mean()) > 0


def test_zero_semantic_gain_returns_the_frozen_text_exactly():
    engine = _engine(readout="semantic")
    with torch.no_grad():
        engine.mixer.semantic_gain.fill_(0.0)
    query, memory, candidate_text, label_text = _episode()
    out = engine(query, memory, candidate_text, label_text,
                 generator=torch.Generator().manual_seed(0))
    from model.evidence.rows import evidence_label_tokens
    expected = evidence_label_tokens(memory, candidate_text, label_text)[out["selected"]]
    engine_labels = engine.mixer(
        retrieval_score=out["scores"].gather(1, out["selected"]),
        candidate_text=candidate_text,
        query_feature=query.feature, query_descriptor=query.descriptor,
        evidence_feature=memory.feature[out["selected"]],
        evidence_descriptor=memory.descriptor[out["selected"]],
        evidence_label_text=expected,
        candidate_slot=torch.arange(1, len(candidate_text) + 1),
        evidence_slot=torch.zeros_like(out["selected"]),
        evidence_group=torch.zeros_like(out["selected"]),
    )["label_vector"]
    assert torch.allclose(engine_labels, F.normalize(expected, dim=-1), atol=1e-6)


def test_refined_label_vectors_stay_unit_norm():
    """The readout is a cosine. A drifting norm would silently reweight evidence."""
    engine = _engine(readout="semantic")
    with torch.no_grad():
        engine.mixer.semantic_gain.fill_(0.4)
    query, memory, candidate_text, label_text = _episode()
    out = engine(query, memory, candidate_text, label_text,
                 generator=torch.Generator().manual_seed(0))
    refined = engine.mixer(
        retrieval_score=out["scores"].gather(1, out["selected"]),
        candidate_text=candidate_text,
        query_feature=query.feature, query_descriptor=query.descriptor,
        evidence_feature=memory.feature[out["selected"]],
        evidence_descriptor=memory.descriptor[out["selected"]],
        evidence_label_text=F.normalize(torch.randn(*out["selected"].shape, TEXT_DIM), dim=-1),
        candidate_slot=torch.arange(1, len(candidate_text) + 1),
        evidence_slot=torch.zeros_like(out["selected"]),
        evidence_group=torch.zeros_like(out["selected"]),
    )["label_vector"]
    assert torch.allclose(refined.norm(dim=-1), torch.ones_like(refined.norm(dim=-1)), atol=1e-5)


def test_gradient_reaches_the_encoder_features_through_the_engine():
    """Phase B fine-tunes the encoder. If the only path to a feature were the top-k selection —
    which is not differentiable — the encoder would receive nothing."""
    engine = _engine().train()
    query, memory, candidate_text, label_text = _episode(n_memory=64)
    query.feature.requires_grad_(True)
    memory.feature.requires_grad_(True)
    out = engine(query, memory, candidate_text, label_text)
    F.cross_entropy(out["logits"] / 0.1,
                    torch.zeros(len(out["logits"]), dtype=torch.long)).backward()
    assert float(query.feature.grad.abs().sum()) > 0
    touched = int((memory.feature.grad.abs().sum(dim=1) > 0).sum())
    assert touched >= len(out["selected"].unique())


# ------------------------------------------------------------------ scorer efficiency
def test_the_interaction_heads_match_their_reference_formulation():
    """The heads are computed as one batched matmul rather than an einsum, for dispatch. That is a
    performance rewrite, so it needs the formulation it replaced as a reference."""
    spec = AttentionSpec(d_model=32, n_heads=4)
    scorer = PairScorer(spec).eval()
    qf, mf = torch.randn(6, 32), torch.randn(11, 32)
    qd = F.normalize(torch.randn(6, TEXT_DIM), dim=-1)
    md = F.normalize(torch.randn(11, TEXT_DIM), dim=-1)
    query_heads = scorer._joint_heads(qf, qd, scorer.query_proj)
    memory_heads = scorer._joint_heads(mf, md, scorer.memory_proj)
    batched = torch.bmm(query_heads.transpose(0, 1), memory_heads.permute(1, 2, 0)).permute(1, 2, 0)
    reference = torch.einsum("qhk,mhk->qmh", query_heads, memory_heads)
    assert torch.allclose(batched, reference, atol=1e-5)


def test_the_reduced_precision_pair_head_does_not_change_what_gets_retrieved():
    """The pair head runs in bfloat16 on CUDA because it is bandwidth-bound — 71% of the scorer's
    forward pass, and 8.6x faster at reduced precision.

    The claim that this is free has to be checked where it matters, which is not the value of the
    score but the SELECTION it drives: top-k is hard, so a row that changes side changes which
    evidence exists and which encoder rows get gradient. Checked at gains far above the 0.02 the
    learned term starts at, so the bound survives the head growing.
    """
    torch.manual_seed(0)
    spec = AttentionSpec(d_model=64, n_heads=4)
    scorer = PairScorer(spec).eval()
    query_feature, memory_feature = torch.randn(16, 64), torch.randn(600, 64)
    query_desc = F.normalize(torch.randn(16, TEXT_DIM), dim=-1)
    memory_desc = F.normalize(torch.randn(600, TEXT_DIM), dim=-1)

    query_heads = scorer._joint_heads(query_feature, query_desc, scorer.query_proj)
    memory_heads = scorer._joint_heads(memory_feature, memory_desc, scorer.memory_proj)
    interaction = torch.einsum("qhk,mhk->qmh", query_heads, memory_heads)
    interaction = interaction / (scorer.cfg.interaction_dim ** 0.5)
    feature_cos = (F.normalize(query_feature, dim=-1)
                   @ F.normalize(memory_feature, dim=-1).t())
    desc_cos = F.normalize(query_desc, dim=-1) @ F.normalize(memory_desc, dim=-1).t()
    stacked = torch.cat([feature_cos.unsqueeze(-1), desc_cos.unsqueeze(-1), interaction], dim=-1)

    exact = scorer._pair_head(stacked).detach()
    reduced = F.linear(
        F.gelu(F.linear(stacked.bfloat16(), scorer.pair_head_in.weight.bfloat16(),
                        scorer.pair_head_in.bias.bfloat16())),
        scorer.pair_head_out.weight.bfloat16(),
    ).squeeze(-1).float().detach()
    assert float((exact - reduced).abs().mean() / exact.std()) < 0.02

    base = (scorer.base_gain * feature_cos).detach()
    for gain in (0.02, 1.0, 5.0):
        chosen = (base + gain * exact).topk(64, dim=1).indices
        approx = (base + gain * reduced).topk(64, dim=1).indices
        overlap = [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(chosen, approx)]
        assert min(overlap) >= 63, (gain, min(overlap))

    assert exact.dtype == stacked.dtype            # CPU stays exact, so tests are a reference


# ------------------------------------------------------------------ why the scalar readout stays
def test_a_candidate_blind_weighting_collapses_to_one_vector_per_query():
    """The structural reason ``readout="weights"`` is the default.

    If the evidence weight does not vary with the candidate, the readout factorises:

        logit_c = sum_m w_m <l_m, t_c> = < sum_m w_m l_m , t_c > = <v_q, t_c>

    The episode is then ONE vector scored against every candidate — a CLIP-shaped model in which
    no candidate can consult different evidence from any other. The rectification in `vote` is the
    only thing standing in the way, and on this corpus's label vocabulary it fires on 0.6% of
    (row, candidate) pairs, so it does not rescue the structure.
    """
    torch.manual_seed(0)
    n_rows, n_cand = 8, 5
    label_vector = F.normalize(torch.randn(1, n_rows, TEXT_DIM), dim=-1)
    candidate_text = F.normalize(torch.randn(n_cand, TEXT_DIM), dim=-1)
    shared = torch.softmax(torch.randn(1, n_rows), dim=1)

    factorised = torch.einsum("qm,qmz->qz", shared, label_vector) @ candidate_text.T
    direct = torch.einsum("qm,qmz,cz->qc", shared, label_vector, candidate_text)
    assert torch.allclose(direct, factorised, atol=1e-5)


def test_the_scalar_readout_weights_each_candidate_differently():
    """What the collapse above costs, and what step 6 buys: with per-candidate weights the model
    performs one weighted read of memory PER CANDIDATE, so two candidates can be supported by
    genuinely different rows. Checked with nothing enrolled, because the enrolled-row mask is a
    second source of per-candidate variation and would confound it."""
    engine = _engine(readout="weights", top_k=8)
    with torch.no_grad():
        engine.mixer.correction_gain.fill_(1.0)
    query, memory, candidate_text, label_text = _episode(n_memory=48)
    bare = SensorRows(**{
        **{f: getattr(memory, f) for f in memory.__dataclass_fields__},
        "enrolled_candidate": torch.full_like(memory.enrolled_candidate, -1),
    })
    log_weight = engine(query, bare, candidate_text, label_text,
                        generator=torch.Generator().manual_seed(0))["log_weight"]
    across_candidates = (log_weight - log_weight.mean(dim=2, keepdim=True)).abs().mean()
    assert float(across_candidates.detach()) > 1e-3

    blind = _engine(readout="semantic", top_k=8)
    blind_weight = blind(query, bare, candidate_text, label_text,
                         generator=torch.Generator().manual_seed(0))["log_weight"]
    # Float32 mean of identical values, so exact zero is not guaranteed — but it is the
    # broadcast retrieval score in every candidate column, three orders below the other arm.
    assert float((blind_weight - blind_weight.mean(dim=2, keepdim=True)).abs().max().detach()) < 1e-6


def test_the_score_couples_patch_and_text_rather_than_summing_separate_scores():
    """The user-facing claim about step 3: ONE learned mechanism over (patch vector, description)
    together, not a signal scorer and a text scorer added up.

    The discriminating quantity is the mixed second difference over one memory row's two inputs:

        score(f1, d1) + score(f2, d2) - score(f1, d2) - score(f2, d1)

    For ANY additive decomposition s1(features) + s2(descriptions) this is exactly zero, whatever
    the two mechanisms are. A genuinely joint scorer leaves a residue, because the pair head and
    the interaction heads couple the two. The feature-only cosine anchor cancels out of the
    difference automatically, so this isolates the learned part.
    """
    torch.manual_seed(0)
    spec = AttentionSpec(d_model=32, n_heads=4)
    scorer = PairScorer(spec).eval()
    with torch.no_grad():
        scorer.residual_gain.fill_(1.0)
    query_feature = torch.randn(1, 32)
    query_desc = F.normalize(torch.randn(1, TEXT_DIM), dim=-1)
    f1, f2 = torch.randn(1, 32), torch.randn(1, 32)
    d1, d2 = (F.normalize(torch.randn(1, TEXT_DIM), dim=-1) for _ in range(2))

    def score(feature, desc):
        return float(scorer(query_feature, query_desc, feature, desc).detach())

    mixed = score(f1, d1) + score(f2, d2) - score(f1, d2) - score(f2, d1)
    assert abs(mixed) > 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to reproduce")
def test_mixed_precision_keeps_small_learned_residuals_in_the_forward_pass():
    """BF16 at score magnitude ~10 used to round the 0.02-scale residual entirely away."""
    engine = _engine().cuda().train()
    query, memory, candidate_text, label_text = _episode()
    query, memory = query.to("cuda"), memory.to("cuda")
    candidate_text, label_text = candidate_text.cuda(), label_text.cuda()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = engine(query, memory, candidate_text, label_text,
                        generator=torch.Generator().manual_seed(1))
    picked = result["scores"].gather(1, result["selected"]).unsqueeze(-1)
    correction = (result["log_weight"] - picked).abs().mean()
    assert result["scores"].dtype == torch.float32
    assert result["log_weight"].dtype == torch.float32
    assert float(correction.detach()) > 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to reproduce")
def test_a_cpu_generator_works_with_cuda_rows():
    """torch.Generator() is CPU by default and seeding one is how deployment gets reproducible
    slot assignments, so the engine must accept that pairing with CUDA rows. Found by the debug
    sweep as a crash on the very first GPU episode."""
    engine = _engine().cuda()
    query, memory, candidate_text, label_text = _episode()
    to_cuda = lambda rows: SensorRows(**{
        name: (None if getattr(rows, name) is None else getattr(rows, name).cuda())
        for name in rows.__dataclass_fields__})
    out = engine(to_cuda(query), to_cuda(memory), candidate_text.cuda(), label_text.cuda(),
                 generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out["logits"]).all()


def test_retrieval_alignment_loss_prefers_semantically_near_labels():
    """The aux loss must be minimized when the score ranks rows by LABEL-TEXT relevance, which is
    the property the measured bottleneck needs. Checked against a deliberately wrong ranking."""
    from training.tokenizer.episodic import retrieval_alignment_loss

    torch.manual_seed(0)
    label_text = F.normalize(torch.randn(10, TEXT_DIM), dim=-1)
    query_label = torch.tensor([0, 1])
    memory_label = torch.arange(10)
    relevance = label_text[query_label] @ label_text[memory_label].t()
    aligned = retrieval_alignment_loss(relevance * 10, query_label, memory_label, label_text)
    inverted = retrieval_alignment_loss(-relevance * 10, query_label, memory_label, label_text)
    uniform = retrieval_alignment_loss(torch.zeros(2, 10), query_label, memory_label, label_text)
    assert float(aligned) < float(uniform) < float(inverted)
    assert float(aligned) > 0                      # it is a cross-entropy, never negative


def test_retrieval_alignment_loss_reaches_the_scorer():
    from training.tokenizer.episodic import retrieval_alignment_loss

    engine = _engine().train()
    query, memory, candidate_text, label_text = _episode()
    scores = engine.scorer(query.feature, query.descriptor, memory.feature, memory.descriptor)
    retrieval_alignment_loss(scores, query.label, memory.label, label_text).backward()
    touched = [n for n, p in engine.scorer.named_parameters()
               if p.grad is not None and float(p.grad.abs().sum()) > 0]
    assert len(touched) >= 5


def test_residual_branches_start_as_small_perturbations():
    """Standard transformer init, and it was missing.

    Without scaling the residual-branch output projections by 1/sqrt(2*n_layers), the first block
    dominates the residual stream: measured on real episode tokens, layer 0's attention update was
    |dx|/|x| = 2.55 — it overwrote the representation it was meant to refine — and layer 1 then sat
    at 99% of uniform attention entropy with nothing left to discriminate.
    """
    torch.manual_seed(0)
    spec = AttentionSpec(d_model=128, n_heads=4, ffn_mult=2, dropout=0.0)
    for n_layers in (2, 4):
        stack = SetAttentionStack(spec, n_layers).eval()
        # tokens shaped like the mixer's: unit-norm content plus three unit-norm identity vectors
        x = sum(F.normalize(torch.randn(8, 128, 128), dim=-1) for _ in range(4))
        normed = stack.attn_norm[0](x)
        qkv = stack.qkv[0](normed).view(8, 128, 3, spec.n_heads, spec.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(8, 128, spec.d_model)
        ratio = float(stack.proj[0](attended).norm(dim=-1).mean() / x.norm(dim=-1).mean())
        assert ratio < 0.5, (n_layers, ratio)


def test_the_fixed_retrieval_stand_in_has_no_parameters():
    """A ladder rung that disables retrieval learning must really disable it, not merely shrink it."""
    from model.evidence.retrieval_scorer import PairScorerConfig

    engine = EvidenceEngine(None, EngineConfig(
        spec=AttentionSpec(d_model=64, n_heads=4), top_k=8, mixing="off",
        scorer=PairScorerConfig(learned=False),
    ))
    assert sum(p.numel() for p in engine.parameters() if p.requires_grad) == 0
    query, memory, candidate_text, label_text = _episode()
    out = engine(query, memory, candidate_text, label_text)
    # with no mixer the weight is the retrieval score alone, identical across candidates
    lw = out["log_weight"]
    assert torch.isfinite(out["logits"]).all()
    assert float((lw - lw.mean(dim=2, keepdim=True)).abs().max()) < 1e-6
