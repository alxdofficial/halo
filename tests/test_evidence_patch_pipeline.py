"""Regression tests for schema-v2 patch evidence, learned retrieval, and confidence."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from model.evidence.confidence import EvidenceConfidenceHead, confidence_features
from model.evidence.decoder import DecoderConfig, EvidenceDecoder
from model.evidence.patch_retrieval import PatchSubspaceRetriever
from training.evidence.build_memory import label_config_balanced_keep
from training.evidence.bank_guard import assert_patch_bank
from training.evidence.patch_episodes import (
    PatchTable,
    assemble_evidence,
    build_allowed_mask,
    queries_from_encoded,
)
from training.evidence.train_patch_decoder import (
    choose_candidates,
    family_holdout_labels,
    load_activity_families,
    run_patch_episode,
    sample_queries,
)


def _bank():
    # Four windows, two patches each. Windows 0/1 are simultaneous placements of event 10.
    window = torch.arange(4).repeat_interleave(2)
    y = torch.tensor([0, 0, 1, 2])
    subj = torch.tensor([0, 0, 1, 2])
    cfg = torch.tensor([0, 1, 0, 1])
    event = torch.tensor([10, 10, 20, 30])
    verified = torch.tensor([True, True, False, False])
    patch = {
        "Z": torch.randn(8, 8).half(),
        "y": y[window],
        "subj": subj[window],
        "cfg": cfg[window],
        "sensor": cfg[window],
        "window": window,
        "event": event[window],
        "event_verified": verified[window],
        "time": torch.tensor([0.5, 1.5] * 4),
        "duration": torch.ones(8),
        "resolution": torch.tensor([0, 1] * 4),
    }
    return {
        "schema_version": 2,
        "patch_embed_probe": torch.zeros(1),
        "Z": torch.randn(4, 8).half(),
        "y": y, "subj": subj, "cfg": cfg, "event": event,
        "event_verified": verified,
        "cfg_rate_hz": {0: 50.0, 1: 100.0},
        "patch": patch,
    }


def test_patch_bank_foreign_keys_and_event_expansion():
    bank = _bank()
    assert_patch_bank(bank, context="test")
    table = PatchTable(bank)
    query = table.gather_queries(torch.tensor([0]), "cpu")
    assert query.mask.sum() == 4
    assert set(query.window[query.mask].tolist()) == {0, 1}
    assert set(query.sensor[query.mask].tolist()) == {0, 1}


def test_index_coreset_stratifies_label_by_config_and_resolution():
    bank = _bank()
    rows = PatchTable(bank).sample_index_rows(
        torch.ones(4, dtype=torch.bool), per_label=4, rng=np.random.default_rng(1)
    )
    label0 = rows[bank["patch"]["y"][rows].eq(0)]
    groups = set(zip(
        bank["patch"]["cfg"][label0].tolist(),
        bank["patch"]["resolution"][label0].tolist(),
    ))
    assert groups == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_memory_label_cap_is_configuration_balanced():
    labels = torch.tensor([0] * 110 + [1] * 5)
    configs = torch.tensor([0] * 100 + [1] * 10 + [0] * 5)
    keep = label_config_balanced_keep(labels, configs, max_per_label=20, seed=7)
    kept_label0 = configs[keep & labels.eq(0)]
    assert len(kept_label0) == 20
    assert torch.bincount(kept_label0, minlength=2).tolist() == [10, 10]
    assert int((keep & labels.eq(1)).sum()) == 5


def test_phase_b_queries_balance_configs_and_temper_subjects():
    # One label: config 0 has a 90/10 subject split, config 1 has ten windows. Hierarchical
    # sampling should make configs ~50/50 and the large subject ~75% within config 0, not 90%.
    y = torch.zeros(110, dtype=torch.long)
    cfg = torch.tensor([0] * 100 + [1] * 10)
    subj = torch.tensor([0] * 90 + [1] * 10 + [2] * 10)
    pool = torch.arange(len(y))
    draws = []
    rng = np.random.default_rng(8)
    for _ in range(2000):
        draws.append(int(sample_queries(
            pool, torch.tensor([0]), y, 1, rng,
            config_ids=cfg, subject_ids=subj, label_alpha=0.0,
        )[0]))
    drawn = torch.tensor(draws)
    config0_fraction = float(cfg[drawn].eq(0).float().mean())
    config0 = drawn[cfg[drawn].eq(0)]
    large_subject_fraction = float(subj[config0].eq(0).float().mean())
    assert 0.45 < config0_fraction < 0.55
    assert 0.70 < large_subject_fraction < 0.80


def test_external_query_packing_uses_noncolliding_structural_ids():
    encoded = {
        "patch_Z": torch.randn(5, 8),
        "patch_window": torch.tensor([0, 0, 1, 1, 1]),
        "patch_time": torch.tensor([0.5, 1.5, 0.5, 1.5, 2.5]),
        "patch_duration": torch.ones(5),
        "patch_resolution": torch.tensor([0, 1, 0, 0, 1]),
    }
    query = queries_from_encoded(encoded, torch.tensor([0, 1]), "cpu", sensor_id=-1)
    assert query.mask.sum(1).tolist() == [2, 3]
    assert (query.window[query.mask] < 0).all()
    assert query.sensor[query.mask].eq(-1).all()


def test_patch_bank_guard_rejects_wrong_parent_metadata():
    bank = _bank()
    bank["patch"]["event"][0] = 999
    with pytest.raises(SystemExit, match="patch.event disagrees"):
        assert_patch_bank(bank, context="test")


def test_patch_bank_guard_requires_valid_sensor_identity():
    bank = _bank()
    del bank["patch"]["sensor"]
    with pytest.raises(SystemExit, match="missing \\['sensor'\\]"):
        assert_patch_bank(bank, context="test")

    bank = _bank()
    bank["patch"]["sensor"][0] = -1
    with pytest.raises(SystemExit, match="nonnegative integer structural sensor ids"):
        assert_patch_bank(bank, context="test")


def test_allowed_mask_excludes_subject_event_window_and_caps_support_by_window():
    bank = _bank()
    table = PatchTable(bank)
    query = table.gather_queries(torch.tensor([2]), "cpu")
    rows = torch.arange(8)
    allowed = build_allowed_mask(
        bank["patch"], rows, query, torch.tensor([1]), torch.tensor([0, 1, 2]),
        truth_present=True, true_support=0, other_support=1, config_mode="any",
        rng=np.random.default_rng(2),
    )
    # Query's own subject/event/window and every true-label support row are excluded.
    assert not allowed[..., bank["patch"]["subj"].eq(1)].any()
    assert not allowed[..., bank["patch"]["event"].eq(20)].any()
    assert not allowed[..., bank["patch"]["y"].eq(1)].any()
    # Label 0's two placements are one verified physical event, so support=1 retains both sensors.
    retained_label0 = rows[allowed[0, 0] & bank["patch"]["y"].eq(0)]
    assert len(torch.unique(bank["patch"]["window"][retained_label0])) == 2
    assert len(retained_label0) == 4


def test_query_absent_excludes_every_config_in_a_multisensor_query():
    bank = _bank()
    query = PatchTable(bank).gather_queries(torch.tensor([0]), "cpu")
    rows = torch.arange(8)
    cross = build_allowed_mask(
        bank["patch"], rows, query, torch.tensor([0]), torch.tensor([0, 1, 2]),
        truth_present=True, true_support=None, other_support=None,
        config_mode="cross", rng=np.random.default_rng(4),
    )
    assert cross.any()  # each sensor can use configurations belonging to the other query sensor
    absent = build_allowed_mask(
        bank["patch"], rows, query, torch.tensor([0]), torch.tensor([0, 1, 2]),
        truth_present=True, true_support=None, other_support=None,
        config_mode="query_absent", rng=np.random.default_rng(4),
    )
    assert not absent.any()  # this tiny bank has only the two configurations in the query session


def test_subspace_retrieval_is_independent_masked_and_differentiable():
    torch.manual_seed(3)
    retriever = PatchSubspaceRetriever(8, n_subspaces=3, subspace_dim=4, ema_decay=0.5)
    query = F.normalize(torch.randn(2, 2, 8), dim=-1)
    memory = F.normalize(torch.randn(9, 8), dim=-1)
    index = retriever.build_index(memory)
    allowed = torch.ones(2, 2, 9, dtype=torch.bool)
    allowed[0, 0, :8] = False
    result = retriever.retrieve(query, index, allowed, k=4)
    assert result.index.shape == (2, 2, 3, 4)
    assert result.valid[0, 0].sum() == 3  # one eligible row independently in each head
    assert set(result.index[0, 0][result.valid[0, 0]].tolist()) == {8}
    score = retriever.score_selected(query, memory, result.index)
    score[result.valid].sum().backward()
    assert retriever.proj.grad is not None
    assert float(retriever.proj.grad.abs().sum()) > 0


def test_subspace_retrieval_ignores_padded_query_slots_without_requiring_memory():
    retriever = PatchSubspaceRetriever(8, 2, 4)
    query = F.normalize(torch.randn(1, 3, 8), dim=-1)
    memory = F.normalize(torch.randn(5, 8), dim=-1)
    allowed = torch.zeros(1, 3, 5, dtype=torch.bool)
    allowed[:, :2] = True
    result = retriever.retrieve(
        query, retriever.build_index(memory), allowed, k=2,
        query_mask=torch.tensor([[True, True, False]]),
    )
    assert result.valid[:, :2].all()
    assert not result.valid[:, 2].any()


def test_subspace_ema_update_tracks_the_online_projection():
    torch.manual_seed(4)
    retriever = PatchSubspaceRetriever(8, 3, 4, ema_decay=0.5)
    old = retriever.ema_proj.clone()
    with torch.no_grad():
        retriever.proj.add_(1)
    retriever.update_ema()
    assert torch.allclose(retriever.ema_proj, old.lerp(retriever.proj, 0.5))


def test_assemble_evidence_caps_windows_and_normalizes_head_resolution_groups():
    from model.evidence.patch_retrieval import PatchRetrieval

    bank = _bank()
    # B=1,Q=1,H=2,K=2, with two heads selecting patches from both resolutions.
    retrieval = PatchRetrieval(
        index=torch.tensor([[[[0, 2], [1, 3]]]]),
        score=torch.ones(1, 1, 2, 2),
        valid=torch.ones(1, 1, 2, 2, dtype=torch.bool),
    )
    online = torch.tensor([[[[0.9, 0.8], [0.7, 0.6]]]], requires_grad=True)
    result = assemble_evidence(
        retrieval, online, torch.arange(8), bank["patch"],
        max_evidence=4, max_per_window=2, max_per_label=4, tau=0.1,
    )
    assert torch.allclose(result.weights.sum(1), torch.ones(1))
    assert len(torch.unique(result.head[result.mask])) == 2
    result.weights.sum().backward()
    assert online.grad is not None


def _decoder_inputs(B=2, Q=3, K=5, C=4, d=16, text=12):
    gen = torch.Generator().manual_seed(9)
    zq = torch.randn(B, Q, d, generator=gen)
    zev = torch.randn(B, K, d, generator=gen)
    ev_text = F.normalize(torch.randn(B, K, text, generator=gen), dim=-1)
    candidate = F.normalize(torch.randn(B, C, text, generator=gen), dim=-1)
    weights = torch.softmax(torch.randn(B, K, generator=gen), dim=1)
    return dict(zq=zq, zev=zev, ev_label_text=ev_text, w_retr=weights,
                cand_text=candidate)


def test_candidate_decoder_is_identity_at_init_and_candidate_permutation_equivariant():
    torch.manual_seed(10)
    cfg = DecoderConfig(
        d_model=16, text_dim=12, n_heads=4, n_layers=1,
        candidate_tokens=True, candidate_layers=1, structural_metadata=True,
    )
    dec = EvidenceDecoder(cfg).eval()
    inputs = _decoder_inputs()
    with torch.no_grad():
        logits = dec(**inputs)
    votes = torch.relu(torch.einsum(
        "bkt,bct->bkc",
        F.normalize(inputs["ev_label_text"], dim=-1),
        F.normalize(inputs["cand_text"], dim=-1),
    ))
    reference = cfg.out_scale_init * torch.einsum("bk,bkc->bc", inputs["w_retr"], votes)
    assert torch.allclose(logits, reference, atol=1e-5)

    # Open the candidate residual so the test exercises candidate attention, not only init identity.
    with torch.no_grad():
        dec.candidate_refiner.weight.normal_(0, 0.02)
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        a = dec(**inputs)
        b = dec(**{**inputs, "cand_text": inputs["cand_text"][:, perm]})
    assert torch.allclose(a[:, perm], b, atol=1e-5)


def test_multiquery_masks_and_structural_metadata_are_operational():
    torch.manual_seed(11)
    dec = EvidenceDecoder(DecoderConfig(
        d_model=16, text_dim=12, n_heads=4, n_layers=1,
        candidate_tokens=True, candidate_layers=1, structural_metadata=True,
    )).eval()
    with torch.no_grad():
        dec.refiner[-1].weight.normal_(0, 0.02)
    inputs = _decoder_inputs()
    q_mask = torch.tensor([[True, True, False], [True, True, False]])
    metadata = dict(
        q_mask=q_mask,
        q_time=torch.tensor([[0.5, 1.5, 99.0]] * 2),
        q_duration=torch.tensor([[1.0, 1.0, 99.0]] * 2),
        q_resolution=torch.tensor([[0, 1, 1]] * 2),
        q_sensor_id=torch.tensor([[0, 1, 2]] * 2),
    )
    changed_padding = {**metadata}
    changed_padding["q_time"] = metadata["q_time"].clone()
    changed_padding["q_time"][:, 2] = -500.0
    with torch.no_grad():
        a = dec(**inputs, **metadata)
        b = dec(**inputs, **changed_padding)
    assert torch.allclose(a, b, atol=1e-5)


def test_same_sensor_relation_bias_is_trainable_and_permutation_safe():
    torch.manual_seed(13)
    dec = EvidenceDecoder(DecoderConfig(
        d_model=16, text_dim=12, n_heads=4, n_layers=1,
        candidate_tokens=True, candidate_layers=1, structural_metadata=True,
    )).train()
    inputs = _decoder_inputs()
    B, Q = inputs["zq"].shape[:2]
    q_sensor = torch.tensor([[0, 0, 1]] * B)
    ev_sensor = torch.tensor([[0, 1, 1, 2, 2]] * B)
    with torch.no_grad():
        dec.refiner[-1].weight.normal_(0, 0.02)
    dec(**inputs, q_sensor_id=q_sensor, ev_sensor_id=ev_sensor).sum().backward()
    assert dec.same_sensor_bias.grad is not None

    dec.eval()
    perm = torch.tensor([3, 0, 4, 1, 2])
    permuted = {
        **inputs,
        "zev": inputs["zev"][:, perm],
        "ev_label_text": inputs["ev_label_text"][:, perm],
        "w_retr": inputs["w_retr"][:, perm],
    }
    with torch.no_grad():
        a = dec(**inputs, q_sensor_id=q_sensor, ev_sensor_id=ev_sensor)
        b = dec(**permuted, q_sensor_id=q_sensor, ev_sensor_id=ev_sensor[:, perm])
    assert torch.allclose(a, b, atol=1e-5)


def test_confidence_features_are_finite_and_candidate_entropy_is_normalized():
    evidence2 = torch.tensor([[1.0, 1.0]])
    evidence8 = torch.ones(1, 8)
    score = torch.tensor([[0.8, 0.7]])
    weight = torch.tensor([[0.5, 0.5]])
    votes2 = torch.ones(1, 2, 2)
    votes8 = torch.ones(1, 2, 8)
    f2 = confidence_features(evidence2, score, votes2, weight)
    f8 = confidence_features(evidence8, score, votes8, weight)
    assert torch.isfinite(f2).all() and torch.isfinite(f8).all()
    assert torch.allclose(f2[:, 2], f8[:, 2], atol=1e-6)  # normalized concentration
    head = EvidenceConfidenceHead()
    assert head(f2).shape == (1,)


def test_patch_episode_smoke_has_finite_gradients_and_attribution():
    torch.manual_seed(12)
    bank = _bank()
    table = PatchTable(bank)
    retriever = PatchSubspaceRetriever(8, 2, 4)
    dec = EvidenceDecoder(DecoderConfig(
        d_model=8, text_dim=6, n_heads=2, n_layers=1,
        candidate_tokens=True, candidate_layers=1, structural_metadata=True,
    ))
    index_rows = torch.arange(len(bank["patch"]["Z"]))
    memory = F.normalize(bank["patch"]["Z"].float(), dim=-1)
    memory_index = retriever.build_index(memory)
    text = F.normalize(torch.randn(3, 6), dim=-1)
    logits, aux = run_patch_episode(
        dec, retriever, table, bank, index_rows, memory_index,
        torch.tensor([0]), torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), text, text,
        truth_present=True, true_support=0, other_support=None,
        config_mode="any", rng=np.random.default_rng(4), topk_per_head=2,
        max_evidence=4, max_per_window=2, max_per_label=4, tau=0.1,
        query_window_mask=torch.ones(4, dtype=torch.bool),
    )
    loss = F.cross_entropy(logits, torch.tensor([0]))
    loss.backward()
    assert torch.isfinite(logits).all()
    assert retriever.proj.grad is not None
    assert {"evidence_index", "evidence_sensor", "evidence_head", "evidence_query_patch"} <= set(aux)


def test_memory_support_caps_apply_to_non_candidate_labels():
    bank = _bank()
    query = PatchTable(bank).gather_queries(torch.tensor([2]), "cpu")
    rows = torch.arange(8)
    # Candidate set could be [1, 2], but memory label 0 is independently visible and capped.
    allowed = build_allowed_mask(
        bank["patch"], rows, query, torch.tensor([1]), torch.tensor([0, 1]),
        truth_present=True, true_support=0, other_support=1,
        config_mode="any", rng=np.random.default_rng(3),
    )
    retained = rows[allowed[0, 0] & bank["patch"]["y"].eq(0)]
    assert len(torch.unique(bank["patch"]["window"][retained])) == 2


def test_activity_family_mapping_is_complete_and_drives_family_distractors():
    import json

    vocab = json.load(open("data/labels/global_labels.json"))["labels"]
    family_ids, families, fingerprint = load_activity_families(vocab)
    assert len(family_ids) == len(vocab)
    assert len(fingerprint) == 16
    walking = vocab.index("walking")
    candidate = choose_candidates(
        torch.tensor([walking]), 6, len(vocab),
        F.normalize(torch.randn(len(vocab), 8), dim=-1),
        F.normalize(torch.randn(len(vocab), 8), dim=-1),
        truth_present=True, mode="motion_family", rng=np.random.default_rng(5),
        family_ids=family_ids,
    )
    distractors = candidate[candidate.ne(walking)]
    assert torch.isin(family_ids[distractors], family_ids[torch.tensor([walking])]).any()
    heldout = family_holdout_labels(family_ids, torch.arange(len(vocab)), 1)
    chosen_family = torch.unique(family_ids[heldout])
    assert len(chosen_family) == 1
    expected = torch.nonzero(family_ids.eq(chosen_family[0]), as_tuple=True)[0]
    assert set(heldout.tolist()) == set(expected.tolist())
