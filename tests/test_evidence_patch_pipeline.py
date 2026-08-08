"""Regression tests for source-aware patch evidence, learned retrieval, and confidence."""

from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from data.scripts.eda.grid_io import GridRef
from model.evidence.confidence import (
    EvidenceConfidenceHead,
    binary_auprc,
    confidence_features,
    expected_calibration_error,
)
from model.evidence.decoder import DecoderConfig, EvidenceDecoder
from model.evidence.patch_retrieval import PatchSubspaceRetriever
from training.evidence.build_memory import archive_budget_balanced_keep, label_config_balanced_keep
from training.evidence.bank_guard import assert_patch_bank
from training.evidence.patch_episodes import (
    EpisodeMemoryView,
    PatchTable,
    assemble_evidence,
    balanced_memory_log_prior,
    build_allowed_mask,
    build_episode_memory_view,
    describe_episode_composition,
    queries_from_encoded,
    simultaneous_stream_pairs,
)
from training.evidence.policy import PHYSICAL_VIEW_MODES, PhaseBPolicy
from training.evidence.live_encoder import PatchViewSpec, SourcePatchEncoder
from training.tokenizer.eval_transfer import encode_dataset_detailed
from training.evidence.train_patch_decoder import (
    EpisodeCurriculum,
    _episode_view_specs,
    choose_candidates,
    decode_adaptation_episode,
    family_holdout_labels,
    load_activity_families,
    parameter_gradient_norm,
    prepare_adaptation_views,
    run_patch_episode,
    sample_queries,
    soft_retrieval_temperature,
)
from training.evidence.subject_style import SubjectStyle, apply_subject_style
from training.evidence.runtime_memory import build_enrollment_memory
from training.evidence.eval_enrollment import (
    _support_and_query_rows,
    build_paired_enrollment_plans,
    paired_subject_summary,
)
from training.evidence.device import resolve_device


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
        "ordinal": torch.tensor([0, 1] * 4),
    }
    return {
        "schema_version": 3,
        "population_fp": "test-population",
        "patch_embed_probe": torch.zeros(1),
        "Z": torch.randn(4, 8).half(),
        "y": y, "subj": subj, "cfg": cfg, "event": event,
        "event_verified": verified,
        "source_row": torch.arange(4),
        "source_alignment": "native",
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
        torch.ones(4, dtype=torch.bool), windows_per_label=2, rng=np.random.default_rng(1)
    )
    label0 = rows[bank["patch"]["y"][rows].eq(0)]
    groups = set(zip(
        bank["patch"]["cfg"][label0].tolist(),
        bank["patch"]["resolution"][label0].tolist(),
    ))
    assert groups == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_index_coreset_does_not_prefer_low_config_ids_when_budget_is_tight():
    bank = _bank()
    n = 20
    bank["Z"] = torch.randn(n, 8).half()
    bank["y"] = torch.zeros(n, dtype=torch.long)
    bank["subj"] = torch.arange(n)
    bank["cfg"] = torch.arange(n)
    bank["event"] = torch.arange(n)
    bank["event_verified"] = torch.zeros(n, dtype=torch.bool)
    bank["source_row"] = torch.arange(n)
    bank["patch"] = {
        "Z": torch.randn(n, 8).half(), "y": bank["y"], "subj": bank["subj"],
        "cfg": bank["cfg"], "sensor": bank["cfg"], "window": torch.arange(n),
        "event": bank["event"], "event_verified": bank["event_verified"],
        "time": torch.full((n,), 0.5), "duration": torch.ones(n),
        "resolution": torch.zeros(n, dtype=torch.long),
        "ordinal": torch.zeros(n, dtype=torch.long),
    }
    table = PatchTable(bank)
    first = table.sample_index_rows(
        torch.ones(n, dtype=torch.bool), 5, np.random.default_rng(1)
    )
    second = table.sample_index_rows(
        torch.ones(n, dtype=torch.bool), 5, np.random.default_rng(2)
    )
    assert len(first) == len(second) == 5
    assert set(first.tolist()) != set(range(5))
    assert set(first.tolist()) != set(second.tolist())


def test_memory_label_cap_is_configuration_balanced():
    labels = torch.tensor([0] * 110 + [1] * 5)
    configs = torch.tensor([0] * 100 + [1] * 10 + [0] * 5)
    keep = label_config_balanced_keep(labels, configs, max_per_label=20, seed=7)
    kept_label0 = configs[keep & labels.eq(0)]
    assert len(kept_label0) == 20
    assert torch.bincount(kept_label0, minlength=2).tolist() == [10, 10]
    assert int((keep & labels.eq(1)).sum()) == 5


def test_global_archive_budget_is_exact_and_preserves_rare_labels():
    labels = torch.tensor([0] * 100 + [1] * 10 + [2] * 2)
    configs = torch.tensor([0] * 50 + [1] * 50 + [0] * 10 + [1, 2])
    keep = archive_budget_balanced_keep(labels, configs, budget=40, seed=3)
    assert int(keep.sum()) == 40
    assert int((keep & labels.eq(2)).sum()) == 2
    assert set(configs[keep & labels.eq(0)].tolist()) == {0, 1}


def test_phase_b_policy_derives_non_conflicting_retrieval_limits():
    policy = PhaseBPolicy(evidence_budget=64)
    assert policy.max_per_window == 4
    assert policy.max_per_label == 12
    assert policy.max_per_window <= policy.max_per_label <= policy.evidence_budget
    assert policy.topk_per_subspace(16) == 8
    assert policy.topk_per_subspace(32) == 4


def test_explicit_cuda_request_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="pass --device cpu explicitly"):
        resolve_device("cuda")
    assert resolve_device("cpu") == torch.device("cpu")


def test_adaptation_episode_removes_candidate_background_and_equalizes_support():
    n_labels, windows_per_label = 3, 4
    y = torch.arange(n_labels).repeat_interleave(windows_per_label)
    n = len(y)
    window = torch.arange(n)
    patch = {
        "Z": torch.randn(n, 8), "y": y, "subj": torch.arange(n),
        "cfg": torch.zeros(n, dtype=torch.long), "sensor": torch.zeros(n, dtype=torch.long),
        "window": window, "event": window, "event_verified": torch.zeros(n, dtype=torch.bool),
        "time": torch.ones(n), "duration": torch.ones(n),
        "resolution": torch.zeros(n, dtype=torch.long), "ordinal": torch.zeros(n, dtype=torch.long),
    }
    bank = {
        "Z": torch.randn(n, 8), "y": y, "subj": torch.arange(n),
        "cfg": torch.zeros(n, dtype=torch.long), "event": window,
        "event_verified": torch.zeros(n, dtype=torch.bool), "patch": patch,
    }
    query = PatchTable(bank).gather_queries(torch.tensor([0, 4, 8]), "cpu")
    view = build_episode_memory_view(
        patch, torch.arange(n), query, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]),
        support_count=2, episode_type="same_subject_enrollment", label_mode="random_alias",
        rng=np.random.default_rng(3),
    )
    assert view.support_units_per_candidate.tolist() == [2, 2, 2]
    for label in range(n_labels):
        candidate_rows = patch["y"].eq(label)
        assert torch.equal(
            view.allowed[0, 0, candidate_rows], view.support_mask[candidate_rows]
        )
    with pytest.raises(ValueError, match="unanswerable"):
        build_episode_memory_view(
            patch, torch.arange(n), query, torch.tensor([0, 1, 2]),
            torch.tensor([0, 1, 2]), support_count=0,
            episode_type="semantic_zero_support", label_mode="random_alias",
            rng=np.random.default_rng(3),
        )

    absent_query = PatchTable(bank).gather_queries(torch.tensor([0]), "cpu")
    absent = build_episode_memory_view(
        patch, torch.arange(n), absent_query, torch.tensor([0]), torch.tensor([1, 2]),
        support_count=1, episode_type="ordinary_few_support", label_mode="coherent",
        rng=np.random.default_rng(4), truth_present=False,
    )
    assert not bool(absent.allowed[..., patch["y"].eq(0)].any())
    for label in (1, 2):
        label_rows = patch["y"].eq(label)
        assert torch.equal(
            absent.allowed[0, 0, label_rows], absent.support_mask[label_rows]
        )
    with pytest.raises(ValueError, match="cannot include"):
        build_episode_memory_view(
            patch, torch.arange(n), absent_query, torch.tensor([0]), torch.tensor([0, 1]),
            support_count=1, episode_type="ordinary_few_support", label_mode="coherent",
            rng=np.random.default_rng(4), truth_present=False,
        )


def test_episode_curriculum_is_exactly_balanced_and_soft_tau_is_bounded():
    curriculum = EpisodeCurriculum(np.random.default_rng(4))
    specs = [curriculum.sample() for _ in range(40)]
    types = [spec.episode_type for spec in specs]
    physical_views = [spec.physical_view_mode for spec in specs]
    assert {name: types.count(name) for name in set(types)} == {
        name: 10 for name in set(types)
    }
    assert {name: physical_views.count(name) for name in PHYSICAL_VIEW_MODES} == {
        name: 20 for name in PHYSICAL_VIEW_MODES
    }
    assert soft_retrieval_temperature(0) == pytest.approx(0.20)
    assert soft_retrieval_temperature(500) == pytest.approx(0.07)
    assert soft_retrieval_temperature(5000) == pytest.approx(0.07)


def test_episode_view_specs_clean_is_exact_identity_and_augmented_is_not():
    bank = _bank()
    query = PatchTable(bank).gather_queries(torch.tensor([2]), "cpu")
    support_mask = torch.zeros(8, dtype=torch.bool)
    support_mask[0] = True
    view = EpisodeMemoryView(
        allowed=torch.ones(1, query.Z.shape[1], 8, dtype=torch.bool),
        support_mask=support_mask,
        support_candidate=torch.zeros(8, dtype=torch.long),
        candidate_ids=torch.tensor([0, 1]),
        query_label=torch.tensor([1]),
        support_units_per_candidate=torch.tensor([1, 0]),
        episode_type="same_subject_enrollment",
        label_mode="coherent",
    )
    rows = torch.arange(8)
    clean_query, clean_support = _episode_view_specs(
        query, view, rows, bank["patch"], np.random.default_rng(1),
        physical_view_mode="clean",
    )
    assert all(spec == PatchViewSpec() for spec in clean_query + clean_support)

    augmented_query, augmented_support = _episode_view_specs(
        query, view, rows, bank["patch"], np.random.default_rng(1),
        physical_view_mode="augmented",
    )
    assert all(spec.subject_style is not None for spec in augmented_query + augmented_support)
    assert all(spec.generic_seed is not None for spec in augmented_query + augmented_support)
    with pytest.raises(ValueError, match="physical_view_mode"):
        _episode_view_specs(
            query, view, rows, bank["patch"], np.random.default_rng(1),
            physical_view_mode="obsolete",
        )


def test_soft_backward_reaches_nonselected_memory_and_straight_through_is_hard_forward():
    torch.manual_seed(5)
    retriever = PatchSubspaceRetriever(8, n_subspaces=2, subspace_dim=4)
    query = F.normalize(torch.randn(1, 1, 8), dim=-1).requires_grad_()
    memory = F.normalize(torch.randn(4, 8), dim=-1).detach().requires_grad_()
    allowed = torch.ones(1, 1, 4, dtype=torch.bool)
    candidate_weights = torch.tensor([
        [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0],
    ])
    prior = torch.full((4,), -np.log(4.0))
    soft = retriever.soft_candidate_logits(
        query, memory, allowed, candidate_weights, prior, tau=0.2,
        selected_index=torch.tensor([[[[0], [0]]]]),
    )
    hard = torch.tensor([[3.0, -1.0]], requires_grad=True)
    straight_through = hard + soft.logits - soft.logits.detach()
    assert torch.equal(straight_through.detach(), hard.detach())
    F.cross_entropy(straight_through, torch.tensor([1])).backward()
    assert memory.grad is not None
    assert float(memory.grad[1:].abs().sum()) > 0
    assert query.grad is not None and float(query.grad.abs().sum()) > 0


def test_subject_style_preserves_gravity_scale_shape_and_masked_slots():
    rate = 50.0
    t = np.arange(300) / rate
    data = np.zeros((300, 6), dtype=np.float32)
    data[:, 0] = 0.2 * np.sin(2 * np.pi * 1.5 * t)
    data[:, 2] = 1.0
    before_mean = data[:, :3].mean(0)
    styled = apply_subject_style(
        data, rate, [True, True, True, False, False, False],
        SubjectStyle(1.08, 1.1, 0.9, 0.2),
    )
    assert styled.shape == data.shape
    assert np.isfinite(styled).all()
    assert np.linalg.norm(styled[:, :3].mean(0) - before_mean) < 0.05
    assert np.count_nonzero(styled[:, 3:]) == 0


def test_balanced_soft_prior_has_equal_label_mass():
    bank = _bank()
    prior = balanced_memory_log_prior(bank["patch"], torch.arange(8), "cpu").exp()
    masses = [float(prior[bank["patch"]["y"].eq(label)].sum()) for label in (0, 1, 2)]
    assert max(masses) - min(masses) < 1e-6


def test_runtime_enrollment_appends_equal_support_without_mutating_base_bank():
    bank = _bank()
    original_patch_count = len(bank["patch"]["Z"])
    encoded = {
        "patch_Z": torch.randn(8, 8),
        "patch_window": torch.arange(4).repeat_interleave(2),
        "patch_time": torch.tensor([0.5, 1.5] * 4),
        "patch_duration": torch.ones(8),
        "patch_resolution": torch.tensor([0, 1] * 4),
    }
    canonical = F.normalize(torch.randn(3, 6), dim=-1)
    candidates = F.normalize(torch.randn(2, 6), dim=-1)
    memory = build_enrollment_memory(
        bank, torch.arange(8), F.normalize(bank["patch"]["Z"].float(), dim=-1),
        encoded, torch.tensor([0, 1]), torch.tensor([0, 1]), canonical, candidates,
    )
    assert len(bank["patch"]["Z"]) == original_patch_count
    assert len(memory.index_rows) == 12
    assert memory.support_mask.sum() == 4
    assert memory.support_units_per_candidate.tolist() == [1, 1]
    query = queries_from_encoded(
        encoded, torch.tensor([2, 3]), "cpu", sensor_id=memory.runtime_sensor_id
    )
    view = memory.episode_view(query, torch.tensor([0, 1]), label_mode="coherent")
    assert view.allowed.all()
    assert torch.equal(view.query_label, memory.candidate_ids)
    assert bool(query.sensor.eq(memory.runtime_sensor_id).all())

    zero = build_enrollment_memory(
        bank, torch.arange(8), F.normalize(bank["patch"]["Z"].float(), dim=-1),
        encoded, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
        canonical, candidates,
    )
    assert len(zero.index_rows) == 8
    assert zero.support_units_per_candidate.tolist() == [0, 0]


def test_same_subject_enrollment_excludes_the_whole_support_execution():
    labels = np.asarray(["walk"] * 6 + ["sit"] * 6, dtype=object)
    subjects = np.asarray(["s1"] * 12, dtype=object)
    executions = np.asarray(
        ["walk_a"] * 3 + ["walk_b"] * 3 + ["sit_a"] * 3 + ["sit_b"] * 3,
        dtype=object,
    )
    support, position, query = _support_and_query_rows(
        labels, subjects, executions, "s1", ["walk", "sit"],
        support_count=1, mode="same_subject", rng=np.random.default_rng(5),
    )
    assert position.tolist() == [0, 1]
    assert not set(executions[support]).intersection(executions[query])
    assert set(labels[query]) == {"walk", "sit"}


def test_paired_enrollment_plan_uses_nested_support_and_fixed_queries():
    labels, subjects, executions = [], [], []
    for subject in ("s1", "s2"):
        for label in ("walk", "sit"):
            for execution in range(3):
                labels.extend([label, label])
                subjects.extend([subject, subject])
                executions.extend([f"{subject}:{label}:{execution}"] * 2)
    plans, coverage = build_paired_enrollment_plans(
        np.asarray(labels, dtype=object),
        np.asarray(subjects, dtype=object),
        np.asarray(executions, dtype=object),
        ["walk", "sit"],
        requested_support=[0, 1, 2, 4], mode="same_subject", seed=7,
    )
    assert coverage["support_ceiling"] == 2
    assert coverage["subjects"] == 2
    for plan in plans:
        assert all(len(rows) == 2 for rows in plan.support_rows)
        support_executions = {
            executions[row] for rows in plan.support_rows for row in rows
        }
        query_executions = {executions[row] for row in plan.query_rows}
        assert support_executions.isdisjoint(query_executions)
        assert all(rows[:1] == (rows[0],) for rows in plan.support_rows)


def test_paired_enrollment_rejects_window_level_execution_ids():
    labels = np.asarray(["walk", "sit"] * 4, dtype=object)
    subjects = np.asarray(["s1"] * len(labels), dtype=object)
    executions = np.asarray([f"window_{index}" for index in range(len(labels))], dtype=object)
    plans, coverage = build_paired_enrollment_plans(
        labels, subjects, executions, ["walk", "sit"],
        requested_support=[0, 1, 2], mode="same_subject", seed=1,
    )
    assert plans == []
    assert coverage["status"] == "unverified_window_level_execution_ids"


def test_enrollment_summary_uses_paired_subject_differences():
    records = {
        "s1": {
            "f1_macro": 80.0, "identity_f1_macro": 60.0,
            "support_removed_f1_macro": 50.0,
            "support_label_shuffled_f1_macro": 40.0,
            "prototype_f1_macro": 70.0, "ridge_head_f1_macro": 75.0,
        },
        "s2": {
            "f1_macro": 40.0, "identity_f1_macro": 50.0,
            "support_removed_f1_macro": 45.0,
            "support_label_shuffled_f1_macro": 30.0,
            "prototype_f1_macro": 35.0, "ridge_head_f1_macro": 45.0,
        },
    }
    summary = paired_subject_summary(records, seed=3, samples=200)
    assert summary["learned_f1_macro"] == 60.0
    assert summary["adaptation_gain_over_identity"] == 5.0
    assert summary["gain_over_support_removed"] == 12.5
    assert summary["adaptation_gain_over_identity_ci95"] == [-10.0, 20.0]
    assert summary["independent_unit"] == "subject"


def test_gradient_telemetry_is_zero_before_tokenizer_finetuning_activates():
    module = torch.nn.Linear(3, 2)
    norm = parameter_gradient_norm(module.parameters(), torch.device("cpu"))
    assert norm.shape == ()
    assert float(norm) == 0.0


def test_source_patch_encoder_recovers_rows_once_and_preserves_gradients(tmp_path, monkeypatch):
    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.use_duration_embedding = False
            self.eval_resolution_pair = (0.5, 1.0)
            self.min_resolution_ratio = 1.75
            self.text_conditioning = "per_channel"

        def forward(self, patches, sampling_rate_hz, patch_len_samples, channel_texts, positions,
                    patch_padding_mask=None, **kwargs):
            scalar = patches.mean(dim=(2, 3)) * self.scale
            per_patch = scalar.unsqueeze(-1).repeat(1, 1, 4)
            mask = patch_padding_mask.to(per_patch.dtype)
            pooled = (per_patch * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            return {"pooled": pooled, "per_patch": per_patch}

    data = np.arange(2 * 120 * 6, dtype=np.float32).reshape(2, 120, 6)
    np.save(tmp_path / "data.npy", data)
    ref = GridRef(
        dataset="wisdm", stream="phone_pocket", alignment="native", rate_hz=50.0,
        channels=("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
        mask=(True,) * 6, labels=("walking", "walking"), subjects=("a", "b"),
        event_ids=("a", "b"), event_ids_explicit=False, shape=data.shape, grid_dir=tmp_path,
    )
    monkeypatch.setattr("training.evidence.live_encoder.discover_grids", lambda _: [ref])
    encoder = TinyEncoder()
    encoded = encode_dataset_detailed(
        encoder, data, ["x"] * 6, torch.device("cpu"), 50.0,
        channel_mask=[True] * 6,
    )
    parent = encoded["patch_window"].long()
    counts = torch.bincount(parent, minlength=2)
    starts = counts.cumsum(0) - counts
    ordinal = torch.arange(len(parent)) - torch.repeat_interleave(starts, counts)
    bank = {
        "schema_version": 3, "d_model": 4,
        "source_alignment": "native",
        "Z": encoded["pooled"], "source_row": torch.arange(2),
        "cfg": torch.zeros(2, dtype=torch.long),
        "cfg_names": {0: ref.key}, "corpus": {"datasets": [ref.dataset]},
        "patch": {
            "Z": encoded["patch_Z"], "window": parent,
            "cfg": torch.zeros(len(parent), dtype=torch.long), "ordinal": ordinal,
            "time": encoded["patch_time"], "duration": encoded["patch_duration"],
            "resolution": encoded["patch_resolution"],
        },
    }
    source = SourcePatchEncoder(bank, "cpu", verify_fingerprint=False)
    rows = torch.tensor([0, len(parent) - 1, 0])
    live = source.encode_patch_rows(rows, encoder, requires_grad=True)
    live.square().mean().backward()
    assert live.shape == (3, 4)
    assert torch.equal(live[0], live[2])
    assert encoder.scale.grad is not None and float(encoder.scale.grad.abs()) > 0

    encoder.zero_grad(set_to_none=True)
    specs = [PatchViewSpec(None, 17), PatchViewSpec(None, 19), PatchViewSpec(None, 17)]
    augmented = source.encode_patch_rows_with_views(
        rows, specs, encoder, requires_grad=True
    )
    repeated = source.encode_patch_rows_with_views(
        rows, specs, encoder, requires_grad=True
    )
    assert torch.equal(augmented, repeated)
    assert torch.equal(augmented[0], augmented[2])
    augmented.square().mean().backward()
    assert encoder.scale.grad is not None and float(encoder.scale.grad.abs()) > 0


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
        bank["patch"], rows, query, torch.tensor([1]),
        truth_present=True, true_support=0, config_mode="any",
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
        bank["patch"], rows, query, torch.tensor([0]),
        truth_present=True, true_support=None,
        config_mode="cross", rng=np.random.default_rng(4),
    )
    assert cross.any()  # each sensor can use configurations belonging to the other query sensor
    absent = build_allowed_mask(
        bank["patch"], rows, query, torch.tensor([0]),
        truth_present=True, true_support=None,
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
    retrieval_head = torch.tensor([[0, 1, 2, 3, 0]] * 2)
    with torch.no_grad():
        logits = dec(**inputs, ev_retrieval_head=retrieval_head)
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


def test_retrieval_subspace_identity_reaches_evidence_attention():
    torch.manual_seed(14)
    dec = EvidenceDecoder(DecoderConfig(
        d_model=16, text_dim=12, n_heads=4, n_layers=1,
        candidate_tokens=True, candidate_layers=1, structural_metadata=True,
        n_retrieval_heads=4,
    )).train()
    with torch.no_grad():
        dec.refiner[-1].weight.normal_(0, 0.02)
    heads = torch.tensor([[0, 1, 2, 3, 0]] * 2)
    dec(**_decoder_inputs(), ev_retrieval_head=heads).sum().backward()
    assert dec.retrieval_head_emb.weight.grad is not None
    assert float(dec.retrieval_head_emb.weight.grad[1:].abs().sum()) > 0


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


def test_confidence_quality_metrics_reward_correct_ranking_and_calibration():
    target = np.asarray([0, 0, 1, 1], dtype=bool)
    good = np.asarray([0.05, 0.2, 0.8, 0.95])
    reversed_score = good[::-1]
    assert binary_auprc(good, target) == pytest.approx(1.0)
    assert binary_auprc(reversed_score, target) < 0.5
    assert expected_calibration_error(good, target) < 0.2


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
        torch.tensor([0]), torch.tensor([0, 1, 2]), text, text,
        truth_present=True, true_support=0,
        config_mode="any", rng=np.random.default_rng(4),
        policy=PhaseBPolicy(evidence_budget=8),
        query_window_mask=torch.ones(4, dtype=torch.bool),
    )
    loss = F.cross_entropy(logits, torch.tensor([0]))
    loss.backward()
    assert torch.isfinite(logits).all()
    assert retriever.proj.grad is not None
    assert {"evidence_index", "evidence_sensor", "evidence_head", "evidence_query_patch"} <= set(aux)


def test_live_patch_episode_backpropagates_into_the_online_tokenizer():
    class TinyOnlineTokenizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Parameter(torch.eye(8) + 0.01 * torch.randn(8, 8))

    class FakeLiveSource:
        def __init__(self, bank):
            self.bank = bank

        def reencode_query(self, query, encoder, *, requires_grad):
            assert requires_grad
            z = (query.Z @ encoder.projection) * query.mask.unsqueeze(-1)
            return replace(query, Z=z)

        def reencode_evidence(self, evidence, encoder, *, requires_grad):
            assert requires_grad
            base = self.bank["patch"]["Z"].float()[evidence.index]
            return (base @ encoder.projection) * evidence.mask.unsqueeze(-1)

    torch.manual_seed(21)
    bank = _bank()
    table = PatchTable(bank)
    retriever = PatchSubspaceRetriever(8, 2, 4)
    decoder = EvidenceDecoder(DecoderConfig(
        d_model=8, text_dim=6, n_heads=2, n_layers=1,
        candidate_tokens=True, candidate_layers=1, structural_metadata=True,
    ))
    # Match the real fine-tuning phase, which begins only after the identity heads have opened.
    with torch.no_grad():
        decoder.refiner[-1].weight.normal_(0, 0.02)
        decoder.candidate_refiner.weight.normal_(0, 0.02)
        for block in decoder.blocks:
            block.ls1.fill_(1.0)
            block.ls2.fill_(1.0)
    tokenizer = TinyOnlineTokenizer()
    index_rows = torch.arange(len(bank["patch"]["Z"]))
    memory_index = retriever.build_index(F.normalize(bank["patch"]["Z"].float(), dim=-1))
    text = F.normalize(torch.randn(3, 6), dim=-1)
    logits, aux = run_patch_episode(
        decoder, retriever, table, bank, index_rows, memory_index,
        torch.tensor([2]), torch.tensor([0, 1, 2]), text, text,
        truth_present=True, true_support=0, config_mode="any",
        rng=np.random.default_rng(8), policy=PhaseBPolicy(evidence_budget=8),
        query_window_mask=torch.ones(4, dtype=torch.bool),
        live_source=FakeLiveSource(bank), live_encoder=tokenizer,
        live_requires_grad=True,
    )
    F.cross_entropy(logits, torch.tensor([1])).backward()
    assert aux["retrieval_scores"].requires_grad
    assert tokenizer.projection.grad is not None
    assert float(tokenizer.projection.grad.abs().sum()) > 0


def test_adaptation_straight_through_path_backpropagates_into_online_tokenizer():
    class TinyOnlineTokenizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Parameter(torch.eye(8) + 0.01 * torch.randn(8, 8))

    class FakeAdaptationSource:
        def __init__(self, bank):
            self.bank = bank

        def reencode_query_views(self, query, encoder, specs, *, requires_grad):
            value = query.Z @ encoder.projection
            if not requires_grad:
                value = value.detach()
            return replace(query, Z=value * query.mask.unsqueeze(-1))

        def encode_patch_rows_with_views(self, rows, specs, encoder, *, requires_grad):
            value = self.bank["patch"]["Z"].float()[rows] @ encoder.projection
            return value if requires_grad else value.detach()

        def reencode_evidence(self, evidence, encoder, *, requires_grad):
            value = self.bank["patch"]["Z"].float()[evidence.index] @ encoder.projection
            return value * evidence.mask.unsqueeze(-1)

    torch.manual_seed(22)
    bank = _bank()
    table = PatchTable(bank)
    query = table.gather_queries(torch.tensor([2]), "cpu")
    rows = torch.arange(8)
    candidates = torch.tensor([1, 2])
    view = build_episode_memory_view(
        bank["patch"], rows, query, torch.tensor([1]), candidates,
        support_count=0, episode_type="semantic_zero_support", label_mode="coherent",
        rng=np.random.default_rng(4),
    )
    retriever = PatchSubspaceRetriever(8, 2, 4)
    decoder = EvidenceDecoder(DecoderConfig(
        d_model=8, text_dim=6, n_heads=2, n_layers=1,
        candidate_tokens=True, candidate_layers=1,
        structural_metadata=True, support_role=True,
    ))
    with torch.no_grad():
        decoder.refiner[-1].weight.normal_(0, 0.02)
        decoder.candidate_refiner.weight.normal_(0, 0.02)
        for block in decoder.blocks:
            block.ls1.fill_(1.0)
            block.ls2.fill_(1.0)
    tokenizer = TinyOnlineTokenizer()
    selector_z = F.normalize(bank["patch"]["Z"].float(), dim=-1)
    memory_index = retriever.build_index(selector_z)
    text = F.normalize(torch.tensor([
        [1.0, 0.2, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.6, 0.8, 0.0, 0.0, 0.0, 0.0],
    ]), dim=-1)
    logits, aux = decode_adaptation_episode(
        decoder, retriever, bank, rows, selector_z, memory_index,
        query, view, text, text[candidates],
        balanced_memory_log_prior(bank["patch"], rows, "cpu"),
        policy=PhaseBPolicy(evidence_budget=8), soft_tau=0.2,
        rng=np.random.default_rng(9),
        live_source=FakeAdaptationSource(bank),
        selector_encoder=tokenizer, online_encoder=tokenizer,
        online_requires_grad=True,
    )
    assert torch.equal(logits.detach(), aux["hard_logits"].detach())
    F.cross_entropy(logits, torch.tensor([0])).backward()
    assert tokenizer.projection.grad is not None
    assert float(tokenizer.projection.grad.abs().sum()) > 0


def test_adaptation_identity_control_matches_untrained_decoder_forward():
    torch.manual_seed(23)
    bank = _bank()
    rows = torch.arange(len(bank["patch"]["Z"]))
    query = PatchTable(bank).gather_queries(torch.tensor([2]), "cpu")
    candidates = torch.tensor([1, 2])
    view = build_episode_memory_view(
        bank["patch"], rows, query, torch.tensor([1]), candidates,
        support_count=0, episode_type="semantic_zero_support", label_mode="coherent",
        rng=np.random.default_rng(7),
    )
    retriever = PatchSubspaceRetriever(8, 2, 4)
    decoder = EvidenceDecoder(DecoderConfig(
        d_model=8, text_dim=6, n_heads=2, n_layers=1,
        candidate_tokens=True, candidate_layers=1,
        structural_metadata=True, support_role=True,
    )).eval()
    selector_z = F.normalize(bank["patch"]["Z"].float(), dim=-1)
    text = F.normalize(torch.randn(3, 6), dim=-1)
    _, aux = decode_adaptation_episode(
        decoder, retriever, bank, rows, selector_z,
        retriever.build_index(selector_z), query, view, text, text[candidates],
        balanced_memory_log_prior(bank["patch"], rows, "cpu"),
        policy=PhaseBPolicy(evidence_budget=8), soft_tau=0.2,
        rng=np.random.default_rng(8),
    )
    assert torch.allclose(aux["hard_logits"], aux["identity_logits"], atol=1e-6)


def test_no_grad_adaptation_skips_the_soft_backward_route(monkeypatch):
    torch.manual_seed(24)
    bank = _bank()
    rows = torch.arange(len(bank["patch"]["Z"]))
    query = PatchTable(bank).gather_queries(torch.tensor([2]), "cpu")
    candidates = torch.tensor([1, 2])
    view = build_episode_memory_view(
        bank["patch"], rows, query, torch.tensor([1]), candidates,
        support_count=0, episode_type="semantic_zero_support", label_mode="coherent",
        rng=np.random.default_rng(7),
    )
    retriever = PatchSubspaceRetriever(8, 2, 4)
    decoder = EvidenceDecoder(DecoderConfig(
        d_model=8, text_dim=6, n_heads=2, n_layers=1,
        candidate_tokens=True, candidate_layers=1,
        structural_metadata=True, support_role=True,
    )).eval()
    selector_z = F.normalize(bank["patch"]["Z"].float(), dim=-1)
    text = F.normalize(torch.randn(3, 6), dim=-1)

    def forbidden_soft_route(*args, **kwargs):
        raise AssertionError("evaluation called the backward-only soft route")

    monkeypatch.setattr(retriever, "soft_candidate_logits", forbidden_soft_route)
    with torch.no_grad():
        logits, aux = decode_adaptation_episode(
            decoder, retriever, bank, rows, selector_z,
            retriever.build_index(selector_z), query, view, text, text[candidates],
            balanced_memory_log_prior(bank["patch"], rows, "cpu"),
            policy=PhaseBPolicy(evidence_budget=8), soft_tau=0.2,
            rng=np.random.default_rng(8),
        )
    assert torch.equal(logits, aux["hard_logits"])
    assert "soft_logits" not in aux


def test_clean_frozen_adaptation_reuses_stored_vectors_without_live_encoding():
    class ForbiddenLiveSource:
        def __getattr__(self, name):
            raise AssertionError(f"clean frozen path called {name}")

    bank = _bank()
    rows = torch.arange(len(bank["patch"]["Z"]))
    query = PatchTable(bank).gather_queries(torch.tensor([2]), "cpu")
    view = build_episode_memory_view(
        bank["patch"], rows, query, torch.tensor([1]), torch.tensor([1, 2]),
        support_count=0, episode_type="semantic_zero_support", label_mode="coherent",
        rng=np.random.default_rng(7),
    )
    retriever = PatchSubspaceRetriever(8, 2, 4)
    selector_z = F.normalize(bank["patch"]["Z"].float(), dim=-1)
    memory_index = retriever.build_index(selector_z)
    selector_query, online_query, memory, episode_index = prepare_adaptation_views(
        query, view, bank, rows, selector_z, memory_index, retriever,
        rng=np.random.default_rng(9), live_source=ForbiddenLiveSource(),
        selector_encoder=object(), online_requires_grad=False, physical_view_mode="clean",
        reuse_stored_clean=True,
    )
    assert selector_query is query and online_query is query
    assert memory is selector_z and episode_index is memory_index


def test_clean_ema_adaptation_reencodes_instead_of_reusing_stale_bank_vectors():
    class CountingLiveSource:
        def __init__(self):
            self.query_calls = 0

        def reencode_query_views(self, query, encoder, specs, *, requires_grad):
            self.query_calls += 1
            return replace(query, Z=query.Z + 0.25)

        def encode_patch_rows_with_views(self, rows, specs, encoder, *, requires_grad):
            return torch.zeros(len(rows), 8)

    bank = _bank()
    rows = torch.arange(len(bank["patch"]["Z"]))
    query = PatchTable(bank).gather_queries(torch.tensor([2]), "cpu")
    view = build_episode_memory_view(
        bank["patch"], rows, query, torch.tensor([1]), torch.tensor([1, 2]),
        support_count=0, episode_type="semantic_zero_support", label_mode="coherent",
        rng=np.random.default_rng(7),
    )
    retriever = PatchSubspaceRetriever(8, 2, 4)
    selector_z = F.normalize(bank["patch"]["Z"].float(), dim=-1)
    source = CountingLiveSource()
    selector_query, online_query, _, _ = prepare_adaptation_views(
        query, view, bank, rows, selector_z, retriever.build_index(selector_z), retriever,
        rng=np.random.default_rng(9), live_source=source, selector_encoder=object(),
        online_requires_grad=False, physical_view_mode="clean", reuse_stored_clean=False,
    )
    assert source.query_calls == 1
    assert selector_query is online_query
    assert not torch.equal(selector_query.Z, query.Z)


def test_nontruth_labels_follow_the_shared_active_index_policy():
    bank = _bank()
    query = PatchTable(bank).gather_queries(torch.tensor([2]), "cpu")
    rows = torch.arange(8)
    # Candidate labels do not constrain the physical memory index.
    allowed = build_allowed_mask(
        bank["patch"], rows, query, torch.tensor([1]),
        truth_present=True, true_support=0,
        config_mode="any", rng=np.random.default_rng(3),
    )
    retained = rows[allowed[0, 0] & bank["patch"]["y"].eq(0)]
    assert len(torch.unique(bank["patch"]["window"][retained])) == 2


def _single_stream_bank():
    """One label recorded on ONE stream by two different people, plus a distractor label."""
    window = torch.arange(4).repeat_interleave(2)
    y = torch.tensor([0, 0, 1, 1])
    subj = torch.tensor([0, 1, 0, 1])
    cfg = torch.zeros(4, dtype=torch.long)          # every window on the same stream
    event = torch.arange(4)
    verified = torch.zeros(4, dtype=torch.bool)
    patch = {
        "Z": torch.randn(8, 8).half(), "y": y[window], "subj": subj[window],
        "cfg": cfg[window], "sensor": cfg[window], "window": window,
        "event": event[window], "event_verified": verified[window],
        "time": torch.tensor([0.5, 1.5] * 4), "duration": torch.ones(8),
        "resolution": torch.tensor([0, 1] * 4), "ordinal": torch.tensor([0, 1] * 4),
    }
    return {"schema_version": 3, "Z": torch.randn(4, 8).half(), "y": y, "subj": subj,
            "cfg": cfg, "event": event, "event_verified": verified,
            "source_row": torch.arange(4), "patch": patch}


def test_cross_subject_support_works_for_a_label_recorded_on_only_one_stream():
    # 35 of the 93 corpus labels exist on a single stream. Requiring acquisition disjointness would
    # drop every one of them from this regime; only the person has to differ.
    bank = _single_stream_bank()
    query = PatchTable(bank).gather_queries(torch.tensor([0]), "cpu")   # label 0, subject 0
    view = build_episode_memory_view(
        bank["patch"], torch.arange(8), query, torch.tensor([0]), torch.tensor([0]),
        support_count=1, episode_type="cross_subject_few_support",
        label_mode="coherent", rng=np.random.default_rng(0),
    )
    support = view.support_rows
    assert len(support), "a single-stream label must still be able to supply cross-subject support"
    assert set(bank["patch"]["subj"][support].tolist()) == {1}, "support must be another person"
    assert set(bank["patch"]["cfg"][support].tolist()) == {0}, "same stream is allowed, not excluded"


def test_episode_composition_names_what_the_support_actually_was():
    bank = _single_stream_bank()
    query = PatchTable(bank).gather_queries(torch.tensor([0]), "cpu")
    view = build_episode_memory_view(
        bank["patch"], torch.arange(8), query, torch.tensor([0]), torch.tensor([0]),
        support_count=1, episode_type="cross_subject_few_support",
        label_mode="coherent", rng=np.random.default_rng(0),
    )
    report = describe_episode_composition(bank["patch"], torch.arange(8), query, view)
    assert report["enrolled_examples"] == 1
    assert report["performed_by_a_different_person"] == 1
    assert report["performed_by_the_same_person"] == 0
    assert report["from_the_query_s_own_stream"] == 1
    assert report["from_a_second_stream_worn_simultaneously"] == 0
    assert report["synthetic_persona"].startswith("a different persona")


def test_simultaneous_stream_pairs_separate_synchronous_from_independent_streams():
    # Streams 0/1 share event 10 (captured together); stream 2 covers the same activity alone.
    cfg = torch.tensor([0, 1, 2, 2])
    event = torch.tensor([10, 10, 20, 30])
    assert simultaneous_stream_pairs(cfg, event) == {frozenset((0, 1))}


def test_composition_separates_paired_rig_support_from_independent_stream_support():
    # Streams 0 and 1 are a worn-together rig (event 10 spans both). The query is a LATER execution
    # on stream 0, so its support on stream 1 is a different session on the paired device.
    window = torch.arange(4).repeat_interleave(2)
    y = torch.zeros(4, dtype=torch.long)
    subj = torch.tensor([0, 0, 1, 1])
    cfg = torch.tensor([0, 1, 1, 0])
    event = torch.tensor([10, 10, 20, 30])
    verified = torch.tensor([True, True, False, False])
    patch = {
        "Z": torch.randn(8, 8).half(), "y": y[window], "subj": subj[window],
        "cfg": cfg[window], "sensor": cfg[window], "window": window,
        "event": event[window], "event_verified": verified[window],
        "time": torch.tensor([0.5, 1.5] * 4), "duration": torch.ones(8),
        "resolution": torch.tensor([0, 1] * 4), "ordinal": torch.tensor([0, 1] * 4),
    }
    bank = {"schema_version": 3, "Z": torch.randn(4, 8).half(), "y": y, "subj": subj,
            "cfg": cfg, "event": event, "event_verified": verified,
            "source_row": torch.arange(4), "patch": patch}
    pairs = simultaneous_stream_pairs(bank["cfg"], bank["event"])
    assert pairs == {frozenset((0, 1))}
    query = PatchTable(bank).gather_queries(torch.tensor([3]), "cpu")   # stream 0, subject 1
    view = build_episode_memory_view(
        bank["patch"], torch.arange(8), query, torch.tensor([0]), torch.tensor([0]),
        support_count=1, episode_type="ordinary_few_support",
        label_mode="coherent", rng=np.random.default_rng(1),
    )
    report = describe_episode_composition(bank["patch"], torch.arange(8), query, view, pairs)
    assert report["enrolled_examples"] == 1
    assert report["from_a_second_stream_worn_simultaneously"] == 1
    assert report["from_the_query_s_own_stream"] == 0
    assert report["synthetic_persona"] == "none applied"


def test_composition_reports_both_own_and_paired_streams_in_one_support_event():
    window = torch.arange(4).repeat_interleave(2)
    y = torch.zeros(4, dtype=torch.long)
    subj = torch.tensor([0, 0, 1, 1])
    cfg = torch.tensor([0, 1, 0, 1])
    event = torch.tensor([10, 10, 20, 20])
    verified = torch.ones(4, dtype=torch.bool)
    patch = {
        "Z": torch.randn(8, 8).half(), "y": y[window], "subj": subj[window],
        "cfg": cfg[window], "sensor": cfg[window], "window": window,
        "event": event[window], "event_verified": verified[window],
        "time": torch.tensor([0.5, 1.5] * 4), "duration": torch.ones(8),
        "resolution": torch.tensor([0, 1] * 4), "ordinal": torch.tensor([0, 1] * 4),
    }
    bank = {
        "schema_version": 3, "Z": torch.randn(4, 8).half(), "y": y, "subj": subj,
        "cfg": cfg, "event": event, "event_verified": verified,
        "source_row": torch.arange(4), "patch": patch,
    }
    pairs = simultaneous_stream_pairs(bank["cfg"], bank["event"])
    query = PatchTable(bank).gather_queries(torch.tensor([0]), "cpu")
    view = build_episode_memory_view(
        bank["patch"], torch.arange(8), query, torch.tensor([0]), torch.tensor([0]),
        support_count=1, episode_type="ordinary_few_support",
        label_mode="coherent", rng=np.random.default_rng(2),
    )
    report = describe_episode_composition(bank["patch"], torch.arange(8), query, view, pairs)
    assert report["enrolled_examples"] == 1
    assert report["from_the_query_s_own_stream"] == 1
    assert report["from_a_second_stream_worn_simultaneously"] == 1


def test_composition_does_not_attribute_distractor_support_to_unrelated_queries():
    bank = _single_stream_bank()
    query = PatchTable(bank).gather_queries(torch.tensor([0]), "cpu")
    view = build_episode_memory_view(
        bank["patch"], torch.arange(8), query, torch.tensor([0]), torch.tensor([0, 1]),
        support_count=1, episode_type="ordinary_few_support",
        label_mode="coherent", rng=np.random.default_rng(3),
    )
    report = describe_episode_composition(bank["patch"], torch.arange(8), query, view)
    assert report["enrolled_examples_without_matching_query"] == 1
    assert report["performed_by_a_different_person"] == 1
    assert report["from_the_query_s_own_stream"] == 1


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
