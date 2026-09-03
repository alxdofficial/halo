"""Tests for the Task-2 personal-normative ruler, its objective and scoring."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dataclasses import replace

from applications.motion_monitoring.data.compatibility import sensor_compatibility_key
from applications.motion_monitoring.task2.contracts import (
    BoundedExecution,
    ExecutionEpisode,
    collate_execution_episodes,
)
from applications.motion_monitoring.task2.episodes import (
    ExecutionRecord,
    Task2BatchBuilder,
    relation_between,
    validate_batch,
)
from applications.motion_monitoring.task2.losses import RulerLossConfig, ruler_loss
from applications.motion_monitoring.task2.model import ChangeRuler
from applications.motion_monitoring.task2.modifications import (
    MODIFICATIONS,
    NUISANCES,
    apply_modification,
    apply_nuisance,
)
from applications.motion_monitoring.task2.personal import personal_operating_point
from applications.motion_monitoring.task2.scoring import (
    personal_change_report,
    physical_change_report,
)
from applications.motion_monitoring.task2.smoke import (
    build_demo_batch,
    make_records,
    run_synthetic_smoke,
)
from applications.motion_monitoring.task2.training import train_step


KEY = sensor_compatibility_key(
    device="smartwatch",
    placement="wrist",
    channels=("acc_x", "acc_y", "acc_z"),
    gravity_state="present",
)
_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
OTHER_KEY = sensor_compatibility_key(
    device="smartphone",
    placement="pocket",
    channels=("acc_x", "acc_y", "acc_z"),
    gravity_state="present",
)


def _execution(name: str, *, subject: str = "s1", key=KEY, dataset: str = "unit", length: int = 6):
    values = torch.randn(length, 4, generator=torch.Generator().manual_seed(len(name)))
    edges = torch.arange(length + 1, dtype=torch.float32)
    return BoundedExecution(
        embeddings=values.float(),
        patch_intervals_sec=torch.stack((edges[:-1], edges[1:]), dim=-1),
        patch_mask=torch.ones(length, dtype=torch.bool),
        dataset=dataset,
        subject_id=subject,
        session_id=f"{subject}_session",
        execution_id=name,
        task_id="reach",
        sensor_config=key,
    )


def test_ruler_starts_at_the_transparent_cosine_floor() -> None:
    references = tuple(_execution(f"r{index}") for index in range(3))
    episode = ExecutionEpisode(accepted_references=references, query=_execution("q"), episode_kind="accepted_query")
    batch = collate_execution_episodes([episode])
    ruler = ChangeRuler(4, phase_bins=8)
    ruler.eval()
    with torch.no_grad():
        learned = ruler(batch).distances
    floor = personal_change_report(batch, None).raw_distance
    assert learned.tolist() == pytest.approx(floor.tolist(), abs=1e-5)


def test_loss_pushes_negatives_away_and_scales_the_margin_with_severity() -> None:
    references = tuple(_execution(f"r{index}") for index in range(2))
    episodes = [
        ExecutionEpisode(accepted_references=references, query=_execution("q"), episode_kind="accepted_query"),
        ExecutionEpisode(
            accepted_references=references,
            query=_execution("m"),
            episode_kind="modified_query",
            severity=1.0,
            modification_kind="retime",
        ),
    ]
    batch = collate_execution_episodes(episodes)

    class _Output:
        distances = torch.tensor([0.10, 0.30])
        phase_residuals = torch.zeros(2, 8)
        residual_vectors = torch.zeros(2, 8, 4)
        reference_phase = torch.zeros(2, 2, 8, 4)
        query_phase = torch.zeros(2, 8, 4)
        evidence_attention = torch.zeros(2, 8, 16)

    # positive 0.10, negative 0.30, margin 0.2 -> hinge exactly zero
    loss = ruler_loss(_Output(), batch, RulerLossConfig(margin=0.2, pull_weight=0.0))
    assert float(loss.ranking) == pytest.approx(0.0, abs=1e-6)
    assert loss.ranking_count == 1
    # a wider margin is not satisfied by the same distances
    wider = ruler_loss(_Output(), batch, RulerLossConfig(margin=0.4, pull_weight=0.0))
    assert float(wider.ranking) > 0.0
    # halving the declared severity halves the required margin
    half = collate_execution_episodes(
        [
            episodes[0],
            ExecutionEpisode(
                accepted_references=references,
                query=_execution("m"),
                episode_kind="modified_query",
                severity=0.5,
                modification_kind="retime",
            ),
        ]
    )
    assert float(ruler_loss(_Output(), half, RulerLossConfig(margin=0.4, pull_weight=0.0)).ranking) < float(
        wider.ranking
    )


def test_loss_never_pairs_across_reference_sets() -> None:
    first = tuple(_execution(f"a{index}") for index in range(2))
    second = tuple(_execution(f"b{index}", subject="s2") for index in range(2))
    episodes = [
        ExecutionEpisode(accepted_references=first, query=_execution("qa"), episode_kind="accepted_query"),
        ExecutionEpisode(
            accepted_references=second,
            query=_execution("qb", subject="s2"),
            episode_kind="accepted_query",
        ),
    ]
    batch = collate_execution_episodes(episodes)

    class _Output:
        distances = torch.tensor([0.1, 0.2])
        evidence_attention = torch.zeros(2, 8, 16)

    loss = ruler_loss(_Output(), batch, RulerLossConfig())
    assert loss.ranking_count == 0  # two positives, no negative anywhere


def test_batch_builder_refuses_to_cross_configurations_or_datasets() -> None:
    good = [
        ExecutionEpisode(
            accepted_references=(_execution("r0"), _execution("r1")),
            query=_execution("q"),
            episode_kind="accepted_query",
        ),
        ExecutionEpisode(
            accepted_references=(_execution("r0"), _execution("r1")),
            query=_execution("o", subject="s2"),
            episode_kind="other_subject_query",
        ),
    ]
    validate_batch(good)
    crossed = [
        good[0],
        ExecutionEpisode(
            accepted_references=(_execution("p0", key=OTHER_KEY), _execution("p1", key=OTHER_KEY)),
            query=_execution("p2", key=OTHER_KEY, subject="s2"),
            episode_kind="other_subject_query",
        ),
    ]
    with pytest.raises(ValueError, match="sensor compatibility keys"):
        validate_batch(crossed)
    other_dataset = [
        good[0],
        ExecutionEpisode(
            accepted_references=(_execution("d0", dataset="other"), _execution("d1", dataset="other")),
            query=_execution("d2", dataset="other", subject="s2"),
            episode_kind="other_subject_query",
        ),
    ]
    with pytest.raises(ValueError, match="cross source datasets"):
        validate_batch(other_dataset)
    with pytest.raises(ValueError, match="needs at least one negative"):
        validate_batch([good[0]])


def test_relation_between_uses_day_then_session() -> None:
    def record(name, *, day, session):
        execution = _execution(name)
        execution = type(execution)(
            embeddings=execution.embeddings,
            patch_intervals_sec=execution.patch_intervals_sec,
            patch_mask=execution.patch_mask,
            dataset=execution.dataset,
            subject_id=execution.subject_id,
            session_id=session,
            execution_id=name,
            task_id=execution.task_id,
            sensor_config=execution.sensor_config,
        )
        return ExecutionRecord(execution=execution, key=KEY, day=day)

    anchor = record("a", day="d1", session="s1")
    assert relation_between(anchor, record("b", day="d2", session="s2")) == "different_day"
    assert relation_between(anchor, record("c", day="d1", session="s2")) == "different_session"
    assert relation_between(anchor, record("d", day="d1", session="s1")) == "same_session"


@pytest.mark.parametrize("kind", sorted(MODIFICATIONS))
def test_modifications_change_the_signal_and_are_deterministic(kind: str) -> None:
    values = np.sin(np.linspace(0, 8 * np.pi, 200))[:, None] * np.ones((1, 6))
    kwargs = {"sampling_rate_hz": 100.0, "channels": _CHANNELS}
    first = apply_modification(values, kind=kind, severity=0.8, seed=7, **kwargs)
    second = apply_modification(values, kind=kind, severity=0.8, seed=7, **kwargs)
    assert np.allclose(first, second)
    assert first.shape[1] == values.shape[1]
    if first.shape == values.shape:
        assert not np.allclose(first, values)
    else:
        assert first.shape[0] != values.shape[0]
    with pytest.raises(ValueError):
        apply_modification(values, kind=kind, severity=0.0, seed=1, **kwargs)


@pytest.mark.parametrize("kind", sorted(NUISANCES))
def test_nuisances_preserve_channel_count_and_are_deterministic(kind: str) -> None:
    values = np.random.default_rng(0).normal(size=(200, 6))
    kwargs = {"sampling_rate_hz": 100.0, "channels": _CHANNELS}
    first = apply_nuisance(values, kind=kind, seed=3, **kwargs)
    assert np.allclose(first, apply_nuisance(values, kind=kind, seed=3, **kwargs))
    assert first.shape[1] == values.shape[1]


def test_tremor_frequency_uses_the_declared_sampling_rate() -> None:
    rate = 100.0
    values = np.zeros((1000, 6), dtype=np.float64)
    values[:, 2] = np.sin(2 * np.pi * np.arange(1000) / 100.0)
    changed = apply_modification(
        values,
        kind="added_tremor",
        severity=1.0,
        seed=17,
        sampling_rate_hz=rate,
        channels=_CHANNELS,
    )
    residual = changed[:, 0] - values[:, 0]
    frequencies = np.fft.rfftfreq(len(residual), d=1.0 / rate)
    peak = frequencies[1:][np.argmax(np.abs(np.fft.rfft(residual))[1:])]
    assert 4.0 <= peak <= 6.0


def test_personal_operating_point_is_reference_limited_below_four() -> None:
    features = torch.tensor([[0.1, 0.2], [0.12, 0.18]], dtype=torch.float64)
    point = personal_operating_point(features)
    assert point.reference_limited
    assert np.isnan(point.personal_limit95)
    three = torch.tensor([[0.1, 0.2], [0.12, 0.18], [0.11, 0.21]], dtype=torch.float64)
    assert personal_operating_point(three).reference_limited
    grown = torch.tensor([[0.1, 0.2], [0.12, 0.18], [0.11, 0.21], [0.10, 0.19]], dtype=torch.float64)
    full = personal_operating_point(grown)
    assert not full.reference_limited
    assert np.isfinite(full.personal_limit95) and full.personal_limit95 > 0


def test_scoring_reports_per_person_thresholds_for_both_arms() -> None:
    references = tuple(_execution(f"r{index}") for index in range(4))
    episode = ExecutionEpisode(
        accepted_references=references, query=_execution("q"), episode_kind="accepted_query"
    )
    batch = collate_execution_episodes([episode])
    floor = personal_change_report(batch, None)
    ruler = personal_change_report(batch, ChangeRuler(4, phase_bins=8).eval())
    for report in (floor, ruler):
        assert torch.isfinite(report.joint_deviation).all()
        assert torch.isfinite(report.personal_limit95).all()
        assert not bool(report.reference_limited.any())
    assert floor.joint_deviation.tolist() == pytest.approx(ruler.joint_deviation.tolist(), abs=1e-5)


def test_personal_scoring_uses_the_learned_query_refinement() -> None:
    references = tuple(_execution(f"r{index}") for index in range(4))
    episode = ExecutionEpisode(
        accepted_references=references, query=_execution("q"), episode_kind="accepted_query"
    )
    batch = collate_execution_episodes([episode])
    ruler = ChangeRuler(4, phase_bins=8).eval()
    baseline = personal_change_report(batch, ruler)
    baseline_reference, _ = ruler.reference_residuals(batch)
    with torch.no_grad():
        ruler.refinement.weight.normal_(mean=0.0, std=1.0)
        ruler.refinement.bias.fill_(0.5)
    changed = personal_change_report(batch, ruler)
    changed_reference, _ = ruler.reference_residuals(batch)
    assert not torch.allclose(baseline.raw_distance, changed.raw_distance)
    assert not torch.allclose(baseline_reference, changed_reference)


def test_learned_personal_scoring_is_invariant_to_batch_padding() -> None:
    first = ExecutionEpisode(
        accepted_references=tuple(_execution(f"a{index}", length=5) for index in range(4)),
        query=_execution("qa", length=6),
        episode_kind="accepted_query",
    )
    second = ExecutionEpisode(
        accepted_references=tuple(_execution(f"b{index}", length=10) for index in range(5)),
        query=_execution("qb", length=12),
        episode_kind="accepted_query",
    )
    ruler = ChangeRuler(4, phase_bins=8).eval()
    with torch.no_grad():
        ruler.refinement.weight.normal_(mean=0.0, std=0.1)
    alone = personal_change_report(collate_execution_episodes([first]), ruler)
    padded = personal_change_report(collate_execution_episodes([first, second]), ruler)
    assert float(padded.joint_deviation[0]) == pytest.approx(
        float(alone.joint_deviation[0]), abs=1e-6
    )
    assert float(padded.personal_limit95[0]) == pytest.approx(
        float(alone.personal_limit95[0]), abs=1e-6
    )


def test_raw_physical_control_uses_the_same_personal_reference_protocol() -> None:
    def physical(name: str, offset: float):
        execution = _execution(name, length=5)
        features = torch.full((5, 1), 1.0 + offset)
        return replace(
            execution,
            physical_features=features,
            physical_feature_mask=torch.ones_like(features, dtype=torch.bool),
            physical_feature_names=("acc_magnitude_mean_g",),
        )

    references = tuple(physical(f"r{index}", 0.001 * index) for index in range(4))
    report = physical_change_report(references, physical("query", 0.2))
    assert np.isfinite(report["joint_deviation"])
    assert np.isfinite(report["personal_limit95"])
    assert report["exceeds_personal_limit"]


def test_smoke_trains_without_nonfinite_gradients() -> None:
    result = run_synthetic_smoke(steps=4, subjects=4, seed=5)
    assert result.nonfinite_gradients == 0
    assert result.episodes > 0
    assert np.isfinite(result.first_loss) and np.isfinite(result.last_loss)
    assert 0.0 <= result.floor_auroc <= 1.0 and 0.0 <= result.ruler_auroc <= 1.0


def test_zero_initialized_refinement_wakes_the_context_path() -> None:
    _, batch = build_demo_batch(subjects=5, seed=17, groups=3)
    model = ChangeRuler(batch.query_embeddings.shape[-1], phase_bins=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = train_step(model, batch, optimizer)
    second = train_step(model, batch, optimizer)
    # Exact floor initialization intentionally blocks the upstream context path
    # on step one, but the updated refinement must expose gradients immediately.
    assert first.parameter_grad_norms["refinement.weight"] > 0
    assert second.parameter_grad_norms["query_attention.in_proj_weight"] > 0
    assert second.parameter_grad_norms["reference_encoder.layers.0.self_attn.in_proj_weight"] > 0


def test_phase_resampling_uses_physical_time_and_handles_one_patch() -> None:
    from applications.motion_monitoring.task2.model import resample_to_phase

    values = torch.tensor([[[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]])
    intervals = torch.tensor([[[0.0, 1.0], [1.0, 2.0], [2.0, 9.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    resampled = resample_to_phase(values, intervals, mask, bins=3)
    # The last patch is seven seconds long, so its centre sits late in phase and
    # the midpoint interpolates between the first two patches, not the last two.
    assert resampled.shape == (1, 3, 2)
    assert resampled[0, 0].tolist() == pytest.approx([0.0, 0.0])
    assert resampled[0, -1].tolist() == pytest.approx([2.0, 4.0])
    single = resample_to_phase(
        values[:, :1], intervals[:, :1], mask[:, :1], bins=4
    )
    assert single.shape == (1, 4, 2)
    assert torch.allclose(single[0], values[0, 0].expand(4, 2))


def test_padding_and_reference_order_do_not_change_the_distance() -> None:
    references = tuple(_execution(f"r{index}", length=5 + index) for index in range(3))
    query = _execution("q", length=7)
    ruler = ChangeRuler(4, phase_bins=8).eval()
    base = collate_execution_episodes(
        [ExecutionEpisode(accepted_references=references, query=query, episode_kind="accepted_query")]
    )
    reordered = collate_execution_episodes(
        [
            ExecutionEpisode(
                accepted_references=tuple(reversed(references)),
                query=query,
                episode_kind="accepted_query",
            )
        ]
    )
    with torch.no_grad():
        assert ruler(base).distances.tolist() == pytest.approx(
            ruler(reordered).distances.tolist(), abs=1e-5
        )
    # Padding a shorter episode alongside a longer one must not move its score.
    long_references = tuple(_execution(f"L{index}", length=12) for index in range(3))
    padded = collate_execution_episodes(
        [
            ExecutionEpisode(accepted_references=references, query=query, episode_kind="accepted_query"),
            ExecutionEpisode(
                accepted_references=long_references,
                query=_execution("Q", length=14),
                episode_kind="accepted_query",
            ),
        ]
    )
    with torch.no_grad():
        assert float(ruler(padded).distances[0]) == pytest.approx(
            float(ruler(base).distances[0]), abs=1e-5
        )


def test_ruler_preserves_dtype_in_half_precision() -> None:
    references = tuple(_execution(f"r{index}") for index in range(2))
    episode = ExecutionEpisode(
        accepted_references=references, query=_execution("q"), episode_kind="accepted_query"
    )
    batch = collate_execution_episodes([episode])
    half = batch.to("cpu")
    half = type(half)(
        **{
            name: (value.half() if isinstance(value, torch.Tensor) and value.is_floating_point() else value)
            for name, value in half.__dict__.items()
        }
    )
    ruler = ChangeRuler(4, phase_bins=8).half().eval()
    with torch.no_grad():
        output = ruler(half)
    assert output.distances.dtype == torch.float16
    assert torch.isfinite(output.distances).all()


def _pool(**kwargs):
    from applications.motion_monitoring.task2.smoke import make_pool

    return make_pool(**kwargs)


def test_builder_selects_pre_materialised_variants_and_prefers_other_days() -> None:
    pool = _pool(subjects=4, seed=5)
    builder = Task2BatchBuilder(pool, reference_count=3, positives_per_episode=2, seed=3)
    episodes, plans = builder.build_batch(groups=2)
    validate_batch(episodes)
    kinds = {episode.episode_kind for episode in episodes}
    assert {"accepted_query", "modified_query", "other_subject_query"} <= kinds
    for episode in episodes:
        assert episode.query.execution_id not in {
            item.execution_id for item in episode.accepted_references
        }
    # Every modification is one the corpus declared, with its severity carried.
    for episode in episodes:
        if episode.episode_kind == "modified_query":
            assert episode.modification_kind in MODIFICATIONS
            assert 0.0 < episode.severity <= 1.0
    assert all(plan.modified >= 1 for plan in plans)


def test_a_query_is_never_a_variant_of_its_own_reference() -> None:
    pool = _pool(subjects=4, seed=12)
    builder = Task2BatchBuilder(pool, reference_count=3, positives_per_episode=2, seed=6)
    roots = {record.execution.execution_id: record.root_id for record in pool}
    for offset in range(8):
        episodes, _ = builder.build_batch(groups=2, seed=400 + offset)
        for episode in episodes:
            reference_roots = {roots[item.execution_id] for item in episode.accepted_references}
            assert roots[episode.query.execution_id] not in reference_roots


def test_a_batch_never_mixes_two_compatibility_keys() -> None:
    from applications.motion_monitoring.task2.episodes import ACCELERATION_ONLY

    pool = _pool(subjects=4, seed=15)
    reduced = []
    for record in pool[: len(pool) // 2]:
        execution = record.execution
        reduced.append(
            replace(
                record,
                key=OTHER_KEY,
                execution=type(execution)(
                    embeddings=execution.embeddings,
                    patch_intervals_sec=execution.patch_intervals_sec,
                    patch_mask=execution.patch_mask,
                    dataset=execution.dataset,
                    subject_id=execution.subject_id,
                    session_id=execution.session_id,
                    execution_id=f"pocket:{execution.execution_id}",
                    task_id=execution.task_id,
                    sensor_config=OTHER_KEY,
                ),
                origin_execution_id=f"pocket:{record.root_id}",
            )
        )
    builder = Task2BatchBuilder(pool + reduced, reference_count=3, positives_per_episode=2, seed=9)
    for offset in range(8):
        episodes, plans = builder.build_batch(groups=2, seed=500 + offset)
        validate_batch(episodes)
        assert len({plan.key for plan in plans}) == 1
    assert ACCELERATION_ONLY == ("acc_x", "acc_y", "acc_z")


def test_declared_dataset_mixture_can_exclude_a_source() -> None:
    pool = _pool(subjects=4, seed=6)
    second = [
        replace(
            record,
            source_dataset="other_source",
            execution=type(record.execution)(
                embeddings=record.execution.embeddings,
                patch_intervals_sec=record.execution.patch_intervals_sec,
                patch_mask=record.execution.patch_mask,
                dataset="other_source",
                subject_id=record.execution.subject_id,
                session_id=record.execution.session_id,
                execution_id=f"other:{record.execution.execution_id}",
                task_id=record.execution.task_id,
                sensor_config=record.execution.sensor_config,
            ),
            origin_execution_id=f"other:{record.root_id}",
        )
        for record in pool
    ]
    builder = Task2BatchBuilder(
        pool + second,
        reference_count=3,
        positives_per_episode=2,
        dataset_weights={"synthetic_task2": 1.0, "other_source": 0.0},
        seed=8,
    )
    for offset in range(8):
        episodes, _ = builder.build_batch(groups=1, seed=200 + offset)
        assert {episode.query.dataset for episode in episodes} == {"synthetic_task2"}
    with pytest.raises(ValueError, match="weight zero"):
        Task2BatchBuilder(
            pool + second,
            dataset_weights={"synthetic_task2": 0.0, "other_source": 0.0},
            seed=8,
        ).build_batch(groups=1)


def test_dataset_weights_are_not_multiplied_by_number_of_configurations() -> None:
    pool = _pool(subjects=4, seed=16)
    second = [
        replace(
            record,
            source_dataset="other_source",
            execution=replace(
                record.execution,
                dataset="other_source",
                execution_id=f"other:{record.execution.execution_id}",
            ),
            origin_execution_id=f"other:{record.root_id}",
        )
        for record in pool
    ]
    second_extra_config = [
        replace(
            record,
            key=OTHER_KEY,
            execution=replace(
                record.execution,
                sensor_config=OTHER_KEY,
                execution_id=f"other-pocket:{record.execution.execution_id}",
            ),
            origin_execution_id=f"other-pocket:{record.root_id}",
        )
        for record in second
    ]
    builder = Task2BatchBuilder(
        pool + second + second_extra_config,
        reference_count=3,
        positives_per_episode=2,
        dataset_weights={"synthetic_task2": 1.0, "other_source": 1.0},
    )
    selected = []
    for seed in range(400):
        _, plans = builder.build_batch(groups=1, seed=seed)
        selected.append(plans[0].dataset)
    other_share = selected.count("other_source") / len(selected)
    assert 0.4 < other_share < 0.6


def test_batch_builder_never_silently_returns_fewer_groups() -> None:
    builder = Task2BatchBuilder(
        _pool(subjects=4, seed=17), reference_count=3, positives_per_episode=2
    )
    with pytest.raises(ValueError, match="independent reference sets"):
        builder.build_batch(groups=len(builder.eligible_groups) + 1)


def test_relation_summary_reports_the_same_session_share() -> None:
    from applications.motion_monitoring.task2.episodes import relation_summary

    builder = Task2BatchBuilder(
        _pool(subjects=5, seed=7), reference_count=3, positives_per_episode=2, seed=7
    )
    plans = []
    for offset in range(5):
        _, batch_plans = builder.build_batch(groups=2, seed=300 + offset)
        plans.extend(batch_plans)
    summary = relation_summary(plans)
    assert summary["reference_sets"] == len(plans)
    assert 0.0 <= summary["same_session_share"] <= 1.0
    assert sum(summary["positive_relations"].values()) > 0
    assert set(summary["positive_relations"]) <= {"same_session", "different_session", "different_day"}


def test_record_rejects_an_inconsistent_variant_declaration() -> None:
    pool = _pool(subjects=3, seed=2)
    clean = next(record for record in pool if record.variant == "clean")
    with pytest.raises(ValueError, match="modification kind"):
        replace(clean, variant="modified")
    with pytest.raises(ValueError, match="severity"):
        replace(clean, variant="modified", modification_kind="retime", severity=0.0)
    with pytest.raises(ValueError, match="unknown variant"):
        replace(clean, variant="mystery")


def test_a_variant_inherits_the_acquisition_day_of_its_origin() -> None:
    """A nuisance variant may be an accepted query, so its day must be real.

    Without this the variant reads as same-session and the cross-day positives the
    objective deliberately over-weights are undercounted.
    """

    from applications.motion_monitoring.data.contracts import RawRecording, SensorStream
    from applications.motion_monitoring.task2.records import DAY_RESOLVERS

    times = np.arange(8, dtype=np.float64) / 50.0
    stream = SensorStream(
        stream_id="watch_wrist",
        placement="dominant_wrist",
        device="WearOS smartwatch",
        timestamps_sec=times,
        values=np.ones((len(times), 3), dtype=np.float32),
        channels=("acc_x", "acc_y", "acc_z"),
        valid=np.ones((len(times), 3), dtype=np.bool_),
        gravity_state="present",
        nominal_rate_hz=50.0,
        # The variant adapter copies the source stream metadata verbatim.
        metadata={"source_epoch_start_sec": 1753780098.579},
    )
    variant = RawRecording(
        dataset="task2_modified_v1",
        recording_id="task2_modified_v1:harmes:x:0:modified_0",
        subject_id="pp01",
        session_id="harmes:pp01:0101",
        streams=(stream,),
        metadata={"origin_dataset": "harmes", "variant": "modified"},
    )
    assert DAY_RESOLVERS["task2_modified_v1"](variant) == "2025-07-29"

    # A source with no day axis stays None rather than inventing one, and a
    # self-referential origin cannot recurse.
    crossfit_variant = RawRecording(
        dataset="task2_modified_v1",
        recording_id="task2_modified_v1:crossfit:y:0:modified_0",
        subject_id="s1",
        session_id="crossfit:exercise:3",
        streams=(stream,),
        metadata={"origin_dataset": "crossfit", "variant": "modified"},
    )
    assert DAY_RESOLVERS["task2_modified_v1"](crossfit_variant) is None
    looping = RawRecording(
        dataset="task2_modified_v1",
        recording_id="task2_modified_v1:loop",
        subject_id="s1",
        session_id="s",
        streams=(stream,),
        metadata={"origin_dataset": "task2_modified_v1"},
    )
    assert DAY_RESOLVERS["task2_modified_v1"](looping) is None
