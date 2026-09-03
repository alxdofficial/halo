import math
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from applications.motion_monitoring.task1 import (
    DifferentiableSubsequenceMatcher,
    SyntheticDetectionDataset,
    collate_detection_episodes,
    event_detection_metrics,
    train_step,
)
from applications.motion_monitoring.task1.matcher import TemporalMatch
from applications.motion_monitoring.evaluation_manifests import (
    Task1EvaluationUnit,
    TaskEvaluationManifest,
)
from applications.motion_monitoring.task1 import full_evaluation
from applications.motion_monitoring.task1 import train_full
from applications.motion_monitoring.task1.train_full import (
    _event_prefix,
    average_precision,
    fix_operating_point,
    select_common_units,
    split_by_subject,
)


def test_synthetic_episodes_are_deterministic_and_include_absent_examples():
    first = SyntheticDetectionDataset(
        4, feature_dim=8, query_patches=24, reference_patches=4, seed=11
    )
    second = SyntheticDetectionDataset(
        4, feature_dim=8, query_patches=24, reference_patches=4, seed=11
    )
    for index in range(4):
        assert torch.equal(
            first[index].reference.embeddings, second[index].reference.embeddings
        )
        assert torch.equal(
            first[index].query.embeddings, second[index].query.embeddings
        )
        assert torch.equal(first[index].targets_sec, second[index].targets_sec)

    absent = SyntheticDetectionDataset(
        1,
        feature_dim=8,
        query_patches=24,
        reference_patches=4,
        target_present_probability=0.0,
    )[0]
    assert absent.targets_sec.shape == (0, 2)


def test_short_smoke_training_reports_loss_metrics_and_gradient_health():
    dataset = SyntheticDetectionDataset(
        12, feature_dim=8, query_patches=24, reference_patches=4, seed=19
    )
    batch = collate_detection_episodes([dataset[index] for index in range(12)])
    model = DifferentiableSubsequenceMatcher(8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    results = [train_step(model, batch, optimizer) for _ in range(4)]

    assert all(math.isfinite(result.loss) for result in results)
    telemetry = results[-1].telemetry
    assert telemetry["gradient_finite"] == 1.0
    assert telemetry["parameters_with_gradient_fraction"] == 1.0
    assert telemetry["parameters_with_nonzero_gradient_fraction"] == 1.0
    assert telemetry["grad_norm_projection"] > 0
    assert telemetry["grad_norm_score_bias"] > 0
    assert 0 <= telemetry["endpoint_f1"] <= 1
    assert 0 <= telemetry["target_absent_false_alarm_rate"] <= 1
    assert results[-1].loss < results[0].loss


def test_event_metrics_score_boundaries_and_target_absent_false_alarms():
    match = TemporalMatch(2, 5, 2.0, 5.0, 0.1, 3, 1.0)
    metrics = event_detection_metrics(
        [match], torch.tensor([[2.0, 5.0]]), query_duration_sec=120.0
    )
    assert metrics["event_f1"] == 1.0
    assert metrics["matched_event_count"] == 1.0
    assert metrics["event_average_precision"] == 1.0
    assert metrics["event_recall_at_operating_point"] == 1.0
    assert metrics["mean_absolute_count_error"] == 0.0
    assert metrics["mean_onset_error_sec"] == 0.0
    assert metrics["false_alarms_per_hour"] == 0.0

    absent = event_detection_metrics(
        [match], torch.empty(0, 2), query_duration_sec=120.0
    )
    assert absent["event_f1"] == 0.0
    assert absent["event_average_precision"] == 0.0
    assert absent["false_alarms_per_hour"] == 30.0


def test_calibration_prefix_matches_event_metric_at_every_threshold():
    matches = [
        TemporalMatch(0, 2, 0.0, 2.0, 0.1, 2, 1.0),
        TemporalMatch(1, 3, 1.0, 3.0, 0.2, 2, 1.0),
        TemporalMatch(5, 7, 5.0, 7.0, 0.3, 2, 1.0),
        TemporalMatch(8, 10, 8.0, 10.0, 0.4, 2, 1.0),
    ]
    targets = torch.tensor([[0.0, 2.0], [5.0, 7.0]])
    scores, cumulative, target_count = _event_prefix(
        matches, targets, iou_threshold=0.5
    )

    for threshold in (np.nextafter(0.1, -np.inf), 0.1, 0.2, 0.3, 0.4):
        detection_count = int(np.searchsorted(scores, threshold, side="right"))
        true_positive = int(cumulative[detection_count - 1]) if detection_count else 0
        fast = (
            true_positive,
            detection_count - true_positive,
            target_count - true_positive,
        )
        exact = event_detection_metrics(
            matches,
            targets,
            query_duration_sec=10.0,
            iou_threshold=0.5,
            score_threshold=threshold,
        )
        assert fast == (
            int(exact["true_positive_count"]),
            int(exact["false_positive_count"]),
            int(exact["false_negative_count"]),
        )


def test_full_evaluation_passes_frozen_threshold_mapping_to_each_unit(monkeypatch):
    unit = Task1EvaluationUnit(
        dataset="source",
        query_cache_index=0,
        query_recording_id="query",
        query_subject_id="subject",
        query_stream_id="imu",
        reference_cache_index=1,
        reference_recording_id="reference",
        reference_subject_id="other",
        reference_stream_id="imu",
        reference_event_index=0,
        label="motion",
        target_intervals_sec=((1.0, 2.0),),
        target_present=True,
        reference_interval_sec=(0.0, 1.0),
        reference_rule="test",
    )
    manifest = TaskEvaluationManifest(
        schema_version=2,
        name="test",
        task="task1",
        cohort_fingerprint="cohort",
        seed=1,
        protocol={"split": "test"},
        units=(asdict(unit),),
        exclusions=(),
        fingerprint="manifest",
    )
    thresholds = {"2to3": 0.2, "4to15": 0.3, "ge16": 0.4, "global": 0.5}
    seen = []

    def fake_unit_metrics(*args, score_threshold, **kwargs):
        seen.append(score_threshold)
        return {
            "true_positive_count": 1.0,
            "false_positive_count": 0.0,
            "false_negative_count": 0.0,
            "query_hours": 1.0,
            "onset_absolute_error_sum_sec": 0.0,
            "offset_absolute_error_sum_sec": 0.0,
            "mean_absolute_count_error": 0.0,
            "query_subject_id": "subject",
            "reference_positions": 2,
            "duration_strata": {},
        }

    monkeypatch.setattr(full_evaluation, "_unit_metrics", fake_unit_metrics)
    results = full_evaluation.evaluate_task1_test(
        manifest,
        {"source": object()},
        object(),
        score_threshold=thresholds,
    )

    assert seen == [thresholds]
    assert results[0].metrics["event_f1"] == 1.0


def test_common_units_are_bound_to_the_current_encoder(tmp_path):
    manifest = TaskEvaluationManifest(
        schema_version=2,
        name="test",
        task="task1",
        cohort_fingerprint="cohort",
        seed=1,
        protocol={"split": "test"},
        units=({"unit": 0}, {"unit": 1}),
        exclusions=(),
        fingerprint="task-fingerprint",
    )
    path = tmp_path / "common.json"
    path.write_text(
        json.dumps(
            {
                "task_manifest_fingerprint": manifest.fingerprint,
                "selected_unit_indices": [1],
                "representations": {
                    "encoder": {"encoder_provenance": {"kind": "expected"}}
                },
            }
        )
    )

    selected, _ = select_common_units(
        manifest, path, representation_provenance={"kind": "expected"}
    )
    assert selected.units == ({"unit": 1},)
    with pytest.raises(ValueError, match="not built for this encoder"):
        select_common_units(
            manifest, path, representation_provenance={"kind": "different"}
        )


def test_stratified_recall_uses_only_operating_point_detections(monkeypatch):
    unit = Task1EvaluationUnit(
        dataset="source",
        query_cache_index=0,
        query_recording_id="query",
        query_subject_id="subject",
        query_stream_id="imu",
        reference_cache_index=1,
        reference_recording_id="reference",
        reference_subject_id="other",
        reference_stream_id="imu",
        reference_event_index=0,
        label="motion",
        target_intervals_sec=((0.0, 2.0), (5.0, 7.0)),
        target_present=True,
        reference_interval_sec=(0.0, 1.0),
        reference_rule="test",
    )
    matches = [
        TemporalMatch(0, 2, 0.0, 2.0, 0.1, 2, 1.0),
        TemporalMatch(5, 7, 5.0, 7.0, 0.9, 2, 1.0),
    ]
    episode = SimpleNamespace(
        reference=SimpleNamespace(embeddings=torch.zeros(2, 3)),
        query=SimpleNamespace(intervals_sec=torch.tensor([[0.0, 1.0], [9.0, 10.0]])),
        targets_sec=torch.tensor(unit.target_intervals_sec),
        metadata={"reference_positions": 2},
    )
    monkeypatch.setattr(
        full_evaluation,
        "unit_matches",
        lambda *args, **kwargs: (matches, episode),
    )

    metrics = full_evaluation._unit_metrics(
        unit,
        object(),
        object(),
        score_threshold={"2to3": 0.5, "global": 0.5},
        model=None,
        nms_iou=0.3,
        match_iou=0.5,
    )

    assert metrics["true_positive_count"] == 1.0
    assert metrics["duration_strata"]["ge2s"] == {"matched": 1.0, "total": 2.0}


def _unit(subject: str, targets=((1.0, 2.0),), relation: str = "cross_subject") -> Task1EvaluationUnit:
    return Task1EvaluationUnit(
        dataset="source",
        query_cache_index=0,
        query_recording_id=f"query-{subject}",
        query_subject_id=subject,
        query_stream_id="imu",
        reference_cache_index=1,
        reference_recording_id="reference",
        reference_subject_id="other",
        reference_stream_id="imu",
        reference_event_index=0,
        label="motion",
        target_intervals_sec=targets,
        target_present=bool(targets),
        reference_interval_sec=(0.0, 1.0),
        reference_rule="test",
        reference_relation=relation,
    )


def test_split_by_subject_is_deterministic_and_never_empties_a_side():
    units = [_unit(f"s{i}") for i in range(5)]
    fit_a, held_a = split_by_subject(units, seed=3, heldout_fraction=0.2)
    fit_b, held_b = split_by_subject(units, seed=3, heldout_fraction=0.2)
    assert [u.query_subject_id for u in held_a] == [u.query_subject_id for u in held_b]
    assert held_a and fit_a
    assert not ({u.query_subject_id for u in held_a} & {u.query_subject_id for u in fit_a})
    with pytest.raises(ValueError):
        split_by_subject([_unit("solo")], seed=3, heldout_fraction=0.2)


def test_operating_point_holdout_uses_two_of_seven_subjects():
    _, heldout = split_by_subject(
        [_unit(f"s{i}") for i in range(7)], seed=20260831, heldout_fraction=0.3
    )
    assert len({unit.query_subject_id for unit in heldout}) == 2


def test_operating_point_respects_the_false_alarm_budget(monkeypatch):
    # Two one-hour queries: every unit yields one true positive (score 0.1) and
    # three false alarms (scores 0.2, 0.3, 0.4). Budget 1 FA/h over 2 h allows 2
    # false alarms -> threshold must be the second-smallest false-alarm score.
    units = [_unit("a"), _unit("b")]

    def fake_unit_matches(unit, *args, **kwargs):
        matches = [
            TemporalMatch(0, 2, 1.0, 2.0, 0.1, 2, 1.0),
            TemporalMatch(5, 7, 5.0, 6.0, 0.2, 2, 1.0),
            TemporalMatch(8, 10, 8.0, 9.0, 0.3, 2, 1.0),
            TemporalMatch(11, 13, 11.0, 12.0, 0.4, 2, 1.0),
        ]
        episode = SimpleNamespace(
            reference=SimpleNamespace(embeddings=torch.zeros(2, 3)),
            query=SimpleNamespace(intervals_sec=torch.tensor([[0.0, 1.0], [3599.0, 3600.0]])),
            targets_sec=torch.tensor(unit.target_intervals_sec),
            metadata={"reference_positions": 2},
        )
        return matches, episode

    monkeypatch.setattr(train_full, "unit_matches", fake_unit_matches)
    point = fix_operating_point(
        units, {"source": object()}, object(), None, false_alarm_budget_per_hour=1.0
    )
    assert point["holdout_false_alarms_allowed"] == 2
    assert point["threshold"] == pytest.approx(0.2)
    assert point["holdout_false_alarms"] == 2
    assert point["metrics"]["false_alarms_per_hour"] <= 1.0
    assert point["metrics"]["event_recall"] == 1.0
    assert point["average_precision"] == pytest.approx(1.0)

    strict = fix_operating_point(
        units, {"source": object()}, object(), None, false_alarm_budget_per_hour=0.0
    )
    assert strict["holdout_false_alarms"] == 0
    assert strict["metrics"]["event_recall"] == 1.0


def test_average_precision_pools_ranked_detections():
    # unit 1: TP at 0.1, FP at 0.2; unit 2: FP at 0.15, TP at 0.3 -> ranked
    # 0.1(TP) 0.15(FP) 0.2(FP) 0.3(TP): AP = (1/1 + 2/4) / 2 = 0.75
    evaluated = [
        {"scores": np.array([0.1, 0.2]), "cumulative_tp": np.array([1, 1]), "target_count": 1},
        {"scores": np.array([0.15, 0.3]), "cumulative_tp": np.array([0, 1]), "target_count": 1},
    ]
    assert average_precision(evaluated) == pytest.approx(0.75)
    assert math.isnan(average_precision([{"scores": np.array([]), "cumulative_tp": np.array([]), "target_count": 0}]))
