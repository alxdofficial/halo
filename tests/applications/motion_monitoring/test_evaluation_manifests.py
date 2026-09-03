from __future__ import annotations

from pathlib import Path

import pytest

from applications.motion_monitoring.evaluation_manifests import (
    Task1EvaluationUnit,
    TaskEvaluationManifest,
    _retain_complete_relation_sets,
    read_task_manifest,
    validate_task_manifest,
)
from applications.motion_monitoring.data.manifests import CohortManifest


def test_checked_in_task_manifests_are_self_signed() -> None:
    root = Path("applications/motion_monitoring/manifests")
    expected = {
        "TASK1_TRAIN_V2.json": "task1",
        "TASK1_TEST_V2.json": "task1",
        "TASK2_TRAIN_V1.json": "task2",
        "TASK2_TEST_V1.json": "task2",
        "TASK3_TRAIN_V2.json": "task3",
        "TASK3_TEST_V2.json": "task3",
    }
    for name, task in expected.items():
        manifest = read_task_manifest(root / name)
        assert manifest.task == task
        # The Task-2 train manifest declares a pool, not a fixed episode list.
        if name != "TASK2_TRAIN_V1.json":
            assert len(manifest.units) > 0
        else:
            assert manifest.protocol["sources"] and manifest.protocol["dataset_weights"]
    test = read_task_manifest(root / "TASK1_TEST_V2.json")
    relations = {row["reference_relation"] for row in test.units}
    assert relations == {"cross_subject", "same_subject"}
    assert {row["dataset"] for row in test.units} == {"c_mhad", "openpack"}
    assert not (root / "TASK1_DEVELOPMENT_V2.json").exists()
    # Task 3 has no development split either: its threshold comes from held-out
    # training identities under a declared false-edge budget.
    assert not (root / "TASK3_DEVELOPMENT_V2.json").exists()
    task3 = read_task_manifest(root / "TASK3_TEST_V2.json")
    assert {row["dataset"] for row in task3.units} == {"oca", "openpack"}
    train3 = read_task_manifest(root / "TASK3_TRAIN_V2.json")
    assert {row["dataset"] for row in train3.units} == {"opportunity", "synth_long_v1"}


def test_choose_reference_honours_the_declared_relation() -> None:
    from applications.motion_monitoring.data.contracts import EventInterval
    from applications.motion_monitoring.data.manifests import CohortEntry
    from applications.motion_monitoring.evaluation_manifests import _choose_reference

    def entry(recording, subject):
        return CohortEntry(
            dataset="source",
            recording_id=recording,
            cache_index=0,
            subject_id=subject,
            session_id=recording,
            leakage_group=subject,
            split="test",
            stream_ids=("imu",),
            event_counts={},
        )

    event = EventInterval(start_sec=0.0, end_sec=1.0, label="motion")
    query = entry("q", "alice")
    candidates = [
        (entry("q", "alice"), "imu", 0, event),      # same recording: never
        (entry("a2", "alice"), "imu", 0, event),     # same person, other recording
        (entry("b1", "bob"), "imu", 0, event),       # other person
    ]
    same = _choose_reference(candidates, query=query, label="motion", seed=1, relation="same_subject")
    cross = _choose_reference(candidates, query=query, label="motion", seed=1, relation="cross_subject")
    assert same is not None and same[0].recording_id == "a2"
    assert cross is not None and cross[0].subject_id == "bob"
    only_self = [candidates[0], candidates[1]]
    assert _choose_reference(only_self, query=query, label="motion", seed=1, relation="cross_subject") is None
    with pytest.raises(ValueError, match="unknown reference relation"):
        _choose_reference(candidates, query=query, label="motion", seed=1, relation="sibling")


def test_relation_comparison_keeps_only_paired_query_label_trials() -> None:
    def unit(label: str, relation: str) -> Task1EvaluationUnit:
        return Task1EvaluationUnit(
            dataset="source",
            query_cache_index=0,
            query_recording_id="query",
            query_subject_id="subject",
            query_stream_id="imu",
            reference_cache_index=1,
            reference_recording_id=f"reference-{relation}",
            reference_subject_id=relation,
            reference_stream_id="imu",
            reference_event_index=0,
            label=label,
            target_intervals_sec=((1.0, 2.0),),
            target_present=True,
            reference_interval_sec=(0.0, 1.0),
            reference_rule="test",
            reference_relation=relation,
        )

    retained, exclusions = _retain_complete_relation_sets(
        [
            unit("paired", "same_subject"),
            unit("paired", "cross_subject"),
            unit("unpaired", "cross_subject"),
        ],
        ("same_subject", "cross_subject"),
    )

    assert {(item.label, item.reference_relation) for item in retained} == {
        ("paired", "same_subject"),
        ("paired", "cross_subject"),
    }
    assert exclusions == [
        {
            "dataset": "source",
            "recording_id": "query",
            "stream_id": "imu",
            "query_interval_sec": None,
            "label": "unpaired",
            "target_present": True,
            "reason": "incomplete reference-relation set",
            "available_relations": ["cross_subject"],
            "required_relations": ["cross_subject", "same_subject"],
        }
    ]


def test_task_manifest_rejects_a_different_cohort() -> None:
    cohort_manifest = CohortManifest(
        schema_version=1,
        name="cohort",
        seed=1,
        development_fraction=0.2,
        cache_fingerprints={},
        entries=(),
        fingerprint="cohort-fingerprint",
    )
    manifest = TaskEvaluationManifest(
        schema_version=1,
        name="blocked",
        task="task2",
        cohort_fingerprint="wrong",
        seed=1,
        protocol={"status": "blocked"},
        units=(),
        exclusions=({"reason": "missing truth"},),
        fingerprint="unused",
    )
    with pytest.raises(ValueError, match="different cohort"):
        validate_task_manifest(manifest, cohort_manifest, {})


def test_task2_test_manifest_is_a_frozen_set_of_personal_comparisons() -> None:
    from collections import Counter

    from applications.motion_monitoring.evaluation_manifests import Task2EvaluationUnit

    manifest = read_task_manifest(
        Path("applications/motion_monitoring/manifests/TASK2_TEST_V1.json")
    )
    units = [Task2EvaluationUnit(**row) for row in manifest.units]
    assert {unit.dataset for unit in units} == {"monipar", "kneepad"}
    roles = Counter((unit.dataset, unit.role) for unit in units)
    for dataset in ("monipar", "kneepad"):
        assert roles[(dataset, "accepted_query")] > 0
        assert roles[(dataset, "changed_query")] > 0
    for unit in units:
        # A comparison is always within one person and one task, and a query is
        # never part of the reference set it is scored against.
        assert unit.query_cache_index not in unit.reference_cache_indices
        assert len(unit.reference_cache_indices) == len(set(unit.reference_cache_indices))
        assert unit.reference_cache_indices
    # Every declared change carries the evidence for the label.
    for unit in units:
        if unit.role != "changed_query":
            continue
        if unit.dataset == "monipar":
            assert unit.change_evidence["score_margin"] >= 1
            assert "strict_change" in unit.change_evidence
            reference_scores = unit.change_evidence["reference_scores"]
            assert len(reference_scores) == 4
            assert len(set(reference_scores)) == 1
            expected_margin = abs(
                unit.change_evidence["mds_updrs_bradykinesia"]
                - reference_scores[0]
            )
            assert unit.change_evidence["score_margin"] == expected_margin
        else:
            assert unit.change_evidence["execution_variant"] != "correct"
            assert unit.change_evidence["trial_index"] >= 1
    # MoniPar compares across weeks; KneE-PAD is one visit and says so.
    assert {u.relation for u in units if u.dataset == "monipar"} == {"different_day"}
    assert {u.relation for u in units if u.dataset == "kneepad"} == {"same_session"}


def test_monipar_serves_reliability_and_responsiveness_as_separate_cells() -> None:
    """One reference rule cannot answer both questions, so MoniPar declares two.

    The stable clinician-scored rule is right for responsiveness but discards
    every unscored subject, which is the only between-week noise floor the
    protocol has: KneE-PAD is a single visit and cannot supply one.
    """

    from collections import Counter

    from applications.motion_monitoring.evaluation_manifests import Task2EvaluationUnit

    manifest = read_task_manifest(
        Path("applications/motion_monitoring/manifests/TASK2_TEST_V1.json")
    )
    units = [Task2EvaluationUnit(**row) for row in manifest.units]
    cells = Counter((unit.dataset, unit.cell) for unit in units)
    assert cells[("monipar", "between_week_reliability")] > 0
    assert cells[("monipar", "clinician_rated_change")] > 0
    assert cells[("kneepad", "known_difference")] > 0
    assert all(unit.cell != "unspecified" for unit in units)

    reliability = [u for u in units if u.cell == "between_week_reliability"]
    responsiveness = [u for u in units if u.cell == "clinician_rated_change"]
    # Reliability is an accepted-only cell: a weekly repeat is accepted by
    # construction and this cell never infers a change label.
    assert {u.role for u in reliability} == {"accepted_query"}
    assert all("week" in u.change_evidence for u in reliability)
    # It must reach more people than the clinician-scored cell, which is the
    # whole reason it exists.
    assert len({u.subject_id for u in reliability}) > len(
        {u.subject_id for u in responsiveness}
    )
    # Responsiveness keeps the stricter label and its evidence.
    assert {u.role for u in responsiveness} == {"accepted_query", "changed_query"}
    for unit in responsiveness:
        if unit.role == "changed_query":
            assert unit.change_evidence["score_margin"] >= 1
    # Exclusions say which cell dropped a series and why.
    assert {row.get("cell") for row in manifest.exclusions} >= {"clinician_rated_change"}


def test_monipar_cells_are_validated_under_their_own_rules() -> None:
    """The clinician-score rule belongs to the responsiveness cell alone.

    Applying it to the reliability cell rejected every unscored weekly repeat,
    which is the whole cohort that cell exists to measure.
    """

    from applications.motion_monitoring.evaluation_manifests import Task2EvaluationUnit

    manifest = read_task_manifest(
        Path("applications/motion_monitoring/manifests/TASK2_TEST_V1.json")
    )
    units = [Task2EvaluationUnit(**row) for row in manifest.units]
    reliability = [u for u in units if u.cell == "between_week_reliability"]
    assert reliability
    # The cell that trips the old rule: unscored queries are the majority of it.
    unscored = [u for u in reliability if "mds_updrs_bradykinesia" not in u.change_evidence]
    assert unscored, "reliability cell should carry unscored weekly repeats"
    assert all(u.role == "accepted_query" for u in reliability)
    assert all("week" in u.change_evidence for u in reliability)
