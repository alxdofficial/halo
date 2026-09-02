from __future__ import annotations

from pathlib import Path

import pytest

from applications.motion_monitoring.evaluation_manifests import (
    TaskEvaluationManifest,
    read_task_manifest,
    validate_task_manifest,
)
from applications.motion_monitoring.data.manifests import CohortManifest


def test_checked_in_task_manifests_are_self_signed() -> None:
    root = Path("applications/motion_monitoring/manifests")
    manifests = [
        read_task_manifest(root / name)
        for name in (
            "TASK1_TRAIN_V1.json",
            "TASK1_DEVELOPMENT_V1.json",
            "TASK1_TEST_V1.json",
            "TASK2_TEST_V1.json",
            "TASK3_TRAIN_V1.json",
            "TASK3_DEVELOPMENT_V1.json",
            "TASK3_TEST_V1.json",
        )
    ]
    assert [manifest.task for manifest in manifests] == [
        "task1",
        "task1",
        "task1",
        "task2",
        "task3",
        "task3",
        "task3",
    ]
    assert len(manifests[0].units) > 0
    assert len(manifests[1].units) > 0
    assert len(manifests[2].units) > 0
    assert not manifests[3].units
    assert len(manifests[4].units) > 0
    assert len(manifests[5].units) > 0
    assert len(manifests[6].units) > 0


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
