from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from applications.motion_monitoring.data.cache import CachedRecordingDataset, write_recording
from applications.motion_monitoring.data.contracts import EventInterval, RawRecording, SensorStream
from applications.motion_monitoring.data.manifests import (
    build_cohort_manifest,
    read_cohort_manifest,
    validate_manifest_caches,
    write_cohort_manifest,
)


def _recording(dataset: str, subject: str, index: int, *, duplicate: bool = False):
    stream = SensorStream(
        stream_id="wrist",
        placement="wrist",
        device="watch",
        timestamps_sec=np.arange(20, dtype=np.float64) / 10,
        values=np.ones((20, 3), dtype=np.float32),
        channels=("acc_x", "acc_y", "acc_z"),
        valid=np.ones((20, 3), dtype=np.bool_),
        gravity_state="present",
        nominal_rate_hz=10.0,
    )
    return RawRecording(
        dataset=dataset,
        recording_id=f"{dataset}-{index}",
        subject_id=subject,
        session_id=f"session-{index}",
        streams=(stream,),
        events=(EventInterval(0.2, 1.2, "move", "event"),),
        metadata={"duplicates_parent_exercise_signal": duplicate},
    )


def _cache(tmp_path, dataset: str, recordings) -> CachedRecordingDataset:
    root = tmp_path / dataset
    rows = []
    for index, recording in enumerate(recordings):
        directory = f"row-{index}"
        write_recording(recording, root / directory)
        rows.append({"dataset": dataset, "directory": directory})
    (root / "cache.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": dataset,
                "recording_count": len(rows),
            }
        ),
        encoding="utf-8",
    )
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return CachedRecordingDataset(root, validate_provenance=False)


def test_manifest_is_deterministic_disjoint_and_excludes_duplicate_views(tmp_path):
    train = _cache(
        tmp_path,
        "train_source",
        (
            _recording("train_source", "s1", 1),
            _recording("train_source", "s1", 2),
            _recording("train_source", "s2", 3),
            _recording("train_source", "s3", 4, duplicate=True),
        ),
    )
    test = _cache(
        tmp_path,
        "test_source",
        (_recording("test_source", "e1", 1),),
    )
    kwargs = dict(
        caches={"train_source": train, "test_source": test},
        name="test",
        training_datasets=("train_source",),
        evaluation_datasets=("test_source",),
        seed=7,
        development_fraction=0.5,
    )
    first = build_cohort_manifest(**kwargs)
    second = build_cohort_manifest(**kwargs)

    assert first == second
    assert len(first.entries) == 4
    assert {entry.split for entry in first.entries_for(dataset="test_source")} == {
        "test"
    }
    subject_splits = {
        entry.split
        for entry in first.entries_for(dataset="train_source")
        if entry.subject_id == "s1"
    }
    assert len(subject_splits) == 1


def test_manifest_round_trip_and_cache_drift_detection(tmp_path):
    train = _cache(
        tmp_path,
        "train_source",
        (_recording("train_source", "s1", 1), _recording("train_source", "s2", 2)),
    )
    test = _cache(
        tmp_path, "test_source", (_recording("test_source", "e1", 1),)
    )
    caches = {"train_source": train, "test_source": test}
    manifest = build_cohort_manifest(
        caches,
        name="test",
        training_datasets=("train_source",),
        evaluation_datasets=("test_source",),
    )
    path = tmp_path / "cohort.json"
    write_cohort_manifest(manifest, path)
    loaded = read_cohort_manifest(path)
    validate_manifest_caches(loaded, caches)
    assert loaded == manifest

    with pytest.raises(ValueError, match="cohort membership changed"):
        validate_manifest_caches(
            replace(loaded, entries=loaded.entries[:-1]), caches
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["split"] = "train"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        read_cohort_manifest(path)


def test_extended_roles_split_only_where_declared_and_round_trip(tmp_path):
    synthetic = _cache(
        tmp_path,
        "synthetic_source",
        (_recording("synthetic_source", "bg:1", 1), _recording("synthetic_source", "bg:2", 2)),
    )
    dev_only = _cache(tmp_path, "dev_source", (_recording("dev_source", "d1", 1),))
    split_eval = _cache(
        tmp_path,
        "split_source",
        tuple(_recording("split_source", f"p{index}", index) for index in range(1, 9)),
    )
    test = _cache(tmp_path, "test_source", (_recording("test_source", "e1", 1),))
    caches = {
        "synthetic_source": synthetic,
        "dev_source": dev_only,
        "split_source": split_eval,
        "test_source": test,
    }
    manifest = build_cohort_manifest(
        caches,
        name="roles",
        train_only_datasets=("synthetic_source",),
        development_only_datasets=("dev_source",),
        split_evaluation_datasets=("split_source",),
        evaluation_datasets=("test_source",),
        evaluation_development_fraction=0.25,
    )
    splits = {
        dataset: {entry.split for entry in manifest.entries_for(dataset=dataset)}
        for dataset in caches
    }
    assert splits["synthetic_source"] == {"train"}
    assert splits["dev_source"] == {"development"}
    assert splits["test_source"] == {"test"}
    assert splits["split_source"] == {"development", "test"}
    split_dev = [
        entry.leakage_group
        for entry in manifest.entries_for(dataset="split_source")
        if entry.split == "development"
    ]
    assert len(set(split_dev)) == 2
    assert manifest.roles == {
        "dev_source": "development_only",
        "split_source": "split_evaluation",
        "synthetic_source": "train_only",
        "test_source": "evaluation",
    }

    path = tmp_path / "roles.json"
    write_cohort_manifest(manifest, path)
    loaded = read_cohort_manifest(path)
    assert loaded == manifest
    validate_manifest_caches(loaded, caches)

    # Roles are fingerprinted: a role edit is a different cohort.
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["roles"]["dev_source"] = "train_only"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        read_cohort_manifest(path)

    # Every declared dataset must have a loaded cache and vice versa.
    with pytest.raises(ValueError):
        build_cohort_manifest(
            caches, name="bad", train_only_datasets=("synthetic_source",)
        )

    # Two-role cohorts keep their original (unextended) fingerprint payload.
    legacy = build_cohort_manifest(
        {"synthetic_source": synthetic, "test_source": test},
        name="legacy",
        training_datasets=("synthetic_source",),
        evaluation_datasets=("test_source",),
    )
    assert legacy.roles == {}
    assert "roles" not in json.loads(
        json.dumps(legacy, default=lambda value: getattr(value, "__dict__", str(value)))
    ) or legacy.roles == {}
