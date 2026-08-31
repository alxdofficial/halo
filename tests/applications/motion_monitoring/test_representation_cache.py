from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from applications.motion_monitoring.data.cache import CachedRecordingDataset, write_recording
from applications.motion_monitoring.data.contracts import EventInterval, RawRecording, SensorStream
from applications.motion_monitoring.data.manifests import build_cohort_manifest
from applications.motion_monitoring.representation_cache import (
    CachedMotionSequenceDataset,
    build_representation_cache,
)
from applications.motion_monitoring.sequence import PhysicalProjectionEncoder


def _recording(dataset: str, subject: str, index: int) -> RawRecording:
    time = np.arange(40, dtype=np.float64) / 20
    values = np.column_stack(
        [np.sin(time), np.cos(time), np.ones_like(time)]
    ).astype(np.float32)
    stream = SensorStream(
        stream_id="wrist",
        placement="wrist",
        device="watch",
        timestamps_sec=time,
        values=values,
        channels=("acc_x", "acc_y", "acc_z"),
        valid=np.ones_like(values, dtype=np.bool_),
        gravity_state="present",
        nominal_rate_hz=20.0,
    )
    return RawRecording(
        dataset=dataset,
        recording_id=f"{dataset}-{index}",
        subject_id=subject,
        session_id=f"session-{index}",
        streams=(stream,),
        events=(EventInterval(0.2, 1.2, "move", "event"),),
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


def _cohort(tmp_path):
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
    return caches, manifest


def test_representation_cache_round_trip_is_keyed_and_provenance_bound(tmp_path):
    caches, manifest = _cohort(tmp_path)
    encoder = PhysicalProjectionEncoder(embedding_dim=8).requires_grad_(False)
    output = tmp_path / "representations"
    build_representation_cache(
        output,
        manifest,
        caches,
        encoder,
        encoder_provenance={"kind": "test", "weights": "fixed"},
        datasets={"train_source"},
        splits={"train", "development"},
    )
    cached = CachedMotionSequenceDataset(
        output, manifest_fingerprint=manifest.fingerprint
    )

    assert len(cached) == 2
    sequence = cached.get("train_source", "train_source-1", "wrist")
    expected = encoder.encode_recording(caches["train_source"][0], stream_id="wrist")
    assert torch.equal(sequence.intervals_sec, expected.intervals_sec)
    assert torch.equal(sequence.valid, expected.valid)
    assert torch.allclose(sequence.embeddings, expected.embeddings)
    assert cached.metadata["encoder_provenance"]["weights"] == "fixed"

    with pytest.raises(ValueError, match="different cohort"):
        CachedMotionSequenceDataset(output, manifest_fingerprint="wrong")


def test_representation_cache_rejects_trainable_encoder_and_existing_output(tmp_path):
    caches, manifest = _cohort(tmp_path)
    encoder = PhysicalProjectionEncoder(embedding_dim=8)
    with pytest.raises(ValueError, match="must be frozen"):
        build_representation_cache(
            tmp_path / "bad",
            manifest,
            caches,
            encoder,
            encoder_provenance={"kind": "test"},
        )

    encoder.requires_grad_(False)
    output = tmp_path / "representations"
    build_representation_cache(
        output,
        manifest,
        caches,
        encoder,
        encoder_provenance={"kind": "test"},
        limit=1,
    )
    with pytest.raises(FileExistsError):
        build_representation_cache(
            output,
            manifest,
            caches,
            encoder,
            encoder_provenance={"kind": "test"},
            limit=1,
        )


class _DetachedButTrainable(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))


def test_write_policy_is_based_on_encoder_state_not_incidental_no_grad(tmp_path):
    caches, manifest = _cohort(tmp_path)
    with pytest.raises(ValueError, match="must be frozen"):
        build_representation_cache(
            tmp_path / "bad",
            manifest,
            caches,
            _DetachedButTrainable(),
            encoder_provenance={"kind": "test"},
        )
