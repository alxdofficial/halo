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
    BoundedSegment,
    CachedMotionSequenceDataset,
    bounded_representation_id,
    build_bounded_representation_cache,
    build_representation_cache,
    require_complete_representations,
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


def test_limited_representation_cache_cannot_be_used_for_official_common_units(tmp_path):
    caches, manifest = _cohort(tmp_path)
    encoder = PhysicalProjectionEncoder(embedding_dim=8).requires_grad_(False)
    output = tmp_path / "pilot"
    build_representation_cache(
        output,
        manifest,
        caches,
        encoder,
        encoder_provenance={"kind": "test"},
        limit=1,
    )
    with pytest.raises(ValueError, match="rejects --limit pilot"):
        require_complete_representations(CachedMotionSequenceDataset(output))


def test_representation_cache_resumes_valid_staging_sequences(tmp_path):
    caches, manifest = _cohort(tmp_path)
    encoder = PhysicalProjectionEncoder(embedding_dim=8).requires_grad_(False)
    partial = tmp_path / "partial"
    output = tmp_path / "resumed"
    build_representation_cache(
        partial,
        manifest,
        caches,
        encoder,
        encoder_provenance={"kind": "test"},
        limit=1,
    )
    partial.rename(output.with_name(f".{output.name}.staging"))

    build_representation_cache(
        output,
        manifest,
        caches,
        encoder,
        encoder_provenance={"kind": "test"},
        resume=True,
    )

    cached = CachedMotionSequenceDataset(output)
    assert len(cached) == 3
    assert cached.get("train_source", "train_source-1", "wrist").embeddings.shape[1] == 8


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


def _second_cohort(tmp_path, caches):
    """A different cohort sharing ``test_source`` with the first one."""

    extra = _cache(
        tmp_path, "synthetic_source", (_recording("synthetic_source", "bg:1", 1),)
    )
    new_caches = {"test_source": caches["test_source"], "synthetic_source": extra}
    manifest = build_cohort_manifest(
        new_caches,
        name="second",
        train_only_datasets=("synthetic_source",),
        evaluation_datasets=("test_source",),
    )
    return new_caches, manifest


def test_cache_accepts_other_cohort_on_matching_raw_fingerprints_and_unions(tmp_path):
    from applications.motion_monitoring.representation_cache import (
        MotionSequenceUnion,
        open_representations,
    )

    caches, first = _cohort(tmp_path)
    encoder = PhysicalProjectionEncoder(embedding_dim=8).requires_grad_(False)
    natural = tmp_path / "natural"
    build_representation_cache(
        natural, first, caches, encoder, encoder_provenance={"kind": "test", "weights": "w"}
    )
    new_caches, second = _second_cohort(tmp_path, caches)
    assert second.fingerprint != first.fingerprint

    # Same raw canonical cache for test_source -> exposed; train_source hidden.
    bound = CachedMotionSequenceDataset(
        natural,
        manifest_fingerprint=second.fingerprint,
        cache_fingerprints=second.cache_fingerprints,
    )
    assert bound.datasets == ("test_source",)
    with pytest.raises(KeyError):
        bound.get("train_source", "train_source-1", "wrist")

    synthetic = tmp_path / "synthetic"
    build_representation_cache(
        synthetic, second, new_caches, encoder,
        encoder_provenance={"kind": "test", "weights": "w"},
        datasets={"synthetic_source"},
        stride_seconds=0.5,
    )
    union = open_representations([natural, synthetic], cohort=second)
    assert isinstance(union, MotionSequenceUnion)
    assert union.datasets == ("synthetic_source", "test_source")
    assert union.get("synthetic_source", "synthetic_source-1", "wrist").embeddings.shape[1] == 8
    assert union.get("test_source", "test_source-1", "wrist").embeddings.shape[1] == 8
    assert json.loads(json.dumps(union.metadata))["datasets"] == ["synthetic_source", "test_source"]
    assert union.metadata["stride_seconds_by_dataset"] == {
        "synthetic_source": 0.5,
        "test_source": 1.0,
    }

    # A cache serving no dataset of the cohort is refused outright.
    lonely = _cache(tmp_path, "lonely", (_recording("lonely", "x", 1),))
    lonely_test = _cache(tmp_path, "lonely_test", (_recording("lonely_test", "y", 1),))
    lonely_cohort = build_cohort_manifest(
        {"lonely": lonely, "lonely_test": lonely_test},
        name="lonely",
        train_only_datasets=("lonely",),
        evaluation_datasets=("lonely_test",),
    )
    with pytest.raises(ValueError, match="shares no canonical cache"):
        open_representations([natural], cohort=lonely_cohort)

    # Overlapping caches are allowed only when their sequence keys are disjoint.
    with pytest.raises(ValueError, match="duplicate sequence identities"):
        open_representations([natural, natural], cohort=second)

    # Different encoders cannot be unioned either.
    other = tmp_path / "other_encoder"
    build_representation_cache(
        other, second, new_caches, encoder,
        encoder_provenance={"kind": "test", "weights": "different"},
        datasets={"synthetic_source"},
    )
    with pytest.raises(ValueError, match="different encoders"):
        open_representations([natural, other], cohort=second)


def test_bounded_representation_sees_only_the_event_and_unions_with_timeline(tmp_path):
    from applications.motion_monitoring.representation_cache import open_representations

    caches, manifest = _cohort(tmp_path)
    encoder = PhysicalProjectionEncoder(embedding_dim=8).requires_grad_(False)
    provenance = {"kind": "test", "weights": "bounded"}
    timeline = tmp_path / "timeline"
    bounded = tmp_path / "bounded"
    build_representation_cache(
        timeline,
        manifest,
        caches,
        encoder,
        encoder_provenance=provenance,
        datasets={"train_source"},
        stride_seconds=0.25,
    )
    segment = BoundedSegment(
        dataset="train_source",
        cache_index=0,
        recording_id="train_source-1",
        stream_id="wrist",
        event_index=0,
        start_sec=0.2,
        end_sec=1.2,
    )
    build_bounded_representation_cache(
        bounded,
        manifest,
        caches,
        encoder,
        [segment],
        encoder_provenance=provenance,
        stride_seconds=0.25,
    )
    union = open_representations([timeline, bounded], cohort=manifest)
    derived = union.get(
        "train_source", bounded_representation_id("train_source-1", 0), "wrist"
    )
    assert float(derived.intervals_sec[0, 0]) >= 0.2 - 1e-6
    assert float(derived.intervals_sec[-1, 1]) <= 1.2 + 1e-6
    assert union.get("train_source", "train_source-1", "wrist").recording_id == "train_source-1"
