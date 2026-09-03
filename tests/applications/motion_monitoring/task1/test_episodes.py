import json

import numpy as np
import pytest
import torch

from applications.motion_monitoring.data.cache import write_recording
from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)
from applications.motion_monitoring.task1.episodes import (
    CachedEventPair,
    CachedEventPairDataset,
    DetectionEpisode,
    EmbeddingSequence,
    audit_cached_event_pairs,
    collate_detection_episodes,
    episode_from_recordings,
)


def _sequence(length=8, feature_dim=3, *, invalid=()):
    embeddings = torch.arange(length * feature_dim, dtype=torch.float32).reshape(
        length, feature_dim
    )
    intervals = torch.column_stack(
        [torch.arange(length), torch.arange(1, length + 1)]
    ).float()
    valid = torch.ones(length, dtype=torch.bool)
    valid[list(invalid)] = False
    return EmbeddingSequence(embeddings + 1, intervals, valid)


def _recording(recording_id, event, *, placement="wrist"):
    timestamps = np.arange(8, dtype=np.float64) + 0.5
    values = np.column_stack([timestamps, timestamps + 1, timestamps + 2]).astype(
        np.float32
    )
    stream = SensorStream(
        stream_id="wrist_acc",
        placement=placement,
        device="watch",
        timestamps_sec=timestamps,
        values=values,
        channels=("acc_x", "acc_y", "acc_z"),
        valid=np.ones_like(values, dtype=bool),
        gravity_state="present",
        nominal_rate_hz=1.0,
    )
    return RawRecording(
        dataset="test_source",
        recording_id=recording_id,
        subject_id=f"subject_{recording_id}",
        session_id=recording_id,
        streams=(stream,),
        events=(event,),
    )


def test_collate_preserves_padding_join_guards_and_target_absence():
    present = DetectionEpisode(
        _sequence(3),
        _sequence(7),
        torch.tensor([[2.0, 5.0]]),
        loss_valid=torch.tensor([False, True, True, True, True, True, True]),
    )
    absent = DetectionEpisode(_sequence(4), _sequence(5), torch.empty(0, 2))

    batch = collate_detection_episodes([present, absent], endpoint_tolerance_sec=0.1)

    assert batch.reference.shape == (2, 4, 3)
    assert batch.query.shape == (2, 7, 3)
    assert not batch.query_valid[1, 5:].any()
    assert not batch.loss_valid[0, 0]
    assert batch.endpoint_targets[0, 4]
    assert not batch.endpoint_targets[1].any()
    assert batch.target_valid.tolist() == [[True], [False]]


def test_episode_from_recordings_uses_independent_event_and_excludes_guards():
    reference_recording = _recording("reference", EventInterval(1.0, 4.0, "pour"))
    query_recording = _recording("query", EventInterval(4.0, 7.0, "pour"))

    episode = episode_from_recordings(
        reference_recording,
        query_recording,
        _sequence(),
        _sequence(),
        label="pour",
        reference_event_index=0,
        guard_intervals_sec=((0.0, 1.5),),
    )

    assert len(episode.reference.embeddings) == 3
    assert episode.targets_sec.tolist() == [[4.0, 7.0]]
    assert episode.loss_valid.tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert episode.metadata["reference_recording_id"] == "reference"

    with pytest.raises(ValueError, match="independent recordings"):
        episode_from_recordings(
            reference_recording,
            reference_recording,
            _sequence(),
            _sequence(),
            label="pour",
            reference_event_index=0,
        )


def test_episode_rejects_a_target_cut_by_the_query_crop():
    reference_recording = _recording("reference", EventInterval(1.0, 3.0, "pour"))
    query_recording = _recording("query", EventInterval(3.0, 6.0, "pour"))

    with pytest.raises(ValueError, match="incomplete at the query crop boundary"):
        episode_from_recordings(
            reference_recording,
            query_recording,
            _sequence(),
            _sequence(length=4),
            label="pour",
            reference_event_index=0,
        )


def test_cached_event_pair_dataset_loads_real_cache_contract(tmp_path):
    reference = _recording("reference", EventInterval(1.0, 4.0, "pour"))
    query = _recording("query", EventInterval(4.0, 7.0, "pour"))
    write_recording(reference, tmp_path / "reference")
    write_recording(query, tmp_path / "query")
    (tmp_path / "cache.json").write_text(
        json.dumps(
            {"schema_version": 1, "dataset": "test_source", "recording_count": 2}
        )
    )
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"dataset": "test_source", "directory": "reference"}),
                json.dumps({"dataset": "test_source", "directory": "query"}),
            ]
        )
        + "\n"
    )

    def provider(recording, stream_id):
        assert recording.streams[0].stream_id == stream_id
        return _sequence()

    dataset = CachedEventPairDataset(
        tmp_path,
        [CachedEventPair(0, 1, "pour", 0, "wrist_acc", "wrist_acc")],
        provider,
        validate_provenance=False,
    )

    episode = dataset[0]
    assert episode.targets_sec.tolist() == [[4.0, 7.0]]
    assert episode.metadata["dataset"] == "test_source"


def test_cached_pair_rejects_cross_placement_shortcut(tmp_path):
    reference = _recording("reference", EventInterval(1.0, 4.0, "pour"))
    query = _recording(
        "query", EventInterval(4.0, 7.0, "pour"), placement="ankle"
    )
    write_recording(reference, tmp_path / "reference")
    write_recording(query, tmp_path / "query")
    (tmp_path / "cache.json").write_text(
        json.dumps(
            {"schema_version": 1, "dataset": "test_source", "recording_count": 2}
        )
    )
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"dataset": "test_source", "directory": "reference"}),
                json.dumps({"dataset": "test_source", "directory": "query"}),
            ]
        )
        + "\n"
    )

    dataset = CachedEventPairDataset(
        tmp_path,
        [CachedEventPair(0, 1, "pour", 0, "wrist_acc", "wrist_acc")],
        lambda recording, stream_id: _sequence(),
        validate_provenance=False,
    )
    with pytest.raises(ValueError, match="incompatible acquisition configurations"):
        dataset[0]


def test_episode_rejects_one_sided_sensor_metadata():
    reference = _recording("reference", EventInterval(1.0, 4.0, "pour"))
    query = _recording("query", EventInterval(4.0, 7.0, "pour"))
    configured = EmbeddingSequence(
        **{
            **_sequence().__dict__,
            "metadata": {
                "device": "smartwatch",
                "placement": "wrist",
                "channels": ("acc_x", "acc_y", "acc_z"),
                "gravity_state": "present",
            },
        }
    )
    with pytest.raises(ValueError, match="both declare sensor configurations"):
        episode_from_recordings(
            reference,
            query,
            configured,
            _sequence(),
            label="pour",
            reference_event_index=0,
        )


def test_cached_pair_audit_filters_invalid_events_before_training(tmp_path):
    reference = _recording("reference", EventInterval(1.0, 5.0, "pour"))
    query = _recording("query", EventInterval(4.0, 7.0, "pour"))
    write_recording(reference, tmp_path / "reference")
    write_recording(query, tmp_path / "query")
    (tmp_path / "cache.json").write_text(
        json.dumps(
            {"schema_version": 1, "dataset": "test_source", "recording_count": 2}
        )
    )
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"dataset": "test_source", "directory": "reference"}),
                json.dumps({"dataset": "test_source", "directory": "query"}),
            ]
        )
        + "\n"
    )
    pairs = (
        CachedEventPair(0, 1, "pour", 0, "wrist_acc", "wrist_acc"),
    )

    def provider(recording, stream_id):
        del stream_id
        return _sequence(invalid=(2,)) if recording.recording_id == "reference" else _sequence()

    audit = audit_cached_event_pairs(
        tmp_path, pairs, provider, validate_provenance=False
    )
    assert not audit.eligible_pairs
    assert len(audit.rejected_pairs) == 1
    assert "invalid embedding gap" in audit.rejected_pairs[0].reason


def test_reference_cannot_bridge_an_invalid_embedding_gap():
    reference = _recording("reference", EventInterval(1.0, 5.0, "pour"))
    query = _recording("query", EventInterval(4.0, 7.0, "pour"))
    with pytest.raises(ValueError, match="invalid embedding gap"):
        episode_from_recordings(
            reference,
            query,
            _sequence(invalid=(2,)),
            _sequence(),
            label="pour",
            reference_event_index=0,
        )


def test_synchronized_views_are_rejected_and_guards_only_mask_supervision():
    event = EventInterval(1.0, 4.0, "pour")
    reference = _recording("reference", event)
    synchronized = RawRecording(
        dataset=reference.dataset,
        recording_id="second_sensor_view",
        subject_id=reference.subject_id,
        session_id=reference.session_id,
        streams=reference.streams,
        events=(event,),
    )
    with pytest.raises(ValueError, match="synchronized views"):
        episode_from_recordings(
            reference,
            synchronized,
            _sequence(),
            _sequence(),
            label="pour",
            reference_event_index=0,
        )

    query = _recording("query", EventInterval(4.0, 7.0, "pour"))
    episode = episode_from_recordings(
        reference,
        query,
        _sequence(),
        _sequence(),
        label="pour",
        reference_event_index=0,
        guard_intervals_sec=((5.0, 6.0),),
    )
    assert episode.alignment_valid.all()
    assert episode.loss_valid.tolist() == [True, True, True, True, True, False, True, True]


def test_cropped_views_of_one_source_recording_are_not_independent():
    event = EventInterval(1.0, 4.0, "pour")
    reference = _recording("reference-crop", event)
    query = _recording("query-crop", EventInterval(2.0, 4.0, "pour"))
    reference = RawRecording(
        **{**reference.__dict__, "metadata": {"source_recording_id": "root"}}
    )
    query = RawRecording(
        **{**query.__dict__, "metadata": {"source_recording_id": "root"}}
    )
    with pytest.raises(ValueError, match="independent recordings"):
        episode_from_recordings(
            reference,
            query,
            _sequence(),
            _sequence(),
            label="pour",
            reference_event_index=0,
        )


def test_episode_preserves_large_absolute_clock_precision():
    offset = 1_634_178_333.0
    intervals = torch.column_stack(
        [torch.arange(8, dtype=torch.float64), torch.arange(1, 9, dtype=torch.float64)]
    ) + offset
    sequence = EmbeddingSequence(
        torch.randn(8, 3), intervals, torch.ones(8, dtype=torch.bool)
    )
    reference = _recording("reference", EventInterval(offset + 1, offset + 4, "pour"))
    query = _recording("query", EventInterval(offset + 4, offset + 7, "pour"))
    episode = episode_from_recordings(
        reference,
        query,
        sequence,
        sequence,
        label="pour",
        reference_event_index=0,
    )
    assert episode.query.intervals_sec.dtype == torch.float64
    assert episode.targets_sec.dtype == torch.float64
    assert torch.all(episode.targets_sec[:, 1] > episode.targets_sec[:, 0])
