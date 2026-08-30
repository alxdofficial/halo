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


def _recording(recording_id, event):
    timestamps = np.arange(8, dtype=np.float64) + 0.5
    values = np.column_stack([timestamps, timestamps + 1, timestamps + 2]).astype(
        np.float32
    )
    stream = SensorStream(
        stream_id="wrist_acc",
        placement="wrist",
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


def test_synchronized_views_and_targets_crossing_guards_are_rejected():
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
    with pytest.raises(ValueError, match="invalid or guarded"):
        episode_from_recordings(
            reference,
            query,
            _sequence(),
            _sequence(),
            label="pour",
            reference_event_index=0,
            guard_intervals_sec=((5.0, 6.0),),
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
