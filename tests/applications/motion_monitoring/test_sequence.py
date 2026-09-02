from __future__ import annotations

import numpy as np
import pytest
import torch
from dataclasses import replace

from applications.motion_monitoring.data.contracts import RawRecording, SensorStream
from applications.motion_monitoring.sequence import (
    MotionSequence,
    PhysicalProjectionEncoder,
    localization_intervals,
    measured_rate_hz,
    stream_to_patch_batch,
)


def _recording(*, missing: bool = False) -> RawRecording:
    timestamps = np.arange(0.0, 3.0, 0.02)
    values = np.stack(
        [
            np.sin(timestamps),
            np.cos(timestamps),
            np.ones_like(timestamps),
            timestamps,
            timestamps * 2,
            timestamps * 3,
        ],
        axis=1,
    ).astype(np.float32)
    valid = np.ones_like(values, dtype=np.bool_)
    if missing:
        valid[50:100, 0] = False
        values[50:100, 0] = np.nan
    stream = SensorStream(
        stream_id="watch",
        placement="right_wrist",
        device="smartwatch",
        timestamps_sec=timestamps,
        values=values,
        channels=("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"),
        valid=valid,
        gravity_state="present",
        nominal_rate_hz=50.0,
    )
    return RawRecording(
        dataset="synthetic",
        recording_id="recording",
        subject_id="subject",
        session_id="session",
        streams=(stream,),
    )


def test_native_patching_preserves_physical_time_and_missingness() -> None:
    recording = _recording(missing=True)
    batch = stream_to_patch_batch(recording.streams[0])
    assert measured_rate_hz(recording.streams[0]) == pytest.approx(50.0)
    assert batch.patches.shape == (3, 50, 6)
    assert batch.patch_intervals_sec[:, 0].tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert batch.patch_valid.tolist() == [True, False, True]
    assert batch.patches[1, :, 0].eq(0).all()
    assert batch.physical_feature_mask[[0, 2], 1:5].all()
    assert not batch.physical_feature_mask[1, 1:5].any()


def test_partial_tail_is_honest_and_not_resampled() -> None:
    recording = _recording()
    stream = recording.streams[0]
    shortened = SensorStream(
        **{
            **stream.__dict__,
            "timestamps_sec": stream.timestamps_sec[:125],
            "values": stream.values[:125],
            "valid": stream.valid[:125],
        }
    )
    batch = stream_to_patch_batch(shortened)
    assert batch.patch_len.tolist() == [50, 50, 25]
    assert batch.durations_sec[-1].item() == pytest.approx(0.5, abs=1e-5)


def test_large_absolute_clock_keeps_subsecond_boundaries() -> None:
    recording = _recording()
    stream = recording.streams[0]
    shifted = SensorStream(
        **{
            **stream.__dict__,
            "timestamps_sec": stream.timestamps_sec + 1_634_178_333.0,
        }
    )
    batch = stream_to_patch_batch(shifted)
    assert batch.patch_intervals_sec.dtype == torch.float64
    assert torch.all(batch.durations_sec > 0.99)


def test_physical_projection_exports_normalized_sequence_and_gradients() -> None:
    encoder = PhysicalProjectionEncoder(embedding_dim=12)
    sequence = encoder.encode_recording(_recording())
    assert isinstance(sequence, MotionSequence)
    assert sequence.embeddings.shape == (3, 12)
    assert torch.allclose(
        sequence.embeddings.norm(dim=1), torch.ones(3), atol=1e-5
    )
    loss = sequence.embeddings[:, 0].sum()
    loss.backward()
    assert encoder.projection[1].weight.grad is not None
    assert torch.isfinite(encoder.projection[1].weight.grad).all()


def test_localization_cells_do_not_inherit_a_wide_receptive_field() -> None:
    sequence = PhysicalProjectionEncoder(embedding_dim=12).encode_recording(_recording())
    wide_support = torch.tensor(
        [[0.0, 5.0], [1.0, 6.0], [2.0, 7.0]], dtype=torch.float64
    )
    sequence = replace(sequence, intervals_sec=wide_support)

    cells = localization_intervals(sequence)

    torch.testing.assert_close(
        cells,
        torch.tensor([[0.0, 3.0], [3.0, 4.0], [4.0, 7.0]], dtype=torch.float64),
    )
