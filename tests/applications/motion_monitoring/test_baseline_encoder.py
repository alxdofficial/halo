from __future__ import annotations

import numpy as np
import pytest
import torch

import baselines
from baselines.base import BaselineAdapter, InputContract
from applications.motion_monitoring.baseline_encoder import BaselineMotionEncoder
from applications.motion_monitoring.data.contracts import RawRecording, SensorStream
from baselines.unimts.adapter import _joint_for, _resample_to_20hz


class _LengthAwareAdapter(BaselineAdapter):
    name = "test_length_aware"
    contract = InputContract(
        channels=("acc_x", "acc_y", "acc_z"), rate_hz=10.0, window_sec=2.0
    )

    def setup(self, device):
        return {"lengths": []}

    def window_features(self, stream, state, device):
        state["lengths"].extend([stream.windows.shape[1]] * stream.n_windows)
        mean = stream.windows[:, :, :3].mean(axis=1)
        spread = stream.windows[:, :, :3].std(axis=1)
        return np.concatenate((mean, spread), axis=1).astype(np.float32)


def _recording() -> RawRecording:
    timestamps = np.arange(0.0, 2.5, 0.1)
    values = np.stack(
        (np.ones_like(timestamps), timestamps + 0.5, np.square(timestamps) + 0.1),
        axis=1,
    ).astype(np.float32)
    stream = SensorStream(
        stream_id="watch",
        placement="right_wrist",
        device="watch",
        timestamps_sec=timestamps,
        values=values,
        channels=("acc_x", "acc_y", "acc_z"),
        valid=np.ones_like(values, dtype=np.bool_),
        gravity_state="present",
        nominal_rate_hz=10.0,
    )
    return RawRecording("test", "recording", "subject", "session", (stream,))


def test_baseline_encoder_uses_native_window_and_preserves_partial_length(
    monkeypatch,
) -> None:
    adapter = _LengthAwareAdapter()
    monkeypatch.setitem(baselines.REGISTRY, adapter.name, adapter)
    encoder = BaselineMotionEncoder(adapter.name)

    sequence = encoder.encode_recording(_recording(), stride_seconds=1.0)

    assert sequence.embeddings.shape == (3, 6)
    assert sorted(encoder._state["lengths"]) == [5, 15, 20]
    assert torch.allclose(sequence.embeddings.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert not any(parameter.requires_grad for parameter in encoder.parameters())


def test_application_encoder_uses_feature_only_setup(monkeypatch, tmp_path) -> None:
    artifact = tmp_path / "encoder.pt"
    artifact.write_bytes(b"released encoder")

    class FeatureOnlyAdapter(_LengthAwareAdapter):
        name = "test_feature_only"

        def setup(self, device):
            raise AssertionError("native prediction setup must not run")

        def setup_features(self, device):
            return {"lengths": [], "feature_only": True}

        def feature_artifacts(self, state):
            return {"checkpoint": artifact}

    adapter = FeatureOnlyAdapter()
    monkeypatch.setitem(baselines.REGISTRY, adapter.name, adapter)
    encoder = BaselineMotionEncoder(adapter.name)
    sequence = encoder.encode_recording(_recording())
    provenance = encoder.provenance()

    assert sequence.embeddings.shape[1] == 6
    assert encoder._state["feature_only"]
    assert provenance["artifacts"]["checkpoint"]["bytes"] == len(b"released encoder")
    assert len(provenance["artifacts"]["checkpoint"]["sha256"]) == 64


def test_application_encoder_rejects_missing_feature_artifact(monkeypatch, tmp_path) -> None:
    class MissingArtifactAdapter(_LengthAwareAdapter):
        name = "test_missing_feature_artifact"

        def feature_artifacts(self, state):
            return {"checkpoint": tmp_path / "missing.pt"}

    adapter = MissingArtifactAdapter()
    monkeypatch.setitem(baselines.REGISTRY, adapter.name, adapter)
    with pytest.raises(FileNotFoundError, match="feature artifact"):
        BaselineMotionEncoder(adapter.name).provenance()


def test_unimts_preserves_one_source_sample_for_wrap_padding() -> None:
    source = np.ones((1, 1, 3), dtype=np.float32)
    resampled = _resample_to_20hz(source, rate_hz=50.0)
    assert resampled.shape == (1, 1, 3)


def test_unimts_application_stream_mapping_preserves_laterality() -> None:
    class Stream:
        dataset = "test"

        def __init__(self, stream: str) -> None:
            self.stream = stream

    assert _joint_for(Stream("left_arm_left_wrist")) == 17
    assert _joint_for(Stream("right_arm_right_wrist")) == 21
    assert _joint_for(Stream("imu0_right upper arm")) == 19
    assert _joint_for(Stream("imu2_left upper arm")) == 15
