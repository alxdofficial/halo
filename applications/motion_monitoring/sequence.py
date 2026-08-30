"""Shared timestamped representation contract for the application tasks.

The application layer keeps source recordings at their native sampling rate.  This
module creates physical-time patches without resampling and adapts either HALO or a
small physical-feature projection to the same ``MotionSequence`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from applications.motion_monitoring.data.contracts import (
    CANONICAL_CHANNELS,
    RawRecording,
    SensorStream,
)


PHYSICAL_FEATURE_NAMES = (
    "valid_fraction",
    "acc_magnitude_mean_g",
    "acc_magnitude_std_g",
    "acc_dynamic_rms_g",
    "acc_jerk_rms_g_per_s",
    "gyro_magnitude_mean_rad_per_s",
    "gyro_magnitude_std_rad_per_s",
    "gyro_dynamic_rms_rad_per_s",
    "gyro_jerk_rms_rad_per_s2",
)


@dataclass(frozen=True)
class NativePatchBatch:
    """One native-rate sensor stream represented as padded physical-time patches."""

    patches: torch.Tensor
    patch_len: torch.Tensor
    patch_intervals_sec: torch.Tensor
    patch_valid: torch.Tensor
    channel_mask: torch.Tensor
    channel_valid_fraction: torch.Tensor
    sampling_rate_hz: float
    source_rate_hz: float
    stream: SensorStream
    physical_features: torch.Tensor
    physical_feature_mask: torch.Tensor

    def __post_init__(self) -> None:
        p, storage, channels = self.patches.shape
        if channels != len(CANONICAL_CHANNELS) or storage <= 0 or p <= 0:
            raise ValueError("patches must have shape [P, storage, 6]")
        if self.patch_len.shape != (p,) or self.patch_intervals_sec.shape != (p, 2):
            raise ValueError("patch lengths or physical intervals disagree with patches")
        if self.patch_valid.shape != (p,) or self.patch_valid.dtype != torch.bool:
            raise ValueError("patch_valid must be a boolean [P] tensor")
        if self.channel_mask.shape != (channels,) or self.channel_mask.dtype != torch.bool:
            raise ValueError("channel_mask must be a boolean [6] tensor")
        if self.channel_valid_fraction.shape != (p, channels):
            raise ValueError("channel validity must have shape [P, 6]")
        if self.physical_features.shape != (p, len(PHYSICAL_FEATURE_NAMES)):
            raise ValueError("physical feature shape disagrees with feature names")
        if self.physical_feature_mask.shape != self.physical_features.shape:
            raise ValueError("physical feature values and masks must have identical shapes")
        if not torch.isfinite(self.patches).all() or not torch.isfinite(self.physical_features).all():
            raise ValueError("patch data and physical summaries must be finite")
        if not torch.all(self.patch_intervals_sec[:, 1] > self.patch_intervals_sec[:, 0]):
            raise ValueError("every patch must cover positive physical time")

    @property
    def positions_sec(self) -> torch.Tensor:
        return self.patch_intervals_sec.mean(dim=1)

    @property
    def durations_sec(self) -> torch.Tensor:
        return self.patch_intervals_sec[:, 1] - self.patch_intervals_sec[:, 0]

    def to(self, device: torch.device | str) -> "NativePatchBatch":
        return NativePatchBatch(
            patches=self.patches.to(device),
            patch_len=self.patch_len.to(device),
            patch_intervals_sec=self.patch_intervals_sec.to(device),
            patch_valid=self.patch_valid.to(device),
            channel_mask=self.channel_mask.to(device),
            channel_valid_fraction=self.channel_valid_fraction.to(device),
            sampling_rate_hz=self.sampling_rate_hz,
            source_rate_hz=self.source_rate_hz,
            stream=self.stream,
            physical_features=self.physical_features.to(device),
            physical_feature_mask=self.physical_feature_mask.to(device),
        )


@dataclass(frozen=True)
class MotionSequence:
    """Normalized patch embeddings and their auditable physical-time provenance."""

    embeddings: torch.Tensor
    intervals_sec: torch.Tensor
    valid: torch.Tensor
    physical_features: torch.Tensor
    physical_feature_mask: torch.Tensor
    physical_feature_names: tuple[str, ...]
    dataset: str
    recording_id: str
    subject_id: str
    session_id: str
    stream_id: str
    placement: str
    device: str
    channels: tuple[str, ...]
    gravity_state: str
    sampling_rate_hz: float

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2 or not len(self.embeddings):
            raise ValueError("embeddings must be non-empty [P, D]")
        p = len(self.embeddings)
        if self.intervals_sec.shape != (p, 2) or self.valid.shape != (p,):
            raise ValueError("intervals and validity must align with embeddings")
        if self.valid.dtype != torch.bool:
            raise ValueError("sequence validity must be boolean")
        if self.physical_features.shape[0] != p:
            raise ValueError("physical summaries must align with embeddings")
        if self.physical_features.shape != self.physical_feature_mask.shape:
            raise ValueError("physical values and masks must have identical shape")
        if self.physical_features.shape[1] != len(self.physical_feature_names):
            raise ValueError("physical feature names disagree with values")
        if not torch.isfinite(self.embeddings).all():
            raise ValueError("embeddings must be finite")
        norms = torch.linalg.vector_norm(self.embeddings[self.valid], dim=-1)
        if len(norms) and not torch.allclose(norms, torch.ones_like(norms), atol=2e-4, rtol=2e-4):
            raise ValueError("valid MotionSequence embeddings must be unit normalized")

    @property
    def duration_sec(self) -> float:
        return float((self.intervals_sec[-1, 1] - self.intervals_sec[0, 0]).item())


class MotionEncoder(Protocol):
    def encode_recording(
        self,
        recording: RawRecording,
        *,
        stream_id: str | None = None,
        patch_seconds: float = 1.0,
        stride_seconds: float = 1.0,
    ) -> MotionSequence: ...


def measured_rate_hz(stream: SensorStream) -> float:
    """Robust rate estimate used for storage sizing, never to rewrite source time."""

    differences = np.diff(np.asarray(stream.timestamps_sec, dtype=np.float64))
    if not len(differences):
        if stream.nominal_rate_hz is None:
            raise ValueError("a one-sample stream requires a declared nominal rate")
        return float(stream.nominal_rate_hz)
    return float(1.0 / np.median(differences))


def select_stream(recording: RawRecording, stream_id: str | None = None) -> SensorStream:
    if stream_id is not None:
        matches = [stream for stream in recording.streams if stream.stream_id == stream_id]
        if len(matches) != 1:
            raise KeyError(f"recording {recording.recording_id!r} has no unique stream {stream_id!r}")
        return matches[0]
    # Prefer the richest stream; break ties by the stable source order.
    return max(recording.streams, key=lambda stream: len(stream.channels))


def _modality_features(
    values: np.ndarray,
    valid: np.ndarray,
    timestamps: np.ndarray,
    columns: Sequence[int],
    *,
    acceleration: bool,
) -> tuple[list[float], list[bool]]:
    if len(columns) != 3:
        return [0.0] * 4, [False] * 4
    triad_valid = valid[:, columns].all(axis=1)
    if not np.any(triad_valid):
        return [0.0] * 4, [False] * 4
    triad = values[triad_valid][:, columns].astype(np.float64, copy=False)
    times = timestamps[triad_valid]
    magnitude = np.linalg.norm(triad, axis=1)
    mean = float(magnitude.mean())
    std = float(magnitude.std())
    dynamic = triad - triad.mean(axis=0, keepdims=True)
    dynamic_rms = float(np.sqrt(np.mean(dynamic * dynamic)))
    if len(triad) >= 2:
        dt = np.diff(times)
        keep = dt > 0
        derivative = np.diff(triad, axis=0)[keep] / dt[keep, None]
        jerk = float(np.sqrt(np.mean(derivative * derivative))) if len(derivative) else 0.0
        jerk_valid = bool(len(derivative))
    else:
        jerk, jerk_valid = 0.0, False
    return [mean, std, dynamic_rms, jerk], [True, True, True, jerk_valid]


def stream_to_patch_batch(
    stream: SensorStream,
    *,
    patch_seconds: float = 1.0,
    stride_seconds: float = 1.0,
    min_valid_fraction: float = 0.8,
    max_patch_samples: int = 512,
) -> NativePatchBatch:
    """Window one stream on its source clock without resampling or interpolation."""

    if patch_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("patch and stride durations must be positive")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("minimum valid fraction must be in (0, 1]")
    timestamps = np.asarray(stream.timestamps_sec, dtype=np.float64)
    source_values = np.asarray(stream.values, dtype=np.float32)
    source_valid = np.asarray(stream.valid, dtype=np.bool_)
    channel_lookup = {name: index for index, name in enumerate(stream.channels)}
    canonical_values = np.zeros((len(timestamps), len(CANONICAL_CHANNELS)), dtype=np.float32)
    canonical_valid = np.zeros_like(canonical_values, dtype=np.bool_)
    channel_mask = np.zeros(len(CANONICAL_CHANNELS), dtype=np.bool_)
    for canonical_index, name in enumerate(CANONICAL_CHANNELS):
        if name not in channel_lookup:
            continue
        source_index = channel_lookup[name]
        live = source_valid[:, source_index]
        canonical_values[live, canonical_index] = source_values[live, source_index]
        canonical_valid[:, canonical_index] = live
        channel_mask[canonical_index] = True

    rate = measured_rate_hz(stream)
    # Include a final honest partial patch. Searchsorted below determines its real support.
    first, stop = float(timestamps[0]), float(timestamps[-1]) + 1.0 / rate
    starts = np.arange(first, stop, stride_seconds, dtype=np.float64)
    bounds: list[tuple[int, int, float, float]] = []
    for start in starts:
        end = min(start + patch_seconds, stop)
        left = int(np.searchsorted(timestamps, start, side="left"))
        right = int(np.searchsorted(timestamps, end, side="left"))
        if right <= left:
            continue
        count = right - left
        if count > max_patch_samples:
            raise ValueError(
                f"{count} native samples exceed max_patch_samples={max_patch_samples}; "
                "increase storage rather than silently truncating the patch"
            )
        represented_end = min(end, float(timestamps[right - 1]) + 1.0 / rate)
        bounds.append((left, right, start, represented_end))
    if not bounds:
        raise ValueError("stream did not produce any non-empty physical-time patch")

    storage = max(right - left for left, right, _, _ in bounds)
    patches = np.zeros((len(bounds), storage, len(CANONICAL_CHANNELS)), dtype=np.float32)
    lengths = np.zeros(len(bounds), dtype=np.int64)
    # Source clocks may be Unix timestamps around 1e9 seconds. Float32 cannot
    # distinguish one-second boundaries at that magnitude, so provenance stays FP64.
    intervals = np.zeros((len(bounds), 2), dtype=np.float64)
    fractions = np.zeros((len(bounds), len(CANONICAL_CHANNELS)), dtype=np.float32)
    patch_valid = np.zeros(len(bounds), dtype=np.bool_)
    physical = np.zeros((len(bounds), len(PHYSICAL_FEATURE_NAMES)), dtype=np.float32)
    physical_mask = np.zeros_like(physical, dtype=np.bool_)
    present_indices = np.flatnonzero(channel_mask)
    acc_columns = [i for i, name in enumerate(CANONICAL_CHANNELS) if name.startswith("acc_") and channel_mask[i]]
    gyro_columns = [i for i, name in enumerate(CANONICAL_CHANNELS) if name.startswith("gyro_") and channel_mask[i]]

    for patch_index, (left, right, start, end) in enumerate(bounds):
        count = right - left
        patches[patch_index, :count] = canonical_values[left:right]
        lengths[patch_index] = count
        intervals[patch_index] = (start, end)
        fractions[patch_index] = canonical_valid[left:right].mean(axis=0)
        patch_valid[patch_index] = bool(
            len(present_indices)
            and np.all(fractions[patch_index, present_indices] >= min_valid_fraction)
        )
        valid_fraction = float(fractions[patch_index, present_indices].mean())
        physical[patch_index, 0] = valid_fraction
        physical_mask[patch_index, 0] = True
        acc_values, acc_mask = _modality_features(
            canonical_values[left:right], canonical_valid[left:right], timestamps[left:right],
            acc_columns, acceleration=True,
        )
        gyro_values, gyro_mask = _modality_features(
            canonical_values[left:right], canonical_valid[left:right], timestamps[left:right],
            gyro_columns, acceleration=False,
        )
        physical[patch_index, 1:5] = acc_values
        physical_mask[patch_index, 1:5] = acc_mask
        physical[patch_index, 5:9] = gyro_values
        physical_mask[patch_index, 5:9] = gyro_mask

    return NativePatchBatch(
        patches=torch.from_numpy(patches),
        patch_len=torch.from_numpy(lengths),
        patch_intervals_sec=torch.from_numpy(intervals),
        patch_valid=torch.from_numpy(patch_valid),
        channel_mask=torch.from_numpy(channel_mask),
        channel_valid_fraction=torch.from_numpy(fractions),
        sampling_rate_hz=rate,
        source_rate_hz=float(stream.nominal_rate_hz or rate),
        stream=stream,
        physical_features=torch.from_numpy(physical),
        physical_feature_mask=torch.from_numpy(physical_mask),
    )


def _motion_sequence(
    recording: RawRecording,
    batch: NativePatchBatch,
    embeddings: torch.Tensor,
) -> MotionSequence:
    safe = torch.where(batch.patch_valid.unsqueeze(-1), embeddings, torch.zeros_like(embeddings))
    # A rejected patch remains explicit and finite; it must not contribute to task losses.
    normalized = F.normalize(safe, dim=-1, eps=1e-8)
    return MotionSequence(
        embeddings=normalized,
        intervals_sec=batch.patch_intervals_sec.to(normalized.device),
        valid=batch.patch_valid.to(normalized.device),
        physical_features=batch.physical_features.to(normalized.device),
        physical_feature_mask=batch.physical_feature_mask.to(normalized.device),
        physical_feature_names=PHYSICAL_FEATURE_NAMES,
        dataset=recording.dataset,
        recording_id=recording.recording_id,
        subject_id=recording.subject_id,
        session_id=recording.session_id,
        stream_id=batch.stream.stream_id,
        placement=batch.stream.placement,
        device=batch.stream.device,
        channels=tuple(batch.stream.channels),
        gravity_state=batch.stream.gravity_state,
        sampling_rate_hz=batch.sampling_rate_hz,
    )


class PhysicalProjectionEncoder(nn.Module):
    """Cheap trainable control used for plumbing and gradient smoke tests."""

    def __init__(self, embedding_dim: int = 32) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding dimension must be positive")
        self.projection = nn.Sequential(
            nn.LayerNorm(len(PHYSICAL_FEATURE_NAMES)),
            nn.Linear(len(PHYSICAL_FEATURE_NAMES), embedding_dim),
        )

    def encode_recording(
        self,
        recording: RawRecording,
        *,
        stream_id: str | None = None,
        patch_seconds: float = 1.0,
        stride_seconds: float = 1.0,
    ) -> MotionSequence:
        stream = select_stream(recording, stream_id)
        batch = stream_to_patch_batch(
            stream, patch_seconds=patch_seconds, stride_seconds=stride_seconds
        ).to(next(self.parameters()).device)
        values = batch.physical_features * batch.physical_feature_mask.to(
            batch.physical_features.dtype
        )
        return _motion_sequence(recording, batch, self.projection(values))


class HaloMotionEncoder(nn.Module):
    """Adapt a ``SetTokenizerEncoder`` to complete native-time recordings."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        max_patches_per_call: int = 256,
        context_patches: int = 16,
    ) -> None:
        super().__init__()
        if max_patches_per_call <= 2 * context_patches:
            raise ValueError("chunk size must exceed twice the temporal context")
        self.encoder = encoder
        self.max_patches_per_call = int(max_patches_per_call)
        self.context_patches = int(context_patches)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: torch.device | str = "cpu",
        trainable: bool = False,
        max_patches_per_call: int = 256,
        context_patches: int = 16,
    ) -> "HaloMotionEncoder":
        from training.tokenizer.eval_transfer import build_encoder

        device = torch.device(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        encoder = build_encoder(payload, device, training=trainable)
        encoder.requires_grad_(trainable)
        return cls(
            encoder,
            max_patches_per_call=max_patches_per_call,
            context_patches=context_patches,
        )

    @staticmethod
    def _text_metadata(stream: SensorStream) -> tuple[list[str], list[str], list[int]]:
        roles = ["x-axis", "y-axis", "z-axis", "x-axis", "y-axis", "z-axis"]
        present_acc = all(name in stream.channels for name in CANONICAL_CHANNELS[:3])
        present_gyro = all(name in stream.channels for name in CANONICAL_CHANNELS[3:])
        sensors: list[str] = []
        accel_id = gyro_id = 0
        if present_acc:
            accel_id = len(sensors)
            sensors.append(
                f"a {stream.device} accelerometer on the {stream.placement}; "
                f"gravity {stream.gravity_state}"
            )
        if present_gyro:
            gyro_id = len(sensors)
            sensors.append(f"a {stream.device} gyroscope on the {stream.placement}")
        if not sensors:
            raise ValueError("HALO requires a complete accelerometer or gyroscope xyz triad")
        sensor_id = [accel_id] * 3 + [gyro_id] * 3
        return roles, sensors, sensor_id

    def _encode_chunk(
        self,
        batch: NativePatchBatch,
        start: int,
        end: int,
    ) -> torch.Tensor:
        device = next(self.encoder.parameters()).device
        stream = batch.stream
        roles, sensors, sensor_id = self._text_metadata(stream)
        patches = batch.patches[start:end]
        expected_storage = getattr(getattr(self.encoder, "filterbank", None), "S", None)
        if expected_storage is not None:
            expected_storage = int(expected_storage)
            if patches.shape[1] > expected_storage:
                raise ValueError(
                    f"native patch needs {patches.shape[1]} samples but the HALO checkpoint "
                    f"supports only {expected_storage}"
                )
            if patches.shape[1] < expected_storage:
                patches = F.pad(patches, (0, 0, 0, expected_storage - patches.shape[1]))
        positions = (
            batch.positions_sec[start:end] - batch.patch_intervals_sec[0, 0]
        ).to(torch.float32)
        kwargs = dict(
            patch_durations=batch.durations_sec[start:end].unsqueeze(0),
            channel_mask=batch.channel_mask.unsqueeze(0),
            patch_padding_mask=batch.patch_valid[start:end].unsqueeze(0),
            sensor_texts=[sensors],
            sensor_id=torch.tensor([sensor_id], dtype=torch.long, device=device),
            source_rate_hz=torch.tensor([batch.source_rate_hz], device=device),
        )
        if bool(getattr(self.encoder, "use_sensor_bias_conditioning", False)):
            raise ValueError(
                "legacy stream-statistic-conditioned HALO checkpoints are not supported by the "
                "application adapter; use a checkpoint with use_sensor_bias_conditioning=False"
            )
        output = self.encoder(
            patches.unsqueeze(0),
            torch.tensor([batch.sampling_rate_hz], device=device),
            batch.patch_len[start:end].unsqueeze(0),
            [roles],
            positions.unsqueeze(0),
            **kwargs,
        )
        if "per_patch" not in output:
            raise KeyError("HALO encoder did not return per_patch representations")
        return output["per_patch"][0]

    def encode_recording(
        self,
        recording: RawRecording,
        *,
        stream_id: str | None = None,
        patch_seconds: float = 1.0,
        stride_seconds: float = 1.0,
    ) -> MotionSequence:
        stream = select_stream(recording, stream_id)
        device = next(self.encoder.parameters()).device
        batch = stream_to_patch_batch(
            stream, patch_seconds=patch_seconds, stride_seconds=stride_seconds
        ).to(device)
        count = len(batch.patches)
        core = self.max_patches_per_call - 2 * self.context_patches
        pieces: list[torch.Tensor] = []
        for core_start in range(0, count, core):
            core_end = min(count, core_start + core)
            input_start = max(0, core_start - self.context_patches)
            input_end = min(count, core_end + self.context_patches)
            encoded = self._encode_chunk(batch, input_start, input_end)
            pieces.append(encoded[core_start - input_start : core_end - input_start])
        embeddings = torch.cat(pieces, dim=0)
        if len(embeddings) != count:
            raise RuntimeError("chunked HALO export lost or duplicated physical patches")
        return _motion_sequence(recording, batch, embeddings)
