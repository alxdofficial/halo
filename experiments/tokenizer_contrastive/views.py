"""Paired sensor-only augmentations for contrastive training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from data.scripts.augmentations import AugmentationConfig, IMUAugmenter, IMUSample


@dataclass(frozen=True)
class ViewConfig:
    rotation_probability: float
    jitter_probability: float
    time_shift_probability: float


def channel_descriptions(channels: tuple[str, ...], gravity_state: str) -> list[str]:
    gravity = "includes gravity" if gravity_state == "present" else "gravity removed"
    descriptions = []
    for channel in channels:
        family, axis = channel.split("_", maxsplit=1)
        if family == "acc":
            descriptions.append(f"Accelerometer {axis}-axis ({gravity})")
        elif family == "gyro":
            descriptions.append(f"Gyroscope {axis}-axis")
        else:
            descriptions.append(channel)
    return descriptions


class PairedViewMaker:
    """Create two independent views while preserving the activity label."""

    def __init__(self, config: ViewConfig):
        aug = AugmentationConfig.none()
        aug.rotation_3d.enabled = config.rotation_probability > 0
        aug.rotation_3d.p = config.rotation_probability
        # Linear acceleration and angular velocity are valid 3D vectors even when gravity
        # is absent, so the contrastive rotation study deliberately covers both gravity states.
        aug.rotation_3d.require_gravity = False
        aug.jitter.enabled = config.jitter_probability > 0
        aug.jitter.p = config.jitter_probability
        aug.time_shift.enabled = config.time_shift_probability > 0
        aug.time_shift.p = config.time_shift_probability
        self.augmenter = IMUAugmenter(aug)

    def _one(
        self,
        data: torch.Tensor,
        channels: tuple[str, ...],
        descriptions: list[str],
        rate_hz: float,
        label: str,
        dataset: str,
    ) -> torch.Tensor:
        sample = IMUSample(
            data=data.clone(),
            channel_names=list(channels),
            sampling_rate=rate_hz,
            channel_descriptions=list(descriptions),
            label=label,
            dataset_name=dataset,
        )
        result = self.augmenter(sample).data
        if result.shape != data.shape:
            raise ValueError(f"Value-only contrastive views must preserve shape: {result.shape} != {data.shape}")
        return result

    def __call__(
        self,
        data: torch.Tensor,
        channels: tuple[str, ...],
        gravity_state: str,
        rate_hz: float,
        label: str,
        dataset: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        descriptions = channel_descriptions(channels, gravity_state)
        return (
            self._one(data, channels, descriptions, rate_hz, label, dataset),
            self._one(data, channels, descriptions, rate_hz, label, dataset),
        )
