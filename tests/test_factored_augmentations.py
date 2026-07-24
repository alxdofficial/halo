"""Factored text metadata must stay synchronized with physics/channel augmentations."""

import random

import numpy as np
import torch

from data.scripts.augmentations import AugmentationConfig, IMUAugmenter, IMUSample
from training.tokenizer.pretrain_data import CHANNELS, stream_sensor_texts


def _sample() -> IMUSample:
    roles, sensors, sensor_id = stream_sensor_texts("hhar", "phone_waist")
    data = torch.zeros(360, 6)
    data[:, 0] = 1.0
    data[:, 1] = 0.05 * torch.sin(torch.linspace(0, 20, 360))
    return IMUSample(
        data=data,
        channel_names=list(CHANNELS),
        sampling_rate=60.0,
        channel_descriptions=[
            f"{name}; includes gravity" if name.startswith("acc") else name
            for name in CHANNELS
        ],
        label="walking",
        dataset_name="hhar",
        channel_mask=[True] * 6,
        role_descriptions=roles,
        sensor_descriptions=sensors,
        sensor_id=sensor_id,
        gravity_state="present",
    )


def test_gravity_removal_updates_factored_sensor_text_and_state():
    cfg = AugmentationConfig.none()
    cfg.gravity.enabled = True
    cfg.gravity.p = 1.0
    out = IMUAugmenter(cfg)(_sample())
    assert out.gravity_state == "removed"
    assert "gravity removed" in out.sensor_descriptions[0].lower()
    assert "includes gravity" not in out.sensor_descriptions[0].lower()
    assert out.data[:, :3].mean(0).norm() < 0.1


def test_gyro_dropout_drops_the_gyro_sensor():
    cfg = AugmentationConfig.none()
    cfg.channel_dropout.enabled = True
    cfg.channel_dropout.p = 1.0
    out = IMUAugmenter(cfg)(_sample())
    assert out.channel_names == list(CHANNELS[:3])          # only the accel triad survives
    assert out.sensor_id == [0, 0, 0]
    # accel & gyro are separate modality-level sensors: dropping the gyro group REMOVES the gyro
    # sensor entirely (no phantom "accelerometer only" phrase on a shared description).
    assert len(out.sensor_descriptions) == 1
    assert "accelerometer" in out.sensor_descriptions[0].lower()
    assert "gyroscope" not in out.sensor_descriptions[0].lower()


def test_text_dropout_respects_channel_budget_without_erasing_sensor_identity():
    cfg = AugmentationConfig.none()
    cfg.channel_text_dropout.enabled = True
    cfg.channel_text_dropout.p = 1.0
    cfg.channel_text_dropout.max_frac = 0.5
    random.seed(12)
    np.random.seed(12)
    sample = _sample()
    original_sensor = list(sample.sensor_descriptions)
    out = IMUAugmenter(cfg)(sample)
    dropped = [i for i, text in enumerate(out.role_descriptions)
               if text == cfg.channel_text_dropout.neutral]
    assert 1 <= len(dropped) <= 3
    assert out.sensor_descriptions == original_sensor
    assert all(out.channel_descriptions[i] == cfg.channel_text_dropout.neutral for i in dropped)


def test_padding_only_accelerometer_is_not_treated_as_physical_gravity():
    cfg = AugmentationConfig.none()
    cfg.gravity.enabled = True
    cfg.gravity.p = 1.0
    roles, sensors, sensor_id = stream_sensor_texts(
        "synthetic", "watch_wrist", has_accel=False, has_gyro=True
    )
    sample = IMUSample(
        data=torch.zeros(120, 6),
        channel_names=list(CHANNELS),
        sampling_rate=50.0,
        channel_descriptions=["accelerometer includes gravity"] * 3 + ["gyroscope"] * 3,
        channel_mask=[False, False, False, True, True, True],
        role_descriptions=roles,
        sensor_descriptions=sensors,
        sensor_id=sensor_id,
        gravity_state=None,
    )
    out = IMUAugmenter(cfg)(sample)
    assert out.gravity_state is None
    # has_accel=False -> only a gyroscope sensor is advertised; no phantom accelerometer sensor.
    assert not any("accelerometer" in s.lower() for s in out.sensor_descriptions)
    assert any("gyroscope" in s.lower() for s in out.sensor_descriptions)
