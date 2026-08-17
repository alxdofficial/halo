from __future__ import annotations

import numpy as np
import torch

from training.evidence.admissibility_gate import TableObservations
from training.evidence.admissibility_text import (
    flatten_training_views, observation_text_views, sensor_text_variants,
)


def _observations() -> TableObservations:
    return TableObservations(
        sensor_text=[
            "a watch accelerometer on the left wrist with gravity present, recorded alongside a gyroscope",
            "a phone gyroscope in the right pocket, recorded alongside an accelerometer",
        ],
        concept_text=["walking", "sitting"],
        value=np.asarray([0.8, 0.2], dtype=np.float32),
        stream_key=["a/wrist::accel", "b/pocket::gyro"],
        concept=["walking", "sitting"],
    )


def test_sensor_paraphrases_preserve_load_bearing_configuration_facts():
    text = _observations().sensor_text[0]
    variants = sensor_text_variants(text, 6, seed=7)
    assert variants[0] == text
    assert len(variants) == 6
    for variant in variants:
        for fact in ("left wrist", "gravity present", "alongside"):
            assert fact in variant
        assert any(term in variant for term in ("gyroscope", "gyro", "angular rate sensor"))


def test_text_views_are_reproducible_and_cell_balanced():
    observations = _observations()
    first = observation_text_views(observations, count=4, seed=11)
    second = observation_text_views(observations, count=4, seed=11)
    assert first == second
    assert all(len(row) == 4 for row in first.sensor + first.concept)

    sensor = torch.randn(2, 4, 8)
    concept = torch.randn(2, 4, 8)
    _, _, target = flatten_training_views(sensor, concept, observations.value)
    assert torch.allclose(target, torch.tensor([0.8] * 4 + [0.2] * 4))
