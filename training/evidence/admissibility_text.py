"""Controlled text views for admissibility fitting and refinement.

Every view of one measured sensor/concept cell keeps the same physical target.  The variants are
deliberately conservative: sensor wording changes only terms already declared safe by the Phase-A
text augmentation, while concept wording comes only from training-dataset synonym/template tables.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import torch

from data.scripts.augmentations import _CH_SYNONYMS
from training.evidence.admissibility_gate import TableObservations
from training.evidence.labeltext import label_variant_rows


@dataclass(frozen=True)
class ObservationTextViews:
    """Text views with shape ``(physical cells, views)`` before embedding."""

    sensor: list[list[str]]
    concept: list[list[str]]

    @property
    def n_views(self) -> int:
        return len(self.sensor[0]) if self.sensor else 0

    def flattened(self) -> tuple[list[str], list[str]]:
        return (
            [value for row in self.sensor for value in row],
            [value for row in self.concept for value in row],
        )


def _stable_index(seed: int, text: str, view: int, pattern: str, n: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{text}\0{view}\0{pattern}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % n


def sensor_text_variants(text: str, count: int, seed: int = 0) -> list[str]:
    """Canonical sensor text plus deterministic, meaning-preserving surface variants.

    Placement, side, gravity convention, units, and partner-modality clauses are never matched by
    the substitution table and therefore cannot change.  A fixed function is used instead of the
    random Phase-A augmentation so artifact fitting is exactly reproducible.
    """
    if count < 1:
        raise ValueError("text-view count must be positive")
    variants = [text]
    for view in range(1, count):
        value = text
        for pattern, options in _CH_SYNONYMS:
            if not re.search(pattern, value, flags=re.I):
                continue
            alternatives = [option for option in options
                            if option.lower() != re.search(pattern, value, flags=re.I).group(0).lower()]
            if not alternatives:
                continue
            selected = alternatives[_stable_index(seed, text, view, pattern, len(alternatives))]
            value = re.sub(pattern, selected, value, flags=re.I)
        variants.append(value)
    return variants


def observation_text_views(
    observations: TableObservations,
    count: int = 4,
    seed: int = 0,
) -> ObservationTextViews:
    """Build a fixed number of equally weighted text views for every physical cell."""
    if count < 1:
        raise ValueError("text-view count must be positive")
    unique_sensors = list(dict.fromkeys(observations.sensor_text))
    sensor_views = {
        text: sensor_text_variants(text, count, seed=seed) for text in unique_sensors
    }

    unique_concepts = list(dict.fromkeys(observations.concept))
    label_rows = label_variant_rows(
        unique_concepts, count, seed=seed, use_descriptions=False, train_only=True,
    )
    concept_position = {label: index for index, label in enumerate(unique_concepts)}
    concept_views = {
        label: [row[concept_position[label]] for row in label_rows]
        for label in unique_concepts
    }
    return ObservationTextViews(
        sensor=[sensor_views[text] for text in observations.sensor_text],
        concept=[concept_views[label] for label in observations.concept],
    )


def embed_text_views(views: ObservationTextViews, embed) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed text views once, returning two ``(N,V,D)`` tensors."""
    sensor_text, concept_text = views.flattened()
    if not sensor_text:
        raise ValueError("cannot embed an empty observation table")
    sensor = embed(sensor_text).reshape(len(views.sensor), views.n_views, -1)
    concept = embed(concept_text).reshape(len(views.concept), views.n_views, -1)
    return sensor, concept


def flatten_training_views(
    sensor: torch.Tensor,
    concept: torch.Tensor,
    target: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten views while repeating every physical target exactly the same number of times."""
    if sensor.shape != concept.shape or sensor.ndim != 3:
        raise ValueError("sensor and concept views must have matching (N,V,D) shapes")
    values = torch.as_tensor(target, dtype=sensor.dtype, device=sensor.device)
    if len(values) != sensor.shape[0]:
        raise ValueError("target count does not match physical-cell count")
    return (
        sensor.reshape(-1, sensor.shape[-1]),
        concept.reshape(-1, concept.shape[-1]),
        values.repeat_interleave(sensor.shape[1]),
    )
