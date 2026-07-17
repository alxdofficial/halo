"""Serializable configuration for the contrastive tokenizer experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


DEFAULT_LABELS = (
    "sitting",
    "standing",
    "cycling",
    "walking",
    "walking_upstairs",
    "walking_downstairs",
    "jogging",
)

DEFAULT_DATASETS = (
    "uci_har",
    "hhar",
    "pamap2",
    "wisdm",
    "kuhar",
    "unimib_shar",
    "hapt",
    "mhealth",
    "capture24",
)


@dataclass
class ExperimentConfig:
    labels: tuple[str, ...] = DEFAULT_LABELS
    datasets: tuple[str, ...] = DEFAULT_DATASETS
    alignment: str = "harmonised"
    seed: int = 20260714
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    max_windows_per_domain_label: int = 1500

    patch_seconds: float = 1.5
    dft_size: int = 256
    token_dim: int = 128
    representation_dim: int = 128
    projection_dim: int = 64
    dropout: float = 0.1

    loss_mode: str = "supervised"
    temperature: float = 0.1
    rotation_probability: float = 1.0
    jitter_probability: float = 0.0
    time_shift_probability: float = 0.0

    classes_per_batch: int = 6
    samples_per_class: int = 4
    steps_per_epoch: int = 200
    validation_steps: int = 30
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    num_workers: int = 0
    epoch_mode: str = "fixed_steps"

    legacy_root: str = str(Path(__file__).resolve().parents[3] / "legacy_code")

    def validate(self) -> None:
        if self.alignment != "harmonised":
            raise ValueError("This first experiment requires fixed-shape harmonised grids")
        if not 0 <= self.rotation_probability <= 1:
            raise ValueError("rotation_probability must be in [0, 1]")
        if not 0 <= self.jitter_probability <= 1:
            raise ValueError("jitter_probability must be in [0, 1]")
        if not 0 <= self.time_shift_probability <= 1:
            raise ValueError("time_shift_probability must be in [0, 1]")
        if self.loss_mode not in {"supervised", "instance"}:
            raise ValueError("loss_mode must be 'supervised' or 'instance'")
        if self.epoch_mode not in {"fixed_steps", "full_coverage"}:
            raise ValueError("epoch_mode must be 'fixed_steps' or 'full_coverage'")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be below 1")
        if self.classes_per_batch < 2 or self.samples_per_class < 1:
            raise ValueError("batches require at least two classes and one sample per class")
        if self.patch_seconds <= 0 or self.dft_size < 2:
            raise ValueError("patch_seconds and dft_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        known = {field.name for field in fields(cls)}
        kwargs = {key: value for key, value in values.items() if key in known}
        for name in ("labels", "datasets"):
            if name in kwargs:
                kwargs[name] = tuple(kwargs[name])
        config = cls(**kwargs)
        config.validate()
        return config

    @classmethod
    def smoke(cls, legacy_root: str | None = None) -> "ExperimentConfig":
        config = cls(
            labels=("sitting", "walking"),
            datasets=("uci_har", "hhar"),
            max_windows_per_domain_label=24,
            token_dim=32,
            representation_dim=32,
            projection_dim=16,
            classes_per_batch=2,
            samples_per_class=2,
            steps_per_epoch=2,
            validation_steps=1,
            epochs=1,
        )
        if legacy_root is not None:
            config.legacy_root = legacy_root
        return config

    @classmethod
    def full_train_corpus(cls, legacy_root: str | None = None) -> "ExperimentConfig":
        """Use every primary-training window and every label represented in train."""
        config = cls(
            labels=(),
            max_windows_per_domain_label=0,
            classes_per_batch=8,
            samples_per_class=4,
            steps_per_epoch=0,
            validation_steps=50,
            epochs=10,
            epoch_mode="full_coverage",
        )
        if legacy_root is not None:
            config.legacy_root = legacy_root
        return config
