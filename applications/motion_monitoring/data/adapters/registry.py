"""Registry for application-dataset adapters.

Task code should use this module rather than importing a source-specific reader.
The imports remain lazy so optional format dependencies are needed only for the
dataset being read.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from applications.motion_monitoring.data.contracts import RawRecording


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    module: str
    default_role: str
    cache_policy: str = "materialize"
    # Canonical caches this adapter synthesizes from (``derived`` policy only).
    # A derived dataset has no frozen source payload; its provenance is the
    # provenance of the caches it was generated from plus the generator code.
    derived_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cache_policy not in {"materialize", "stream", "derived"}:
            raise ValueError(f"unsupported cache policy: {self.cache_policy}")
        if (self.cache_policy == "derived") != bool(self.derived_from):
            raise ValueError(
                f"{self.name}: derived_from must be set exactly for the derived policy"
            )


ADAPTERS: dict[str, AdapterSpec] = {
    "alameda": AdapterSpec(
        "alameda",
        "applications.motion_monitoring.data.adapters.alameda",
        "task2_evaluation",
        "stream",
    ),
    "aidlab_har": AdapterSpec(
        "aidlab_har",
        "applications.motion_monitoring.data.adapters.aidlab_har",
        "training_development",
    ),
    "c_mhad": AdapterSpec(
        "c_mhad",
        "applications.motion_monitoring.data.adapters.c_mhad",
        "evaluation",
    ),
    "crossfit": AdapterSpec(
        "crossfit",
        "applications.motion_monitoring.data.adapters.crossfit",
        "training_development",
    ),
    "cops": AdapterSpec(
        "cops",
        "applications.motion_monitoring.data.adapters.cops",
        "task2_evaluation",
        "stream",
    ),
    "monipar": AdapterSpec(
        "monipar",
        "applications.motion_monitoring.data.adapters.monipar",
        "task1_evaluation",
        "stream",
    ),
    "oca": AdapterSpec(
        "oca",
        "applications.motion_monitoring.data.adapters.oca",
        "evaluation",
    ),
    "openpack": AdapterSpec(
        "openpack",
        "applications.motion_monitoring.data.adapters.openpack",
        "training_development",
    ),
    "recofit": AdapterSpec(
        "recofit",
        "applications.motion_monitoring.data.adapters.recofit",
        "training_development",
    ),
    "wear": AdapterSpec(
        "wear",
        "applications.motion_monitoring.data.adapters.wear",
        "evaluation",
    ),
    # Task-1 synthetic wrist-IMU training corpus: CrossFit single-repetition
    # clips inserted into RecoFit backgrounds (TASK1_REFERENCE_RESOLUTION_SPEC
    # section C). Training only; never a development or test source.
    "synth_wrist_v1": AdapterSpec(
        "synth_wrist_v1",
        "applications.motion_monitoring.data.adapters.synth_wrist_v1",
        "task1_training",
        "derived",
        derived_from=("crossfit", "recofit"),
    ),
}


def adapter_names() -> tuple[str, ...]:
    return tuple(ADAPTERS)


def materializable_adapter_names() -> tuple[str, ...]:
    """Adapters whose source payloads should be copied into canonical caches."""

    return tuple(
        name
        for name, spec in ADAPTERS.items()
        if spec.cache_policy in {"materialize", "derived"}
    )


def iter_recordings(
    dataset: str,
    *,
    root: Path | None = None,
    limit: int | None = None,
) -> Iterator[RawRecording]:
    """Yield recordings from one registered source using its native adapter."""

    try:
        spec = ADAPTERS[dataset]
    except KeyError as error:
        choices = ", ".join(ADAPTERS)
        raise KeyError(
            f"unknown application dataset {dataset!r}; choose from {choices}"
        ) from error
    module = import_module(spec.module)
    yield from module.iter_recordings(root=root, limit=limit)
