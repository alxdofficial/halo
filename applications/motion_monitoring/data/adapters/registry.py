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


ADAPTERS: dict[str, AdapterSpec] = {
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
}


def adapter_names() -> tuple[str, ...]:
    return tuple(ADAPTERS)


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
