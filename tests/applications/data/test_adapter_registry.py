from __future__ import annotations

import pytest

from applications.motion_monitoring.data.adapters.registry import (
    ADAPTERS,
    adapter_names,
    iter_recordings,
)


def test_adapter_registry_has_frozen_new_source_set() -> None:
    assert adapter_names() == (
        "aidlab_har",
        "c_mhad",
        "crossfit",
        "oca",
        "openpack",
        "recofit",
        "wear",
    )
    assert {spec.default_role for spec in ADAPTERS.values()} == {
        "training_development",
        "evaluation",
    }


def test_adapter_registry_rejects_unknown_source() -> None:
    with pytest.raises(KeyError, match="unknown application dataset"):
        next(iter_recordings("not_a_dataset", limit=1))
