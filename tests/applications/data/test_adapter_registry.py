from __future__ import annotations

import pytest

from applications.motion_monitoring.data.adapters.registry import (
    ADAPTERS,
    adapter_names,
    iter_recordings,
    materializable_adapter_names,
)


def test_adapter_registry_has_frozen_new_source_set() -> None:
    assert adapter_names() == (
        "alameda",
        "aidlab_har",
        "c_mhad",
        "crossfit",
        "cops",
        "monipar",
        "oca",
        "openpack",
        "recofit",
        "wear",
        "synth_wrist_v1",
    )
    assert {spec.default_role for spec in ADAPTERS.values()} == {
        "training_development",
        "evaluation",
        "task1_evaluation",
        "task1_training",
        "task2_evaluation",
    }
    assert materializable_adapter_names() == (
        "aidlab_har",
        "c_mhad",
        "crossfit",
        "oca",
        "openpack",
        "recofit",
        "wear",
        "synth_wrist_v1",
    )
    assert ADAPTERS["synth_wrist_v1"].cache_policy == "derived"
    assert ADAPTERS["synth_wrist_v1"].derived_from == ("crossfit", "recofit")
    assert {
        name for name, spec in ADAPTERS.items() if spec.cache_policy == "stream"
    } == {"alameda", "cops", "monipar"}


def test_adapter_registry_rejects_unknown_source() -> None:
    with pytest.raises(KeyError, match="unknown application dataset"):
        next(iter_recordings("not_a_dataset", limit=1))
