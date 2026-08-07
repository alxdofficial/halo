"""Regression tests for NFI-FARED duplicate annotation reconciliation."""

import numpy as np
import pytest

from data.datasets.nfi_fared.convert import merge_duplicate_labels


def test_duplicate_merge_recovers_activity_from_idle_annotation():
    merged, promoted = merge_duplicate_labels(
        [np.array(["walking", "no activity", "no activity"]),
         np.array(["walking", "train", "no activity"])],
        ["exp1.csv", "exp4.csv"],
    )
    assert merged.tolist() == ["walking", "train", "no activity"]
    assert promoted == 1


def test_duplicate_merge_allows_raw_labels_with_same_canonical():
    merged, promoted = merge_duplicate_labels(
        [np.array(["train"]), np.array(["bus"])],
        ["exp1.csv", "exp2.csv"],
    )
    assert merged.tolist() == ["train"]
    assert promoted == 0


def test_duplicate_merge_rejects_conflicting_real_activities():
    with pytest.raises(ValueError, match="conflicting mapped labels"):
        merge_duplicate_labels(
            [np.array(["walking"]), np.array(["running"])],
            ["exp1.csv", "exp2.csv"],
        )
