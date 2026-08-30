"""Audited Task-0 stream and annotation policies for canonical application data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Mapping


@dataclass(frozen=True)
class Task0DatasetPolicy:
    streams: FrozenSet[str]
    annotation_kinds: FrozenSet[str]
    exhaustive_background: bool
    required_excluded_labels: Mapping[str, FrozenSet[str]] = field(default_factory=dict)


EVALUATION_POLICIES: Mapping[str, Task0DatasetPolicy] = {
    "c_mhad": Task0DatasetPolicy(
        streams=frozenset({"imu"}),
        annotation_kinds=frozenset({"event"}),
        exhaustive_background=False,
    ),
    "wear": Task0DatasetPolicy(
        streams=frozenset({"right_arm", "left_arm"}),
        annotation_kinds=frozenset({"activity"}),
        exhaustive_background=True,
    ),
    "oca": Task0DatasetPolicy(
        streams=frozenset({"imu0", "imu2"}),
        annotation_kinds=frozenset({"sample_label_run"}),
        exhaustive_background=True,
        required_excluded_labels={"sample_label_run": frozenset({"null"})},
    ),
}


CALIBRATION_POLICIES: Mapping[str, Task0DatasetPolicy] = {
    "openpack": Task0DatasetPolicy(
        streams=frozenset({"atr01"}),
        annotation_kinds=frozenset({"fine_action", "operation", "box_cycle"}),
        exhaustive_background=True,
        required_excluded_labels={
            "fine_action": frozenset({"ignore", "unknown", "system error"}),
            "operation": frozenset({"null", "system error"}),
        },
    ),
    "recofit": Task0DatasetPolicy(
        streams=frozenset({"right_forearm_imu"}),
        annotation_kinds=frozenset({"set"}),
        exhaustive_background=True,
    ),
}


def validate_policy(
    dataset: str,
    *,
    stream_id: str,
    annotation_kinds: set[str],
    excluded_labels: set[str],
    exhaustive_background: bool,
    calibration: bool,
    allow_exploratory: bool,
) -> None:
    policies = CALIBRATION_POLICIES if calibration else EVALUATION_POLICIES
    policy = policies.get(dataset)
    errors: list[str] = []
    if policy is None:
        errors.append("dataset has no audited Task-0 policy")
    else:
        if stream_id not in policy.streams:
            errors.append(f"stream {stream_id!r} is not an audited Task-0 stream")
        if (
            len(annotation_kinds) != 1
            or not annotation_kinds <= policy.annotation_kinds
        ):
            errors.append(
                "select exactly one audited annotation level from "
                f"{sorted(policy.annotation_kinds)}"
            )
        if exhaustive_background and not policy.exhaustive_background:
            errors.append("background is not exhaustively annotated for this protocol")
        selected_kind = (
            next(iter(annotation_kinds)) if len(annotation_kinds) == 1 else None
        )
        required_labels = policy.required_excluded_labels.get(
            selected_kind, frozenset()
        )
        missing_labels = required_labels - excluded_labels
        if missing_labels:
            errors.append(
                "this annotation level requires excluding background labels "
                f"{sorted(missing_labels)}"
            )
    if errors and not allow_exploratory:
        raise ValueError(
            f"invalid audited Task-0 policy for {dataset}: " + "; ".join(errors)
        )
