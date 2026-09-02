"""Acquisition compatibility rules for cross-recording application examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import CANONICAL_CHANNELS, SensorStream


@dataclass(frozen=True, order=True)
class SensorCompatibilityKey:
    """Properties that must agree when signals are paired or joined.

    Exact device model and native sampling rate are deliberately absent. They may
    be treated as nuisance variation within one device family, whereas changing
    device family, placement, axes, or gravity convention changes the observation.
    """

    device_family: str
    placement: str
    channels: tuple[str, ...]
    gravity_state: str


def device_family(device: str) -> str:
    """Map known acquisition descriptions to a conservative physical form factor."""

    normalized = "_".join(device.strip().lower().replace("-", " ").split())
    if not normalized:
        raise ValueError("sensor device must be non-empty")
    if any(word in normalized for word in ("smartphone", "phone", "iphone")):
        return "phone"
    if any(
        word in normalized
        for word in ("smartwatch", "watch", "geneactiv", "bangle")
    ):
        return "watch"
    if any(
        word in normalized
        for word in ("imu", "shimmer", "bno055", "tsnd151", "razor")
    ):
        return "research_imu"
    return f"unclassified:{normalized}"


def sensor_compatibility_key(
    *,
    device: str,
    placement: str,
    channels: Sequence[str],
    gravity_state: str,
) -> SensorCompatibilityKey:
    normalized_placement = "_".join(
        placement.strip().lower().replace("-", " ").split()
    )
    if not normalized_placement:
        raise ValueError("sensor placement must be non-empty")
    channel_set = set(channels)
    if not channel_set or any(
        channel not in CANONICAL_CHANNELS for channel in channel_set
    ):
        raise ValueError(f"unsupported canonical channels: {tuple(channels)}")
    ordered_channels = tuple(
        channel for channel in CANONICAL_CHANNELS if channel in channel_set
    )
    if gravity_state not in {"present", "removed", "unknown"}:
        raise ValueError(f"invalid gravity state: {gravity_state}")
    return SensorCompatibilityKey(
        device_family=device_family(device),
        placement=normalized_placement,
        channels=ordered_channels,
        gravity_state=gravity_state,
    )


def stream_compatibility_key(stream: SensorStream) -> SensorCompatibilityKey:
    return sensor_compatibility_key(
        device=stream.device,
        placement=stream.placement,
        channels=stream.channels,
        gravity_state=stream.gravity_state,
    )


def require_compatible_streams(first: SensorStream, second: SensorStream) -> None:
    first_key = stream_compatibility_key(first)
    second_key = stream_compatibility_key(second)
    if first_key != second_key:
        raise ValueError(
            "paired sensor streams have incompatible acquisition configurations: "
            f"{first_key} != {second_key}"
        )
