"""Phone/watch deployment channel policy for HALO data preprocessing.

Raw converted datasets intentionally remain lossless. This module defines the
deployment-scoped view consumed by HALO and EDA: one physical phone or watch
stream with acceleration and, when trustworthy and co-located, gyroscope data.
Every curated frame therefore has exactly three or six sensor channels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


STANDARD_CHANNEL_ORDER = (
    "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
)

PRIMARY_TRAIN_DATASETS = (
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

PRIMARY_EVAL_DATASETS = (
    "motionsense",
    "realworld",
    "mobiact",
    "shoaib",
    "inclusivehar",
)

EXCLUDED_PRIMARY_DATASETS = {
    "dsads": "torso/limb IMUs do not match the phone-pocket/waist or watch-wrist deployment",
    "harth": "lower-back and thigh accelerometers are retained only as a placement stress test",
    "opportunity": "back and upper/lower-arm IMUs are appendix-only, not phone/watch inputs",
    "recgym": "per-axis min-max normalization destroyed physical scale and gravity",
}


@dataclass(frozen=True)
class StreamSpec:
    dataset: str
    stream_id: str
    device_profile: str
    placement: str
    required: Mapping[str, Tuple[str, ...]]
    optional: Mapping[str, Tuple[str, ...]]
    gravity_state: str
    role: str = "primary"
    session_contains: Tuple[str, ...] = ()
    session_excludes: Tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class CuratedMetadata:
    dataset: str
    stream_id: str
    device_profile: str
    placement: str
    gravity_state: str
    channels: Tuple[str, ...]
    source_channels: Mapping[str, Tuple[str, ...]]
    note: str


def _xyz(prefix: str) -> Dict[str, Tuple[str, ...]]:
    return {f"acc_{axis}": (f"{prefix}{axis}",) for axis in "xyz"}


def _gyro(prefix: str) -> Dict[str, Tuple[str, ...]]:
    return {f"gyro_{axis}": (f"{prefix}{axis}",) for axis in "xyz"}


def _total_acc(acc_prefix: str, gravity_prefix: str) -> Dict[str, Tuple[str, ...]]:
    return {
        f"acc_{axis}": (f"{acc_prefix}{axis}", f"{gravity_prefix}{axis}")
        for axis in "xyz"
    }


_GENERIC_ACC = _xyz("acc_")
_GENERIC_GYRO = _gyro("gyro_")


STREAM_SPECS: Tuple[StreamSpec, ...] = (
    # Training datasets.
    StreamSpec("uci_har", "phone_waist", "phone", "waist",
               _xyz("total_acc_"), _gyro("body_gyro_"), "present"),
    StreamSpec("hhar", "phone_waist", "phone", "waist",
               _GENERIC_ACC, _GENERIC_GYRO, "present"),
    StreamSpec("pamap2", "watch_wrist", "watch", "wrist-mounted hand",
               _xyz("hand_acc16_"), _gyro("hand_gyro_"), "present",
               note="Uses the +/-16g wrist IMU; chest, ankle, 6g, mag, temperature, HR, and invalid orientation are pruned."),
    StreamSpec("wisdm", "phone_pocket", "phone", "pocket",
               _xyz("phone_accel_"), _gyro("phone_gyro_"), "present",
               session_contains=("phone_",), session_excludes=("_gyro_",),
               note="Legacy conversion stores acceleration and gyro separately; gyro is optional until the converter emits merged IMU sessions."),
    StreamSpec("wisdm", "watch_wrist", "watch", "wrist",
               _xyz("watch_accel_"), _gyro("watch_gyro_"), "present",
               session_contains=("watch_",), session_excludes=("_gyro_",),
               note="Legacy conversion stores acceleration and gyro separately; gyro is optional until the converter emits merged IMU sessions."),
    StreamSpec("kuhar", "phone_waist", "phone", "waist",
               _GENERIC_ACC, _GENERIC_GYRO, "removed"),
    StreamSpec("unimib_shar", "phone_pocket", "phone", "trouser pocket",
               _GENERIC_ACC, {}, "present"),
    StreamSpec("hapt", "phone_waist", "phone", "waist",
               _GENERIC_ACC, _GENERIC_GYRO, "present"),
    StreamSpec("mhealth", "watch_wrist", "watch", "right wrist",
               _xyz("arm_acc_"), {}, "present",
               note="Low-reliability sample-and-hold wrist gyro is pruned with chest, ankle, ECG, and magnetometer channels."),
    StreamSpec("capture24", "watch_wrist", "watch", "dominant wrist",
               _GENERIC_ACC, {}, "present"),

    # Primary evaluation datasets.
    StreamSpec("motionsense", "phone_front_pocket", "phone", "front pocket",
               _total_acc("acc_", "gravity_"), _GENERIC_GYRO, "present",
               note="Total acceleration is reconstructed from iOS userAcceleration + gravity; attitude is QA-only."),
    StreamSpec("realworld", "phone_waist", "phone", "waist",
               _GENERIC_ACC, _GENERIC_GYRO, "present",
               note="Gyro is retained only when the converted waist stream actually contains a complete finite triad."),
    StreamSpec("mobiact", "phone_trouser_pocket", "phone", "trouser pocket",
               _GENERIC_ACC, _GENERIC_GYRO, "present"),
    StreamSpec("shoaib", "phone_right_pocket", "phone", "right trouser pocket",
               _xyz("right_pocket_acc_"), _gyro("right_pocket_gyro_"), "present"),
    StreamSpec("inclusivehar", "phone_waist", "phone", "waist",
               _total_acc("acc_", "gravity_"), _GENERIC_GYRO, "present",
               note="Total acceleration is reconstructed from iOS userAcceleration + gravity; attitude is QA-only."),

    # Deployment-plausible diagnostic views, never mixed into the primary score.
    StreamSpec("shoaib", "phone_left_pocket", "phone", "left trouser pocket",
               _xyz("left_pocket_acc_"), _gyro("left_pocket_gyro_"), "present", role="diagnostic"),
    StreamSpec("shoaib", "phone_belt", "phone", "belt/holster",
               _xyz("belt_acc_"), _gyro("belt_gyro_"), "present", role="diagnostic"),
    StreamSpec("shoaib", "watch_wrist_proxy", "watch_proxy", "right wrist",
               _xyz("wrist_acc_"), _gyro("wrist_gyro_"), "present", role="diagnostic",
               note="A wrist-mounted smartphone is a placement proxy, not a true smartwatch."),
    StreamSpec("harth", "stress_lower_back", "non_deployment", "lower back",
               _xyz("back_acc_"), {}, "present", role="stress"),
    StreamSpec("harth", "stress_thigh", "non_deployment", "thigh",
               _xyz("thigh_acc_"), {}, "present", role="stress"),
)


_BY_DATASET: Dict[str, Tuple[StreamSpec, ...]] = {}
for _dataset in {spec.dataset for spec in STREAM_SPECS}:
    _BY_DATASET[_dataset] = tuple(spec for spec in STREAM_SPECS if spec.dataset == _dataset)


def policy_fingerprint() -> str:
    """Stable fingerprint used to invalidate cached HALO session indexes."""
    payload = [
        {
            "dataset": s.dataset,
            "stream_id": s.stream_id,
            "device_profile": s.device_profile,
            "placement": s.placement,
            "required": dict(s.required),
            "optional": dict(s.optional),
            "gravity_state": s.gravity_state,
            "role": s.role,
            "session_contains": s.session_contains,
            "session_excludes": s.session_excludes,
        }
        for s in STREAM_SPECS
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def stream_specs(dataset: str, role: Optional[str] = "primary") -> Tuple[StreamSpec, ...]:
    specs = _BY_DATASET.get(dataset, ())
    return specs if role is None else tuple(spec for spec in specs if spec.role == role)


def get_stream_spec(dataset: str, stream_id: str) -> StreamSpec:
    for spec in _BY_DATASET.get(dataset, ()):
        if spec.stream_id == stream_id:
            return spec
    raise KeyError(f"No deployment stream {dataset}/{stream_id}")


def session_stream_specs(
    dataset: str,
    session_id: str,
    role: str = "primary",
) -> Tuple[StreamSpec, ...]:
    """Return policy streams that can be represented by a converted session id."""
    matches = []
    for spec in stream_specs(dataset, role):
        if spec.session_contains and not all(token in session_id for token in spec.session_contains):
            continue
        if any(token in session_id for token in spec.session_excludes):
            continue
        matches.append(spec)
    return tuple(matches)


def _sources_available(frame: pd.DataFrame, sources: Sequence[str]) -> bool:
    if not all(source in frame.columns for source in sources):
        return False
    return all(np.isfinite(pd.to_numeric(frame[source], errors="coerce")).any() for source in sources)


def channel_names_for_frame(frame: pd.DataFrame, spec: StreamSpec) -> Tuple[str, ...]:
    """Return the standardized 3- or 6-channel schema available in ``frame``."""
    missing = [name for name, sources in spec.required.items() if not _sources_available(frame, sources)]
    if missing:
        raise ValueError(
            f"{spec.dataset}/{spec.stream_id}: missing required deployment channels {missing}; "
            f"available columns={list(frame.columns)}"
        )

    names = list(spec.required)
    optional_names = list(spec.optional)
    if optional_names and all(_sources_available(frame, spec.optional[name]) for name in optional_names):
        names.extend(optional_names)
    return tuple(name for name in STANDARD_CHANNEL_ORDER if name in names)


def curate_frame(frame: pd.DataFrame, spec: StreamSpec) -> Tuple[pd.DataFrame, CuratedMetadata]:
    """Select one deployment stream and rename/derive it to the common channel schema.

    A tuple of source columns means sum them elementwise. This is used only for
    iOS total acceleration reconstruction (userAcceleration + gravity).
    """
    channel_names = channel_names_for_frame(frame, spec)
    source_map = {**spec.required, **spec.optional}
    out = pd.DataFrame(index=frame.index)
    if "timestamp_sec" in frame.columns:
        out["timestamp_sec"] = pd.to_numeric(frame["timestamp_sec"], errors="coerce")

    for output_name in channel_names:
        sources = source_map[output_name]
        values = np.zeros(len(frame), dtype=np.float64)
        for source in sources:
            values += pd.to_numeric(frame[source], errors="coerce").to_numpy(dtype=np.float64)
        out[output_name] = values

    if "activity" in frame.columns:
        out["activity"] = frame["activity"].values

    metadata = CuratedMetadata(
        dataset=spec.dataset,
        stream_id=spec.stream_id,
        device_profile=spec.device_profile,
        placement=spec.placement,
        gravity_state=spec.gravity_state,
        channels=channel_names,
        source_channels={name: tuple(source_map[name]) for name in channel_names},
        note=spec.note,
    )
    return out.reset_index(drop=True), metadata


def curate_session(
    dataset: str,
    session_id: str,
    frame: pd.DataFrame,
    stream_id: Optional[str] = None,
    role: str = "primary",
) -> Tuple[pd.DataFrame, CuratedMetadata]:
    if stream_id is not None:
        spec = get_stream_spec(dataset, stream_id)
        if spec.role != role:
            raise ValueError(f"{dataset}/{stream_id} has role={spec.role}, requested role={role}")
    else:
        matches = session_stream_specs(dataset, session_id, role)
        if len(matches) != 1:
            raise ValueError(
                f"{dataset}/{session_id}: expected exactly one {role} deployment stream, "
                f"found {[s.stream_id for s in matches]}"
            )
        spec = matches[0]
    return curate_frame(frame, spec)


def source_channel_is_allowed(dataset: str, channel_name: str, role: str = "primary") -> bool:
    """Whether a raw channel participates in any selected deployment stream."""
    return any(
        channel_name in sources
        for spec in stream_specs(dataset, role)
        for sources in (*spec.required.values(), *spec.optional.values())
    )


def channel_description(metadata: CuratedMetadata, channel_name: str) -> str:
    modality = "accelerometer" if channel_name.startswith("acc_") else "gyroscope"
    axis = channel_name[-1].upper()
    gravity = ""
    if modality == "accelerometer":
        gravity = "; gravity removed" if metadata.gravity_state == "removed" else "; includes gravity"
    return (
        f"{metadata.device_profile} {modality} {axis}-axis at {metadata.placement}{gravity}"
    )


def all_source_channels(dataset: str, role: str = "primary") -> Tuple[str, ...]:
    channels = {
        source
        for spec in stream_specs(dataset, role)
        for sources in (*spec.required.values(), *spec.optional.values())
        for source in sources
    }
    return tuple(sorted(channels))
