"""KneE-PAD knee-rehabilitation trials as bounded Task-2 executions.

Thirty-one patients with a diagnosed knee pathology performed three prescribed
exercises unsupervised in physiotherapy centres while wearing eight Delsys Trigno
Avanti units over thigh and calf muscle bellies. The publishers then curated what
patients actually did wrong and identified two recurring incorrect variants per
exercise. That provenance is why this is Task 2's controlled known-difference
cell: the deviations are real patient errors sorted after the fact, not faults
performed to order (docs/tasks/TASK2_CHANGE_QUANTIFICATION.md section 7.2).

Two contract choices matter and are deliberate:

* the event **label is the base exercise**, not the released variant name. Task 2
  compares executions of one task, so correct squats must be able to serve as the
  reference set for an incorrect squat. Encoding the variant in the label would
  make them different tasks that are never compared. The variant is carried in
  metadata as ``execution_variant`` with an ``accepted`` flag.
* each of the eight sensors becomes its **own stream**, so each carries its own
  placement and therefore its own compatibility key. A batch can then never mix
  a rectus femoris trace with a gastrocnemius one.

Every subject was recorded in a single visit, so all repeats here are
within-session. That is honest for a known-difference cell and is why this source
cannot contribute a between-day reliability estimate.

Muscle-belly placement sits outside the phone and watch deployment envelope the
repository declares, so results from this source are reported as a cross-placement
stress cell rather than as consumer-wearable performance.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)


_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1] / "sources" / "kneepad" / "sessions"
)
_RATE_HZ = 2000.0 / 13.5  # 148.148..., the released Trigno IMU rate
_AXES = ("x", "y", "z")
_SENSORS = (
    "right_rectus_femoris",
    "right_hamstrings",
    "right_tibialis_anterior",
    "right_gastrocnemius",
    "left_rectus_femoris",
    "left_hamstrings",
    "left_tibialis_anterior",
    "left_gastrocnemius",
)
_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
# Guard: the converter emits acceleration in g, so a resting limb reads near 1.
_ACC_G_RANGE = (0.5, 1.5)
_MIN_SAMPLES = 32

# Released label -> (base exercise, accepted, variant name).
_LABELS = {
    "squat": ("squat", True, "correct"),
    "squat_with_the_injured_leg_placed_in_front": (
        "squat", False, "injured_leg_in_front",
    ),
    "squat_with_weight_transfer_onto_the_healthy_leg": (
        "squat", False, "weight_transfer_to_healthy_leg",
    ),
    "seated_leg_extension": ("seated_leg_extension", True, "correct"),
    "seated_leg_extension_with_the_limb_lifted_from_the_chair": (
        "seated_leg_extension", False, "limb_lifted_from_chair",
    ),
    "seated_leg_extension_without_full_range_of_motion": (
        "seated_leg_extension", False, "reduced_range_of_motion",
    ),
    "walking": ("walking", True, "correct"),
    "walking_with_the_injured_knee_fully_extended_and_the_hip_abducted": (
        "walking", False, "knee_extended_hip_abducted",
    ),
    "walking_with_the_injured_knee_not_fully_extended": (
        "walking", False, "knee_not_fully_extended",
    ),
}


def iter_recordings(
    root: Path | None = None,
    limit: int | None = None,
) -> Iterator[RawRecording]:
    """Yield one bounded execution per released trial, on eight sensor streams."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    source = _DEFAULT_ROOT if root is None else Path(root)
    if not source.is_dir():
        raise FileNotFoundError(f"KneE-PAD sessions not found at {source}")
    columns = [f"{sensor}_{channel}" for sensor in _SENSORS for channel in _CHANNELS]
    yielded = 0
    for path in sorted(source.glob("*/data.parquet")):
        if limit is not None and yielded >= limit:
            return
        trial = path.parent.name
        frame = pd.read_parquet(path, columns=["timestamp_sec", "activity", "subject", *columns])
        if len(frame) < _MIN_SAMPLES:
            continue
        released = frame["activity"].astype(str).unique().tolist()
        if len(released) != 1:
            raise ValueError(f"{path} mixes released labels: {released}")
        try:
            label, accepted, variant = _LABELS[released[0]]
        except KeyError as error:
            raise ValueError(f"{path}: unknown released label {released[0]!r}") from error
        subjects = frame["subject"].astype(str).unique().tolist()
        if len(subjects) != 1:
            raise ValueError(f"{path} mixes subjects: {subjects}")
        subject_id = subjects[0]
        timestamps = frame["timestamp_sec"].to_numpy(dtype=np.float64)
        if not np.all(np.diff(timestamps) > 0):
            raise ValueError(f"{path}: clock is not strictly increasing")

        streams = []
        for sensor in _SENSORS:
            values = frame[[f"{sensor}_{channel}" for channel in _CHANNELS]].to_numpy(
                dtype=np.float32
            )
            valid = np.isfinite(values)
            magnitude = float(np.median(np.linalg.norm(np.nan_to_num(values[:, :3]), axis=1)))
            if not _ACC_G_RANGE[0] <= magnitude <= _ACC_G_RANGE[1]:
                raise ValueError(
                    f"{path}: {sensor} median |acc| {magnitude:.3f} is outside the "
                    f"documented g range {_ACC_G_RANGE}"
                )
            streams.append(
                SensorStream(
                    stream_id=sensor,
                    placement=sensor,
                    device="Delsys Trigno Avanti IMU",
                    timestamps_sec=timestamps,
                    values=np.nan_to_num(values),
                    channels=_CHANNELS,
                    valid=valid,
                    gravity_state="present",
                    nominal_rate_hz=_RATE_HZ,
                    metadata={
                        "acceleration_unit": "g",
                        "gyroscope_unit": "rad/s",
                        "placement_class": "muscle_belly",
                        "outside_consumer_deployment_envelope": True,
                    },
                )
            )
        period = 1.0 / _RATE_HZ
        yield RawRecording(
            dataset="kneepad",
            recording_id=f"kneepad:{trial}",
            subject_id=subject_id,
            # One visit per patient: every repeat here is within-session.
            session_id=f"kneepad:{subject_id}:visit",
            streams=tuple(streams),
            events=(
                EventInterval(
                    start_sec=float(timestamps[0]),
                    end_sec=float(timestamps[-1] + period),
                    label=label,
                    annotation_kind="bounded_execution",
                    metadata={
                        "released_label": released[0],
                        "execution_variant": variant,
                        "accepted": accepted,
                        "source": "released_trial_extent",
                    },
                ),
            ),
            split="evaluation",
            metadata={
                "trial": trial,
                "released_label": released[0],
                "execution_variant": variant,
                "accepted": accepted,
                "bounded_execution_annotations": True,
                "single_visit_cohort": True,
            },
        )
        yielded += 1
