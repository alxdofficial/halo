"""Assembled long wrist recordings for Task-3 training (doc section 10.3).

CrossFit ships one repetition per array, about 3 s. A clip that short cannot serve
Task 3's training shape at all: it hosts only the shortest of the five candidate
scales, contains a single execution so no cross-instance positive pair can form
inside it, and has no background, while deployment searches a timeline that is
mostly non-target.

The decisive reason is boundary realism. Deployment supplies no event boundaries;
the candidate grid has to find them. Training on pre-trimmed clips hands the
boundary over for free, so the matcher never learns to tolerate a candidate that
starts early, ends late, or straddles two executions, and the boundary IoU and
start/end error metrics would be measured on a condition training never faced.
Here the inserted extents are known exactly while the grid must still locate them.

Construction, per recording: a background drawn from CrossFit sets of a *different*
exercise (not only the `Null` recordings, which cover just 7 subjects, whereas
different-label sets reach all 50 at matched wrist placement and double as
labelled distractors), into which several repetitions of one primary exercise are
spliced, plus repetitions of other exercises as explicit negatives.

Section 5's prohibition is respected: the whole-recording transform is applied to
background and inserts together, never to the inserted events alone, so "was
spliced" carries no information about the label. The mandatory check is the
shuffled-identity leak control in ``task3/controls.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from applications.motion_monitoring.data.cache import CachedRecordingDataset
from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)


DATASET = "synth_long_v1"
_DATA_ROOT = Path(__file__).resolve().parents[1]

SYNTHESIS_CONFIG: dict[str, Any] = {
    "version": 1,
    "seed": 20260903,
    "source_dataset": "crossfit",
    "donor_kind": "repetition",
    "background_kind": "exercise_sequence",
    "recordings": 600,
    "recording_seconds": 240.0,
    "primary_inserts": [4, 5, 6, 7],
    "distractor_inserts": [2, 3, 4],
    "distractor_exercises": 3,
    "edge_margin_sec": 4.0,
    "insert_spacing_sec": 2.0,
    "crossfade_sec": 0.3,
    # Applied to background and inserts together, after assembly.
    "recording_rotation_max_deg": 15.0,
    "noise_sd_acc_g": 0.01,
    "noise_sd_gyro_rad_s": 0.02,
    "donor_duration_bounds_sec": [1.0, 8.0],
    "min_background_seconds": 60.0,
}
_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")


def config_digest(config: dict[str, Any] = SYNTHESIS_CONFIG) -> str:
    return sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_root(dataset: str, root: Path | None) -> Path:
    base = Path(root) if root is not None else _DATA_ROOT / "sources"
    return base / dataset / "processed" / "canonical_v1"


def _rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    angle = float(rng.uniform(-np.deg2rad(max_deg), np.deg2rad(max_deg)))
    axis = rng.normal(size=3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def _apply_rotation(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    out = values.copy()
    out[:, :3] = values[:, :3] @ rotation.T
    out[:, 3:6] = values[:, 3:6] @ rotation.T
    return out


def _load_bank(cache: CachedRecordingDataset, config: dict[str, Any]):
    """Donor repetitions grouped by exercise, and background sets by exercise."""

    low, high = config["donor_duration_bounds_sec"]
    donors: dict[str, list[dict[str, Any]]] = {}
    backgrounds: dict[str, list[dict[str, Any]]] = {}
    for recording in cache:
        stream = recording.streams[0]
        if tuple(stream.channels) != _CHANNELS:
            continue
        values = np.asarray(stream.values, dtype=np.float64)
        rate = float(stream.nominal_rate_hz or 100.0)
        for event in recording.events:
            duration = event.end_sec - event.start_sec
            if event.annotation_kind == config["donor_kind"]:
                if not low <= duration <= high or not np.asarray(stream.valid).all():
                    continue
                donors.setdefault(event.label, []).append(
                    {
                        "values": values,
                        "rate": rate,
                        "clip_id": recording.recording_id,
                        "subject": recording.subject_id,
                        "label": event.label,
                    }
                )
            elif event.annotation_kind == config["background_kind"]:
                if duration < config["min_background_seconds"]:
                    continue
                backgrounds.setdefault(event.label, []).append(
                    {
                        "values": values,
                        "rate": rate,
                        "recording_id": recording.recording_id,
                        "subject": recording.subject_id,
                        "label": event.label,
                    }
                )
    return donors, backgrounds


def iter_recordings(
    root: Path | None = None,
    limit: int | None = None,
    config: dict[str, Any] = SYNTHESIS_CONFIG,
) -> Iterator[RawRecording]:
    """Yield assembled wrist recordings with exact inserted-execution extents."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    cache = CachedRecordingDataset(_canonical_root(config["source_dataset"], root))
    donors, backgrounds = _load_bank(cache, config)
    exercises = sorted(label for label, items in donors.items() if len(items) >= 8)
    if len(exercises) < 2 or not backgrounds:
        raise ValueError("the CrossFit cache cannot supply donors and backgrounds")

    total = config["recordings"] if limit is None else min(config["recordings"], limit)
    for index in range(total):
        rng = np.random.default_rng(
            int(sha256(f"{config['seed']}:{index}".encode()).hexdigest()[:8], 16)
        )
        primary = exercises[int(rng.integers(len(exercises)))]
        # Background is a different exercise, so it is a labelled distractor
        # rather than an unknown region.
        background_labels = sorted(set(backgrounds) - {primary})
        if not background_labels:
            continue
        background_label = background_labels[int(rng.integers(len(background_labels)))]
        pool = backgrounds[background_label]
        base = pool[int(rng.integers(len(pool)))]
        rate = base["rate"]
        length = int(round(config["recording_seconds"] * rate))
        source = base["values"]
        if len(source) < length:
            repeats = int(np.ceil(length / len(source)))
            source = np.tile(source, (repeats, 1))
        signal = source[:length].copy()

        margin = config["edge_margin_sec"]
        spacing = config["insert_spacing_sec"]
        fade = int(round(config["crossfade_sec"] * rate))
        occupied: list[tuple[float, float]] = []
        events: list[EventInterval] = []
        counts: dict[str, int] = {}

        plan = [(primary, int(rng.choice(config["primary_inserts"])))]
        # The background is a real set of ``background_label``, so that motion is
        # already present unlabelled throughout the recording. Inserting it again
        # as a distractor would create labelled and unlabelled copies of the same
        # motion and make the unassigned regions quietly wrong.
        others = [item for item in exercises if item not in {primary, background_label}]
        rng.shuffle(others)
        for label in others[: config["distractor_exercises"]]:
            plan.append((label, int(rng.choice(config["distractor_inserts"]))))

        for label, count in plan:
            for _ in range(count):
                clips = donors[label]
                clip = clips[int(rng.integers(len(clips)))]
                donor = np.asarray(clip["values"], dtype=np.float64)
                span = len(donor) / rate
                placed = False
                for _attempt in range(40):
                    start = float(
                        rng.uniform(margin, config["recording_seconds"] - margin - span)
                    )
                    stop = start + span
                    if all(
                        stop + spacing <= left or start >= right + spacing
                        for left, right in occupied
                    ):
                        placed = True
                        break
                if not placed:
                    continue
                begin = int(round(start * rate))
                end = begin + len(donor)
                if end > length:
                    continue
                # Match the donor's mean gravity to the background at the seam so
                # the splice is not a step discontinuity.
                donor_gravity = donor[:, :3].mean(axis=0)
                local_gravity = signal[begin:end, :3].mean(axis=0)
                left = donor_gravity / max(float(np.linalg.norm(donor_gravity)), 1e-9)
                right = local_gravity / max(float(np.linalg.norm(local_gravity)), 1e-9)
                axis = np.cross(left, right)
                norm = float(np.linalg.norm(axis))
                if norm > 1e-8:
                    angle = float(np.arctan2(norm, float(np.dot(left, right))))
                    axis = axis / norm
                    cross = np.array(
                        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
                    )
                    align = (
                        np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
                    )
                    donor = _apply_rotation(donor, align)
                blend = donor.copy()
                if fade > 0 and len(blend) > 2 * fade:
                    ramp = np.linspace(0.0, 1.0, fade)[:, None]
                    blend[:fade] = ramp * blend[:fade] + (1 - ramp) * signal[begin : begin + fade]
                    blend[-fade:] = (1 - ramp) * blend[-fade:] + ramp * signal[end - fade : end]
                signal[begin:end] = blend
                occupied.append((start, stop))
                occurrence = counts.get(label, 0)
                counts[label] = occurrence + 1
                events.append(
                    EventInterval(
                        start_sec=start,
                        end_sec=stop,
                        label=label,
                        annotation_kind="inserted_execution",
                        metadata={
                            "donor_clip_id": clip["clip_id"],
                            "donor_subject_id": clip["subject"],
                            "occurrence_index": occurrence,
                            "role": "primary" if label == primary else "distractor",
                            "guard_sec": config["crossfade_sec"],
                        },
                    )
                )
        if counts.get(primary, 0) < 2:
            # Without two executions of the primary exercise the recording cannot
            # contribute a positive pair, which is the whole point.
            continue

        # One transform for the whole recording, background and inserts alike.
        signal = _apply_rotation(signal, _rotation(rng, config["recording_rotation_max_deg"]))
        signal[:, :3] += rng.normal(0.0, config["noise_sd_acc_g"], signal[:, :3].shape)
        signal[:, 3:6] += rng.normal(0.0, config["noise_sd_gyro_rad_s"], signal[:, 3:6].shape)
        events.sort(key=lambda item: item.start_sec)
        yield RawRecording(
            dataset=DATASET,
            recording_id=f"{DATASET}:{index:05d}",
            subject_id=f"bg:{base['subject']}",
            session_id=f"{DATASET}:{index:05d}",
            streams=(
                SensorStream(
                    stream_id="wrist_imu",
                    placement="wrist",
                    device="off-the-shelf smartwatch",
                    timestamps_sec=np.arange(length, dtype=np.float64) / rate,
                    values=signal.astype(np.float32),
                    channels=_CHANNELS,
                    valid=np.ones((length, len(_CHANNELS)), dtype=bool),
                    gravity_state="present",
                    nominal_rate_hz=rate,
                    metadata={"assembled": True},
                ),
            ),
            events=tuple(events),
            split="train",
            metadata={
                "primary_exercise": primary,
                "background_exercise": background_label,
                "background_recording_id": base["recording_id"],
                "inserted_executions": len(events),
                "primary_occurrences": counts.get(primary, 0),
                "bounded_execution_annotations": True,
                # Only the inserted extents are labelled; the background is a
                # different exercise and is a labelled distractor, not empty.
                "exhaustive_annotation": False,
                "synthesis_config": config_digest(config),
            },
        )
