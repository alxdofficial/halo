"""Lazy adapter for the authored CrossFit smartwatch NumPy release."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re

import numpy as np

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
    split_at_clock_gaps,
)


_SOURCE_RATE_HZ = 100.0
_SOURCE_STEP_SEC = 1.0 / _SOURCE_RATE_HZ
_STANDARD_GRAVITY_M_S2 = 9.80665
_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
_PAPER_PARTICIPANT_COUNT = 54
_RELEASE_PARTICIPANT_COUNT = 57
_PSEUDO_REPETITION_FILES = frozenset(
    {
        "Burpees_495_9.npy",
        "Push ups_276_13.npy",
        "Squats_158_10.npy",
        "Squats_261_10.npy",
        "Squats_281_13.npy",
        "Squats_310_9.npy",
    }
)

_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "sources" / "crossfit" / "raw"
_EXERCISE_PATTERN = re.compile(r"^(?P<label>.+)_(?P<exercise_id>\d+)$")
_REPETITION_PATTERN = re.compile(
    r"^(?P<label>.+)_(?P<exercise_id>\d+)_(?P<repetition_index>\d+)$"
)


@dataclass(frozen=True)
class _SourceSpec:
    path: Path
    label: str
    exercise_id: int
    subject_id: str
    repetition_index: int | None
    repetition_indices: tuple[int, ...]


def _resolve_release(root: Path | None) -> tuple[Path, Path]:
    requested = Path(root) if root is not None else _DEFAULT_ROOT
    requested = requested.expanduser().resolve()
    data_candidates = (
        requested
        / "HAR_Crossfit_Sensors_Data"
        / "data"
        / "constrained_workout"
        / "preprocessed_numpy_data",
        requested / "data" / "constrained_workout" / "preprocessed_numpy_data",
        requested / "constrained_workout" / "preprocessed_numpy_data",
        requested,
    )
    data_root = next(
        (
            candidate
            for candidate in data_candidates
            if (candidate / "np_exercise_data").is_dir()
            and (candidate / "np_reps_data").is_dir()
        ),
        None,
    )
    if data_root is None:
        raise FileNotFoundError(
            f"CrossFit authored NumPy directories were not found below {requested}"
        )

    map_candidates = (
        requested / "HAR_Crossfit_Sensors_Code" / "participant_ex_code_map.txt",
        (
            data_root.parents[3]
            / "HAR_Crossfit_Sensors_Code"
            / "participant_ex_code_map.txt"
            if len(data_root.parents) > 3
            else data_root / "participant_ex_code_map.txt"
        ),
        requested / "participant_ex_code_map.txt",
    )
    participant_map = next((path for path in map_candidates if path.is_file()), None)
    if participant_map is None:
        raise FileNotFoundError(
            f"CrossFit participant-to-exercise map was not found below {requested}"
        )
    return data_root, participant_map


def _parse_exercise_path(path: Path) -> tuple[str, int]:
    match = _EXERCISE_PATTERN.fullmatch(path.stem)
    if match is None or match.group("label") != path.parent.name:
        raise ValueError(f"unexpected CrossFit exercise filename: {path}")
    return match.group("label"), int(match.group("exercise_id"))


def _parse_repetition_path(path: Path) -> tuple[str, int, int]:
    match = _REPETITION_PATTERN.fullmatch(path.stem)
    if match is None or match.group("label") != path.parent.name:
        raise ValueError(f"unexpected CrossFit repetition filename: {path}")
    return (
        match.group("label"),
        int(match.group("exercise_id")),
        int(match.group("repetition_index")),
    )


def _participant_index(path: Path) -> tuple[dict[int, str], dict[str, tuple[int, ...]]]:
    participant_to_exercises = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(participant_to_exercises, dict):
        raise ValueError("CrossFit participant map must be a JSON object")

    exercise_to_participant: dict[int, str] = {}
    normalized_participant_map: dict[str, tuple[int, ...]] = {}
    for raw_participant, exercise_ids in participant_to_exercises.items():
        participant = str(raw_participant)
        normalized_ids = tuple(int(exercise_id) for exercise_id in exercise_ids)
        normalized_participant_map[participant] = normalized_ids
        for exercise_id in normalized_ids:
            if exercise_id in exercise_to_participant:
                raise ValueError(f"duplicate CrossFit exercise id {exercise_id}")
            exercise_to_participant[exercise_id] = participant

    return exercise_to_participant, normalized_participant_map


def _quality_slices(timestamps: np.ndarray, valid: np.ndarray) -> Iterator[slice]:
    """Keep partial channel loss masked and split rows with no usable IMU sample."""

    if valid.all():
        yield slice(0, len(timestamps))
        return

    for clock_slice in split_at_clock_gaps(
        timestamps,
        max_gap_sec=_SOURCE_STEP_SEC * 1.5,
    ):
        usable = valid[clock_slice].any(axis=1)
        changes = np.flatnonzero(np.diff(np.pad(usable.astype(np.int8), (1, 1))) != 0)
        for start, end in changes.reshape(-1, 2):
            yield slice(clock_slice.start + int(start), clock_slice.start + int(end))


def _load_source_array(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    authored = np.load(path, allow_pickle=False)
    if authored.ndim != 2 or authored.shape[0] != 18 or authored.shape[1] == 0:
        raise ValueError(f"unexpected CrossFit array shape {authored.shape} in {path}")

    timestamps = np.arange(authored.shape[1], dtype=np.float64) * _SOURCE_STEP_SEC
    source_values = authored[:6].T
    valid = np.isfinite(source_values)
    values = np.array(source_values, dtype=np.float32, copy=True)
    values[~valid] = 0.0
    values[:, :3] /= np.float32(_STANDARD_GRAVITY_M_S2)
    return timestamps, values, valid


@lru_cache(maxsize=4)
def _source_manifest(
    data_root: Path,
    participant_map_path: Path,
) -> tuple[tuple[_SourceSpec, ...], tuple[str, ...]]:
    exercise_to_participant, participant_to_exercises = _participant_index(
        participant_map_path
    )
    exercise_paths = sorted(
        (data_root / "np_exercise_data").glob("*/*.npy"),
        key=lambda path: (path.parent.name.casefold(), _parse_exercise_path(path)[1]),
    )
    repetition_paths = sorted(
        (data_root / "np_reps_data").glob("*/*.npy"),
        key=lambda path: (
            path.parent.name.casefold(),
            _parse_repetition_path(path)[1],
            _parse_repetition_path(path)[2],
        ),
    )
    label_by_exercise = {
        exercise_id: label
        for exercise_path in exercise_paths
        for label, exercise_id in (_parse_exercise_path(exercise_path),)
    }
    if set(label_by_exercise) != set(exercise_to_participant):
        raise ValueError(
            "CrossFit participant map and authored exercise arrays contain different ids"
        )
    null_only_participants = tuple(
        sorted(
            (
                participant
                for participant, exercise_ids in participant_to_exercises.items()
                if exercise_ids
                and all(
                    label_by_exercise[exercise_id].casefold() == "null"
                    for exercise_id in exercise_ids
                )
            ),
            key=lambda participant: int(participant.removeprefix("P")),
        )
    )

    repetitions_by_exercise: dict[int, list[tuple[Path, str, int]]] = defaultdict(list)
    for repetition_path in repetition_paths:
        repetition_label, exercise_id, repetition_index = _parse_repetition_path(
            repetition_path
        )
        repetitions_by_exercise[exercise_id].append(
            (repetition_path, repetition_label, repetition_index)
        )

    specs: list[_SourceSpec] = []
    seen_exercises: set[int] = set()
    for exercise_path in exercise_paths:
        label, exercise_id = _parse_exercise_path(exercise_path)
        if exercise_id in seen_exercises:
            raise ValueError(f"duplicate CrossFit exercise array {exercise_id}")
        seen_exercises.add(exercise_id)
        try:
            subject_id = exercise_to_participant[exercise_id]
        except KeyError as error:
            raise ValueError(f"unmapped CrossFit exercise id {exercise_id}") from error

        repetitions = repetitions_by_exercise.pop(exercise_id, [])
        repetition_indices = tuple(item[2] for item in repetitions)
        specs.append(
            _SourceSpec(
                path=exercise_path,
                label=label,
                exercise_id=exercise_id,
                subject_id=subject_id,
                repetition_index=None,
                repetition_indices=repetition_indices,
            )
        )
        for repetition_path, repetition_label, repetition_index in repetitions:
            if repetition_label != label:
                raise ValueError(
                    f"CrossFit repetition label {repetition_label!r} disagrees with "
                    f"parent exercise label {label!r} for id {exercise_id}"
                )
            specs.append(
                _SourceSpec(
                    path=repetition_path,
                    label=repetition_label,
                    exercise_id=exercise_id,
                    subject_id=subject_id,
                    repetition_index=repetition_index,
                    repetition_indices=repetition_indices,
                )
            )

    if repetitions_by_exercise:
        orphan_ids = sorted(repetitions_by_exercise)
        raise ValueError(f"CrossFit repetitions lack parent exercises: {orphan_ids}")
    return tuple(specs), null_only_participants


def _recordings_from_array(
    *,
    path: Path,
    label: str,
    exercise_id: int,
    subject_id: str,
    repetition_index: int | None,
    repetition_indices: tuple[int, ...],
    null_only_participants: tuple[str, ...],
) -> Iterator[RawRecording]:
    timestamps, values, valid = _load_source_array(path)
    is_repetition = repetition_index is not None
    is_background = label.casefold() == "null"
    is_fragment = is_repetition and (
        path.name in _PSEUDO_REPETITION_FILES or values.shape[0] < 20
    )
    source_level = "repetition" if is_repetition else "exercise"
    base_recording_id = f"crossfit:exercise:{exercise_id}"
    if is_repetition:
        base_recording_id += f":rep:{repetition_index}"

    common_metadata = {
        "source_file": str(path),
        "source_level": source_level,
        "source_label": label,
        "exercise_id": exercise_id,
        "parent_exercise_recording_id": f"crossfit:exercise:{exercise_id}",
        "repetition_index": repetition_index,
        "is_derived_repetition_slice": is_repetition,
        "duplicates_parent_exercise_signal": is_repetition,
        "available_repetition_indices": repetition_indices,
        "source_timeline": "authored_interpolated_10_ms_grid",
        "source_array_channels": 18,
        "selected_source_channels": tuple(range(6)),
        "acceleration_source_unit": "m/s^2",
        "acceleration_output_unit": "g",
        "gyroscope_source_unit": "rad/s",
        "gyroscope_output_unit": "rad/s",
        "paper_participant_count": _PAPER_PARTICIPANT_COUNT,
        "release_participant_count": _RELEASE_PARTICIPANT_COUNT,
        "participant_count_mismatch": True,
        "null_only_participant_codes": null_only_participants,
        "participant_is_null_only": subject_id in null_only_participants,
        "pseudo_repetition_fragment": is_fragment,
        "event_excluded": is_fragment,
        "event_exclusion_reason": (
            "8-16-sample authored pseudo-repetition fragment" if is_fragment else None
        ),
        "source_clock_gap_recovery": (
            "unavailable_after_authors_interpolated the raw clocks"
        ),
    }

    slices = tuple(_quality_slices(timestamps, valid))
    if not slices:
        raise ValueError(f"CrossFit array has no usable wrist IMU samples: {path}")
    for part_index, quality_slice in enumerate(slices):
        part_timestamps = timestamps[quality_slice]
        part_values = values[quality_slice]
        part_valid = valid[quality_slice]
        recording_id = base_recording_id
        if len(slices) > 1:
            recording_id += f":quality-part:{part_index}"

        events: tuple[EventInterval, ...] = ()
        if not is_fragment:
            events = (
                EventInterval(
                    start_sec=float(part_timestamps[0]),
                    end_sec=float(part_timestamps[-1] + _SOURCE_STEP_SEC),
                    label=label,
                    annotation_kind=(
                        "background"
                        if is_background
                        else "repetition" if is_repetition else "exercise_sequence"
                    ),
                    metadata={
                        "boundary_source": "authored_array_extent",
                        "right_open": True,
                        "exercise_id": exercise_id,
                        "repetition_index": repetition_index,
                    },
                ),
            )

        yield RawRecording(
            dataset="crossfit",
            recording_id=recording_id,
            subject_id=subject_id,
            session_id=f"crossfit:exercise:{exercise_id}",
            streams=(
                SensorStream(
                    stream_id="wrist_imu",
                    placement="wrist",
                    device="off-the-shelf smartwatch",
                    timestamps_sec=part_timestamps,
                    values=part_values,
                    channels=_CHANNELS,
                    valid=part_valid,
                    gravity_state="present",
                    nominal_rate_hz=_SOURCE_RATE_HZ,
                    metadata={
                        "authored_rate_hz": _SOURCE_RATE_HZ,
                        "orientation_channels_excluded": True,
                        "ankle_channels_excluded_for_consumer_compatibility": True,
                        "invalid_value_sentinel": 0.0,
                    },
                ),
            ),
            events=events,
            metadata={
                **common_metadata,
                "quality_part_index": part_index,
                "quality_part_count": len(slices),
                "source_start_sec": float(part_timestamps[0]),
                "source_end_sec": float(part_timestamps[-1] + _SOURCE_STEP_SEC),
            },
        )


def iter_recordings(
    root: Path | None = None,
    limit: int | None = None,
) -> Iterator[RawRecording]:
    """Yield full exercises followed by their authored repetition slices lazily.

    The release contains pre-interpolated 18-channel arrays rather than original
    sensor clocks. This adapter preserves that authored 10 ms physical-time grid
    and never performs additional interpolation.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    if limit == 0:
        return

    data_root, participant_map_path = _resolve_release(root)
    source_specs, null_only_participants = _source_manifest(
        data_root,
        participant_map_path,
    )
    emitted = 0
    for spec in source_specs:
        if limit is not None and emitted >= limit:
            return
        for recording in _recordings_from_array(
            path=spec.path,
            label=spec.label,
            exercise_id=spec.exercise_id,
            subject_id=spec.subject_id,
            repetition_index=spec.repetition_index,
            repetition_indices=spec.repetition_indices,
            null_only_participants=null_only_participants,
        ):
            yield recording
            emitted += 1
            if limit is not None and emitted >= limit:
                return
