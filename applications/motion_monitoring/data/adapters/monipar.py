"""MoniPar weekly smartwatch protocols as bounded Task-2 executions.

One recording is one weekly visit: a subject walks once through the released
MDS-UPDRS-derived protocol while wearing a TicWatch S2 on the more affected
wrist (the dominant wrist for controls). Seven of the nine protocol items occur
exactly once per visit, so each is one bounded execution of the same task by the
same person on the same acquisition configuration, one week apart -- the unit
Task 2 compares (``docs/tasks/TASK2_CHANGE_QUANTIFICATION.md`` section 7.2).

``postural_transition`` (median 18 runs per visit) and ``resting`` (median 10)
are not single-run items and are emitted only as ``protocol_state`` tracks. A
resting run additionally becomes a bounded execution when the released per-sample
rest-tremor track labels the whole run with one MDS-UPDRS grade; that subset is
small (38 runs, grades 0/1/2 = 30/6/2) and is carried as metadata, not as a
supported evaluation cell.

Clinical labels shipped with the release and attached here:

* weekly MDS-UPDRS bradykinesia scores for items 3.4 finger tapping, 3.5 hand
  movements and 3.6 pronation-supination, for the six supervised PD subjects
  (``MONIPAR SUBJECTS DATA.xlsx``, sheet 2). Score column ``j`` is week ``j+1``:
  the only subject with fewer scores (sup03, six) is also the only subject whose
  released weeks stop at six, and its two blank scores are the last two columns.
* per-sample rest-tremor grades 0/1/2 for selected supervised and remote
  subjects (``*_TREMOR_LABEL.mat``); ``-1`` marks unlabelled signal.

Acceleration is released in m/s^2 with gravity present and is converted to g.
Each visit is resampled from its own measured clock onto a true 50 Hz grid with
the same anti-aliased rule as ``data/datasets/monipar/convert.py``: the released
millisecond clock is bimodal (49.87-52.85 Hz) against a documented 50 Hz, and a
5.7 % shift would move the tremor bands this dataset exists to measure.
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)


_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1] / "sources" / "monipar" / "downloads" / "x"
)
_RATE_HZ = 50.0
_RATE_TOLERANCE_HZ = 0.05
_GRAVITY_MS2 = 9.80665
_CHANNELS = ("acc_x", "acc_y", "acc_z")
# Guard: a still wrist reads ~9.81 in m/s^2. g would land near 1.0 and milli-g
# near 1000; either would silently corrupt the gravity-bearing channels.
_ACC_MS2_RANGE = (8.0, 11.5)

# (raw file stem, MATLAB variable, subject prefix, cohort)
_SUBGROUPS = (
    ("MONIPAR_PD_SUPERVISED", "SUPERVISED_RAWDATA", "sup", "pd_supervised"),
    ("MONIPAR_PD_REMOTE", "REMOTE_RAWDATA", "rem", "pd_remote"),
    ("MONIPAR_HEALTHYCONTROL", "HEALTHYCONTROL_RAWDATA", "hc", "healthy_control"),
)
_TREMOR_FILES = {
    "sup": ("MONIPAR_PD_SUPERVISED_TREMOR_LABEL", "SUPERVISED_TREMOR_LABEL"),
    "rem": ("MONIPAR_PD_REMOTE_TREMOR_LABEL", "REMOTE_TREMOR_LABEL"),
}

_EXERCISES = {
    0: "postural_transition",
    1: "resting",
    2: "postural_hand_tremor",
    3: "moving_hands_to_the_chest",
    4: "finger_tapping",
    5: "rapid_hand_opening_and_closing",
    6: "pronation_supination",
    7: "arising_from_a_chair",
    8: "gait",
}
# Items that occur exactly once per visit; verified per visit before use.
TASK2_EXECUTION_LABELS = (
    "arising_from_a_chair",
    "finger_tapping",
    "gait",
    "moving_hands_to_the_chest",
    "postural_hand_tremor",
    "pronation_supination",
    "rapid_hand_opening_and_closing",
)
_MULTI_RUN_LABELS = ("postural_transition", "resting")

_BRADYKINESIA_SHEET = "MDS UPDRS Brady Supervised Grp."
_BRADYKINESIA_ITEMS = {
    "3.4 FINGER TAPPING": "finger_tapping",
    "3.5 HAND MOVEMENTS": "rapid_hand_opening_and_closing",
    "3.6 PRONATION-SUPINATION MOVEMENTS OF HANDS": "pronation_supination",
}


def _source_root(root: Path | None) -> Path:
    source = _DEFAULT_ROOT if root is None else Path(root)
    if not source.is_dir():
        raise FileNotFoundError(f"MoniPar release not found at {source}")
    return source


@lru_cache(maxsize=4)
def _bradykinesia_scores(source: Path) -> dict[tuple[str, int, str], int]:
    """(subject, week, label) -> weekly clinician MDS-UPDRS bradykinesia score."""

    import pandas as pd

    path = source / "MONIPAR SUBJECTS DATA.xlsx"
    frame = pd.read_excel(path, sheet_name=_BRADYKINESIA_SHEET, header=None)
    scores: dict[tuple[str, int, str], int] = {}
    label: str | None = None
    for _, row in frame.iterrows():
        item = row[2]
        if isinstance(item, str) and item.strip() in _BRADYKINESIA_ITEMS:
            label = _BRADYKINESIA_ITEMS[item.strip()]
        subject = row[3]
        if not isinstance(subject, str) or not subject.startswith("SUPERVISED_"):
            continue
        if label is None:
            raise ValueError(f"{path}: a score row precedes its MDS-UPDRS item")
        index = int(subject.split("_")[1])
        for offset, value in enumerate(row[4:12].tolist()):
            if isinstance(value, str) or value is None:
                continue
            if isinstance(value, float) and np.isnan(value):
                continue
            scores[(f"sup{index:02d}", offset + 1, label)] = int(value)
    if not scores:
        raise ValueError(f"{path}: no weekly bradykinesia scores were parsed")
    return scores


@lru_cache(maxsize=4)
def _tremor_cells(source: Path) -> dict[str, Any]:
    from scipy.io import loadmat

    cells: dict[str, Any] = {}
    for prefix, (stem, variable) in _TREMOR_FILES.items():
        path = source / f"{stem}.mat"
        if path.exists():
            cells[prefix] = loadmat(path)[variable]
    return cells


@lru_cache(maxsize=4)
def _raw_cells(source: Path, stem: str, variable: str):
    from scipy.io import loadmat

    path = source / f"{stem}.mat"
    if not path.exists():
        raise FileNotFoundError(f"missing MoniPar release file: {path}")
    return loadmat(path)[variable]


def _measured_rate_hz(timestamps_ms: np.ndarray) -> float:
    span = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
    if span <= 0:
        raise ValueError("MoniPar trial clock does not advance")
    return (len(timestamps_ms) - 1) / span


def _to_native_rate(
    acceleration: np.ndarray, codes: np.ndarray, tremor: np.ndarray, source_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anti-alias resample to exactly 50 Hz; categorical tracks ride along."""

    if abs(source_hz - _RATE_HZ) <= _RATE_TOLERANCE_HZ:
        return acceleration, codes, tremor
    from scipy.signal import resample_poly

    ratio = Fraction(_RATE_HZ / source_hz).limit_denominator(2000)
    values = resample_poly(acceleration, up=ratio.numerator, down=ratio.denominator, axis=0)
    index = np.clip(
        np.round(np.linspace(0, len(codes) - 1, len(values))).astype(int),
        0,
        len(codes) - 1,
    )
    return values, codes[index], tremor[index]


def _runs(labels: np.ndarray) -> list[tuple[int, int]]:
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], changes, [len(labels)]))
    return [(int(start), int(stop)) for start, stop in zip(bounds[:-1], bounds[1:])]


def _activity_events(
    labels: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[EventInterval, ...]:
    if len(labels) != len(timestamps) or not len(labels):
        raise ValueError("MoniPar labels and timestamps must be non-empty and aligned")
    period = 1.0 / _RATE_HZ
    return tuple(
        EventInterval(
            start_sec=float(timestamps[start]),
            end_sec=float(timestamps[stop - 1] + period),
            label=str(labels[start]),
            annotation_kind="protocol_state",
            metadata={
                "source": "sample_labels",
                "week_session": True,
                "independent_repetition": False,
            },
        )
        for start, stop in _runs(labels)
    )


def _execution_events(
    labels: np.ndarray,
    timestamps: np.ndarray,
    tremor: np.ndarray,
    *,
    subject_id: str,
    week: int,
    scores: dict[tuple[str, int, str], int],
) -> tuple[EventInterval, ...]:
    """One bounded execution per single-run protocol item, plus labelled rest runs."""

    period = 1.0 / _RATE_HZ
    runs = _runs(labels)
    counts: dict[str, int] = {}
    for start, _ in runs:
        counts[str(labels[start])] = counts.get(str(labels[start]), 0) + 1
    events: list[EventInterval] = []
    seen: dict[str, int] = {}
    for start, stop in runs:
        label = str(labels[start])
        run_index = seen.get(label, 0)
        seen[label] = run_index + 1
        grades = np.unique(tremor[start:stop])
        grade = int(grades[0]) if len(grades) == 1 and grades[0] >= 0 else None
        if label in TASK2_EXECUTION_LABELS:
            if counts[label] != 1:
                # The single-execution rule is verified per visit, never assumed.
                continue
        elif not (label == "resting" and grade is not None):
            continue
        metadata: dict[str, Any] = {
            "source": "sample_labels",
            "week": week,
            "run_index": run_index,
            "runs_in_visit": counts[label],
            "single_run_item": label in TASK2_EXECUTION_LABELS,
        }
        score = scores.get((subject_id, week, label))
        if score is not None:
            metadata["mds_updrs_bradykinesia"] = int(score)
            metadata["mds_updrs_rater"] = "clinician, weekly in-clinic assessment"
        if grade is not None:
            metadata["mds_updrs_rest_tremor"] = grade
        events.append(
            EventInterval(
                start_sec=float(timestamps[start]),
                end_sec=float(timestamps[stop - 1] + period),
                label=label,
                annotation_kind="bounded_execution",
                metadata=metadata,
            )
        )
    return tuple(events)


def iter_recordings(
    root: Path | None = None,
    limit: int | None = None,
) -> Iterator[RawRecording]:
    """Yield one weekly visit per recording in canonical g units."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    source = _source_root(root)
    scores = _bradykinesia_scores(source)
    tremor_cells = _tremor_cells(source)
    yielded = 0
    for stem, variable, prefix, cohort in _SUBGROUPS:
        cells = _raw_cells(source, stem, variable)
        tremor_group = tremor_cells.get(prefix)
        for subject_index in range(cells.shape[0]):
            subject_id = f"{prefix}{subject_index + 1:02d}"
            for week_index in range(cells.shape[1]):
                if limit is not None and yielded >= limit:
                    return
                trial = cells[subject_index, week_index]
                if np.size(trial) == 0:
                    continue  # the subject missed that week
                trial = np.asarray(trial, dtype=np.float64)
                if trial.shape[1] != 5:
                    raise ValueError(
                        f"{subject_id} week {week_index + 1}: expected 5 columns"
                    )
                if not np.isfinite(trial).all():
                    raise ValueError(f"{subject_id} week {week_index + 1}: non-finite samples")
                timestamps_ms, acceleration, codes = trial[:, 0], trial[:, 1:4], trial[:, 4]
                if not np.all(np.diff(timestamps_ms) > 0):
                    raise ValueError(f"{subject_id} week {week_index + 1}: clock is not monotonic")
                unknown = set(np.unique(codes).astype(int)) - set(_EXERCISES)
                if unknown:
                    raise ValueError(
                        f"{subject_id} week {week_index + 1}: unknown label codes {unknown}"
                    )
                magnitude = float(np.median(np.linalg.norm(acceleration, axis=1)))
                if not _ACC_MS2_RANGE[0] <= magnitude <= _ACC_MS2_RANGE[1]:
                    raise ValueError(
                        f"{subject_id} week {week_index + 1}: median |acc| {magnitude:.3f} is "
                        f"outside the documented m/s^2 range {_ACC_MS2_RANGE}"
                    )
                if tremor_group is None or tremor_group[subject_index, week_index].size == 0:
                    tremor = np.full(len(codes), -1, dtype=np.int64)
                else:
                    tremor = np.asarray(
                        tremor_group[subject_index, week_index], dtype=np.int64
                    ).ravel()
                    if len(tremor) != len(codes):
                        raise ValueError(
                            f"{subject_id} week {week_index + 1}: tremor track length differs"
                        )
                source_hz = _measured_rate_hz(timestamps_ms)
                values, codes, tremor = _to_native_rate(
                    acceleration, codes.astype(int), tremor, source_hz
                )
                labels = np.asarray([_EXERCISES[int(code)] for code in codes])
                timestamps = np.arange(len(values), dtype=np.float64) / _RATE_HZ
                week = week_index + 1
                session_id = f"{subject_id}_w{week:02d}"
                yield RawRecording(
                    dataset="monipar",
                    recording_id=f"monipar:{session_id}",
                    subject_id=subject_id,
                    session_id=session_id,
                    streams=(
                        SensorStream(
                            stream_id="watch_wrist",
                            placement="wrist",
                            device="TicWatch S2 smartwatch",
                            timestamps_sec=timestamps,
                            values=(values / _GRAVITY_MS2).astype(np.float32),
                            channels=_CHANNELS,
                            valid=np.ones((len(values), len(_CHANNELS)), dtype=bool),
                            gravity_state="present",
                            nominal_rate_hz=_RATE_HZ,
                            metadata={
                                "source_acceleration_unit": "m/s^2",
                                "output_acceleration_unit": "g",
                                "measured_source_rate_hz": source_hz,
                                "resampled_to_native_rate": bool(
                                    abs(source_hz - _RATE_HZ) > _RATE_TOLERANCE_HZ
                                ),
                            },
                        ),
                    ),
                    events=(
                        *_activity_events(labels, timestamps),
                        *_execution_events(
                            labels,
                            timestamps,
                            tremor,
                            subject_id=subject_id,
                            week=week,
                            scores=scores,
                        ),
                    ),
                    split="evaluation",
                    metadata={
                        "cohort": cohort,
                        "week": week,
                        "visit_kind": "weekly_protocol",
                        "bounded_execution_annotations": True,
                        "independent_repetition_annotations": False,
                        "multi_run_items": list(_MULTI_RUN_LABELS),
                    },
                )
                yielded += 1
