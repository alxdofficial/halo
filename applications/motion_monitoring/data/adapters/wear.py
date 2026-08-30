"""Lazy raw-timeline adapter for the WEAR outdoor sports dataset."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import lru_cache
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)


_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "sources" / "wear"
_SOURCE_RATE_HZ = 50.0
_CHANNELS = ("acc_x", "acc_y", "acc_z")
_SESSION_RE = re.compile(r"sbj_(?P<index>\d+)")
_SUBJECT_ALIASES = {18: 0, 19: 14}
_ARM_COLUMNS = {
    "right_arm": ("right_arm_acc_x", "right_arm_acc_y", "right_arm_acc_z"),
    "left_arm": ("left_arm_acc_x", "left_arm_acc_y", "left_arm_acc_z"),
}
_PLACEMENTS = {"right_arm": "right_wrist", "left_arm": "left_wrist"}
_REQUIRED_COLUMNS = (
    "sbj_id",
    *_ARM_COLUMNS["right_arm"],
    *_ARM_COLUMNS["left_arm"],
    "label",
)
_CSV_DTYPES = {
    "sbj_id": np.int16,
    **{column: np.float32 for columns in _ARM_COLUMNS.values() for column in columns},
}


def _resolve_raw_root(root: Path | None) -> Path:
    candidate = _DEFAULT_ROOT if root is None else Path(root)
    options = (candidate, candidate / "raw")
    for option in options:
        if (option / "inertial_50hz").is_dir() and (
            option / "annotations_60fps"
        ).is_dir():
            return option
    raise FileNotFoundError(
        f"WEAR raw inertial and annotation folders not found beneath {candidate}"
    )


def _annotation_signature(record: Mapping[str, object]) -> str:
    return json.dumps(
        {
            "duration": record.get("duration"),
            "fps": record.get("fps"),
            "annotations": record.get("annotations"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@lru_cache(maxsize=4)
def _load_annotations(
    annotation_root: Path,
) -> tuple[
    dict[str, Mapping[str, object]], dict[str, tuple[str, ...]], Mapping[str, int]
]:
    records: dict[str, Mapping[str, object]] = {}
    signatures: dict[str, str] = {}
    sources: dict[str, list[str]] = {}
    label_dict: Mapping[str, int] | None = None

    files = sorted(annotation_root.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"no WEAR annotation JSON files found in {annotation_root}"
        )

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != "Wear":
            raise ValueError(f"unexpected WEAR annotation version in {path.name}")
        current_label_dict = payload.get("label_dict")
        database = payload.get("database")
        if not isinstance(current_label_dict, Mapping) or not isinstance(
            database, Mapping
        ):
            raise ValueError(f"invalid WEAR annotation schema in {path.name}")
        if label_dict is None:
            label_dict = current_label_dict
        elif dict(current_label_dict) != dict(label_dict):
            raise ValueError(f"WEAR label dictionaries disagree in {path.name}")

        for session_id, record in database.items():
            if not isinstance(session_id, str) or not isinstance(record, Mapping):
                raise ValueError(f"invalid WEAR database entry in {path.name}")
            signature = _annotation_signature(record)
            if session_id in signatures and signatures[session_id] != signature:
                raise ValueError(f"WEAR annotations disagree for {session_id}")
            signatures.setdefault(session_id, signature)
            records.setdefault(session_id, record)
            sources.setdefault(session_id, []).append(path.name)

    if label_dict is None:
        raise ValueError("WEAR annotations contain no label dictionary")
    return records, {key: tuple(value) for key, value in sources.items()}, label_dict


def _positive_row_runs(labels: np.ndarray) -> tuple[tuple[str, float, float], ...]:
    labels = np.asarray(labels, dtype=object)
    normalized = np.where(pd.isna(labels), "null", labels)
    blank = normalized == ""
    if np.any(blank):
        normalized[blank] = "null"
    changes = np.flatnonzero(normalized[1:] != normalized[:-1]) + 1
    bounds = np.concatenate(([0], changes, [len(normalized)]))
    return tuple(
        (
            str(normalized[start]),
            float(start / _SOURCE_RATE_HZ),
            float(stop / _SOURCE_RATE_HZ),
        )
        for start, stop in zip(bounds[:-1], bounds[1:])
        if str(normalized[start]).lower() != "null"
    )


def _annotation_intervals(
    record: Mapping[str, object],
    *,
    duration_sec: float,
    label_dict: Mapping[str, int],
) -> tuple[EventInterval, ...]:
    annotations = record.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("WEAR annotation record has no annotations list")

    events: list[EventInterval] = []
    cursor = 0.0
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            raise ValueError("invalid WEAR interval annotation")
        label = annotation.get("label")
        segment = annotation.get("segment")
        frame_segment = annotation.get("segment (frames)")
        label_id = annotation.get("label_id")
        if (
            not isinstance(label, str)
            or not isinstance(segment, list)
            or len(segment) != 2
            or not isinstance(frame_segment, list)
            or len(frame_segment) != 2
            or not isinstance(label_id, int)
        ):
            raise ValueError("invalid WEAR interval fields")
        if label not in label_dict or int(label_dict[label]) != label_id:
            raise ValueError(f"WEAR interval has inconsistent label id for {label!r}")

        start_sec, end_sec = (float(segment[0]), float(segment[1]))
        start_frame, end_frame = (float(frame_segment[0]), float(frame_segment[1]))
        if not np.allclose(
            (start_frame / 60.0, end_frame / 60.0),
            (start_sec, end_sec),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(f"WEAR frame and second intervals disagree for {label!r}")
        if start_sec < cursor - 1e-9 or end_sec > duration_sec + 1e-9:
            raise ValueError(
                f"WEAR interval falls outside the inertial timeline: {label!r}"
            )

        if start_sec > cursor:
            events.append(
                EventInterval(
                    start_sec=cursor,
                    end_sec=start_sec,
                    label="NULL",
                    annotation_kind="background",
                    metadata={"usage": "scoring_only", "source": "json_complement"},
                )
            )
        events.append(
            EventInterval(
                start_sec=start_sec,
                end_sec=end_sec,
                label=label,
                annotation_kind="activity",
                metadata={
                    "usage": "scoring_only",
                    "source": "interval_json",
                    "source_label_id": label_id,
                    "source_frame_segment": (start_frame, end_frame),
                },
            )
        )
        cursor = end_sec

    if cursor < duration_sec:
        events.append(
            EventInterval(
                start_sec=cursor,
                end_sec=duration_sec,
                label="NULL",
                annotation_kind="background",
                metadata={"usage": "scoring_only", "source": "json_complement"},
            )
        )
    return tuple(events)


def _reconcile_row_labels(
    row_labels: np.ndarray,
    events: tuple[EventInterval, ...],
) -> dict[str, object]:
    row_runs = list(_positive_row_runs(row_labels))
    json_runs = [
        (event.label, event.start_sec, event.end_sec)
        for event in events
        if event.annotation_kind == "activity"
    ]
    matched = 0
    row_only: list[tuple[str, float, float]] = []
    start_offsets_samples: set[int] = set()
    row_index = 0
    for label, json_start, json_end in json_runs:
        while row_index < len(row_runs) and row_runs[row_index][0] != label:
            row_only.append(row_runs[row_index])
            row_index += 1
        if row_index >= len(row_runs):
            raise ValueError(f"WEAR row labels are missing JSON activity {label!r}")
        row_label, row_start, row_end = row_runs[row_index]
        expected_row_start = max(0.0, json_start - 1.0 / _SOURCE_RATE_HZ)
        if (
            row_label != label
            or not np.isclose(row_start, expected_row_start, rtol=0.0, atol=1e-9)
            or not np.isclose(row_end, json_end, rtol=0.0, atol=1e-9)
        ):
            raise ValueError(
                "WEAR row labels and JSON annotations disagree for "
                f"{label!r}: row=({row_start}, {row_end}), "
                f"json=({json_start}, {json_end})"
            )
        matched += 1
        start_offsets_samples.add(round((json_start - row_start) * _SOURCE_RATE_HZ))
        row_index += 1
    row_only.extend(row_runs[row_index:])
    return {
        "json_activity_interval_count": len(json_runs),
        "row_label_activity_run_count": len(row_runs),
        "matched_activity_interval_count": matched,
        "row_label_only_intervals": tuple(row_only),
        "json_start_offsets_samples": tuple(sorted(start_offsets_samples)),
    }


def _sensor_stream(
    frame: pd.DataFrame,
    timestamps_sec: np.ndarray,
    *,
    stream_id: str,
) -> SensorStream:
    columns = _ARM_COLUMNS[stream_id]
    values = frame.loc[:, columns].to_numpy(dtype=np.float32, copy=True)
    valid = np.isfinite(values)
    return SensorStream(
        stream_id=stream_id,
        placement=_PLACEMENTS[stream_id],
        device="Bangle.js smartwatch",
        timestamps_sec=timestamps_sec,
        values=values,
        channels=_CHANNELS,
        valid=valid,
        gravity_state="present",
        nominal_rate_hz=_SOURCE_RATE_HZ,
        metadata={
            "side": stream_id.removesuffix("_arm"),
            "source_placement": stream_id,
            "source_columns": columns,
            "source_acceleration_unit": "g",
            "application_compatible_placement": True,
            "missing_value_policy": "explicit_validity_mask_no_imputation",
        },
    )


def iter_recordings(
    root: Path | None = None,
    limit: int | None = None,
) -> Iterator[RawRecording]:
    """Yield WEAR sessions with consumer-compatible arm streams at native release rate."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    if limit == 0:
        return

    raw_root = _resolve_raw_root(root)
    annotation_records, annotation_sources, label_dict = _load_annotations(
        raw_root / "annotations_60fps"
    )
    csv_paths = sorted(
        (raw_root / "inertial_50hz").glob("sbj_*.csv"),
        key=lambda path: int(path.stem.removeprefix("sbj_")),
    )
    if not csv_paths:
        raise FileNotFoundError(f"no WEAR inertial CSV files found in {raw_root}")

    for yielded, path in enumerate(csv_paths):
        if limit is not None and yielded >= limit:
            return
        match = _SESSION_RE.fullmatch(path.stem)
        if match is None:
            raise ValueError(f"unexpected WEAR session filename: {path.name}")
        source_index = int(match.group("index"))
        session_id = path.stem
        annotation_record = annotation_records.get(session_id)
        if annotation_record is None:
            raise ValueError(f"WEAR has no interval annotations for {session_id}")

        frame = pd.read_csv(
            path,
            usecols=list(_REQUIRED_COLUMNS),
            dtype=_CSV_DTYPES,
            low_memory=False,
        )
        source_ids = frame["sbj_id"].dropna().unique()
        if len(source_ids) != 1 or int(source_ids[0]) != source_index:
            raise ValueError(f"WEAR subject column disagrees with {path.name}")
        duration_sec = len(frame) / _SOURCE_RATE_HZ
        declared_duration_sec = float(annotation_record.get("duration", np.nan))
        if not np.isfinite(declared_duration_sec):
            raise ValueError(f"WEAR has invalid declared duration for {session_id}")
        timestamps_sec = np.arange(len(frame), dtype=np.float64) / _SOURCE_RATE_HZ
        events = _annotation_intervals(
            annotation_record,
            duration_sec=duration_sec,
            label_dict=label_dict,
        )
        reconciliation = _reconcile_row_labels(frame["label"].to_numpy(), events)

        canonical_subject_index = _SUBJECT_ALIASES.get(source_index, source_index)
        official_partition = "training" if source_index < 18 else "testing"
        recording = RawRecording(
            dataset="wear",
            recording_id=f"wear:{session_id}",
            subject_id=f"sbj_{canonical_subject_index}",
            session_id=session_id,
            streams=tuple(
                _sensor_stream(frame, timestamps_sec, stream_id=stream_id)
                for stream_id in ("right_arm", "left_arm")
            ),
            events=events,
            split=official_partition,
            metadata={
                "source_file": path.name,
                "source_subject_id": session_id,
                "identity_alias_applied": canonical_subject_index != source_index,
                "official_partition": official_partition,
                "annotation_usage": "scoring_only",
                "annotation_sources": annotation_sources[session_id],
                "annotation_fps": int(annotation_record.get("fps", 0)),
                "json_declared_duration_sec": declared_duration_sec,
                "inertial_duration_sec": duration_sec,
                "declared_duration_difference_sec": declared_duration_sec
                - duration_sec,
                "source_timing": "row_index_at_postprocessed_fixed_50_hz",
                "source_placements": (
                    "right_arm",
                    "right_leg",
                    "left_leg",
                    "left_arm",
                ),
                "preferred_application_streams": ("right_arm", "left_arm"),
                "excluded_source_placements": ("right_leg", "left_leg"),
                **reconciliation,
            },
        )
        del frame
        yield recording
