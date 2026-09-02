"""Targeted integrity and semantic audit for Task-2 longitudinal sources."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

from applications.motion_monitoring.data.adapters.alameda import (
    ELIGIBLE_CAMPAIGNS,
    _archive_path as alameda_archive_path,
    _campaign,
)
from applications.motion_monitoring.data.adapters.cops import (
    _HOURLY_RE,
    _archive_paths as cops_archive_paths,
    _diary as cops_diary,
)
from applications.motion_monitoring.data.adapters.registry import iter_recordings
from applications.motion_monitoring.data.compatibility import sensor_compatibility_key
from applications.motion_monitoring.sequence import measured_rate_hz


_DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "data" / "inspection" / "task2_audit.json"
)


def _audit_recording(recording) -> dict[str, Any]:
    violations: list[str] = []
    streams = []
    for stream in recording.streams:
        rate = measured_rate_hz(stream)
        if stream.nominal_rate_hz is not None:
            relative_error = abs(rate - stream.nominal_rate_hz) / stream.nominal_rate_hz
            if relative_error > 0.02:
                violations.append(
                    f"{stream.stream_id}: measured rate differs from nominal by {relative_error:.1%}"
                )
        if not np.isfinite(stream.values[stream.valid]).all():
            violations.append(f"{stream.stream_id}: non-finite valid values")
        acceleration = [
            index
            for index, name in enumerate(stream.channels)
            if name.startswith("acc_")
        ]
        magnitude_quantiles = None
        if len(acceleration) == 3:
            magnitude = np.linalg.norm(stream.values[:, acceleration], axis=1)
            magnitude_quantiles = np.quantile(magnitude, [0.01, 0.5, 0.99]).tolist()
            if (
                stream.gravity_state == "present"
                and not 0.5 <= magnitude_quantiles[1] <= 1.5
            ):
                violations.append(
                    f"{stream.stream_id}: gravity-present median magnitude is "
                    f"{magnitude_quantiles[1]:.3f} g"
                )
        streams.append(
            {
                "stream_id": stream.stream_id,
                "placement": stream.placement,
                "device": stream.device,
                "channels": list(stream.channels),
                "samples": len(stream.values),
                "duration_sec": float(
                    stream.timestamps_sec[-1] - stream.timestamps_sec[0]
                ),
                "measured_rate_hz": rate,
                "nominal_rate_hz": stream.nominal_rate_hz,
                "valid_fraction": float(stream.valid.mean()),
                "acceleration_magnitude_quantiles_g": magnitude_quantiles,
                "compatibility_key": asdict(
                    sensor_compatibility_key(
                        device=stream.device,
                        placement=stream.placement,
                        channels=stream.channels,
                        gravity_state=stream.gravity_state,
                    )
                ),
            }
        )
    starts = [float(stream.timestamps_sec[0]) for stream in recording.streams]
    stops = [float(stream.timestamps_sec[-1]) for stream in recording.streams]
    for event in recording.events:
        if event.start_sec < min(starts) or event.end_sec > max(stops) + 0.05:
            violations.append(f"event {event.label!r} falls outside sensor support")
    bounded_events = bool(recording.metadata.get("bounded_event_annotations", False))
    bounded_executions = bool(
        recording.metadata.get("bounded_execution_annotations", False)
    )
    if (bounded_events or bounded_executions) != bool(recording.events):
        violations.append("bounded-event declaration disagrees with event availability")
    if bounded_executions and not bool(
        recording.metadata.get("independent_repetition_annotations", True)
    ):
        violations.append("execution annotations cannot deny independent repetitions")
    event_durations = np.asarray(
        [event.end_sec - event.start_sec for event in recording.events],
        dtype=np.float64,
    )
    return {
        "recording_id": recording.recording_id,
        "subject_id": recording.subject_id,
        "session_id": recording.session_id,
        "event_count": len(recording.events),
        "event_labels": sorted({event.label for event in recording.events}),
        "event_duration_sec": (
            {
                "min": float(event_durations.min()),
                "median": float(np.median(event_durations)),
                "max": float(event_durations.max()),
                "under_5_sec": int((event_durations < 5.0).sum()),
            }
            if len(event_durations)
            else None
        ),
        "metadata": dict(recording.metadata),
        "streams": streams,
        "violations": violations,
    }


def run(*, samples_per_source: int = 2) -> dict[str, Any]:
    if samples_per_source <= 0:
        raise ValueError("samples_per_source must be positive")
    sources: dict[str, Any] = {}
    for dataset in ("alameda", "cops"):
        samples = [
            _audit_recording(recording)
            for recording in iter_recordings(dataset, limit=samples_per_source)
        ]
        sources[dataset] = {
            "sample_count": len(samples),
            "samples": samples,
            "violation_count": sum(len(sample["violations"]) for sample in samples),
        }

    alameda_path = alameda_archive_path(None)
    with ZipFile(alameda_path) as archive:
        present_campaigns = {
            name.split("/")[2]
            for name in archive.namelist()
            if name.startswith("PD GeneActiv Dataset/GeneActiv Recordings/")
            and len(name.split("/")) > 3
        }
        alameda_days: Counter[str] = Counter(
            _campaign(name)
            for name in archive.namelist()
            if name.endswith(".parquet") and _campaign(name) in ELIGIBLE_CAMPAIGNS
        )
    alameda_linked = 0
    alameda_unlinked = 0
    for recording in iter_recordings("alameda"):
        if recording.metadata.get("clinical_linkage_status") == "linked_within_14_days":
            alameda_linked += 1
        else:
            alameda_unlinked += 1
    cops_paths = cops_archive_paths(None)
    cops_hours = 0
    cops_bilateral_hours = 0
    cops_diary_linked_hours = 0
    cops_scored_hours = 0
    for path in cops_paths:
        with ZipFile(path) as archive:
            diary = cops_diary(archive, path.stem)
            diary_by_zip = {}
            for _, row in diary.iterrows():
                for column in ("WearableDataLeftZIP", "WearableDataRightZIP"):
                    name = row.get(column)
                    if isinstance(name, str) and name.strip():
                        diary_by_zip[name.strip()] = row
            groups: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
            for member in archive.namelist():
                match = _HOURLY_RE.fullmatch(Path(member).name)
                if match is None:
                    continue
                groups[(match.group("day"), match.group("hours"))][
                    match.group("side")
                ] = member
            cops_hours += len(groups)
            cops_bilateral_hours += sum(
                set(sides) == {"left", "right"} for sides in groups.values()
            )
            for members in groups.values():
                rows = [
                    diary_by_zip[PurePosixPath(member).name]
                    for member in members.values()
                    if PurePosixPath(member).name in diary_by_zip
                ]
                if not rows:
                    continue
                cops_diary_linked_hours += 1
                row = rows[0]
                if any(
                    pd.notna(pd.to_numeric(row.get(name), errors="coerce"))
                    for name in (
                        "KinesiaScore",
                        "TremorScore",
                        "FreezingScore",
                        "FallScore",
                    )
                ):
                    cops_scored_hours += 1
    coverage = {
        "alameda": {
            "eligible_campaigns": len(ELIGIBLE_CAMPAIGNS),
            "eligible_campaigns_present": len(ELIGIBLE_CAMPAIGNS & present_campaigns),
            "eligible_daily_recordings": int(sum(alameda_days.values())),
            "daily_recordings_by_campaign": dict(sorted(alameda_days.items())),
            "daily_recordings_linked_to_clinical_visit_within_14_days": alameda_linked,
            "daily_recordings_without_clinical_link": alameda_unlinked,
            "role": "exploratory free-living long-term state association",
        },
        "cops": {
            "participant_archives": len(cops_paths),
            "compressed_bytes": sum(path.stat().st_size for path in cops_paths),
            "hourly_recordings": cops_hours,
            "bilateral_hourly_recordings": cops_bilateral_hours,
            "unilateral_hourly_recordings": cops_hours - cops_bilateral_hours,
            "diary_linked_hourly_recordings": cops_diary_linked_hours,
            "hourly_recordings_with_numeric_symptom_score": cops_scored_hours,
            "role": "short-term free-living symptom-state fluctuation",
        },
    }
    blockers = []
    if coverage["alameda"]["eligible_campaigns_present"] != len(ELIGIBLE_CAMPAIGNS):
        blockers.append("ALAMEDA eligible campaign set is incomplete")
    if coverage["cops"]["participant_archives"] != 66:
        blockers.append("COPS corpus does not contain all 66 participant archives")
    if any(source["violation_count"] for source in sources.values()):
        blockers.append("sampled longitudinal recordings violate the raw-data contract")
    return {
        "status": "ready_for_representation_smokes" if not blockers else "blocked",
        "sources": sources,
        "coverage": coverage,
        "blockers": blockers,
        "warnings": [],
        "semantic_boundary": (
            "ALAMEDA and COPS are free-living state sources and must not be relabeled as "
            "action executions. Controlled known-change development uses PHYTMO and KneE-PAD."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-source", type=int, default=2)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(samples_per_source=args.samples_per_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
