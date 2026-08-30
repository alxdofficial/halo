"""Run reproducible real-payload checks over application dataset adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from applications.motion_monitoring.data.adapters.registry import (
    adapter_names,
    iter_recordings,
)
from applications.motion_monitoring.data.contracts import RawRecording, SensorStream


def _sample_period(stream: SensorStream) -> float:
    if len(stream.timestamps_sec) > 1:
        return float(np.median(np.diff(stream.timestamps_sec)))
    if stream.nominal_rate_hz:
        return 1.0 / stream.nominal_rate_hz
    return 0.0


def _validate_recording(recording: RawRecording) -> dict[str, Any]:
    if any(len(stream.timestamps_sec) < 2 for stream in recording.streams):
        raise ValueError(f"{recording.recording_id}: stream has fewer than two samples")

    observed_start = min(
        float(stream.timestamps_sec[0]) for stream in recording.streams
    )
    observed_end = max(
        float(stream.timestamps_sec[-1]) + _sample_period(stream)
        for stream in recording.streams
    )
    for event in recording.events:
        if (
            event.start_sec < observed_start - 1e-6
            or event.end_sec > observed_end + 1e-6
        ):
            raise ValueError(
                f"{recording.recording_id}: event {event.label!r} "
                f"[{event.start_sec}, {event.end_sec}] exceeds observed sensor span "
                f"[{observed_start}, {observed_end}]"
            )

    samples = sum(len(stream.timestamps_sec) for stream in recording.streams)
    invalid = sum(int((~stream.valid).sum()) for stream in recording.streams)
    values = sum(int(stream.valid.size) for stream in recording.streams)
    rates = [1.0 / _sample_period(stream) for stream in recording.streams]
    return {
        "recording_id": recording.recording_id,
        "subject_id": recording.subject_id,
        "session_id": recording.session_id,
        "streams": len(recording.streams),
        "samples": samples,
        "events": len(recording.events),
        "duration_sec": observed_end - observed_start,
        "rates_hz": rates,
        "invalid_values": invalid,
        "value_count": values,
        "invalid_fraction": invalid / values if values else 0.0,
    }


def verify_dataset(dataset: str, *, limit: int) -> dict[str, Any]:
    rows = [
        _validate_recording(recording)
        for recording in iter_recordings(dataset, limit=limit)
    ]
    if not rows:
        raise ValueError(f"{dataset}: adapter yielded no recordings")
    recording_ids = [row["recording_id"] for row in rows]
    if len(recording_ids) != len(set(recording_ids)):
        raise ValueError(
            f"{dataset}: duplicate recording identifiers in verification sample"
        )
    rates = [rate for row in rows for rate in row["rates_hz"]]
    return {
        "status": "pass",
        "recordings_checked": len(rows),
        "streams_checked": sum(row["streams"] for row in rows),
        "samples_checked": sum(row["samples"] for row in rows),
        "events_checked": sum(row["events"] for row in rows),
        "duration_sec_checked": sum(row["duration_sec"] for row in rows),
        "rate_hz_min": min(rates),
        "rate_hz_median": float(np.median(rates)),
        "rate_hz_max": max(rates),
        "invalid_fraction": (
            sum(row["invalid_values"] for row in rows)
            / sum(row["value_count"] for row in rows)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", metavar="DATASET")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    unknown = sorted(set(args.datasets) - set(adapter_names()))
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}")

    datasets = args.datasets or list(adapter_names())
    report = {
        "schema_version": 1,
        "recordings_per_dataset_limit": args.limit,
        "datasets": {name: verify_dataset(name, limit=args.limit) for name in datasets},
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
