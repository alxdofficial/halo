"""Probe released baseline encoders on real Task-2 longitudinal streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from applications.motion_monitoring.baseline_encoder import (
    OPTIONAL_APPLICATION_BASELINES,
    PRIMARY_APPLICATION_BASELINES,
    BaselineMotionEncoder,
)
from applications.motion_monitoring.data.adapters.registry import iter_recordings
from applications.motion_monitoring.data.examples import crop_recording


_DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "smoke"
    / "task2_baseline_data_probe.json"
)


def _stream_crop(recording, stream, duration_sec: float):
    start = float(stream.timestamps_sec[0])
    stop = min(
        start + duration_sec,
        float(stream.timestamps_sec[-1]) + 1.0 / (stream.nominal_rate_hz or 1.0),
    )
    return crop_recording(
        recording,
        start,
        stop,
        recording_suffix=f"baseline-probe-{stream.stream_id}",
        retain_overlapping_events=False,
    )


def run(
    baselines: list[str],
    *,
    device: str,
    duration_sec: float = 20.0,
) -> dict[str, Any]:
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    recordings = {
        dataset: next(iter_recordings(dataset, limit=1))
        for dataset in ("alameda", "cops")
    }
    results: dict[str, Any] = {}
    for name in baselines:
        encoder = BaselineMotionEncoder(name, device=device)
        cases: list[dict[str, Any]] = []
        for dataset, recording in recordings.items():
            for stream in recording.streams:
                crop = _stream_crop(recording, stream, duration_sec)
                started = perf_counter()
                sequence = encoder.encode_recording(crop, stream_id=stream.stream_id)
                live = sequence.embeddings[sequence.valid]
                if not len(live):
                    raise ValueError(
                        f"{name}/{dataset}/{stream.stream_id}: no valid embeddings"
                    )
                singular = torch.linalg.svdvals(live.float())
                probability = singular / singular.sum().clamp_min(1e-12)
                effective_rank = float(
                    torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum())
                )
                cases.append(
                    {
                        "dataset": dataset,
                        "recording_id": recording.recording_id,
                        "stream_id": stream.stream_id,
                        "placement": stream.placement,
                        "channels": list(stream.channels),
                        "source_rate_hz": stream.nominal_rate_hz,
                        "embedding_shape": list(sequence.embeddings.shape),
                        "valid_embeddings": int(sequence.valid.sum().item()),
                        "effective_rank": effective_rank,
                        "elapsed_sec": perf_counter() - started,
                    }
                )
        results[name] = {
            "status": "pass",
            "provenance": encoder.provenance(),
            "cases": cases,
        }
    return {
        "status": "mechanical_compatibility_only",
        "device": device,
        "duration_sec": duration_sec,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(PRIMARY_APPLICATION_BASELINES),
        choices=(*PRIMARY_APPLICATION_BASELINES, *OPTIONAL_APPLICATION_BASELINES),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--duration-sec", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(
        args.baselines,
        device=args.device,
        duration_sec=args.duration_sec,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
