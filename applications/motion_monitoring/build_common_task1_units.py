"""Freeze the Task-1 unit intersection representable by every compared encoder."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.evaluation_manifests import (
    Task1EvaluationUnit,
    read_task_manifest,
)
from applications.motion_monitoring.representation_cache import CachedMotionSequenceDataset
from applications.motion_monitoring.task1.episodes import (
    crop_sequence,
    episode_from_recordings,
    from_motion_sequence,
)


def _parse(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("representation must use NAME=PATH")
    return name, Path(path)


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--representation", action="append", type=_parse, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_task_manifest(args.manifest)
    representations = {
        name: CachedMotionSequenceDataset(path) for name, path in args.representation
    }
    recording_caches = {
        dataset: open_cache(dataset)
        for dataset in sorted({str(row["dataset"]) for row in manifest.units})
    }
    selected = []
    excluded = []
    counts = Counter()
    for index, raw in enumerate(manifest.units):
        unit = Task1EvaluationUnit(**raw)
        reference_recording = recording_caches[unit.dataset][unit.reference_cache_index]
        query_recording = recording_caches[unit.dataset][unit.query_cache_index]
        failed = []
        reasons = {}
        for name, cache in representations.items():
            try:
                query_sequence = from_motion_sequence(
                    cache.get(
                        unit.dataset,
                        unit.query_recording_id,
                        unit.query_stream_id,
                    )
                )
                if getattr(unit, "query_interval_sec", None) is not None:
                    block_start, block_end = map(float, unit.query_interval_sec)
                    query_sequence = crop_sequence(query_sequence, block_start, block_end)
                episode_from_recordings(
                    reference_recording,
                    query_recording,
                    from_motion_sequence(
                        cache.get(
                            unit.dataset,
                            unit.reference_recording_id,
                            unit.reference_stream_id,
                        )
                    ),
                    query_sequence,
                    label=unit.label,
                    reference_event_index=unit.reference_event_index,
                    target_intervals_sec=unit.target_intervals_sec,
                    reference_interval_sec=getattr(unit, "reference_interval_sec", None),
                )
            except ValueError as error:
                failed.append(name)
                reasons[name] = str(error)
        if failed:
            excluded.append(
                {"unit_index": index, "encoders": failed, "reasons": reasons}
            )
            counts[(unit.dataset, "excluded")] += 1
        else:
            selected.append(index)
            counts[(unit.dataset, "selected")] += 1
    payload = {
        "schema_version": 1,
        "task_manifest_fingerprint": manifest.fingerprint,
        "representations": {
            name: cache.metadata["encoder_provenance"]
            for name, cache in representations.items()
        },
        "selected_unit_indices": selected,
        "excluded": excluded,
        "counts": {
            f"{dataset}/{status}": count
            for (dataset, status), count in sorted(counts.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
