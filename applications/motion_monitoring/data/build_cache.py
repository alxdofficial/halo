"""Build lossless, map-style caches from the application raw adapters."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from applications.motion_monitoring.data.adapters.registry import (
    ADAPTERS,
    iter_recordings,
    materializable_adapter_names,
)
from applications.motion_monitoring.data.cache import (
    CACHE_SCHEMA_VERSION,
    _storage_name,
    cache_provenance,
    verify_source_payload,
    write_recording,
)


DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parent / "sources"


def build_dataset_cache(dataset: str, *, force: bool = False) -> dict[str, object]:
    try:
        spec = ADAPTERS[dataset]
    except KeyError as error:
        raise KeyError(f"unknown application dataset {dataset!r}") from error
    if spec.cache_policy not in {"materialize", "derived"}:
        raise ValueError(
            f"{dataset}: adapter is stream-only; building a canonical cache would "
            "duplicate or expand a large existing payload"
        )
    verify_source_payload(dataset)
    output = DEFAULT_SOURCE_ROOT / dataset / "processed" / "canonical_v1"
    staging = output.with_name(f".{output.name}.building")
    if output.exists() and not force:
        raise FileExistsError(
            f"cache already exists: {output}; pass --force to rebuild"
        )
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    try:
        for recording in iter_recordings(dataset):
            directory_name = _storage_name(recording.recording_id)
            write_recording(recording, staging / directory_name)
            rows.append(
                {
                    "recording_id": recording.recording_id,
                    "subject_id": recording.subject_id,
                    "session_id": recording.session_id,
                    "split": recording.split,
                    "directory": directory_name,
                    "stream_count": len(recording.streams),
                    "event_count": len(recording.events),
                }
            )
        if not rows:
            raise ValueError(f"{dataset}: raw adapter yielded no recordings")
        (staging / "manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        (staging / "cache.json").write_text(
            json.dumps(
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "dataset": dataset,
                    "recording_count": len(rows),
                    "source": (
                        "deterministic synthesis from canonical caches"
                        if spec.cache_policy == "derived"
                        else "lossless raw-adapter output"
                    ),
                    "resampled": False,
                    "windowed": False,
                    "imputed": False,
                    "provenance": cache_provenance(dataset),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"dataset": dataset, "recordings": len(rows), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", choices=materializable_adapter_names())
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    worker_count = min(args.workers, len(args.datasets))
    if worker_count == 1:
        reports = [
            build_dataset_cache(name, force=args.force) for name in args.datasets
        ]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(build_dataset_cache, name, force=args.force)
                for name in args.datasets
            ]
            reports = [future.result() for future in futures]
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
