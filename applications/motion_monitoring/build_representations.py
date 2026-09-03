"""Build a frozen HALO or released-baseline representation cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.baseline_encoder import (
    OPTIONAL_APPLICATION_BASELINES,
    PRIMARY_APPLICATION_BASELINES,
    BaselineMotionEncoder,
)
from applications.motion_monitoring.data.manifests import read_cohort_manifest
from applications.motion_monitoring.evaluation_manifests import (
    Task1EvaluationUnit,
    Task2EvaluationUnit,
    read_task_manifest,
)
from applications.motion_monitoring.representation_cache import (
    BoundedSegment,
    build_bounded_representation_cache,
    build_representation_cache,
    file_sha256,
)
from applications.motion_monitoring.sequence import HaloMotionEncoder


def _bounded_kind(value: str) -> tuple[str, str]:
    dataset, separator, kind = value.partition("=")
    if not separator or not dataset or not kind:
        raise argparse.ArgumentTypeError("bounded kind must use DATASET=ANNOTATION_KIND")
    return dataset, kind


def _manifest_segments(path: Path, caches) -> list[BoundedSegment]:
    task = read_task_manifest(path)
    segments: list[BoundedSegment] = []
    if task.task == "task1":
        for raw in task.units:
            unit = Task1EvaluationUnit(**raw)
            start, end = unit.reference_interval_sec
            segments.append(
                BoundedSegment(
                    unit.dataset,
                    unit.reference_cache_index,
                    unit.reference_recording_id,
                    unit.reference_stream_id,
                    unit.reference_event_index,
                    float(start),
                    float(end),
                )
            )
    elif task.task == "task2":
        for raw in task.units:
            unit = Task2EvaluationUnit(**raw)
            specifications = list(
                zip(
                    unit.reference_cache_indices,
                    unit.reference_recording_ids,
                    unit.reference_event_indices,
                )
            ) + [
                (unit.query_cache_index, unit.query_recording_id, unit.query_event_index)
            ]
            for cache_index, recording_id, event_index in specifications:
                recording = caches[unit.dataset][cache_index]
                event = recording.events[event_index]
                segments.append(
                    BoundedSegment(
                        unit.dataset,
                        cache_index,
                        recording_id,
                        unit.stream_id,
                        event_index,
                        float(event.start_sec),
                        float(event.end_sec),
                    )
                )
    else:
        raise ValueError("bounded representation manifests support Task 1 and Task 2")
    return segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--baseline",
        choices=(*PRIMARY_APPLICATION_BASELINES, *OPTIONAL_APPLICATION_BASELINES),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--splits", nargs="+", choices=("train", "development", "test"))
    parser.add_argument(
        "--patch-seconds",
        type=float,
        help="override the encoder's native receptive field (HALO defaults to 1 second)",
    )
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument(
        "--bounded-task-manifest",
        type=Path,
        action="append",
        default=[],
        help="encode each Task-1 reference or Task-2 execution independently",
    )
    parser.add_argument(
        "--bounded-kind",
        type=_bounded_kind,
        action="append",
        default=[],
        help="encode every event of DATASET=ANNOTATION_KIND independently",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="validate and reuse complete sequences in an interrupted staging cache",
    )
    args = parser.parse_args()

    manifest = read_cohort_manifest(args.manifest)
    selected_datasets = set(args.datasets or manifest.cache_fingerprints)
    unknown = selected_datasets - set(manifest.cache_fingerprints)
    if unknown:
        raise ValueError(f"datasets are absent from the cohort manifest: {sorted(unknown)}")
    caches = {
        dataset: open_cache(dataset)
        for dataset in manifest.cache_fingerprints
    }
    if args.baseline is not None:
        encoder = BaselineMotionEncoder(args.baseline, device=args.device)
        provenance = encoder.provenance()
        patch_seconds = args.patch_seconds
    else:
        encoder = HaloMotionEncoder.from_checkpoint(
            args.checkpoint, device=args.device, trainable=False
        )
        provenance = {
            "kind": "halo",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint),
        }
        patch_seconds = 1.0 if args.patch_seconds is None else args.patch_seconds
    encoder.eval()
    if args.bounded_task_manifest or args.bounded_kind:
        if args.splits or args.limit:
            raise ValueError("bounded caches do not accept --splits or --limit")
        segments: list[BoundedSegment] = []
        for path in args.bounded_task_manifest:
            segments.extend(_manifest_segments(path, caches))
        entries = {
            (entry.dataset, entry.cache_index, entry.recording_id): entry
            for entry in manifest.entries
        }
        for dataset, kind in args.bounded_kind:
            if dataset not in selected_datasets:
                raise ValueError(f"bounded dataset {dataset!r} was not selected")
            for (entry_dataset, cache_index, recording_id), entry in entries.items():
                if entry_dataset != dataset:
                    continue
                recording = caches[dataset][cache_index]
                for event_index, event in enumerate(recording.events):
                    if event.annotation_kind != kind:
                        continue
                    for stream_id in entry.stream_ids:
                        segments.append(
                            BoundedSegment(
                                dataset,
                                cache_index,
                                recording_id,
                                stream_id,
                                event_index,
                                float(event.start_sec),
                                float(event.end_sec),
                            )
                        )
        build_bounded_representation_cache(
            args.output,
            manifest,
            caches,
            encoder,
            segments,
            encoder_provenance=provenance,
            patch_seconds=patch_seconds,
            stride_seconds=args.stride_seconds,
            force=args.force,
            resume=args.resume,
        )
    else:
        build_representation_cache(
            args.output,
            manifest,
            caches,
            encoder,
            encoder_provenance=provenance,
            datasets=selected_datasets,
            splits=set(args.splits) if args.splits else None,
            patch_seconds=patch_seconds,
            stride_seconds=args.stride_seconds,
            limit=args.limit,
            force=args.force,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
