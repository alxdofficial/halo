"""Build a frozen HALO representation cache for an application cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.data.manifests import read_cohort_manifest
from applications.motion_monitoring.representation_cache import (
    build_representation_cache,
    file_sha256,
)
from applications.motion_monitoring.sequence import HaloMotionEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--splits", nargs="+", choices=("train", "development", "test"))
    parser.add_argument("--patch-seconds", type=float, default=1.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
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
    encoder = HaloMotionEncoder.from_checkpoint(
        args.checkpoint, device=args.device, trainable=False
    )
    encoder.eval()
    build_representation_cache(
        args.output,
        manifest,
        caches,
        encoder,
        encoder_provenance={
            "kind": "halo",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint),
        },
        datasets=selected_datasets,
        splits=set(args.splits) if args.splits else None,
        patch_seconds=args.patch_seconds,
        stride_seconds=args.stride_seconds,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()
