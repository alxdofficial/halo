"""Build the frozen application cohort manifest from canonical caches."""

from __future__ import annotations

import argparse
from pathlib import Path

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.data.manifests import (
    build_cohort_manifest,
    write_cohort_manifest,
)


DEFAULT_TRAINING = ("openpack", "crossfit", "aidlab_har", "recofit")
DEFAULT_EVALUATION = ("c_mhad", "wear", "oca")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="application_cohort_v1")
    parser.add_argument("--training", nargs="+", default=list(DEFAULT_TRAINING))
    parser.add_argument("--evaluation", nargs="+", default=list(DEFAULT_EVALUATION))
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--development-fraction", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "manifests" / "COHORT_V1.json",
    )
    args = parser.parse_args()
    datasets = tuple(dict.fromkeys((*args.training, *args.evaluation)))
    caches = {dataset: open_cache(dataset) for dataset in datasets}
    manifest = build_cohort_manifest(
        caches,
        name=args.name,
        training_datasets=args.training,
        evaluation_datasets=args.evaluation,
        seed=args.seed,
        development_fraction=args.development_fraction,
    )
    write_cohort_manifest(manifest, args.output)
    counts: dict[str, int] = {}
    for entry in manifest.entries:
        key = f"{entry.dataset}/{entry.split}"
        counts[key] = counts.get(key, 0) + 1
    print(f"wrote {args.output} ({manifest.fingerprint})")
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
