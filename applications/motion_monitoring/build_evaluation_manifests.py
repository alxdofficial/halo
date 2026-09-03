"""Build the immutable Task-3 manifests from COHORT_V1.

Task 1 builds its own cohort and manifests with
``applications.motion_monitoring.task1.build_manifests_v2``; Task 2 with
``applications.motion_monitoring.task2.build_manifests_v1``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from applications.motion_monitoring.data.manifests import read_cohort_manifest
from applications.motion_monitoring.evaluation_manifests import (
    build_task3_development_manifest,
    build_task3_train_manifest,
    build_task3_test_manifest,
    write_task_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path(__file__).resolve().parent / "manifests" / "COHORT_V1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "manifests",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    cohort = read_cohort_manifest(args.cohort)
    manifests = (
        build_task3_train_manifest(cohort, seed=args.seed),
        build_task3_development_manifest(cohort, seed=args.seed),
        build_task3_test_manifest(cohort, seed=args.seed),
    )
    for manifest in manifests:
        split = str(manifest.protocol.get("split", "test")).upper()
        path = args.output_dir / f"{manifest.task.upper()}_{split}_V1.json"
        write_task_manifest(manifest, path)
        print(
            f"{manifest.task}: {len(manifest.units)} units, "
            f"{len(manifest.exclusions)} exclusions, {manifest.fingerprint} -> {path}"
        )


if __name__ == "__main__":
    main()
