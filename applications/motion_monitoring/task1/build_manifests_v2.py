"""Build the Task-1 V2 cohort and split manifests (spec sections C and F).

Roles (TASK1_REFERENCE_RESOLUTION_SPEC.md section F.2, two data roles only):

* train      = ``synth_wrist_v1`` only (synthetic; the operating point is fixed
               on its held-out background subjects, never on natural data)
* evaluation = C-MHAD and OpenPack, every subject; reported under both the
               ``cross_subject`` and ``same_subject`` reference relations

There is no development split. COHORT_V1 and the Task-3 manifests are untouched;
Task 1 gets its own cohort so its roles can diverge from the shared cohort
without invalidating the per-dataset representation caches (see
``representation_cache``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.data.manifests import (
    build_cohort_manifest,
    write_cohort_manifest,
)
from applications.motion_monitoring.evaluation_manifests import (
    build_task1_test_manifest,
    build_task1_train_manifest,
    write_task_manifest,
)


COHORT_NAME = "cohort_task1_v2"
TRAIN_ONLY = ("synth_wrist_v1",)
EVALUATION = ("c_mhad", "openpack")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "manifests",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    datasets = sorted(TRAIN_ONLY + EVALUATION)
    caches = {dataset: open_cache(dataset) for dataset in datasets}
    cohort = build_cohort_manifest(
        caches,
        name=COHORT_NAME,
        seed=args.seed,
        train_only_datasets=TRAIN_ONLY,
        evaluation_datasets=EVALUATION,
    )
    cohort_path = args.output_dir / "COHORT_TASK1_V2.json"
    write_cohort_manifest(cohort, cohort_path)
    print(f"cohort: {len(cohort.entries)} entries, {cohort.fingerprint} -> {cohort_path}")
    for dataset in datasets:
        splits = {}
        for entry in cohort.entries_for(dataset=dataset):
            splits.setdefault(entry.split, set()).add(entry.leakage_group)
        print(
            f"  {dataset:15s} "
            + ", ".join(f"{split}={len(groups)} groups" for split, groups in sorted(splits.items()))
        )

    manifests = (
        build_task1_train_manifest(cohort, caches, seed=args.seed, name="task1_train_v2"),
        build_task1_test_manifest(cohort, caches, seed=args.seed, name="task1_test_v2"),
    )
    for manifest in manifests:
        split = str(manifest.protocol["split"]).upper()
        path = args.output_dir / f"TASK1_{split}_V2.json"
        write_task_manifest(manifest, path)
        present = sum(1 for unit in manifest.units if unit["target_present"])
        relations = {}
        for unit in manifest.units:
            relations[unit["reference_relation"]] = relations.get(unit["reference_relation"], 0) + 1
        print(
            f"{manifest.name}: {len(manifest.units)} units ({present} present; "
            + ", ".join(f"{k}={v}" for k, v in sorted(relations.items()))
            + f"), {len(manifest.exclusions)} exclusions, {manifest.fingerprint} -> {path}"
        )


if __name__ == "__main__":
    main()
