"""Freeze the Task-2 train-pool and test-unit intersection across encoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.data.manifests import read_cohort_manifest
from applications.motion_monitoring.evaluation_manifests import (
    Task2EvaluationUnit,
    read_task_manifest,
    validate_task_manifest,
)
from applications.motion_monitoring.representation_cache import (
    open_representations,
    require_complete_representations,
)
from applications.motion_monitoring.task2.evaluate_v1 import _execution
from applications.motion_monitoring.task2.records import build_record_pool, record_identity


def _parse(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("representation must use NAME=PATH")
    return name, Path(path)


def _roots(cache) -> list[str]:
    return list(cache.metadata.get("member_roots", [str(cache.root)]))


def _stride(cache, dataset: str) -> float:
    return float(
        cache.metadata.get("stride_seconds_by_dataset", {}).get(
            dataset, cache.metadata.get("stride_seconds", 1.0)
        )
    )


def main() -> None:
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=root / "manifests/COHORT_TASK2_V1.json")
    parser.add_argument(
        "--train-manifest", type=Path, default=root / "manifests/TASK2_TRAIN_V1.json"
    )
    parser.add_argument(
        "--test-manifest", type=Path, default=root / "manifests/TASK2_TEST_V1.json"
    )
    parser.add_argument("--representation", action="append", type=_parse, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cohort = read_cohort_manifest(args.cohort)
    train = read_task_manifest(args.train_manifest)
    test = read_task_manifest(args.test_manifest)
    if train.task != "task2" or train.protocol.get("split") != "train":
        raise ValueError("--train-manifest must be the Task-2 train manifest")
    if test.task != "task2" or test.protocol.get("split") != "test":
        raise ValueError("--test-manifest must be the Task-2 test manifest")
    if train.cohort_fingerprint != cohort.fingerprint or test.cohort_fingerprint != cohort.fingerprint:
        raise ValueError("Task-2 manifests belong to another cohort")
    grouped: dict[str, list[Path]] = {}
    for name, path in args.representation:
        grouped.setdefault(name, []).append(path)
    representations = {
        name: open_representations(paths, cohort=cohort) for name, paths in grouped.items()
    }
    for cache in representations.values():
        require_complete_representations(cache)
    datasets = sorted(
        {
            *(str(row["dataset"]) for row in train.protocol["sources"]),
            *(str(row["dataset"]) for row in test.units),
        }
    )
    for dataset in datasets:
        strides = {name: _stride(cache, dataset) for name, cache in representations.items()}
        if len(set(strides.values())) != 1:
            raise ValueError(
                f"Task-2 representations use unequal temporal strides for {dataset}: {strides}"
            )

    recording_caches = {dataset: open_cache(dataset) for dataset in datasets}
    validate_task_manifest(test, cohort, recording_caches)
    train_datasets = [str(row["dataset"]) for row in train.protocol["sources"]]
    identities_by_encoder = {
        name: {
            record_identity(record)
            for record in build_record_pool(train_datasets, cache, strict=False)
        }
        for name, cache in representations.items()
    }
    selected_train = sorted(set.intersection(*identities_by_encoder.values()))
    if not selected_train:
        raise ValueError("the common Task-2 training execution intersection is empty")

    selected_test: list[int] = []
    exclusions: list[dict[str, object]] = []
    for unit_index, raw in enumerate(test.units):
        unit = Task2EvaluationUnit(**raw)
        failed: dict[str, str] = {}
        for name, cache in representations.items():
            source = recording_caches[unit.dataset]
            refs = [
                _execution(source, cache, unit.dataset, index, event, unit.stream_id)
                for index, event in zip(
                    unit.reference_cache_indices, unit.reference_event_indices
                )
            ]
            query = _execution(
                source,
                cache,
                unit.dataset,
                unit.query_cache_index,
                unit.query_event_index,
                unit.stream_id,
            )
            if query is None or any(item is None for item in refs):
                failed[name] = "missing or invalid bounded representation"
        if failed:
            exclusions.append({"unit_index": unit_index, "encoders": failed})
        else:
            selected_test.append(unit_index)
    if not selected_test:
        raise ValueError("the common Task-2 test-unit intersection is empty")

    payload = {
        "schema_version": 1,
        "train_manifest_fingerprint": train.fingerprint,
        "test_manifest_fingerprint": test.fingerprint,
        "representations": {
            name: {
                "encoder_provenance": cache.metadata["encoder_provenance"],
                "roots": _roots(cache),
                "stride_seconds_by_dataset": {
                    dataset: _stride(cache, dataset) for dataset in datasets
                },
                "patch_seconds": cache.metadata.get("patch_seconds"),
            }
            for name, cache in representations.items()
        },
        "selected_train_execution_ids": selected_train,
        "selected_test_unit_indices": selected_test,
        "excluded_test_units": exclusions,
        "counts": {
            "train_execution_intersection": len(selected_train),
            "test_unit_intersection": len(selected_test),
            "test_units_excluded": len(exclusions),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
