"""Recalibrate an existing Task-1 head on a frozen common development set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from applications.motion_monitoring.data.examples import open_cache
from applications.motion_monitoring.data.manifests import read_cohort_manifest
from applications.motion_monitoring.evaluation import fingerprint_protocol
from applications.motion_monitoring.evaluation_manifests import read_task_manifest
from applications.motion_monitoring.representation_cache import open_representations
from applications.motion_monitoring.task1.model import DifferentiableSubsequenceMatcher
from applications.motion_monitoring.task1.train_full import calibrate, select_common_units


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=root / "manifests/COHORT_V1.json")
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=root / "manifests/TASK1_DEVELOPMENT_V1.json",
    )
    parser.add_argument("--common-development-units", type=Path, required=True)
    parser.add_argument("--representations", type=Path, nargs="+", required=True)
    parser.add_argument("--head-directory", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cohort = read_cohort_manifest(args.cohort)
    manifest = read_task_manifest(args.development_manifest)
    manifest, common = select_common_units(manifest, args.common_development_units)
    assert common is not None
    representations = open_representations(args.representations, cohort=cohort)
    checkpoint = torch.load(
        args.head_directory / "task1_head.pt", map_location="cpu", weights_only=False
    )
    model = DifferentiableSubsequenceMatcher(
        int(checkpoint["feature_dim"]),
        projection_dim=int(checkpoint["projection_dim"]),
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    caches = {
        dataset: open_cache(dataset)
        for dataset in sorted({str(row["dataset"]) for row in manifest.units})
    }
    report_path = args.head_directory / "task1_training.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["direct"] = calibrate(manifest, caches, representations, None)
    report["learned"] = calibrate(manifest, caches, representations, model)
    report["development_common_unit_count"] = len(manifest.units)
    report["development_common_unit_fingerprint"] = fingerprint_protocol(common)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "development_common_unit_count": len(manifest.units),
                "direct": report["direct"],
                "learned": report["learned"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
