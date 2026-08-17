"""Fail-loud readiness audit for the matched zero-shot and enrollment protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import baselines
from baselines.base import BaselineAdapter
from eval.enrollment_protocol import ACTION_REGIMES, iter_cells, load_manifest


PAPER_BASELINES = ("harnet", "crosshar", "limubert", "unimts", "imagebind", "normwear")


def audit(manifest_path: Path, baseline_names=PAPER_BASELINES) -> dict:
    manifest = load_manifest(manifest_path, validate_grids=True)
    blockers, warnings = [], []
    if len(manifest["seeds"]) < 5:
        blockers.append("fewer than five serialized support seeds")
    if manifest["support_counts"][:5] != [0, 1, 2, 4, 8]:
        blockers.append("main k curve is not exactly 0,1,2,4,8")

    zero_datasets = {
        cell["dataset"] for _, cell in iter_cells(manifest, kinds=["zero_shot"])
        if cell["status"] == "ok"
    }
    positive_datasets = {
        cell["dataset"] for _, cell in iter_cells(manifest, kinds=["enrollment"])
        if cell["status"] == "ok"
    }
    expected = set(manifest["datasets"])
    if zero_datasets != expected:
        blockers.append(f"missing zero-shot datasets: {sorted(expected - zero_datasets)}")
    if positive_datasets != expected:
        warnings.append(
            "no valid positive-k relation for datasets: "
            f"{sorted(expected - positive_datasets)}"
        )

    baseline_status = {}
    for name in baseline_names:
        adapter = baselines.REGISTRY.get(name)
        if adapter is None:
            blockers.append(f"baseline adapter is not registered: {name}")
            continue
        features_overridden = type(adapter).window_features is not BaselineAdapter.window_features
        candidates_overridden = (
            type(adapter).predict_candidates is not BaselineAdapter.predict_candidates
            or adapter.tier in {"conse", "cosine"}
        )
        cached_prediction = (
            type(adapter).predict_candidates_from_features
            is not BaselineAdapter.predict_candidates_from_features
        )
        baseline_status[name] = {
            "adapter": f"{type(adapter).__module__}.{type(adapter).__name__}",
            "frozen_features": features_overridden,
            "candidate_override": candidates_overridden,
            "cached_feature_prediction": cached_prediction,
        }
        if not features_overridden:
            blockers.append(f"{name}: no frozen window feature interface")
        if not candidates_overridden:
            blockers.append(f"{name}: cannot score the manifest candidate roster")
        if not cached_prediction:
            blockers.append(f"{name}: k=0 would require a second sensor encoding pass")

    enrollment = [cell for _, cell in iter_cells(manifest, kinds=["enrollment"])]
    ceilings = {}
    secondary_ready = 0
    for cell in enrollment:
        ceilings[str(cell["support_ceiling"])] = ceilings.get(str(cell["support_ceiling"]), 0) + 1
        secondary_ready += int(cell.get("secondary_high_support", {}).get("status") == "ok")
    return {
        "ready": not blockers,
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "datasets": manifest["datasets"],
        "action_regimes": {key: list(value) for key, value in ACTION_REGIMES.items()},
        "zero_shot_datasets": sorted(zero_datasets),
        "positive_k_datasets": sorted(positive_datasets),
        "positive_relation_support_ceilings": ceilings,
        "secondary_k16_relations": secondary_ready,
        "baseline_status": baseline_status,
        "blockers": blockers,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.manifest)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
