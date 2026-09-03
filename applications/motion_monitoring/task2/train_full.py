"""Fit the Task-2 ruler on the declared training pool.

Two data roles only: this reads the train manifest's pool declaration, draws
compatibility-clean batches under a seed, and writes a checkpoint bound to the
cohort, the manifest and the encoder. It never touches the evaluation manifest,
and it fits no threshold: evaluation derives a personal reference-only operating
limit from each person's accepted reference set.

The run record carries the positive-relation mix. That is the number to read
first: a ruler trained almost entirely on within-session positives has never been
asked to tolerate a remount, and the evaluation's nuisance false-alarm rate on
between-week repeats is where that would show up.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from applications.motion_monitoring.data.manifests import read_cohort_manifest
from applications.motion_monitoring.evaluation_manifests import read_task_manifest
from applications.motion_monitoring.representation_cache import open_representations
from .contracts import collate_execution_episodes
from .episodes import Task2BatchBuilder, relation_summary
from .losses import RulerLossConfig
from .model import ChangeRuler
from .records import build_record_pool, record_identity
from .training import train_step


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps <= 0 or args.groups <= 0 or args.telemetry_every <= 0:
        raise ValueError("steps, groups, and telemetry interval must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError(
            "learning rate and gradient clip must be positive; weight decay must be non-negative"
        )
    cohort = read_cohort_manifest(args.cohort)
    manifest = read_task_manifest(args.train_manifest)
    if manifest.task != "task2" or manifest.protocol.get("split") != "train":
        raise ValueError("--train-manifest must be the Task-2 train manifest")
    if manifest.cohort_fingerprint != cohort.fingerprint:
        raise ValueError("Task-2 train manifest belongs to another cohort")
    representations = open_representations(args.representations, cohort=cohort)
    common_text = args.common_units.read_text(encoding="utf-8")
    common = json.loads(common_text)
    if common.get("train_manifest_fingerprint") != manifest.fingerprint:
        raise ValueError("common Task-2 units belong to another train manifest")
    provenance = representations.metadata["encoder_provenance"]
    if not any(
        row.get("encoder_provenance") == provenance
        for row in common.get("representations", {}).values()
    ):
        raise ValueError("common Task-2 units were not built for this encoder")
    datasets = [str(row["dataset"]) for row in manifest.protocol["sources"]]

    started = time.time()
    pool = build_record_pool(datasets, representations, strict=False)
    allowed = set(common["selected_train_execution_ids"])
    available = {record_identity(record) for record in pool}
    missing = allowed - available
    if missing:
        raise ValueError(
            f"representation cache is missing {len(missing)} common Task-2 executions"
        )
    pool = [record for record in pool if record_identity(record) in allowed]
    if not pool:
        raise ValueError("the declared training pool produced no usable executions")
    pool_composition: dict[str, int] = {}
    for record in pool:
        key = f"{record.dataset}/{record.variant}"
        pool_composition[key] = pool_composition.get(key, 0) + 1
    for source in manifest.protocol["dataset_weights"]:
        if not any(key.startswith(f"{source}/") for key in pool_composition):
            raise ValueError(f"declared Task-2 source {source!r} produced no usable executions")
        if not any(
            key.startswith(f"{source}/") and not key.endswith("/clean")
            for key in pool_composition
        ):
            raise ValueError(f"declared Task-2 source {source!r} produced no modified variants")
    builder = Task2BatchBuilder(
        pool,
        reference_count=int(manifest.protocol["reference_count"]),
        positives_per_episode=args.positives_per_episode,
        modified_per_episode=args.modified_per_episode,
        other_subject_per_episode=args.other_subject_per_episode,
        dataset_weights=dict(manifest.protocol["dataset_weights"]),
        seed=args.seed,
    )
    if not builder.eligible_groups:
        raise ValueError("no reference-set group has enough executions for this episode shape")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = ChangeRuler(
        pool[0].execution.embeddings.shape[1],
        phase_bins=args.phase_bins,
        projection_dim=args.projection_dim,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_config = RulerLossConfig(margin=args.margin, pull_weight=args.pull_weight)
    telemetry: list[dict[str, Any]] = []
    plans = []
    for step in range(args.steps):
        episodes, batch_plans = builder.build_batch(groups=args.groups, seed=args.seed + step)
        plans.extend(batch_plans)
        batch = collate_execution_episodes(episodes).to(args.device)
        result = train_step(model, batch, optimizer, loss_config=loss_config, grad_clip=args.grad_clip)
        if step == 0 or (step + 1) % args.telemetry_every == 0 or step + 1 == args.steps:
            telemetry.append({"step": step + 1, **vars(result)})
    model.eval()

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "embedding_dim": pool[0].execution.embeddings.shape[1],
            "phase_bins": args.phase_bins,
            "projection_dim": args.projection_dim,
            "model_state_dict": model.state_dict(),
            "cohort_fingerprint": cohort.fingerprint,
            "train_manifest_fingerprint": manifest.fingerprint,
            "representation_provenance": representations.metadata["encoder_provenance"],
            "pool_composition": pool_composition,
            "common_unit_fingerprint": sha256(common_text.encode("utf-8")).hexdigest(),
            "config": vars(args),
        },
        args.output / "task2_head.pt",
    )
    report = {
        "task": "task2",
        "status": "trained",
        "cohort_fingerprint": cohort.fingerprint,
        "train_manifest_fingerprint": manifest.fingerprint,
        "representation_provenance": representations.metadata["encoder_provenance"],
        "pool_executions": len(pool),
        "pool_composition": pool_composition,
        "common_unit_fingerprint": sha256(common_text.encode("utf-8")).hexdigest(),
        "eligible_groups": len(builder.eligible_groups),
        "steps": args.steps,
        "seconds": time.time() - started,
        "relation_mix": relation_summary(plans),
        "telemetry": telemetry,
    }
    (args.output / "task2_training.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=root / "manifests/COHORT_TASK2_V1.json")
    parser.add_argument(
        "--train-manifest", type=Path, default=root / "manifests/TASK2_TRAIN_V1.json"
    )
    parser.add_argument("--representations", type=Path, nargs="+", required=True)
    parser.add_argument("--common-units", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--positives-per-episode", type=int, default=2)
    parser.add_argument("--modified-per-episode", type=int, default=2)
    parser.add_argument("--other-subject-per-episode", type=int, default=1)
    parser.add_argument("--phase-bins", type=int, default=8)
    parser.add_argument("--projection-dim", type=int)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--pull-weight", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--telemetry-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    report = train(args)
    print(json.dumps({k: v for k, v in report.items() if k != "telemetry"}, indent=2, default=str))


if __name__ == "__main__":
    main()
