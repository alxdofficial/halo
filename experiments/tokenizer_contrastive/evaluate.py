"""Evaluate a contrastive tokenizer checkpoint without fitting a classifier."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .data import (
    ContrastiveWindowDataset,
    DomainBalancedBatchSampler,
    build_index,
)
from .metrics import evaluate_loader
from .train import make_loader, resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--legacy-root", default=None)
    parser.add_argument("--rotation-probability", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, config, payload = load_checkpoint(
        args.checkpoint, device, legacy_root=args.legacy_root
    )
    seed_everything(config.seed)
    refs, records, index_summary = build_index(config)
    selected = records[args.split]
    if not selected:
        raise ValueError(f"No records available for split {args.split!r}")
    evaluation_config = replace(
        config,
        rotation_probability=args.rotation_probability,
        jitter_probability=0.0,
        time_shift_probability=0.0,
    )
    evaluation_config.validate()
    dataset = ContrastiveWindowDataset(refs, selected, evaluation_config)
    sampler = DomainBalancedBatchSampler(
        selected,
        config.classes_per_batch,
        config.samples_per_class,
        args.steps,
        config.seed + 2,
    )
    loader = make_loader(dataset, sampler, config)
    metrics = evaluate_loader(
        model, loader, device, config.temperature, config.loss_mode
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": payload["epoch"],
        "split": args.split,
        "evaluation_rotation_probability": args.rotation_probability,
        "metrics": metrics,
        "split_counts": index_summary["counts"][args.split],
    }
    destination = args.output or args.checkpoint.parent / f"evaluation_{args.split}.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
