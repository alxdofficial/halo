"""Train the HALO physical-Hz tokenizer with paired supervised contrastive views."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import ExperimentConfig
from .data import (
    ContrastiveWindowDataset,
    CoverageBalancedBatchSampler,
    DomainBalancedBatchSampler,
    build_index,
    collate_windows,
)
from .losses import supervised_contrastive_loss
from .metrics import evaluate_loader
from .model import ContrastiveTokenizerEncoder, tokenizer_source_metadata


OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError("device must be cpu, cuda, or auto")


def make_loader(
    dataset: ContrastiveWindowDataset,
    sampler: DomainBalancedBatchSampler,
    config: ExperimentConfig,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_windows,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=config.num_workers > 0,
        pin_memory=False,
    )


def train_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    config: ExperimentConfig,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        rates = batch["rate_hz"].to(device)
        mask = batch["channel_mask"].to(device)
        lengths = batch["window_length"].to(device)
        labels = batch["label_id"].to(device)
        output_a = model(batch["view_a"].to(device), rates, mask, lengths)
        output_b = model(batch["view_b"].to(device), rates, mask, lengths)
        features = torch.stack([output_a["projection"], output_b["projection"]], dim=1)
        loss = supervised_contrastive_loss(
            features, labels, temperature=config.temperature, mode=config.loss_mode
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def run_training(
    config: ExperimentConfig,
    output_dir: Path,
    device: torch.device,
    overwrite: bool = False,
) -> dict:
    config.validate()
    seed_everything(config.seed)
    refs, records, index_summary = build_index(config)
    existing = list(output_dir.iterdir()) if output_dir.exists() else []
    if existing and not overwrite:
        raise FileExistsError(
            f"Run directory {output_dir} is not empty; choose another --run-name or pass --overwrite"
        )
    if overwrite:
        for path in existing:
            if path.is_file() and path.name != ".gitignore":
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    (output_dir / "index_summary.json").write_text(
        json.dumps(index_summary, indent=2) + "\n"
    )

    train_dataset = ContrastiveWindowDataset(refs, records["train"], config)
    evaluation_config = replace(
        config,
        rotation_probability=1.0,
        jitter_probability=0.0,
        time_shift_probability=0.0,
    )
    validation_dataset = ContrastiveWindowDataset(
        refs, records["validation"], evaluation_config
    )
    if config.epoch_mode == "full_coverage":
        train_sampler = CoverageBalancedBatchSampler(
            records["train"], config.classes_per_batch, config.samples_per_class,
            config.seed,
        )
    else:
        train_sampler = DomainBalancedBatchSampler(
            records["train"], config.classes_per_batch, config.samples_per_class,
            config.steps_per_epoch, config.seed,
        )
    validation_sampler = DomainBalancedBatchSampler(
        records["validation"], config.classes_per_batch, config.samples_per_class,
        config.validation_steps, config.seed + 1,
    )
    train_loader = make_loader(train_dataset, train_sampler, config)
    validation_loader = make_loader(validation_dataset, validation_sampler, config)

    model = ContrastiveTokenizerEncoder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    source_metadata = tokenizer_source_metadata(config.legacy_root)
    (output_dir / "tokenizer_source.json").write_text(
        json.dumps(source_metadata, indent=2) + "\n"
    )

    history = []
    validation_sampler.set_epoch(0)
    seed_everything(config.seed + 100_000)
    initial_validation = evaluate_loader(
        model, validation_loader, device, config.temperature, config.loss_mode
    )
    initial_metrics = {
        "epoch": 0,
        "train_loss": None,
        "validation": initial_validation,
    }
    history.append(initial_metrics)
    (output_dir / "metrics.jsonl").write_text(json.dumps(initial_metrics) + "\n")
    save_checkpoint(
        output_dir / "initial.pt", model, optimizer, config, 0, initial_metrics
    )
    save_checkpoint(
        output_dir / "best.pt", model, optimizer, config, 0, initial_metrics
    )
    print(json.dumps(initial_metrics, sort_keys=True))
    best_validation_loss = initial_validation["loss"]
    for epoch in range(1, config.epochs + 1):
        seed_everything(config.seed + epoch)
        train_sampler.set_epoch(epoch)
        validation_sampler.set_epoch(epoch)
        train_loss = train_epoch(model, train_loader, optimizer, device, config)
        seed_everything(config.seed + 100_000 + epoch)
        validation = evaluate_loader(
            model, validation_loader, device, config.temperature, config.loss_mode
        )
        metrics = {"epoch": epoch, "train_loss": train_loss, "validation": validation}
        history.append(metrics)
        with (output_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        save_checkpoint(
            output_dir / "last.pt", model, optimizer, config, epoch, metrics
        )
        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, config, epoch, metrics
            )
        print(json.dumps(metrics, sort_keys=True))

    summary = {
        "output_dir": str(output_dir.resolve()),
        "device": str(device),
        "best_validation_loss": best_validation_loss,
        "epochs": config.epochs,
        "validation_view": "two independent SO(3) rotations",
        "initial": history[0],
        "last": history[-1],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.smoke:
        config = ExperimentConfig.smoke(args.legacy_root)
    elif args.full_train_corpus:
        config = ExperimentConfig.full_train_corpus(args.legacy_root)
    elif args.config is not None:
        config = ExperimentConfig.from_dict(json.loads(args.config.read_text()))
        if args.legacy_root is not None:
            config.legacy_root = args.legacy_root
    else:
        config = ExperimentConfig()
        if args.legacy_root is not None:
            config.legacy_root = args.legacy_root
    if args.labels:
        config.labels = tuple(args.labels)
    if args.datasets:
        config.datasets = tuple(args.datasets)
    if args.rotation_probability is not None:
        config.rotation_probability = args.rotation_probability
    if args.loss_mode is not None:
        config.loss_mode = args.loss_mode
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.steps_per_epoch is not None:
        config.steps_per_epoch = args.steps_per_epoch
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    config.validate()
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--legacy-root", default=None)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--rotation-probability", type=float, default=None)
    parser.add_argument("--loss-mode", choices=("supervised", "instance"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full-train-corpus", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.smoke and args.full_train_corpus:
        parser.error("--smoke and --full-train-corpus are mutually exclusive")

    config = parse_config(args)
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or OUTPUT_ROOT / run_name
    summary = run_training(
        config, output_dir, resolve_device(args.device), overwrite=args.overwrite
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
