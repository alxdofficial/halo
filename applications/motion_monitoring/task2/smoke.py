"""Short synthetic smoke for the Task-2 personal-normative ruler.

Builds compatibility-clean batches from synthetic per-subject trajectories, runs
a handful of optimizer steps, and reports whether the ruler separates accepted
repeats from declared modifications better than the untrained cosine floor. It
exercises the code path end to end; it is not evidence about real data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from applications.motion_monitoring.data.compatibility import sensor_compatibility_key
from .contracts import BoundedExecution, collate_execution_episodes
from .episodes import ExecutionRecord, Task2BatchBuilder, relation_summary
from .losses import RulerLossConfig
from .metrics import paired_within_series_auroc
from .modifications import MODIFICATIONS, NUISANCES, apply_modification, apply_nuisance
from .model import ChangeRuler
from .scoring import personal_change_report
from .training import train_step


_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
_RATE_HZ = 50.0


def _key(channels=_CHANNELS):
    return sensor_compatibility_key(
        device="smartwatch",
        placement="wrist",
        channels=tuple(channels),
        gravity_state="present",
    )


_KEY = _key()


@dataclass(frozen=True)
class SmokeResult:
    steps: int
    episodes: int
    first_loss: float
    last_loss: float
    first_separation: float | None
    last_separation: float | None
    floor_auroc: float
    ruler_auroc: float
    nonfinite_gradients: int
    reference_attention_fraction: float
    relation_mix: dict[str, Any]


def _signal(length: int, channels: int, rng: np.random.Generator, *, offset: np.ndarray) -> np.ndarray:
    time = np.linspace(0.0, 1.0, length)
    columns = [np.sin(2 * np.pi * (1 + index % 4) * time) for index in range(channels)]
    values = np.stack(columns, axis=-1) + offset
    return values + 0.02 * rng.normal(size=values.shape)


def _embed(
    values: np.ndarray,
    *,
    dataset: str,
    subject: str,
    session: str,
    execution: str,
    task: str,
    channels=_CHANNELS,
) -> BoundedExecution:
    """Stand in for a frozen encoder: one patch per 0.5 s of physical time.

    Released encoders pad a reduced channel set into the canonical six-channel
    layout and carry a validity mask, so the embedding width does not change when
    the gyroscope is dropped; only the declared configuration does. The stand-in
    reproduces that, otherwise a channel view would silently resize the ruler.
    """

    array = np.asarray(values, dtype=np.float32)
    if tuple(channels) != _CHANNELS:
        padded = np.zeros((array.shape[0], len(_CHANNELS)), dtype=np.float32)
        for column, name in enumerate(channels):
            padded[:, _CHANNELS.index(name)] = array[:, column]
        array = padded
    per_patch = max(2, int(0.5 * _RATE_HZ))
    count = max(2, array.shape[0] // per_patch)
    patches = np.stack(
        [array[index * per_patch : (index + 1) * per_patch].mean(axis=0) for index in range(count)]
    )
    edges = np.arange(count + 1, dtype=np.float64) * (per_patch / _RATE_HZ)
    return BoundedExecution(
        embeddings=torch.from_numpy(patches).float(),
        patch_intervals_sec=torch.from_numpy(np.stack((edges[:-1], edges[1:]), axis=-1)).float(),
        patch_mask=torch.ones(count, dtype=torch.bool),
        dataset=dataset,
        subject_id=subject,
        session_id=session,
        execution_id=execution,
        task_id=task,
        sensor_config=_key(channels),
    )


def make_records(
    *, subjects: int = 6, executions: int = 6, channels: int = 6, seed: int = 11
) -> tuple[list[ExecutionRecord], dict[str, np.ndarray]]:
    """Synthetic executions plus the raw signals the batch builder transforms."""

    rng = np.random.default_rng(seed)
    records: list[ExecutionRecord] = []
    raw: dict[str, np.ndarray] = {}
    for subject_index in range(subjects):
        subject = f"subject_{subject_index:02d}"
        offset = 0.05 * rng.normal(size=channels)
        for execution_index in range(executions):
            day = f"day_{execution_index % 3}"
            session = f"{subject}_{day}"
            execution_id = f"{subject}_exec_{execution_index}"
            values = _signal(120 + 10 * (execution_index % 3), channels, rng, offset=offset)
            raw[execution_id] = values
            records.append(
                ExecutionRecord(
                    execution=_embed(
                        values,
                        dataset="synthetic_task2",
                        subject=subject,
                        session=session,
                        execution=execution_id,
                        task="repeated_reach",
                    ),
                    key=_KEY,
                    day=day,
                )
            )
    return records, raw


def make_pool(*, subjects: int = 6, seed: int = 11, modified_per_execution: int = 2):
    """Clean records plus pre-materialised variants, mirroring the derived corpus.

    Real runs read variants from ``task2_modified_v1``; the smoke makes the same
    shapes in memory so the batch builder is exercised on the selection path it
    actually uses.
    """

    records, raw = make_records(subjects=subjects, seed=seed)
    pool = list(records)
    kinds = sorted(MODIFICATIONS)
    nuisance_kinds = sorted(NUISANCES)
    rng = np.random.default_rng(seed)
    for record in records:
        values = raw[record.root_id]
        for index in range(modified_per_execution):
            kind = kinds[int(rng.integers(len(kinds)))]
            severity = float(0.3 + 0.7 * rng.random())
            modified = apply_modification(
                values,
                kind=kind,
                severity=severity,
                seed=int(rng.integers(2**31)),
                sampling_rate_hz=_RATE_HZ,
                channels=_CHANNELS,
            )
            pool.append(
                replace(
                    record,
                    execution=_variant_execution(record, modified, f"modified_{index}"),
                    variant="modified",
                    modification_kind=kind,
                    severity=severity,
                    origin_execution_id=record.root_id,
                )
            )
        nuisance_kind = nuisance_kinds[int(rng.integers(len(nuisance_kinds)))]
        nuisanced = apply_nuisance(
            values,
            kind=nuisance_kind,
            seed=int(rng.integers(2**31)),
            sampling_rate_hz=_RATE_HZ,
            channels=_CHANNELS,
        )
        pool.append(
            replace(
                record,
                execution=_variant_execution(record, nuisanced, "nuisance"),
                variant="nuisance",
                nuisance_kind=nuisance_kind,
                origin_execution_id=record.root_id,
            )
        )
    return pool


def _variant_execution(record: ExecutionRecord, values: np.ndarray, suffix: str) -> BoundedExecution:
    return _embed(
        values,
        dataset=record.execution.dataset,
        subject=record.execution.subject_id,
        session=record.execution.session_id,
        execution=f"{record.execution.execution_id}:{suffix}",
        task=record.execution.task_id,
    )


def build_demo_batch(*, subjects: int = 4, seed: int = 11, groups: int = 2):
    """One compatibility-clean batch of synthetic episodes, for tests and demos."""

    builder = Task2BatchBuilder(
        make_pool(subjects=subjects, seed=seed),
        reference_count=4,
        positives_per_episode=2,
        seed=seed,
    )
    episodes, _ = builder.build_batch(groups=groups, seed=seed)
    return episodes, collate_execution_episodes(episodes)


def run_synthetic_smoke(
    *,
    steps: int = 12,
    subjects: int = 6,
    seed: int = 11,
    device: str = "cpu",
) -> SmokeResult:
    if steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    pool = make_pool(subjects=subjects, seed=seed)
    builder = Task2BatchBuilder(pool, reference_count=4, positives_per_episode=2, seed=seed)
    model = ChangeRuler(pool[0].execution.embeddings.shape[1], phase_bins=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    telemetry = []
    plans = []
    episodes_seen = 0
    for step in range(steps):
        episodes, batch_plans = builder.build_batch(groups=2, seed=seed + step)
        batch = collate_execution_episodes(episodes).to(device)
        episodes_seen += len(episodes)
        plans.extend(batch_plans)
        telemetry.append(train_step(model, batch, optimizer, loss_config=RulerLossConfig()))

    episodes, _ = builder.build_batch(groups=3, seed=seed + 999)
    batch = collate_execution_episodes(episodes).to(device)
    model.eval()
    # Both arms travel the same deployment path: phase residual, personal
    # envelope, per-person reference limit. Only the residual space differs.
    ruler_distances = personal_change_report(batch, model).joint_deviation
    floor = personal_change_report(batch, None).joint_deviation

    def by_subject(values: torch.Tensor):
        accepted: dict[str, list[float]] = {}
        changed: dict[str, list[float]] = {}
        for index, episode in enumerate(episodes):
            subject = episode.accepted_references[0].subject_id
            target = accepted if episode.is_positive else changed if episode.is_negative else None
            if target is not None:
                target.setdefault(subject, []).append(float(values[index]))
        return accepted, changed

    floor_auroc = paired_within_series_auroc(*by_subject(floor))["mean_auroc"]
    ruler_auroc = paired_within_series_auroc(*by_subject(ruler_distances))["mean_auroc"]
    return SmokeResult(
        steps=steps,
        episodes=episodes_seen,
        first_loss=telemetry[0].total_loss,
        last_loss=telemetry[-1].total_loss,
        first_separation=telemetry[0].separation,
        last_separation=telemetry[-1].separation,
        floor_auroc=floor_auroc,
        ruler_auroc=ruler_auroc,
        nonfinite_gradients=sum(item.nonfinite_gradients for item in telemetry),
        reference_attention_fraction=telemetry[-1].reference_attention_fraction,
        relation_mix=relation_summary(plans),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--subjects", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_synthetic_smoke(
        steps=args.steps, subjects=args.subjects, seed=args.seed, device=args.device
    )
    payload: dict[str, Any] = asdict(result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
