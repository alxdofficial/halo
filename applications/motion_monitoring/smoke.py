"""Short real-cache optimizer smoke tests for all three application tasks.

This command verifies data loading, common encoder wiring, finite losses, and
gradient flow. It is deliberately not a training recipe or a reported result.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from applications.motion_monitoring.data.examples import (
    crop_event,
    crop_query_around_event,
    crop_recording,
    find_event_pair,
    iter_events,
    open_cache,
)
from applications.motion_monitoring.sequence import (
    HaloMotionEncoder,
    MotionEncoder,
    PhysicalProjectionEncoder,
)
from applications.motion_monitoring.task1 import (
    DifferentiableSubsequenceMatcher,
    collate_detection_episodes,
    episode_from_recordings,
    from_motion_sequence as task1_sequence,
    train_step as task1_train_step,
)
from applications.motion_monitoring.task2 import (
    BoundedExecution,
    ChangeMetricHead,
    ExecutionPair,
    collate_execution_pairs,
    from_motion_sequence as task2_execution,
    initialize_change_threshold,
    train_step as task2_train_step,
)
from applications.motion_monitoring.task2.smoke import SYNTHETIC_TARGETS
from applications.motion_monitoring.task3 import (
    RecurrentMotionMetric,
    assign_event_targets,
    collate_motion_sequences,
    event_batch_from_recordings,
    initialize_affinity_threshold,
    pool_multiscale_candidates,
)
from applications.motion_monitoring.task3.training import train_step as task3_train_step


def _encoder_grad_norm(encoder: torch.nn.Module) -> tuple[float, float]:
    trainable = [
        parameter for parameter in encoder.parameters() if parameter.requires_grad
    ]
    if not trainable:
        return 0.0, 1.0
    gradients = [
        parameter.grad.detach().float()
        for parameter in trainable
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0, 0.0
    norm = torch.stack([gradient.square().sum() for gradient in gradients]).sum().sqrt()
    finite = torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all()
    return float(norm), float(finite)


def _different_recording_pair(dataset: str, annotation_kind: str):
    return find_event_pair(
        open_cache(dataset),
        annotation_kind=annotation_kind,
        same_label=True,
        different_recordings=True,
    )


def _same_subject_pair(dataset: str, annotation_kind: str):
    first_by_key = {}
    for example in iter_events(open_cache(dataset), annotation_kind=annotation_kind):
        key = (example.recording.subject_id, example.event.label)
        prior = first_by_key.get(key)
        if prior is not None and prior.execution_id != example.execution_id:
            return prior, example
        first_by_key[key] = example
    raise LookupError(f"{dataset} has no within-subject repeated {annotation_kind}")


def _build_encoder(args: argparse.Namespace) -> MotionEncoder:
    if args.encoder == "physical":
        return PhysicalProjectionEncoder(args.embedding_dim).to(args.device)
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --encoder halo")
    return HaloMotionEncoder.from_checkpoint(
        args.checkpoint,
        device=args.device,
        trainable=args.train_encoder,
    )


def _task1_smoke(
    encoder: torch.nn.Module,
    *,
    steps: int,
    train_encoder: bool,
) -> dict[str, object]:
    reference_example, query_example = _different_recording_pair("openpack", "fine_action")
    reference_recording = crop_event(reference_example)
    query_recording = crop_query_around_event(query_example, duration_sec=30.0)

    def build_batch():
        reference_sequence = encoder.encode_recording(reference_recording)
        query_sequence = encoder.encode_recording(query_recording)
        event_index = next(
            index
            for index, event in enumerate(reference_recording.events)
            if event.label == reference_example.event.label
            and event.annotation_kind == reference_example.event.annotation_kind
        )
        episode = episode_from_recordings(
            reference_recording,
            query_recording,
            task1_sequence(reference_sequence),
            task1_sequence(query_sequence),
            label=reference_example.event.label,
            reference_event_index=event_index,
        )
        return collate_detection_episodes([episode]).to(
            next(encoder.parameters()).device
        )

    first_batch = build_batch()
    model = DifferentiableSubsequenceMatcher(first_batch.reference.shape[-1]).to(
        first_batch.reference.device
    )
    parameters = list(model.parameters())
    if train_encoder:
        parameters += [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-3)
    rows = []
    for _ in range(steps):
        result = task1_train_step(model, build_batch(), optimizer)
        encoder_norm, encoder_finite = _encoder_grad_norm(encoder)
        rows.append(
            {
                "loss": result.loss,
                **result.telemetry,
                "encoder_grad_norm": encoder_norm,
                "encoder_gradient_finite": encoder_finite,
            }
        )
    return {
        "dataset": "openpack",
        "label": reference_example.event.label,
        "steps": steps,
        "query_patches": int(first_batch.query_valid.sum()),
        "reference_patches": int(first_batch.reference_valid.sum()),
        "target_count": int(first_batch.target_valid.sum()),
        "first": rows[0],
        "last": rows[-1],
    }


def _task2_smoke(
    encoder: torch.nn.Module,
    *,
    steps: int,
    train_encoder: bool,
) -> dict[str, object]:
    first, second = _same_subject_pair("crossfit", "repetition")
    reference_recording = crop_event(first)
    comparison_recording = crop_event(second)

    def build_batch():
        reference_sequence = encoder.encode_recording(reference_recording)
        comparison_sequence = encoder.encode_recording(comparison_recording)
        reference = task2_execution(
            reference_sequence, execution_id=first.execution_id, task_id=first.event.label
        )
        comparison = task2_execution(
            comparison_sequence,
            execution_id=second.execution_id,
            task_id=first.event.label,
        )
        accepted = ExecutionPair(
            reference=reference,
            comparison=comparison,
            pair_kind="accepted_variation",
            change_targets=torch.zeros(len(SYNTHETIC_TARGETS)),
            target_mask=torch.ones(len(SYNTHETIC_TARGETS), dtype=torch.bool),
            target_specs=SYNTHETIC_TARGETS,
        )
        changed_embeddings = comparison.embeddings.clone()
        phase = torch.linspace(
            0.0,
            torch.pi,
            len(changed_embeddings),
            device=changed_embeddings.device,
            dtype=changed_embeddings.dtype,
        )
        changed_embeddings[:, 0] = changed_embeddings[:, 0] + 0.75 * torch.sin(phase)
        changed = BoundedExecution(
            embeddings=F.normalize(changed_embeddings, dim=-1),
            patch_intervals_sec=comparison.patch_intervals_sec,
            patch_mask=comparison.patch_mask,
            dataset=comparison.dataset,
            subject_id=comparison.subject_id,
            session_id=comparison.session_id,
            execution_id=f"{comparison.execution_id}/synthetic-change",
            task_id=comparison.task_id,
        )
        known_change = ExecutionPair(
            reference=reference,
            comparison=changed,
            pair_kind="known_change",
            change_targets=torch.tensor([0.2, -0.2, 0.2, 0.1]),
            target_mask=torch.ones(len(SYNTHETIC_TARGETS), dtype=torch.bool),
            target_specs=SYNTHETIC_TARGETS,
        )
        return collate_execution_pairs([accepted, known_change]).to(
            next(encoder.parameters()).device
        )

    first_batch = build_batch()
    model = ChangeMetricHead(
        embedding_dim=first_batch.reference_embeddings.shape[-1],
        target_dim=len(SYNTHETIC_TARGETS),
        phase_bins=8,
    ).to(first_batch.reference_embeddings.device)
    initialize_change_threshold(model, first_batch)
    parameters = list(model.parameters())
    if train_encoder:
        parameters += [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-3)
    rows = []
    for _ in range(steps):
        result = task2_train_step(model, build_batch(), optimizer)
        encoder_norm, encoder_finite = _encoder_grad_norm(encoder)
        rows.append(
            {
                **asdict(result),
                "encoder_grad_norm": encoder_norm,
                "encoder_gradient_finite": encoder_finite,
            }
        )
    return {
        "dataset": "crossfit",
        "task": first.event.label,
        "steps": steps,
        "first": rows[0],
        "last": rows[-1],
    }


def _task3_smoke(
    encoder: torch.nn.Module,
    *,
    steps: int,
    train_encoder: bool,
) -> dict[str, object]:
    source = open_cache("openpack")[0]
    stream = source.streams[0]
    recording = crop_recording(
        source,
        float(stream.timestamps_sec[0]),
        float(stream.timestamps_sec[0]) + 300.0,
        recording_suffix="task3-smoke",
    )

    def build_candidates():
        sequence = encoder.encode_recording(recording)
        timeline = collate_motion_sequences([sequence])
        events = event_batch_from_recordings(
            [recording],
            [sequence],
            annotation_kind="fine_action",
            exhaustive=False,
        ).to(timeline.embeddings.device)
        candidates = pool_multiscale_candidates(
            timeline.embeddings,
            timeline.intervals_sec,
            timeline.valid,
            durations_sec=(2.0, 4.0, 8.0),
            candidate_stride_sec=1.0,
        )
        targets = assign_event_targets(candidates, events, positive_iou=0.5)
        return candidates, targets, int(events.event_mask.sum())

    initial_candidates, initial_targets, event_count = build_candidates()
    model = RecurrentMotionMetric(
        initial_candidates.embeddings.shape[-1],
        projection_dim=min(32, initial_candidates.embeddings.shape[-1]),
    ).to(initial_candidates.embeddings.device)
    initialize_affinity_threshold(model, initial_candidates, initial_targets)
    parameters = list(model.parameters())
    if train_encoder:
        parameters += [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-3)
    rows = []
    for _ in range(steps):
        candidates, targets, _ = build_candidates()
        result = task3_train_step(model, optimizer, candidates, targets)
        encoder_norm, encoder_finite = _encoder_grad_norm(encoder)
        rows.append(
            {
                "loss": result.loss,
                **result.telemetry,
                "encoder_grad_norm": encoder_norm,
                "encoder_gradient_finite": encoder_finite,
            }
        )
    return {
        "dataset": "openpack",
        "annotation_kind": "fine_action",
        "steps": steps,
        "event_count": event_count,
        "candidate_count": initial_candidates.valid_count,
        "first": rows[0],
        "last": rows[-1],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    encoder = _build_encoder(args)
    train_encoder = args.train_encoder or args.encoder == "physical"
    return {
        "status": "mechanical_smoke_only",
        "encoder": args.encoder,
        "train_encoder": train_encoder,
        "device": str(args.device),
        "task1": _task1_smoke(encoder, steps=args.steps, train_encoder=train_encoder),
        "task2": _task2_smoke(encoder, steps=args.steps, train_encoder=train_encoder),
        "task3": _task3_smoke(encoder, steps=args.steps, train_encoder=train_encoder),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encoder", choices=("physical", "halo"), default="physical")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train-encoder", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
