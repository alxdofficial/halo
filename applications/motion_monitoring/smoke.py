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
from applications.motion_monitoring.task1.matcher import best_full_timeline_match
from applications.motion_monitoring.task1.training import event_detection_metrics
from applications.motion_monitoring.task2 import (
    BoundedExecution,
    ChangeRuler,
    ExecutionEpisode,
    binary_auroc,
    binary_operating_metrics,
    collate_execution_episodes,
    from_motion_sequence as task2_execution,
    personal_change_report,
    train_step as task2_train_step,
)
from applications.motion_monitoring.task3 import (
    RecurrentMotionMetric,
    assign_event_targets,
    collate_motion_sequences,
    direct_cosine_affinity,
    event_batch_from_recordings,
    initialize_affinity_threshold,
    pool_multiscale_candidates,
    scoped_pair_indices,
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


def _midpoint_operating_metrics(
    positive_scores: torch.Tensor, negative_scores: torch.Tensor
) -> tuple[float, dict[str, float]]:
    """Calibrate a diagnostic threshold halfway between class medians."""

    threshold = float(
        0.5 * (positive_scores.detach().median() + negative_scores.detach().median())
    )
    scores = torch.cat((positive_scores, negative_scores))
    targets = torch.cat(
        (
            torch.ones_like(positive_scores, dtype=torch.bool),
            torch.zeros_like(negative_scores, dtype=torch.bool),
        )
    )
    return threshold, binary_operating_metrics(scores, targets, threshold=threshold)


def _different_recording_pair(dataset: str, annotation_kind: str):
    return find_event_pair(
        open_cache(dataset),
        annotation_kind=annotation_kind,
        same_label=True,
        different_recordings=True,
    )


def _independent_examples(
    dataset: str,
    annotation_kind: str,
    *,
    label: str,
    count: int,
    same_subject: bool = False,
):
    """Select deterministic same-label examples from distinct recordings."""

    selected = []
    recording_ids: set[str] = set()
    subject_id: str | None = None
    for example in iter_events(
        open_cache(dataset),
        annotation_kind=annotation_kind,
        include_labels={label},
    ):
        recording_id = example.recording.recording_id
        if recording_id in recording_ids:
            continue
        if same_subject and subject_id is not None and example.recording.subject_id != subject_id:
            continue
        selected.append(example)
        recording_ids.add(recording_id)
        subject_id = subject_id or example.recording.subject_id
        if len(selected) == count:
            return tuple(selected)
    raise LookupError(
        f"{dataset} has fewer than {count} independent {annotation_kind} examples for {label!r}"
    )


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
    reference_example, query_example = _independent_examples(
        "openpack", "operation", label="Assemble Box", count=2, same_subject=True
    )
    reference_recording = crop_event(reference_example)
    query_recording = crop_query_around_event(query_example, duration_sec=60.0)

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
    reference = first_batch.reference[0, first_batch.reference_valid[0]].detach().cpu().numpy()
    query = first_batch.query[0, first_batch.query_valid[0]].detach().cpu().numpy()
    intervals = first_batch.query_intervals_sec[
        0, first_batch.query_valid[0]
    ].detach().cpu().numpy()
    direct_match = best_full_timeline_match(reference, query, intervals)
    direct_metrics = event_detection_metrics(
        [direct_match],
        first_batch.targets_sec[0, first_batch.target_valid[0]].detach().cpu(),
        query_duration_sec=float(intervals[-1, 1] - intervals[0, 0]),
        score_threshold=float("inf"),
    )
    frozen_batch = None if train_encoder else first_batch
    model = DifferentiableSubsequenceMatcher(first_batch.reference.shape[-1]).to(
        first_batch.reference.device
    )
    parameters = list(model.parameters())
    if train_encoder:
        parameters += [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-3)
    rows = []
    for _ in range(steps):
        result = task1_train_step(model, frozen_batch or build_batch(), optimizer)
        encoder_norm, encoder_finite = _encoder_grad_norm(encoder)
        rows.append(
            {
                "loss": result.loss,
                **result.telemetry,
                "encoder_grad_norm": encoder_norm,
                "encoder_gradient_finite": encoder_finite,
            }
        )
    learned_matches = model.detect(
        first_batch.reference[0, first_batch.reference_valid[0]],
        first_batch.query[0],
        first_batch.query_intervals_sec[0].detach().cpu().numpy(),
        score_threshold=float("inf"),
        query_valid=first_batch.query_valid[0] & first_batch.loss_valid[0],
        max_detections=1,
    )
    learned_event_metrics = event_detection_metrics(
        learned_matches,
        first_batch.targets_sec[0, first_batch.target_valid[0]].detach().cpu(),
        query_duration_sec=float(intervals[-1, 1] - intervals[0, 0]),
        score_threshold=float("inf"),
    )
    return {
        "dataset": "openpack",
        "label": reference_example.event.label,
        "steps": steps,
        "query_patches": int(first_batch.query_valid.sum()),
        "reference_patches": int(first_batch.reference_valid.sum()),
        "target_count": int(first_batch.target_valid.sum()),
        "frozen_direct": {
            "method": "constrained_subsequence_dtw_best_match",
            "match_score": direct_match.score,
            **direct_metrics,
        },
        "frozen_learned": {
            "method": "projected_soft_dtw_endpoint_head",
            "hard_dtw_event_readout": learned_event_metrics,
            "first": rows[0],
            "last": rows[-1],
        },
        "first": rows[0],
        "last": rows[-1],
    }


def _task2_smoke(
    encoder: torch.nn.Module,
    *,
    steps: int,
    train_encoder: bool,
) -> dict[str, object]:
    # A mechanical smoke needs a real encoder input but does not define a Task-2
    # cohort. Use independently recorded same-label OpenPack actions here. The
    # manifest-bound HARMES/CrossFit protocol owns reportable fitting.
    examples = _independent_examples(
        "openpack", "operation", label="Assemble Box", count=5, same_subject=True
    )
    reference_examples = examples[:4]
    comparison_example = examples[4]
    reference_recordings = tuple(crop_event(item) for item in reference_examples)
    comparison_recording = crop_event(comparison_example)

    def build_batch():
        reference_sequences = tuple(
            encoder.encode_recording(recording) for recording in reference_recordings
        )
        comparison_sequence = encoder.encode_recording(comparison_recording)
        references = tuple(
            task2_execution(
                sequence,
                execution_id=example.execution_id,
                task_id=example.event.label,
            )
            for sequence, example in zip(
                reference_sequences, reference_examples, strict=True
            )
        )
        comparison = task2_execution(
            comparison_sequence,
            execution_id=comparison_example.execution_id,
            task_id=comparison_example.event.label,
        )
        accepted = ExecutionEpisode(
            accepted_references=references,
            query=comparison,
            episode_kind="accepted_query",
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
            sensor_config=comparison.sensor_config,
        )
        changed_episode = ExecutionEpisode(
            accepted_references=references,
            query=changed,
            episode_kind="modified_query",
            severity=1.0,
            modification_kind="latent_smoke_only",
        )
        return collate_execution_episodes([accepted, changed_episode]).to(
            next(encoder.parameters()).device
        )

    first_batch = build_batch()
    direct = personal_change_report(first_batch, None)
    targets = torch.tensor([False, True], device=direct.joint_deviation.device)
    direct_auroc = binary_auroc(
        direct.joint_deviation,
        targets,
    )
    frozen_batch = None if train_encoder else first_batch
    model = ChangeRuler(
        embedding_dim=first_batch.reference_embeddings.shape[-1],
        phase_bins=8,
    ).to(first_batch.reference_embeddings.device)
    parameters = list(model.parameters())
    if train_encoder:
        parameters += [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-3)
    rows = []
    for _ in range(steps):
        result = task2_train_step(model, frozen_batch or build_batch(), optimizer)
        encoder_norm, encoder_finite = _encoder_grad_norm(encoder)
        rows.append(
            {
                **asdict(result),
                "encoder_grad_norm": encoder_norm,
                "encoder_gradient_finite": encoder_finite,
            }
        )
    model.eval()
    learned = personal_change_report(first_batch, model)
    learned_auroc = binary_auroc(learned.joint_deviation, targets)
    return {
        "dataset": "openpack",
        "task": comparison_example.event.label,
        "steps": steps,
        "frozen_direct": {
            "method": "phase_cosine_personal_robust_statistics",
            "auroc": direct_auroc,
            "accepted_score": float(direct.joint_deviation[0]),
            "changed_score": float(direct.joint_deviation[1]),
            "accepted_above_personal_limit": bool(
                direct.exceeds_personal_limit[0]
            ),
            "changed_above_personal_limit": bool(
                direct.exceeds_personal_limit[1]
            ),
            "reference_limited_fraction": float(direct.reference_limited.float().mean()),
        },
        "frozen_learned": {
            "method": "set_conditioned_change_ruler",
            "auroc": learned_auroc,
            "accepted_score": float(learned.joint_deviation[0]),
            "changed_score": float(learned.joint_deviation[1]),
            "accepted_above_personal_limit": bool(
                learned.exceeds_personal_limit[0]
            ),
            "changed_above_personal_limit": bool(
                learned.exceeds_personal_limit[1]
            ),
            "first": rows[0],
            "last": rows[-1],
        },
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
    direct = direct_cosine_affinity(initial_candidates, initial_targets)
    frozen_candidates = None if train_encoder else (initial_candidates, initial_targets)
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
        if frozen_candidates is None:
            candidates, targets, _ = build_candidates()
        else:
            candidates, targets = frozen_candidates
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
    positive_pairs, negative_pairs = scoped_pair_indices(
        initial_candidates, initial_targets
    )
    flat_candidates = initial_candidates.embeddings.reshape(
        -1, initial_candidates.embeddings.shape[-1]
    )
    with torch.no_grad():
        learned_positive = model.pair_logits(
            flat_candidates[positive_pairs[:, 0]],
            flat_candidates[positive_pairs[:, 1]],
        )
        learned_negative = model.pair_logits(
            flat_candidates[negative_pairs[:, 0]],
            flat_candidates[negative_pairs[:, 1]],
        )
    direct_threshold, direct_operating = _midpoint_operating_metrics(
        direct.positive_scores, direct.negative_scores
    )
    learned_operating = binary_operating_metrics(
        torch.cat((learned_positive, learned_negative)),
        torch.cat(
            (
                torch.ones_like(learned_positive, dtype=torch.bool),
                torch.zeros_like(learned_negative, dtype=torch.bool),
            )
        ),
        threshold=0.0,
    )
    return {
        "dataset": "openpack",
        "annotation_kind": "fine_action",
        "steps": steps,
        "event_count": event_count,
        "candidate_count": initial_candidates.valid_count,
        "frozen_direct": {
            "method": "raw_cosine_candidate_affinity",
            "pair_auroc": direct.auroc,
            "pair_auprc": direct.auprc,
            "diagnostic_threshold": direct_threshold,
            **direct_operating,
            "positive_score": float(direct.positive_scores.mean()),
            "negative_score": float(direct.negative_scores.mean()),
        },
        "frozen_learned": {
            "method": "projected_cosine_affinity_head",
            "diagnostic_threshold": 0.0,
            **learned_operating,
            "first": rows[0],
            "last": rows[-1],
        },
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
