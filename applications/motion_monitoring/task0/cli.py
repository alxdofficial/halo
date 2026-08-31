"""Fit and run the Task-0 physical event-proposal detector."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from applications.motion_monitoring.data.adapters.registry import ADAPTERS
from applications.motion_monitoring.data.build_cache import DEFAULT_SOURCE_ROOT
from applications.motion_monitoring.data.cache import CachedRecordingDataset
from applications.motion_monitoring.data.contracts import EventInterval
from applications.motion_monitoring.task0.contracts import (
    EvidenceConfig,
    ProposalConfig,
    RefinementConfig,
)
from applications.motion_monitoring.task0.calibration import (
    CalibrationCase,
    calibrate_thresholds,
)
from applications.motion_monitoring.task0.detector import Task0Detector
from applications.motion_monitoring.task0.evidence import extract_physical_evidence
from applications.motion_monitoring.task0.metrics import evaluate_intervals
from applications.motion_monitoring.task0.plotting import plot_task0_timeline
from applications.motion_monitoring.task0.policies import validate_policy


def _cache_path(dataset: str) -> Path:
    return DEFAULT_SOURCE_ROOT / dataset / "processed" / "canonical_v1"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _recordings(dataset: str, limit: int | None) -> Sequence:
    cached = CachedRecordingDataset(_cache_path(dataset))
    return cached[:limit] if limit is not None else cached


def _fit(args: argparse.Namespace) -> None:
    evaluation_sources = [
        dataset
        for dataset in args.datasets
        if ADAPTERS[dataset].default_role == "evaluation"
    ]
    if evaluation_sources and not args.allow_evaluation_fit:
        raise ValueError(
            "refusing to fit Task-0 preprocessing on evaluation sources: "
            + ", ".join(evaluation_sources)
        )

    identities: list[str] = []
    sequence_count = 0

    def sequences() -> Iterator:
        nonlocal sequence_count
        for dataset in args.datasets:
            for recording in _recordings(dataset, args.limit_per_dataset):
                requested = set(args.stream_id or ())
                selected_streams = [
                    stream
                    for stream in recording.streams
                    if not requested or stream.stream_id in requested
                ]
                if not selected_streams:
                    continue
                identities.append(f"{recording.dataset}:{recording.recording_id}")
                for stream in selected_streams:
                    sequence_count += 1
                    yield extract_physical_evidence(recording, stream, evidence_config)

    evidence_config = EvidenceConfig(
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        gravity_cutoff_hz=args.gravity_cutoff_hz,
    )
    proposal_config = ProposalConfig(
        start_threshold=args.start_threshold,
        continue_threshold=args.continue_threshold,
        minimum_duration_seconds=args.minimum_duration_seconds,
        merge_gap_seconds=args.merge_gap_seconds,
        merge_floor=args.merge_floor,
    )
    refinement_config = RefinementConfig(
        enabled=not args.no_refinement,
        penalty=args.pelt_penalty,
        minimum_segment_seconds=args.pelt_minimum_segment_seconds,
        jump_seconds=args.pelt_jump_seconds,
        boundary_margin_seconds=args.boundary_margin_seconds,
    )
    detector = Task0Detector.fit(
        sequences(),
        evidence_config=evidence_config,
        proposal_config=proposal_config,
        refinement_config=refinement_config,
    )
    digest = sha256("\n".join(sorted(identities)).encode("utf-8")).hexdigest()
    detector.fit_provenance.update(
        {
            "datasets": list(args.datasets),
            "recording_count": len(identities),
            "stream_sequence_count": sequence_count,
            "recording_identity_sha256": digest,
            "evaluation_source_override": bool(evaluation_sources),
        }
    )
    detector.save(args.output)
    print(json.dumps({"output": str(args.output), **detector.fit_provenance}, indent=2))


def _detect(args: argparse.Namespace) -> None:
    detector = Task0Detector.load(args.model)
    detector_sha256 = _file_sha256(args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    proposal_count = 0
    recording_count = 0
    with output.open("w", encoding="utf-8") as destination:
        for recording in _recordings(args.dataset, args.limit):
            recording_count += 1
            proposals = detector.detect_recording(
                recording,
                stream_ids=(args.stream_id,) if args.stream_id else None,
            )
            for proposal in proposals:
                proposal = replace(
                    proposal,
                    metadata={
                        **proposal.metadata,
                        "detector_sha256": detector_sha256,
                    },
                )
                destination.write(json.dumps(proposal.to_dict(), sort_keys=True) + "\n")
                proposal_count += 1
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "recordings": recording_count,
                "proposals": proposal_count,
                "output": str(output),
                "detector_sha256": detector_sha256,
            },
            indent=2,
        )
    )


def _calibrate(args: argparse.Namespace) -> None:
    if (
        ADAPTERS[args.dataset].default_role == "evaluation"
        and not args.allow_evaluation_calibration
    ):
        raise ValueError(
            f"refusing to calibrate Task-0 thresholds on evaluation source {args.dataset!r}"
        )
    if not args.confirm_exhaustive_background:
        raise ValueError(
            "event-F1 calibration requires --confirm-exhaustive-background; "
            "otherwise unmatched proposals cannot be treated as false positives"
        )
    detector = Task0Detector.load(args.model)
    parent_detector_sha256 = _file_sha256(args.model)
    annotation_kinds = set(args.annotation_kind)
    excluded_labels = {label.casefold() for label in (args.exclude_label or ())}
    validate_policy(
        args.dataset,
        stream_id=args.stream_id,
        annotation_kinds=annotation_kinds,
        excluded_labels=excluded_labels,
        exhaustive_background=True,
        calibration=True,
        allow_exploratory=args.allow_exploratory_policy,
    )
    cases = []
    identities = []
    for recording in _recordings(args.dataset, args.limit):
        streams = [
            stream for stream in recording.streams if stream.stream_id == args.stream_id
        ]
        if len(streams) != 1:
            raise KeyError(
                f"{recording.recording_id}: expected one stream named {args.stream_id!r}, "
                f"found {len(streams)}"
            )
        events = tuple(
            event
            for event in recording.events
            if event.annotation_kind in annotation_kinds
            and event.label.casefold() not in excluded_labels
        )
        identities.append(recording.recording_id)
        cases.append(
            CalibrationCase(
                extract_physical_evidence(
                    recording, streams[0], detector.evidence_config
                ),
                events,
            )
        )
    selected, rows = calibrate_thresholds(
        cases,
        detector.scaler,
        detector.proposal_config,
        start_thresholds=args.start_thresholds,
        continue_thresholds=args.continue_thresholds,
        iou_threshold=args.iou_threshold,
        refinement_config=detector.refinement_config,
    )
    detector.proposal_config = selected
    detector.fit_provenance["threshold_calibration"] = {
        "dataset": args.dataset,
        "stream_id": args.stream_id,
        "annotation_kinds": sorted(annotation_kinds),
        "excluded_labels": sorted(excluded_labels),
        "recording_count": len(identities),
        "recording_identity_sha256": sha256(
            "\n".join(sorted(identities)).encode("utf-8")
        ).hexdigest(),
        "parent_detector_sha256": parent_detector_sha256,
        "iou_threshold": args.iou_threshold,
        "refinement_config": asdict(detector.refinement_config),
        "exploratory_policy_override": args.allow_exploratory_policy,
        "selected": asdict(selected),
        "grid": [asdict(row) for row in rows],
    }
    detector.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "recording_count": len(identities),
                "selected": asdict(selected),
                "best_event_f1": max(row.event_f1 for row in rows),
            },
            indent=2,
        )
    )


def _evaluate(args: argparse.Namespace) -> None:
    detector = Task0Detector.load(args.model)
    detector_sha256 = _file_sha256(args.model)
    included_labels = set(args.include_label or ())
    excluded_labels = {label.casefold() for label in (args.exclude_label or ())}
    annotation_kinds = set(args.annotation_kind)
    if (
        args.exhaustive_background
        and included_labels
        and not args.allow_exploratory_policy
    ):
        raise ValueError(
            "audited exhaustive evaluation is class-agnostic and cannot select only "
            "a subset of positive labels"
        )
    validate_policy(
        args.dataset,
        stream_id=args.stream_id,
        annotation_kinds=annotation_kinds,
        excluded_labels=excluded_labels,
        exhaustive_background=args.exhaustive_background,
        calibration=False,
        allow_exploratory=args.allow_exploratory_policy,
    )
    all_proposals = []
    all_events: list[EventInterval] = []
    per_recording = []
    elapsed_duration = 0.0

    for recording in _recordings(args.dataset, args.limit):
        streams = [
            stream for stream in recording.streams if stream.stream_id == args.stream_id
        ]
        if len(streams) != 1:
            raise KeyError(
                f"{recording.recording_id}: expected one stream named {args.stream_id!r}, "
                f"found {len(streams)}"
            )
        stream = streams[0]
        events = tuple(
            event
            for event in recording.events
            if event.annotation_kind in annotation_kinds
            and (not included_labels or event.label in included_labels)
            and event.label.casefold() not in excluded_labels
        )
        proposals = tuple(detector.detect_stream(recording, stream))
        median_dt = float(np.median(np.diff(stream.timestamps_sec)))
        recording_start = float(stream.timestamps_sec[0])
        duration = float(stream.timestamps_sec[-1] + median_dt - recording_start)
        metrics = evaluate_intervals(
            proposals,
            events,
            exhaustive_background=args.exhaustive_background,
            recording_start_sec=recording_start,
            recording_duration_sec=duration,
            iou_threshold=args.iou_threshold,
            boundary_tolerance_sec=args.boundary_tolerance_seconds,
        )
        per_recording.append(
            {"recording_id": recording.recording_id, "metrics": asdict(metrics)}
        )
        all_proposals.extend(
            replace(
                proposal,
                start_sec=elapsed_duration + proposal.start_sec - recording_start,
                end_sec=elapsed_duration + proposal.end_sec - recording_start,
            )
            for proposal in proposals
        )
        all_events.extend(
            replace(
                event,
                start_sec=elapsed_duration + event.start_sec - recording_start,
                end_sec=elapsed_duration + event.end_sec - recording_start,
            )
            for event in events
        )
        elapsed_duration += duration

    if elapsed_duration <= 0:
        raise ValueError("evaluation selected no recording duration")
    aggregate = evaluate_intervals(
        all_proposals,
        all_events,
        exhaustive_background=args.exhaustive_background,
        recording_start_sec=0.0,
        recording_duration_sec=elapsed_duration,
        iou_threshold=args.iou_threshold,
        boundary_tolerance_sec=args.boundary_tolerance_seconds,
    )
    report = {
        "schema_version": 1,
        "dataset": args.dataset,
        "stream_id": args.stream_id,
        "annotation_kinds": sorted(annotation_kinds),
        "included_labels": sorted(included_labels),
        "excluded_labels": sorted(excluded_labels),
        "exhaustive_background": args.exhaustive_background,
        "recording_count": len(per_recording),
        "recording_duration_sec": elapsed_duration,
        "detector_sha256": detector_sha256,
        "exploratory_policy_override": args.allow_exploratory_policy,
        "aggregate": asdict(aggregate),
        "per_recording": per_recording,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "per_recording"},
            indent=2,
        )
    )


def _plot(args: argparse.Namespace) -> None:
    detector = Task0Detector.load(args.model)
    recordings = _recordings(args.dataset, None)
    matches = [
        index
        for index, row in enumerate(recordings.rows)
        if row["recording_id"] == args.recording_id
    ]
    if len(matches) != 1:
        raise KeyError(
            f"expected one recording named {args.recording_id!r}, found {len(matches)}"
        )
    recording = recordings[matches[0]]
    streams = [
        stream for stream in recording.streams if stream.stream_id == args.stream_id
    ]
    if len(streams) != 1:
        raise KeyError(
            f"{recording.recording_id}: expected one stream named {args.stream_id!r}, "
            f"found {len(streams)}"
        )
    annotation_kinds = set(args.annotation_kind or ())
    excluded_labels = {label.casefold() for label in (args.exclude_label or ())}
    events = tuple(
        event
        for event in recording.events
        if (not annotation_kinds or event.annotation_kind in annotation_kinds)
        and event.label.casefold() not in excluded_labels
    )
    output = plot_task0_timeline(
        recording,
        streams[0],
        detector,
        events=events,
        output=args.output,
    )
    print(json.dumps({"output": str(output), "event_count": len(events)}, indent=2))


def _add_shared_fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window-seconds", type=float, default=0.5)
    parser.add_argument("--stride-seconds", type=float, default=0.1)
    parser.add_argument("--gravity-cutoff-hz", type=float, default=0.3)
    parser.add_argument("--start-threshold", type=float, default=3.0)
    parser.add_argument("--continue-threshold", type=float, default=1.5)
    parser.add_argument("--minimum-duration-seconds", type=float, default=0.5)
    parser.add_argument("--merge-gap-seconds", type=float, default=0.25)
    parser.add_argument("--merge-floor", type=float, default=0.75)
    parser.add_argument("--pelt-penalty", type=float, default=5.0)
    parser.add_argument("--pelt-minimum-segment-seconds", type=float, default=0.3)
    parser.add_argument("--pelt-jump-seconds", type=float, default=0.1)
    parser.add_argument("--boundary-margin-seconds", type=float, default=0.5)
    parser.add_argument("--no-refinement", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser(
        "fit", help="fit development-only robust scaling"
    )
    fit_parser.add_argument("datasets", nargs="+", choices=tuple(ADAPTERS))
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.add_argument("--limit-per-dataset", type=int)
    fit_parser.add_argument("--stream-id", action="append")
    fit_parser.add_argument("--allow-evaluation-fit", action="store_true")
    _add_shared_fit_arguments(fit_parser)
    fit_parser.set_defaults(function=_fit)

    detect_parser = subparsers.add_parser(
        "detect", help="write event proposals as JSONL"
    )
    detect_parser.add_argument("dataset", choices=tuple(ADAPTERS))
    detect_parser.add_argument("--model", type=Path, required=True)
    detect_parser.add_argument("--output", type=Path, required=True)
    detect_parser.add_argument("--limit", type=int)
    detect_parser.add_argument("--stream-id")
    detect_parser.set_defaults(function=_detect)

    calibrate_parser = subparsers.add_parser(
        "calibrate", help="select thresholds on exhaustive development annotations"
    )
    calibrate_parser.add_argument("dataset", choices=tuple(ADAPTERS))
    calibrate_parser.add_argument("--model", type=Path, required=True)
    calibrate_parser.add_argument("--output", type=Path, required=True)
    calibrate_parser.add_argument("--stream-id", required=True)
    calibrate_parser.add_argument("--annotation-kind", action="append", required=True)
    calibrate_parser.add_argument("--exclude-label", action="append")
    calibrate_parser.add_argument(
        "--start-thresholds", nargs="+", type=float, default=(2.0, 2.5, 3.0, 4.0)
    )
    calibrate_parser.add_argument(
        "--continue-thresholds", nargs="+", type=float, default=(0.5, 1.0, 1.5)
    )
    calibrate_parser.add_argument("--iou-threshold", type=float, default=0.5)
    calibrate_parser.add_argument("--limit", type=int)
    calibrate_parser.add_argument(
        "--confirm-exhaustive-background", action="store_true"
    )
    calibrate_parser.add_argument("--allow-evaluation-calibration", action="store_true")
    calibrate_parser.add_argument("--allow-exploratory-policy", action="store_true")
    calibrate_parser.set_defaults(function=_calibrate)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate one explicit stream and annotation level"
    )
    evaluate_parser.add_argument("dataset", choices=tuple(ADAPTERS))
    evaluate_parser.add_argument("--model", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--stream-id", required=True)
    evaluate_parser.add_argument("--annotation-kind", action="append", required=True)
    evaluate_parser.add_argument("--include-label", action="append")
    evaluate_parser.add_argument("--exclude-label", action="append")
    evaluate_parser.add_argument("--exhaustive-background", action="store_true")
    evaluate_parser.add_argument("--iou-threshold", type=float, default=0.5)
    evaluate_parser.add_argument(
        "--boundary-tolerance-seconds", type=float, default=0.5
    )
    evaluate_parser.add_argument("--limit", type=int)
    evaluate_parser.add_argument("--allow-exploratory-policy", action="store_true")
    evaluate_parser.set_defaults(function=_evaluate)

    plot_parser = subparsers.add_parser(
        "plot", help="render one proposal and reference timeline"
    )
    plot_parser.add_argument("dataset", choices=tuple(ADAPTERS))
    plot_parser.add_argument("--model", type=Path, required=True)
    plot_parser.add_argument("--recording-id", required=True)
    plot_parser.add_argument("--stream-id", required=True)
    plot_parser.add_argument("--annotation-kind", action="append")
    plot_parser.add_argument("--exclude-label", action="append")
    plot_parser.add_argument("--output", type=Path, required=True)
    plot_parser.set_defaults(function=_plot)

    args = parser.parse_args()
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if (
        getattr(args, "limit_per_dataset", None) is not None
        and args.limit_per_dataset <= 0
    ):
        parser.error("--limit-per-dataset must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
