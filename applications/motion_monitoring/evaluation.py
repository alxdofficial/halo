"""Strict result contract and comparison tables for application Tasks 1-3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


PRIMARY_METRICS = {
    "task1": (
        "event_precision",
        "event_recall",
        "event_f1",
        "false_alarms_per_hour",
        "mean_onset_error_sec",
        "mean_offset_error_sec",
        "mean_absolute_count_error",
    ),
    "task2": (
        "change_sensitivity",
        "change_specificity",
        "change_balanced_accuracy",
        "accepted_false_alarm_rate",
    ),
    "task3": (
        "occurrence_precision",
        "occurrence_recall",
        "bcubed_f1",
        "mean_fragments_per_true_motif",
        "false_occurrences_per_hour",
        "mean_absolute_count_error",
        "matched_mean_iou",
    ),
}


def fingerprint_protocol(protocol: Mapping[str, object]) -> str:
    """Hash a finite JSON protocol record with stable ordering."""

    if not protocol:
        raise ValueError("evaluation protocol must be non-empty")
    try:
        rendered = json.dumps(
            protocol, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError("evaluation protocol must be finite JSON data") from error
    return sha256(rendered.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ApplicationEvaluation:
    task: str
    encoder: str
    dataset: str
    split: str
    cohort_fingerprint: str
    protocol_fingerprint: str
    protocol: Mapping[str, object]
    metrics: Mapping[str, float]
    counts: Mapping[str, int]
    encoder_provenance: Mapping[str, object]
    status: str = "complete"

    def __post_init__(self) -> None:
        if self.task not in PRIMARY_METRICS:
            raise ValueError(f"unknown application task: {self.task!r}")
        if self.split not in {"development", "test"}:
            raise ValueError("evaluation split must be development or test")
        if self.status not in {"complete", "diagnostic"}:
            raise ValueError("result status must be complete or diagnostic")
        identities = (
            self.encoder,
            self.dataset,
            self.cohort_fingerprint,
            self.protocol_fingerprint,
        )
        if any(not value for value in identities):
            raise ValueError("evaluation identities must be non-empty")
        missing = set(PRIMARY_METRICS[self.task]) - set(self.metrics)
        if missing:
            raise ValueError(
                f"{self.task} result is missing primary metrics: {sorted(missing)}"
            )
        for name, value in self.metrics.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"metric {name!r} must be finite")
        if not self.counts or any(
            not isinstance(value, int) or value <= 0 for value in self.counts.values()
        ):
            raise ValueError("evaluation counts must be non-empty positive integers")
        if not self.encoder_provenance:
            raise ValueError("encoder provenance must be non-empty")
        if self.protocol_fingerprint != fingerprint_protocol(self.protocol):
            raise ValueError(
                "evaluation protocol fingerprint does not match its record"
            )

    @property
    def reportable(self) -> bool:
        return self.status == "complete" and self.split == "test"

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> "ApplicationEvaluation":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def validate_comparison(results: Sequence[ApplicationEvaluation]) -> None:
    if not results:
        raise ValueError("comparison requires at least one result")
    reference = results[0]
    dimensions = (
        "task",
        "dataset",
        "split",
        "cohort_fingerprint",
        "protocol_fingerprint",
        "status",
    )
    for result in results[1:]:
        mismatched = [
            name
            for name in dimensions
            if getattr(result, name) != getattr(reference, name)
        ]
        if mismatched:
            raise ValueError(
                "cannot compare results with different " + ", ".join(mismatched)
            )
    encoders = [result.encoder for result in results]
    if len(encoders) != len(set(encoders)):
        raise ValueError("comparison contains duplicate encoder rows")


def render_markdown(results: Sequence[ApplicationEvaluation]) -> str:
    validate_comparison(results)
    metrics = PRIMARY_METRICS[results[0].task]
    header = ["encoder", *metrics]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(metrics)) + " |",
    ]
    for result in sorted(results, key=lambda item: item.encoder.lower()):
        lines.append(
            "| "
            + " | ".join(
                [
                    result.encoder,
                    *[f"{float(result.metrics[name]):.4f}" for name in metrics],
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"
