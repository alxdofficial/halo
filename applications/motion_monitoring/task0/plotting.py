"""Diagnostic timeline visualization for Task-0 proposal review."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from applications.motion_monitoring.data.contracts import (
    EventInterval,
    RawRecording,
    SensorStream,
)
from applications.motion_monitoring.task0.contracts import MotionProposal
from applications.motion_monitoring.task0.detector import Task0Detector
from applications.motion_monitoring.task0.evidence import (
    extract_physical_evidence,
    standardized_motion_score,
)


def plot_task0_timeline(
    recording: RawRecording,
    stream: SensorStream,
    detector: Task0Detector,
    *,
    events: Sequence[EventInterval] = (),
    output: Path,
    max_raw_points: int = 100_000,
) -> Path:
    """Plot raw vector magnitude, motion evidence, proposals, and references."""

    if max_raw_points <= 0:
        raise ValueError("max_raw_points must be positive")
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Task-0 plotting requires matplotlib; install HALO with the task0 extra"
        ) from error

    evidence = extract_physical_evidence(recording, stream, detector.evidence_config)
    proposals = detector.detect_evidence(evidence)
    scores, score_valid = standardized_motion_score(evidence, detector.scaler)
    origin = float(stream.timestamps_sec[0])
    raw_stride = max(1, int(np.ceil(len(stream.timestamps_sec) / max_raw_points)))
    raw_indices = np.arange(0, len(stream.timestamps_sec), raw_stride)
    raw_time = stream.timestamps_sec[raw_indices] - origin

    figure, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
    for names, label, color in (
        (("acc_x", "acc_y", "acc_z"), "acceleration magnitude (g)", "#1f77b4"),
        (("gyro_x", "gyro_y", "gyro_z"), "angular speed (rad/s)", "#9467bd"),
    ):
        if not all(name in stream.channels for name in names):
            continue
        channel_indices = [stream.channels.index(name) for name in names]
        valid = stream.valid[raw_indices][:, channel_indices].all(axis=1)
        values = stream.values[raw_indices][:, channel_indices]
        magnitude = np.linalg.norm(values, axis=1)
        magnitude[~valid] = np.nan
        axes[0].plot(raw_time, magnitude, color=color, linewidth=0.7, label=label)

    evidence_time = evidence.centers_sec - origin
    axes[1].plot(
        evidence_time[score_valid],
        scores[score_valid],
        color="#222222",
        linewidth=1.0,
        label="robust motion evidence",
    )
    axes[1].axhline(
        detector.proposal_config.start_threshold,
        color="#d62728",
        linestyle="--",
        linewidth=1.0,
        label="start threshold",
    )
    axes[1].axhline(
        detector.proposal_config.continue_threshold,
        color="#ff7f0e",
        linestyle=":",
        linewidth=1.0,
        label="continuation threshold",
    )

    for proposal_index, proposal in enumerate(proposals):
        label = "proposal" if proposal_index == 0 else None
        for axis in axes:
            axis.axvspan(
                proposal.start_sec - origin,
                proposal.end_sec - origin,
                color="#ff7f0e",
                alpha=0.18,
                label=label,
            )
    for event_index, event in enumerate(events):
        label = "reference event" if event_index == 0 else None
        for axis in axes:
            axis.axvspan(
                event.start_sec - origin,
                event.end_sec - origin,
                facecolor="none",
                edgecolor="#2ca02c",
                linewidth=1.2,
                label=label,
            )

    axes[0].set_ylabel("physical magnitude")
    axes[1].set_ylabel("standardized evidence")
    axes[1].set_xlabel("seconds from recording start")
    axes[0].set_title(
        f"{recording.dataset} | {recording.recording_id} | {stream.stream_id}"
    )
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            axis.legend(
                unique.values(), unique.keys(), loc="upper right", frameon=False
            )
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output
