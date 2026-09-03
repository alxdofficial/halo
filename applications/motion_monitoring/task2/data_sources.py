"""Frozen Task-2 source roles and event-selection contracts.

This module deliberately describes source selection, not episode construction.
Task 2 trains only from HARMES and CrossFit and evaluates on MoniPar and KneE-PAD.

Decided 2026-09-02, after measuring the alternatives:

* **DUO-GAIT is dropped.** Its wrist signals live in a 2.9 GB Zenodo archive that
  repeatedly truncated and then began returning 502; the targeted range-request
  path failed the same way. Only 2 of 16 subjects arrived with paired conditions,
  which cannot support an induced-change cell.
* **KneE-PAD is the second evaluation cell (added 2026-09-02).** A single-source
  evaluation was too thin, and this one costs no download: it is already converted,
  is absent from the checkpoint's 18 training datasets, and supplies many
  subject-exercise groups with at least five correct trials and at least one
  incorrect variant. Its deviations were curated from what patients actually did
  wrong while unsupervised, which is closer to real change than faults performed
  to order. Muscle-belly placement puts it outside the consumer deployment
  envelope, so it is reported as a cross-placement stress cell.
* **Faulty Exercises IMU is rejected.** Its only edge over KneE-PAD was being
  unseen by pretraining, which KneE-PAD also is, and its 923 repetitions over 16
  subjects and 13 classes leave roughly four correct executions per group, below
  the five an episode needs.

The two cells are complementary and neither substitutes for the other. MoniPar
carries between-week nuisance false alarms and a small clinician-rated
bradykinesia change cell on a consumer watch.
KneE-PAD carries the controlled known-difference test on a research placement,
within a single visit, so it contributes no between-day estimate. What neither
supplies is induced physiological change; an unsegmented longitudinal source must
never be substituted to fill that gap.
"""

from __future__ import annotations

from dataclasses import dataclass

from applications.motion_monitoring.data.contracts import EventInterval, RawRecording


@dataclass(frozen=True)
class Task2SourceSpec:
    """The admissible event unit for one source in the Task-2 protocol."""

    dataset: str
    role: str
    event_kind: str
    description: str


TASK2_SOURCE_SPECS = (
    Task2SourceSpec(
        dataset="harmes",
        role="train",
        event_kind="bounded_execution",
        description="same-person, cross-session dominant-wrist ADLs",
    ),
    Task2SourceSpec(
        dataset="crossfit",
        role="train",
        event_kind="repetition",
        description="within-set smartwatch repetitions and synthetic modifications",
    ),
    Task2SourceSpec(
        dataset="monipar",
        role="evaluation",
        event_kind="bounded_execution",
        description="weekly, single-run wrist protocol executions",
    ),
    Task2SourceSpec(
        dataset="kneepad",
        role="evaluation",
        event_kind="bounded_execution",
        description=(
            "controlled known-difference cell: correct plus two curated incorrect "
            "variants per knee exercise; converted by the local reviewed adapter"
        ),
    ),
)

TASK2_TRAIN_DATASETS = ("harmes", "crossfit")
TASK2_READY_EVALUATION_DATASETS = ("monipar", "kneepad")
TASK2_DEFERRED_EVALUATION_DATASETS = ()


def spec_for(dataset: str) -> Task2SourceSpec:
    """Return the frozen Task-2 source specification for ``dataset``."""

    for spec in TASK2_SOURCE_SPECS:
        if spec.dataset == dataset:
            return spec
    raise KeyError(f"{dataset!r} is not a Task-2 source")


def is_selected_event(recording: RawRecording, event: EventInterval) -> bool:
    """Whether one source annotation is a valid Task-2 execution candidate.

    The MoniPar adapter retains uniformly tremor-graded resting intervals as
    metadata for an extension. They are bounded executions in the raw cache but
    not part of the primary single-run protocol evaluation, so exclude them here.
    """

    spec = spec_for(recording.dataset)
    if spec.role == "evaluation_deferred" or event.annotation_kind != spec.event_kind:
        return False
    if recording.dataset == "monipar" and event.label == "resting":
        return False
    return True


def validate_selected_recording(recording: RawRecording) -> None:
    """Check source metadata needed before an execution enters Task-2 data."""

    spec = spec_for(recording.dataset)
    if spec.role == "evaluation_deferred":
        raise ValueError(f"{recording.dataset} is not wired into Task 2 yet")
    if not recording.streams:
        raise ValueError(f"{recording.recording_id}: no sensor stream")
    if (
        recording.dataset == "harmes"
        and recording.streams[0].placement != "dominant_wrist"
    ):
        raise ValueError(f"{recording.recording_id}: HARMES must use its dominant wrist")
    if recording.dataset == "kneepad" and not recording.metadata.get("single_visit_cohort"):
        raise ValueError(f"{recording.recording_id}: KneE-PAD repeats are within one visit")
    if recording.dataset == "monipar" and recording.streams[0].channels != (
        "acc_x",
        "acc_y",
        "acc_z",
    ):
        raise ValueError(f"{recording.recording_id}: MoniPar must remain acceleration-only")
