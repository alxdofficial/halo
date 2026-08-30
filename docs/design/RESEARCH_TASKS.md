# Research tasks

> **Agreed scope, 2026-08-30.** These are the four active application tasks on the
> `application-motion-monitoring` branch.

The four tasks are components of one applied movement-monitoring system. Individually they are
deliberately small. The research contribution is the complete path from continuous wearable data to
inspectable detection, longitudinal change, and recurrent-motion outputs. Task 0 is established
infrastructure rather than a claimed activity-recognition advance.

## Shared terms

- **Execution:** one bounded performance of a movement.
- **Reference:** one or more independently recorded executions supplied as examples.
- **Continuous recording:** an interval that may contain zero, one, or many executions.
- **Motion motif:** a recurring, temporally structured sensor pattern not yet assigned human
  meaning.
- **Activity:** a reference-supplied or human-confirmed movement with operational meaning.
- **Personal baseline:** the distribution of variation among accepted reference executions for one
  person, device setup, and task.
- **Candidate motion event:** a bounded interval with observable, temporally coherent human motion;
  it is not yet assumed to be intentional or meaningful.

## Task 0: event proposal and segmentation

### Question

Where does coherent human motion occur in a continuous recording, and what are its start and end
boundaries?

Task 0 screens near-still periods, sensor noise, and recording defects, then proposes structured
motion for later analysis. It cannot reliably distinguish deliberate action from every incidental
movement and does not assign an activity name. IMU alone cannot prove intent, so the operational
target is observable **coherent motion**, with intent supplied later by an enrolled reference,
synchronized video, or human confirmation.

### Input and output

```text
continuous recording
    -> candidate motion intervals, boundary confidence, and motion-coherence scores
```

### Required properties

1. preserve native timestamps and hard recording boundaries;
2. detect intervals across a practical range of physical durations;
3. screen stillness and sensor artifacts while retaining uncertain movement;
4. produce stable boundaries under small changes to the analysis window;
5. retain uncertain intervals instead of forcing an activity decision; and
6. support synchronized video as privileged training or annotation evidence while remaining
   IMU-only at deployment.

Task 0 is shared infrastructure. Task 1 may search its proposals, Task 2 compares confirmed
proposals, and Task 3 clusters recurring proposals. Direct full-timeline matching remains a Task-1
control so segmentation errors are visible rather than silently limiting detection.

The detailed contract is owned by
[`TASK0_EVENT_SEGMENTATION.md`](../tasks/TASK0_EVENT_SEGMENTATION.md).

## Task 1: arbitrary task detection

### Question

Given one or a few reference executions, did the movement occur in a later recording, and where?

The reference can be a named rehabilitation exercise, an arbitrary personal task such as "exercise
one," or a movement whose language label is irrelevant. The detector must work when the later
execution differs in speed, amplitude, ordinary repetition variability, session, or device mounting.

### Input and output

```text
reference executions + continuous recording
    -> event intervals, match scores, and retrieved reference evidence
```

### Primary application

A clinician records several acceptable repetitions in clinic. The system scans a later home
session, counts detected repetitions, and returns the corresponding intervals for inspection.

### Required controls

- Raw-signal subsequence dynamic time warping.
- Engineered physical features with the same matcher.
- Frozen HALO and released external encoders with the same temporal matcher.
- Sessions containing no target occurrence.
- Hard negatives from the same person, device, and activity family.

Synthetic insertion of an augmented excerpt is a unit and stress test only. A publication result
must use independent real executions.

The full Task-1 design, including the 120-second episode contract, construction rules, matcher,
training arms, and sealed evaluation datasets, is owned by
[`TASK1_ARBITRARY_DETECTION.md`](../tasks/TASK1_ARBITRARY_DETECTION.md).

## Task 2: activity difference quantification

### Question

For executions known to represent the same task, which differences are ordinary personal variation
and which form a persistent change over time?

The system separates properties that temporal alignment should ignore from properties that should
remain visible. For example, alignment can compare movement shape despite different speed, while
duration is reported separately rather than discarded.

### Outputs

- aligned latent-shape deviation;
- phase-local deviation showing where the movement changed;
- duration and cadence differences;
- intensity and frequency-content differences;
- smoothness and repetition-consistency measures;
- uncertainty relative to the person's accepted baseline; and
- nearest historical executions for inspection.

Per-execution deviations are passed through a robust longitudinal state estimate. This is a
low-pass model over successive executions or sessions, not filtering of the raw movement itself. It
suppresses isolated outliers while retaining persistent drift and its uncertainty.

### Interpretation rule

Without external ground truth, the output is **difference**, not quality or improvement. A quality,
correctness, fatigue, or clinical score requires clinician labels, known execution perturbations,
or validated physical measurements.

### Primary application

Track whether a rehabilitation exercise changes across days or weeks and whether the change is
larger than normal test-retest and remounting variability.

The complete Task-2 alignment, measurement, training-data, reliability, and visualization protocol
is owned by [`TASK2_CHANGE_QUANTIFICATION.md`](../tasks/TASK2_CHANGE_QUANTIFICATION.md).

## Task 3: recurrent motion discovery

### Question

Which coherent motion motifs recur frequently in an unlabeled occupational recording?

The system does not begin with a reference or activity vocabulary. It clusters Task-0 proposals
using a same-motion metric trained on arbitrary labeled event identities. It reports count, duration,
cadence, and within-shift change. Evaluation identities are unseen and their labels remain hidden
during clustering. A raw-timeline motif search remains a non-learned control for proposal-stage
misses.

### Input and output

```text
continuous recording
    -> recurring motif clusters
    -> count, duration, cadence, compactness, examples, and timeline locations
    -> human confirmation or rejection
```

"More than ten repetitions per day" is a useful configurable review rule, not a universal scientific
threshold. It filters which learned identity clusters are displayed; it does not define cluster
membership. The interface should expose minimum recurrence, bout gap, duration range, and review
budget while evaluation reports complete precision-recall behavior across thresholds.

### Coherent motion versus noise

IMU data cannot determine whether a person intended an action. The operational definition of a
coherent motif is therefore based on observable evidence:

1. recurrence in several non-overlapping intervals;
2. stable internal temporal ordering after limited time warping;
3. greater within-cluster similarity than similarity to local background;
4. boundaries that remain stable under small changes to the analysis window;
5. recurrence across independent periods where possible; and
6. enough duration and signal energy to exclude sensor quantization and near-still duplicates.

Repeated arm flailing may satisfy these conditions. It remains a motion motif until a human confirms
that it represents meaningful work. This human-in-the-loop step is part of the intended design, not
an evaluation workaround.

### Primary application

An ergonomist records a work shift, reviews the most frequent motion motifs, names the relevant
tasks, and promotes selected motifs into Task-1 references for future monitoring. Task 2 then tracks
whether those tasks drift over a shift or across days.

The complete Task-3 motif-search, clustering, data-construction, review, and evaluation protocol is
owned by [`TASK3_RECURRENT_MOTION_DISCOVERY.md`](../tasks/TASK3_RECURRENT_MOTION_DISCOVERY.md).

## How the tasks fit together

```text
continuous recording -> Task 0: propose motion intervals
                              |
demonstrated reference -------+----------> Task 1: detect occurrences
                              |                    |
                              +-> Task 3: cluster -+
                                                   |
                                                   v
                                          Task 2: quantify change
```

The common technical primitive is a timestamped sequence of patch embeddings plus physical sensor
measurements. Task 0 bounds candidate motion; the later tasks match, compare, or cluster it.

## Success criteria

The program is useful only if it demonstrates all of the following:

- reliable motion proposal and boundary estimation at a practically low false-proposal rate;
- reliable enrolled-event detection at a practically low false-alarm rate;
- change scores that exceed raw-signal and engineered-feature controls;
- test-retest stability under device remounting and ordinary execution variability;
- motif discovery that recovers repeated actions without labeling every possible activity; and
- outputs that a clinician or ergonomist can inspect rather than only one opaque scalar.

The measurement plan is defined in [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md).
