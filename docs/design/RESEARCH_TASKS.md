# Research tasks

> **Agreed scope, 2026-08-27.** These are the three active application tasks on the
> `application-motion-monitoring` branch.

## Shared terms

- **Execution:** one bounded performance of a movement.
- **Reference:** one or more independently recorded executions supplied as examples.
- **Continuous recording:** an interval that may contain zero, one, or many executions.
- **Motion motif:** a recurring, temporally structured sensor pattern not yet assigned human
  meaning.
- **Activity:** a reference-supplied or human-confirmed movement with operational meaning.
- **Personal baseline:** the distribution of variation among accepted reference executions for one
  person, device setup, and task.

## Task 1: arbitrary activity detection

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

## Task 2: activity difference quantification

### Question

For two executions known to represent the same task, how are they different?

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

### Interpretation rule

Without external ground truth, the output is **difference**, not quality or improvement. A quality,
correctness, fatigue, or clinical score requires clinician labels, known execution perturbations,
or validated physical measurements.

### Primary application

Track whether a rehabilitation exercise changes across days or weeks and whether the change is
larger than normal test-retest and remounting variability.

## Task 3: recurrent motion discovery

### Question

Which coherent motion motifs recur frequently in an unlabeled occupational recording?

The system does not begin with a reference or activity vocabulary. It searches the latent timeline
for non-overlapping subsequences that recur with similar temporal structure, clusters those
occurrences, and reports their count, duration, cadence, and within-shift change.

### Input and output

```text
continuous recording
    -> recurring motif clusters
    -> count, duration, cadence, compactness, examples, and timeline locations
    -> human confirmation or rejection
```

"More than ten repetitions per day" is a useful configurable review rule, not a universal scientific
threshold. The interface should expose minimum recurrence, duration range, and review budget while
the evaluation reports complete precision-recall behavior across thresholds.

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

## How the tasks fit together

```text
demonstrated reference ---------------------> Task 1: detect occurrences
                                                     |
unlabeled occupational stream -> Task 3: discover --+
                                                     |
                                                     v
                                            Task 2: quantify change
```

The common technical primitive is a timestamped sequence of patch embeddings plus physical sensor
measurements. The tasks differ in whether the movement is supplied, compared, or discovered.

## Success criteria

The program is useful only if it demonstrates all of the following:

- reliable event detection at a practically low false-alarm rate;
- change scores that exceed raw-signal and engineered-feature controls;
- test-retest stability under device remounting and ordinary execution variability;
- motif discovery that recovers repeated actions without labeling every possible activity; and
- outputs that a clinician or ergonomist can inspect rather than only one opaque scalar.

The measurement plan is defined in [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md).
