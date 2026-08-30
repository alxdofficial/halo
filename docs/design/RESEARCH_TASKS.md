# Research tasks

> **Agreed scope, 2026-08-30.** These are the three active application tasks on `main`.

The contribution is one applied movement-monitoring system. Each task consumes a complete recording
or independently bounded executions and performs the temporal localization needed for its own output.
A generic motion-proposal stage is not a prerequisite.

## Shared terms

- **Execution:** one bounded performance of a movement.
- **Reference:** one or more independently recorded executions supplied as examples.
- **Continuous recording:** an interval that may contain zero, one, or many executions.
- **Temporal candidate:** a physical-time interval produced during dense matching or multiscale
  search; several overlapping candidates may describe the same occurrence.
- **Motion motif:** a recurring, temporally structured sensor pattern not yet assigned human meaning.
- **Activity:** a reference-supplied or human-confirmed movement with operational meaning.
- **Personal baseline:** the distribution of accepted execution variation for one person, setup, and
  task.

## Task 1: arbitrary task detection

### Question

Given one or a few reference executions, did the movement occur in a later complete recording, and
where?

```text
reference executions + complete recording
    -> subsequence alignment across timestamped patch embeddings
    -> event intervals, match scores, and aligned reference evidence
```

The movement name is irrelevant. Constrained subsequence DTW handles bounded speed change and returns
the aligned start and end directly. Overlapping detections are consolidated in physical time. A
generic motion detector must not prevent the matcher from inspecting any part of the recording.

Training may use naturally continuous recordings with exact event annotations. When these are
insufficient, independent isolated executions can be inserted into compatible real background with
other actions as distractors. Synthetic timelines are training data; primary evaluation uses natural
continuous recordings with real per-instance boundaries and target-absent periods.

Required controls are raw-signal DTW, engineered physical features, frozen HALO, and released external
encoders under the same matcher. The full contract is owned by
[`TASK1_ARBITRARY_DETECTION.md`](../tasks/TASK1_ARBITRARY_DETECTION.md).

## Task 2: activity difference quantification

### Question

For independently bounded executions known to represent the same task, which differences are ordinary
personal variation and which form a persistent change over time?

The executions are aligned, then latent shape and interpretable physical measurements are reported
separately. A robust personal reference distribution estimates ordinary test-retest variation. A
longitudinal state model suppresses isolated outliers while retaining persistent drift and its
uncertainty.

```text
confirmed reference executions + later confirmed execution
    -> temporal alignment
    -> latent and physical differences over movement phase
    -> change relative to personal variation and measurement noise
```

Task 2 does not require generic event proposals. Source intervals, user-guided recordings, or Task-1
detections can supply executions. Its core evaluation uses source boundaries so change measurement is
not confounded with localization error.

Without external ground truth, the output is **difference**, not correctness, fatigue, clinical
improvement, or disease. Those interpretations require known execution variants, clinician labels, or
validated measurements. The full contract is owned by
[`TASK2_CHANGE_QUANTIFICATION.md`](../tasks/TASK2_CHANGE_QUANTIFICATION.md).

## Task 3: recurrent motion discovery

### Question

Which temporally structured motions recur frequently in an unlabeled occupational recording?

Task 3 searches the complete recording at several physical durations. The raw recording is encoded
once at a fine temporal stride; adjacent patch embeddings are pooled into a temporal pyramid rather
than re-encoding each duration. Every candidate retains its physical start, end, scale, and score.

```text
complete recording
    -> timestamped base patch embeddings
    -> dense multiscale temporal candidates
    -> same-motion similarity graph
    -> temporal consolidation of overlapping candidates
    -> recurring motif clusters, examples, counts, and exposure
```

Training annotations define arbitrary event-equivalence identities. Candidates overlapping the same
source event are positives; candidates covering explicitly different events are negatives; ambiguous
partial overlaps are ignored. Labels are hidden during discovery and used only to construct training
targets or score the output. Unlabeled time is negative only when the source annotation is exhaustive.

Temporal non-maximum suppression or weighted interval selection consolidates overlapping candidates
that describe one occurrence. This makes localization part of motif discovery rather than a separate
front end. A reviewer confirms, rejects, or names the resulting motif. The full contract is owned by
[`TASK3_RECURRENT_MOTION_DISCOVERY.md`](../tasks/TASK3_RECURRENT_MOTION_DISCOVERY.md).

## Optional motion-proposal baseline

The implemented physical evidence, hysteresis, and bounded-PELT detector remains an optional speed
baseline. It can screen long still periods before Task 1 or Task 3, but every primary evaluation also
runs directly on the complete timeline. Proposal recall therefore cannot silently cap task quality.
This method is documented in
[`MOTION_PROPOSAL_BASELINE.md`](../methods/MOTION_PROPOSAL_BASELINE.md).

## How the tasks fit together

```text
demonstrated reference + complete recording -> Task 1: detect occurrences
                                                    |
complete unlabeled recording -------------> Task 3: discover motifs
                                                    |
confirmed or detected executions ------------------+
                                                    v
                                          Task 2: quantify change
```

The shared primitive is a timestamped sequence of patch embeddings plus physical sensor measurements.
Task 1 performs reference-conditioned localization; Task 3 performs reference-free dense recurrence
search; Task 2 measures confirmed executions.

## Success criteria

- enrolled-event detection at a declared false-alarm rate;
- change scores that improve on raw and engineered physical controls where latent structure helps;
- test-retest stability under ordinary remounting and execution variation;
- motif discovery that recovers repeated instances without being told the number of clusters;
- accurate physical-time boundaries after task-specific decoding; and
- outputs that a clinician or ergonomist can inspect.

Dataset eligibility and annotation strength are owned by
[`ANNOTATION_INVENTORY.md`](../data/ANNOTATION_INVENTORY.md). The measurement plan is defined in
[`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md).
