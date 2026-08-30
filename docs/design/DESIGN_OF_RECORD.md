# Design of record: propose, demonstrate, detect, compare, and discover

> **Design of record, 2026-08-30.** This replaces the former Phase-A/Phase-B label-prediction
> architecture on `main`. The initial implementation is
> representation-agnostic: HALO and released external encoders enter through the same interface.

## 1. System boundary

The system accepts phone, smartwatch, or bounded consumer-wearable IMU recordings and produces a
timestamped movement representation. It does not require activity names for its core operations.

```text
native-rate IMU + acquisition metadata
    -> faithful encoder adapter
    -> timestamped patch embeddings
    -> event proposal followed by task-specific matching, alignment, or motif discovery
    -> inspectable events and measurements
```

The raw data and physical measurements remain available beside the latent representation. This is
necessary because latent distance alone cannot explain whether a difference reflects speed,
intensity, orientation, device remounting, or an actual change in execution.

## 2. Common representation contract

Every encoder adapter returns a `MotionSequence` with:

- one normalized embedding per temporal patch or sliding analysis frame;
- start time, end time, and represented duration for every embedding;
- sensor modality, placement, sampling rate, gravity state, and channel-validity metadata;
- execution, session, subject, dataset, and source-recording provenance;
- aligned raw acceleration and gyroscope references; and
- parameter-free physical summaries required by Task 2.

Adapters may use a model's native temporal features. A model exposing only a pooled embedding is
applied with a common physical-time sliding window and stride. Duplicate windows never receive extra
weight merely because one model emits a denser temporal grid.

## 3. Task 0: event proposal and segmentation

Task 0 locates candidate intervals of coherent human motion and estimates their boundaries. The
primary method uses compact physical signal evidence, hysteresis, and PELT; a learned temporal
localization head over frozen encoder timelines is a supervised comparison. Synchronized video may
provide privileged boundary supervision during development, but deployment remains IMU-only.

Task 0 does not infer intent or assign activity labels. Its full contract is owned by
[`TASK0_EVENT_SEGMENTATION.md`](../tasks/TASK0_EVENT_SEGMENTATION.md).

## 4. Task 1: reference-to-stream detection

Task 1 uses independent reference executions and constrained cosine-cost subsequence DTW over a
continuous `MotionSequence`. The initial implementation is non-parametric; learned projections are
later comparison arms rather than prerequisites. Its complete episode, matching, action-proposal,
training, and evaluation contract is owned by
[`TASK1_ARBITRARY_DETECTION.md`](../tasks/TASK1_ARBITRARY_DETECTION.md).

Task-0 proposals provide an efficient candidate search, but direct matching against the full
timeline remains a required control.

## 5. Task 2: aligned difference measurement

The reference and current execution are aligned once, then measured through separate views:

1. **Latent shape:** mean and phase-local embedding cost along the alignment path.
2. **Timing:** total duration, phase duration, cadence, and time-warp ratio.
3. **Intensity:** acceleration and angular-velocity magnitude summaries.
4. **Frequency:** dominant and band-limited motion energy in physical hertz.
5. **Smoothness and stability:** jerk-based summaries, repetition variability, and alignment spread.
6. **Configuration sensitivity:** change under remounting, placement, rate, or channel availability.

Several accepted baseline executions define a personal reference distribution. A later execution is
reported as changed only relative to that measured distribution and a test-retest noise floor.

The system reports the aligned raw traces, physical summaries, latent deviation over movement phase,
and nearest historical examples. It does not map deviation to "better" or "worse" without an
external direction label.

The complete contract is owned by
[`TASK2_CHANGE_QUANTIFICATION.md`](../tasks/TASK2_CHANGE_QUANTIFICATION.md).

## 6. Task 3: recurrent motion discovery

### 6.1 Candidate generation

Task-0 proposals are searched for recurring temporal structure. Final motif occurrences must be
non-overlapping. A direct raw-timeline motif search remains a control so proposal-stage misses are
measured.

### 6.2 Recurrence evidence

Training datasets provide bounded events and arbitrary action identities. A balanced pairwise loss
teaches a small metric whether two independently recorded events have the same source identity.
Names are not embedded, cross-dataset equivalence is not assumed, and complete identities are held
out for evaluation.

At deployment, a recurrence graph connects candidate intervals whose calibrated same-motion
probability passes a development-set threshold. Clustering extracts groups of repeated motifs. Raw
or engineered-feature matrix-profile search and frozen-embedding DTW remain non-learned controls.
The implementation reports cluster compactness, nearest-negative separation, occurrence count,
duration spread, and boundary stability rather than returning an unqualified cluster id.

### 6.3 Human confirmation

The interface ranks motif clusters by recurrence and cumulative exposure, shows representative
executions and their timeline locations, and asks a clinician or ergonomist to confirm, reject, or
name them. A confirmed motif becomes a Task-1 reference. Intent is never inferred from recurrence
alone.

The complete contract is owned by
[`TASK3_RECURRENT_MOTION_DISCOVERY.md`](../tasks/TASK3_RECURRENT_MOTION_DISCOVERY.md).

## 7. Learning boundary

Task 3 includes a small pairwise metric because arbitrary event-equivalence supervision is the
mechanism being tested. The first milestone still freezes every encoder. If representation quality
is the measured limitation, the next arm fine-tunes the encoder through the same metric:

- positives are independent executions of the same movement;
- hard negatives are different movements from the same subject and configuration;
- cross-session positives are preferred over augmented copies;
- augmentations model sensor nuisance but never manufacture the only positive evidence; and
- test tasks, subjects, and sessions remain unseen.

The metric may be reused by Tasks 1 and 2 where appropriate. Task 0 keeps its established proposal
methods. Separate complex decoders are not introduced until the simple metric fails for a measured
reason.

## 8. Role of the current HALO encoder

HALO contributes physical-time features, temporal patch embeddings, explicit missing-information
masks, and heterogeneous sensor metadata. These properties are hypotheses, not assumed advantages.
They are tested against raw signal methods and frozen released encoders under the same downstream
pipeline. [`ENCODER_HYPOTHESES.md`](ENCODER_HYPOTHESES.md) owns the cross-task literature analysis,
mechanism audit, prior evidence, and matched ablation gates for those hypotheses.

The previous candidate-label interface, zero-shot classifier, memory-label voting, admissibility
gate, and retrieve-mix-vote Phase B are not part of this design. Their code remains temporarily for
reproducibility while application code is built; their documentation is removed from the active
surface and remains recoverable from commit `32267b6`.

## 9. Implementation principles

- Use physical seconds, not sample counts, for every duration and stride.
- Preserve source timestamps and hard recording/session boundaries.
- Keep reference, query, and motif occurrences execution-disjoint in evaluation.
- Separate latent measurements from interpretable physical measurements.
- Treat synthetic streams as tests, not publication evidence.
- Keep the first system non-parametric and auditable.
- Never describe recurrence as intent or difference as clinical improvement without ground truth.
