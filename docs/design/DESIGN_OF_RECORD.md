# Design of record: demonstrate, detect, compare, and discover

> **Design of record, 2026-08-27.** This replaces the former Phase-A/Phase-B label-prediction
> architecture on the `application-motion-monitoring` branch. The initial implementation is
> representation-agnostic: HALO and released external encoders enter through the same interface.

## 1. System boundary

The system accepts phone, smartwatch, or bounded consumer-wearable IMU recordings and produces a
timestamped movement representation. It does not require activity names for its core operations.

```text
native-rate IMU + acquisition metadata
    -> faithful encoder adapter
    -> timestamped patch embeddings
    -> task-specific matching, alignment, or motif discovery
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

## 3. Task 1: reference-to-stream detection

### 3.1 Reference construction

Each reference execution remains an individual sequence. When several references are available,
the system stores their distribution rather than averaging away timing and execution variation.

### 3.2 Matching

For reference patch `i` and candidate patch `j`, the initial latent cost is:

```text
cost[i,j] = 1 - cosine(reference_embedding[i], candidate_embedding[j])
```

Subsequence dynamic time warping or an equivalent monotonic alignment finds candidate intervals
whose patch sequence matches the reference while permitting bounded speed variation. Duration and
warping penalties prevent an arbitrarily stretched fragment from receiving a good score.

The first design is non-parametric. A learned Siamese projection or match head is introduced only
after frozen representations, raw DTW, and physical-feature controls are measured.

### 3.3 Event formation

Overlapping detections are merged with temporal non-maximum suppression. Thresholds are selected on
development subjects using event precision-recall or a false-alarms-per-hour target, never on test
recordings. Reference variability and optional target-absent calibration recordings provide
per-deployment uncertainty.

## 4. Task 2: aligned difference measurement

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

## 5. Task 3: recurrent motion discovery

### 5.1 Candidate generation

The continuous embedding timeline is searched over a bounded set of physical durations. Candidate
subsequences overlap during search but final motif occurrences must be non-overlapping. Near-still
and duplicate sensor intervals are screened using explicit signal-quality rules.

### 5.2 Recurrence evidence

The initial implementation compares two established floors:

- raw or engineered-feature matrix-profile/subsequence matching; and
- the same motif search over frozen encoder embeddings.

A recurrence graph connects candidate intervals whose aligned distance passes a development-set
threshold. Clustering extracts groups of repeated motifs. The implementation must report cluster
compactness, separation from local background, occurrence count, duration spread, and boundary
stability rather than returning an unqualified cluster id.

### 5.3 Human confirmation

The interface ranks motif clusters by recurrence and cumulative exposure, shows representative
executions and their timeline locations, and asks a clinician or ergonomist to confirm, reject, or
name them. A confirmed motif becomes a Task-1 reference. Intent is never inferred from recurrence
alone.

## 6. Optional learning

The first milestone freezes every encoder. If the representation is the limiting factor, the next
model is a small shared metric projection trained episodically:

- positives are independent executions of the same movement;
- hard negatives are different movements from the same subject and configuration;
- cross-session positives are preferred over augmented copies;
- augmentations model sensor nuisance but never manufacture the only positive evidence; and
- test tasks, subjects, and sessions remain unseen.

The head is trained for verification and reused by Tasks 1-3. Separate complex decoders are not
introduced until a simpler shared metric fails for a measured reason.

## 7. Role of the current HALO encoder

HALO contributes physical-time features, temporal patch embeddings, explicit missing-information
masks, and heterogeneous sensor metadata. These properties are hypotheses, not assumed advantages.
They are tested against raw signal methods and frozen released encoders under the same downstream
pipeline.

The previous candidate-label interface, zero-shot classifier, memory-label voting, admissibility
gate, and retrieve-mix-vote Phase B are not part of this design. Their code remains temporarily for
reproducibility while application code is built; their documentation is removed from the active
surface and remains recoverable from commit `32267b6`.

## 8. Implementation principles

- Use physical seconds, not sample counts, for every duration and stride.
- Preserve source timestamps and hard recording/session boundaries.
- Keep reference, query, and motif occurrences execution-disjoint in evaluation.
- Separate latent measurements from interpretable physical measurements.
- Treat synthetic streams as tests, not publication evidence.
- Keep the first system non-parametric and auditable.
- Never describe recurrence as intent or difference as clinical improvement without ground truth.
