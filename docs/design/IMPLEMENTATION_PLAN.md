# Application implementation plan

> **Plan of record, 2026-08-30.** Each stage establishes a simple complete-timeline floor before a
> learned component is added.

**Current progress.** Seven application sources have immutable payload checksums, verified native-time
adapters, and lossless map-style caches. The temporal annotation inventory measures their real event
structure and records the source capabilities of reused datasets. Task 1 has a tested encoder-agnostic
subsequence matcher with physical-time output and duplicate consolidation. The common
`MotionSequence` export, immutable task manifests, and complete evaluations remain open.

## Stage 0: protocol and annotation audit

1. Maintain one authoritative temporal annotation inventory.
2. Reconstruct XRF V2 and HARMES complete timelines without Phase-A excerpt assumptions.
3. Define the serialized `MotionSequence` contract.
4. Freeze application train/development/test roles and upstream-checkpoint provenance.
5. Freeze deterministic manifests containing subjects, recordings, references, target intervals,
   target-absent time, annotation scope, and split fingerprints.
6. Select thresholds and checkpoints on development subjects before sealed evaluation.

**Exit condition:** no reference/query leakage, every negative interval is supported by its annotation
contract, and every result regenerates from one manifest fingerprint.

## Stage 1: common representation export

1. Export timestamped HALO patch embeddings without pooling away movement phase.
2. Add faithful temporal adapters for the primary author-released encoders.
3. Export raw IMU and parameter-free physical measurements beside every latent sequence.
4. Measure temporal resolution, throughput, memory, and device coverage.

**Exit condition:** every encoder produces the same `MotionSequence` schema in physical time.

## Stage 2: Task-1 full-timeline floors

1. **Implemented:** open-begin/open-end cosine subsequence DTW with bounded local warp slope.
2. **Implemented:** trace every feasible endpoint into a physical-time candidate interval.
3. **Implemented:** score thresholding and temporal non-maximum suppression for multiple detections.
4. Wire raw-signal, physical-feature, HALO, and released-encoder timelines into the same matcher.
5. Fit thresholds on target-present and target-absent development recordings.
6. Build natural and synthetic training episodes without leaking joins or augmentation watermarks.
7. Produce event timelines, alignment paths, false-alarm curves, count error, and boundary error.

**Exit condition:** event AP and recall at a declared false-alarms-per-hour operating point are
available for every representation on natural continuous recordings.

## Stage 3: Task-2 measurement

1. Reuse Task-1 alignment for phase-local latent deviation.
2. Add duration, cadence, intensity, physical-frequency, smoothness, and stability measurements.
3. Estimate personal baseline variability, remounting noise, and minimum detectable change.
4. Fit a robust longitudinal state over per-execution deviations.
5. Validate known execution differences and held-out longitudinal association separately.
6. Build phase and longitudinal plots with uncertainty and nearest historical examples.

**Exit condition:** known changes are separated from ordinary measurement noise without using quality
or clinical language absent external validation.

## Stage 4: Task-3 dense recurrence discovery

1. Encode each complete recording once at a fine physical-time stride.
2. Pool adjacent embeddings over a small declared duration set to form a temporal pyramid.
3. Assign candidate/event overlap targets from exact-boundary training sources; ignore ambiguous
   partial overlaps and incompletely annotated negatives.
4. Implement pooled cosine, constrained DTW, and variable-length matrix-profile controls.
5. Train the balanced same-motion metric on arbitrary source identities.
6. Consolidate overlapping multiscale candidates with a frozen temporal decoding rule.
7. Calibrate recurrence thresholds on held-out subjects and identities.
8. Build recurrence graphs, unassigned candidates, motif diagnostics, and human review output.
9. Report oracle source-interval matching and complete-timeline discovery separately.

**Exit condition:** repeated-event coverage, false motif rate, count error, fragmentation, boundary
quality, and review burden are measured on hidden-label continuous recordings.

## Stage 5: optional encoder adaptation

Only if a frozen comparison identifies a representation limitation:

1. fine-tune the encoder through the unchanged task loss;
2. keep independent positive executions and same-context hard negatives;
3. hold out subjects, sessions, identities, and datasets as separate generalization tests; and
4. compare frozen and fine-tuned encoders at matched operating points and compute cost.

Do not restore candidate-label voting, an evidence transformer, or a complex curriculum unless a
measured failure specifically requires it.

## Optional motion-proposal speed arm

The existing physical-evidence/hysteresis/PELT implementation may screen long still intervals. It is
not on the critical path and cannot replace complete-timeline evaluation. Retain it only if it yields
a useful runtime reduction at negligible downstream recall loss.

## Applied validation

Collect a focused phone/watch study only if public data cannot close the evidence gap. It should
include independently remounted sessions, confirmed executions, controlled changes, long
target-absent periods, synchronized video, and clinician or ergonomist review.

## Repository cleanup rule

`main` keeps only the active application design. Historical zero-shot and evidence-engine work remains
recoverable from `archive/pre-application-main-20260830` at commit `32267b6`; it is not duplicated in
active documentation.
