# Application implementation plan

> **Plan of record, 2026-08-30.** This is intentionally staged. Each stage establishes a simple
> floor before adding a learned component.

## Stage 0: protocol and data audit

1. Create an `applications/motion_monitoring/` package separate from the old zero-shot and Phase-B
   evaluation code.
2. Define the serialized `MotionSequence` contract from `DESIGN_OF_RECORD.md`.
3. Audit execution, subject, session, and timestamp boundaries for MoniPar, SPAR, Upper Limb Use,
   KneE-PAD, PHYTMO, MM-Fit, C-MHAD, WEAR, OCA, OpenPack, CrossFit, Bodyweight Exercise
   Segmentation, AIDLAB-HAR, RecoFit, CaRa, DWC/ExRAC, and any prospective collection.
4. Freeze application train/development/test roles and record which HALO checkpoints consumed each
   source during pretraining.
5. Build one deterministic manifest containing reference executions, continuous searches, target-
   absent recordings, and known event intervals.
6. Select and freeze every HALO and downstream checkpoint on development subjects only; record its
   hash, training-source provenance, and selection metric before opening sealed test results.

**Exit condition:** no reference/query leakage, and every result can be regenerated from one
manifest fingerprint.

## Stage 1: common representation export

1. Export timestamped HALO patch embeddings without pooling away movement phase.
2. Add faithful temporal adapters for the three primary author-released checkpoint baselines;
   document any sliding-window adapter used when a model lacks native temporal output.
3. Export raw IMU and parameter-free physical measurements beside every latent sequence.
4. Measure representation throughput, temporal resolution, memory, and device coverage.

**Exit condition:** every encoder produces the same `MotionSequence` schema and preserves physical
time.

## Stage 2: Task-0 event proposal and segmentation

The implementation contract for this stage is
[`TASK0_EVENT_SEGMENTATION.md`](../tasks/TASK0_EVENT_SEGMENTATION.md).

1. Implement the energy-only hysteresis floor.
2. Implement the primary compact dynamic-acceleration/angular-speed evidence, hysteresis, and
   `ruptures` PELT refinement with all durations expressed in seconds.
3. Fit feature scales, thresholds, PELT penalty, `min_size`, and `jump` on development subjects and
   freeze them.
4. Audit each evaluation source for exhaustive background annotation. Evaluate false proposals only
   where valid; otherwise report annotated-event recall, boundary quality, and reviewed unmatched
   proposals.
5. Evaluate over/under-segmentation and the Task-1/Task-3 delta from oracle intervals to Task-0
   proposals.
6. Benchmark an established temporal-localization implementation over raw or frozen
   representations only after the statistical floor is fixed.
7. Evaluate synchronized-video supervision as an optional development arm where available.

**Exit condition:** coherent motion intervals are recovered at a declared false-proposal rate and
proposal-stage misses can be distinguished from downstream matching failures.

## Stage 3: Task-1 floors

The implementation contract for this stage is
[`TASK1_ARBITRARY_DETECTION.md`](../tasks/TASK1_ARBITRARY_DETECTION.md).

1. Implement raw subsequence DTW.
2. Implement physical-feature subsequence matching.
3. Implement cosine-cost subsequence alignment over each frozen representation.
4. Add development-only threshold selection, event merging, and target-absent evaluation.
5. Produce event timelines and retrieved-reference visualizations.

**Exit condition:** cross-session event AP, false alarms per hour, count error, and boundary error
are available for all representations.

## Stage 4: Task-2 measurement

The implementation contract for this stage is
[`TASK2_CHANGE_QUANTIFICATION.md`](../tasks/TASK2_CHANGE_QUANTIFICATION.md).

1. Reuse Task-1 alignment to produce phase-local latent deviation.
2. Add duration, cadence, intensity, physical-frequency, smoothness, and stability measurements.
3. Estimate personal baseline variability and remounting noise.
4. Fit a robust longitudinal state over per-execution deviations and expose persistent drift,
   uncertainty, and isolated outliers separately.
5. Validate known differences on development data and clinical association on held-out data.
6. Build longitudinal plots with uncertainty and nearest historical examples.

**Exit condition:** the system distinguishes measurement noise from known execution changes and does
not use quality language without external validation.

## Stage 5: Task-3 motif discovery

The implementation contract for this stage is
[`TASK3_RECURRENT_MOTION_DISCOVERY.md`](../tasks/TASK3_RECURRENT_MOTION_DISCOVERY.md).

1. Build deterministic train/development/test event manifests from exact-boundary datasets.
2. Implement pooled-cosine, constrained-DTW, and raw matrix-profile controls.
3. Train the balanced pairwise same-motion metric on arbitrary source identities.
4. Calibrate the score threshold on held-out subjects and complete held-out identities.
5. Build recurrence graphs, unassigned events, bout grouping, and cluster diagnostics.
6. Evaluate first with oracle event boundaries and then with Task-0 proposals.
7. Rank motifs by recurrence and cumulative exposure, then add human confirmation and promotion into
   a Task-1 reference.

**Exit condition:** repeated-action event recall, false motif rate, count error, fragmentation, and
review burden are measured on hidden-label recordings.

## Stage 6: optional encoder adaptation

Only if the frozen-encoder Task-3 metric identifies a representation limitation:

1. Fine-tune the encoder through the same pairwise loss without changing the task definition.
2. Keep same-subject/same-device hard negatives and independent positive executions.
3. Hold out subjects, sessions, datasets, and activity identities as separate generalization tests.
4. Compare frozen and fine-tuned encoders at matched thresholds and compute cost.
5. Reuse the learned metric in Tasks 1 or 2 only when it improves their declared metrics.

Do not restore a candidate-label transformer, evidence vote, or complex curriculum unless a measured
failure of the shared metric specifically requires it.

## Stage 7: applied validation

Collect a small prospective phone/watch study if public data cannot close the evidence gap:

- multiple independent sessions and device remountings;
- several rehabilitation or occupational movements;
- accepted references and deliberately varied executions;
- long target-absent background periods;
- synchronized video and clinician/ergonomist review; and
- enough subjects for subject-level uncertainty.

## Repository cleanup rule

The branch keeps only active application documentation and implementation references. The complete
zero-shot, k-curve, evidence-engine, and historical result record remains recoverable from the branch
point at commit `32267b6`; it is not duplicated under an active-looking archive directory.
