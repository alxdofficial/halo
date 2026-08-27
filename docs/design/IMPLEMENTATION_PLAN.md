# Application implementation plan

> **Plan of record, 2026-08-27.** This is intentionally staged. Each stage establishes a simple
> floor before adding a learned component.

## Stage 0: protocol and data audit

1. Create an `applications/motion_monitoring/` package separate from the old zero-shot and Phase-B
   evaluation code.
2. Define the serialized `MotionSequence` contract from `DESIGN_OF_RECORD.md`.
3. Audit execution, subject, session, and timestamp boundaries for MoniPar, SPAR, Upper Limb Use,
   KneE-PAD, PHYTMO, MM-Fit, and any occupational collection.
4. Freeze application train/development/test roles and record which HALO checkpoints consumed each
   source during pretraining.
5. Build one deterministic manifest containing reference executions, continuous searches, target-
   absent recordings, and known event intervals.

**Exit condition:** no reference/query leakage, and every result can be regenerated from one
manifest fingerprint.

## Stage 1: common representation export

1. Export timestamped HALO patch embeddings without pooling away movement phase.
2. Add faithful temporal adapters for the four author-released checkpoint baselines.
3. Export raw IMU and parameter-free physical measurements beside every latent sequence.
4. Measure representation throughput, temporal resolution, memory, and device coverage.

**Exit condition:** every encoder produces the same `MotionSequence` schema and preserves physical
time.

## Stage 2: Task-1 floors

1. Implement raw subsequence DTW.
2. Implement physical-feature subsequence matching.
3. Implement cosine-cost subsequence alignment over each frozen representation.
4. Add development-only threshold selection, event merging, and target-absent evaluation.
5. Produce event timelines and retrieved-reference visualizations.

**Exit condition:** cross-session event AP, false alarms per hour, count error, and boundary error
are available for all representations.

## Stage 3: Task-2 measurement

1. Reuse Task-1 alignment to produce phase-local latent deviation.
2. Add duration, cadence, intensity, physical-frequency, smoothness, and stability measurements.
3. Estimate personal baseline variability and remounting noise.
4. Validate known differences on development data and clinical association on held-out data.
5. Build longitudinal plots with uncertainty and nearest historical examples.

**Exit condition:** the system distinguishes measurement noise from known execution changes and does
not use quality language without external validation.

## Stage 4: Task-3 motif discovery

1. Implement raw matrix-profile or equivalent subsequence-motif floors.
2. Run the same duration search over frozen embedding timelines.
3. Build recurrence graphs, overlap suppression, clustering, and cluster-quality diagnostics.
4. Rank motifs by recurrence and cumulative exposure.
5. Add human confirmation and promotion of a motif into a Task-1 reference.

**Exit condition:** repeated-action event recall, false motif rate, count error, fragmentation, and
review burden are measured on hidden-label recordings.

## Stage 5: optional metric learning

Only if the frozen representations are inadequate:

1. Train a small Siamese projection on independent same-movement executions.
2. Use same-subject/same-device hard negatives.
3. Hold out subjects, sessions, datasets, and activity identities as separate generalization tests.
4. Reuse the one learned metric in all three tasks.
5. Compare against the frozen input representation at matched thresholds and compute cost.

Do not restore a candidate-label transformer, evidence vote, or complex curriculum unless a measured
failure of the shared metric specifically requires it.

## Stage 6: applied validation

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
