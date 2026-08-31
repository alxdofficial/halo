# Application implementation plan

> **Plan of record, 2026-08-31.** Each stage establishes a simple complete-timeline floor before a
> learned component is added.

**Current progress.** Seven application sources have immutable payload checksums, verified native-time
adapters, and lossless map-style caches. The temporal annotation inventory measures their real event
structure and records the source capabilities of reused datasets. The runtime `MotionSequence`
contract now exports native-clock patch intervals, validity, normalized embeddings, physical
summaries, and provenance through both HALO and a cheap physical-feature control. All three task
pipelines have short real-cache optimizer smokes with finite head and encoder gradients. These are
mechanical checks, not trained models or results. The seven-source `COHORT_V1` manifest and shared
frozen-representation cache are implemented. The manifest contains 864 nonduplicated
training/development recordings and 277 sealed-test recordings and rejects the released WEAR and OCA
partitions as split authorities. HARMES/MoniPar application adapters, complete representation
caches, calibrated operating points, long training, and complete evaluations remain open.

Reproduce the cross-task mechanical check with:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.smoke --steps 3 --device cpu
```

The same command accepts `--encoder halo --checkpoint <phase-a.pt>` and optional
`--train-encoder` for a one-step GPU gradient-path check. Do not treat its deliberately tiny,
development-source metrics as application performance.

## Stage 0: protocol and annotation audit

1. Maintain one authoritative temporal annotation inventory.
2. Reconstruct XRF V2 and HARMES complete timelines without Phase-A excerpt assumptions.
3. **Implemented:** define, validate, and serialize `MotionSequence` without losing physical time.
4. **Implemented for seven canonical sources:** freeze application train/development/test roles and
   upstream-cache provenance.
5. **Recording cohort implemented; task episodes open:** freeze deterministic manifests containing
   subjects, recordings, references, target intervals, target-absent time, annotation scope, and
   split fingerprints.
6. Select thresholds and checkpoints on development subjects before sealed evaluation.

**Exit condition:** no reference/query leakage, every negative interval is supported by its annotation
contract, and every result regenerates from one manifest fingerprint.

## Stage 1: common representation export

1. **Implemented and smoke-tested:** export timestamped frozen HALO patch embeddings without pooling
   away movement phase.
2. Add faithful temporal adapters for the primary author-released encoders.
3. Export raw IMU and parameter-free physical measurements beside every latent sequence.
4. Measure temporal resolution, throughput, memory, and device coverage.

**Exit condition:** every selected encoder produces the same cached `MotionSequence` schema in
physical time. HALO passes; released baseline cache adapters remain open.

## Stage 2: Task-1 full-timeline floors

1. **Implemented:** open-begin/open-end cosine subsequence DTW with bounded local warp slope.
2. **Implemented:** trace every feasible endpoint into a physical-time candidate interval.
3. **Implemented:** score thresholding and temporal non-maximum suppression for multiple detections.
4. **Partially implemented:** physical-feature and HALO timelines use the same matcher; raw-signal
   and released-encoder adapters remain open.
5. Fit thresholds on target-present and target-absent development recordings.
6. **Implemented mechanically:** natural cache episodes plus deterministic independent-view,
   retiming, distractor, target-absent, validity, and join-guard test episodes. Pair preflight records
   data-quality exclusions before loading, and guards break alignment paths rather than only masking
   endpoint losses.
7. Produce event timelines, alignment paths, false-alarm curves, count error, and boundary error.

**Exit condition:** event AP and recall at a declared false-alarms-per-hour operating point are
available for every representation on natural continuous recordings.

## Stage 3: Task-2 measurement

1. **Implemented mechanically:** phase-normalized latent residual curves for bounded executions.
2. **Partially implemented:** duration and IMU magnitude/dynamic/jerk summaries are exported;
   cadence, frequency, smoothness, and stability reporting remain open.
3. **Personal joint-variation fit implemented:** estimate robust personal center, measurement-floor
   scaling, and shrinkage-regularized covariance; remounting noise and minimum detectable change
   calibration remain open.
4. Fit a robust longitudinal state over per-execution deviations.
5. **Mechanically implemented:** masked, unit-scaled known-change classification/regression and
   accepted-variation controls; real longitudinal association remains open.
6. Build phase and longitudinal plots with uncertainty and nearest historical examples.

**Exit condition:** known changes are separated from ordinary measurement noise without using quality
or clinical language absent external validation.

## Stage 4: Task-3 dense recurrence discovery

1. **Implemented mechanically:** encode complete recording crops at a fine physical-time stride.
2. **Implemented:** pool adjacent embeddings over declared physical durations, with candidate output
   invariant to heterogeneous batch padding.
3. **Implemented:** assign candidate/event overlap targets from exact-boundary training sources; ignore ambiguous
   partial overlaps and incompletely annotated negatives.
4. Implement pooled cosine, constrained DTW, and variable-length matrix-profile controls.
5. **Implemented mechanically:** train a balanced same-motion metric on scoped arbitrary identities.
6. **Implemented:** consolidate overlapping multiscale candidates with temporal NMS.
7. Calibrate recurrence thresholds on held-out subjects and identities.
8. **Partially implemented:** mutual-neighbor recurrence graphs, unassigned candidates, and motif
   diagnostics exist; human-review artifacts remain open.
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
