# Application implementation plan

> **Plan of record, 2026-09-02.** The three tasks are mechanically implemented. This document only
> records the remaining execution order; task definitions belong in their task documents and data
> membership belongs in the signed manifests.

## Current state

- Native-time adapters and provenance-bound raw caches exist for the active sources.
- HALO and the author-released HARNet, UniMTS, and NormWear encoders export the shared timestamped
  `MotionSequence` contract. ImageBind remains optional.
- Task 1 V2, Task 2 V1, and Task 3 V1 have separate signed cohort/task manifests.
- Task 1 and Task 2 require independently bounded representation caches where an enrolled or
  compared execution must not see surrounding-recording context.
- The all-task physical-encoder smoke and the released-baseline smoke exercise the current heads,
  losses, masks, and gradient paths. Smoke scores are not application results.

## Shared gate

For each selected encoder:

1. Build complete representation caches with raw-cache and checkpoint provenance.
2. Freeze the cross-encoder common-unit intersection before fitting a task head.
3. Run the non-learned control and the identically configured learned head.
4. Select global operating points without test data where the task needs one.
5. Evaluate every signed test unit and report each dataset separately with subject-level
   uncertainty.
6. Record trainable parameter count, fitting time, inference time, and exclusions.

Do not fine-tune an encoder until its frozen direct-versus-learned comparison is complete. An
end-to-end HALO arm is a later system measurement, not a replacement for that matched comparison.

## Task 1

1. Build full query-timeline and independently bounded reference caches for `COHORT_TASK1_V2`.
2. Run `build_common_task1_units.py` across all encoders at one temporal stride.
3. Fit the small matcher on `TASK1_TRAIN_V2` using only its synthetic training-subject hold-out for
   the operating threshold.
4. Evaluate all paired same-subject and cross-subject units in `TASK1_TEST_V2` on C-MHAD and
   OpenPack.
5. Report event precision, recall, F1, false alarms per hour, count error, and boundary error for
   direct constrained DTW and the learned projection.

The exact data and reference rules are in
[`TASK1_REFERENCE_RESOLUTION_SPEC.md`](../tasks/TASK1_REFERENCE_RESOLUTION_SPEC.md).

## Task 2

1. Rebuild `task2_modified_v1` and `TASK2_{TRAIN,TEST}_V1` whenever their provenance changes.
2. Build independently bounded representations for HARMES, CrossFit, generated variants, MoniPar,
   and KneE-PAD.
3. Run `build_common_task2_units.py` across encoders at one temporal stride.
4. Fit `ChangeRuler` only on the declared train pool. Clinical labels and evaluation queries never
   enter fitting.
5. Evaluate the frozen cosine floor, learned ruler, and raw physical-summary control on the same
   common units.
6. Report MoniPar and KneE-PAD separately. Unsupported reliability strata remain visible.

The exact training, personal-limit, and metric rules are in
[`TASK2_CHANGE_QUANTIFICATION.md`](../tasks/TASK2_CHANGE_QUANTIFICATION.md).

## Task 3

1. Build complete-timeline caches for `COHORT_TASK3_V2`.
2. Fit the affinity head on `TASK3_TRAIN_V2` (Opportunity gestures plus the assembled
   `synth_long_v1` wrist timelines), sampling event-anchored batches so every batch contains
   independent executions of one identity.
3. Fix both readout thresholds a priori on held-out training identities under a 5% false-edge
   budget. There is no development split.
4. Evaluate all `TASK3_TEST_V2` timelines on OpenPack and OCA. C-MHAD and WEAR are excluded:
   with a median of one occurrence per identity they cannot measure recurrence.
5. Report occurrence precision/recall, false occurrences per hour, count error, motif coverage,
   cluster purity together with fragmentation, boundary quality, and runtime.

The exact candidate and recurrence rules are in
[`TASK3_RECURRENT_MOTION_DISCOVERY.md`](../tasks/TASK3_RECURRENT_MOTION_DISCOVERY.md).

## Mechanical checks

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.smoke --steps 3 --device cpu
```

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.baseline_smoke \
  --baselines harnet unimts normwear --tasks task1 task2 task3 \
  --steps 3 --device cuda
```

These commands test plumbing only. Complete results must bind the task manifest, common-unit file,
representation provenance, head checkpoint, and operating-point provenance.
