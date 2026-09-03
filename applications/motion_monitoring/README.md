# Movement-monitoring application code

This package implements the three application tasks defined in
[`docs/design/RESEARCH_TASKS.md`](../../docs/design/RESEARCH_TASKS.md). The authoritative algorithms,
data roles, and evaluation rules remain in the linked design documents; this file only maps those
contracts to code.

## Shared runtime

- `data/`: native-time dataset adapters, immutable caches, and validation tools.
- `sequence.py`: one timestamped `MotionSequence` contract for HALO and control encoders.
- `baseline_encoder.py`: released HARNet, UniMTS, and NormWear checkpoints adapted to the same
  temporal contract; ImageBind is an optional appendix control.
- `training.py`: low-cost optimizer and gradient-health checks shared by development smokes.
- `smoke.py`: one short real-cache training check spanning all three tasks.
- `evaluation.py`: strict task/dataset/split/provenance result records and comparable tables.

Phase A and Phase B train the upstream HALO representation. In the application study, released
external encoders are frozen and receive a separately trained small module for each task. HALO is
measured both frozen under that same protocol and with a task-specific end-to-end fine-tuning arm.
Before end-to-end fitting, every encoder must pass the paired frozen direct-versus-learned utility
gate in [`docs/design/EVALUATION_PROTOCOL.md`](../../docs/design/EVALUATION_PROTOCOL.md). This keeps
episode construction fixed and isolates whether a gain comes from the representation or the small
task head.

## Task packages

- `task1/`: bounded-reference matching against a complete query timeline.
- `task2/`: set-conditioned within-person execution comparison and known-change targets.
- `task3/`: dense multiscale candidates and arbitrary-identity recurrence learning.

Run the mechanical integration check with the required environment:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.smoke --steps 3 --device cpu
```

Use `--encoder halo --checkpoint <phase-a.pt>` to exercise the real HALO encoder. Add
`--train-encoder` only for a short end-to-end gradient-path check. This command does not perform an
official training run and its tiny development-source metrics are not application results.

Smoke every primary released encoder against every task head with:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.baseline_smoke \
  --baselines harnet unimts normwear --tasks task1 task2 task3 --device cuda
```

Probe the same encoders directly on exploratory ALAMEDA and COPS longitudinal streams with:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.longitudinal_extension.baseline_data_probe --device cuda
```

Both commands are mechanical compatibility checks. Official comparisons require the frozen cohort
and task-manifest fingerprints plus the metrics in `docs/design/EVALUATION_PROTOCOL.md`. Task 1 and
Task 3 select global operating points without test data; Task 2 instead derives a personal limit
from each accepted deployment reference set.

## Task 1 and Task 2 launch sequence

Use one directory per frozen encoder. The directory must contain only caches made by that encoder
at one declared temporal stride. The commands below use a HALO checkpoint; replace
`--checkpoint <encoder.pt>` with `--baseline harnet`, `--baseline unimts`, or
`--baseline normwear` for a released frozen control. Build every encoder's caches first, then run
the common-unit command **once with every compared encoder**. All task heads and evaluations must
reuse those same common-unit files; a per-encoder "common" file would allow different models to be
scored on different examples. Do not use `--limit` caches in this sequence: the common-unit
builders reject them.

### Task 1: arbitrary task detection

Task 1 needs full timelines for query recordings and independently bounded inputs for reference
executions. Build both before freezing the common train and test intersections:

```bash
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
ROOT=applications/motion_monitoring
ART=$ROOT/artifacts/representations
COMMON=$ROOT/artifacts/common
OUT=$ART/<encoder>/task1

$PY -m applications.motion_monitoring.task1.build_manifests_v2
$PY -m applications.motion_monitoring.build_representations \
  --manifest $ROOT/manifests/COHORT_TASK1_V2.json --checkpoint <encoder.pt> \
  --datasets synth_wrist_v1 c_mhad openpack --stride-seconds 1.0 \
  --output $OUT/timelines
$PY -m applications.motion_monitoring.build_representations \
  --manifest $ROOT/manifests/COHORT_TASK1_V2.json --checkpoint <encoder.pt> \
  --stride-seconds 1.0 --output $OUT/references \
  --bounded-task-manifest $ROOT/manifests/TASK1_TRAIN_V2.json \
  --bounded-task-manifest $ROOT/manifests/TASK1_TEST_V2.json
$PY -m applications.motion_monitoring.build_common_task1_units \
  --cohort $ROOT/manifests/COHORT_TASK1_V2.json \
  --manifest $ROOT/manifests/TASK1_TRAIN_V2.json \
  --representation harnet=$ART/harnet/task1/timelines \
  --representation harnet=$ART/harnet/task1/references \
  --representation unimts=$ART/unimts/task1/timelines \
  --representation unimts=$ART/unimts/task1/references \
  --representation normwear=$ART/normwear/task1/timelines \
  --representation normwear=$ART/normwear/task1/references \
  --output $COMMON/task1_train.json
$PY -m applications.motion_monitoring.build_common_task1_units \
  --cohort $ROOT/manifests/COHORT_TASK1_V2.json \
  --manifest $ROOT/manifests/TASK1_TEST_V2.json \
  --representation harnet=$ART/harnet/task1/timelines \
  --representation harnet=$ART/harnet/task1/references \
  --representation unimts=$ART/unimts/task1/timelines \
  --representation unimts=$ART/unimts/task1/references \
  --representation normwear=$ART/normwear/task1/timelines \
  --representation normwear=$ART/normwear/task1/references \
  --output $COMMON/task1_test.json
$PY -m applications.motion_monitoring.task1.train_full \
  --cohort $ROOT/manifests/COHORT_TASK1_V2.json \
  --train-manifest $ROOT/manifests/TASK1_TRAIN_V2.json \
  --common-train-units $COMMON/task1_train.json \
  --representations $OUT/timelines $OUT/references --output $OUT/head --device cuda
```

Run evaluation twice: once without `--head-directory` for the frozen direct-DTW floor, and once
with `--head-directory $OUT/head` for the learned matcher. Both use the exact same test units.

```bash
$PY -m applications.motion_monitoring.task1.evaluate_v2 \
  --cohort $ROOT/manifests/COHORT_TASK1_V2.json \
  --train-manifest $ROOT/manifests/TASK1_TRAIN_V2.json \
  --test-manifest $ROOT/manifests/TASK1_TEST_V2.json \
  --common-train-units $COMMON/task1_train.json \
  --common-test-units $COMMON/task1_test.json \
  --representations $OUT/timelines $OUT/references --output $OUT/direct_test.json --device cuda
```

### Task 2: change quantification

Task 2 compares independently bounded executions only. The representation cache must include both
the held-out units and every declared train-pool execution:

```bash
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
ROOT=applications/motion_monitoring
ART=$ROOT/artifacts/representations
COMMON=$ROOT/artifacts/common
OUT=$ART/<encoder>/task2

$PY -m applications.motion_monitoring.task2.build_manifests_v1
$PY -m applications.motion_monitoring.build_representations \
  --manifest $ROOT/manifests/COHORT_TASK2_V1.json --checkpoint <encoder.pt> \
  --stride-seconds 1.0 --output $OUT/executions \
  --bounded-task-manifest $ROOT/manifests/TASK2_TEST_V1.json \
  --bounded-kind harmes=bounded_execution \
  --bounded-kind crossfit=repetition \
  --bounded-kind task2_modified_v1=bounded_execution
$PY -m applications.motion_monitoring.build_common_task2_units \
  --cohort $ROOT/manifests/COHORT_TASK2_V1.json \
  --train-manifest $ROOT/manifests/TASK2_TRAIN_V1.json \
  --test-manifest $ROOT/manifests/TASK2_TEST_V1.json \
  --representation harnet=$ART/harnet/task2/executions \
  --representation unimts=$ART/unimts/task2/executions \
  --representation normwear=$ART/normwear/task2/executions \
  --output $COMMON/task2.json
$PY -m applications.motion_monitoring.task2.train_full \
  --cohort $ROOT/manifests/COHORT_TASK2_V1.json \
  --train-manifest $ROOT/manifests/TASK2_TRAIN_V1.json \
  --representations $OUT/executions --common-units $COMMON/task2.json \
  --output $OUT/head --device cuda
$PY -m applications.motion_monitoring.task2.evaluate_v1 \
  --cohort $ROOT/manifests/COHORT_TASK2_V1.json \
  --test-manifest $ROOT/manifests/TASK2_TEST_V1.json \
  --train-manifest $ROOT/manifests/TASK2_TRAIN_V1.json \
  --representations $OUT/executions --common-units $COMMON/task2.json \
  --output $OUT/direct_test.json --device cuda
```

Repeat the final command with `--head-directory $OUT/head` and a different output filename for the
learned ruler. This is intentionally not a threshold-fitting flag: every Task 2 operating limit is
derived from that person's accepted reference executions at evaluation time.
