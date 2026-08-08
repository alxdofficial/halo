# Phase-B Training Intent

Status: executable contract as of 2026-08-08.

## Purpose

Phase B is not a classifier over the corpus vocabulary. It learns how to use a labeled memory as
an adaptation mechanism. At deployment, a new movement name and a few labeled executions can be
inserted into memory without changing model weights. The model must recognize later executions from
physical similarity, retrieve the supplied examples, read their episode-local labels, and choose
among the labels allowed for that task.

The rehabilitation use case is the design target: clinic enrollment and home measurement may use
different devices, and a useful label may be patient-specific rather than a canonical HAR class.
The corpus labels are training material for learning evidence use; they are not a fixed output head.

## One Bank, Two Distinct Concepts

The live Phase-B trainer has one physical corpus archive. Enrollment is simulated as an
episode-local overlay on that archive: selected rows receive a `PROVIDED_SUPPORT` role and a
temporary label, while stored canonical labels remain immutable.

A frozen per-patient reference snapshot for longitudinal deviation analysis is a later analytics
feature. It is not a second Phase-B training bank and is not implemented by the predictor trainer.

## Episode Recipe

Training cycles equally through four regimes:

1. `semantic_zero_support`: coherent candidate names, no examples of candidate concepts in memory.
2. `ordinary_few_support`: 1, 2, 4, or 8 examples per candidate; independent acquisition variation.
3. `cross_subject_few_support`: support and query use different virtual-subject styles, different
   physical subjects, and different acquisition configurations when selecting support.
4. `same_subject_enrollment`: support and query share one virtual-subject style but receive
   independent acquisition variation.

Supported episodes use four or eight candidates. Every candidate receives exactly the same number
of independent event/window support units. Ordinary examples of all candidate concepts are removed
from background memory before support is restored. The exact query window and verified synchronous
event are always excluded.

Positive-support episodes split equally between coherent label text and neutral random aliases such
as `routine amber`. One alias mapping is shared by candidate tokens and support rows within an
episode and changes between episodes. Random aliases are forbidden at zero support because that task
would contain no information connecting the name to the movement.

## Physical Views

The transform order is:

```text
raw source window
-> virtual-subject style (persistent within the episode role)
-> independent acquisition augmentation per execution
-> frozen or online Phase-A tokenizer
```

Virtual-subject style varies pace, dynamic acceleration amplitude, gyro amplitude, and smoothness.
Dynamic acceleration is scaled around a low-pass gravity estimate, so gravity direction and scale
remain intact. Generic Phase-B augmentation uses mild noise, gain, and SO(3) rotation. It does not
change duration, channel layout, units, gravity convention, or patch ordinals.

`calibrate_subject_style.py` was run on 3,180 subject groups across 330 matched
stream-and-label strata. Observed subject p10/p90 ratios were approximately 0.67/1.41 for pace,
0.58/1.72 for dynamic acceleration, and 0.59/1.63 for gyro RMS. The live 0.88-1.12 pace and
0.85-1.15 amplitude ranges are therefore conservative; the report is persisted at
`training/evidence/outputs/subject_style_calibration.json`.

## Retrieval Gradient

Inference is hard top-k. Training uses the same hard decoder numerically, plus an all-eligible-memory
backward surrogate:

```text
training_logits = hard_logits + soft_logits - stop_gradient(soft_logits)
```

The soft route starts at temperature 0.20 and reaches 0.07 over 500 steps. Its row prior gives equal
mass to label, source window, resolution, and represented duration, preventing fine patch grids or
large labels from winning by row count. It updates the online retrieval projection and query path
even when useful support was outside hard top-k. The estimator is intentionally biased; hard-forward
behavior, support recall, retained soft mass, and hard/soft gradient norms are monitored.

## Memory and Fine-Tuning

- archive: 250,000 balanced source windows, CPU resident;
- rotating active view: up to 16 windows per label, refreshed every 100 steps;
- final evidence budget: 64 patches per query; other retrieval caps are derived;
- default tokenizer: frozen, but query and support are re-encoded after physical augmentation;
- optional `ema_finetune`: detached EMA tokenizer keys select neighbors, while raw query, provided
  support, and selected background evidence are re-forwarded through the online tokenizer.

## Objective and Confidence

The predictor has one objective: candidate-set cross-entropy. There is no corpus-classification
auxiliary loss, explicit unknown candidate, subject-adversarial term, or metric-learning loss.

After predictor training, it is frozen. A separate confidence head predicts `correct AND
answerable` from evidence/retrieval diagnostics using adaptation-distributed truth-present episodes
and coherent truth-absent episodes. Confidence does not change candidate logits.

## Required Gates

- hard-forward logits are exactly unchanged by the soft-backward estimator;
- non-selected eligible rows receive retrieval gradient;
- support count is equal across candidates and exact event/window leakage is absent;
- random aliases are consistent within an episode and change across episodes;
- held-out-family adaptation improves as support increases;
- removing support lowers true-candidate probability in random-alias canaries;
- consistently permuting aliases does not materially change canonical predictions;
- no candidate-position, memory-row, subspace, or always-unknown collapse appears in telemetry.

The external protocol is implemented by `training/evidence/eval_enrollment.py`. It appends held-out
labeled support patches to an active corpus view without gradients and reports same-subject and
cross-subject support curves. Its `--random-aliases` mode isolates memory binding from semantic label
transfer.

Telemetry is written approximately every minute to
`training/evidence/outputs/telemetry/{phase_b_telemetry.jsonl,phase_b_telemetry_latest.json}`.
