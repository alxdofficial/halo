# HALO docs

> ## If you are writing the paper, read exactly these, in this order
> 1. [**design/MOTIVATION.md**](design/MOTIVATION.md) — the thesis. *Why* HALO exists and why it is
>    not trivial. If a framing cannot survive the rebuttals in its §2, it is not the contribution.
> 2. [**design/POSITIONING.md**](design/POSITIONING.md) — what the result is *for*, and how to report
>    it. ⚠️ its argument is live, its **numbers are pre-v4** — take numbers only from the block below.
> 3. [**design/PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md**](design/PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md)
>    — what Phase A actually is, as built, and the artifact handed to Phase B.
> 4. [**design/PHASE_B_TRAINING_INTENT.md**](design/PHASE_B_TRAINING_INTENT.md) — what Phase B is,
>    why memory is an adaptation mechanism, and the evidence required before claiming it works.
> 5. [**results/RESULTS.md**](results/RESULTS.md) — **the current headline result.** The current
>    HALO checkpoint and all six external encoders on the shared manifest, with the promoted 1-NN
>    readout separated from the learned retrieve-mix-vote ablation.
> 6. [**results/PHASE_B_TRAINING_STATUS.md**](results/PHASE_B_TRAINING_STATUS.md) — the current
>    Phase-B run history, matched tables, confirmed defects, and next-run requirements.
>
> Then take every number from Results below, and nothing at all from
> [`archive/`](archive/README.md).

## Current Matched Results

**2026-08-24.** The promoted checkpoint is `PB-04-CK-DENSE`, selected at step 13,000 from
`training/tokenizer/outputs/e2e_pb04_continuous_dense_35k_20260824/best.pt`. It gives HALO's
strongest zero-shot point estimate, although the seven-dataset difference from fixed PB-04 is not
resolved statistically. For k>=1, the supported deployment rule is 1-NN over enrolled executions;
the learned retrieve-mix-vote readout is reported as an underperforming ablation.

PB-04 and all six external encoders were scored on the same sealed `adaptation_v1` manifest: seven
held-out datasets, five seeds, and execution-disjoint support/query sets. Aggregate tables,
per-dataset tables, checkpoint comparisons, and the exact protocol are maintained only in
[`results/RESULTS.md`](results/RESULTS.md) to prevent duplicate numbers from becoming stale.

### Retracted — do not cite
- the **49.5 "beats harnet"** evidence-decoder headline — retracted twice: first for eval-label text
  contamination plus eval-tuned hyperparameters, then again after the vocabulary fix;
- the **r = −0.973** seen-vs-unseen correlation — re-measured at −0.328, p = 0.47;
- the **learnable filterbank** as a contribution — measured inert; the gain was multiresolution.

### Also do not cite
- the **random-alias training objective** as part of the recipe — removed from the default on
  2026-08-22 (`--alias-episode-fraction` defaults to 0.0). It remains available behind the flag,
  with an optional curriculum (`--alias-warmup-steps` / `--alias-ramp-steps`).
- the **"Phase-B plateaus at 0.45 / the signal is exhausted"** verdict — falsified: it was a
  cosine learning-rate schedule collapsing by 6k steps. A 90k run reaches ≈0.51–0.54. But note the
  reciprocal caution: the 0.5424 "best @ 35k" is roughly what a max over 76 validation points of a
  flat plateau (mean 0.512, sd 0.0135) yields by chance, so **best-checkpoint scores carry a
  selection inflation of about +0.03**.

### The number trap
Add to the do-not-cite list: the Stage-2 pair **23.75 / 9.81** (yesterday's headline) and the
protocol-v4 triple **45.7 / 42.9 / 34.4**, which `APPLICATIONS.md` still asserted as current.
Any figure produced before the vocabulary fix (**59 labels**) is not comparable to the table above:
40.4, 42.7, 45.1, 46.1, 47.3, 49.5. **Never pair a number from `pretrain_native` with one from
`pretrain_fixed_mr`** — they are different encoders.

## `design/` — the contribution

**Thesis and framing**
- [**MOTIVATION.md**](design/MOTIVATION.md) — one language interface for unseen labels *and* unseen
  acquisition configs; open-set labels alone is table stakes (ConSE); channel count is not a claim.
- [**POSITIONING.md**](design/POSITIONING.md) — is this worth pursuing, what it is for, the k-curve
  and controlled-shift protocols. ⚠️ numbers pre-v4.
- [TEXT_CONDITIONING.md](design/TEXT_CONDITIONING.md) — the factored per-sensor identity
  (device + placement + modality) plus intra-sensor axis role, as implemented.

**The two phases, as built**
- [**COMPACT_EVIDENCE_ENGINE.md**](design/COMPACT_EVIDENCE_ENGINE.md) — the implemented PB-04
  retrieve-mix-vote ablation. The promoted deployment path uses the same PB-04 encoder with direct
  1-NN enrollment because the learned correction reduces held-out performance.
- [**DESIGN_AUDIT_20260821.md**](design/DESIGN_AUDIT_20260821.md) — the stage-by-stage verification
  record: what is proven by test or measurement, what is an open risk, and the four methodology
  rules (each of which was violated once, at cost).
- [DESIGN_OF_RECORD.md](design/DESIGN_OF_RECORD.md) — **Phase A only.** Its Phase-B half (the
  admissibility gate, `cosine/τ + log(admissibility)`, the closed-form Stage-1 predictor,
  `sensor_bias` in the trunk) describes components the code no longer has, and its *thesis*
  paragraph is built on the admissibility claim the gate was meant to instantiate. Bannered.
- [**PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md**](design/PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md) —
  Phase-A record and the handoff contract. Commands live in
  [`training/tokenizer/README.md`](../training/tokenizer/README.md).
- [PHASE_B_TRAINING_INTENT.md](design/PHASE_B_TRAINING_INTENT.md) — Phase-B *motivation* (memory
  as an adaptation mechanism, and the standard of evidence it demands). ⚠️ **its mechanism sections
  are superseded** — required artifacts, prediction rule, and alias handling all describe the
  retired gate design. Bannered.
- [**PHASE_B_TRAINING_STATUS.md**](results/PHASE_B_TRAINING_STATUS.md) — the current readiness and
  experiment ledger. (There is no longer an "active admissibility design" — that mechanism was
  removed; see the compact engine above.)
- [PHASE_B_STEP0_CONTROL.md](results/PHASE_B_STEP0_CONTROL.md) — ⚠️ **conclusions reversed.**
  Measured on the 2026-08-09 trainer, which no longer exists; "training destroys zero-shot" and
  "prototype/ridge beat both arms everywhere" are both contradicted by the mixer and the compact
  engine. Bannered; quote no number from it.
- [AUGMENTATIONS.md](design/AUGMENTATIONS.md) — augmentation policy and the told-vs-not-told
  experiment. ⚠️ **that experiment has never been run** (its closest proxy, the 2026-08-11 parity
  gate, came back inert), and the doc's "active configuration" table misreports the defaults.

**Planning and open issues**
- [REMEDIATION_PLAN.md](design/REMEDIATION_PLAN.md) — historical July audit ledger. Its file paths
  and status markers are not the current implementation contract; use the Phase-A and Phase-B
  records above for current readiness.
- [DATA_SCALING_PLAN.md](design/DATA_SCALING_PLAN.md) — what corpus is reachable and the case for
  broadening.

**Literature**
- [EVIDENCE_ENGINE_RESEARCH.md](design/EVIDENCE_ENGINE_RESEARCH.md) — dated July 2026 literature
  synthesis and decision input. It is not a design or configuration contract; current motivation
  and decisions live only in `MOTIVATION.md`, `POSITIONING.md`, and `PHASE_B_TRAINING_INTENT.md`.
- [LITERATURE_HAR_LLM_2026-08.md](design/LITERATURE_HAR_LLM_2026-08.md) — August 2026 sweep of the
  HAR + LLM/foundation-model literature. Same status: input, not contract.

**Proposed — not decided, not built.** Do not describe these as part of the system.
- [APPLICATIONS.md](design/APPLICATIONS.md) — downstream applications. ⚠️ **its §0 premise is
  retracted and its conclusion inverted**: it argues from protocol-v4 numbers that we lose on
  accuracy, which is no longer true for the enrollment regime it is about. Bannered.
- [LANGUAGE_HIERARCHY.md](design/LANGUAGE_HIERARCHY.md) — language at every level; a second act.
- [**CONTINUOUS_KERNEL_FRONTEND.md**](design/CONTINUOUS_KERNEL_FRONTEND.md) — experimental CNN whose
  kernels are continuous curves in **real time** (seconds), sampled at the recording rate. A
  standalone implementation and focused tests exist, but it is not wired into the encoder or any
  trainer. Its decisive test remains cross-rate transfer rather than headline accuracy.

## `data/` — the corpus
- [DATA_PIPELINE.md](data/DATA_PIPELINE.md) — source → curate → unit→g → resample → window → grids.
- [DATA_HETEROGENEITY.md](data/DATA_HETEROGENEITY.md) — per-dataset normalization decisions and why.
- [DATASET_EXPANSION_2026-08.md](data/DATASET_EXPANSION_2026-08.md) — implemented 2026-08 expansion:
  six default training additions, KneE-PAD as an explicit short/stress source, and three held-out
  evaluation sources, with verified converter decisions and the matched-versus-expanded rule.
- [DATASET_EXPANSION_AUDIT_2026-08-11.md](data/DATASET_EXPANSION_AUDIT_2026-08-11.md) — the converter
  audit behind that expansion. Every finding is fixed; read it before trusting any enrollment or
  simultaneity claim about the ten new sources.

## `results/` — the measured record
- [**RESULTS.md**](results/RESULTS.md) — **the current headline**: PB-04-CK-DENSE zero-shot and 1-NN
  enrollment against the complete external-model roster, all matched controls, every k, and
  per-dataset tables.
- [ADAPTATION_TABLE_20260822.md](results/ADAPTATION_TABLE_20260822.md) — historical August 22 table;
  retained for auditability and explicitly superseded by `RESULTS.md`.
- [EVAL_HARNESS_AUDIT_20260822.md](results/EVAL_HARNESS_AUDIT_20260822.md) — verification that the
  eval path is sound and that every baseline is used as its developers intended. Two deviations
  found and fixed in code (LiMU-BERT accel scale; UniMTS label text); LiMU-BERT needs a re-pretrain
  before its row is valid again, and an adapter guard enforces that.
- [PHASE_B_DIAGNOSIS_20260820.md](results/PHASE_B_DIAGNOSIS_20260820.md) — why Phase-B training
  looked stuck. Its **mechanism** findings stand (retrieval ranks by acquisition configuration
  ×7.0; same-activity/different-device support rows at the 39th percentile; names-vs-signals
  r = 0.11). Its **"more steps do not help"** conclusion was later falsified by the 90k run.
- [PHASE_B_TRAINING_STATUS.md](results/PHASE_B_TRAINING_STATUS.md) and
  [PHASE_B_STEP0_CONTROL.md](results/PHASE_B_STEP0_CONTROL.md) — the parked relational-decoder
  history and the step-0 control that retired its headline. Also linked from `design/` above.

## `baselines/` — who we compare against
- [BASELINES.md](baselines/BASELINES.md) — roster, verified input contracts, frozen-vs-self-train.
  ⚠️ two contract statements were corrected on 2026-08-22 — see
  [`results/EVAL_HARNESS_AUDIT_20260822.md`](results/EVAL_HARNESS_AUDIT_20260822.md).
- [BASELINE_FAIRNESS_POLICY.md](baselines/BASELINE_FAIRNESS_POLICY.md) — the treatment contract.
  ⚠️ its "identical 6-channel 60 Hz tensor" invariant describes the design, **not** the executed
  path (scoring runs `non_harmonised`); see the correction in its §2.

## [`archive/`](archive/README.md) — superseded records
Ten documents that were authoritative when written and are not now: the pre-consolidation evidence
engine design and findings, two superseded results snapshots, three superseded preflight/quality
records, and the falsified learnable-tokenizer arm. **Nothing there describes the current system.**
Each carries a banner naming its live replacement.

## Parked on branch `pose-pretext-exploration` (not on main)
- `POSE_PRETEXT_LITERATURE.md` — IMU→pose pretext. **Killed by literature**: the config-invariance
  premise is backwards, and IMUCoCo (UIST 2025) already published the mechanism.
- `ENROLLMENT_BY_DEMONSTRATION.md` — repetition-mined enrollment. Alive but a larger pivot.

## Conventions
- One concern per folder; add new docs to the matching subfolder **and link them here**.
- Facts verified against papers or code carry a date.
- A doc whose central claim is falsified gets a banner and **moves to `archive/`** — kept, not
  deleted, because the record of what we believed and why is part of the work. What changed in the
  2026-08-08 consolidation is that stale docs no longer sit beside live ones.
- Current zero-shot and enrollment numbers live in the table above and in `results/RESULTS.md`,
  generated from `eval/adaptation_results/`. The
  `eval/results/` store is protocol-v4 and stale for every model. Generated output READMEs are
  artifact indexes, not competing reports.
- A comparison is only citable if every row shares one manifest fingerprint. Mixing protocol
  versions is rejected by `eval/assemble_table.py` by design; do not work around it.
- Cross-session context lives in the memory files (`~/.claude/.../memory/halo-*.md`).
