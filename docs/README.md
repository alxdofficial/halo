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
> 5. [**results/ADAPTATION_TABLE_20260822.md**](results/ADAPTATION_TABLE_20260822.md) — **the
>    current headline result.** The compact evidence engine against all seven baselines on the
>    shared manifest, every method at every k. Supersedes the 2026-08-19 table.
> 6. [**results/PHASE_B_TRAINING_STATUS.md**](results/PHASE_B_TRAINING_STATUS.md) — the current
>    Phase-B run history, matched tables, confirmed defects, and next-run requirements.
>
> Then take every number from Results below, and nothing at all from
> [`archive/`](archive/README.md).

## Current Matched Results

**2026-08-22.** The current model is the **compact evidence engine** (`halo_compact`): filterbank
tokenizer → 3-layer temporal trunk (d=128, one row per (patch, sensor)) → **plain cosine**
retrieval → hard top-64 → evidence mixer → text-cosine vote. **1,126,024 trainable parameters.**
Checkpoint `training/tokenizer/outputs/long_4h_20260821/best.pt`. Scored on the shared
`adaptation_v1` manifest (61 cells · 7 held-out datasets · 5 seeds · execution-disjoint
support/query), fingerprint-identical to every baseline row. Full table:
[`results/ADAPTATION_TABLE_20260822.md`](results/ADAPTATION_TABLE_20260822.md).

### Zero-shot, k = 0 — each model's own shipped mechanism

| model | ordinary macro F1 | specialized-novel macro F1 |
|---|---:|---:|
| CrossHAR + ConSE | **37.01** | 10.88 |
| **HALO compact engine** | 36.95 | 8.75 |
| HARNet + ConSE | 33.82 | 11.40 |
| UniMTS (own text tower) | 32.70 | **19.24** |
| LiMU-BERT + ConSE | 27.60 | 9.11 |
| ImageBind (own text tower) | 11.38 | 8.15 |
| NormWear (L1 text match) | 5.08 | 3.58 |

HALO's row is the only one produced with **no fitted head** — the engine's native
retrieve→mix→vote rule. The specialized-novel 8.75 is weak and expected: the memory bank holds no
clinical motions and their names do not project onto the training vocabulary. That is precisely
the case enrollment exists for — one enrolled example takes it to ≈43.

### Enrollment, k ≥ 1 — and the nuance that matters

**HALO is best in 35 of 40 method×k enrollment columns**, including *every* specialized-novel
column at every k, at d=128 (baselines 512–2048). The five it loses are all `ordinary` at high k
(nearest k=4/8/16, linear_head k=8/16) — everyday activities with many labelled examples, where a
larger frozen feature has room to be fitted. Per-dataset margins are positive in 34/44
comparisons against the strongest baseline per column, typically +1 to +7 F1.

> ⚠️ **Read this before citing the 35/40.** Those enrollment columns fit *generic* heads
> (nearest/prototype/ridge/linear_head) on each model's **frozen features**, with identical code
> for every model. So they demonstrate that **our encoder's representation** is strong — they do
> **not** demonstrate that our *engine* works. The engine's own mechanism is scored in exactly one
> place: zero-shot k=0. Scoring the engine at k ≥ 1 (enrolled rows placed in the memory bank,
> native rule, versus ridge on our own features) is **not yet done** and is the decisive
> outstanding experiment.

Caveats bound to this table: `limubert` rows predate the 2026-08-22 accel-scale fix
([`results/EVAL_HARNESS_AUDIT_20260822.md`](results/EVAL_HARNESS_AUDIT_20260822.md) F1) and may
understate it — its backbone re-pretrain is pending; `unimts` zero-shot predates the label-text
ensemble fix (F2). Enrollment methods read no label text, so those columns are unaffected. The
older protocol-v4 (93-label) zero-shot table is stale **for every model** and is indexed in
[`results/RESULTS.md`](results/RESULTS.md).

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
- [**DESIGN_OF_RECORD.md**](design/DESIGN_OF_RECORD.md) — the current architecture decision record:
  the three per-(patch, sensor) vectors, the front end, `sensor_bias`, the admissibility gate and
  its three guards, the Phase-B prediction rule, and the build ledger with what is deleted and what
  is still pending. When this and an older design doc disagree, this one wins.
- [**PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md**](design/PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md) —
  Phase-A record and the handoff contract. Commands live in
  [`training/tokenizer/README.md`](../training/tokenizer/README.md).
- [**PHASE_B_TRAINING_INTENT.md**](design/PHASE_B_TRAINING_INTENT.md) — the sole Phase-B motivation
  and training contract. Commands live in
  [`training/evidence/README.md`](../training/evidence/README.md).
- [**PHASE_B_TRAINING_STATUS.md**](results/PHASE_B_TRAINING_STATUS.md) — the authoritative current
  readiness and experiment ledger. Its completed tables are explicitly marked as parked relational
  experiments and do not validate the active admissibility design.
- [**PHASE_B_STEP0_CONTROL.md**](results/PHASE_B_STEP0_CONTROL.md) — what Phase-B training actually
  buys, measured against the system at initialisation. Training pushes zero-shot below its chance
  floor by destroying a ConSE-like bridge the untrained mechanism already had; at k=2 cross-subject
  it is a wash, the decoder's +1.25 exactly cancelling the retriever's −1.27.
- [AUGMENTATIONS.md](design/AUGMENTATIONS.md) — augmentation policy and the told-vs-not-told experiment.

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
- [APPLICATIONS.md](design/APPLICATIONS.md) — what a HAR foundation model could make possible
  downstream, chosen so the pitch does not rest on accuracy we do not have.
- [LANGUAGE_HIERARCHY.md](design/LANGUAGE_HIERARCHY.md) — language at every level; a second act.

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
- [**ADAPTATION_TABLE_20260822.md**](results/ADAPTATION_TABLE_20260822.md) — **the current
  headline**: compact engine vs seven baselines, all methods × k, both regimes, plus per-dataset
  zero-shot. Supersedes `ADAPTATION_TABLE_20260819.md`.
- [RESULTS.md](results/RESULTS.md) — project-wide results index; carries the headline table inline.
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
- Zero-shot and enrollment numbers live in the table above and in
  `results/ADAPTATION_TABLE_20260822.md`, generated from `eval/adaptation_results/`. The
  `eval/results/` store is protocol-v4 and stale for every model. Generated output READMEs are
  artifact indexes, not competing reports.
- A comparison is only citable if every row shares one manifest fingerprint. Mixing protocol
  versions is rejected by `eval/assemble_table.py` by design; do not work around it.
- Cross-session context lives in the memory files (`~/.claude/.../memory/halo-*.md`).
