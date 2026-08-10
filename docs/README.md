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
> 5. [**results/PHASE_B_TRAINING_STATUS.md**](results/PHASE_B_TRAINING_STATUS.md) — the current
>    Phase-B run history, matched tables, confirmed defects, and next-run requirements.
>
> Then take every number from Results below, and nothing at all from
> [`archive/`](archive/README.md).

## Zero-Shot Baseline Results

Generated from `eval/results/` by `python -m eval.assemble_table`, which refuses to assemble a
mixed-protocol table. **56 cells** (8 models × 7 zero-shot test sets), all **protocol v4 (93 labels)**,
as of 2026-08-06.

| harnet | halo_evidence | crosshar | unimts | halo | limubert | imagebind | normwear |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **45.7** | 42.9 | 42.8 | 34.7 | 34.4 | 32.2 | 11.4 | 5.1 |

**harnet (frozen UK-Biobank) is ahead in this completed zero-shot baseline protocol.** These numbers
precede the current Phase-B adaptation run and must not be mixed with its enrollment tables. The
current Phase-B empirical status is owned by
[`results/PHASE_B_TRAINING_STATUS.md`](results/PHASE_B_TRAINING_STATUS.md).

### Retracted — do not cite
- the **49.5 "beats harnet"** evidence-decoder headline — retracted twice: first for eval-label text
  contamination plus eval-tuned hyperparameters, then again after the vocabulary fix;
- the **r = −0.973** seen-vs-unseen correlation — re-measured at −0.328, p = 0.47;
- the **learnable filterbank** as a contribution — measured inert; the gain was multiresolution.

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
- [**PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md**](design/PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md) —
  Phase-A record and the handoff contract. Commands live in
  [`training/tokenizer/README.md`](../training/tokenizer/README.md).
- [**PHASE_B_TRAINING_INTENT.md**](design/PHASE_B_TRAINING_INTENT.md) — the sole Phase-B motivation
  and training contract. Commands live in
  [`training/evidence/README.md`](../training/evidence/README.md).
- [**PHASE_B_TRAINING_STATUS.md**](results/PHASE_B_TRAINING_STATUS.md) — historical results from the
  superseded vote/soft-retrieval run, retained as the empirical motivation for the relational
  design. Its §4
  unweighted three-cell mean is retired; see the step-0 control below for the weighting that
  replaces it.
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

**Proposed — not decided, not built.** Do not describe these as part of the system.
- [APPLICATIONS.md](design/APPLICATIONS.md) — what a HAR foundation model could make possible
  downstream, chosen so the pitch does not rest on accuracy we do not have.
- [LANGUAGE_HIERARCHY.md](design/LANGUAGE_HIERARCHY.md) — language at every level; a second act.

## `data/` — the corpus
- [DATA_PIPELINE.md](data/DATA_PIPELINE.md) — source → curate → unit→g → resample → window → grids.
- [DATA_HETEROGENEITY.md](data/DATA_HETEROGENEITY.md) — per-dataset normalization decisions and why.
- [DATASET_EXPANSION_2026-08.md](data/DATASET_EXPANSION_2026-08.md) — **proposal, nothing acquired.**
  Candidate training/eval datasets for the rehabilitation-tracking framing, with verified access and
  disk cost. Its §6 records where it conflicts with `design/DATA_SCALING_PLAN.md`'s frozen-corpus rule.

## `baselines/` — who we compare against
- [BASELINES.md](baselines/BASELINES.md) — roster, verified input contracts, frozen-vs-self-train.
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
- Zero-shot baseline numbers live in the table above, generated from `eval/results/`. Phase-B
  adaptation numbers live only in `results/PHASE_B_TRAINING_STATUS.md`; generated output READMEs are
  artifact indexes, not competing reports.
- Cross-session context lives in the memory files (`~/.claude/.../memory/halo-*.md`).
