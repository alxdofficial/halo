# HALO docs

Organized into subfolders by concern.

> ## Read these three first, in this order
> 1. [**design/MOTIVATION.md**](design/MOTIVATION.md) — *why* HALO exists (the thesis).
> 2. [**design/POSITIONING.md**](design/POSITIONING.md) — whether the thesis is worth pursuing, what
>    the result would actually be **for**, and how to report it. Written 2026-07-21 after a hard look
>    at the numbers.
> 3. [**design/EVIDENCE_ENGINE_FINDINGS.md**](design/EVIDENCE_ENGINE_FINDINGS.md) (STATUS block) —
>    the authoritative current empirical position. **Any number in any other doc is subordinate to
>    this one.**
>
> ### ⚠️ State of results as of 2026-07-21
> **`eval/results/` is EMPTY and every headline number in these docs is provisional.** The split
> manifest changed (452 → 482 subjects, `hapt` cohort-aliased), the probe is now seeded, and cache
> fingerprints are now enforced — so all four ConSE heads are stale and must be refit **together**
> before any table is valid. Pre-fix numbers are preserved byte-identical in
> `eval/results_archive/2026-07-20_pre-vocab-fix/` (tag `results-pre-vocab-fix`).
>
> **Retracted claims — do not cite:** the 49.5 "beats harnet" evidence-decoder headline (twice: first
> for eval-label text contamination + eval-tuned hyperparameters, then again after the vocabulary
> fix); the `r = −0.973` seen-vs-unseen correlation (re-measured at −0.328, p = 0.47); the learnable
> filterbank as a contribution (measured **inert** — the gain was multiresolution).
>
> **⚠️ Never pair a number from `pretrain_native` with one from `pretrain_fixed_mr`.** They are
> different encoders. The valid same-encoder pairs are **84.3 / 39.6** (`pretrain_native`, supervised
> probe vs zero-shot ConSE) and **42.7 / 45.1 / 46.1** (`pretrain_fixed_mr`, ConSE vs untrained
> retrieval vs trained decoder). `RESULTS_V2.md` reports HALO at **40.4** because it uses
> `pretrain_native`.

## `design/` — the contribution: what we're building and why

**Thesis and positioning**
- [**MOTIVATION.md**](design/MOTIVATION.md) — one language interface for unseen labels *and* unseen
  acquisition configs; open-set-labels-alone is table stakes (ConSE); channel-count is not a claim.
- [**POSITIONING.md**](design/POSITIONING.md) — is zero-shot HAR worth pursuing, what the deliverable
  is *for* (operational properties, not accuracy), the k-curve, the controlled-shift protocol.
- [LANGUAGE_HIERARCHY.md](design/LANGUAGE_HIERARCHY.md) — second-act design: language at every level.
- [**TEXT_CONDITIONING.md**](design/TEXT_CONDITIONING.md) — factoring channel text into a
  per-sensor identity (device+placement+modality) + trivial intra-sensor role; current-code mapping
  and the fold-in plan. ⚠️ prior-art lane is crowded — see MOTIVATION §2b.

**Evidence engine** (read FINDINGS before the others)
- [**EVIDENCE_ENGINE_FINDINGS.md**](design/EVIDENCE_ENGINE_FINDINGS.md) — ⭐ current empirical truth,
  including the retractions and what replaced them.
- [EVIDENCE_ENGINE.md](design/EVIDENCE_ENGINE.md) — original design rationale (2026-07-13/14).
  Preserved as written; the empirical picture has since moved.
- [EVIDENCE_ENGINE_BUILD_PLAN.md](design/EVIDENCE_ENGINE_BUILD_PLAN.md) — milestone plan. Status
  lines are **stale**: much is built and several gates were later failed.
- [EVIDENCE_ENGINE_TIER2.md](design/EVIDENCE_ENGINE_TIER2.md) — Tier-2 architecture. Its T2.2/T2.3
  "GATE PASSED" claims are **retracted**; retained with strike-through as a record.
- [EVIDENCE_ENGINE_RESEARCH.md](design/EVIDENCE_ENGINE_RESEARCH.md) — 2026-07-14 literature synthesis.

**Pretraining / tokenizer**
- [AUGMENTATIONS.md](design/AUGMENTATIONS.md) — augmentation policy + the told-vs-not-told experiment.
- [LEARNABLE_TOKENIZER_ARM.md](design/LEARNABLE_TOKENIZER_ARM.md) — ⚠️ hypothesis **falsified**: the
  learnable filterbank is inert; multiresolution did the work.
- [NATIVE_PRETRAIN_PREFLIGHT.md](design/NATIVE_PRETRAIN_PREFLIGHT.md) — provenance for the
  `pretrain_native` run.
- [PIPELINE_A_PREFLIGHT.md](design/PIPELINE_A_PREFLIGHT.md) — older 8-dataset / 57-label / 60 Hz
  audit. **Superseded** by NATIVE_PRETRAIN_PREFLIGHT for corpus facts.
- [TOKENIZER_QUALITY.md](design/TOKENIZER_QUALITY.md) — quality battery on an **11k-step** encoder;
  predates the current checkpoints.

**Planning**
- [REMEDIATION_PLAN.md](design/REMEDIATION_PLAN.md) — every known correctness/fairness issue, with
  status. Includes the 2026-07-21 debug-sweep table (fixed vs still open).
- [DATA_SCALING_PLAN.md](design/DATA_SCALING_PLAN.md) — corpus is ~290–547 h reachable; the case for
  broadening.

## `data/` — the corpus and how it's built
- [DATA_PIPELINE.md](data/DATA_PIPELINE.md) — source → curate → unit→g → resample → window → grids.
- [DATA_HETEROGENEITY.md](data/DATA_HETEROGENEITY.md) — per-dataset normalization decisions and why.

## `baselines/` — who we compare against and how
- [BASELINES.md](baselines/BASELINES.md) — roster, verified input contracts, frozen-vs-self-train.
- [BASELINE_FAIRNESS_POLICY.md](baselines/BASELINE_FAIRNESS_POLICY.md) — treatment contract. ⚠️ its
  "identical 6-channel 60 Hz tensor" invariant describes the design, **not** the executed path
  (scoring runs `non_harmonised`); see the correction in §2.
- [RESULTS_V2.md](baselines/RESULTS_V2.md) — ⚠️ **superseded** pre-vocabulary-fix snapshot.
- [RESULTS_PRELIMINARY.md](baselines/RESULTS_PRELIMINARY.md) — ⚠️ **superseded**, older still.

## Parked directions (branch `pose-pretext-exploration`, not on main)
- `POSE_PRETEXT_LITERATURE.md` — IMU→pose as a pretext task. **Killed by literature**: the
  config-invariance premise is backwards, and IMUCoCo (UIST 2025) already published the mechanism.
- `ENROLLMENT_BY_DEMONSTRATION.md` — repetition-mined few-shot enrollment. Alive but a larger pivot;
  prior art unchecked.

## Conventions
- One concern per folder; add new docs to the matching subfolder **and link them here**.
- Facts verified against papers/code carry a date.
- A doc whose central claim is later falsified gets a **banner at the top**, and is kept rather than
  deleted — the record of what we believed and why is part of the work.
- Cross-session context lives in the memory files (`~/.claude/.../memory/halo-*.md`).
