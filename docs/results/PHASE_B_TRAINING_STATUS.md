# Phase-B Training Status

> **Historical empirical record for the superseded vote/soft-retrieval Phase-B run.** This document
> preserves completed-run findings, comparison tables, and the defects that motivated the current
> relational learned-query design. It is not configuration guidance, and none of its checkpoints
> are compatible with the current trainer. The
> normative motivation and architecture contract remain in
> [`PHASE_B_TRAINING_INTENT.md`](../design/PHASE_B_TRAINING_INTENT.md). Generated tensors, telemetry,
> and per-cell results live under `training/evidence/outputs/diagnostics/` and are indexed here rather
> than interpreted in additional competing reports.
>
> Last updated: 2026-08-09. The sealed external test roster has **not** been consumed.
> No full run of the current relational learned-query design has been completed yet.

## 1. Current Verdict

Phase B is a **development-level mild success**, not a finished result and not ready for a final
claim. The clean replay demonstrates genuine use of enrolled examples, but also exposes a retrieval
bottleneck, a validation-selector failure, and destructive specialization after the best external
development checkpoint.

The original claim that training peaked at step 200 was wrong. That run crossed an episode-builder
code change during resume and compared scores from two different validation protocols. Under one
continuous protocol, useful adaptation continued through step 1000 and the internal metric peaked at
step 1800.

The evidence engine is not trapped in an early local minimum:

- external neutral-alias adaptation improves from 60.03 F1 at step 200 to 66.73 at step 1000;
- decoder and retriever parameters continue moving substantially after step 200;
- training canary accuracy continues rising through the run;
- gradients remain finite, no clipping events occur, and the frozen tokenizer does not drift.

The actual failure is that the optimization target and training distribution permit the model to
keep improving internally while losing the external adaptation behavior we care about.

## 2. What Phase-B Training Does

1. The selected Phase-A encoder converts each sensor window into contextualized patch embeddings.
2. `build_memory.py` stores detached patch embeddings with label, dataset, subject, configuration,
   event, window, sensor, timing, and source-row metadata.
3. Each optimizer step samples one episodic task: a candidate-label set, one support-memory overlay,
   a query batch, a label presentation mode, and a clean or augmented physical view.
4. The learned multi-subspace retriever performs hard top-k selection from the active memory view.
   A scaled soft all-memory path supplies backward-only gradients to non-selected rows.
5. The evidence decoder receives query patches, retrieved patches, retrieved labels, support roles,
   candidate-label tokens, structural metadata, and retrieval weights.
6. Evidence-set attention, evidence-label refinement, candidate refinement, and retrieval-prior
   reweighting produce candidate logits.
7. The sole task loss is candidate-set cross-entropy on answerable episodes.

The four episode regimes are cycled evenly:

- coherent semantic zero support;
- ordinary few support;
- genuine cross-subject few support;
- same-subject enrollment, with augmented views sharing one virtual-subject style across support and
  query.

Supported episodes sample `k` from `1,2,4,8`, candidate counts from `4,8,12,16`, coherent or
episode-local neutral label text, and clean or augmented physical views.

## 3. Completed Clean Replay

| setting | value |
|---|---:|
| Phase-A mode | frozen |
| optimizer | AdamW |
| steps | 3000 |
| query batch | 64 |
| peak learning rate | `5e-4` |
| warmup | 100 steps |
| schedule | cosine decay to zero |
| weight decay | `0.01` |
| decoder | 3 layers, 4 heads |
| evidence budget | 64 |
| seed | `20260725` |
| internal validation | every 200 steps |

The replay was stopped at step 1000 only to preserve weights, then resumed from the exact trainer
state without code or protocol changes. Evaluator-ready weights were retained at steps 100, 200,
400, 600, 800, 1000, and every 200 steps thereafter.

### Fixed external-development trajectory

This checkpoint curve uses neutral episode-local aliases and fixed development episodes:

- MotionSense and RealWorld: genuine same-subject enrollment, `k=1/2`;
- Shoaib: genuine cross-subject support, `k=1/2`;
- six equally weighted dataset/regime/support cells.

| step | full evidence engine F1 | identity retrieval-vote F1 | decoder gain |
|---:|---:|---:|---:|
| 100 | 61.73 | 58.67 | +3.05 |
| 200 | 60.03 | 59.21 | +0.82 |
| 400 | 62.43 | 58.66 | +3.77 |
| 600 | 63.00 | 59.40 | +3.60 |
| 800 | 63.43 | 59.14 | +4.29 |
| **1000** | **66.73** | **61.39** | **+5.34** |
| 1200 | 60.06 | 60.76 | -0.71 |
| 1800 | 59.74 | 59.74 | -0.01 |
| 3000 | 59.98 | 60.65 | -0.67 |

The current internal selector instead chooses step 1800 at internal macro balanced accuracy 0.3472.
Across the ten checkpoints with both measurements, internal macro balanced accuracy is negatively
associated with external F1 (Spearman rho `-0.709`, nominal `p=0.0217`). This is diagnostic with
only ten points, but sufficient to reject the current internal score as a checkpoint selector.

## 4. Matched Method Comparison

The following table uses the clean step-1000 weights, the exact same query cohorts, and identical
per-subject candidate sets. Values are the unweighted mean of three cell-level macro F1 scores over
2,953 query windows. “Full evidence engine” is HALO's primary Phase-B path; every other column is a
control.

| label presentation | support per candidate | full evidence engine | identity retrieval vote | ConSE | prototype | ridge |
|---|---:|---:|---:|---:|---:|---:|
| coherent activity names | 0 | 30.43 | 32.86 | 29.33 | N/A | N/A |
| coherent activity names | 1 | 57.13 | 56.77 | 29.33 | 74.23 | 74.52 |
| coherent activity names | 2 | 64.29 | 64.08 | 29.33 | 79.76 | 80.25 |
| neutral episode-local aliases | 1 | 61.68 | 57.54 | N/A | 74.23 | 74.52 |
| neutral episode-local aliases | 2 | 71.78 | 65.25 | N/A | 79.76 | 80.25 |

Definitions:

- **Identity retrieval vote:** the same learned retriever and top-k evidence as the full engine, but
  candidate scores are direct retrieval-weighted votes from stored evidence labels. This is the
  weighted-kNN-like control that isolates the learned evidence decoder.
- **ConSE:** the current frozen Phase-A encoder plus a newly refitted 93-label classifier and semantic
  ConSE bridge, restricted to the exact episode candidate set. It does not consume support and is
  therefore constant across `k`. It is undefined for arbitrary neutral aliases.
- **Prototype:** normalized mean of enrolled support embeddings per candidate followed by cosine
  classification.
- **Ridge:** a deterministic closed-form L2-regularized linear classifier fitted on enrolled support.

### Direct support-use controls

| label presentation | support | full engine | support removed | support labels shuffled |
|---|---:|---:|---:|---:|
| coherent | 1 | 57.13 | 30.43 | 18.86 |
| coherent | 2 | 64.29 | 30.43 | 16.23 |
| neutral aliases | 1 | 61.68 | 30.68 | 18.59 |
| neutral aliases | 2 | 71.78 | 30.68 | 12.99 |

Removing support eliminates the gain, and assigning incorrect labels to support is more damaging.
The engine is demonstrably reading enrolled examples and their labels.

## 5. Mechanical Status and Remaining Defects

### B1. Resume did not bind the validation protocol or code identity — fixed 2026-08-09

Fixed validation canaries are constructed before resume loading in
`training/evidence/train_patch_decoder.py`. Previously, resume restored `best`, `best_step`, and
`best_state` while validating only the bank fingerprint and CLI configuration. This is how the
original step-200 checkpoint retained a score from an incompatible validation protocol.

Trainer-state schema v2 now stores the training regime, a SHA-256 fingerprint of every behavior-
defining Phase-B source file, and a structured SHA-256 fingerprint of the complete deterministically
rebuilt train/validation canary state. Resume requires exact equality before restoring the prior best
state. Legacy states are intentionally rejected as unsafe.

Verified by the two-step smoke, v2 resume guard, and fingerprint unit tests.

### B2. Checkpoint selection optimizes the wrong metric — high

`training/evidence/train_patch_decoder.py:2566-2572` selects solely on internal
`macro_cell_ba`. That mixture includes zero support, `k=4/8`, coherent labels, synthetic episode
views, and internal folds. It selected step 1800 even though fixed real-subject `k=1/2` adaptation
peaked at step 1000.

Required correction: define a development selector centered on absolute `k=1/2` adaptation and
gain over identity, with no catastrophic per-domain regression. Keep prototype/ridge as reporting
guards. Freeze the development procedure before touching the sealed test roster.

### B3. Intermediate evaluator checkpoints were not retained — fixed 2026-08-09

Every validation now writes a complete evaluator-ready artifact to
`<output-stem>.milestones/step_NNNNNN.pt`, including model state, protocol metadata, source/canary
fingerprints, exact checkpoint step, and contemporaneous metrics. The `.last.pt` artifact remains the
single resumable optimizer/RNG state. The final output remains the internally selected predictor.

The smoke verified separate step-1 and step-2 milestone predictors and their metadata.

### B4. Exact soft-gradient ablation crashed telemetry — fixed 2026-08-09

Soft-path probes, telemetry fields, and console fields are now conditional on the estimator having
run. A two-step exact-zero smoke completed without producing misleading placeholder soft metrics.

### B5. The frozen-mode default batch does not reflect the profiled 4090 launch — low

**Fixed 2026-08-09.** The ambiguous `--batch` option was removed. The live trainer defaults to eight
independent episodes with eight queries each and records both dimensions in its run contract.
A two-step real-bank 8x8 smoke on the local RTX 4090 measured 0.55-0.82 seconds per optimizer step
and 1.29 GiB peak allocated VRAM, excluding startup and periodic validation. This is a launch-path
measurement, not a stable full-run throughput estimate.

## 6. Training-Design Constraints Preventing Full Learning

These are evidence-backed diagnoses or focused hypotheses, not confirmed code-corruption bugs.

### 6.1 Query batch size is not episode diversity — highest-priority design issue

One optimizer step constructs one candidate set and one support-memory overlay, then draws the full
query batch from that shared episode. Batch 64 therefore means 64 correlated query windows, not 64
independent adaptation tasks. At step 1000 the model had seen 64,000 queries but only 1,000 support
draws, candidate sets, and memory overlays.

This was enough to learn shortcuts and domain-specific refinements quickly, while providing much
less task diversity than the query count suggested. **Fixed 2026-08-09:** each live optimizer update
now contains eight independently constructed episodes with eight queries apiece. Candidate sets,
support overlays, aliases, distractors, and physical views are episode-local.

Increasing the immutable archive or active-memory size does not address this problem: it still
produces one task per optimizer step. With fixed top-k it can reduce the probability that enrolled
support is retrieved. Memory size and episode count are therefore separate axes; the next experiment
should increase independently constructed episodes per step while keeping memory capacity fixed.

### 6.2 Retrieval is the primary capability bottleneck

At step 1000, positive-support recall at top-k is only about 0.353. Identity voting reaches 61.39 F1,
the full decoder reaches 66.73, while direct prototype and ridge controls reach 76.99 and 77.39 on
the same neutral-alias cells. The representation contains useful information, but the evidence
engine often cannot access the enrolled item that a direct support method receives automatically.

The support-lane proposal was considered and rejected because it creates a second manually privileged
retrieval path. The live design keeps retrieval entirely learned and query driven. Exact enrolled
rows instead supervise a multiple-instance boundary objective over eligible memory: the best
true-support patch is promoted above the final evidence-budget cutoff without adding it to the
forward roster or treating every background analogue as irrelevant.

This was also a credit-assignment problem. The historical all-memory soft backward estimator was
biased relative to the hard forward computation and often poorly aligned with it. That estimator has
been removed; support-boundary supervision is the sole missed-known-positive gradient path.

### 6.3 The learning rate remains aggressive after the external optimum

The run uses `5e-4`, a 100-step warmup, and a 3000-step cosine schedule. Learning rate is still about
`3.9e-4` at step 1000 and `3.4e-4` at step 1200. During that interval external F1 falls 6.68 points,
and several projection/refinement components move by 10-27% of their prior norm. This is consistent
with overshoot or rapid specialization, not convergence into a stable local minimum.

The next-run default is now `2e-4` with a 300-step warmup and the same 3000-step cosine horizon. This
is a predeclared correction based on the historical trajectory, not evidence that `2e-4` is optimal;
matched short development runs must still test sensitivity before a sealed evaluation.

### 6.4 Training candidate-set sizes miss important deployment regimes

The historical run sampled `4,8,12,16` candidates. The matched external enrollment cells contain two
RealWorld candidates, three MotionSense candidates, and seven Shoaib candidates per subject. The
architecture supports these counts, but it never trains directly on the common two- and three-way
decision regimes.

**Fixed 2026-08-09.** Candidate counts now cover `2,3,4,8,12,16`, preserving large tasks while matching
the common two- and three-way external cells.

### 6.5 Coherent semantics and support adaptation interfere

Neutral aliases outperform coherent labels at both `k=1` and `k=2`. With neutral aliases, unrelated
canonical background labels have little textual similarity to the candidates, so enrolled support
is easy to identify. With coherent labels, semantically related global-memory distractors can vote
for candidates and the decoder does not reliably suppress them despite receiving an explicit support
role.

This is evidence that the model has learned episode-local label binding better than semantic-plus-
support reasoning. Before adding losses, measure support-versus-background attention and pooling by
label mode. The live response is greater independent-episode diversity plus direct support-boundary
supervision for learned retrieval, rather than another auxiliary classifier or a privileged lane.

### 6.6 Decoder changes are not bounded after identity initialization

The decoder is exactly identity-like only at initialization. Evidence text, candidate text, and
pooling weights can later move without a per-query fallback constraint. This allows the learned path
to reduce a strong identity result, as seen on RealWorld after step 1000.

First fix selection and retrieval. If destructive overrides persist, use an explicit residual over
identity logits with a small, observable per-query gate or bounded correction. Do not add this merely
to improve the current development table; test it as a predeclared do-no-harm ablation.

### 6.7 Historical hard-forward/soft-backward retrieval was a biased estimator

In the superseded run, the hard and soft retriever gradients were frequently weakly or negatively
aligned, and hard top-k retained little all-memory soft mass. A matched near-zero-soft run also
performed worse (61.81 versus 66.73 external F1 at step 1000), so that experiment did not isolate a
clean estimator benefit. The current trainer removes this backward-only path rather than carrying a
biased computation that cannot be explained as the forward model.

Differentiable top-k methods exist, including optimal-transport and sparse convex relaxations, but
they introduce a second approximation and additional compute. They remain a matched future ablation,
not default machinery. See [Xie et al. 2020](https://proceedings.neurips.cc/paper/2020/hash/ec24a54d62ce57ba93a531b460fa8d18-Abstract.html)
and [Sander et al. 2023](https://proceedings.mlr.press/v202/sander23a.html).

### 6.8 Retrieval credit-assignment audit after the redesign

A fixed-memory replay across the historical step-0 through step-3000 retrievers used 24 held query
windows, four declared support executions per query, 14,905 active patches, and the same final active
memory for every checkpoint. It found:

- final roster Jaccard versus initialization fell to 0.406, so retrieval did explore rather than
  remaining locked to initialization;
- consecutive-checkpoint Jaccard rose to 0.980 at step 3000, showing expected late freezing as cosine
  learning rate reached zero;
- median best-support rank improved from 142 to 13 by step 2400, but final support recall stayed near
  0.29-0.33 and support mass near 0.02;
- the historical all-positive loss therefore moved the projection without reliably improving access.

The defect was mathematical: every query patch/head pulled every patch from every support execution,
although one activity execution contains distinct temporal phases. The live v10 objective instead
uses a set-to-set maximum and the final evidence-budget boundary. A 30-step real-bank diagnostic of
this implementation found task and weighted-support retriever gradients of comparable scale (mean
norms 0.160 and 0.091) with mean cosine 0.032, so they are not fighting. Candidate CE promoted
selected true-support scores 70.7% of the time and selected background scores 47.3% of the time,
confirming that non-support analogues can receive favorable task credit once selected. Fixed-canary
rosters changed for 43.8% of queries while retaining 0.977 mean overlap at step 30.

This is a mechanism smoke test, not evidence of final quality. The next full development run must show
that fixed-canary support rank/recall, identity gain, and external development adaptation improve
together. Dense-retrieval precedent supports explicit known-positive supervision and hard-negative
boundaries ([DPR](https://arxiv.org/abs/2004.04906)); latent retrievers such as
[REALM](https://arxiv.org/abs/2002.08909) and [RAG](https://arxiv.org/abs/2005.11401) still optimize
only a bounded retrieved set. None of those results implies that a soft all-memory surrogate is
automatically faithful for this IMU evidence mechanism.

## 7. Recommended Next Training Sequence

1. Freeze a development checkpoint selector aligned to `k=1/2` adaptation and identity gain.
2. The multi-episode batch and two-/three-candidate curriculum are now implemented.
3. ~~Keep retrieval learned and query driven; measure the support-boundary objective rather than
   adding a manually privileged support allocation.~~ **Superseded 2026-08-10:** the support-boundary
   objective was removed; the candidate loss is now the only objective. Retrieval remains learned and
   query driven.
4. Run short, matched development experiments over learning rate and decay. Preserve the same
   checkpoint grid and at least two seeds.
5. Select one recipe using fixed development `k=1/2` coherent and neutral-alias curves, identity
   gain, support-removal/shuffle controls, and per-domain regression guards.
6. Only then run a full trajectory and evaluate the sealed test roster once.

No additional pretraining loss is currently justified. The prototype/ridge ceiling shows that the
frozen representation already contains substantially more usable information than Phase B extracts;
the immediate problem is episodic exposure, retrieval access, optimization, and selection.

## 8. Canonical Artifact Index

All paths below are relative to the repository root.

- Clean replay raw artifacts:
  `training/evidence/outputs/diagnostics/phase_b_20260808/clean_replay/`
- Aggregate checkpoint trajectory: `clean_replay/checkpoint_summary.csv`
- Internal trajectory: `clean_replay/internal_trajectory.csv`
- External per-cell trajectory: `clean_replay/external_alias_trajectory.csv`
- Matched comparison: `clean_replay/comparison_table.csv`
- Matched per-cell comparison: `clean_replay/comparison_table_by_cell.csv`
- Parameter trajectory: `clean_replay/parameter_trajectory.csv`
- Training-window telemetry summary: `clean_replay/training_windows.csv`
- Raw telemetry: `clean_replay/telemetry/` and `clean_replay/telemetry_resume/`
- Evaluator-ready checkpoints: `clean_replay/predictors/`
- Full trainer states: `clean_replay/checkpoints/`
- Step-1000 component ablations: `clean_replay/ablations/` and
  `clean_replay/step1000_output_path_ablations.csv`
- Current matched ConSE artifacts: `clean_replay/conse_current/`
- Original interrupted-run forensic artifacts:
  `training/evidence/outputs/diagnostics/phase_b_20260808/`
- Machine-readable provenance:
  `training/evidence/outputs/diagnostics/phase_b_20260808/MANIFEST.json`

Key weight identities:

| artifact | SHA-256 |
|---|---|
| externally best measured step-1000 predictor | `f8841f6037431a512367d89e4dee16267d22297abe0d206a0f9d6c27cb147ced` |
| internally selected step-1800 predictor | `26e6bc0090bef1224e656efd701fb8c15b3da7301bd8b2c1043e790e58d206c9` |
| final step-3000 trainer state | `ab47354df57ba5d39b723aa976e9695aca7ab383294f24c4fdc377d3f4717315` |
| current matched ConSE head | `97de10dca12331ee7280e28173ff66c0bad3a1e879dd5b37349fb444fc3c0810` |

## 9. Interpretation Boundaries

- These are development diagnostics, not final test estimates.
- Step 1000 was identified after examining the development trajectory.
- The matched comparison covers three datasets and one Phase-B seed.
- The original interrupted run is retained only to document the protocol failure.
- Prototype and ridge use pooled support embeddings and direct support access; their advantage is a
  capability ceiling and a mechanism diagnostic, not proof that the evidence engine can never close
  the gap.
- “Cross subject” in the Shoaib external cell means genuinely different recorded people. Synthetic
  virtual-subject styling is additional training augmentation, not the definition of that split.
