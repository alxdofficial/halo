# Phase-B Training Intent

> **Canonical Phase-B motivation and executable contract. Read this before configuring, launching,
> or interpreting Phase-B training.**
>
> Status: implementation-aligned as of 2026-08-08. Phase A is complete; the first full run of this
> Phase-B recipe has not yet established its empirical claims.

This document owns both the reason Phase B exists and the constraints that make its training match
that reason. `training/evidence/README.md` owns commands. `PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md`
records the completed Phase-A recipe and the handoff between phases. The older evidence-engine
documents are historical research records, not configuration guidance.

## 1. The Core Thesis

**Phase B is not a classifier over the corpus vocabulary with retrieval attached. It learns to use
labeled memory as an adaptation mechanism.**

Corpus activity labels are examples from which the model learns how evidence should be used. They
are not a fixed deployment vocabulary. At deployment, a new movement name and a few labeled
executions can be added to memory without an optimizer step or a new model artifact. The engine
must retrieve those examples from physical similarity, bind them to the episode-local name, and
choose among the labels allowed for the task.

The decisive test is therefore not only in-corpus accuracy. It is whether the same frozen engine can
use a handful of previously unseen examples, under a previously unseen label string and changed
acquisition conditions, to improve recognition immediately.

## 2. Deployment Motivation

The design target is personalized rehabilitation monitoring. A clinician records a patient
performing a prescribed movement a few times. The patient later performs it at home using their own
watch or phone. The useful concept may be specific to the patient, impairment, and exercise rather
than one of the canonical HAR labels in a public dataset.

This exposes four limitations of ordinary supervised HAR:

1. **The deployment concepts are not enumerable in advance.** A phrase such as `physio arm exercise
   one` may identify a patient-specific movement that no global corpus labels exactly.
2. **Per-user fine-tuning is operationally expensive even when optimization is cheap.** It creates
   model artifacts that must be stored, versioned, validated, and rolled back per deployment.
3. **Memory enrollment has a concrete deletion boundary.** Removing a user's enrolled rows removes
   those examples without retraining. This benefit applies only while enrollment is non-parametric;
   optional tokenizer fine-tuning changes the privacy and model-management analysis.
4. **Enrollment and query acquisition can differ.** Clinic and home hardware, placement,
   orientation, rate, and gravity convention may not match. A device change must not be interpreted
   as a change in the patient's movement.

This does not imply that memory adaptation will beat a fully supervised, task-specific model when
labels and retraining are plentiful. The intended advantages are immediate enrollment, no fixed
output head, operational reuse of one model, inspectable analogues, and explicit handling of
acquisition heterogeneity.

For the broader project positioning and prior-art caveats, see `MOTIVATION.md`, `POSITIONING.md`,
and `EVIDENCE_ENGINE_RESEARCH.md`. In particular, the claim is not that prior work ignores
heterogeneity. The narrower hypothesis is that configuration-aware representations and example
enrollment can work together across label and acquisition shifts.

## 3. Division of Responsibility

Phase A and Phase B answer different unknowns:

- **Phase A asks how the signal was recorded.** Its physical-Hz filterbank, observability masks,
  per-channel/sensor text conditioning, and heterogeneous augmentations produce contextualized patch
  embeddings while preserving acquisition semantics.
- **Phase B asks what the movement should be called in this task.** It retrieves labeled patch
  evidence, contextualizes the query, evidence, and allowed labels together, and predicts among the
  runtime candidates.

Phase B cannot recover information that the sensor did not measure, nor can it separate labels that
are physically indistinguishable from the available IMU. Candidate labels constrain the answer; they
do not make an unobservable distinction observable.

## 4. Terms Used by the Trainer

- **Candidate labels:** the labels allowed as outputs for the current episode. They are runtime
  inputs, not classifier slots or memory-capacity parameters.
- **Corpus archive:** the CPU-resident, immutable collection of Phase-A patch embeddings and source
  metadata from which active memory views are drawn.
- **Active memory:** a rotating, balanced subset of archive windows used for retrieval at a given
  training interval.
- **Memory label:** the immutable canonical label stored with an ordinary corpus row.
- **Provided support:** a labeled execution deliberately enrolled for one candidate in the current
  episode. Its rows receive the `PROVIDED_SUPPORT` role and an episode-local label overlay; the
  archive is not mutated.
- **True support:** provided support for the query's ground-truth candidate.
- **Other support:** equally sized provided support for the other candidate labels. Supplying every
  candidate equally prevents support count from revealing the answer.
- **Zero support:** no candidate concept is represented in memory. The model must rely on coherent
  label semantics and transferable corpus evidence.
- **Random alias:** a neutral temporary name such as `routine amber`, shared by one candidate token
  and all of its support rows within an episode. It tests whether memory binding works without known
  activity language.

There is one physical Phase-B archive, not separate corpus and enrollment banks. Enrollment during
training and evaluation is an episode-local overlay on that archive.

## 5. Patch Memory and Retrieval

The live predictor operates on patch embeddings, not pooled session embeddings. The bank retains
pooled vectors for controls and provenance, but primary retrieval is independently performed for
each valid query patch in four learned projected subspaces. Patch rows retain label, subject,
dataset/configuration, source window, verified event, sensor, physical time, duration, resolution,
and raw-source provenance.

The standard capacity policy is deliberately small:

- archive budget: at most 250,000 source windows, CPU resident; the global label/configuration
  balancing policy applies when the source population exceeds that cap. **Measured 2026-08-08: the
  current corpus yields 248,351 encoded windows, so the cap does not bind and no balancing is
  applied.** The archive is therefore whatever the 50,000-per-stream scan ceiling produced, and is
  strongly imbalanced (`sitting` 28,648 windows against a 30-window minimum). Balance is restored
  downstream by the active view, which equalises to 16 windows per label;
- active view: up to 16 source windows per label, refreshed every 100 steps;
- evidence budget: 64 final patch rows per query;
- candidate counts: four, eight, twelve, or sixteen;
- support counts: one, two, four, or eight independent executions per candidate.

Retrieval K and per-window/per-label contribution limits are derived from the evidence budget. They
are not independent knobs. One support execution is one verified physical event, including its
synchronous placements; unpaired data uses one source window.

Exact query windows and verified synchronous events are excluded from memory. Candidate concepts
are removed from ordinary background memory before the selected support rows are restored. This is
what prevents the model from solving a random-alias episode through a hidden canonical-label path.

## 6. Candidate-Conditioned Evidence

The decoder receives a permutation-equivariant set containing query patches and retrieved evidence.
A query token combines its physical patch vector with the `QUERY` role and an explicit no-label
embedding. Each evidence item is one bound token: its retrieved physical patch and attached label
text are summed with either the `EVIDENCE` or `PROVIDED_SUPPORT` role. Keeping them in one token
prevents attention from losing which label belongs to which physical exemplar. Evidence also carries
the learned retrieval-subspace identity that selected it. Candidate labels are separate tokens with
the `CANDIDATE` role; candidate self-attention and cross-attention read the physical query/evidence
set. Patch center, duration, resolution, sensor/source membership, and validity are supplied as
structural metadata.

Candidate tokens are intentionally enabled. This resolves the older wording discrepancy: **there
are no per-label classifier parameters, but the generic decoder is trained to reason over frozen
text embeddings of the labels supplied at runtime.** An unseen label string is reachable because it
is encoded through the same frozen text interface, not because the model has a learned output row
for that label.

The decoder emits candidate logits from retrieval-weighted evidence. There is no explicit
`UNKNOWN` candidate. Ambiguity and lack of support are handled by a separate confidence stage so
the predictor is not rewarded for always choosing a safe unknown output.

## 7. Episodic Adaptation Curriculum

Training cycles equally through four regimes:

1. `semantic_zero_support`: coherent candidate names and no examples of candidate concepts in
   memory.
2. `ordinary_few_support`: balanced support for every candidate with independent mild acquisition
   variation.
3. `cross_subject_few_support`: support and query come from different physical subjects — a hard
   constraint — and receive different virtual-subject styles. **Acquisition disjointness is allowed
   but not required.** Support drawn from another stream happens wherever the label supports it and
   is counted, never demanded; see §7.1.
4. `same_subject_enrollment`: support and query share one virtual-subject style but receive
   independent acquisition variation, simulating enrollment and later recognition for one person.
   Note that this regime places no constraint on *real* subject identity — the shared persona is
   synthetic, and the underlying executions may come from different people.

### 7.1 Why cross-configuration is an allower, not a prerequisite

Measured on the 2026-08-08 bank: **0 of 93 labels have only one subject**, so person-disjointness is
always satisfiable. But **35 of 93 labels exist in exactly one stream** — `applying_hand_cream`
(harmes only), all seven `falling_*` labels, `ironing`, `floor_cleaning` and others. Requiring
acquisition disjointness alongside subject disjointness made an eight-candidate episode feasible
essentially never (measured 0.00%–1.5% depending on support count), because every candidate had to
satisfy it simultaneously. Enforcing it would silently drop those 35 labels from the regime that most
directly rehearses the deployment story.

Each episode therefore records what it actually enrolled, in recorded executions: how many came from
a different person, from the query's own stream, from a second stream worn simultaneously (a
synchronised rig such as xrf_v2's placements or nfi_fared's back+wrist), and from an independently
recorded stream (wisdm's phone and watch, or two different studies). Those counts appear per batch in
the telemetry and in the periodic training line.

**Scope limit for the paper.** Cross-configuration enrollment is only testable on the 58 labels that
appear on more than one stream — 33 of them via synchronised multi-placement rigs, 25 via two
different studies. No code change extends this; only a corpus with the same activity recorded on more
than one device would.

Supported episodes use one, two, four, or eight independent support executions for every candidate.
They split evenly between coherent activity text and random aliases. Random aliases are forbidden
at zero support because the episode would contain no information connecting an arbitrary word to a
movement.

Candidate distractors include random, language-near, motion-family-near, and physically confusable
labels. Complete motion families are reserved from predictor training for validation. Results must
be stratified by support count and episode regime; averaging them together would hide whether the
adaptation mechanism works.

## 8. Physical Episode Views

Training cycles exactly 50/50 between two physical-view modes. The **clean** mode re-encodes the
unaltered source query and support executions. The **augmented** mode follows this order:

```text
raw source window
-> virtual-subject style, when required by the episode regime
-> independent acquisition augmentation for that execution
-> frozen or online Phase-A tokenizer
```

Virtual-subject style varies pace, dynamic-acceleration amplitude, gyro amplitude, and smoothness.
Dynamic acceleration is scaled around a low-pass gravity estimate so gravity direction and scale
remain intact. Generic Phase-B acquisition augmentation uses mild noise, gain, and a shared SO(3)
rotation for co-located accelerometer and gyroscope channels. It does not change units, gravity
convention, channel layout, patch ordinal, or nominal session duration.

The style ranges were calibrated over 3,180 subject groups and 330 matched label/configuration
strata. The persisted report is `training/evidence/outputs/subject_style_calibration.json`; the live
ranges are intentionally conservative relative to the observed subject variation.

Persistent style and independent acquisition noise serve different purposes. Reusing a
label-specific noise transform would create an augmentation watermark, so acquisition randomness is
sampled per execution rather than used as a candidate identity.

The clean half closes the train/deployment gap: ordinary runtime queries and enrollment examples are
not synthetically transformed. Held-out validation runs each identical episode through both modes and
reports `clean_macro_cell_ba` and `augmented_macro_cell_ba` separately; checkpoint selection uses their
equal-weight mean. A model may therefore earn robustness without hiding a regression on clean input.

## 9. Hard Retrieval and Gradient Flow

Inference uses hard top-k retrieval. During training, the numerical forward remains exactly that
hard path. A balanced soft vote over all eligible rows in the active memory is attached only in the
backward pass:

```text
training_logits = hard_logits + soft_logits - stop_gradient(soft_logits)
```

The soft temperature anneals from 0.20 to 0.07 over 500 steps. Its prior balances label, source
window, resolution, and represented duration so large labels and dense short-patch grids do not win
by row count. The estimator is biased by design, but it gives the retrieval projection and query
path a learning signal when useful support falls outside hard top-k.

Required telemetry includes exact hard-forward equivalence, non-selected-row gradient, hard/soft
gradient norms, retained soft mass, true-support recall, selected-row entropy, candidate entropy,
candidate margin, row/subspace utilization, cross-subspace top-k overlap, identity-control gain, and
support-removal effects.

## 10. Tokenizer Modes

The default mode freezes Phase A. Raw query and support windows are still re-encoded after their
episode-specific physical transforms, but no tokenizer gradients are retained.

The optional `ema_finetune` mode is a warm-started experiment, not joint training from scratch:

1. A detached EMA tokenizer supplies stable keys for selecting neighbors from the archive.
2. The raw query, provided support, and selected background windows are reloaded.
3. Those bounded windows are re-forwarded through the online tokenizer with gradients.
4. Candidate cross-entropy updates the online tokenizer, retriever, and decoder.
5. EMA keys are refreshed in deterministic shards, and inference uses the saved EMA tokenizer.

This avoids retaining a graph for the full bank while still supporting end-to-end fine-tuning after
selection. It also means a fine-tuned deployment no longer has the same deletion and model-versioning
properties as pure non-parametric enrollment; those modes must not be conflated in claims.

## 11. Objective and Confidence

The predictor has one objective: candidate-set cross-entropy on answerable, truth-present episodes.
There is no corpus-classification auxiliary head, metric-learning loss, subject-adversarial loss,
EDL predictor loss, repeatability penalty, or explicit unknown target.

After predictor training, the predictor is frozen. A separate confidence head is trained with BCE to
predict `correct AND answerable` from evidence and retrieval diagnostics. Truth-present and
truth-absent samples use the same physical augmentation, support-overlay, retrieval, and decoder
path; truth-absent samples differ only in that their true concept is excluded from candidates and
memory. Confidence never changes candidate
logits. Thresholds and operating points must be selected on validation data and reported with ECE,
AUROC, and risk/coverage rather than treated as intrinsic uncertainty guarantees.

## 12. Non-Negotiable Controls

- Keep the identity-initialized decoder control beside every trained result. Earlier learned
  evidence variants underperformed their untrained controls; a new component must beat its own
  initialization rather than only an unrelated baseline.
- Do not add a corpus-vocabulary classification head. It would reward memorizing training labels and
  hide failure on runtime candidates.
- Do not train only at high support, with a fixed candidate roster, or with same-configuration
  support. Those choices raise the easy training score while removing the claimed capability.
- Do not give only the true candidate support. Equal support per candidate prevents count leakage.
- Do not use random aliases at zero support or tune thresholds on evaluation labels.
- Do not introduce subject-invariance objectives by default. Subject-specific movement character is
  needed for enrollment, while acquisition nuisances are the intended robustness target.
- Do not claim longitudinal clinical tracking from the classification engine alone.

## 13. Required Evidence Before Making the Claim

1. Support curves at zero, one, two, four, and eight examples per candidate, reported separately for
   same-subject and cross-subject/configuration enrollment. Enrollment is drawn from corrected native
   grids in distinct source-execution units; every window from a chosen same-subject support
   execution is excluded from that subject's query set.
2. A random-alias support curve showing that examples, not familiar label semantics, drive the gain.
3. Support-removal and alias-permutation canaries showing that the predictor reads provided support
   rather than candidate position or canonical-label leakage.
4. Held-out-family and held-out-dataset evaluation, with clear wording about whether the motion,
   label string, subject, or only dataset is unseen.
5. Cross-configuration enrollment results showing that support recorded on one configuration helps
   a query from another.
6. Identity-control comparisons and telemetry ruling out candidate-position, memory-row, retrieval-
   subspace, or low-entropy selection collapse.
7. Confidence ECE, AUROC, and risk/coverage on held-out families and truth-absent episodes.

An improvement in ordinary in-corpus accuracy that degrades these adaptation tests is a regression
against the purpose of Phase B.

## 14. Future Longitudinal Analysis Is Separate

A frozen per-patient reference snapshot may later support questions such as `how has this execution
changed since enrollment?` or `how far is this from a healthy exemplar?`. That is an analytics use
of the same patch representation and retrieval machinery, not a second bank in the current Phase-B
trainer.

Such a feature would require a validated continuous deviation readout and repeatability metrics such
as ICC, SEM, and minimum detectable change. Reference snapshots must be immutable and bound to a
timepoint; automatically appending later sessions would move the reference and could hide real
change. None of these longitudinal claims are implemented or established by the current predictor.

## 15. Implementation Map

- recipe constants: `training/evidence/policy.py`;
- memory construction and provenance: `training/evidence/build_memory.py` and `bank_guard.py`;
- episode overlays and leakage masks: `training/evidence/patch_episodes.py`;
- labels and aliases: `training/evidence/episode_labels.py` and `labeltext.py`;
- physical views: `training/evidence/subject_style.py` and `data/scripts/augmentations.py`;
- retrieval and decoder: `model/evidence/patch_retrieval.py` and `decoder.py`;
- predictor and confidence stages: `training/evidence/train_patch_decoder.py` and
  `train_patch_confidence.py`;
- deployment-style enrollment evaluation: `training/evidence/eval_enrollment.py`;
- live health output: `training/evidence/telemetry.py`.

Before a real Phase-B run, rebuild the memory bank from the selected Phase-A checkpoint. Existing
pre-redesign banks do not satisfy the current patch schema and provenance contract. Use
`training/evidence/README.md` as the single source for launch commands.
