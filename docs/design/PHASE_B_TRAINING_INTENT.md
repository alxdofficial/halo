# Phase-B Training Intent

> **Canonical Phase-B motivation and executable contract. Read this before configuring, launching,
> or interpreting Phase-B training.**
>
> Status: implementation-aligned as of 2026-08-09. Phase A is complete. The current relational,
> learned-query Phase-B path has passed unit and synthetic integration tests but has not yet been
> trained at scale. [`../results/PHASE_B_TRAINING_STATUS.md`](../results/PHASE_B_TRAINING_STATUS.md)
> records the superseded run that motivated this design. The sealed test roster has not been used.

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

For the broader project positioning and prior-art caveats, see `MOTIVATION.md` and
`POSITIONING.md`. In particular, the claim is not that prior work ignores
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
  downstream by the active view, which caps every label at 16 windows;
- active view: up to 16 source windows per label, refreshed every 5 steps. Half of each capped
  label budget reserves distinct executions from one randomly rotated anchor subject; the remainder
  is configuration- and subject-balanced. This preserves the repeated-person executions required
  by the same-subject curriculum without allowing one subject to occupy the whole view;
- evidence budget: 64 final patch rows per query. During the first third of training, a wider
  shortlist begins at 256 rows with retrieval-prior temperature 0.20 and anneals to the exact
  64-row, temperature-0.07 deployment recipe;
- candidate counts: an integer sampled from two through sixteen;
- support counts: an integer sampled from one through eight independent executions per enrolled
  candidate.

Retrieval K is derived from the evidence budget. The final roster is the highest-scoring unique
memory rows across query patches and learned subspaces; no per-label or per-window cap alters that
ranking. One support identity is one verified physical event; all rows from that event that are
present in the active view are bound together. The active window budget does not promise that every
synchronous placement is present. Unpaired data uses one source window.

Exact query windows and verified synchronous events are excluded from memory. Candidate concepts
are removed from ordinary background memory before the selected support rows are restored. This is
what prevents the model from solving a random-alias episode through a hidden canonical-label path.

## 6. Candidate-Conditioned Evidence

The sole decoder is a relational transformer over `[candidate names; background label names; query
patches; retrieved evidence]`. Candidate, label, query, evidence, and provided-support roles are
explicit embeddings. Randomized episode-local coreference slots bind a retrieved evidence row to
its candidate or background-label token without placing text directly on the physical row. Source
window groups are randomized independently, so neither slot nor group identity can become a stable
label shortcut. Query/evidence configuration, subject, sensor, and physical time are supplied as
relations or structural metadata.

Candidate tokens are intentionally enabled. This resolves the older wording discrepancy: **there
are no per-label classifier parameters, but the generic decoder is trained to reason over frozen
text embeddings of the labels supplied at runtime.** An unseen label string is reachable because it
is encoded through the same frozen text interface, not because the model has a learned output row
for that label.

A candidate logit is the readout on its token, and nothing else. Selected cosine scores are first
converted to a shift-invariant log-relative prior over the valid evidence rows:

```text
retrieval bias = log_softmax(score / temperature) + log(valid evidence count)
```

Equal scores therefore contribute zero bias, adding a common offset changes nothing, and only
within-roster preferences affect attention. This is the numerical contract across Phase B: Phase-A
patch vectors, label-text vectors, and retrieval projections are L2 normalized at the decoder or
retrieval boundary; the relative prior
has the same scale as ordinary attention logits; summed token features pass through LayerNorm; and
the candidate readout is trained with ordinary unscaled cross-entropy. Because selection is a hard
top-k over frozen memory vectors, the relative attention prior is the only differentiable route from
the candidate loss back to the retriever projection. `dL/ds < 0` means "increase this row relative
to the other selected rows," rather than rewarding the absolute common mode of a high cosine.

An earlier revision made the logit a zero-initialized residual over a closed-form base — prototype
similarity where the retriever had surfaced a candidate's enrolled rows, a label-text vote
otherwise — on the argument that this gave a non-random floor and made the learned contribution
directly measurable. Measurement rejected it. On the real bank the prototype branch was reachable
for only **15.5%** of decisions, because the retriever selects a candidate's declared support just
21.7% of the time for the true candidate and 2.0% for the others; so ~85% of base scores were the
label-text vote this decoder exists to replace. Worse, the two branches sat **6.4 logits apart**
(602:1 odds), so the base decided by *which branch a candidate landed on* rather than by any
comparison. The instrument had become the model, and it is gone.

Enrollment remains fully available: declared support carries the `PROVIDED_SUPPORT` role and a
coreference slot binding it to its candidate, so the stack can compare the query against it
directly. It has to learn that comparison rather than be handed a closed-form version of it.

There is no explicit `UNKNOWN` candidate. Confidence calibration code remains available as a parked
follow-up experiment, but it is not part of the current Phase-B launch or claim.

## 7. Episodic Adaptation Curriculum

Training covers four regimes:

1. `semantic_zero_support`: coherent candidate names and no examples of candidate concepts in
   memory.
2. `ordinary_few_support`: enrolled candidates receive balanced support with independent mild
   acquisition variation; enrollment may be partial or full.
3. `cross_subject_few_support`: support and query come from different physical subjects — a hard
   constraint — and receive different virtual-subject styles. **Acquisition disjointness is allowed
   but not required.** Support drawn from another stream happens wherever the label supports it and
   is counted, never demanded; see §7.1.
4. `same_subject_enrollment`: support and query are distinct executions from the same real subject.
   Augmented views additionally share one virtual-subject style while receiving independent
   acquisition variation. Clean views therefore rehearse real same-person enrollment rather than
   silently pairing two different people.

One optimizer step contains eight episodes with eight query executions each. One coherent partial-
support episode is paired with a zero-support counterfactual that reuses its exact query, candidates,
candidate phrasing, and physical query view; only the support-memory overlay differs. One fully
enrolled random-alias episode is also guaranteed so every loss condition is represented. Remaining
episodes are independent draws. Apart from the counterfactual pair, every episode has its own
candidate set, support overlay, label aliases, physical view/persona, and support count; only the
immutable archive and trainable model parameters are shared. Gradients are accumulated across the
eight episode forwards and applied once.

Partial enrollment is a random proper subset of
candidates receiving support while the remaining candidate concepts stay erased from memory; it
falls out of drawing the enrolled count uniformly from one to the candidate count rather than being
a named stratum. Random aliases remain full-support only because an arbitrary name with no binding
example is unanswerable. The realized mix is reported in telemetry.

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

**Scope limit for the paper.** Cross-configuration enrollment is only testable where the same
activity and subject identifiers exist on more than one stream. The external evaluator now builds
those support/query pairs explicitly and skips unsupported pairs; no resampling or synthetic pairing
can substitute for a corpus that actually recorded both configurations.

Under the current external roster, Shoaib supplies this cross-configuration check only in the
development protocol. None of the sealed datasets supplies a valid cross-configuration enrollment
cohort. TNDA-HAR has one subject and window-level execution identifiers, so it contributes only the
coherent zero-support condition; it cannot support a genuine same- or cross-subject enrollment
claim. Sealed adaptation claims therefore come from InclusiveHAR, USC-HAD, and UT-Complex, and the
paper must report that scope rather than pooling unsupported TNDA-HAR cells.

During training, an enrolled candidate receives an independently sampled integer from one through
eight support executions; an unenrolled candidate receives zero. Fixed validation and external
reporting curves use one, two, four, and eight so every checkpoint is compared at identical,
interpretable anchors. Full-support episodes enroll every candidate, while partial episodes enroll
a random proper subset. Random aliases are forbidden at zero or partial support because an
unenrolled arbitrary name would contain no information connecting it to a movement.

Candidate difficulty is staged: the first 20% of training uses 2-4 candidates with 25% hard
distractors, the next 40% uses 4-8 with 50% hard distractors, and the remainder uses the complete
2-16 range with 75% hard distractors. Hardness is the mean of text cosine and physical-centroid
cosine; other distractors are random. Complete motion families are reserved from predictor training for validation. Training uses
only rows whose subject and configuration are both in the training partition. Validation queries use
all three excluded quadrants: held subject, held configuration, and both held. Fixed validation
canaries include coherent partial, coherent full, random-alias full, and semantic zero-support
recipes at every supported `k`, each evaluated through clean and augmented views. Four semantic
zero-support recipes cover candidate counts 2, 4, 8, and 16. The default 48
base canaries are the complete 16-recipe by three-transfer-condition Cartesian product, so fold
metrics cannot be confounded by different episode mixtures. Internal held-family canaries use
zero-support, ordinary, and
cross-subject episodes; real same-subject enrollment is measured by the external evaluator because
the held-subject memory partition intentionally contains no rows from its query people. Results must
be stratified by support count and episode regime; averaging them together would hide whether the
adaptation mechanism works.

## 8. Physical Episode Views

Training samples two physical-view modes with equal probability. With a frozen tokenizer, the **clean**
mode uses stored clean vectors whose equivalence to live encoding is guarded by the bank probe; the
fine-tuning mode uses live clean forwards. The **augmented** mode follows this order:

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
reports `clean_macro_cell_ba` and `augmented_macro_cell_ba` separately. Checkpoint selection uses the
fixed held-family low-support cells described below, which contain both views equally. A model may
therefore earn robustness without hiding a regression on clean input.

## 9. Learned Query Retrieval and Gradient Flow

After the episode defines which rows physically exist in memory and applies leakage exclusions, the
forward roster is selected only by learned query-to-memory similarity in projected physical
subspaces. No fixed support bonus, label-text similarity, configuration weight, per-label/window
retrieval cap, manually constructed eligibility roster, or backward-only vote can add or promote a
row. Enrollment identity is used only as an episode label, loss target, and telemetry annotation.

Hard top-k selection is non-differentiable. Candidate cross-entropy trains the scores of selected
support and background evidence together with the relational decoder, reaching the retriever through
the attention bias those scores apply. That is the only retriever objective.

An earlier revision added a multiple-instance support-boundary loss over eligible memory, promoting
the best true-support match above the evidence-budget cutoff. It has been removed. It was introduced
when the decoder could not reach the retriever at all (`grad_residual_to_retriever` measured exactly
0.0 in 8/8 probes), a premise the attention bias removed; its positives were the *true* candidate's
support, making it label-conditioned during training and absent at evaluation; and it was the only
mechanism in the system that scored against rows outside the forward roster, contradicting the
explicit decision to accept gradient for selected rows only. The accepted cost is that a missed
support row can no longer be promoted.

Required telemetry includes assembled true-support recall, fixed-canary roster churn, selected-row
and retrieval-prior entropy, candidate entropy and margin, candidate-to-role attention mass,
row/subspace utilization, raw pre-deduplication cross-subspace overlap, retriever/decoder gradient
ratio, component gradients, and support-removal/label-shuffle effects.

Telemetry is interpreted per episode type, enrollment shape, support count, candidate count, label
mode, and physical view rather than only as a global average. Optimization uses standard candidate
cross-entropy, averaged equally over the eight independent episodes. `CE / log(candidate_count)` is
reported only as a chance-relative diagnostic; using it for gradients over-weighted two-candidate
episodes by roughly three times relative to sixteen-candidate episodes.
Expected modules expose component gradient norms, non-finite gradients stop before an optimizer
update. Every launch immediately
replaces stale health output with a run-identified heartbeat. A CPU-only monitor turns the latest
snapshot into green/warning/critical health, a text summary, and an optional plot.

## 10. Tokenizer Modes

The default mode freezes Phase A. Raw query and support windows are still re-encoded after their
episode-specific physical transforms, but no tokenizer gradients are retained.

The optional `ema_finetune` mode is a warm-started experiment, not joint training from scratch:

1. A detached EMA tokenizer supplies stable keys for selecting neighbors from the archive.
2. The raw query, provided support, and selected background windows are reloaded.
3. Those bounded windows are re-forwarded through the online tokenizer with gradients.
4. Candidate cross-entropy updates the online tokenizer, retriever, and decoder.
5. The small active key view is fully refreshed on its normal 5-step cadence, and inference uses
   the saved EMA tokenizer.

This avoids retaining a graph for the full bank while still supporting end-to-end fine-tuning after
selection. It also means a fine-tuned deployment no longer has the same deletion and model-versioning
properties as pure non-parametric enrollment; those modes must not be conflated in claims.

## 11. Objective and Confidence

The objective is candidate-set cross-entropy on answerable, truth-present episodes, and nothing
else. Per-query CE is reduced separately for semantic zero support, partial-enrollment queries whose
candidate is unenrolled, coherent enrolled queries, and alias enrolled queries; the four present
group means receive equal weight. There is no auxiliary retrieval term, corpus-classification head, subject-adversarial loss,
EDL predictor loss, repeatability penalty, or explicit unknown target.

The default run is 3,000 optimizer steps at learning rate `2e-4`, with a 300-step linear warmup and
cosine decay to zero. At eight episodes and eight queries per step this is 24,000 instantiated
adaptation episodes and 192,000 query executions; 3,000 support/zero pairs intentionally
share their query context.

Checkpoint selection is predeclared and uses fixed held-family `k=1/2` canaries. Its adaptation scalar is
`low_k_balanced_accuracy + 0.5 * (low_k_balanced_accuracy - identity_balanced_accuracy)`: absolute
quality is primary, while learned gain over the matched identity control also matters. A checkpoint
is eligible only when its fixed zero-support score remains within one point of the stronger of the
step-zero zero-support score and its matched current identity score. Step zero is retained as the
fallback if every trained milestone violates that floor. Every
validation milestone is saved, so the selector can be audited against external development curves.

The separate confidence head and truth-absent episode generator are parked. If resumed, predictor
weights remain frozen and calibration must be reported with ECE, AUROC, and risk/coverage; none of
those claims belong to the current Phase-B experiment.

## 12. Non-Negotiable Controls

- Keep the identity-initialized decoder control beside every trained result. Earlier learned
  evidence variants underperformed their untrained controls; a new component must beat its own
  initialization rather than only an unrelated baseline.
- Do not add a corpus-vocabulary classification head. It would reward memorizing training labels and
  hide failure on runtime candidates.
- Do not train only at high support, with a fixed candidate roster, or with same-configuration
  support. Those choices raise the easy training score while removing the claimed capability.
- Do not give only the true candidate support. Within the enrolled subset, every candidate receives
  the same support count; partial episodes choose the subset independently of the query truth.
- Do not use random aliases at zero support or tune thresholds on evaluation labels.
- Do not introduce subject-invariance objectives by default. Subject-specific movement character is
  needed for enrollment, while acquisition nuisances are the intended robustness target.
- Do not claim longitudinal clinical tracking from the classification engine alone.

## 13. Required Evidence Before Making the Claim

1. Support curves at zero, one, two, four, and eight examples per enrolled candidate, reported
   separately for full and half-candidate partial enrollment and for same-subject, cross-subject,
   same-configuration, and cross-configuration conditions. Enrollment is drawn from corrected native
   grids in distinct source-execution units; every window from a chosen same-subject support
   execution is excluded from that subject's query set.
2. A random-alias support curve showing that examples, not familiar label semantics, drive the gain.
3. Support-removal and support-label-shuffle canaries showing that the predictor reads provided
   support. Consistent label-renaming agreement is reported separately as a naming-stability check;
   it is not evidence of support use.
4. Held-out-family and held-out-dataset evaluation, with clear wording about whether the motion,
   label string, subject, or only dataset is unseen.
5. Cross-configuration enrollment results showing that support recorded on one configuration helps
   a query from another.
6. Identity-control comparisons and telemetry ruling out candidate-position, memory-row, retrieval-
   subspace, or low-entropy selection collapse.
7. External enrollment curves must use one fixed subject/candidate/query cohort and nested,
   execution-independent support prefixes. Report unsupported cells instead of substituting windows
   or changing the cohort. Compare against support removal, shuffled support labels, prototypes, and
   a fitted few-shot head on exactly the same support/query split.
8. Development uses the registered external development roster. The external test roster is selected
   explicitly only after design and checkpoint choices are frozen; report seen and unseen activity
   separately for Phase-A vocabulary exposure and Phase-B candidate-training exposure; also record
   whether the dataset, subject, and exact label string were seen.
9. Treat the subject as the independent unit for uncertainty. Report paired subject-bootstrap
    intervals for adaptation gains over identity, support-removed, shuffled-label, prototype, and
    fitted-head controls. Keep development, sealed-test, and custom artifacts in distinct files.

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
- retrieval and decoder: `model/evidence/patch_retrieval.py` and `relational_decoder.py`;
- predictor stage: `training/evidence/train_patch_decoder.py`;
- parked confidence experiment: `training/evidence/train_patch_confidence.py`;
- deployment-style enrollment evaluation: `training/evidence/eval_enrollment.py`;
- live health output: `training/evidence/telemetry.py`.

Before a real Phase-B run, rebuild the memory bank from the selected Phase-A checkpoint. Existing
pre-redesign banks do not satisfy the current patch schema and provenance contract. Use
`training/evidence/README.md` as the single source for launch commands.
