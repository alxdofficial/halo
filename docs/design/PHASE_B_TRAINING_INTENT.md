# Phase-B Training Intent

> Canonical motivation and executable contract for the current Phase-B design.
>
> Status: implementation-aligned as of 2026-08-17. The relational evidence decoder is not the
> active design. Phase B now starts with a closed-form predictor and a small admissibility gate.

`docs/design/DESIGN_OF_RECORD.md` gives the full design history and evidence. This document states
only the current training and evaluation contract. `training/evidence/README.md` gives commands.

## 1. Goal

HALO should use a labeled memory as an adaptation mechanism. A caller supplies candidate labels and
may enroll examples for some or all of them. HALO then predicts one of those candidates without
fitting a separate classifier for each person or task.

The memory cannot recover information absent from the sensors. Candidate labels restrict the answer
set; they do not make physically indistinguishable activities separable.

## 2. Terms

- **Candidate:** a label allowed as an answer for the current prediction.
- **Corpus row:** a labeled sensor-patch embedding from the Phase-A training corpus.
- **Enrolled row:** a sensor-patch embedding supplied for one episode candidate.
- **Query row:** a sensor-patch embedding from the recording being classified.
- **Admissibility:** whether a sensor configuration can provide evidence about a candidate concept.
- **k:** the number of independent enrolled executions supplied per enrolled candidate.

The archive stores rows per patch and per physical sensor. Accelerometer and gyroscope rows are
separate. Every row retains its modality, gravity convention, sensor description, sensor bias,
label, subject, stream, source window, physical time, duration, and resolution.

## 3. Required Artifacts

Phase B requires all three artifacts to come from the same sensor-granularity Phase-A checkpoint:

1. A schema-5 memory bank built with `build_memory --sensor-rows`. Schema 5 stores and fingerprints
   the exact per-sensor descriptor text, modality, and gravity convention used at encoding time.
2. A schema-2 resolvability table measured only on Phase-A training subjects and datasets.
3. An admissibility-gate artifact fitted from that table and bound to the exact memory bank.

The development and sealed test datasets must not contribute to gate fitting. Validation subjects
from Phase A are also excluded. Artifact fingerprints make a stale checkpoint, bank, text path, or
sensor table fail before evaluation.

The completed Phase-A source is
`training/tokenizer/outputs/phase_a_fixed_1s_rotation_20260817/best.pt`. Current persisted Phase-B
artifacts predate it and must be rebuilt; their filenames do not imply that they are current.

## 4. Resolvability Measurement

For each physical sensor and activity, the measurement asks whether that sensor's embedding can
separate the activity from the other activities in the same protocol. It uses subject-disjoint kNN
and maps chance performance to zero.

Accelerometer and gyroscope are measured independently. A stream-level vector pooled across both
sensors is not a valid target for a per-sensor gate. Simultaneous multi-placement datasets provide a
separate paired contrast, where subject, event, and clock are held fixed.

The fitted table is training data for the gate. It is not an independent validation result after
fitting. Generalization is measured with held-out concept, dataset, and body-region folds.

## 5. Prediction

For every query sensor patch and candidate:

1. **Compatibility:** retain evidence with the same modality. Accelerometer rows must also use the
   same gravity convention. Gyroscope rows do not have a gravity convention.
2. **Admissibility:** score both the query sensor description and evidence sensor description against
   the candidate concept. Combine the two scores with a geometric mean.
3. **Retrieval:** rank candidate-row choices by `cosine / temperature + log(admissibility)`. The
   learned term is continuous; only physical incompatibility is a hard exclusion. Convert these
   scores to nonnegative evidence mass with one query-wide numerical stabilizer.
4. **Voting:** an enrolled row votes for the candidate to which it was explicitly bound. A corpus row
   votes through similarity between its label text and the candidate text.
5. **Merge:** sum votes across query patches and sensors.

The sensor-bias similarity term is disabled by default because it can reproduce dataset identity.
Any experiment that enables it must report retrieval provenance with the term both on and off.

## 6. Coherent Labels and Arbitrary Aliases

With coherent activity names, semantic admissibility and corpus label-text voting are enabled. This
is the zero-support path and also permits corpus evidence to supplement enrollment.

With arbitrary aliases, the words intentionally have no activity meaning. Semantic admissibility
and corpus label-text voting are therefore disabled. Only explicitly enrolled rows can vote for an
alias. The query-side physical compatibility filter still applies. This prevents an arbitrary word
such as `protocol amber` from being penalized because its language embedding resembles no known action.

## 7. Stage 1: Closed Form

Stage 1 fits the low-rank admissibility gate to the train-only resolvability measurements. It does
not train a relational decoder or learned retrieval projection. The evaluation compares:

- admissibility-gated retrieval;
- the same retrieval and voting rule with admissibility set to one;
- support removed;
- support labels cyclically shifted;
- prototype classification; and
- a fitted L2 ridge head.

External evaluation reports k = 0, 1, 2, 4, and 8, partial and full enrollment, and genuine same-
subject, cross-subject, same-configuration, and cross-configuration cohorts where the dataset can
support those claims. The subject is the independent unit for confidence intervals.

The present evaluator constructs a fixed nested curve within each run, but its candidates are the
labels feasible for that subject and requested support ceiling. This is a conditional
support-feasible benchmark, not automatically the dataset's complete label roster. Candidate count
and coverage must be reported beside every cell. A fixed full-roster arm is required before making a
closed-label-set claim.

The archive and the active retrieval population are distinct. The current evaluator balances a
working set of 16 source windows per corpus label before retrieval; it does not search all 250,000
archived windows. Result artifacts record both populations. The 16-window policy remains a
development variable to validate, not an invisible property of “the memory bank.”

## 8. Stage 2: Optional Gate Refinement

Stage 2 is eligible only if Stage 1 is useful. It keeps the encoder, bank features, retrieval rule,
and voting rule fixed and refines only the small admissibility gate.

Training uses candidate cross-entropy. Admissibility enters a full soft distribution over all
physically compatible candidate-row choices in a bounded, label-balanced working memory. There is no
discrete top-k in the training path, so every compatible row can receive credit. Validation and
deployment apply top-k to the same adjusted score, then sum the resulting soft evidence mass.

Episodes vary candidate count, support count, and partial versus full enrollment. Query and support
come from distinct recorded executions. The candidate labels are removed from ordinary corpus
memory, then only the sampled enrolled executions are restored with explicit candidate bindings.
This prevents hidden same-label corpus evidence from masquerading as adaptation. Coherent label
paraphrases vary the wording while preserving the activity meaning.

A small replay penalty keeps the gate aligned with train-only physical resolvability measurements.
This is a semantic anchor for the same gate, not a learned decoder or an additional representation
objective. Its cells are limited to the Stage-2 training concepts. Internally held concepts were
still present in the Stage-1 warm start, so internal validation measures "not refined on," not
"never observed." External development datasets remain the model-selection evidence.

Stage 2 must retain the Stage-1 controls. A lower training loss is not sufficient. The refined gate
must beat its own step-zero warm start and the gate-disabled rule under exact hard retrieval, improve
external enrollment results, and retain configuration-dependent skill under held-out folds. A large
gap between soft-training and hard-validation behavior is evidence to study a differentiable sparse
top-k operator; it is not a reason to add one preemptively.

## 9. Parked Code

The learned subspace retriever, relational decoder, curriculum, synthetic subject styles, and their
old checkpoints remain only for reproducing earlier experiments. They are not the default Phase-B
trainer and must not be described as the current model. New Phase-B commands use
`predictor_mode='admissibility_gate'` and a schema-5 bank.

## 10. Source of Truth

- measurement: `training/evidence/resolvability.py`;
- gate and fitting: `training/evidence/admissibility_gate.py` and `gate_predictor.py`;
- controlled text views: `training/evidence/admissibility_text.py`;
- prediction: `training/evidence/admissible_retrieval.py`;
- optional gate refinement: `training/evidence/train_admissibility_gate.py`;
- memory construction and guards: `training/evidence/build_memory.py` and `bank_guard.py`;
- external enrollment evaluation: `training/evidence/eval_enrollment.py`;
- commands: `training/evidence/README.md`.
