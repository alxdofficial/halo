# Phase-B step-0 control — what Phase-B training actually buys

> Historical control that motivated the relational decoder now specified by
> [`../design/PHASE_B_TRAINING_INTENT.md`](../design/PHASE_B_TRAINING_INTENT.md).
> Measured 2026-08-09. Development roster only; the sealed test roster was not touched.

## 1. Why this arm had to exist

Every Phase-B number on record compares a trained checkpoint against another trained checkpoint.
Milestone predictors start at step 100, and the "identity retrieval vote" control reuses the
**trained** retriever, so it isolates the decoder and says nothing about the system as a whole. Two
readings therefore fit the existing data equally well:

- Phase-B training works, and the engine's score is the product of that training; or
- the frozen Phase-A representation plus retrieval plus a text ensemble already works, and Phase-B
  training is neutral.

`training/evidence/make_step0_predictor.py` closes the gap. It packages the decoder and retriever
**at initialisation** in the evaluator's own artifact schema, so `eval_enrollment` scores it through
the identical path — same cohorts, same candidate sets, same code. Nothing differs but the weights.

The arm is only interpretable if the decoder really is identity-at-init, so the builder asserts that
on a synthetic forward before writing: maximum logit gap **1.5e-8** against the closed-form
retrieval vote. It holds on real data too — the step-0 arm's "full engine" and "identity vote"
columns agree to ≤0.25 F1 on every cell, the residue being argmax ties.

Having both arms also decomposes the gain, which no previous measurement could:

```
step 0                 untrained retriever + untrained decoder
step 1000 identity     trained   retriever + untrained-equivalent decoder
step 1000 engine       trained   retriever + trained decoder
```

## 2. Result

Query-weighted macro F1 over the cells both arms scored. Same-subject and cross-subject are
reported separately: they are different protocols over different query cohorts, and pooling them
would also double-count shoaib, whose unsupported same-subject curve falls back to the
cross-subject cohort at k=0.

**Cross-subject — 17,871 queries, the bulk of the evidence**

| support | step 0 | step 1000 | Δ from training | prototype | ridge | chance |
|---:|---:|---:|---:|---:|---:|---:|
| k=0 | **16.18** | **9.44** | **−6.74** | n/a | n/a | 13.77 |
| k=1 | 45.34 | 46.53 | +1.19 | 52.85 | 52.28 | 13.77 |
| k=2 | 49.41 | 49.39 | **−0.02** | 54.95 | 54.42 | 13.77 |

**Same-subject — 2,953 queries at k=0, 853 at k≥1**

| support | step 0 | step 1000 | Δ from training | prototype | ridge | chance |
|---:|---:|---:|---:|---:|---:|---:|
| k=0 | **24.92** | **15.53** | **−9.39** | n/a | n/a | 20.93 |
| k=1 | 45.93 | 52.34 | +6.41 | 74.45 | 75.30 | 37.30 |
| k=2 | 62.12 | 63.88 | +1.76 | 83.29 | 81.73 | 37.30 |

**Neutral episode-local aliases** — the same cohorts with candidate names replaced by meaningless
strings, so only enrollment can answer:

| mode | support | step 0 | step 1000 | coherent step 1000 | alias advantage |
|---|---:|---:|---:|---:|---:|
| cross-subject | k=1 | 45.42 | 47.62 | 46.53 | +1.09 |
| cross-subject | k=2 | 49.53 | 50.90 | 49.39 | +1.51 |
| same-subject | k=1 | 47.22 | **61.90** | 52.34 | **+9.56** |
| same-subject | k=2 | 60.84 | **74.02** | 63.88 | **+10.14** |

Four conclusions follow. The first changes the diagnosis; the fourth is new.

### 2.1 Phase-B training destroys zero-shot rather than failing to create it

The redesign document's §1.3 concluded that zero-shot "has no substrate": Phase A is label-free, so
nothing maps a window to a label name. That is true of the *trained* system and false of the
untrained one.

At k=0 the untrained system is **above** its chance floor in both regimes (16.18 vs 13.77;
24.92 vs 20.93) and the trained system is **below** it in both (9.44; 15.53). The worst single cell
is shoaib cross-subject, 2,100 queries: **25.44 untrained → 5.32 trained**, against a 14.29 chance
floor and a 3.57 constant-predictor floor.

The substrate was always there. With a candidate's own examples erased from memory, retrieval still
returns semantically *neighbouring* background rows, and their label text bridges to the candidate
names — ConSE, running for free inside the mechanism. Closed-vocabulary episodic cross-entropy then
trains it away.

This matters because a capability being destroyed is a different problem from one that never
existed, and it is a fixable one. It is also the third recurrence of "trained is not better than
untrained" in this project's Phase-B record.

### 2.2 On the bulk of the queries, training at k=2 is a wash — and the decomposition says why

Cross-subject, k=2, 17,871 queries:

| arm | macro F1 | what changed |
|---|---:|---|
| step 0 | 49.41 | — |
| step 1000, identity vote | 48.14 | retriever training: **−1.27** |
| step 1000, full engine | 49.39 | decoder training: **+1.25** |

The decoder's gain almost exactly cancels the retriever's loss. That is not a tuning problem: it is
what happens when the retrieved set *is* the prediction, so the retriever is optimised through a
biased straight-through surrogate while the decoder compensates downstream.

> **2026-08-10.** The straight-through surrogate described above no longer exists. The retriever is
> now reached only by the candidate loss, through the additive attention bias each evidence row's
> retrieval score applies. This paragraph records the mechanism in place when the control was run.

### 2.3 The gap to close belongs to the mechanism, not to the training

Prototype and ridge — closed-form, no Phase-B parameters — beat both arms in every supported cell,
by 5 points cross-subject and by **20 points** same-subject at k=2 (83.29 vs 63.88). Training closes
none of that: the trained engine is 1.76 above the untrained one and 19.41 below the prototype.

This is the direct empirical case for the redesign's Stage 3. Making `base_c` a prototype similarity
starts the model at the prototype column before a single gradient step; the prototype numbers above
*are* that base. The remaining distance is exactly what averaging in representation space provides
and voting does not.

### 2.4 Training makes real activity names actively harmful

The alias condition strips all meaning from candidate names, so only enrolled examples can answer.
It should be *harder*. Same-subject, k=2:

| | coherent names | neutral aliases | alias advantage |
|---|---:|---:|---:|
| step 0 | 62.12 | 60.84 | −1.28 |
| step 1000 | 63.88 | **74.02** | **+10.14** |
| step 1000, learned − identity | +2.91 | **+12.06** | — |

At initialisation the two conditions are equivalent, as they should be. After training the model is
**10 points better when the names are meaningless**, and its decoder contributes four times as much.
Phase-B training has not merely failed to exploit semantics — it has learned that real names are a
liability and performs best when they carry nothing.

That is the behavioural signature of §1.1. When every candidate decision is routed through a text
cosine with a 0.467 common mode and a 0.151 margin, the optimiser's best available move is to
suppress the semantic channel entirely. It cannot separate "these two names are similar because the
activities are related" from "these two names are similar because SBERT puts everything at 0.27".
The cost is §2.1: suppressing that channel is exactly what removes zero-shot.

The redesign addresses this at the root rather than by re-weighting. Coreference binding is a
discrete shared embedding, so enrollment identity never has to be inferred from a cosine; label
semantics enter only where they are genuinely graded, as attention between label tokens and
candidate tokens. The model can then use meaning without having to trust the kernel's common mode.

## 3. Reporting change

The unweighted three-cell mean used by [`PHASE_B_TRAINING_STATUS.md`](PHASE_B_TRAINING_STATUS.md) §4
is retired, on two counts. It weighted RealWorld's 203 queries (3 subjects, a 2-way decision) as one
third of the number; and its three cells were the subset most favourable to training — restricted to
them, training appears to gain +9.4 at k=1 and +3.9 at k=2, where the full matched roster gives
+1.2 and −0.02 cross-subject.

`training/evidence/build_comparison_table.py` replaces the uncommitted script that produced it. It
aggregates query-weighted (reporting the unweighted value alongside), groups by support mode,
restricts to cells every arm scored so the comparison is matched by construction, lists any cell
dropped for that reason, and prints chance and constant-predictor floors next to every score. It
also asserts that the support-free controls — prototype and ridge, which touch neither decoder nor
retriever — are identical across arms, since any difference would mean the arms were not scoring the
same cohort.

## 4. Provenance

- Step-0 artifacts: `training/evidence/outputs/stage1_step0_control/untrained_step0_seed*.pt`,
  built from the step-1000 predictor's metadata so bank, vocabulary, fold and retrieval policy match
  exactly.
- Trained arm: `.../phase_b_20260808/clean_replay/predictors/step_1000.predictor.pt`, the external
  development peak.
- Tables: `comparison.csv`, `comparison_by_cell.csv`, `comparison.json` in the same directory,
  covering both label presentations. Rebuild with:

  ```bash
  python -m training.evidence.build_comparison_table \
    --arm step0    out/enroll_step0_coherent.json    out/enroll_step0_aliases.json \
    --arm step1000 out/enroll_step1000_coherent.json out/enroll_step1000_aliases.json \
    --out-prefix out/comparison
  ```
- The arms record different training regimes (`..._v6_mixed_overlap` and `..._v5`). Re-scoring an
  archived checkpoint under current code requires `--accept-training-regime`, which writes the
  accepted regime into the result so the bypass is auditable rather than silent. Only *training*
  recipes differ; the evaluation path is identical.
- The step-0 run was executed twice, before and after the Stage 2-4 code landed, and reproduced its
  engine column exactly — confirming the new readout did not perturb the vote path. The re-scored
  step-1000 arm also reproduced the archived per-cell numbers exactly (shoaib cross-subject k=0:
  5.32 engine, 27.15 identity).

## 5. What this does not show

- One Phase-A checkpoint, one Phase-B seed, one development roster.
- The step-0 arm's only stochastic component is the retriever's random orthogonal projection; the
  decoder is deterministic at init. Two further seeds are built
  (`untrained_step0_seed20260726/27.pt`) but not scored, so no error bar is claimed.
- Nothing about the relational readout: it is built and unit-tested but has not been trained at
  scale. Its comparison against this table is the next measurement, not a result.
