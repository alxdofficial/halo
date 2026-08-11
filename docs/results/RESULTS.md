# Results — plan of record

> **This is the live results index and experiment plan.** It owns what tables we intend to publish,
> what has actually been measured, and what is blocked. The two other files in this directory are
> banner-flagged historical records of superseded runs and are not plans:
> [`PHASE_B_STEP0_CONTROL.md`](PHASE_B_STEP0_CONTROL.md) (the untrained-control measurement) and
> [`PHASE_B_TRAINING_STATUS.md`](PHASE_B_TRAINING_STATUS.md) (the superseded vote/soft-retrieval run).
>
> Last updated 2026-08-11. **The sealed external test roster has not been consumed.**
>
> **Headline as of the first full run (§2): the mechanism adapts, the training does not.** Untrained
> decoder + trained retriever goes 21.3 → 65.5 macro-F1 from k=0 to k=8 cross-subject. The trained
> decoder is flat at ~13.6 and provably inert to enrollment. Prototype and ridge match the untrained
> arm at full enrollment and cannot enter the k=0 or partial-enrollment regimes at all.

## 1. What we are trying to show

Three questions, three artifacts. They are deliberately not one table: they vary different things
and carry different fairness requirements, and merging them produces a grid that looks authoritative
while comparing incomparable cells.

| | question | artifact | varies | fixed |
|---|---|---|---|---|
| **T1** | At zero target labels, who wins? | table | encoder × dataset | k=0 |
| **T2** | Is adaptation from the mechanism or the representation? | table | encoder × mechanism | k |
| **T3** | Which mechanism owns which label budget? | figure | (encoder, mechanism) × k | — |

T2 and T3 are the same measurement read two ways; T3 is a figure because the crossing points are the
finding and a table hides them.

## 2. First full run of the relational design — measured 2026-08-11

Run `phase_b_20260811`, 3,000 steps, ~20 min, regime
`episodic_memory_adaptation_v19_batched_views_no_k0_selection_gate`. Selected checkpoint **step 800**
by `held_family_low_k_adaptation_score` (0.1672 against step 0's 0.1352) — the first Phase-B run in
which a trained checkpoint won selection on its own merits rather than falling back to step zero.

Scored by `eval_enrollment` on the development roster. Cross-subject, **full** enrollment,
query-weighted over motionsense + realworld + shoaib, **17,871 queries**:

| mechanism | k=0 | k=1 | k=2 | k=4 | k=8 |
|---|---:|---:|---:|---:|---:|
| HALO memory — **trained** decoder | 15.3 | 13.6 | 13.5 | 13.6 | 13.6 |
| HALO memory — **untrained** decoder | 21.3 | 49.7 | 54.0 | 61.2 | **65.5** |
| prototype (closed form) | n/a | 51.1 | 54.6 | 58.9 | 65.3 |
| ridge head (fitted) | n/a | 50.7 | 53.8 | 59.1 | 65.2 |
| *canary:* support removed | — | 15.3 | 15.3 | 15.3 | 15.3 |
| *canary:* labels shuffled | — | 13.2 | 13.2 | 13.1 | 12.9 |

Three findings, in order of consequence.

### 2.1 The trained decoder is inert to enrollment, not merely weak

It scores ~13.6 at every budget, and its own canaries bracket it: removing the enrolled examples
gives 15.3 (**better**), shuffling their labels gives 13.2. Per-cell the gap is extreme — motionsense
cross-subject `k=8` is **18.3 trained against 71.7 untrained**. The checkpoint's
`support_removal_true_probability_drop` of +0.0009 was reporting this accurately.

### 2.2 The mechanism adapts strongly with no Phase-B training at all

The untrained-decoder arm — trained retriever, decoder at initialisation — runs **21.3 → 49.7 → 54.0
→ 61.2 → 65.5**. That is the label-efficiency curve the project exists to produce, on
subject-disjoint queries, with no gradient step at enrollment time.

### 2.3 Prototype and ridge match it at full enrollment, and beat it at k=1

51.1 and 50.7 against 49.7 at `k=1`; all three converge near 65 by `k=8`. The pre-registered kill
criterion therefore fires, and harder than anticipated: it is not only that the *learned* decoder
fails to earn its parameters — the *retrieval machinery itself* does not beat a class centroid over
the same frozen features. On this evidence the value sits in the Phase-A representation.

**What survives, and it is not an operational-convenience argument.** Prototype and ridge are
*undefined* unless every candidate has at least one example (`_few_shot_baselines` returns `None`).
So they cannot enter two regimes the memory arm covers:

- **k=0** — candidates that have a name and no examples (memory arm 21.3, prototype nonexistent);
- **partial enrollment** — some candidates demonstrated, others name-only, which is the realistic
  deployment shape.

Do **not** claim prototype/ridge are "extra work we save". A prototype is append-only too, needs no
gradient step, and stores one centroid per class against our ~15 patch rows per enrolled window — we
are the more expensive option at deployment. The defensible claim is *coverage*, not cost.

### 2.4 Why the training fails

| | step 1 | 400 | 1600 | 3000 |
|---|---:|---:|---:|---:|
| train BA | 0.216 | 0.566 | 0.615 | **0.638** |
| **validation BA** | 0.237 | 0.227 | 0.204 | **0.208** |
| support-removal Δp | −0.000 | 0.000 | 0.000 | −0.000 |
| coherent BA | 0.268 | 0.247 | 0.213 | 0.215 |
| alias BA | 0.146 | 0.166 | 0.175 | 0.188 |
| candidate logit spread | 0.18 | 1.76 | 5.49 | **6.78** |

Two hypotheses are ruled out by this table. It is **not** label-text memorisation: coherent and alias
*converge* (0.215 vs 0.188), and coherent declines — a name-reader would hold coherent high and leave
alias at chance. It is **not** learned-then-forgotten: support-removal Δp is ≈0.000 at step 1 and
still ≈0.000 at step 3000, so enrollment is never read at any point. Every enrollment shape scores
the same ~0.21, and zero-support is the best cell.

Meanwhile train BA nearly triples while validation BA is flat from the first measurement, and the
logit spread grows 38×. The decoder minimises cross-entropy by **memorising the training
vocabulary's query signatures rather than learning to compare a query against its support**. On a
held-out family there is nothing memorised and no comparison ability, so it emits a near-constant
answer — which is exactly the flat 13.6 with bracketing canaries observed externally.

**The structural cause is the v19 objective.** Its loss was candidate-set cross-entropy "and nothing
else". Nothing required the answer to depend on support; reading enrollment was one way to lower CE,
memorisation was easier, and the optimiser took it. The current §11 records the v20 replacement.

### 2.5 Implemented next experiment — use the counterfactual pair that is already built

Every optimizer step already constructs an exact support/zero pair: same query, same candidates, same
phrasing, same physical view, differing *only* in the memory overlay
(`counterfactual_role="support"`/`"zero"`). In v19 both halves received independent cross-entropy and
were never compared. V20 adds two paired terms:

1. **p(true | support) > p(true | no support) + margin.** Because the halves are identical apart from
   the overlay, the difference is attributable to enrollment alone. A model answering from the query
   alone yields exactly zero difference, so this is the one gradient memorisation cannot satisfy.
   Mask it to the episodes where the true candidate is actually enrolled — partial enrollment selects
   its subset independently of the query truth, so elsewhere no gain should be expected.
2. **p(true | correct support) > p(true | shuffled-label support).** Same rows, same count, permuted
   labels. This blocks the cheap escape of boosting whichever candidate merely *has* support, forcing
   the content of the examples to matter.

The two terms target precisely the two canaries that have never moved. They are implemented in regime
`episodic_memory_adaptation_v20_counterfactual_support_bootstrap`, together with a 400-step frozen-
retriever decoder bootstrap and an alias support-binding anchor. This is an implementation status,
not a result: v20 still needs a real training run.

### 2.6 A selector problem this exposes

The v19 internal held-family canaries reported gain −0.145 (0.233 against 0.378) while the external cells
showed 13.6 against 65.5. The selector ranked step 800 "best" while the model was inert on every real
cohort. The canaries must be validated against external cells before they are trusted for checkpoint
selection again. V20 replaces the independently drawn/confounded canaries with fixed C=8 nested
k-curves and makes support-use interventions eligibility conditions; the new canaries still require
empirical validation against development data.

### 2.7 Gradient-path audit for v20 — measured 2026-08-11

This is an implementation diagnostic, not a quality result. A 405-step real-bank probe crossed the
400-step support bootstrap and measured a disjoint, exhaustive partition of every decoder parameter.
All 13 groups had 100% backward-graph coverage and finite nonzero gradients at every sampled
checkpoint (steps 100, 200, 300, 400, and 405). At step 405, decoder gradient RMS was `1.43e-3` and
retriever RMS was `4.26e-6`; the retriever had 100% coverage and its direct task-gradient norm was
`1.09e-3`. The norm difference is therefore not a dead path. AdamW's adaptive scaling matters here:
a separate one-step probe at the configured `2e-4` LR moved the online retriever `0.585%` relative to
its EMA copy despite its smaller raw RMS.

| decoder path | gradient RMS at step 405 |
|---|---:|
| candidate readout | 6.18e-3 |
| text projection | 2.68e-3 |
| input / output layer norm | 2.46e-3 / 2.45e-3 |
| role / slot embeddings | 3.85e-4 / 2.32e-4 |
| attention stack | 8.46e-5 |
| query / evidence projections | 7.17e-5 / 7.21e-5 |
| time projection | 5.68e-5 |
| acquisition relations | 1.75e-5 |
| group embeddings | 3.17e-6 |

An accelerated one-step diagnostic also crossed the intentional tokenizer freeze in
`ema_finetune` mode. Filterbank (`1.92e-5` RMS), context transformer (`2.14e-6`), duration
conditioning (`1.23e-6`), and channel conditioning (`7.79e-7`) all had 100% coverage and nonzero
entries. The audit initially found the 256-value Phase-A mask token in the optimizer despite Phase B
never executing masked prediction; v20 now freezes it. Frozen-mode runs correctly report zero
tokenizer/retriever gradients during their declared freeze rather than treating them as failures.

### 2.8 Scope

One checkpoint, one seed, development roster only; the sealed roster was not touched. Dev has zero
unseen concepts, so this measures adaptation to new people and devices, not to new label strings.
Partial-enrollment cells run much lower for the untrained arm (~35–43 against 60–77 full) because
unenrolled candidates can only be answered from text; report them separately rather than pooled. The
coherent-label arm only — the `--random-aliases` arm has not been run.

Artifacts: `training/evidence/outputs/phase_b_20260811/` and
`training/evidence/outputs/eval_enrollment_dev.json`.

## 3. T1 — Zero-shot across models

Rows = model, columns = the 7 held-out datasets **grouped by unseen-vocabulary fraction**, cells =
macro-F1 with subject-stratified 95% CI. Grouping matters: three of the seven sets are fully covered
by the 93-label training vocabulary, so a flat mean dilutes the open-set claim into a plain transfer
number. This grouping must match the paper's `tab:per_dataset`.

Measured (`training/diagnostics/outputs/current_protocol_table.md`, 7/7 datasets, protocol v4,
`vocab_fp=05853ac157dd03ad`):

| model | mean macro-F1 | status |
|---|---:|---|
| harnet | **45.7** | current |
| halo_evidence | 42.9 | **STALE — re-score required** |
| crosshar | 42.8 | current |
| unimts | 34.7 | current |
| halo (ConSE) | 34.4 | **STALE — re-score required** |
| limubert | 32.2 | current |
| imagebind | 11.4 | current |
| normwear | 5.1 | current |

### 3.1 The two HALO rows are stale and the artifacts cannot say so

Every result JSON in `eval/results/` is dated **2026-07-25 to 2026-08-05**. The Phase-A checkpoint
Phase B is training against — `training/tokenizer/outputs/phase_a_headline/best.pt`, step 27,000 —
is dated **2026-08-07**. The six baseline rows do not depend on our encoder and remain valid. The
`halo` and `halo_evidence` rows do, and predate it.

Worse, this is not detectable from the artifact. `_protocol` records only
`{version, n_labels, vocab_fp, split_fp}` — there is **no backbone or checkpoint fingerprint** — so a
row scored with any Phase-A checkpoint looks identical on disk. The Phase-B bank carries `backbone`
and `bank_fp` provenance and a behavioural probe; the zero-shot eval harness carries neither.

**Two actions, both required before T1 ships:**

1. Re-score `halo` and `halo_evidence` on the current Phase-A checkpoint.
2. Stamp a backbone fingerprint into the eval artifact and reject a table that mixes fingerprints —
   the same guard `bank_guard` already applies to Phase-B artifacts.

Until then, do not place a HALO enrollment curve built on the Aug-7 encoder next to a HALO
zero-shot number built on an earlier one and call it one row.

## 4. T2 — Adaptation: mechanism × encoder

The load-bearing design point is that this is a **cross, not a list**. Without the cross, "HALO
adapts well" is unattributable between the mechanism and the features.

| encoder | mechanism | gradient step? | status |
|---|---|---|---|
| HALO | memory, trained decoder | no | free from `eval_enrollment` |
| HALO | memory, untrained step-0 | no | free |
| HALO | prototype | no | free |
| HALO | ridge | yes (head only) | free |
| harnet | prototype | no | **build** |
| harnet | ridge | yes (head only) | **build** |
| harnet | memory / retrieval | no | **build** — the killer control |

Every row also carries two canaries: **support-removed** and **shuffled-support-labels**. A row whose
score does not move under either is not reading its enrollment, and its number means something other
than adaptation. As of the 2026-08-11 pre-run check both canaries sit at ≈0.0 for the untrained
decoder, which is the expected starting state and the thing training has to change.

The four HALO rows are matched by construction: `_few_shot_baselines` in
`training/evidence/eval_enrollment.py` fits the prototype and ridge controls from the *same* support
rows the memory receives, on the same cohort, in the same episode.

### 4.1 What `k` actually counts, and at what granularity enrollment lands

Verified in `_support_and_query_rows`: for each candidate the evaluator draws `k` distinct
executions, then takes **exactly one window from each**. So

> **`k` = k windows per class, one per distinct execution.**

The execution structure is a *leakage constraint* — it forces the k windows to come from k different
bouts rather than from k neighbouring slices of one recording — not a data-volume multiplier. `k=8`
is 8 windows on every dataset, so the unit is comparable across datasets and directly comparable to
any baseline fitted on the same 8 windows.

Those 8 windows land in memory at a finer granularity than they are fitted at:

| level | what it is | role |
|---|---|---|
| execution | a verified physical event (one bout/trial) | leakage unit — the k windows must come from k different ones |
| **window** | one ~6 s window; **this is what `k` counts** | the enrolled example |
| patch | ~15.1 per window (bank: 248,351 windows → 3,745,436 rows) | the retrievable memory row |

So `k=8` enrolls 8 windows ≈ **121 patch rows**, each carrying its own label, subject and candidate
binding, and retrieval matches query patches against those patch rows. There is no pooled per-window
or per-session vector on the live path; the bank retains pooled `Z` only for controls and provenance.

The prototype and ridge controls read the **same 8 windows** through the pooled representation
(`support_encoded["pooled"]`, L2-normalised, centroid or dual-form ridge). Same data, different
granularity — that difference *is* the mechanism being tested, not a confound. Report both `k` and
the realised patch-row count so this is visible.

Build the harnet rows through `eval_enrollment`'s own cohort — swap the feature extractor, reuse the
prototype/ridge path — so the support draw and query set stay identical by construction.

## 5. T3 — Label efficiency (buildable cross-subject; same-subject is what is constrained)

x = `k`, y = macro-F1, one line per (encoder, mechanism), crossing points annotated.

**Cross-subject enrollment supports a full `k = 0,1,2,4,8` curve on six of the seven held-out
datasets, including all three development sets.** Enrolling from people other than the query subject
pools executions across the whole cohort, so the ceiling is set by the dataset's total size rather
than by how many times one person repeated an activity. Measured 2026-08-11 (worst case per label,
holding out the subject that owns the most executions):

| dataset | subjects | cross-subject `k` ceiling (min · med · max over labels) | labels supporting `k=8` |
|---|---:|---|---|
| motionsense | 24 | 46 · 69 · 69 | 6/6 |
| realworld | 15 | 14 · 14 · 18 | 8/8 |
| shoaib | 10 | 9 · 9 · 9 | 7/7 |
| inclusivehar | 20 | 19 · 19 · 19 | 6/6 |
| usc_had | 14 | 65 · 65 · 65 | 12/12 |
| ut_complex | 10 | 9 · 9 · 9 | 13/13 |
| tnda_har | **1** | 0 | 0/8 — single subject, no cross-subject cohort exists |

T3 is therefore **not blocked**: the dev roster alone supports the full curve for every mechanism.
Two limitations remain, and they are narrower than "no curve":

1. **Same-subject** — the literal deployment story, where the clinician records *this* patient — is
   the constrained protocol (§5.2). Report the two protocols separately; never pool them.
2. **Unseen concepts** — dev has none (§5.3), so a dev curve measures adaptation to a new
   *person or device*, not to a new *concept name*.

### 5.1 Why a curve needs repeated executions

To place a point at budget `k`, one subject must have performed the same activity **k+1 separate
times**: `k` to enroll, at least one left to be asked about. Enroll the only recording a subject made
of an activity and there is nothing left to query — the cell does not exist.

The unit has to be a distinct **execution**, not a window. A five-minute walking recording cut into
six-second windows yields ~50 windows, so it is superficially tempting to "enroll 8 windows and query
the rest" — but those windows are seconds apart in one bout: same shoes, same floor, same session,
same gait phase. Scoring that measures near-duplicate retrieval, not adaptation. The evaluator
refuses it, rejecting window-level pseudo-event ids for same-subject adaptation with status
`unverified_window_level_execution_ids`.

### 5.2 Measured executions per (subject, label), native grids, 2026-08-11

| dataset | subjects | labels | executions per (subj, label) | max same-subject `k` |
|---|---:|---:|---|---:|
| motionsense | 24 | 6 | min 2 · med 3 · max 3 | 2 |
| realworld | 15 | 8 | min 1 · med 1 · max 3 | 2 |
| shoaib | 10 | 7 | all 1 | **0** |
| inclusivehar | 20 | 6 | all 1 | **0** |
| ut_complex | 10 | 13 | all 1 | **0** |
| usc_had | 14 | 12 | all 5 | **4** |
| tnda_har | **1** | 8 | 351–461 | (n/a — one subject) |

Three of the seven held-out sets record each subject performing each activity exactly once, so they
admit **no same-subject curve at all**. The dev roster's best possible curve is `k ∈ {0,1,2}`, and
`shoaib` contributes only its `k=0` point. `tnda_har`'s large count is an artifact of having a single
subject — same-subject enrollment is vacuous there, and no subject-disjoint claim is constructible.

Cross-subject enrollment is far less constrained (enroll from person A, query person B needs only two
people per activity), so a cross-subject curve reaches further. But cross-subject is not the
deployment story: "the clinician records *this* patient, who then exercises at home" is same-subject.

### 5.3 A second, independent limitation — no unseen concepts on dev

All 21 dev label strings are already in the 93-label training vocabulary. Even a perfect dev curve
would therefore measure adaptation to a new *subject or device*, never to a new *concept*. The
open-set claim needs strings the corpus never saw, and those sit in the sealed roster (`usc_had` has
5).

So the two limitations are separable and both bind: dev cannot reach past `k=2`, **and** dev cannot
test open-set at any `k`. The only legacy set with both a real curve and unseen labels — `usc_had`,
`k ≤ 4`, 5 unseen — is behind the seal.

**T3's right home is the rehabilitation roster**, where SPAR (20 subjects × 7 exercises × 20
repetitions) supports a genuine k=8 same-subject curve with 7 unseen concepts and left/right as a
free cross-placement axis. Those datasets are converted and gridded
(`data/datasets/{spar,monipar,mmfit,realdisp,forth_trace,phytmo,kneepad,upper_limb_use}/grids/`) but
have **no eval label configs and are not in `PRIMARY_EVAL_DATASETS`**, so they are not scorable yet.
That work is in flight in another working tree — coordinate rather than duplicate.

## 5b. v20 readiness measurement — 2026-08-11

§2.5 proposed paired counterfactual terms as the fix for the inert decoder. They are now built
(`episodic_memory_adaptation_v20_counterfactual_support_bootstrap`), and this section records the
pre-launch measurement rather than the intention.

What changed, beyond the two proposed terms:

- A third matched intervention — correctly bound support versus the **same rows with deranged
  candidate bindings** — so "support present but its labels ignored" is penalised separately from
  "support absent".
- A **400-step decoder bootstrap** with the retriever and tokenizer frozen, C=2-4, k=1-2 and
  alias-heavy draws, so the decoder learns the elementary intervention before the selector moves.
- **Validation curves are now un-confounded.** v19 varied the candidate count across the curve, so
  k=1 and k=8 differed in two things at once. All three curves now fix C=8 and vary only support.
- **Checkpoint eligibility is three mechanism checks**, not a scalar: beat the closed-form
  retrieval-vote control, lose true-label probability when support is removed, and lose it again
  when support labels are deranged. If nothing qualifies, the artifact explicitly declares
  `closed_form_retrieval_vote` instead of silently exporting a random decoder as "trained" — the
  §2.6 selector failure is now structurally impossible.
- The early exploration phase was **inert in v19**: it raised the assembled evidence budget while
  `topk_per_subspace` stayed pinned to the deployment budget, so no additional rows were ever
  retrieved. Raw per-subspace K now derives from the active training budget.

Measured on a 450-step real-config probe (RTX 4090, frozen tokenizer):

| step | counterfactual queries with true support retrieved | support-vs-zero log-p gain | retriever grad |
|---:|---:|---:|---:|
| 100 | 0.330 | −5.953 | 0.0 |
| 200 | 0.238 | +0.0001 | 0.0 |
| 300 | 0.259 | +0.0026 | 0.0 |
| 400 | 0.259 | +0.0074 | 0.0 |
| 450 | 0.401 | +0.0179 | 0.120 |

Three things this establishes, none of which were true in v19:

1. **The objective has substrate.** The counterfactual loss is masked to queries where hard
   retrieval actually exposed true-candidate support; 24-40% qualify, against a 5% monitor alert
   threshold. A near-zero fraction would have made the whole term silently inert — the failure mode
   that produced v19's result — and `counterfactual/selected_query_fraction` now alerts on it.
2. **The quantity that was flat is moving.** Support-versus-zero log-probability gain crosses from
   −5.95 to +0.018 and rises monotonically. That is the direct measurement of the enrollment
   dependence v19 lacked (§2.1: Δp ≈ 0 from step 1). It is small in absolute terms; the run has to
   show it keeps climbing.
3. **The bootstrap releases exactly where declared** — retriever gradient is identically 0.0 through
   step 400 and nonzero at 450.

Runtime: 141 s for 450 steps including validations, so the 3,000-step default projects to roughly
16 minutes. The v19 README figure of 18.5-19 min was withdrawn as unprofiled; this replaces it.

**One comparability caveat for the §2 table.** `eval/data.py` now applies the duplicate/implausible
quality screens to evaluation, which drops 34 flagged windows from `motionsense/phone_front_pocket`,
a Phase-B development stream. The v20 dev numbers are therefore computed on a very slightly smaller
query population than the 2026-08-11 prelim table. The difference is 0.75% of one stream and does not
move any conclusion, but the two tables are not bit-comparable.

## 6. Pending decisions

1. **Does the seal open?** Reporting T3 on the legacy roster today requires `usc_had`, which spends
   the seal. Recommendation: keep it sealed, ship a 3-point dev curve (k=0,1,2) labelled as unable to
   test open-set, and hold T3 for the rehab roster.
2. **Do we build the harnet mechanism rows?** This is the highest-value build in the plan, because it
   is the control that decides whether the contribution is the mechanism or the encoder. Without it
   T2 is a list, not a cross.
3. **Do we add a linear probe at matched k?** It is the rival named in the paper's pre-registered
   kill criterion, and no few-shot harness for baselines currently exists.

## 7. Runs and provenance

- **Completed failed run:** `training/evidence/outputs/phase_b_20260811/` — 3,000 steps, frozen tokenizer,
  evidence budget 64, `--val-every 200`, regime
  `episodic_memory_adaptation_v19_batched_views_no_k0_selection_gate`. Launched 2026-08-11 against
  bank `training/evidence/outputs/memory_bank.pt` (schema 3, 248,351 windows / 3,745,436 patch rows /
  93 labels, bound to Phase-A step 27,000).
- **Next run:** regime `episodic_memory_adaptation_v20_counterfactual_support_bootstrap`. Selection
  now requires nonnegative gain over the retrieval-vote control and positive support-removal and
  support-label-shuffle effects. With no eligible learned checkpoint, the artifact explicitly selects
  the best retrieval-vote control state. Any comparison with v19 is a different objective and
  selection protocol.
- Zero-shot table: `training/diagnostics/outputs/current_protocol_table.md`, assembled by
  `eval/assemble_table.py` from `eval/results/*.json`.
- Enrollment curves: `training/evidence/outputs/eval_enrollment_dev.json` (+ `_aliases`), rebuilt by
  `python -m training.evidence.eval_enrollment --device cuda`.

## 8. Reporting rules for every table here

- Score each dataset against **its own** label strings, never a shared ontology.
- Subject is the independent unit; report paired subject-bootstrap intervals for control deltas.
- Report unsupported cells as unsupported. Never substitute windows or change the cohort to fill one.
- Keep development, sealed-test, and custom artifacts in distinct files.
- Every trained arm appears beside its untrained control.
- State the label-budget unit explicitly wherever `k` appears.
