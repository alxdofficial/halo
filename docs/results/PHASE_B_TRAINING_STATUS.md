# Phase-B Training Status

> **Authoritative Phase-B empirical ledger.** The current behavioral contract is in
> [`PHASE_B_TRAINING_INTENT.md`](../design/PHASE_B_TRAINING_INTENT.md); commands are in
> [`training/evidence/README.md`](../../training/evidence/README.md). This file owns completed-run
> results, current readiness, and their interpretation.
>
> Last updated: 2026-08-22. The current rank-8 per-sensor admissibility experiment and matched
> five-seed adaptation suite are recorded below.
> Its checkpoint was selected on internal and external development results before the frozen test
> roster was opened. No test result was used for fitting, checkpoint selection, or configuration
> changes. Later sections retain historical relational-decoder results for comparison only.

## Current Admissibility Design: Readiness

The current sensor-granularity Phase-A checkpoint, schema-5 bank, train-only resolvability table,
artifact-version-4 gate, Stage-2 checkpoint, and external enrollment curves are complete. The result
validates the pipeline and support mechanism, but it does **not** establish that learned Stage 2
outperforms closed-form retrieval on held-out datasets.

| required artifact | current state | consequence |
|---|---|---|
| Phase-A checkpoint | ready: `phase_a_fixed_1s_rotation_20260817/best.pt`, step 27,000 | valid sensor-row source; 18 training datasets and no Phase-B dev/test overlap |
| schema-5 memory bank | ready: 250,000 windows, 1,350,834 patches, 2,629,972 sensor rows, fp `34e1ce91c843f1b2` | descriptor text/modality/gravity are stored and fingerprinted |
| schema-2 resolvability table | ready: 110 sensors, 1,664 train-only cells, bound to the selected checkpoint | valid gate-fitting source |
| bound admissibility gate | ready: artifact version 4, rank 8, bound to the schema-5 bank | Stage 2 is finite, reaches every gate module, and has completed external evaluation |
| current enrollment curves | complete: 3 development datasets and a matched five-seed suite over 7 test datasets | coherent and arbitrary-label protocols include retrieval, support-removal, label-shuffle, prototype, and ridge controls; six external representations use the same episodes |

The selected Phase-A checkpoint records `git='8e64e39-dirty'`. Its weights, configuration, corpus
fingerprint, and fixed behavioral probes make development use auditable, but the source tree that
created it is not recoverable from a commit alone. A paper-grade rerun should be produced from a
clean, committed revision after the design is frozen.

A matched fixed-one-second transfer evaluation now removes the earlier evaluation-grid confound. The
older `phase_a_sensor_v1_20260813_v2` step-4,000 checkpoint scores **0.617** mean kNN balanced
accuracy across the same seven held-out datasets; the selected fixed-one-second step-27,000 checkpoint
scores **0.509**. The **-0.108** gap is a real representation-quality warning, although the training
recipe and rebuilt corpus still differ, so this comparison does not identify a single cause. Artifacts:
`phase_a_checkpoint_selection_20260816/transfer_{old,new}_fixed1s.json`.

The rebuilt rank-8 warm-start gate had a clear internal warning. Across 16 deterministic held-concept
validation episodes (28 candidate/support cells), mean macro F1 was **0.380** with the gate and
**0.592** with admissibility set to one (delta **-0.211**). Stage 2 recovered this failed
initialization on development data, but the full test result was only at parity with identity
retrieval. The fitted gate must therefore be described as an initialization, not a validated model.

Targeted tests cover gate, retrieval, bank, resolvability, sensor export, and enrollment controls.
A real-data probe exported valid per-sensor rows from Capture24 using the
current checkpoint. It also exposed and fixed one metadata defect: accel-only archive rows were
reconstructed as if a gyroscope had been present.

Three protocol limitations remain before a paper-grade Stage-1 result:

1. The evaluator exposes only 16 archive windows per label. On the existing historical bank this is
   1,488 of 248,351 windows (0.60%). The new evaluator records the actual active count, but the
   memory-size choice still needs a development-only sweep or a chunked full-bank implementation.
2. Candidate sets are support-feasible per subject. In historical same-subject cells this reduced
   MotionSense from six activities to three and RealWorld from eight to two. This is a valid
   conditional few-shot cohort only when reported explicitly; it is not a fixed full-label-set
   benchmark and must not be described as one.
3. The compatibility filter permits only accel-to-accel or gyro-to-gyro retrieval and, for
   accelerometers, only the same gravity convention. This makes cosine comparisons conservative but
   provides no adaptation path across those acquisition differences. In the historical archive,
   gravity-removed streams held 14,964 of 248,351 windows (6.0%), so they formed a small isolated
   retrieval partition. The current schema-5 build must report this partitioning, and a development
   ablation must distinguish “invalid cross-modal cosine” from heterogeneity the encoder can bridge.

Stage-2 gate refinement uses a full soft distribution during training, so every physically
compatible row receives gradient. Validation and deployment truncate the same adjusted score to
top-k. The completed run is numerically healthy and improved development performance, but its
held-out margin over identity retrieval is effectively zero.

## Current Experiment Table

| mechanism | checkpoint/bank | dev | test | status |
|---|---|---:|---:|---|
| current per-sensor admissibility | current 27k / schema 5 / Stage-2 step 1,000 | measured | five matched seeds plus six external representations | pipeline valid; test is near or below identity retrieval, not a learned Stage-2 win |
| parked relational v22 | historical channel checkpoint / schema 3 | measured | not the main v22 run | retained below |
| parked relational checkpoint study | Phase-A 4k and 30k / schema 3 | measured | measured once after selection | retained in Section 9 |
| frozen HARNet enrollment control | released HARNet trunk | measured | measured once | retained in Section 10 |

## Matched Adaptation Suite: 2026-08-17

The paper-facing suite is under `eval/adaptation_results/v1_d85761d_stage2/`; assembled output is
under `eval/adaptation_tables/v1_d85761d_stage2/`. It uses source revision `d85761d`, manifest
fingerprint `1bd89d35f5aed197`, predictor fingerprint `64af1db442d9cc75`, bank fingerprint
`34e1ce91c843f1b2`, and manifest seeds 20260808 through 20260812. The seven test datasets have no
dataset overlap with the 18-source Phase-A corpus, gate fitting data, or memory bank. Supports and
queries are execution-disjoint, k counts executions per candidate, candidate rosters are serialized,
and the headline is an unweighted macro average over datasets. Full support-removal and label-shuffle
controls run on the first seed; positive-k primary curves use five seeds, while deterministic k=0 is
scored once.

During the pre-assembly audit, an initial HALO invocation was found to have used the default Stage-1
gate instead of the selected Stage-2 checkpoint. Those outputs are excluded from this section. The
evaluator now requires an explicit predictor path, and the assembler reports paired intervals
separately by action regime, label mode, and k.

The action regimes are fixed before scoring. **Ordinary** contains InclusiveHAR, USC-HAD, TNDA-HAR,
and UT-Complex: population locomotion and daily activities. **Specialized novel** contains Monipar,
SPAR, and Upper Limb Use: held-out clinical motor assessments, shoulder rehabilitation exercises,
and fine-grained upper-limb tasks outside the Phase-B training ontology. “Novel” describes the
held-out dataset/category regime; it does not claim that every underlying motion is physically
unprecedented. Random aliases are a separate binding control and are not what defines this regime.

At k=0, external models use their matched ConSE zero-shot bridge and HALO has no enrolled support.
At positive k, each external row fits the common 200-step linear head on k labeled executions per
class while freezing its representation; HALO performs gradient-free memory enrollment.

**Ordinary activities, macro F1**

| model | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|---:|
| HALO Stage 2 | 23.75 | 44.79 | 48.89 | 51.36 | 51.18 | 49.32 |
| HARNet | 33.82 | 51.35 | 56.63 | 61.56 | 62.77 | 59.73 |
| CrossHAR | **37.01** | 51.94 | 57.46 | 63.02 | 63.53 | 61.37 |
| LiMU-BERT | 27.60 | 54.26 | **61.00** | 64.80 | 64.65 | 60.29 |
| UniMTS | 32.70 | **54.81** | 59.92 | **65.49** | **66.69** | **65.05** |
| ImageBind | 11.38 | 45.17 | 53.12 | 58.48 | 58.98 | 55.94 |
| NormWear | 5.08 | 35.81 | 42.43 | 46.81 | 46.76 | 44.99 |

**Specialized novel activities, macro F1**

| model | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|---:|
| HALO Stage 2 | 9.81 | 28.62 | 29.20 | 39.74 | 43.10 | 45.53 |
| HARNet | 11.40 | 34.72 | 37.54 | 54.40 | **61.34** | **66.37** |
| CrossHAR | 10.88 | 32.20 | 36.35 | 46.69 | 49.07 | 51.12 |
| LiMU-BERT | 9.11 | 28.95 | 31.12 | 38.29 | 39.40 | 39.96 |
| UniMTS | **19.24** | **38.95** | **39.38** | **55.23** | 61.04 | 65.17 |
| ImageBind | 8.15 | 29.19 | 32.91 | 40.36 | 44.98 | 48.53 |
| NormWear | 3.58 | 25.31 | 25.94 | 33.89 | 36.36 | 38.19 |

The positive-k comparison is a common label-efficiency control, not model-native end-to-end
fine-tuning. k=16 has fewer eligible datasets because some cohorts lack 16 independent executions;
comparisons between models within a given k remain matched.

Paired subject bootstrap intervals support a narrow conclusion. On ordinary activities, Stage 2
minus identity is -0.23 at k=1 (95% CI -0.78 to 0.26), -0.38 at k=2 (-0.77 to -0.00), -1.06 at k=4
(-1.73 to -0.40), and -1.70 at k=8 (-2.68 to -0.77). Specialized activities are statistically at
parity through k=8; k=16 is -0.99 (-1.62 to -0.39). The learned gate therefore does not improve the
enrollment mechanism as a whole.

Arbitrary labels produce the intended neutral-gate behavior: learned and identity retrieval are
exactly equal. Their ordinary curve is 46.26, 50.27, 53.43, 53.35, 50.73; the specialized curve is
30.66, 30.39, 41.35, 44.62, 47.51. On the first seed, removing support reduces the arbitrary-label
dataset-macro range to 1.90-2.58 and shuffling support labels reduces it to 3.17-12.23. This is strong
evidence that labeled enrollment drives prediction, but no evidence that Stage 2 improves arbitrary
label binding because semantic admissibility is deliberately neutral there.

This binding ability is not yet a representation-quality win. Against training-free nearest-support
controls under arbitrary labels, HALO trails LiMU-BERT and UniMTS at every k in both regimes, and
also trails CrossHAR and HARNet at most points. The result validates the protocol and memory
mechanism, while locating the remaining deficit primarily in representation/retrieval quality.

The suite is valid for matched model comparisons within a given k. It has three interpretation
limits. First, k=16 covers fewer datasets because some cohorts lack 16 independent executions;
cross-k changes must be read with the dataset-count column. Second, counterfactual controls have one
seed rather than five. Third, this test roster has now informed diagnosis, so future design selection
requires a newly sealed confirmation roster.

## Initial Rank-8 Run: 2026-08-17

> Superseded for paper comparison by the matched five-seed suite above. Retained as the development
> and checkpoint-selection record.

Artifacts are under `training/evidence/outputs/admissibility_stage2_rank8_20260817/`. Stage 1 fitted
the rank-8 gate in 8.25 seconds. Stage 2 ran for 2,000 optimizer steps with four independent episodes
per step and completed in 103.64 seconds on the local RTX 4090. The internal validation macro F1 was
0.384 before Stage 2, 0.606 at step 500, 0.620 at step 1,000, 0.590 at step 1,500, and 0.584 at step
2,000. Step 1,000 was therefore frozen before external test evaluation.

On the three development datasets, step 1,000 improved all 33 non-tied cells and lost none relative
to the fitted Stage-1 gate. Its cell-macro F1 rose from 38.95 to 49.15. Identity retrieval scored
48.37, so the learned margin was only +0.78 point. Prototype and ridge controls remained stronger at
56.97 and 56.33. This supports checkpoint selection but does not establish a large decoder benefit.

The table below reports unweighted means over valid protocol cells. Coherent rows include zero,
partial, and full enrollment. Arbitrary-label rows require positive full enrollment, so their
absolute F1 must not be compared directly with the coherent rows.

| protocol and split | cells | learned | identity retrieval | prototype | ridge | support removed | labels shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| coherent development | 38 | 49.15 | 48.37 | 56.97 | 56.33 | 34.64 | 25.45 |
| coherent test | 207 | 25.14 | 25.04 | 30.24 | 30.31 | 15.14 | 13.02 |
| arbitrary-label development | 16 | 55.67 | 55.67 | 56.97 | 56.33 | 8.51 | 12.73 |
| arbitrary-label test | 93 | 30.35 | 30.35 | 30.24 | 30.31 | 3.22 | 10.03 |

The paper-facing k curves below use only full-enrollment cells at k greater than zero. This keeps the
learned, identity, prototype, and ridge columns on exactly the same cohort; prototype and ridge are
not defined for partial enrollment. The number of eligible cells falls with k because some
subject/configuration cohorts do not have that many independent support executions.

**Coherent test labels**

| support per candidate | cells | learned | identity retrieval | prototype | ridge |
|---:|---:|---:|---:|---:|---:|
| 0 | 21 | 15.37 | 14.47 | N/A | N/A |
| 1 | 28 | 26.44 | 26.37 | 28.70 | 28.17 |
| 2 | 26 | 27.41 | 27.43 | 28.67 | 28.53 |
| 4 | 22 | 30.86 | 30.86 | 31.17 | 31.42 |
| 8 | 17 | 34.23 | 34.49 | 33.99 | 35.12 |

**Arbitrary test labels**

| support per candidate | cells | learned | identity retrieval | prototype | ridge |
|---:|---:|---:|---:|---:|---:|
| 0 | N/A | N/A | N/A | N/A | N/A |
| 1 | 28 | 27.99 | 27.99 | 28.70 | 28.17 |
| 2 | 26 | 27.69 | 27.69 | 28.67 | 28.53 |
| 4 | 22 | 32.30 | 32.30 | 31.17 | 31.42 |
| 8 | 17 | 35.80 | 35.80 | 33.99 | 35.12 |

Arbitrary-label k=0 is not a difficult zero-shot problem; it is unidentifiable. With a fresh random
one-to-one alias assignment and no labeled example, every permutation between activities and aliases
is equally compatible with the observations. It must be reported as N/A, not as a model score.
Identity retrieval and the learned predictor are equal for k greater than zero because
out-of-vocabulary aliases receive neutral admissibility. Removing the enrolled rows reduces the
arbitrary-label mean to 3.22; shuffling their episode-local label bindings reduces it to 10.03. The
memory therefore provides genuine adaptation, but Stage 2 does not improve that path.

### Historical k=0 Gap, Now Resolved

At the time of this initial run, a matched coherent k=0 table did not exist. The five-seed suite above
has now run HARNet, CrossHAR, UniMTS, LiMU-BERT, ImageBind, and NormWear on the serialized current
query windows and candidate rosters. Arbitrary-label k=0 remains N/A because the class-to-alias
permutation is unidentifiable without enrollment.

Held-out coherent results are heterogeneous:

| dataset | learned | identity retrieval | difference |
|---|---:|---:|---:|
| InclusiveHAR | 32.39 | 30.72 | +1.67 |
| USC-HAD | 34.24 | 33.43 | +0.81 |
| TNDA-HAR | 34.46 | 35.74 | -1.27 |
| UT-Complex | 35.68 | 39.22 | -3.54 |
| Monipar | 28.17 | 27.61 | +0.56 |
| SPAR | 30.09 | 28.91 | +1.18 |
| Upper Limb Use | 19.92 | 20.18 | -0.26 |

**Verdict:** the current code and artifact path are operational, Stage 2 successfully repairs the
poor fitted-gate initialization, and labeled memory demonstrably drives arbitrary-label adaptation.
The scientific result is still modest: coherent test performance is effectively tied with identity
retrieval and remains below prototype/ridge controls, while arbitrary-label predictions are exactly
the retrieval-only result. The matched five-seed suite above confirms the lack of a general learned-
admissibility advantage and retains the UT-Complex regression.

## Current Design Audit and Literature Check

The basic idea is well precedented. [Matching Networks](https://arxiv.org/abs/1606.04080) conditions
prediction on a labeled support set without per-task fine-tuning, and
[Prototypical Networks](https://arxiv.org/abs/1703.05175) shows that a simple metric-space class mean
is a strong few-shot inductive bias. [Tip-Adapter](https://arxiv.org/abs/2207.09519) is the closest
analogue to HALO's current Stage 1: it combines a frozen representation, a labeled cache, and a
separate zero-shot prior, with an optional lightweight fitted refinement. Wearable-HAR reviews also
frame pretrain-then-finetune as the standard comparator rather than something a memory method may
omit; see [Haresamudram et al.](https://arxiv.org/abs/2202.12938). A recent HAR-specific prototype
method reports large one-shot personalization gains using closed-form updates
([Burzer et al.](https://arxiv.org/abs/2606.04798)), reinforcing that prototype and fitted-head curves
are primary baselines, not secondary diagnostics.

What is sound:

- per-sensor rows avoid comparing an accel-only query with a pooled accel-plus-gyro vector;
- modality and gravity compatibility are explicit and label-independent;
- enrolled examples vote by their episode-local binding, so arbitrary labels can work;
- corpus-label voting and support voting provide separate zero- and few-support information sources;
- subject/execution guards, support removal, label shuffling, prototype, ridge, and bank fingerprints
  are appropriate controls.

What should change before a strong result:

1. **Completed: replicate the current result.** Five fixed manifest seeds now confirm parity or a
   small regression rather than a stable learned-gate improvement.
2. **Register the candidate roster independently of query truth.** Keep the support-feasible cohort
   as a named secondary analysis, and add a fixed dataset-roster arm for the main closed-set curve.
3. **Measure memory size.** Sweep the active windows per label on development data, record runtime,
   and freeze one value. If the full archive is needed, use chunked/ANN retrieval rather than hiding
   a 0.6% working set behind the word “bank.”
4. **Strengthen the resolvability target.** The current target uses one deterministic half-subject
   split. Repeated subject-group folds or leave-one-subject-out averages would reduce noise before a
   low-rank gate is fitted to those values.
5. **Separate semantic and enrollment scores.** The current vote mixes corpus text votes and bound
   support votes inside one neighbor softmax. As in cache adapters, report the two components and use
   at most one development-fitted blend coefficient. This is especially important under partial
   enrollment, where the existing prototype/ridge baselines are unavailable.
6. **Diagnose Stage-2 generalization.** The full soft training distribution addresses selected-row
   credit assignment, but the learned gate still loses on UT-Complex and adds nothing for arbitrary
   aliases. Investigate those development analogues before changing the retrieval rule, and retain
   the held-fold physical-correlation guard.
7. **Isolate reproduction code.** Move the relational decoder/evaluator modes behind an explicitly
   historical entry point. The active evaluator currently imports both designs, which is operationally
   safe but makes the current mechanism harder to explain and easier to misconfigure.
8. **Bind the frozen text encoder behavior.** Implemented: schema-5 banks record the
   MiniLM model name, sentence-transformers version, and a fixed output probe. Evaluation rejects a
   changed runtime text space before comparing stored labels with candidate embeddings.
9. **Keep admissibility continuous.** Implemented: training and deployment use the same
   `cosine / temperature + log(admissibility)` score. No independently calibrated veto remains.
10. **State what compatibility does not solve.** Exact modality/gravity matching prevents invalid
    comparisons; it does not demonstrate cross-modality or cross-convention adaptation. Retain the
    conservative rule for the first Stage-1 result, report partition coverage, then ablate only on
    development data before claiming that Phase B handles those forms of heterogeneity.

The evaluator now records the fraction of candidates receiving no evidence and the fraction of
queries for which every candidate score is zero. In the required-answer protocol an all-zero row
still resolves by candidate order, so this telemetry distinguishes a real vote from a forced tie.

## Historical Relational Verdict

The v22 clean arbitrary-label experiment is a **clear development-level adaptation success**.
Unlike the preceding coherent-only run, the learned evidence engine passed every internal mechanism
gate and was selected instead of the closed-form fallback. On genuinely held development datasets,
performance rises consistently as labeled support is added. Removing support or shuffling its
episode-local names destroys that gain.

The result does not establish a final claim. It uses one Phase-B seed, the coherent zero-support path
is weak, and prototype/ridge controls remain several F1 points stronger. The correct next question is
how to preserve zero-support semantics while closing the remaining gap to direct support methods.

## Historical Training Recipe

| setting | value |
|---|---:|
| Phase-A tokenizer | frozen |
| optimizer | AdamW |
| steps | 3,000 |
| independent episodes per step | 8 |
| queries per episode | 8 |
| candidate count | uniform integer from 2 through 16 |
| support per candidate | uniform integer from 0 through 8 |
| k=0 label presentation | coherent activity names |
| k>0 label presentation | fresh neutral aliases per episode |
| signal views | clean stored patch embeddings |
| objective | candidate-set cross-entropy only |
| evidence budget | 64 patch rows |
| learning rate | `2e-4`, 300-step warmup, cosine decay |
| tokenizer fine-tuning | disabled |
| seed | `20260725` |

Every candidate receives the same `k`. A positive-support episode assigns a one-to-one random name
such as `protocol amber` to each candidate and to that candidate's support rows. The assignment is
redrawn for every episode. Retrieval remains learned and query driven; no support row is manually
inserted into top-k evidence.

The run processed 24,000 independent episodes and 192,000 query windows in 601 seconds on the local
RTX 4090. It completed without non-finite values, dead gradient paths, or clipping instability.

## Internal Checkpoint Selection

Step 1000 was selected as the learned relational decoder. It passed all declared requirements:

- learned low-k performance exceeded the closed-form vote;
- support presence improved prediction;
- removing support reduced correct-label probability;
- shuffling support labels reduced correct-label probability.

Fixed held-family C=8 balanced accuracy at the selected checkpoint:

| support per candidate | learned engine | identity retrieval vote |
|---:|---:|---:|
| 0, coherent | 0.142 | 0.338 |
| 1, arbitrary names | 0.350 | 0.259 |
| 2, arbitrary names | 0.360 | 0.301 |
| 4, arbitrary names | 0.400 | 0.354 |
| 8, arbitrary names | 0.455 | 0.389 |

Removing support reduced mean true-label probability by 0.151. Cyclically shuffling support names
reduced it by 0.168. Training and held-family macro BA were nearly identical at selection
(`0.3425` versus `0.3414`), unlike the large gap in the failed coherent-only run.

Later checkpoints retained adaptation but did not improve low-k selection. The final step reached
0.325 low-k BA versus 0.355 at step 1000, while k=8 remained 0.440. This supports checkpointing the
development optimum rather than treating the final optimizer state as authoritative.

## External Development Evaluation

The development roster is MotionSense, RealWorld, and Shoaib. Values below are unweighted means of
available dataset/protocol macro F1 scores. Cross-subject cohorts use genuinely different recorded
people, not synthetic subject transformations.

### Arbitrary-label full enrollment

| relation | k | learned engine | identity vote | prototype | ridge | support removed | labels shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| same subject | 1 | 74.51 | 78.94 | 82.82 | 79.66 | 42.65 | 14.54 |
| same subject | 2 | 77.66 | 80.20 | 80.75 | 79.35 | 42.65 | 14.79 |
| cross subject | 1 | 49.87 | 50.22 | 54.84 | 54.49 | 14.27 | 11.96 |
| cross subject | 2 | 55.29 | 56.70 | 59.40 | 58.83 | 14.27 | 11.01 |
| cross subject | 4 | 61.29 | 64.71 | 63.73 | 64.09 | 14.27 | 10.10 |
| cross subject | 8 | 65.71 | 69.37 | 68.76 | 68.85 | 14.27 | 9.21 |

Same-subject k>0 averages contain MotionSense and RealWorld; Shoaib has no valid paired same-subject
support cohort. Cross-subject averages contain all three datasets and 17,837 query windows per row.

The intervention columns establish mechanism use. For example, cross-subject k=8 falls from 65.71
F1 to 14.27 when support is removed and to 9.21 when support receives incorrect names. The model is
using both the physical examples and the support-to-name binding.

### Coherent names

| relation | k/shape | learned engine | identity vote | prototype | ridge |
|---|---|---:|---:|---:|---:|
| same subject | k=0 | 31.54 | 33.49 | N/A | N/A |
| same subject | k=1 full | 73.93 | 78.92 | 82.82 | 79.66 |
| same subject | k=2 full | 77.56 | 81.33 | 80.75 | 79.35 |
| cross subject | k=0 | 10.09 | 24.56 | N/A | N/A |
| cross subject | k=1 full | 50.52 | 50.82 | 54.84 | 54.49 |
| cross subject | k=2 full | 55.65 | 56.70 | 59.40 | 58.83 |
| cross subject | k=4 full | 61.63 | 64.78 | 63.73 | 64.09 |
| cross subject | k=8 full | 65.92 | 69.54 | 68.76 | 68.85 |

The nearly identical positive-k coherent and arbitrary-name curves show that enrolled evidence, not
activity-name semantics, drives the successful adaptation path. Coherent partial enrollment also
generalizes despite not being trained directly: cross-subject F1 rises from 35.22 at k=1 to 39.91
at k=8, compared with the 10.09 zero-support floor.

## Comparison With the Failed Minimal Run

The v21 run used coherent names for every episode. It reached approximately 0.94 training BA while
held-out BA stayed near 0.19; support removal and label shuffling were nearly inert. External
cross-subject F1 only rose from 11.86 at k=0 to 14.48 at k=8.

Under v22, positive-support candidate names are arbitrary. External coherent-name cross-subject F1
now rises from 10.09 at k=0 to 50.52, 55.65, 61.63, and 65.92 at k=1,2,4,8. This isolates the former
failure: coherent positive-support episodes allowed direct activity classification and did not make
support binding necessary.

Zero-support quality did not improve. It fell modestly on the external development aggregate
(11.86 to 10.09 cross-subject; 34.04 to 31.54 same-subject) and more substantially on the fixed
held-family canary. Uniform sampling over k=0..8 allocates only about one ninth of episodes to the
semantic path, while every positive-k episode trains arbitrary-name binding.

## Historical Limitations

1. The learned engine remains about 3-5 macro F1 points below prototype/ridge controls on most
   cross-subject cells. Retrieval and evidence interpretation therefore still leave usable
   representation quality on the table.
2. Coherent k=0 generalization is weak, especially across subjects. Future work should rebalance or
   separate the semantic path without weakening the successful adaptation objective.
3. Configuration-only internal transfer is markedly weaker than subject-only transfer. The current
   external development roster has no genuine cross-configuration enrollment cohort, so that claim
   cannot yet be tested adequately.
4. This is one Phase-B seed selected on development canaries. Replicates are required before opening
   the sealed test roster.
5. Arbitrary labels are intentionally unanswerable at k=0 and are therefore evaluated only when
   support is present.

## Historical Artifact Index

Historical v22 run:

- root: `training/evidence/outputs/phase_b_v22_alias_support_20260811/`
- selected predictor: `patch_evidence_predictor.pt`
- resumable final state: `patch_evidence_predictor.last.pt`
- milestone predictors: `patch_evidence_predictor.milestones/`
- complete training log: `train.log`
- raw telemetry and rendered plot: `telemetry/`
- arbitrary-label development result: `eval_alias_dev.json`
- coherent-name development result: `eval_coherent_dev.json`

Superseded runs remain available for forensic comparison:

- coherent-only v21: `training/evidence/outputs/phase_b_v21_minimal_20260811/`
- earlier complex v20: `training/evidence/outputs/phase_b_v20_20260811/`
- historical diagnostics: `training/evidence/outputs/diagnostics/phase_b_20260808/`

## Historical Interpretation Boundary

This result demonstrates learned memory adaptation on held development datasets. It does not
establish superiority over direct prototype/ridge adaptation or validate the current design. The
later authorized test readout in Section 9 is a fixed historical analysis, not a development target.

## 9. Phase-A Checkpoint Study (2026-08-17)

This study asks whether a learnable Phase-B decoder can recover information that appears weaker to
closed-form patch retrieval at the 30,000-step Phase-A checkpoint. It is a controlled experiment
with the parked relational decoder, not a change to the current per-sensor admissibility design.

Two independent Phase-B runs used the recipe in Section 2 and seed `20260725`. The Phase-A encoder
was frozen. The step-4,000 arm selected its Phase-B step 3,000 after 692 seconds. The Phase-A
step-30,000 arm selected its Phase-B step 400 after 693 seconds; later training increased the
training-validation gap and did not recover its early selection score. Both runs passed the support
removal and support-label shuffle mechanism checks at their selected checkpoints.

### External development results

The table is query-weighted over MotionSense, RealWorld, and Shoaib. `Identity` is the untrained
retrieval vote using the same bank and enrollment protocol. The arbitrary-name condition removes
useful label semantics and therefore isolates adaptation through enrolled support.

| names | shape | k | step-4k engine | step-4k identity | step-30k engine | step-30k identity |
|---|---|---:|---:|---:|---:|---:|
| coherent | full | 1 | 53.69 | 54.62 | 44.20 | 44.93 |
| coherent | full | 2 | 58.15 | 60.53 | 49.20 | 51.57 |
| coherent | full | 4 | 61.93 | 64.70 | 56.29 | 59.16 |
| coherent | full | 8 | 68.43 | 71.19 | 63.35 | 67.38 |
| coherent | partial | 1 | 35.95 | 34.26 | 29.27 | 31.71 |
| coherent | partial | 2 | 38.17 | 37.23 | 31.72 | 33.27 |
| coherent | partial | 4 | 39.16 | 38.15 | 34.75 | 35.83 |
| coherent | partial | 8 | 40.78 | 40.07 | 35.86 | 37.10 |
| arbitrary | full | 1 | 51.12 | 54.03 | 44.12 | 46.34 |
| arbitrary | full | 2 | 57.34 | 60.12 | 48.79 | 51.99 |
| arbitrary | full | 4 | 64.18 | 65.87 | 56.30 | 59.99 |
| arbitrary | full | 8 | 69.97 | 73.65 | 63.24 | 69.47 |

The step-4,000 representation is better in every positive-support development cell. Its learned
decoder improves over identity retrieval only under partial coherent enrollment; it remains below
identity under full enrollment and arbitrary names. The result does not support the hypothesis that
the 30,000-step representation merely needed a learnable Phase B to expose superior information.

### Frozen test results

The test roster contains InclusiveHAR, USC-HAD, TNDA-HAR, UT-Complex, MONIPAR, SPAR, and Upper Limb
Use. Cells above a dataset's paired-support ceiling are omitted. The table is query-weighted over
the remaining cells. These values are a fixed readout, not a development target.

| names | shape | k | step-4k engine | step-4k identity | step-30k engine | step-30k identity |
|---|---|---:|---:|---:|---:|---:|
| coherent | full | 1 | 31.25 | 33.54 | 32.24 | 33.08 |
| coherent | full | 2 | 36.25 | 40.08 | 36.64 | 38.47 |
| coherent | full | 4 | 39.59 | 45.40 | 40.82 | 43.65 |
| coherent | full | 8 | 43.47 | 48.93 | 44.54 | 47.90 |
| coherent | partial | 1 | 23.15 | 24.63 | 20.79 | 20.73 |
| coherent | partial | 2 | 25.86 | 28.07 | 23.67 | 24.30 |
| coherent | partial | 4 | 27.68 | 30.19 | 26.58 | 27.64 |
| coherent | partial | 8 | 28.66 | 30.91 | 28.26 | 29.70 |
| arbitrary | full | 1 | 32.67 | 32.09 | 32.23 | 33.38 |
| arbitrary | full | 2 | 39.43 | 38.60 | 36.99 | 38.82 |
| arbitrary | full | 4 | 43.86 | 44.07 | 41.19 | 43.81 |
| arbitrary | full | 8 | 47.29 | 48.27 | 44.82 | 48.01 |

The strongest adaptation-specific result is the step-4,000 arbitrary-name test: the learned engine
beats identity at k=1 and k=2, then is approximately tied at k=4 and k=8. This is modest evidence
that the decoder can use unfamiliar enrolled names. It is not a general improvement: coherent
full-enrollment and most individual datasets still favor identity, prototype, or ridge controls.
The step-30,000 engine occasionally exceeds the step-4,000 engine under coherent full enrollment,
but its own identity and prototype controls remain lower. The raw representation therefore still
favors step 4,000.

### Artifacts

- root: `training/evidence/outputs/phase_a_checkpoint_selection_20260816/`
- selected predictors: `step4000/learned_relational.pt`, `step30000/learned_relational.pt`
- raw development/test evaluations: `step*/eval_learned_*_{dev,test}.json`
- query-weighted tables: `comparison_learned_*.{csv,json}`
- training telemetry: `step*/telemetry_relational/`

This historical checkpoint comparison remains one-seed and should not be used for a stable effect
estimate. The current admissibility result uses the five-seed suite at the top of this document.

## 10. Frozen HARNet Representation Control (2026-08-17)

This control tests whether HARNet's released frozen representation supports better enrollment than
HALO's Phase-A step-4,000 representation. It uses the same cross-subject candidate sets, nested
execution support prefixes, query windows, seed, and macro-F1 implementation as Section 9. No HARNet
classifier or HALO evidence-engine parameter is fitted. The compared rules are nearest labeled
support, normalized candidate prototypes, and a deterministic L2 ridge head.

Query-weighted development results over MotionSense, RealWorld, and Shoaib:

| k | HARNet nearest | HARNet prototype | HARNet ridge | HALO identity | HALO prototype | HALO ridge |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44.20 | 44.20 | 43.61 | 54.62 | 57.28 | 55.06 |
| 2 | 49.06 | 48.03 | 48.74 | 60.53 | 58.43 | 57.34 |
| 4 | 52.45 | 53.35 | 54.35 | 64.70 | 61.38 | 60.71 |
| 8 | 57.09 | 57.38 | 61.08 | 71.19 | 67.65 | 67.84 |

The already-authorized frozen test readout gives the same conclusion:

| k | HARNet nearest | HARNet prototype | HARNet ridge | HALO identity | HALO prototype | HALO ridge |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 31.47 | 31.47 | 31.65 | 33.54 | 36.24 | 35.32 |
| 2 | 35.00 | 34.33 | 35.18 | 40.08 | 40.04 | 38.94 |
| 4 | 39.11 | 38.99 | 41.32 | 45.40 | 43.91 | 43.57 |
| 8 | 43.38 | 41.79 | 46.39 | 48.93 | 45.91 | 46.34 |

HARNet is competitive on individual datasets, especially MotionSense, and its ridge head essentially
ties HALO ridge at test k=8. It does not provide a stronger support-conditioned representation
overall. HALO leads clearly at low k and its identity retrieval remains strongest at every aggregate
k. The current positive-support ceiling is therefore primarily in evidence interpretation and
retrieval use, not an obvious representation deficit relative to HARNet.

This control does not explain k=0. HARNet's historical zero-shot row includes a supervised global-
vocabulary probe and ConSE bridge, while this experiment deliberately removes that classifier. The
large k=0 gap should consequently be investigated as semantic grounding and candidate scoring before
it is attributed to the frozen representation.

Artifacts:

- evaluator: `training/evidence/eval_harnet_enrollment.py`;
- development: `training/evidence/outputs/representation_controls/harnet_enrollment_dev.json`;
- test: `training/evidence/outputs/representation_controls/harnet_enrollment_test.json`.

## 11. Matched Encoder-Swap Experiment (2026-08-22)

This experiment asks whether HALO's signal encoder limits the evidence engine. It has exactly three
arms: randomly initialized HALO trained end to end, released HARNet frozen, and released UniMTS
frozen. The baseline arms train only a 512-to-128 projection and the same evidence engine used by
HALO. Candidate sets, episodes, loss, retrieval, mixer, vote, seed, and validation are otherwise
identical.

The common input contract is deliberately conservative: one duration-weighted row per sensor per
six-second source window, complete gravity-present accelerometer triads only. Gyroscope rows are not
fed to accelerometer-only pretrained models, and gravity-removed KU-HAR/XRF-AirPods rows are absent
from all three arms. HARNet receives anti-aliased 30 Hz input with its released center-crop/wrap-pad
rule; UniMTS receives anti-aliased 20 Hz input, its released right-wrap/truncate rule, g-to-m/s2
conversion, and the placement-specific SMPL joint used by its released adapter.

### One-hour matched screen

The corrected GPU profile uses four independently constructed episodes per optimizer step, 128
background windows per episode, top-k 64, BF16, and four data workers on the RTX 4090. The frozen
trunks are evaluated in large batches; UniMTS additionally uses a static-shape compiled forward
whose outputs agree with eager BF16 inference (minimum per-row cosine 0.999992). Compilation is a
runtime detail and is not stored in checkpoints.

| arm | measured ms/step | peak allocated VRAM | 6,000-step train estimate |
|---|---:|---:|---:|
| HALO, random-init end to end | 55.5 | 0.49 GiB | 5.6 min |
| HARNet, frozen trunk | 87.6 | 0.36 GiB | 8.8 min |
| UniMTS, frozen trunk | 316.3 | 3.69 GiB | 31.6 min |

Seven validation passes (step 0 and every 1,000 steps), startup, calibration, and compilation add
several minutes across the three sequential runs. The completed screen took 50.4 minutes. This is a
representation **screen**, not a converged final comparison:
128 is smaller than the production 512-row bank and 6,000 steps is shorter than the 35,000-step
mature schedule. If an arm is still improving at step 6,000, it must be extended before reporting a
final encoder ranking.

Launch each arm sequentially, changing only `--encoder-backbone` and the output directory. HALO uses
the current fixed physical filterbank; the learnable-frontend experiment is not part of this test.

```bash
python -m training.tokenizer.pretrain_episodic --random-init --encoder-comparison \
  --encoder-backbone ours --steps 6000 --warmup-steps 500 --bank-windows 128 \
  --val-every 1000 --out training/tokenizer/outputs/encoder_comparison_screen/ours
python -m training.tokenizer.pretrain_episodic --random-init --encoder-comparison \
  --encoder-backbone harnet --steps 6000 --warmup-steps 500 --bank-windows 128 \
  --val-every 1000 --out training/tokenizer/outputs/encoder_comparison_screen/harnet
python -m training.tokenizer.pretrain_episodic --random-init --encoder-comparison \
  --encoder-backbone unimts --steps 6000 --warmup-steps 500 --bank-windows 128 \
  --val-every 1000 --out training/tokenizer/outputs/encoder_comparison_screen/unimts
```

The earlier `training/tokenizer/outputs/encoder_swap_20260822` pilot is superseded: it duplicated one
baseline window vector across patch rows, fed gyroscope values through accelerometer trunks, omitted
UniMTS placement, admitted gravity-removed streams, and could not reconstruct baseline checkpoints.
Do not use its scores. CPU smoke runs of all three corrected arms, a GPU UniMTS train/validation
smoke, checkpoint reconstruction, and matched GPU profiles pass.

### Screen results

The table reports each arm's checkpoint selected on the mean of the four **coherent** validation
cells. Random aliases were disabled in training, so their diagnostic curve is not used for checkpoint
selection. All arms have corpus fingerprint `e673381a7cb18517`, source head `2c5098e`, and source-patch
hash `04e9cc84...f536`.

| arm | best step | k=0 | k=1 | k=2 | k=4 | coherent mean | initial mean | wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HALO, random-init end to end | 5,000 | 0.4806 | 0.4950 | 0.4780 | 0.4903 | 0.4860 | 0.3017 | 7.70 min |
| HARNet, frozen trunk | 5,000 | **0.5078** | 0.4210 | 0.5148 | 0.5009 | 0.4861 | 0.3335 | 9.30 min |
| UniMTS, frozen trunk | 4,000 | 0.4609 | **0.5115** | **0.5182** | **0.5315** | **0.5055** | **0.4218** | 33.39 min |

The screen gives a qualified answer. UniMTS is best overall by 0.0194 macro-F1, with its advantage
concentrated at k >= 1. HARNet is best at k=0. HALO catches HARNet despite learning its encoder from
scratch, so these results do not support the claim that HALO's encoder is the sole or dominant
bottleneck. They do support a smaller claim: a strong pretrained UniMTS representation improves the
engine's enrollment behavior under this short budget.

The non-monotonic curves, especially HARNet k=1 below k=0, also show that the common evidence engine
does not yet use additional support reliably for every representation. This is a held-out-concept
validation screen on a 128-row bank, not the paper's external test suite and not a mature 512-row,
35,000-step comparison. Do not promote the ranking to a final model claim without the external
evaluation and, for any still-improving arm, a longer matched run.

Artifacts are under `training/tokenizer/outputs/encoder_comparison_screen/{ours,harnet,unimts}`.
Each directory contains `best.pt`, `last.pt`, `log.jsonl`, `run_config.json`, `summary.json`, and the
captured source provenance. The interrupted pre-fix UniMTS pilot was removed from this results path.
