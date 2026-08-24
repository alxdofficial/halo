# HALO — design of record

Agreed 2026-08-11; Phase-A recipe updated 2026-08-18. Supersedes the Phase-A/B design in `PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md`
and `EVIDENCE_ENGINE_*` for anything they disagree on.

> ## ⚠️ Phase-B sections are SUPERSEDED — 2026-08-22
>
> Phase A here is current. **Every Phase-B mechanism section below — the admissibility gate, the
> joint retrieval score `feature cosine / temperature + log(admissibility)`, the closed-form Stage-1
> predictor and its artifacts — describes a design the code no longer implements.** The active
> design is the compact evidence engine
> ([`COMPACT_EVIDENCE_ENGINE.md`](COMPACT_EVIDENCE_ENGINE.md)): plain `cos(patch_q, patch_m)/0.07`
> retrieval with **no gate**, a full-memory soft vote, and a top-64 scalar evidence reranker. The
> reranker can change only one score per retrieved row; the final text/identity vote remains fixed.
> At width 128 the end-to-end model has 869,900 trainable parameters, of which 228,108 belong to
> Phase B. The primary experiment trains two 228,108-parameter rerankers from one identical frozen
> encoder: a zero-shot head on k=0 and an enrollment head on k=1/2/4/8/16. End-to-end training and a
> unified head remain explicit controls rather than the current default claim.
>
> **The thesis paragraph below needs the same care.** Its centrepiece — configuration calibrating
> the *admissibility* of retrieved evidence, so a pocket phone cannot vote on an arm gesture — is
> the claim the deleted gate was built to instantiate. The current reranker can learn soft trust
> from query/evidence descriptions, but has no separately fitted or hard admissibility gate. The
> measured result that retired the explicit gate is in
> `../results/PHASE_B_TRAINING_STATUS.md`: learned admissibility had no held-out advantage over
> setting admissibility to one. Constraint 4 below ("learned readouts have lost to their closed
> forms twice") is the honest frame, and the compact engine is the third attempt at beating them.
>
> **Three further Phase-A corrections (2026-08-22), each verified against code:**
> 1. The `retrieval_tokens` "shallow path" paragraph is stale. Under `trunk="temporal"` the encoder
>    forces `use_sensor_isolated_retrieval=False`, so the shallow branch never fires and the
>    retrieval row **is the full 3-layer trunk output, taken after descriptor conditioning**. The
>    "816k of 7.5M parameters" figure and the depth-1-vs-3 ablation refer to nothing in the code.
>    Sensor isolation is now supplied by the trunk architecture instead — a correct property,
>    reached by a different mechanism than described. (Same error in
>    `PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md`'s VICReg sentence.)
> 2. "Cross-sensor attention operates within a placement in Phase A" — `TemporalTrunk` has no
>    cross-sensor attention.
> 3. "The reference recipe adds only jitter and scale" — as coded the recipe is **fully clean**:
>    `AugmentationConfig.phase_a()` overwrites both probabilities to 0.0 and the CLI flags default
>    to `None`. The 0.5 values are dataclass defaults that never take effect unless passed.
>
> What the current evidence supports instead is in
> [`../results/ADAPTATION_TABLE_20260822.md`](../results/ADAPTATION_TABLE_20260822.md): a compact
> encoder whose frozen features win 35/40 enrollment columns. That is a representation claim, not
> an admissibility claim. Do not carry this thesis paragraph into a draft unmodified. Written to be re-read at the start of a
session; every non-obvious choice carries the reason it was made, because most of them were reached
by eliminating an alternative that looked better.

## Thesis

**Sensor heterogeneity should not be abstracted away. It should be presented alongside the numerical
features, because the retriever must rely on it to decide which retrieved evidence is admissible.**

Contribution statement (for MOTIVATION.md; describes the target, not the current build):

> HALO encodes acquisition configuration rather than normalising it away: placement, modality, and
> sampling regime enter the representation alongside the signal, and the tokenizer marks explicitly
> what a given configuration **cannot resolve**. Classification is retrieval over a bank of labelled
> recordings, with configuration calibrating the admissibility of each retrieved element — a phone in
> a pocket is strong evidence about gait and none at all about what the arms are doing, so however
> similar its signal, it must not vote on an arm gesture. Episodic training varies placement,
> configuration, and label coverage within the retrieval set, so admissibility is learned rather than
> assuming every geometric neighbour is comparable — which is what lets HALO enrol a concept once, on
> whatever hardware is at hand, and report honestly when a deployment cannot observe it.

## Constraints that killed earlier designs — do not re-propose these

1. **No cross-placement generation or transport.** A wrist recording does not contain what the ankle
   saw; carrying a bag changes the wrist and not the ankle, so the map is not a function. Any design
   requiring a model to infer one placement's signal from another is unsound, not merely hard.
2. **Same-site cross-modality IS well-posed.** Accel and gyro at one placement observe the same rigid
   body synchronously. This is the only sensor-to-sensor prediction permitted.
3. **Mechanism novelty is unavailable.** Retrieval, attention over retrieved sets, gating, reranking,
   two-space (ConSE) scoring — all claimed, all checked (see `LITERATURE_HAR_LLM_2026-08.md`).
   The contribution is the admissibility claim and the system that instantiates it, not a new module.
4. **Learned readouts have lost to their closed forms twice** (v19 inert; v20 45.5 vs 65.4). Closed
   form is the design of record until a learned version beats it on a pre-registered gate.
5. **The language-interface negative result is taken** (Haresamudram et al., AAAI 2025 — NLS is
   30-40% below supervised/SSL; causes: sensor heterogeneity + description poverty). Our version is a
   sharper *mechanistic* section, never a headline.

---

# Phase A — the tokenizer

## Data unit

A **sensor** = one modality triad at one placement (accel xyz, or gyro xyz). A 6-channel stream is
**two sensors**. `sensor_id` (already in `pretrain_data.py`) is the grouping.

Per (patch, sensor) the system carries three records, but only the first two enter the encoder:

| name | contents | provenance |
|---|---|---|
| `feature` | filterbank band energies + signed DC + amplitude + observability masks | per patch, computed |
| `text_descriptor` | SBERT of "accelerometer of a smartwatch on the left wrist" | per sensor, frozen artifact |
| `sensor_bias` | activity-invariant channel physics; Phase-B bank metadata only | per sensor, offline closed-form statistics |

`text_descriptor` gets a learnable gated projection into `d_model`. `sensor_bias` is deliberately
excluded from the Phase-A trunk: it is strongly predictive of source dataset and is unavailable for
a truly novel device until statistics have been estimated. It remains an auditable Phase-B bank
field whose retrieval weight defaults to zero.

## Front end — keep

Constant-Q physical-Hz filterbank (32 bands, 0.3-15 Hz, Q=4, `DFT_SIZE=256`), the **Nyquist
observability mask** (`center + 2σ ≤ 0.9·rate/2`), the **low-frequency resolution mask** (cycles
fitting in the window), an amplitude scalar, and the signed DC feature with frozen standardisation.

The two masks are the built instantiation of the thesis at the band level: they state what this
configuration cannot resolve. Nothing else in the wearable literature does this — LSM-2 models data
that is *absent*, not data that is *present but insufficient*.

## Front end — change

- **Fold xyz into one sensor token.** Was per-channel `(B,P,C,d)`, becomes per-sensor `(B,P,S,d)`.
  3x shorter sequences on 6-channel streams. **Requires a per-axis validity mask inside the feature
  block** — otherwise mhealth's degenerate gyro and channel-dropout events silently poison the token
  instead of dropping it.
- **Drop role text.** Axis order is fixed inside the token; modality moves into the sensor text. The
  role/sensor factoring (`FactoredChannelTextFusion`) collapses to sensor-only.
- **Drop the separate patch-duration numeric input.** The resolution mask already encodes duration
  per band, in the structurally correct form, and physical-time RoPE carries temporal extent. A third
  copy is redundant.

## Trunk

~~`DualBranchTransformer`, **3 layers**~~ — **corrected 2026-08-22:** the trunk of record is
`TemporalTrunk`, 3 layers, with **no cross-sensor branch at all**. Phase B hard-rejects anything
else (`pretrain_episodic.py` exits unless `trunk="temporal"` and `token_granularity="sensor"`). Physical-time RoPE in seconds,
never patch index. Sensors carry no positional index; identity is text, so sensor count and order
are free.

The encoder exposes two representations. `retrieval_tokens` apply the first layer's TEMPORAL
sub-block only -- temporal attention plus its residual norm, applied independently per sensor,
before descriptor conditioning and before any cross-sensor mixing. These are the rows stored by
Phase B, so an accelerometer row is unchanged when a gyroscope is absent. The main `tokens` path
uses all layers with cross-sensor attention; it supplies JEPA targets and pooled context.

Two properties of the stored row are bundled in one flag and have very different standing:

* **Sensor isolation is principled.** The bank must be queryable by an accel-only device against
  rows built from six-channel streams; a cross-sensor-mixed row would carry gyroscope information
  the query cannot have. `test_retrieval_rows_are_sensor_isolated` asserts it.
* **Depth 1, and skipping the block's feed-forward network, are NOT design decisions.** They are
  what `temporal_context` happens to compute. The stored path is ~816k of the encoder's 7.5M
  parameters. It currently measures better than full depth (+0.071 on trained weights, -0.001 at
  random init) but that gap is CREATED BY TRAINING, i.e. it reflects the objective shaping the deep
  path wrongly rather than a virtue of shallowness. Retrieval depth 1 vs 2 vs 3, and whether to
  include the FFN, are open ablations.

Cross-sensor attention operates **within a placement** in Phase A. Cross-*placement* fusion is a
Phase-B vote-merge, never an attention operation (constraint 1).

## Augmentations

The reference recipe adds only **jitter and scale** (`--jitter-p`, `--scale-p`), which perturb
amplitude and sensor noise without touching gravity-frame orientation. Every other transform is a
controlled one-at-a-time ablation exposed on the CLI: `--rotation-p` / `--rotation-pairing`,
`--rate-augmentation-p`, `--channel-dropout-p`, `--gravity-p`, `--channel-text-phrase-p`,
`--channel-text-dropout-p`.

Rotation is shared across positive views by default. Independent rotation is an explicit invariance
control and is **known-harmful**: it trained the encoder 0.085 BELOW a random-init trunk, confirmed
three ways (per-dataset regression pattern, one-variable pilots, and the posture canary, where
Shoaib fell 1.000 -> 0.703). It erases the gravity-frame posture and limb-orientation information
the signed DC feature exists to preserve.

Measured 2026-08-18, and recorded so it is not re-litigated: adding the REST of the historical
stack -- gravity, rate, channel dropout, channel-text paraphrase and dropout -- changed held-out
transfer by less than the noise floor, and did NOT reduce subject leakage (~0.55 against the
old-good checkpoint's 0.3146). Whatever gave that checkpoint its selectivity, it is not the
augmentation stack, not multi-resolution, and not descriptor reconstruction; all three were tested
and eliminated.

The JEPA teacher consumes view A. In the clean reference, VICReg's invariance term is zero by
construction and its variance/covariance terms act as collapse and redundancy controls.

## Objectives

**JEPA** — latent masked prediction against an EMA teacher, two active mask strategies:

1. **Time-mask** (existing): mask physical-time intervals of `feature`.
2. **Sensor-mask** (new): mask *all* of one sensor's channels; reconstruct from the other sensor plus
   the masked sensor's `text_descriptor`. `[MASK]` is applied **before** fusion so the
   model knows *which* sensor it must reconstruct. **Restricted to same-placement pairs, enforced
   structurally in the mask planner, with a test.** This is the well-posed successor to the
   cross-placement objective deleted 2026-08-06 — that one was ill-posed by constraint 1; this one is
   rigid-body kinematics.
Descriptor-mask retrieval remains available as an ablation but has zero probability and zero weight in
the reference run.

**Masked physical reconstruction (MAE) — implemented, tested, NULL.** `--mae-weight` replaces JEPA's
EMA-latent target with masked reconstruction of the parameter-free `filterbank.analyze` output
(`losses_repr.masked_analysis_reconstruction_loss`). Motivation: HALO is the only model in our
comparison set whose pretraining target is produced by the model itself, and a self-referential
target is satisfiable by encoding any consistently-reproduced factor, including subject identity.

The 2026-08-18 2x2 measured the main effect of MAE-vs-JEPA at **+0.0027**, against a 0.012 noise
floor -- undetectable, in both the with-fixes and without-fixes cells.

**Why it was a null experiment, measured afterwards:** the target is ~90% predictable for free. A
masked patch's analysis features are recoverable from its neighbours by linear interpolation --
averaging the two adjacent patches already achieves 91.2% of the variance reduction on MotionSense
and 83.5% on RealWorld. Filterbank band energies over one second of human motion are slowly varying,
so the objective asks the encoder to learn what a two-tap average does. Contrast LiMU-BERT, which
reconstructs the normalised RAW waveform: not interpolable, because it requires phase.

HALO cannot currently run raw-signal reconstruction -- the encoder's input is already magnitude-only
(`analyze` discards phase), so the waveform is not recoverable from what it sees. Before concluding
anything about masked reconstruction for HALO, mask contiguous RUNS of patches (2-4 s) so
interpolation cannot bridge the gap, and re-measure the interpolation baseline at that span. **VICReg** retains its variance and covariance terms for collapse prevention. It
is split equally between pooled contextual embeddings and the raw 256-dimensional retrieval rows;
this prevents a healthy projector from hiding collapse in the representation Phase B stores.

## `sensor_bias`

Computed **offline**, per sensor, over Phase-A training subjects only. **Activity-invariant channel
physics only** — pooling over a sensor whose dataset has a skewed activity distribution would
otherwise make this a dataset fingerprint.

| field | computed as | catches |
|---|---|---|
| gravity magnitude | median ‖acc‖ over quiescent patches | recgym min-max ([0,1] → nonsense) |
| gravity presence | median norm of the per-window DC vector | gravity-removed streams |
| noise floor | quiescent energy above the motion band (>~20 Hz is instrument, not person) | sensor grade |
| quantization step | min positive gap in sorted unique value diffs | effective bits |
| clipping | fraction at observed rails | range limits |
| rate fidelity | source acquisition rate / stored array rate from converter metadata | resampling |
| rest bias | median quiescent sensor norm | gyro offset / accelerometer rest magnitude |

Quiescent-patch selection = low-variance percentile per stream. Every field is a **norm or aggregate**
— which is why rotation leaves the descriptor unchanged. That is by construction, not luck.

The artifact carries seven standardized values plus seven support bits. Unsupported measurements
therefore differ from a real value at the corpus mean. Implausible and byte-duplicate windows are
excluded before statistics are estimated.

**BUILT 2026-08-12** — `data/scripts/curate/sensor_bias.py`, artifact
`data/scripts/curate/sensor_bias.json`: **110 sensors across the 18-source expanded Phase-A corpus**.
Normalisation uses training subjects only (58 validation subjects excluded), at most 4,000 valid
windows per stream. The artifact persists the dataset roster, data seed, and validation-subject hash.
It is validated by Phase-B artifact builders, not by the Phase-A trainer.

**Guard status: INCONCLUSIVE, and it cannot be made conclusive on this corpus.**
Nearest-neighbour dataset purity is 0.4211 against a 0.0833 chance. Restricting neighbours to
different placements gives 0.4474 (all 38 scored). This changes the statistic but does not resolve
the central confound: acquisition hardware and processing are generally dataset-specific, and the
corpus lacks an independent same-hardware control across studies. The check is therefore a monitor,
not a pass/fail guard.

The decisive guards are downstream, where the confound does not apply:

1. **Encoder probe — RUN 2026-08-12. FIRES.** `training/tokenizer/probe_provenance.py`, group-disjoint
   (subject) linear readouts over the same embeddings:

   | probe | balanced accuracy | classes |
   |---|---:|---:|
   | dataset identity | **0.7019** | 32 |
   | activity, class-count matched | **0.5002** | 32 |
   | margin | **+0.2017** | |
   | within-placement provenance, excess over chance | 0.5814 | 13 placements |

   At matched class count the encoder reads *which study this is* 20 points more sharply than *what
   the person is doing*, and provenance stays readable even among streams sharing a body location.

   **A correction on the metric:** the first version compared excess-over-chance across a 32-class
   and a 211-class problem and reported a 4.3x ratio. That is not a fair comparison — more classes
   lowers chance while making the task harder, so the two excesses are not on one scale. The
   class-count-matched margin above (+0.20) is the honest number, roughly half the impression the
   ratio gave.

   **Decision:** source provenance already existed without the artifact. The next Phase-A recipe
   removes `sensor_bias` from the trunk and makes retrieval rows sensor-isolated; rerun this probe on
   every candidate checkpoint.
2. **Phase-B retrieval-provenance guard** — does enabling the bias blend shift retrieval toward the
   query's own dataset, against a placement-matched baseline? Measures the harm directly rather than
   proxying it. **Not yet built.**

## Gates

### PARITY GATE — RUN 2026-08-11. Result: **the conditioning is inert.**

`training/tokenizer/outputs/parity_ablation/parity.json`, checkpoint `phase_a_headline/best.pt`
(step 27,000), held-out-config transfer kNN-BA, same subject holdout in both arms:

| dataset | full conditioning | neutral text | gain |
|---|---:|---:|---:|
| motionsense | 0.791 | 0.756 | +0.035 |
| realworld | 0.649 | 0.656 | **−0.007** |
| shoaib | 0.896 | 0.831 | +0.065 |
| inclusivehar | 0.429 | 0.486 | **−0.057** |
| **mean** | **0.691** | **0.683** | **+0.0086** |

Against the 3k-step screening noise floor (sd 0.0065; nothing under ~0.012 is real) the mean gain is
~1.3 sd, and **the sign flips on two of four datasets**. A real effect does not change sign across
half the cohort. Placement, device and gravity text currently contribute nothing.

Note the direction of the test strengthens the conclusion: the model was *trained* with full text and
is *evaluated* with neutral text, so the parity arm is out-of-distribution input. A model that had
learned to depend on placement would degrade. It does not.

This is the measured motivation for the entire Phase-A redesign below — specifically for
descriptor-mask JEPA, which exists to make the descriptor load-bearing, and for removing
`sensor_text_dropout`, which trained the model to cope without it. It is also a reportable finding in
its own right: conditioning-by-concatenated-text, as the field does it, does not survive contact with
a parity control.

| gate | when | criterion |
|---|---|---|
| ~~parity ablation~~ | **DONE 2026-08-11** | **+0.0086, inert.** See above. Re-run after the retrain; the redesign has to move this number or it has failed. |
| fingerprint probe | after retrain | `feature` → dataset ID at chance |
| screening run | before full retrain | 3k steps against noise floor sd 0.0065 (nothing under ~0.012 is real) |
| sensor-text on/off | after retrain | if off loses nothing, sensor text becomes bank-only and the trunk story simplifies |

---

# Resolvability — implementation ready; current measurement pending

The earlier `resolvability.json` was exploratory and is rejected by the current loader. It cannot be
used to fit or validate the current gate for three reasons: it included development and test
datasets, it pooled Phase-A validation subjects into the measurement corpus, and each stream target
pooled accelerometer with gyroscope while the runtime gate acts on one sensor at a time. Its reported
placement gaps motivated the design, but they are not evidence for the current implementation.

The current `training/evidence/resolvability.py` contract is:

- use only datasets recorded in the sensor-granularity Phase-A checkpoint;
- exclude the exact Phase-A validation subjects and all quality-screened windows;
- measure accelerometer and gyroscope independently;
- pool patches within resolution and then average resolutions for each source window;
- use subject-disjoint, one-vs-rest kNN and map chance balanced accuracy to zero; and
- report simultaneous-placement contrasts separately.

The label string is only a grouper in this measurement. No language embedding enters the score. A
new scientific result must be produced from the completed sensor-granularity Phase-A checkpoint.

## The admissibility gate

`training/evidence/admissibility_gate.py`. The train-only measured table is the gate's warm-start
data. It is not an independent validation set after fitting. Held-out concept, dataset, and body-
region folds in `gate_extrapolation.py` are the generalization evidence.

**Why the lookup had to go.** `gate_tensor` keyed on the literal `"<dataset>/<stream>"` string and an
exact label match. Against a novel vocabulary every entry hit the neutral default, so the gate became
a uniform multiplier — and a uniform multiplier **provably cannot change the argmax**. It was doing
nothing in exactly the open-vocabulary case it existed for. Deleted.

**The replacement**, a function of text alone (no curated placement fields, no anchor templates — a
deployment supplies only its sensor's description):

```
u = A · sbert(sensor_text)      A: (r, 384)   "where does this sensor sit"
v = B · sbert(concept_text)     B: (r, 384)   "where does this concept move"
w = sigmoid(u^T LAMBDA v + b)   LAMBDA: (r, r)
```

An additive `f(sensor)+g(concept)` gives a sensor one fixed quality ordering and cannot express cases
where placement A is useful for one concept and placement B is useful for another. The bilinear form
can express that interaction. Rank 1 is a valid signed ablation and can express a two-by-two
inversion; it is not prohibited mathematically. Rank 8 is the current default; ranks 1, 2, and 4
remain registered held-out ablations. Full learned projections are the default; fixed PCA remains an
ablation.

### Prediction path

1. **Hard compatibility** — modality and gravity codes only, so it is label-independent and already
   works unchanged on unseen vocabulary.
2. **Soft admissibility** — score both query and evidence sensor descriptions against the candidate
   and combine them with a scale-preserving geometric mean.
3. **Joint retrieval score** — `feature cosine / temperature + log(admissibility)`, followed by
   per-candidate top-k and a query-wide stabilized exponential that produces evidence mass. This
   preserves candidate-level observability while applying the learned gate exactly once.

`BIAS_BLEND_WEIGHT` defaults to **0**: the provenance probe reads dataset identity out of the pooled
feature at BA 0.702, so a bias-similarity term pulls toward same-dataset retrieval. Enable it only
beside a reported `retrieval_provenance` with the blend on and off.

### Three guards, each making the null observable

| guard | where | fires when |
|---|---|---|
| gate collapse | `AdmissibilityGate.spread`, logged per prediction | the gate flattens to a constant — "admissibility does not help" is a result |
| provenance capture | `retrieval_provenance`, gate on vs off | the gate raises same-dataset retrieval: it learned provenance, not physics |
| latent ↔ measurement | `latent_measurement_correlation` | learned latents do not track the table: the gate found something, but not the claim |

`within_concept_pearson_r` is the load-bearing statistic in the third: it removes per-concept
difficulty and leaves only configuration-dependent structure.

---

# Bank build

Rows are **per patch per sensor**: `[feature, text_descriptor, sensor_bias]` + label + provenance
(`subj`, `cfg`, `event`, `time`, `resolution`). ~7M rows (roughly double, from the per-sensor split).

Provenance and construction metadata are **unchanged** — folds, execution leakage units, and episode
sampling do not care how rows are keyed.

Retrieval rows are computed before cross-sensor attention, so partner-sensor presence is no longer a
latent confound. Partner presence may remain provenance for analysis, but it does not need to be
encoded into the sensor descriptor to explain a representation change.

Payoff: capture24 (accel-only, the largest corpus source) now matches accel queries exactly instead
of smearing a channel-set mismatch into the embedding.

---

# Phase B — Stage 1, fully closed-form

The earlier closed-form control beat the learned decoder, but those numbers used the previous
encoder and bank and are not directly comparable. They motivate this design; Stage 1 must establish
its result again under the current sensor-row protocol.

**Prediction:**

1. **Compatibility filter** — hard: modality accel↔accel / gyro↔gyro, gravity state. Makes the
   comparison meaningful at all. Reads only the row's modality and gravity codes, never the label, so
   it works unchanged on a vocabulary it has never seen.
2. **Soft admissibility** — `AdmissibilityGate` scores the query sensor and evidence sensor against
   the candidate concept. Their geometric mean preserves the fitted gate scale while requiring both
   sides to observe the concept. It enters the retrieval score continuously and is never a fixed
   placement filter or a learned hard exclusion.
3. **Rank** — `feature cosine / temperature + log(admissibility)`. Select top-k separately per
   candidate, then use one query-wide stabilizer across all selected candidate-row choices so query-side
   observability cannot cancel. The `sensor_bias` blend defaults to **0** (the
   provenance probe reads dataset identity at BA 0.702); enable only beside a reported
   `retrieval_provenance` with the blend on and off.
4. **Vote** — enrolled rows vote their bound candidate by identity (aliasing proved this is what
   carries at k≥1); corpus rows vote through label text (the k=0 path).
5. **Merge across the query's sensors** — sum votes, per-sensor weights at most. Cross-placement
   fusion lives here, closed-form.

Larger bank, larger k than the v20 regime. Bigger k also relieves the top-k gradient limitation if
Stage 3 ever runs.

**Stage 1 must answer:**

- Beat prototype and ridge on the same features (the internal fair reference; the old 65.4 is not
  directly comparable across a new encoder and bank).
- **Cross-config enrollment** — enroll on device A, query on device B. This *is* the heterogeneity
  claim, and it is pure evaluation.
- **The 4-minute fine-tuning baseline** (Haresamudram) — the comparison this literature demands.
- **Retrieval-provenance guard** — does enabling the `sensor_bias` blend shift retrieval toward the
  query's own dataset, against a placement-matched baseline? If yes it is fingerprint-matching.

**Stage 2** — optional gate-only refinement. Candidate cross-entropy refines `A`, `B` and `LAMBDA`.
The training path uses a fully soft distribution over every physically compatible row in a bounded,
label-balanced working memory; top-k on the same continuous score is the validation and deployment rule.
The train-only measured table supplies the warm start and a small replay anchor. Query and support
come from distinct executions, candidate labels are removed from corpus memory, and only explicitly
sampled supports are restored. External development results, not fitted cells or training loss,
decide whether Stage 2 replaces Stage 1.

The soft path is intentionally simple. Matching-style soft attention supplies the needed credit
assignment without introducing an optimal-transport or sparse-top-k subsystem. Add a differentiable
sparse selector only if telemetry demonstrates a material gap between soft training and exact hard
validation. The refined gate must also beat its gate-disabled and step-zero controls; otherwise the
closed-form warm start remains the design.

Why this is not the fourth repetition of a failed learning attempt: the three that went net-negative
(decoder 44.2 vs a 46.7 control, Phase-B training below chance, parity +0.0086) were all
**closed-vocabulary classifiers trained with cross-entropy**, which memorise the training label set.
This is ~1.5–3k parameters emitting *one scalar per (config, concept) pair*, constrained linear in a
frozen embedding space. There is nowhere to put a memorised vocabulary. That is an argument from the
parameterisation, which is why all three guards ship before the first run rather than after.

**Stage 3** — tokenizer finetune: frozen-index retrieve → re-forward selected rows with grad →
end-to-end. Needs scheduled bank re-embeds (or a drift threshold on the existing fingerprint probe),
SSL loss retained as a regulariser, learnable projections over `sensor_bias` with frozen raw
statistics, and eligibility gated on beating Stage 1.

**Artifact build pending.** The sensor-granularity Phase-A checkpoint completed on 2026-08-17, so the
model prerequisite now exists. `predictor_mode='admissibility_gate'` still requires a schema-5
bank, train-only schema-2 resolvability table, and bank-bound gate. The persisted files predate the
current checkpoint/vocabulary and are rejected. The wiring fails loudly rather than falling back to
the pooled table, because scoring pooled session vectors under this mode's name would report a
different mechanism.

**Parked, not deleted:** relational decoder, counterfactual objective, retriever training. If a dozen
learned parameters cannot improve a working closed form, that says what a million would do — for a
day of compute instead of a week.

---

## Ledger — build status as of 2026-08-12

**Pre-existing:** filterbank + observability masks, signed DC, dual-branch trunk with physical-time
RoPE, gated conditioning, JEPA + VICReg, episodic construction, execution leakage units,
canaries (support-removed / label-shuffled), prototype + ridge comparators, `live_encoder`, bank
fingerprint guard.

**Built and covered by the current test suite:**

| piece | where |
|---|---|
| parity ablation + `--parity` arm | `training/tokenizer/eval_transfer.py` — **run, result above** |
| augmentation NUISANCE/CONFIG split + `split_by_group()` | `data/scripts/augmentations.py` |
| `sensor_text_dropout` disabled | `data/scripts/augmentations.py` |
| `sensor_bias` computation + guard + train-only artifact (110 sensors) | `data/scripts/curate/sensor_bias.py` |
| `sensor_bias` loader for the data path | `training/tokenizer/pretrain_data.py` |
| `SensorFold` + axis validity mask | `model/tokenizer/sensor_tokens.py` |
| `ConditioningProjection` (learnable MLP over frozen artifact) | `model/tokenizer/sensor_tokens.py` |
| `DescriptorHead` + retrieval-scored loss | `model/tokenizer/sensor_tokens.py` |
| sensor-granularity encoder path (`token_granularity='sensor'`) | `model/tokenizer/encoder.py` |
| sensor-mask + descriptor-mask JEPA planner | `training/tokenizer/losses_repr.py` |
| Phase-B Stage 1 predictor (compat -> soft admissibility/rank -> vote -> merge) | `training/evidence/admissible_retrieval.py` |
| retrieval-provenance guard | `training/evidence/admissible_retrieval.py` |
| `AdmissibilityGate` + warm start + abstention + collapse telemetry | `training/evidence/admissibility_gate.py` |
| held-out body-region/dataset/concept study implementation | `training/evidence/gate_extrapolation.py` |
| gate persistence, bank adapter, `predict_bank` | `training/evidence/gate_predictor.py` |
| `predictor_mode='admissibility_gate'` + schema-5 guard | `training/evidence/eval_enrollment.py` |
| provenance probe on **per-sensor rows**, not just pooled | `training/tokenizer/probe_provenance.py` |
| one shared modality-presence rule (`modalities_present`) | `training/tokenizer/pretrain_data.py` |
| sensor/retrieval/gate tests | `tests/test_sensor_granularity.py`, `tests/test_admissible_retrieval.py`, `tests/test_admissibility_gate.py`, `tests/test_gate_predictor.py` |

**Deleted 2026-08-12, superseded by the gate:**

| piece | why |
|---|---|
| `resolvability.gate_tensor` | string-keyed lookup; on novel vocabulary it defaulted everywhere and became a uniform multiplier that provably could not change the argmax |
| `resolvability_predictor.py` | standalone (placement text, concept text) -> resolvability stage; folded into the gate, which fits the same map jointly with the coupling |
| `admissible_retrieval.admissibility` + `GATE_FLOOR` | replaced by one continuous log-admissibility retrieval prior |

**Trainer integration — built and smoke-tested 2026-08-12:**

- shared config draw across views (`_shared_draw`, generalised `shared_config_seed`)
- `sensor_placement` emitted per batch for physically valid JEPA masking; `sensor_bias` omitted from
  Phase-A batches and retained only for legacy-checkpoint evaluation and Phase-B bank metadata
- `--token-granularity sensor` CLI flag, threaded into cfg and the encoder builder
- `make_sensor_mask_plan` wired, with `descriptor_mask` into both forwards and the
  teacher; `jepa_mask` and the per-resolution diagnostic switched to the sensor presence mask
- descriptor-retrieval loss added to the objective, scored ONLY on sensors whose descriptor was
  actually hidden (an unmasked descriptor was fed to the encoder, so "reconstructing" it is a copy)
- telemetry: descriptor loss/top-1/target rate, gradients for active modules, and effective rank for
  pooled, projector, teacher, and retrieval-row representations

**300-step smoke, both arms, real corpus subset** (5 datasets, batch 16):

| arm | jepa | total | descriptor top1 | AMP skips | val_knn_ba |
|---|---|---|---|---|---|
| sensor | 0.997 → 0.047 | 25.0 → 13.5 | 0.0 → 0.67–1.0 | 2 | 0.2395 |
| channel | 1.031 → 0.041 | 24.2 → 10.6 | n/a | 0 | 0.2789 |

No persistent NaN (the step-1 value is AMP scaler startup). The sensor arm's val is *lower* here, but
300 steps on a 5-dataset subset is far below the 3k-step screening noise floor (sd 0.0065) — it
carries no signal and must not be read as an arm comparison.

**Work after the replacement Phase-A training:**

1. **Build memory only from the development-selected replacement checkpoint.**
   `build_memory.py --sensor-rows` emits `[feature, text_descriptor, sensor_bias]` per patch per
   sensor. Existing banks contain rows from the superseded representation path.
2. **Build the current resolvability table.** The implementation produces train-only per-sensor
   measurements and paired contrasts. The current JSON is historical and intentionally rejected.
3. **Re-run the encoder dataset-ID probe** on pooled and retrieval rows. The old result predates
   sensor-isolated retrieval and direct row-level VICReg.

**Unmeasured and load-bearing:** whether cross-config enrollment works at all; whether the bias term
is physics or fingerprint; whether the redesign moves the parity number off +0.0086.
