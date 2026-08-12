# HALO — design of record

Agreed 2026-08-11. Supersedes the Phase-A/B design in `PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md`
and `EVIDENCE_ENGINE_*` for anything they disagree on. Written to be re-read at the start of a
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

Per (patch, sensor) the model carries three vectors:

| name | contents | provenance |
|---|---|---|
| `feature` | filterbank band energies + signed DC + phase + observability masks | per patch, computed |
| `text_descriptor` | SBERT of "accelerometer of a smartwatch on the left wrist" | per sensor, frozen artifact |
| `sensor_bias` | activity-invariant channel physics | per sensor, offline closed-form statistics |

`text_descriptor` and `sensor_bias` each get a **learnable MLP projection** into `d_model`. The
projections are learnable; the underlying artifacts are frozen. This is what keeps `sensor_bias` a
*measurement* — the moment the statistics themselves are learned, the declared-vs-measured audit
property dies.

## Front end — keep

Constant-Q physical-Hz filterbank (32 bands, 0.3-15 Hz, Q=4, `FB_DFT_SIZE=256`), the **Nyquist
observability mask** (`center + 2σ ≤ 0.9·rate/2`), the **low-frequency resolution mask** (cycles
fitting in the window), the signed DC feature with frozen standardisation, the within-sensor phase
feature.

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

`DualBranchTransformer`: temporal self-attention with **physical-time RoPE in seconds** (never patch
index), plus cross-sensor attention. **Tokens are never merged** — attention makes accel and gyro
mutually aware, which is what makes masked-sensor prediction possible. Sensors carry no positional
index; identity is text, so sensor count and order are free.

Cross-sensor attention operates **within a placement** in Phase A. Cross-*placement* fusion is a
Phase-B vote-merge, never an attention operation (constraint 1).

## Augmentations — split by whether the acquisition changed

| group | members | draw | VICReg invariance |
|---|---|---|---|
| **nuisance** | jitter, small scale, window crop, **text paraphrase** | independent per view | applies |
| **config** | rotation_3d, rate, unit scale, gravity state, sensor dropout | **one draw per window per step, shared across ALL views** | does **not** apply |

"All views" = VICReg view A + VICReg view B + the JEPA teacher's clean view.

**Verified 2026-08-12:** the teacher is NOT a separate augmentation in this codebase — it consumes
the same `patches` tensor as view A (`jepa_teacher.tokenize(patches, ...)`, or the shared fixed-arm
analysis), so it inherits view A's configuration automatically. Sharing the draw between A and B is
therefore sufficient, and the "third view" concern raised while designing this does not apply here.

Implemented by generalising the pre-existing `shared_config_seed` hook, which previously covered only
the sensor-text-dropout *decision*. That was not enough: the physical transforms draw their
PARAMETERS internally (`_random_so3` from `np.random.randn`, rate from `np.random.uniform`, dropout
from `_random.sample`), so a shared decision still left the two views rotated by different rotations
to different rates — the same invariance demand one level down. `_shared_draw` now seeds and restores
both module RNGs around each config transform. Measured: two views of one window come out with
identical rate, channel mask and gravity state, while nuisance draws stay independent.

**Why paraphrase is nuisance, not config:** the grouping rule is "did the underlying acquisition
change," not "which field was touched". Paraphrase changes the wording of an identical
configuration. Keeping it nuisance is what forces the model to work off text *semantics* rather than
memorising ~20 fixed strings — the pressure that has to generalise to an unseen placement description
at deployment.

**`sensor_text_dropout` is REMOVED.** Under VICReg it trained "with descriptor" and "without
descriptor" toward the same representation, which is precisely the mechanism that makes conditioning
inert. Descriptor-mask JEPA supersedes it with a supervised version of the same robustness.

## Objectives

**JEPA** — latent masked prediction against an EMA teacher, three mask strategies:

1. **Time-mask** (existing): mask physical-time intervals of `feature`.
2. **Sensor-mask** (new): mask *all* of one sensor's channels; reconstruct from the other sensor plus
   the masked sensor's `text_descriptor` + `sensor_bias`. `[MASK]` is applied **before** fusion so the
   model knows *which* sensor it must reconstruct. **Restricted to same-placement pairs, enforced
   structurally in the mask planner, with a test.** This is the well-posed successor to the
   cross-placement objective deleted 2026-08-06 — that one was ill-posed by constraint 1; this one is
   rigid-body kinematics.
3. **Descriptor-mask** (new): mask `text_descriptor`, keep `feature` + `sensor_bias`, reconstruct the
   descriptor. **Scored retrievally** — is the true descriptor the nearest among the batch's
   descriptors — not by MSE, which predicting the corpus-mean text would game.

**VICReg** — variance and covariance terms retained as anti-collapse; the invariance MSE applies to
nuisance-group pairs only.

Strategy 3 is what makes the descriptor load-bearing. It replaces the standalone contrastive
descriptor-matching head considered earlier: same effect, one framework, no new loss family.

## `sensor_bias`

Computed **offline**, per sensor, over all recordings from that sensor. **Activity-invariant channel
physics only** — pooling over a sensor whose dataset has a skewed activity distribution would
otherwise make this a dataset fingerprint.

| field | computed as | catches |
|---|---|---|
| gravity magnitude | median ‖acc‖ over quiescent patches | recgym min-max ([0,1] → nonsense) |
| gravity presence | median \|DC\| | kuhar / uci_har gravity-removed (≈0.04) |
| noise floor | quiescent energy above the motion band (>~20 Hz is instrument, not person) | sensor grade |
| quantization step | min positive gap in sorted unique value diffs | effective bits |
| clipping | fraction at rails + rail value | range limits |
| rate fidelity | declared rate vs measured inter-sample statistics | FORTH-TRACE-style clocks |
| dropout structure | gap fraction + gap-length statistics | transmission character |
| gyro rest bias | resting gyro norm | calibration |

Quiescent-patch selection = low-variance percentile per stream. Every field is a **norm or aggregate**
— which is why rotation leaves the descriptor unchanged. That is by construction, not luck.

**OPEN (deferred by user, 2026-08-11):** whether accel and gyro need different field sets, and
pruning to the minimal set that serves the goal. Build the full set, measure, then prune.

**BUILT 2026-08-11** — `data/scripts/curate/sensor_bias.py`, artifact
`data/scripts/curate/sensor_bias.json`: **163 sensors across 34 datasets**. Values verified physical:
accel gravity magnitude 1.000–1.002 g, gyro rest bias 0.012–0.026, gravity fields correctly NaN for
gyro, `noise_floor` NaN on 16% of sensors (streams at ≤40 Hz have Nyquist below the 20 Hz motion band,
so out-of-band content is genuinely unmeasurable — reported unsupported, never filled).

**Guard status: INCONCLUSIVE, and it cannot be made conclusive on this corpus.**
Nearest-neighbour dataset purity is 0.534 against a 0.029 chance. Restricting neighbours to
*different placements* returns **the identical number** (0.5337, all 163 scored) — the exclusion
removes nothing, because every stream carries a unique placement and every dataset a single device
model. **Dataset identity and acquisition hardware are confounded by construction**, so this statistic
cannot separate "descriptor captures channel physics" (intended) from "descriptor is a dataset ID"
(leakage). It is retained as a monitor — a sharp increase across corpus revisions still means
something — but reporting it as a pass would be false assurance.

The decisive guards are downstream, where the confound does not apply:
1. **Encoder probe** — linear probe from the trunk's `feature` output to dataset ID must sit at
   chance. `sensor_bias` enters the trunk (user's call, 2026-08-11), so this tests whether the
   representation absorbed provenance. **Not yet built.**
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

# Resolvability — MEASURED 2026-08-12. The premise holds.

`training/evidence/resolvability.py`, artifact `training/evidence/outputs/resolvability.json`,
80 streams encoded with `phase_a_headline/best.pt`.

**How the score is computed — and what it is NOT.** Per stream: encode windows with the frozen
encoder, hold out half the subjects, and for each label run a one-vs-rest kNN over cosine similarity
in *embedding* space; score balanced accuracy rescaled `2·(ba − 0.5)`, clipped at 0. **The label
strings are used only as groupers — their semantic content never enters the score.** No SBERT, no
text kernel. That matters: if resolvability were derived from linguistic similarity it would be
re-measuring the text interface we already showed is inert (+0.0086), and the result would be
circular. It isn't.

## The paired contrast — simultaneous streams, same events, placement the only variable

| dataset | labels | mean gap | max gap | stream-profile correlation | inverting pairs |
|---|---:|---:|---:|---:|---:|
| realdisp | 31 | 0.508 | 0.882 | **+0.419** | 35/36 (97%) |
| mmfit | 10 | 0.493 | 0.868 | **+0.379** | 2/6 (33%) |
| xrf_v2 | 28 | 0.255 | 0.803 | +0.703 | 14/15 (93%) |
| opportunity | 4 | 0.236 | 0.443 | +0.872 | 1/10 (10%) |
| sp_sw_har | 5 | 0.160 | 0.377 | +0.885 | 0/1 (0%) |
| **overall** | | **0.330** | | **+0.652** | |

Representative rows, all on identical physical events:

| concept | best | worst | gap |
|---|---|---|---:|
| mmfit `bicep_curls` | left_wrist 1.00 | left_ear 0.13 | 0.87 |
| realdisp `jump_rope` | right_lower_arm 0.88 | back 0.00 | 0.88 |
| xrf_v2 `answering_phone` | right_pocket 0.80 | airpods_ear 0.00 | 0.80 |
| mmfit `squats` | right_pocket 0.99 | right_wrist 0.35 | 0.64 |

**Both halves of the claim are confirmed.** The gap establishes that placement determines whether a
concept is witnessable at all. The *correlation* establishes the sharper half: if placements had a
fixed quality ordering, stream profiles would correlate near +1 and a scalar per placement would
suffice — no third argument needed. On the two datasets with rich concept vocabularies (realdisp 31
labels, mmfit 10) correlation falls to ~0.4 and **93–97% of stream pairs invert**: the pocket wins
`squats` and loses `tricep_extensions`, the wrist the reverse. Which placement wins depends on the
concept, which is exactly what makes the gate's `(config, config, concept)` signature necessary.

The two high-correlation datasets (sp_sw_har 0.885, opportunity 0.872) carry only 4–5 coarse
locomotion labels, where every sensor sees roughly the same whole-body posture. That is the expected
boundary of the effect, not a counterexample.

## Honest limits

- **Encoder-dependent.** Scores come from one checkpoint; a better encoder shifts them. What is
  robust is the *contrast*, since every stream passes through the same encoder on the same events.
- **Same-protocol only.** Each label is scored against its own stream's other labels, so the number
  means "can this config tell this concept from its neighbours here", not an absolute.
- **NO GENERALISATION TO UNSEEN PAIRS.** `gate_tensor` is a lookup; unmeasured (stream, label) pairs
  return `default=0.5`, neither veto nor licence. At deployment with a novel label there is no
  estimate. **This is precisely where language would have to do the work** — predicting resolvability
  for an unseen (placement, concept) pair from their descriptions — and it is unbuilt. It is also the
  one remaining place the language interface could earn its keep after failing everywhere else.

---

# Bank build

Rows are **per patch per sensor**: `[feature, text_descriptor, sensor_bias]` + label + provenance
(`subj`, `cfg`, `event`, `time`, `resolution`). ~7M rows (roughly double, from the per-sensor split).

Provenance and construction metadata are **unchanged** — folds, execution leakage units, and episode
sampling do not care how rows are keyed.

Record **partner-sensor presence** in the descriptor. A row encoded alongside a gyro is not the same
as one encoded alone (the trunk attended across sensors), and a query from an accel-only device has
no gyro context. The gate must see this rather than absorb it. Sensor-dropout during pretraining is
the other half of the mitigation.

Payoff: capture24 (accel-only, the largest corpus source) now matches accel queries exactly instead
of smearing a channel-set mismatch into the embedding.

---

# Phase B — Stage 1, fully closed-form

Not a fallback. At 65.4 vs 45.5 the closed form is the champion; this promotes it to the design of
record.

**Prediction:**

1. **Compatibility filter** — hard, index partition: modality accel↔accel / gyro↔gyro, gravity state,
   units. Makes the comparison meaningful at all.
2. **Rank** — `feature` cosine within the admissible pool, additive `sensor_bias` blend.
3. **Admissibility gate** — soft, placement- **and concept**-dependent; initially rule-based.
   **Never hard-filters placement**: wrist evidence can bear on ambulation and not on arm gestures,
   and that distinction is the contribution. Hard filter = "is this comparison interpretable"; soft
   gate = "is this evidence relevant to this concept."
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

**Stage 2** — train the gate and blend scalars only. A handful of parameters over a working closed
form.

**Stage 3** — tokenizer finetune: frozen-index retrieve → re-forward selected rows with grad →
end-to-end. Needs scheduled bank re-embeds (or a drift threshold on the existing fingerprint probe),
SSL loss retained as a regulariser, learnable projections over `sensor_bias` with frozen raw
statistics, and eligibility gated on beating Stage 1.

**Parked, not deleted:** relational decoder, counterfactual objective, retriever training. If a dozen
learned parameters cannot improve a working closed form, that says what a million would do — for a
day of compute instead of a week.

---

## Ledger — build status as of 2026-08-11

**Pre-existing:** filterbank + observability masks, signed DC, phase feature, dual-branch trunk with
physical-time RoPE, gated text fusion, JEPA + VICReg, episodic construction, execution leakage units,
canaries (support-removed / label-shuffled), prototype + ridge comparators, `live_encoder`, bank
fingerprint guard.

**BUILT in this plan** (commits `9b7d75d`, `1ebaf0a`, `dad8ee9`; 475 tests pass):

| piece | where |
|---|---|
| parity ablation + `--parity` arm | `training/tokenizer/eval_transfer.py` — **run, result above** |
| augmentation NUISANCE/CONFIG split + `split_by_group()` | `data/scripts/augmentations.py` |
| `sensor_text_dropout` disabled | `data/scripts/augmentations.py` |
| `sensor_bias` computation + guard + artifact (163 sensors) | `data/scripts/curate/sensor_bias.py` |
| `sensor_bias` loader for the data path | `training/tokenizer/pretrain_data.py` |
| `SensorFold` + axis validity mask | `model/tokenizer/sensor_tokens.py` |
| `ConditioningProjection` (learnable MLP over frozen artifact) | `model/tokenizer/sensor_tokens.py` |
| `DescriptorHead` + retrieval-scored loss | `model/tokenizer/sensor_tokens.py` |
| sensor-granularity encoder path (`token_granularity='sensor'`) | `model/tokenizer/encoder.py` |
| sensor-mask + descriptor-mask JEPA planner | `training/tokenizer/losses_repr.py` |
| Phase-B Stage 1 closed-form predictor | `training/evidence/admissible_retrieval.py` |
| retrieval-provenance guard | `training/evidence/admissible_retrieval.py` |
| 30 gate tests | `tests/test_sensor_granularity.py`, `tests/test_admissible_retrieval.py` |

**TRAINER INTEGRATION — BUILT AND SMOKE-TESTED 2026-08-12** (commit below):

- shared config draw across views (`_shared_draw`, generalised `shared_config_seed`)
- `sensor_bias` + `sensor_placement` emitted per batch, ragged-padded, carried to view B
- `--token-granularity sensor` CLI flag, threaded into cfg and the encoder builder
- `make_sensor_mask_plan` wired, with `descriptor_mask` and `sensor_bias` into both forwards and the
  teacher; `jepa_mask` and the per-resolution diagnostic switched to the sensor presence mask
- descriptor-retrieval loss added to the objective, scored ONLY on sensors whose descriptor was
  actually hidden (an unmasked descriptor was fed to the encoder, so "reconstructing" it is a copy)
- telemetry: `descriptor/loss`, `descriptor/top1`

**300-step smoke, both arms, real corpus subset** (5 datasets, batch 16):

| arm | jepa | total | descriptor top1 | AMP skips | val_knn_ba |
|---|---|---|---|---|---|
| sensor | 0.997 → 0.047 | 25.0 → 13.5 | 0.0 → 0.67–1.0 | 2 | 0.2395 |
| channel | 1.031 → 0.041 | 24.2 → 10.6 | n/a | 0 | 0.2789 |

No persistent NaN (the step-1 value is AMP scaler startup). The sensor arm's val is *lower* here, but
300 steps on a 5-dataset subset is far below the 3k-step screening noise floor (sd 0.0065) — it
carries no signal and must not be read as an arm comparison.

**STILL NOT BUILT:**

1. **`build_memory.py` per-sensor rows** — emit `[feature, text_descriptor, sensor_bias]` per patch
   per sensor (~7M rows) instead of per patch.
2. **`resolvability` estimation** — the (source config, target config, candidate) table the
   admissibility gate consumes. Stage 1 accepts it as an argument; nothing produces it yet. The
   paired multi-stream recordings (sp_sw_har, xrf_v2, opportunity, realdisp, mmfit) are the intended
   source. **This is the critical path: admissibility is the contribution, and a null result here
   changes what Phase A should be optimising for.**
3. **Encoder dataset-ID probe** — the decisive fingerprint guard, since `sensor_bias` enters the trunk.

**Unmeasured and load-bearing:** whether cross-config enrollment works at all; whether the bias term
is physics or fingerprint; whether the redesign moves the parity number off +0.0086.
