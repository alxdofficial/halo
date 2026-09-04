# Recognize by comparison — agreed design and paper shape for the IMWUT submission

**Status: design of record for the `imwut/compare` line. Agreed 2026-09-03, revised the same day (two arms). THINKING STAGE —
nothing in this document is implemented, trained, or evaluated yet.**

This supersedes the clinical / motion-monitoring pivot as the paper target. The
classification-era code (Phase-A tokenizer, evidence/compact engine, baseline adapters, the
`adaptation_v1` evaluation manifest) stays the substrate; the application tasks remain on `main`
untouched.

---

## 0. Why this direction

Supervisor decision, 2026-09-03: clinical venues want a concrete solution to one condition with
data gathered for it; a generic technique that happens to apply to several conditions will not
land there. HALO started as an ML paper and should stay one. The right venue for a well-evaluated
incremental improvement on the HALO line is IMWUT, where the recent comparable work lives
(CrossHAR, GOAT, oneHAR, IMUZero, LanHAR, MASTER, Customizable-FM, Wonderwall).

What we learned across v1 and v2 (see `docs/results/` and the memory ledger) points at one
increment:

- The encoder was never the bottleneck: frozen HALO features linearly separate activities at
  84 macro-F1 while zero-shot through a text bridge sits near 40.
- Comparison against labelled exemplars is the right primitive: the *untrained* retrieve-and-vote
  mechanism already edged harnet (47.5 vs 47.3), and prototype/ridge over frozen features beat
  every trained head we built.
- Plain cosine retrieval over a heterogeneous bank fails for a specific reason: it ranks by
  acquisition configuration (x7 lift) rather than by activity. Learned retrieval was worth
  exactly nothing. The stage should go.
- The one learned component that helped (the attention mixer, +0.12) helped only because the
  language channel was in the loop: a scrambled-vocabulary control inverted the gain.
- Training with the ground-truth label absent from the support set pushed zero-shot below chance
  under closed-vocabulary cross-entropy, while helping k >= 1. Zero-shot is additionally capped by
  the weak relation between signal similarity and label-name similarity (r ~ 0.11).
- End-to-end training from random initialisation collapsed the encoder's effective rank within
  300 steps. Warm-starting from self-supervised pretraining is not optional.

The literature has moved the same way: sensor heterogeneity (user, device, placement) is named as
the dominant barrier in every 2025-26 survey, ZARA (ACL 2026) gets training-free transfer by
reasoning over *retrieved reference statistics* with an LLM, and HARBench (PerCom 2026) now scores
position-robustness and few-shot as separate axes. We take ZARA's structure — features plus
labelled reference recordings — and replace the LLM with a small learned comparator.

---

## 1. Thesis

**Recognise by comparison, not classification.** Given a query window and a handful of labelled
recordings that were acquired under a compatible sensor configuration, a small comparator decides
which examples the query resembles and reads the label off them in language space. The model is
not asked to learn a universal representation of all IMU data; it is asked to embed motion into
features good enough that attention over a compatible support set can make the call.

The contribution we market is the **training curriculum**: how support sets are sampled during
pretraining-plus-fine-tuning so that the comparator learns to compare rather than to memorise a
vocabulary. Nothing in the architecture is claimed as novel.

---

## 2. Model (kept deliberately simple)

| Component | Decision | Rationale |
|---|---|---|
| Front end | **Fixed physical filterbank, single resolution, not learnable** | The learnable arm stayed pinned at its init; simplicity wins. Multiresolution's +0.034 is a documented option we deliberately do not take. |
| Feature extraction | **Existing Phase-A tokenizer + temporal trunk, unchanged** | No evidence that complicating it buys anything. |
| Conditioning | **Rate and patch-duration pathways kept. Acquisition-configuration text is OFF in the core design.** Compatibility is handled at support construction instead (Section 3). | The encoder is trained on every configuration anyway; once support is compatible by construction there is nothing left for the text to tell it. The text pathway is kept in the code for the Section 6 experiment. |
| Comparator | **Attention over the query and every support example** (no retrieval stage, no top-k) | Removes the config-ranking defect and the non-differentiable selection; K is small enough to attend over fully. |
| Feature centering | **ON by default**: each episode's mean feature is subtracted from the query and support rows before similarity and attention | Acquisition configuration is close to a common mode within an episode, since the support all shares the query's key; the previous design's retrieval ranked by configuration at a 7.0x lift. Removing the mean leaves only how rows differ. `--no-center-features` is the ablation. |
| Readout | score(candidate c) = sum over support examples e of  attn(query, e) x cos(text(label_e), text(c)) | The vote we already run. Duplicate or synonymous labels need no homogenisation — they simply contribute through their text similarity. |
| Label / text tower | Frozen sentence encoder (MiniLM), text ensembling kept | Every learned text adapter we tried was net-negative. |
| Size | Compact engine budget, ~1M trainable parameters | Efficiency is part of the story. |

Support examples are **encoded per query, with gradients**, at train time. Nothing is cached.

---

## 3. Support-set contract (the method)

A support set is the list of labelled recordings the comparator sees alongside the query. The
same sampler runs at train and test; only the pool differs.

1. **Compatibility is a filter, not a learned quantity.** Support examples must share the query's
   acquisition configuration family: device family, placement, channel set, gravity state.
   Sampling rate is *not* part of the key (the filterbank is rate-invariant by construction). We
   do not offer a smartwatch example to a pocket-phone query. This is a plain deployment
   consideration; we claim no novelty for it. The compatibility key already exists in the
   application code (`SensorCompatibilityKey`) and is reused as-is.
2. **Never the query, never the query's subject.** Support is subject-disjoint from the query.
3. **Random label subset.** Draw a subset of labels present in the pool, then K recordings across
   that subset. Labels are used verbatim; no canonicalisation, no deduplication.
4. **Ground truth present with fixed probability.** With probability p the query's label is
   among the support labels (few-shot episode); otherwise it is excluded (zero-shot episode). The
   two regimes are trained jointly, not in separate arms.
5. **Zero-shot episodes use a soft target**, the text-similarity distribution over candidates,
   not a hard one-hot. This is the guard against the k = 0 collapse we measured.
6. **Support may span datasets** as long as every example passes the compatibility filter. The
   encoder is trained on all configurations; the filter only constrains what is compared.

Constants (there is no development split; set by judgement before the first run, and free to be
varied *in training* as an experiment — varying them costs nothing and is informative):

| constant | value | note |
|---|---|---|
| K (support size, train) | 32 | drawn per episode; at evaluation K is the swept k of the k-curve |
| p (GT present, train) | 0.5 | joint ZS/FS; a sweep over p in {0.25, 0.5, 0.75} is a cheap experiment |
| label-subset size | uniform in [2, min(8, labels available)] | |
| fine-tune schedule | 35k-50k steps | the plateau we saw at 6k was a schedule artifact |
| checkpoint | final step of the fixed budget | no selection on held-out data |
| seeds | >= 3 | validation-draw variance dominates single runs |

At inference the support size is not a constant at all: the evaluation sweeps k, which is the
k-curve.

---

## 4. Training

**One stage: end to end from random initialisation** (revised 2026-09-04). Encoder and comparator
are trained together on episodes drawn by the Section 3 sampler. Loss = cross-entropy over
candidates for few-shot episodes, soft-target KL for zero-shot episodes.

This corrects an earlier draft that made a Phase-A warm start mandatory on the grounds that
from-scratch training collapsed the encoder's effective rank. The checkpoints contradict it: every
compact-engine checkpoint on disk, including `long_4h_20260821` — the one leading 33 of 40
enrolment columns at selection 0.5424 — records `phase_a_checkpoint=None`. The rank collapse was
measured on a 6,000-step schedule where the step-1,750 "peak" was head saturation, not convergence;
at 35,000 steps the same from-scratch recipe produced the best result the project has.

Rank collapse therefore stays as **telemetry worth watching**, not a demonstrated failure mode, and
`encoder/effective_rank` is logged every validation.

The Phase-A warm start remains available as a second arm (`--phase-a`). Which wins at a 35k
schedule has never been tested head to head; that is an experiment, not a settled question.

---

## 5. Evaluation

- **Protocol**: the frozen `adaptation_v2` manifest — **10** held-out datasets, subject-disjoint,
  70 cells (48 usable), fingerprint `0f98ae25…`. `adaptation_v1` (7 datasets) stays intact so the
  pre-pivot numbers remain comparable; the two carry different protocol names so the assembler
  cannot mix their rows.

  **motionsense, realworld and shoaib were restored 2026-09-04.** They were never unsuitable — they
  were the Phase-B *development* split, withheld so repeated development would not consume the
  datasets meant to support the final claim. That split no longer exists. Restoring them doubles
  the ordinary-activity enrolment cells (5 -> 10) and lifts ordinary zero-shot cells (4 -> 7)
  without losing a single clinical cell, and it decorrelates two things that were confounded: with
  only the seven, 8 of 11 streams were wrist and the hardest labels sat on exactly the streams with
  the narrowest support pools. All three are phone streams carrying ordinary locomotion.

  **To disclose in the paper:** design decisions in the superseded Phase-B line were made while
  looking at these three. No model was ever trained on them, so this is researcher exposure rather
  than data leakage — but the reviewer should hear it from us. `motionsense` additionally has no
  config-compatible training stream, so like `upper_limb_use` its k=0 row is unsupported while its
  k>=1 rows are unaffected.

  `hapt` is permanently excluded and now raises if requested: it is the same 30 subjects as
  `uci_har`, which is in the training corpus.
- **Headline**: enrollment k-curve, k in {1, 2, 4, 8, 16}, macro-F1 with subject-bootstrap CIs,
  mean over >= 3 seeds. Zero-shot (k = 0) is reported as a **disclosed secondary** row using the
  same mechanism with an empty ground-truth slot; we state its cap rather than chase it.
- **Support at test time** comes from the held-out dataset's own enrollment pool, which is
  config-compatible by construction and subject-disjoint by the manifest. Two enrollment modes are
  first-class: **cross-subject** (support from other people) and **same-subject** (support from the
  user's own recordings) — the latter is the deployment story and the ubicomp headline figure.
- **Mandatory control rows**: the *untrained floor* (same mechanism at initialisation) and the
  *step-0 control* (paired against each trained run), following the methodology rule that every
  learned component is guilty until a control clears it.
- **Baselines = released checkpoints only.** Training regimen and data are part of each method;
  we do not match them. Keep harnet, UniMTS, ImageBind, NormWear. Drop CrossHAR and LiMU-BERT
  (we pretrained those ourselves). Audit for released weights before writing: Wonderwall
  (IMWUT'26), IMUZero (IMWUT'25), LanHAR (IMWUT'24), GOAT (IMWUT'24), MOMENT. Every baseline gets
  its own native few-shot rule (1-NN / prototype / linear head, whichever is best for it).
- **Closed-set models do open-set through ConSE** (Norouzi et al. 2014), exactly as before: a
  temperature-calibrated convex combination of label-text embeddings fit on training data only.
  It is the defensible, practical bridge and it is what lets every released checkpoint enter the
  same k-curve sweep.

---

## 6. Two arms, then ablations

**Arm A — explicit compatibility control (the core design).** No acquisition text reaches the
encoder. Support is filtered to the query's configuration family at construction time. This is the
headline configuration and every claim in Section 7 is made about it.

**Arm B — learned compatibility (an experiment, not a core design).** Acquisition text is ON and
support construction allows *anything* — compatible and incompatible examples mixed. The question
is whether the comparator learns on its own to gauge which examples are compatible, which to
trust and which to discount. Two sub-questions fall out of it:

- B1. With mixed support, does Arm B recover Arm A's accuracy (learned filter ~ explicit filter)?
- B2. When *no* perfectly compatible example exists — the support holds only near misses, e.g. a
  different placement on the same device family — can the model still draw something useful from
  them, and how does accuracy degrade with compatibility distance?

Arm B is reported as a result about the model, not as a claim the paper rests on.

**Ablations on Arm A** (each answers one reviewer question):

| Ablation | Question it answers |
|---|---|
| **Episode mean-centering OFF** (`--no-center-features`; centering is the DEFAULT as of 2026-09-04) | Does removing what the support rows have in common force the model to discriminate? Configuration is close to a common mode within an episode — the support all shares the query's key — and the previous design's retrieval ranked by configuration at a 7.0x lift, so this targets the measured defect directly. **Changes the step-0 function: compare raw scores at matched seeds, never paired gain.** |
| From-scratch vs Phase-A warm start | Does one-stage training beat two, at a 35k schedule? Never tested head to head. |
| **Fixed filterbank vs continuous kernel** (`--frontend`) | The encoder's two real front-end modes; see 6.1 — the existing head-to-head is inside the noise and must be re-run matched. |
| Fixed single-res filterbank vs multiresolution vs learnable | Is the simple front end leaving accuracy on the table? |
| Attention comparator vs cosine 1-NN vs prototype over the same encoder | Is the learned comparison worth having? (the untrained floor lives here) |
| Compatible support vs unfiltered support, text OFF | How much does the explicit filter buy on its own? |
| Text vote vs one-hot vote | Is the language channel load-bearing? (plus the scrambled-vocabulary control) |
| p in {0, 0.25, 0.5, 0.75, 1} | Does joint ZS/FS training cost few-shot accuracy? |
| Step-0 control | Did fine-tuning help at all, paired? |

### 6.1 On the two front-end modes

The encoder has always had two genuine front ends — the fixed physical filterbank and the
continuous kernel bank — and both are selectable in the comparison trainer (`--frontend`), verified
to run from scratch at the compact shape.

A head-to-head already exists on disk, and it settles nothing:

| run | frontend | step | selection |
|---|---|---:|---:|
| `e2e_pb04_fixed_filterbank_35k_20260824` | fixed | 10,000 | 0.3642 |
| `e2e_pb04_continuous_dense_35k_20260824` | continuous | 13,000 | 0.3724 |

Continuous leads by 0.0082, but the two runs stopped at different steps, neither recorded a seed,
and the measured screening noise on this setup is sd 0.0065 — the standing rule being that nothing
under about 0.012 is real. So the gap sits inside the noise and was measured off unmatched
schedules. Only the continuous arm has downstream adaptation results on disk, so even the
second-stage comparison is one-sided.

The honest statement is that **we do not know which front end is better**, and the paper must
either say so or run the comparison properly: matched steps, matched seeds, at least three of them,
compared by raw score rather than paired gain (the front end changes the step-0 function). The
design of record stays *fixed* because it is the simpler claim and the one the thesis rests on;
continuous is the challenger, not the default.

## 7. Paper shape

**Working title.** Recognize by Comparison: In-Context Activity Recognition Across Heterogeneous
IMU Setups.

**Claims.**
1. A ~1M-parameter comparator over a fixed physical filterbank, told nothing about the sensor
   configuration but given K compatible labelled recordings at test time, matches or beats
   68M-1.2B released foundation models on enrollment k >= 1 across seven held-out datasets, and
   leads where device and placement shift.
2. The gain comes from the training curriculum — episodic support sampling with joint zero- and
   few-shot regimes and soft zero-shot targets — not from architecture.
3. No label homogenisation is needed; verbatim labels vote through language similarity.
4. The whole system runs on device (parameter, latency and memory table).

**Sections.** Introduction (heterogeneity is a comparison problem) -> Related work (HAR foundation
models, language-aligned HAR, training-free/LLM reasoning, few-shot HAR) -> Method (front end,
comparator, support contract, curriculum) -> Evaluation protocol (manifest, leakage discipline,
baseline contract) -> Results (k-curve table, per-dataset, heterogeneity axes) -> Ablations ->
Cost -> Limitations (zero-shot cap stated plainly; compatibility filter assumes known placement)
-> Conclusion.

**Heterogeneity-axis experiment** (kept from the earlier thesis, now with a mechanism that can use
it): placement shift, rate shift, orientation perturbation, gravity present vs removed — ours vs
baselines under identical inputs.

---

## 8. Reuse map

| Reused as-is | New |
|---|---|
| Phase-A tokenizer + pretraining recipe | Episode sampler with no retrieval and the compatibility filter |
| Evidence mixer (becomes the comparator) | Joint ZS/FS loss with soft zero-shot targets |
| `adaptation_v1` manifest, eval harness, adapters | Baseline weight audit for the 2025-26 comparators |
| Results/methodology tooling (paired gain, step-0 predictor) | Cost table |

---

## 9. Open items before any build

All three are now **DECIDED** (2026-09-03) and recorded in `docs/design/IMWUT_HANDOFF.md` §1:

1. **k=0 mechanism** — comparator over a candidate-excluded, config-compatible corpus draw (not
   ConSE), so one mechanism spans the whole k-curve.
2. **Arm A never sees acquisition text** — text is OFF at pretraining *and* fine-tuning; Arm A and
   Arm B get separate Phase-A runs (~26 min each).
3. **Arm B2 compatibility distance is binary, not a ladder** — distance 1 means same device family
   plus *equivalent* placement (left↔right wrist, pocket variants). Wrist↔ankle and watch↔phone are
   out of scope; we do not claim a graded degradation curve.

Remaining: constants in Section 3 are set by judgement and may be varied in training; baseline
weight audit (Section 5). Target deadline is **Nov 1 2026** (user decision), with the Oct 1 k-curve
gate in `docs/design/IMWUT_BUILD_PLAN.md`. Venue read: `docs/research/IMWUT_VENUE_READ.md`.

**A blocking prerequisite surfaced during the sweep**: `StreamSpec.placement` values are free-text
prose ("the left wrist" vs "left wrist" vs "dominant wrist"), and the existing compatibility key
normalizes only whitespace and case — so the compatible pool fragments. Placement classes must be
built before anything else. See the handoff, §2 and W1.

No implementation, branch beyond this document, or training run is authorised by this document.
