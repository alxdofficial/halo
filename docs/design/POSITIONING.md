# Positioning — what the evidence engine is actually for

> **Status: decision notes, 2026-07-21.** Written to answer one question: *is a HAR model that
> generalizes zero-shot to unseen datasets, configs and labels worth pursuing, and would the end
> result be useful to anyone?*
>
> This is a **small deviation** from the original purpose in `MOTIVATION.md` — it keeps the corpus,
> the encoder, the language interface and the evidence engine. Two **larger** deviations were
> explored and parked on branch `pose-pretext-exploration`:
> `POSE_PRETEXT_LITERATURE.md` (IMU→pose pretext — killed) and `ENROLLMENT_BY_DEMONSTRATION.md`
> (repetition-mined enrollment — alive but a bigger pivot).

---

## 1. Is zero-shot HAR worth pursuing?

Split the question, because the two halves have opposite answers.

**As a product claim: no.**

> ⚠️ **Encoder provenance.** The 84.3 / 39.6 pair is both-`pretrain_native` — that is what makes it
> a valid same-encoder comparison. The evidence-engine numbers (42.7 ConSE / 45.1 untrained / 46.1
> trained) are `pretrain_fixed_mr`. **Do not pair 84.3 with 42.7** — different encoders, and we have
> never measured a supervised probe ceiling on `pretrain_fixed_mr`.

| | macro-F1 |
|---|---|
| our supervised linear probe (`pretrain_native`, frozen) | **84.3** |
| our zero-shot ConSE (`pretrain_native`, same encoder) | **39.6** |
| UniMTS zero-shot → its own full-shot, same paper, same 18 datasets | 34.3 → 87.5 (**53.2-point gap**) |
| IMU2CLIP zero-shot → linear probe, same encoder, adjacent table rows | 18.46 → 62.52 (**44.1-point gap**) |
| Google SensorLM (59.7M hours, 103k people), 20-class zero-shot | **0.29 F1** |

If a few hundred labels double the score, "no labels needed" is a **45-point discount, not a
feature**. Our gap is *smaller* than the field leader's, which means it is normal — but normal is
not the same as acceptable. And SensorLM shows scale is not about to close it.

**As a research problem: wide open.** Honest third-party cross-dataset zero-shot sits at **~29.5**
(AnyMo, 14 unseen datasets); reproductions of UniMTS land at 32.1 and 26.4 against its self-reported
34.3. Our 39.6–42.7 is at or above every honestly-measured published result — *with the caveat that
protocols differ and this is not a like-for-like comparison.*

**Conclusion:** "zero-shot classification accuracy" is the wrong headline. The problem is real and
unsolved; the framing is what fails.

## 2. The reframe: capability at test time, not capacity at train time

We cannot win on data. harnet has ~700k person-days against our ~290–547 hours — roughly a
**58,000×** ratio.

But a large pretrained model has its knowledge **baked in at training time**. It cannot acquire a new
placement, a new device or a new label afterwards. A memory bank can, in O(1), with no gradient step.

That moves the axis of comparison from *"who has the better frozen representation"* to *"who can
acquire new configs and labels at test time"* — a **capability** difference rather than a three-point
F1 delta. Capability differences are far more defensible, and this one is structural.

Supporting evidence that this axis is the right one: **UniMTS's own ablation** attributes its gain to
acquisition-config handling, not label semantics — removing rotation-invariant augmentation costs
**−16.7** macro-F1 and removing the graph/placement encoder **−16.7**, while removing text
augmentation costs only **−5.5**. Config handling is worth ~3× text handling, measured by the leading
zero-shot paper on itself. The barriers the literature names are **sensor heterogeneity** and **poor
activity text**, never label semantics.

## 3. THE APPLICATION ANSWER

**If the memory bank is allowed labeled target data, the value proposition is not accuracy.** Say
this plainly and do not pretend otherwise: where labels exist and accuracy is what matters,
fine-tuning probably wins. The value is in properties fine-tuning does not have.

### 3.1 What the approach is actually good for

* **Operational scale.** A vendor shipping across 50 device × placement combinations maintains **one
  encoder and 50 small banks** instead of 50 models. No per-config retraining, no per-config
  revalidation, no 50-way model registry. The saving is engineering and compliance cost, not F1.
* **Continual addition without regression risk.** A new activity next month is an *append*. No
  retraining cycle, no catastrophic forgetting, and — importantly for anything certified — **no need
  to re-validate the model you already signed off**, because the weights did not move.
* **On-device and private.** Appending exemplars to a bank is feasible on a phone; fine-tuning
  largely is not. The user's data never has to leave the device.
* **Per-user concepts.** A prescribed rehab exercise is defined *per patient*. A shared model
  structurally cannot encode it, no matter how much data it was trained on.
* **Auditability.** Retrieval answers *why*: "classified as X because it matched these five
  exemplars — here they are." A parametric model returns a logit. For clinical or regulated
  deployment this is a genuine, inspectable difference, and it is probably the most underrated
  advantage we have.

### 3.2 What it is NOT good for

* Beating a supervised probe on accuracy when labels are plentiful. It very likely won't.
* Any setting where a single fixed config and a single fixed label set are known in advance — that
  is a solved supervised problem and we should not pretend otherwise.

### 3.3 Why this suits our venue

These are **deployment properties, not accuracy properties** — and MobiCom / UbiComp / IMWUT reward
exactly that. "One model, many deployments, no retraining, inspectable decisions" is much more their
taste than a three-point F1 win, and **it is a story that survives never beating harnet.**

### 3.4 The honest failure condition

If a linear probe beats retrieval at *every* label budget including k=1, then only the operational
arguments remain — and operational arguments alone rarely carry a top venue without a real
deployment behind them. That is the outcome to test for first (§5).

## 4. Reporting: do not switch the metric, add an axis

The tempting move — stop reporting zero-shot, report peak performance after the bank ingests target
data — has two problems.

**It stops being zero-shot.** Once the bank holds labeled target data it is few-shot / transductive
transfer, and the competitor becomes the linear probe, because *fitting a probe on frozen features
with k labels is also seconds of work*. "No gradient step" is a weak differentiator on its own.

**⚠️ It is the exact practice we criticised.** ZARA reaches 81.4 macro-F1 — matching a supervised
probe — **by requiring a labeled in-domain retrieval database at inference**. Our own survey listed
that as one of three ways published zero-shot numbers get inflated (the others: LLaSA training on 4
of its 5 eval datasets; ZeroHAR's within-dataset unseen-class split sold as cross-dataset). If we
adopt the same mechanism and keep the zero-shot label, we commit the same sin.

Also: switching metrics *because the current one looks bad* is result-driven metric selection, and a
reviewer will smell it.

**Instead: report a curve over label budget.** k = 0, 1, 5, 10, 50, full — with the linear probe on
the same axes. k=0 is zero-shot, k=full is supervised. Nothing can be cherry-picked, and the claim
becomes whatever the curve supports.

## 5. The decisive experiment — the k-curve

**This is the experiment for the whole project, not just this sub-idea.** It needs no new data:
existing encoder, existing bank, existing baselines, existing subject splits, existing protocol
stamping.

For each held-out eval cell, and for k ∈ {0, 1, 5, 10, 50, full} labeled target windows per class:

| arm | mechanism |
|---|---|
| **retrieval** | append the k exemplars to the memory bank; no gradient step |
| **probe** | fit the shared 2-layer probe on the same k exemplars, frozen features |
| **harnet probe** | same, on frozen harnet features — the strong-representation control |

Report macro-F1 vs k, with subject-stratified CIs, per axis of shift (§6).

**Hypothesis (pre-registered):** retrieval wins at very low k — nearest-neighbour beats linear
probes when labels are scarce — and probes overtake somewhere in the 10–50 range. If so our
defensible territory is **k ≤ ~10**, which is exactly the enrollment setting, and the whole
positioning becomes internally consistent.

**Kill criterion:** if the probe dominates at every k **including k=1**, retrieval has no regime of
its own and this direction is dead. State that outcome plainly if it happens.

## 6. The controlled-shift ("radar") protocol

A single blended cross-dataset number hides which heterogeneity is actually costing us. Report
**retention** — performance under shift ÷ performance in the matched condition — one axis at a time.

| axis | how we construct a CONTROLLED shift | control quality |
|---|---|---|
| **placement** | wisdm / sp_sw_har / xrf_v2 simultaneous streams — same subject, same instant, only placement differs | **excellent** |
| sampling rate | resample the same windows | exact |
| channel set | drop gyro, keep accelerometer | exact |
| gravity convention | already tracked per stream | exact |
| orientation | apply rotations (existing augmentation machinery) | exact |
| subject | LOSO from the split manifest | exact |
| labels | hold out label groups | exact |
| device/hardware | different datasets — confounded with everything else | poor; must be flagged |

Six or seven cleanly controlled axes from data already on disk. The placement axis is the standout:
simultaneous multi-placement recordings of the same person at the same moment isolate placement from
subject, activity and session entirely.

**Caveat on the radar visual:** polygon area is *not* a principled aggregate — it depends on the
arbitrary axis ordering and over-weights models good at two adjacent axes. Use the radar as a
picture; report per-axis retention as the result; for a scalar use **mean** retention, or **min** for
worst-case robustness.

**The asymmetry this will expose, and we should face it:** on the **label** axis fine-tuning is not
expensive, it is *unavailable* — a class you have labels for is not unseen. On the **config** axes
fine-tuning is available and probably cheap. That is awkward, because `MOTIVATION.md` calls the label
axis table stakes and stakes the contribution on config. The counter — that config adaptation costs a
labeled collection **per deployment, recurring forever** across devices, placements and firmware
revisions — is real but currently *asserted*. The k-curve per axis is what would measure it.

## 7. The undersizing / substitution study (secondary)

With a large encoder you cannot tell whether performance comes from the representation or the
retrieval. **Deliberately shrinking the encoder isolates the non-parametric contribution**, and turns
our data disadvantage into the experimental design.

Study: encoder size × memory-bank size → ZS-XD, with harnet as the fixed large-encoder-no-memory
reference. Question: does small-encoder-plus-memory reach large-encoder-alone, and at what memory
cost? Precedent exists in language modelling, where retrieval substitutes for parameters.

**⚠️ What could kill it, from our own data:** retrieval purity is **flat at 0.68 across all k**,
which says retrieval quality is capped by the representation. If the encoder is the ceiling,
undersizing lowers it and retrieval cannot compensate. Which is exactly why it is worth measuring —
a clean negative ("parameters-for-memory substitution does not hold for wearables") is publishable
and unpublished.

## 8. What this does NOT resolve

* Whether the k ≤ 10 regime is large enough to matter to anyone in practice.
* Whether operational advantages carry a top venue without a deployment.
* Prior art on few-shot / query-by-example HAR, which has **not** been checked — and the pose episode
  showed what happens when we skip that step.

---

## 10. Novelty check (literature sweep, 2026-07-22) — both contributions are in crowded territory

Ran directly against Consensus + Exa, no workflow. The honest verdict: **both candidate
contributions are less novel than hoped, and the pose-idea lesson (assert nothing, verify) applies.**

**Contribution A — language-conditioned sensor/config encoding — largely taken.** GOAT (IMWUT 2024,
device-position text encoding), oneHAR/uniHAR (IMWUT 2025, arbitrary sensor-position configs),
ActivityNarrated (2026, open-vocab + heterogeneous placement + retrieval eval), LanHAR, AnyMo,
MobiDiary all occupy this space; MobiDiary even evaluates on our own `xrf_v2`. Remaining daylight is
**narrow**: the per-sensor free-text factorization (`TEXT_CONDITIONING.md`) feeding a non-parametric
evidence engine. Needs close reading of GOAT/oneHAR/ActivityNarrated to confirm, not assert.

**Contribution B — teach-by-example / retrieval-append — mechanism is taken; the HAR framing may have
a gap.** The append-to-memory, prototype-distance, no-retraining mechanism is heavily established in
vision/VLM/detection (RAC "Online Learning via Memory", T3AR, and many memory-bank TTA methods), and
OFTTA (IMWUT 2023) already brings optimization-free prototype TTA to HAR — but **unsupervised**
(pseudo-labels, cross-person drift), not "append *labeled* exemplars of new configs/labels." The
specific open question — *does appending labeled exemplars match fine-tuning's data-efficiency for
wearable HAR, with calibrated abstention?* — was **not** found answered. One search (few-shot
query-by-example HAR) hit a rate limit and is still unchecked.

**Consequence.** Neither contribution is a clear open lane on its face. The defensible position, if
one survives, is the **combination** — per-sensor language conditioning + non-parametric test-time
acquisition + calibrated abstention + the k-curve showing a low-k regime we own — not any single
component. Before building, close two threads: (1) retry the few-shot/query-by-example HAR search;
(2) close-read GOAT, oneHAR, ActivityNarrated.
