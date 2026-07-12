# Baseline Fairness & Faithfulness Policy

**This is the authoritative policy for how HALO treats every baseline.** It exists so there is
*no* ad-hoc, per-model improvisation: every input decision and every modification we make to a
baseline is justified here, once, against a single contract. If a change to a baseline is not
sanctioned by Section 3, it does not ship.

Supporting detail lives in [`baseline_flexibility.md`](baseline_flexibility.md) (the capability
matrix) and [`BASELINE_IMPLEMENTATION_NOTES.md`](BASELINE_IMPLEMENTATION_NOTES.md) (per-model
published contracts + citations). Where those disagree with this file on a *decision*, this file
wins.

---

## 1. The heterogeneity stack

Real-world phone/watch HAR is heterogeneous along a few independent axes. We model the comparison
as a **stack of tiers**, bottom to top. Each tier is one axis; a model occupies a *level* at each
tier; HALO is `native` at every tier, and that stack *is* the contribution.

| Tier | Axis | Question it answers |
|---|---|---|
| **T0 — Base** | tokenization + feature extraction + prediction | What is the core model, independent of heterogeneity? |
| **T1 — Rate** | sampling-rate heterogeneity | How does it ingest data at an arbitrary Hz? |
| **T2 — Channels / placement / rotation** | channel count, order, modality, gravity-presence, body placement, orientation | How does it know *what each channel is* and *where/how the sensor sits*? |
| **T3 — Labels** | open-set / novel vocabulary | How does it predict a label it was never trained on? |

**Levels** (from `baseline_flexibility.md`): `native` (in-model, lossless) > `resample` /
`pad+mask` (handled but lossy / capacity-wasting) > `fixed` (pushed into preprocessing) > `none`.

### Where each model sits

| Model | T0 weights | T1 Rate | T2 Channels/placement | T3 Labels |
|---|---|---|---|---|
| **CrossHAR** | we pretrain (SSL) | fixed → **60 Hz** | fixed 6-ch, zero-pad+mask; placement-blind | closed → **ConSE** |
| **LiMU-BERT** | we pretrain (SSL) | fixed → **60 Hz** | fixed 6-ch, zero-pad+mask; placement-blind | closed → **ConSE** |
| **ssl / harnet5** | **frozen released** | fixed **30 Hz** | fixed **3-ch acc-only**; wrist-implied | closed → **ConSE** |
| **UniMTS** | **frozen released** | resample to its rate | **native placement** (SMPL joint), acc-only 3-ch | **native text** (CLIP, cosine) |
| **NormWear** | **frozen released** | fixed **65 Hz** | **native** variable real channels + mask; no placement | **native text** (TinyLlama, L1) |
| **HALO (ours)** | we train | **native (invariant)** | **native**: ch-independent + **language conditioning** (placement/sensor/gravity) | **native language alignment** |

The single most important split is **T0 weights**: two models we *pretrain on our corpus*
(CrossHAR, LiMU-BERT, self-supervised) — for those we fully control T1/T2 — and three **frozen
released checkpoints** (harnet, UniMTS, NormWear) whose rate/channel/gravity contract is baked into
the weights and **cannot** be changed without discarding the released model. "We train all
baselines on the same data" is true only for the first group *and* for every ConSE head; the frozen
backbones were pretrained on their own corpora (faithful, and disclosed).

---

## 2. Locked input decisions (one policy, no per-model drift)

One canonical corpus is built **once** and every model derives its input from it by a single
documented transform. Decided 2026-07-11.

- **Source data.** One corpus: accelerometer canonicalized to **g, gravity-present** (via
  `accel_units.to_g`; the only gravity-removed set is kuhar, physically unrecoverable), **acc+gyro**,
  **phone/watch placements only** (pocket / waist / thigh for phone; wrist / arm for watch — see the
  corpus-curation policy). No magnetometer / ECG / orientation / temperature / heart-rate.
- **T1 — rate.** Models we pretrain (CrossHAR, LiMU-BERT) unify at **60 Hz**. Frozen
  models are resampled **from the source to their required native rate** (harnet 30 Hz, NormWear
  65 Hz, UniMTS its 200-sample format) by **one** anti-aliased resampler — per-model *target*, one
  code path. HALO consumes native rate (no resample). *Faithfulness note:* CrossHAR/LiMU-BERT are
  transformers with a configurable `seq_len`; retraining at 60 Hz is done via that config knob
  (a 6 s window becomes 360 timesteps vs the paper's 120 @ 20 Hz), **not** by architectural surgery.
- **T2 — channels.** **Deployment unit = one device** (a phone *or* a watch): every sample is one
  device's channels, so a dataset that recorded phone **and** watch contributes **two** single-device
  samples (double the data, not a wider tensor) — placement is metadata, not channel width, so the
  corpus is **≤ 6 channels** (acc+gyro). Two tensor regimes are built (see §2A): **harmonised**
  (canonical channel order, zero-pad+mask) and **non-harmonised** (native as-collected order /
  variable count). 6-ch models take all 6 (acc-only devices zero-pad the gyro slots, masked);
  acc-only models (harnet, UniMTS) slice the 3 accel channels; HALO takes the variable real set.
  **Zero-pad + mask only — never random/fabricated fill.** Gravity is uniform in the source; any
  gravity removal is the model's *own* internal step (CrossHAR InstanceNorm, NormWear z-score).
- **T3 — labels.** Text-aligned models (UniMTS, NormWear, HALO) use their **native text tower**.
  Closed-vocab models (CrossHAR, LiMU-BERT, harnet) use the **ConSE bridge** (Norouzi 2014):
  softmax over the shared global TRAIN vocabulary → convex combination of frozen-SBERT label
  embeddings, temperature-calibrated on held-out source subjects.

---

## 2A. Channel heterogeneity in depth (the T2 design)

T2 is the richest axis and HALO's headline, so its handling is pinned down here. Refined 2026-07-12.

**Deployment unit = one device.** A phone-or-watch product runs on one device at a time, so the
fundamental sample is one device's `[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]`. A dataset that
recorded phone **and** watch (e.g. wisdm) or many positions (e.g. shoaib) contributes **one sample
per placement** — the same activity seen by a different device is just more data, never a 12-/30-wide
fused tensor. This keeps the max channel count at **6**, is deployment-realistic, matches how the
baselines are built (single device), and lets us evaluate every curated placement as its own sample
(real placement-robustness, not one cherry-picked stream). *(A phone+watch **fusion** showcase — 12
real channels — is a separate HALO-only experiment, not part of the core comparison.)*

**Two orthogonal knobs.** The raw tensor cannot encode placement — six accel/gyro numbers look the
same whether the device is a pocketed phone or a wrist watch. Placement-blindness is therefore a
property of the *tensor*; the only way to be placement-**aware** is a separate metadata-conditioning
signal. These are independent:

| Knob | Values | Nature |
|---|---|---|
| **Tensor cleanliness** | **harmonised** (canonical channel order, fixed shape, zero-pad+mask) ↔ **non-harmonised** (native order / variable count) | a **data** preprocessing choice |
| **Metadata conditioning** | **parity** (neutral text — blind) ↔ **full** (placement/sensor as language) | a **model capability** — HALO-only |

- *harmonised* = fixed 6-channel canonical layout, acc-only streams zero-pad+mask their gyro slots.
  One schema for the whole corpus — the input a fixed-layout model needs.
- *non-harmonised* = the curated stream at its **native width** (3 or 6), canonical order. A
  channel-flexible model (HALO, NormWear) ingests it directly; a fixed-input model must pad it up to
  its width — and because `deployment_policy` already canonicalises channel **order**, that padding
  yields the **same tensor** as the harmonised view. **So harmonised and non-harmonised COINCIDE for
  fixed-input baselines** (proven in `tests/test_baseline_view.py::test_non_harmonised_padded_to_six_collapses_onto_harmonised`).
  The pair is therefore **not** a knob that degrades fixed baselines on the native view; its point is
  that fixed baselines **require** the harmonised fixed schema to run at all, whereas HALO consumes the
  native view **without that alignment labor**. (A dataset-specific channel *scramble* would make fixed
  models degrade, but that is an unfair handicap, §3b.5 — we do not do it.)
- **Baselines are placement-blind regardless** — they have no conditioning mechanism, so the real
  advantage axis is **metadata conditioning**, not harmonised-vs-non-harmonised. HALO reads
  placement/sensor identity as language: HALO-parity (neutral text) lands on the blind baselines, and
  the gap **HALO-full − HALO-parity** *is* the value of the conditioning — the parity row is what
  attributes the win to conditioning rather than architecture.

**Why zero-pad+mask / native order is the accepted method (not imputation).** The two field-standard
neutral ways to feed a fixed-input baseline heterogeneous channels are (a) **zero-pad to the model's
native width** and (b) **restrict to the common accel-only subset**. Both are *null* operations — they
add no information — which is exactly why they are fair. The cross-dataset HAR literature standardises
the shared representation while preserving intrinsic differences (DAGHAR, Napoli et al. 2024 —
harmonises units/rate/gravity/labels/window) and treats *in-model* variable-channel handling as the
published novelty (Channel-Free HAR, Hasegawa 2026; oneHAR, Wei et al. 2025; NormWear, Luo et al.
2024; Wonderwall, Miao et al. 2026). That is the clincher: building a cross-channel adapter or imputer
to feed a baseline would hand it the exact capability those papers publish as their contribution —
i.e. solving the heterogeneity *on its behalf*.

| Channel method | Verdict | Why |
|---|---|---|
| Zero-pad missing channels to native width (+ mask) | ✅ standard | null op; model copes with absence as in deployment |
| Common accel-only subset | ✅ standard (DAGHAR, UniMTS) | uniform, but strips gyro from 6-ch models |
| Impute / synthesise missing channels | ❌ solves it for them | injects info the baseline couldn't get itself |
| Learned cross-channel adapter | ❌ solves it for them | that *is* the flexible models' published contribution |
| Random fill / scrambled order | ❌ unfair handicap | sabotage, not a null op |

**Fairness anchor:** on an acc-only device, **HALO also has only 3 real channels** — it cannot invent
gyro either — so no model ever receives a real channel another is denied. The zero-padded gyro a 6-ch
baseline sees is the same *absence* HALO sees; only the *representation* of that absence (zero-blind
vs mask vs language) differs, and that difference is the capability under test.

---

## 3. The faithfulness contract

**Principle.** Feed each model the input its *published design* requires, and handle each
heterogeneity axis with the *model's own mechanism* where it has one; where it has none, apply the
*single standard bridge* (anti-aliased resample / zero-pad+mask / ConSE) — **never** a
HALO-specific capability. Effort is symmetric across all models.

### 3a. Allowed — still a faithful, fair representation

1. **Resampling** our data (anti-aliased) to a model's required fixed rate. *(T1)*
2. **Channel coercion** into the model's expected layout: fixed-order zero-pad+mask (fixed models),
   per-channel + mask (channel-independent), SMPL-joint placement (UniMTS). *(T2)*
3. **Window fitting**: center-crop / wrap-pad to the model's fixed window length. *(T0/T1)*
4. **(Re)training the backbone on our corpus** for models *designed to be trained on your data* —
   CrossHAR/LiMU-BERT (self-supervised) — keeping the architecture and training recipe as published
   and changing **only** the corpus (and rate/window via the model's own config knobs). *(T0)*
5. **Fitting the ConSE head / temperature** on our **TRAIN** sets, with subject-disjoint splits and
   no target-label access. *(T3)*
6. **Using a model's native text tower** for open-set (UniMTS CLIP, NormWear TinyLlama). *(T3)*
7. **Repairing a released repo's broken code** *to restore its published behavior* (e.g. NormWear's
   GPU CWT path that `.numpy()`s a CUDA tensor). A bug-fix that makes the model run as its authors
   intended is faithful; a behavior *change* is not.
8. **Disclosing N/A** when a model would be fed out-of-contract data (e.g. a gravity-dependent model
   on a gravity-removed set) instead of reporting a physically invalid number.

### 3b. Disallowed — unfaithful or unfair

1. **Altering a frozen model's architecture or input contract** (rate, channel count) that its
   released weights depend on. (Feeding harnet 60 Hz or 6-ch makes its frozen kernels meaningless.)
2. **Granting a baseline a HALO-specific capability** it lacks in its paper — e.g. bolting language
   channel-conditioning onto CrossHAR, or making harnet rate-invariant.
3. **Any use of target labels at inference** (the LiMU-BERT sub-window transition filter we removed
   was exactly this — fixed).
4. **Asymmetric tuning**: more HPO, larger budget, longer schedule, or more augmentation for one
   side than the other.
5. **Random / fabricated channel fill** to satisfy a channel count (only zero-pad + mask).
6. **Reporting out-of-contract data** without the N/A disclosure from 3a.8.
7. **Silent coverage truncation** (top-N, sampling, no-retry) — if coverage is bounded, it is logged.

### 3c. The test to apply

Before any baseline change, answer: *"Would the model's authors recognise this as their method, run
on new data?"* If it only adapts the **data** to the model's published contract (3a.1–3), uses the
model's **own** mechanism, retrains a **train-on-your-data** model on our corpus without architectural
change (3a.4), or restores published behavior (3a.7) — it passes. If it changes what the **model** is,
or gives one side an edge the other lacks — it fails.

---

## 4. Per-baseline ledger

Exactly what we do to each model, and why it passes Section 3.

- **CrossHAR** *(we pretrain, SSL)* — T0 masked-reconstruction Transformer + `Transformer_ft`;
  retrained on our corpus at **60 Hz** via `seq_len`. T1 fixed→60 Hz. T2 fixed 6-ch acc+gyro,
  zero-pad+mask, placement-blind (per-window InstanceNorm ⇒ scale/gravity-immune). T3 closed→ConSE.
  *Faithful:* arch + recipe unchanged, corpus/rate swapped via its own config (3a.4).

- **LiMU-BERT** *(we pretrain, SSL)* — T0 LiMU-BERT encoder + GRU; retrained on our corpus at
  60 Hz. T1 fixed→60 Hz. T2 fixed 6-ch, `÷9.8` accel norm (keeps gravity), zero-pad+mask,
  placement-blind. T3 closed→ConSE. *Modification:* removed the eval-time GT sub-window filter
  (target-label leakage) — a *fairness restoration* (3b.3), not a capability change.

- **ssl / harnet5** *(frozen released — OxWearables, Capture-24 pretrain)* — T0 1D-ResNet, **frozen**.
  T1 fixed **30 Hz** (source resampled to 30, center-cropped to 5 s / 150). T2 fixed **3-ch
  acc-only**, wrist-implied, no gyro. T3 closed→ConSE head fit on our train sets. *Faithful:* no
  backbone change; only data adaptation (3a.1–3) + ConSE (3a.5).

- **UniMTS** *(frozen released — HF)* — T0 acc-only ST-GCN + fine-tuned CLIP text tower, **frozen**.
  T1 resample + wrap-pad to 200. T2 **native placement** (accel written to the dataset's SMPL joint,
  others zero), acc-only 3-ch, needs gravity-present m/s². T3 **native CLIP text** (cosine).
  *Faithful:* uses the model's own placement + text mechanisms; N/A on gravity-removed sets (3a.8).

- **NormWear** *(frozen released — GitHub)* — T0 channel-independent CWT-ViT + MSiTF + TinyLlama,
  **frozen**. T1 fixed **65 Hz** (source resampled to 65). T2 **native** variable real channels
  (per-channel + mask), no placement semantics; per-window z-score removes gravity internally. T3
  **native TinyLlama text** (L1 distance). *Modification:* GPU-CWT bug-fix to restore published
  behavior (3a.7).

- **HALO (ours)** — T0 physical-Hz filterbank tokenizer + dual-branch transformer + per-patch
  semantic head. T1 **native rate-invariant** (no resample; Nyquist masks). T2 **native**
  channel-independent + **language channel-conditioning** (placement + sensor + gravity semantics) +
  SO(3) rotation augmentation. T3 **native language alignment**. The `native` reference at every tier.

---

## 5. How the results read, tier by tier

- **Headline ZS-XD** — every zero-shot model at its native level → HALO's end-to-end advantage.
- **Parity row** — HALO neutralized to a baseline's level (fixed rate + neutral channel text) →
  isolates the T0 base architecture from HALO's T1/T2 heterogeneity mechanisms.
- **T1 ablation** — multi-rate evaluation: fixed-rate baselines resample/degrade, HALO stays flat.
- **T2 comparison** — the advantage axis is **metadata conditioning** (**parity** = neutral text /
  blind ↔ **full** = placement/sensor as language); see §2A. Every fixed baseline is *blind* and, per
  §2A, sees the **same** tensor whether the corpus is harmonised or non-harmonised — so the
  harmonised/non-harmonised pair is a **channel-flexibility** demonstration (HALO and NormWear ingest
  the native non-harmonised view directly; fixed baselines *require* the harmonised schema to run),
  **not** a knob that degrades fixed baselines. HALO-parity lands on the blind baselines; HALO-full
  adds conditioning and rises above them. The gap **HALO-full − HALO-parity** = the value of
  conditioning, and **baselines cannot close it** (no conditioning mechanism).
- **T3** — language alignment (HALO, UniMTS, NormWear) vs ConSE bridge (CrossHAR, LiMU-BERT, harnet).
Each tier is one axis, one policy, one comparison — which is exactly what keeps the code and the
paper clean.
