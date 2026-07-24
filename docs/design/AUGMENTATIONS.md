# Augmentation policy — what each model gets, and why

> ⚠️ **Partially stale (2026-07-24).** Some claims predate the SSL pivot (label IDs used by A2;
> gravity-gated rotation). Current facts: A2 is label-free SimCLR; SO(3) rotation now includes
> gravity-removed streams (`require_gravity=False`); a bounded **sensor-text dropout** was added.
> The authoritative Phase-A recipe table is in
> [`training/tokenizer/README.md`](../../training/tokenizer/README.md).

Augmentations run **on-the-fly** in the training dataloader (per-sample, stochastic) on top of the
pre-materialised grids. This doc answers two questions: (1) which augmentations each model can
*consume*, and (2) how they are *used* across our two experiments.

The policy is dictated by the contribution thesis (`MOTIVATION.md`): **the differentiator is test-time
language conditioning, NOT augmentation exposure.** So a robustness augmentation a fixed baseline can
consume must be applied to the baselines *too* — otherwise a reviewer correctly says "you just gave
HALO more diverse training data," and the conditioning result means nothing. Only augmentations a fixed
model **structurally cannot ingest**, or that **train HALO's language interface**, are HALO-exclusive.

## Active configuration (`AugmentationConfig.default_v2`, used by Phase-1 pretraining)

Source of truth: `data/scripts/augmentations.py`. Applied per-sample in the loader (`pretrain_data.py`).

| Aug | Enabled | p | Params |
|---|---|---|---|
| `jitter` | ✅ | 0.5 | σ = 0.05 |
| `scale` | ✅ | 0.5 | ×[0.9, 1.1] |
| `gravity` (remove gravity) | ✅ | **0.15** | cutoff 0.4 Hz — **lowered from 0.5** (audit: p=0.5 stripped gravity on 52 % of windows, killing the DC/gravity feature on half the corpus) |
| `rotation_3d` (SO(3)) | ✅ | 0.5 | requires gravity present (rotates acc+gyro jointly) |
| `rate` (resample) | ✅ | 0.5 | 15–100 Hz, min 32 samples |
| `channel_dropout` | ✅ | 0.3 | drops the `gyro` group (acc never dropped) |
| `window_crop` (P5) | ✅ | 0.5 | keep a random contiguous ≥50 % sub-window (session-length invariance); floor 32 samples |
| `channel_text_phrase` / `channel_text_dropout` | ✅ | 0.5 / 0.15 | text-side augs (channel-text paraphrase, channel-desc dropout) |
| `label_text` | ❌ **off in Phase-A** | (0.8) | label synonyms/templates — **computed then discarded** in pretraining: A2 contrasts on label **IDs**, not text, so the loader disables it (`pretrain_data.py:185`). A Pipeline-B (language-tower) concern. |
| `time_shift` / `time_warp` / `magnitude_warp` | ❌ off | — | available but disabled by default |

**Rate/length diversity is now REAL, not synthetic (changed 2026-07-18).** HALO trains on the
`native` grids (`build_grids._ALIGNMENTS`): the corpus's **native sampling rates** (20/50/100 Hz) and
**native window lengths**, *not* a 60 Hz resampled base. The 60 Hz "harmonised" grids are the
layout-locked baselines' crutch (CrossHAR/LiMU-BERT need a fixed rate); the filterbank tokenizer is
rate-invariant, so HALO does not. On top of the real native anchors, `rate` (15–100 Hz) interpolates
between them and `window_crop` varies the observation length — both are layout-breaking (Bucket 2),
so a fixed-window/fixed-rate baseline structurally cannot ingest them.

## Three buckets

### Bucket 1 — Symmetric robustness (apply to baselines too)

Layout-preserving: they keep the fixed 6-ch `[acc,gyro]` / 60 Hz contract, so a fixed baseline consumes
an augmented sample exactly like a real one. **Reserving these for HALO would confound conditioning
with augmentation exposure** — so in the conditioning experiment they are applied to HALO **and** the
retrained baselines equally.

| Aug | What it does | Why it's layout-preserving |
|---|---|---|
| `jitter` | additive Gaussian noise | shape/layout/rate intact |
| `scale` | random amplitude factor | amplitude robustness; layout intact |
| `magnitude_warp` | smooth per-timestep magnitude | shape robustness; layout intact |
| `time_warp` | smooth local time distortion (same length) | speed robustness; layout intact |
| `time_shift` | shift within the window | phase robustness; layout intact |
| `gravity` (P1) | remove/add the gravity DC (iOS userAccel ↔ Android total) | still 6-ch/60 Hz — a fixed model **can** train on it |
| `rotation_3d` (P2) | uniform SO(3) rotation of each co-located triad (gravity rotates with accel) | still 6-ch/60 Hz — a fixed model **can** train on it |

> **Note (changed 2026-07-12):** `gravity` and `rotation_3d` were previously filed as HALO-only. That
> was the old "augmentation as HALO's capability" framing. Under the `MOTIVATION.md` thesis they are the
> *core* of the conditioning experiment and therefore **must be symmetric** — the fixed baselines get
> them in training; only HALO can be *told* the transform at test time.

### Bucket 2 — HALO-only by necessity (layout-breaking)

A fixed-layout model **structurally cannot ingest** these — not a boost, a hard incompatibility.

| Aug | Axis changed | Why a fixed model can't take it |
|---|---|---|
| `rate` (P3) | sampling rate | a 60 Hz-fixed model must resample back, which cancels the augmentation |
| `channel_dropout` (P4) | channel count | drops channels → variable width; a fixed 6-ch model can't take it |
| `window_crop` (P5) | observation length | variable-length window → variable token count; a fixed-window model can't take it |

### Bucket 3 — HALO-only by design (interface-training)

These train HALO's **language interface**; a baseline has no text interface to train, so they are
legitimately exclusive.

| Aug | What it trains |
|---|---|
| `channel_text_phrase` | paraphrases the per-channel placement/sensor text |
| `channel_text_dropout` | drops channel metadata for robustness |
| `label_text` | paraphrases the label string (language-aligned label tower) |

## Two experiments, two policies

1. **Headline comparison table.** Each model uses **its own published training recipe** (faithfulness —
   "each at its fullest," `BASELINE_FAIRNESS_POLICY.md`). We do not graft our augmentations onto anyone.
2. **The conditioning demonstration** (the thesis, `MOTIVATION.md` §3). Train HALO **and the retrained
   baselines** (CrossHAR, LiMU-BERT) with the **same Bucket-1 transform augmentation** (equal exposure).
   At test time, apply the transform to the test data and give **only HALO** the acquisition descriptor.
   The gap is then purely *test-time conditioning access* — architectural, not a data advantage.

**Caveat.** Equal-exposure only works for the baselines we retrain (CrossHAR, LiMU-BERT); the **frozen**
baselines (harnet, UniMTS, NormWear) are as-released, so the clean conditioning control is HALO vs the
retrained baselines. Against frozen baselines the comparison is "off-the-shelf product vs HALO."

## Fairness guardrails (from `MOTIVATION.md` §4)

1. **Realistic transforms only** — orientation, gravity, placement, rate. Never an arbitrary corruption
   (channel scramble, additive garbage) engineered to break baselines; that is sabotage and reads as such.
2. **Equal augmentation exposure** in the conditioning experiment — baselines see the same transformed
   data; the only difference is being *told* the transform at test time.
3. **Descriptor ≠ answer** — the acquisition descriptor never leaks the label or target distribution.

This is consistent with `BASELINE_FAIRNESS_POLICY.md`'s asymmetry rule (never give one side more
augmentation than the other): the symmetric bucket is applied to both sides; the HALO-only buckets are
excluded because a fixed model *cannot consume them*, not because we are boosting HALO.
