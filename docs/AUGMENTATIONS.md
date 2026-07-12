# Augmentation policy — what each model gets, and why

Augmentations run **on-the-fly** in the training dataloader (per-sample, stochastic) on top of the
pre-materialised grids. This doc answers two questions: (1) which augmentations each model can
*consume*, and (2) how they are *used* across our two experiments.

The policy is dictated by the contribution thesis (`MOTIVATION.md`): **the differentiator is test-time
language conditioning, NOT augmentation exposure.** So a robustness augmentation a fixed baseline can
consume must be applied to the baselines *too* — otherwise a reviewer correctly says "you just gave
HALO more diverse training data," and the conditioning result means nothing. Only augmentations a fixed
model **structurally cannot ingest**, or that **train HALO's language interface**, are HALO-exclusive.

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
