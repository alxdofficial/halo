# Augmentation viability — which augmentations are safe for which models

Augmentations run **on-the-fly** in the training dataloader (per-sample, stochastic), on top of the
pre-materialised grids. The question here: which of them are safe to apply to **every** model, and
which only make sense for a **heterogeneity-aware** model (HALO).

The dividing line is simple: **does the augmentation preserve the fixed input contract of a
fixed-layout model** (fixed 6-channel `[acc,gyro]`, fixed 60 Hz, no metadata channel)? If yes, it's a
plain signal-space augmentation any model can consume. If it changes the *rate*, *channel set*,
*gravity content*, or *text metadata*, a placement-/rate-blind model either cannot ingest it (you'd
have to resample/repad it back, undoing the augmentation) or is pushed out of its training
distribution — so it belongs to the heterogeneity-aware set only.

## A · Safe for ALL models (signal-space, format-preserving)

These are the classic HAR augmentations. They keep the 6-ch / 60 Hz layout intact, so a fixed
baseline consumes an augmented sample exactly like a real one, and they broadly help generalization.

| Aug | What it does | Why it's safe |
|---|---|---|
| `jitter` | additive Gaussian noise | preserves shape, layout, rate |
| `scale` | multiply by a random amplitude factor | amplitude robustness; layout/rate intact |
| `magnitude_warp` | smooth per-timestep magnitude variation | shape robustness; layout/rate intact |
| `time_warp` | smooth local time distortion (same window length) | speed robustness; layout/rate intact |
| `time_shift` | shift the signal within the window | phase robustness; layout/rate intact |

## B · Heterogeneity-aware only (HALO)

Each of these changes an axis a fixed-layout model is welded to, so applying it to such a model is
either impossible or actively harmful. They are exactly the heterogeneity HALO is built to handle.

| Aug | Axis changed | Why it's HALO-only |
|---|---|---|
| `gravity` (P1) | gravity content | removes/adds the gravity DC (simulates iOS userAcc). A gravity-present-trained fixed model goes out-of-distribution. |
| `rotation_3d` (P2) | orientation / placement | full uniform SO(3) rotation of each co-located triad (gravity rotates with accel). Teaches orientation invariance a fixed phone-in-pocket model isn't meant to have. |
| `rate` (P3) | sampling rate | anti-aliased resample to a random Hz — a 60 Hz-fixed model can't ingest it without resampling back (which cancels the aug). |
| `channel_dropout` (P4) | channel count | drops sensor channels → variable width; a fixed 6-ch model can't take it. |
| `channel_text_phrase` | channel-text metadata | paraphrases the per-channel placement/sensor text — only HALO reads channel text. |
| `channel_text_dropout` | channel-text metadata | drops channel metadata for robustness — HALO-only signal. |
| `label_text` | label text | paraphrases the label string — only language-aligned models (HALO) use it. |

## Fairness note (how this interacts with the baseline contract)

The **retrained** baselines (CrossHAR, LiMU-BERT) are trained with **their own published augmentation
recipe** — that's the faithful choice (`BASELINE_FAIRNESS_POLICY.md` §3a.4). We do **not** graft our
augmentations onto them, and the asymmetry rule (§3b.4) forbids giving one side more augmentation than
its recipe. So set **B is HALO-only in practice**, not because we're boosting HALO, but because those
augmentations produce data a fixed model structurally cannot consume. Set **A** is the pool of
augmentations that *would* be safe if we ever ran a symmetric-augmentation ablation (they don't break
any model's input contract). Frozen baselines (harnet, UniMTS, NormWear) are never trained by us, so
augmentation does not apply to them at all.
