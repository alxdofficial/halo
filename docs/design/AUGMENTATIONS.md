# Augmentation policy — what each model gets, and why

> ## ⚠️ The "Active configuration" table is wrong about the defaults — 2026-08-22
> It lists `jitter` and `scale` as ✅ enabled at p = 0.5. Those are the *dataclass field* defaults,
> which `AugmentationConfig.phase_a()` immediately **overwrites with 0.0**, and the CLI flags default
> to `None`. **As coded, the Phase-A reference recipe is fully clean** — jitter/scale at 0.5 is a
> value someone must pass explicitly. Same wording error in `DESIGN_OF_RECORD.md`.
>
> **Also missing entirely: `AugmentationConfig.phase_b_generic()`**, which is what the episodic
> trainer uses under `--augment` — and it enables **jitter 0.7, scale 0.7, and `rotation_3d` 0.8**.
> That directly contradicts this document's own banner ("independent SO(3) rotation is
> known-harmful, 0.085 below a random-init trunk") and its table listing `rotation_3d` at ❌ 0.0. No
> baseline receives `phase_b_generic`, so it also sits outside the Bucket-1 equal-exposure rule.
> Reconcile before citing any augmentation policy.
>
> **A third fairness arm now exists** and is not covered here: harnet/UniMTS running frozen *inside*
> HALO's own pipeline (`model/tokenizer/baseline_backbone.py`, `--encoder-backbone`). Two facts that
> belong in the fairness section — augmentation is *forbidden* in that arm, and it drops gyroscope
> and gravity-removed rows in **every** arm, so it is not the same data as the headline table.

> **Status 2026-08-22 — the conditioning demonstration below has NOT been run.** It is still the
> thesis's decisive experiment (`MOTIVATION.md` §3) and still unexecuted. Weigh it against the
> counter-evidence recorded in `MOTIVATION.md`'s banner: inference-time descriptor masking left
> cross-configuration retrieval unchanged. Also note the training recipe has moved on — Phase-A
> cross-placement was deleted 2026-08-06 (recipe is JEPA + aug-VICReg), and random-alias episodes
> were removed from the Phase-B default on 2026-08-22.

> **Updated 2026-08-18.** The reference recipe enables **jitter and scale** only; every other
> transform is an explicit, one-at-a-time ablation with a CLI flag. Independent-SO(3) rotation is
> known-harmful (trained 0.085 below a random-init trunk). Adding the rest of the historical stack
> (gravity, rate, channel dropout, text) moved held-out transfer by less than the noise floor and
> did not reduce subject leakage. Evidence:
> [`docs/results/PHASE_A_RECOVERY_20260818.md`](../results/PHASE_A_RECOVERY_20260818.md).
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

## Active configuration (`AugmentationConfig.phase_a`, used by Phase-A pretraining)

Source of truth: `data/scripts/augmentations.py`. Applied per-sample in the loader (`pretrain_data.py`).

| Aug | Enabled | p | Params |
|---|---|---|---|
| `jitter` | ✅ | 0.5 | `--jitter-p`. In the reference recipe: perturbs sensor noise without touching gravity-frame orientation |
| `scale` | ✅ | 0.5 | `--scale-p`. In the reference recipe, same rationale as jitter |
| `gravity` (remove gravity) | ❌ | 0.5 | `--gravity-p`; CONFIG-group. Tested 2026-08-18: no effect beyond noise |
| `rotation_3d` (SO(3)) | ❌ | 0.0 | explicit `--rotation-p`; shared across views unless `--rotation-pairing independent` is requested |
| `rate` (resample) | ❌ | 0.5 | `--rate-augmentation-p`; CONFIG-group. Tested: no effect beyond noise. Native rates already give real diversity |
| `channel_dropout` | ❌ | 0.3 | `--channel-dropout-p`; CONFIG-group. Tested: no effect beyond noise |
| `window_crop` | ❌ | 0.5 | available as an ablation |
| text paraphrase/dropout | ❌ | varies | available as ablations |

The clean recipe makes the first run interpretable and preserves gravity-frame orientation, which is
itself discriminative for posture and limb motion. Each transform is added alone and retained only
when frozen downstream development evaluation shows a repeatable gain. A shared rotation changes the
acquisition configuration without asking VICReg to erase it. Independent rotation is retained only
as the explicit invariance control that reproduced the failed 2026-08-17 recipe. The objectives
remain JEPA plus VICReg.

**Rate/length diversity is now REAL, not synthetic (changed 2026-07-18).** HALO trains on the
`native` grids (`build_grids._ALIGNMENTS`): the corpus's **native sampling rates** (20/50/100 Hz) and
**recording data**, *not* a 60 Hz resampled base. Native grids use non-overlapping six-second contexts
and retain the final shorter context with an explicit valid length. The collate then forms sequential
one-second patches, including an honest final short patch. No synthetic rate or crop transform is part
of the reference run.

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

## Comparison policy

The headline comparison uses only author-released external checkpoints, frozen and evaluated with
their published input contracts. We do not graft HALO augmentations onto them or reproduce their
pretraining. The causal augmentation and conditioning evidence therefore comes from within-HALO
ablations: hold the corpus, checkpoint-selection rule, and evaluation manifest fixed while changing
only the HALO augmentation or descriptor input under study.

## Fairness guardrails (from `MOTIVATION.md` §4)

1. **Realistic transforms only** — orientation, gravity, placement, rate. Never an arbitrary corruption
   (channel scramble, additive garbage) engineered to break baselines; that is sabotage and reads as such.
2. **One change per HALO ablation** — compare the same HALO training recipe with and without the
   transform or conditioning signal; do not attribute a cross-model difference to augmentation.
3. **Descriptor ≠ answer** — the acquisition descriptor never leaks the label or target distribution.

This keeps augmentation attribution separate from the frozen-checkpoint baseline comparison.
