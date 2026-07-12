# Motivation — what HALO is for, and why it is not trivial

This is the load-bearing "why" for the project. Every design choice (curation, the language interface,
the augmentation split, the baseline contract) traces back to the argument here. If a framing in a
paper draft cannot be defended against the rebuttals in §2, it is not the contribution.

## 0. The task

Zero-shot, **open-set** human activity recognition on consumer **phones and smartwatches**: recognize
activities — including ones never labeled in training — from a **single on-device IMU stream**, across
the messy heterogeneity of real deployments (body placement, mounting orientation, sensor set,
gravity convention, sampling rate), **without retraining per device or per configuration**.

## 1. Framings that look like contributions but are not

Two tempting pitches die to a one-line reviewer rebuttal. We state them so nobody resurrects them.

- **"Handles a variable number / order of channels."**
  *Rebuttal:* pad-to-width + mask and canonical reordering are one line of preprocessing. There is no
  research problem here. (Our curated corpus is only **3↔6** channels anyway — one device, acc or
  acc+gyro — so there is not even a scale story to hide behind.) Channel-flexibility is a *mechanism*
  HALO uses to avoid a per-config input head; it is **not** a contribution and must not be pitched as one.

- **"Zero-shot to unseen activity labels."**
  *Rebuttal:* **ConSE** (Norouzi et al., 2014) already bridges a closed-vocabulary classifier to unseen
  labels by interpolating label embeddings. We deliberately equip every closed-vocab baseline with ConSE
  (see `BASELINE_FAIRNESS_POLICY.md`), so open-set-labels is **table stakes**, not a differentiator.
  It is necessary for the task and insufficient as a claim.

## 2. The actual contribution: one language interface for BOTH unseen labels AND unseen acquisition configs

ConSE and a config one-hot are each half of what's needed, and each is limited:

| interface | axis it covers | limitation |
|---|---|---|
| **ConSE** (label→embedding) | *output*: unseen **labels** | label side only; **no** input-side mechanism |
| **config one-hot / extra features** | *input*: **seen** configs | cannot represent a config **not seen at train time** |
| **HALO: per-channel/stream language description** | *input*: **unseen** configs | — |

HALO conditions its **encoder** on a free-text description of how each channel/stream was acquired
(placement, sensor modality, gravity state, mounting orientation / applied transform). Because the
interface is **language**, it generalizes **compositionally to descriptions never seen in training** —
a new placement phrase, a new sensor, a *combination* of transforms. That is the **input-side analogue
of open-set labels**.

> **The thesis in one sentence:** HALO uses a single natural-language interface to achieve zero-shot
> generalization on *both* axes at once — *what to recognize* (unseen labels) and *how it was acquired*
> (unseen sensor configurations / transforms) — and the input axis is one **no** baseline can match:
> ConSE is label-side only, a config one-hot covers only seen configs.

### Why this beats preprocessing / invariance (the crisp version)

Preprocessing and learned invariance must **commit to one fixed policy at train time** and apply it
blindly (resample to X Hz; normalize gravity away; augment for rotation-*invariance*). Conditioning
**defers the policy to test time** and selects it from a language description — including descriptions
and combinations never seen. **That deferral is the thing you cannot preprocess.** And conditioning is
strictly richer than invariance: invariance is *lossy* (rotation-invariance discards the orientation
some activities depend on), whereas conditioning stays orientation-*aware* and adapts only when told.

## 2b. Closest prior art — and the one-line differentiation

The nearest work, and the one a reviewer *will* raise, is **ZeroHAR** (Chowdhury et al., AAAI 2025) —
notably from the same group as UniMTS (shared author). ZeroHAR also conditions on sensor context and
aligns the motion latent to it. **The distinction is the interface:**

> ZeroHAR conditions on a **fixed, closed set of context attributes** (sensor type, Cartesian axis, body
> position — a categorical vector). **HALO conditions on free-form natural-language** channel
> descriptions, which is what lets it **generalize compositionally to acquisition descriptions never
> seen in training** (a new placement phrase, a new sensor, a combination). Fixed attributes cover only
> the configurations enumerated at train time; language does not. That is the same open-set argument as
> for labels, now on the input side.

(No code/weights are publicly released for ZeroHAR as of 2026-07, so it is a **related-work / cite**
target, not a runnable frozen baseline. UniMTS remains the runnable closest competitor, differentiated
by invariance-vs-conditioning per §2.) Adjacent, on the *output* side: **ActivityNarrated** (Ray et al.,
2026) explores open-vocabulary sensor-language recognition — relevant to `LANGUAGE_HIERARCHY.md`.

## 3. The falsifiable claim and the experiment that ConSE cannot rebut

**Claim.** Under realistic, physically-meaningful **input-side** nuisance transforms, a model *told* the
transform in language (HALO) retains accuracy where an equally-trained model that *cannot* be told —
even one with ConSE on the label side — degrades.

**Transforms (all real deployment variation):**
- arbitrary **mounting orientation** — uniform SO(3) rotation with gravity rotating along the device;
- **gravity present vs removed** — Android total-acceleration vs iOS `userAcceleration`;
- **placement** change (pocket ↔ waist ↔ wrist);
- **sampling rate** change.

**Controlled demonstration.** Hold model, weights, and training-time augmentation fixed; vary only
whether the acquisition **descriptor is provided at test time**. `HALO+descriptor` vs `HALO−descriptor`
vs `baseline` isolates the value of input-side conditioning. ConSE cannot close the gap (it never
touches the input); a config one-hot cannot close it for transforms/configs unseen at train time.

## 4. Fairness guardrails — a demonstration, never sabotage

The experiment in §3 is only convincing if a reviewer agrees it is fair. Three rules:

1. **Realistic transforms only.** Every transform must be variation that genuinely occurs in phone/watch
   deployment (orientation, gravity convention, placement, rate). **Never** an arbitrary corruption
   (channel scramble, additive garbage) engineered to break baselines — that is sabotage and reads as such.
2. **Equal augmentation exposure.** Baselines are **trained with the same augmentations** (they see
   rotated / gravity-altered data too). The gap is *not* "baselines never saw it"; it is "baselines
   cannot be **told** which transform applies at test time and switch behavior." The difference is
   architectural, not a training-data advantage.
3. **Conditioning, not cheating.** The descriptor describes the *acquisition*, not the *answer*. It never
   leaks the label or the target distribution.

See `AUGMENTATIONS.md`: the "HALO-only" augmentation set (gravity P1, SO(3) rotation P2, rate P3,
channel-dropout P4, channel-text) is exactly the set that instantiates this thesis — each is a
real-world transform that is describable in language and that a fixed-layout model structurally cannot
be told about.

## 5. What this means for the corpus (no rework)

The data pipeline is unchanged. We curate every dataset to one deployment-realistic phone/watch stream
and preserve its **native heterogeneity** (placement / gravity / rate / modality) as the substrate; the
harmonised view exists only to give fixed baselines an equal footing (`DATA_PIPELINE.md`,
`BASELINE_FAIRNESS_POLICY.md`). The augmentations are the controlled knob that turns that heterogeneity
into the §3 experiment. Nothing here changes the converters or grids — it fixes what we **claim** and
which experiments **lead**.

## 6. One-paragraph pitch (for abstracts / intros)

> Consumer phones and watches produce IMU streams that vary in placement, mounting orientation, sensor
> set, gravity convention, and sampling rate, and are asked to recognize activities never seen at
> training time. Prior zero-shot HAR addresses only the *label* side (e.g. ConSE-style embedding
> bridges), and channel/rate differences are dismissed as cheap preprocessing. HALO is a
> language-aligned IMU foundation model that exposes a **single natural-language interface for both the
> label side and the acquisition side**: it recognizes unseen activities *and* adapts to unseen,
> language-described sensor configurations and transforms — the latter a capability no label-side bridge
> or fixed-config code can provide. We show that under realistic deployment transforms (arbitrary
> orientation, gravity removal, placement and rate changes), a model told the transform in language
> retains accuracy where equally-trained baselines degrade.
