# The language-native hierarchy: conditioning → concepts → labels (with graceful degradation)

A design direction that extends the core thesis (`MOTIVATION.md`). It is a **second act** — to be
implemented after base HALO + the conditioning experiment are validated — but recorded now so we don't
lose it.

## The idea in one sentence

Language is HALO's interface at **every level** of the pipeline — it conditions the *input* on the
acquisition setup, predicts interpretable *intermediate* motion concepts, and matches *output* open-set
labels — and when no label is confident the model **degrades gracefully** to those concepts + a
retrieval hypothesis instead of forcing a wrong guess.

```
raw IMU  →  tokens  →  [ acquisition conditioning ]  →  MOTION CONCEPTS (attributes)  →  label
                          (input-side, MOTIVATION.md)     (new: intermediate, in language)   (open-set)
                                                                     │
                                              if no label is confident ▼
                                       graceful fallback: emit concepts + nearest-exemplar hypothesis
```

## What the intermediate layer is

A small set of **universal, interpretable motion descriptors** predicted *before* the label — e.g.
`stationary | moving`, `low | moderate | high` intensity, `cyclic | acyclic`, `upright | lying`,
cadence / dominant band, `transitioning | steady`. Structurally this is a **concept bottleneck**
(Koh 2020) / **attribute-based zero-shot** layer (Lampert 2009); in HAR specifically, semantic
attributes for zero-shot were done by *NuActiv* (Cheng 2013).

**Honest prior-art note.** Attributes-for-HAR-ZSL is *not* new by itself. What is fresh here is doing it
**inside a language-native foundation model** as a **graceful-degradation + open-set** layer, unified
with the input-side acquisition conditioning — not "we use attributes."

## Why it fits HALO unusually well (the synergy)

HALO's tokenizer already computes the physics these concepts are made of — so they are nearly free and
each grounds in a component that already exists:

| Concept | Grounded in (already in HALO) |
|---|---|
| cyclic / cadence / dominant band | the **physical-Hz filterbank** |
| upright / lying (posture) | the **signed DC / gravity feature** |
| stationary / moving, intensity | signal energy (AC power) |
| transitioning / steady | the **boundary head** |
| "similar to something I've seen" | the **MoCo memory bank** (retrieval) |

Crucially, the concept layer can be **weak-supervised from the signal itself** (variance → intensity,
spectral peak → cadence, gravity direction → posture), so it needs **no manual annotation**.

## Why it strengthens the contribution

1. **It is the label-side twin of the Tier-1 experiment.** Tier-1 gives compositional generalization on
   the *config* axis (unseen acquisition descriptions); concepts give it on the *label* axis (an unseen
   activity is still describable by its attributes). Together: **one language interface, compositional
   generalization on both axes.** That is a *unified* story, not scope-creep.
2. **Graceful degradation is a capability no baseline has.** UniMTS / NormWear / harnet *force* an argmax
   over candidate labels — they cannot say "moving, cyclic, unknown activity." An abstain-with-a-
   description behavior is more honest and more deployable, and it is a clean **open-set recognition**
   (reject-the-unknown) mechanism that is *informative* rather than a bare "unknown."
3. **Interpretability / debuggability** — a legible intermediate layer, which reviewers value and which
   makes failures diagnosable.

## How to build it (cheap, staged)

- A **concept head** on the pooled / per-patch embedding predicting ~6–8 language attributes,
  weak-supervised from signal statistics (no manual labels).
- The **label head conditioned on the concepts** (concept bottleneck) *or* run in parallel — an ablation
  decides which is better.
- A **confidence-gated fallback** at inference: if max label-similarity < a calibrated threshold →
  emit the predicted concepts + the nearest memory-bank exemplar (with provenance) instead of a forced
  label.

## New evaluation axes it unlocks

- **Graceful-degradation quality:** when the label is wrong, are the concepts still right?
- **Open-set rejection:** does it correctly abstain on genuinely-unknown activities (and describe them)?
- **Concept accuracy** vs the weak-derived attributes (a sanity/interpretability check).

These are metrics the baselines structurally cannot report, so they showcase the capability.

## Scope caution

This is a training-objective + inference addition. **Validate base HALO + the input-side conditioning
first** (`MOTIVATION.md` §3) before layering concepts — do not build a tower on an unvalidated base. It
is a strong second act, not the first thing to add.

Related: `MOTIVATION.md` (the core thesis), `AUGMENTATIONS.md` (the conditioning experiment).
