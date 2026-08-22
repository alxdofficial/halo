# Motivation — what HALO is for, and why it is not trivial

This is the load-bearing "why" for the project. Every design choice (curation, the language interface,
the augmentation split, the baseline contract) traces back to the argument here. If a framing in a
paper draft cannot be defended against the rebuttals in §2, it is not the contribution.

> ## ⚠️ Evidence status — 2026-08-22. Read before pitching any of this.
>
> The argument below is unchanged and still the intended contribution. What has changed is that we
> now have measurements, and **the thesis's own decisive experiment has not been run**, while two
> measurements sit in tension with it. Stating that here so no draft asserts what we have not shown.
>
> **Not demonstrated — and the closest proxy came back NEGATIVE.** The full four-transform §3
> experiment has never been executed; it remains a plan in `AUGMENTATIONS.md`. But a weaker version
> *was* run: the **parity gate of 2026-08-11** (`DESIGN_OF_RECORD.md`, "PARITY GATE — Result: the
> conditioning is inert"). Trained with full text, evaluated with neutral text, held-out-config
> transfer: mean gain **+0.0086** against a 0.0065 noise floor — about 1.3 sd — and **the sign flips
> on two of four datasets** (realworld −0.007, inclusivehar −0.057). A real effect does not change
> sign across half the cohort. DESIGN_OF_RECORD calls that result "the measured motivation for the
> entire Phase-A redesign". So the input-side claim is not merely unproven; the one measurement we
> have points the wrong way.
>
> **Counter-evidence to weigh.** (a) Masking the acquisition descriptor at inference left
> cross-configuration retrieval **unchanged** (`DESIGN_AUDIT_20260821.md`); that is an
> inference-time ablation on a model trained *with* descriptors, not the §3 experiment, but it is
> not the result the thesis predicts. (b) The encoder's retrieval feature ranks by **acquisition
> configuration, not activity** (×7.0 lift for same-config rows; same-activity/different-device rows
> at the 39th percentile — `PHASE_B_DIAGNOSIS_20260820.md`). Conditioning is about being *told* a
> config rather than being invariant to it, so this is not a direct refutation — but "the model
> sorts by device" is the opposite of what a reader will expect the config interface to buy.
>
> **What IS demonstrated, and it is not this document's headline.**
> [`../results/ADAPTATION_TABLE_20260822.md`](../results/ADAPTATION_TABLE_20260822.md): the compact
> evidence engine is best in **35 of 40 enrollment columns** at d=128 — including *every*
> clinical/rehab column at every k. That is a **few-shot enrollment / memory-adaptation** result and
> it is the project's strongest evidence. Two qualifications: those columns fit generic heads on
> frozen features, so they demonstrate the **representation**, not the engine; and the zero-shot
> picture is split — ordinary **36.95** (2nd of 7, no fitted head) but specialized-novel **8.75**,
> **5th of 7**, behind UniMTS 19.24 and harnet 11.40. §2's headline is "unseen labels *and* unseen
> configs"; being 5th of 7 on precisely the novel-label regime is the most direct contradiction of
> the label half in this document. Underneath it, names and signals correlate at only **r = 0.11**
> across 105 labels on this corpus — a ceiling on any signal-based label bridge, independent of our
> design.
>
> **One more measurement, and it is the only one that favours the config story.** Our encoder carries
> **more cross-configuration structure than any encoder tested**, including harnet and UniMTS (raw
> kNN lift 2.82 vs 2.36 / 2.59 —
> [`../results/ENCODER_COMPARISON_20260822.md`](../results/ENCODER_COMPARISON_20260822.md)). That
> says the representation has the property; it does **not** say language conditioning produced it —
> the parity gate says the conditioning is inert.
>
> **Honest reading.** The measured strength (few-shot enrollment on unseen, semantically opaque
> activities — the "exercise 1" physiotherapy case) and the written thesis (language conditioning
> on unseen acquisition configurations) are **two different claims**. Either run §3 and earn the
> written one, or re-centre the contribution on what the evidence supports. Do not pitch the
> language-config interface as demonstrated.
>
> ⚠️ **And note a trap in the re-centring option:** random-alias episodes — the only training
> mechanism that produced the "exercise 1" capability — were **removed from the default objective on
> 2026-08-22**, because removing them was a wash on coherent performance. Removal cost that
> capability directly (alias-cell score 0.4906 → 0.2934). Re-centring on the physiotherapy case
> therefore requires **re-enabling `--alias-episode-fraction` and re-measuring**, not just
> rewriting the pitch.

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
| **HALO: per-channel/stream language description** | *input*: **unseen** configs | ⚠️ **not demonstrated** — parity gate inert (+0.0086, sign flips 2/4); descriptor masking leaves cross-config lift unchanged |

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
by invariance-vs-conditioning per §2.)

> ### ⚠️ Prior-art reality check (literature sweep, 2026-07-22) — this lane is CROWDED
>
> An earlier draft treated language-conditioned config handling as close to novel. **A direct
> literature sweep found otherwise**, and this is recorded so we do not overclaim (the same mistake
> that killed the pose-pretext idea by assertion). The differentiation left to us is **narrow** and
> must be established by close reading, not assumed:
>
> - **[GOAT](https://consensus.app/papers/details/ed52d465eb1d5e1f8a35fb611d3ba628/) (IMWUT 2024)** —
>   natural-language supervision over activity labels **and device on-body positions**, with an
>   explicit "novel device position encoding technique," for zero-shot cross-dataset HAR. This is the
>   config-side language conditioning, already published. *Open question that may be our daylight: is
>   GOAT's position encoding categorical/closed, or genuinely free-text?*
> - **[oneHAR / uniHAR](https://doi.org/10.1145/3749509) (IMWUT 2025)** — "one model to fit them all,"
>   LLM-assisted, handles a **variable number and combination of sensor positions**, device types
>   spanning phone/watch/glasses, IMU–text alignment. Directly targets our heterogeneity claim.
> - **[ActivityNarrated](https://consensus.app/papers/details/03e9ee0ed61d5708b0a51b30402f8e61/)
>   (2026)** — a "language-conditioned learning architecture over **variable-length sensor streams and
>   heterogeneous sensor placements**," open-vocabulary, **retrieval-based evaluation**. This is much
>   closer to our whole pitch than the "adjacent, output-side" framing above admitted — it overlaps
>   the open-vocab + heterogeneous-placement + retrieval-eval combination.
> - **[LanHAR](https://consensus.app/papers/details/2bfc914ea3265cb6a0b3c911238bc195/) (2024)**,
>   **[AnyMo](https://arxiv.org/html/2605.22715) (2026)**, **MobiDiary (2026)** — all condition on /
>   normalize across device+placement heterogeneity via language or geometry; MobiDiary even evaluates
>   on **xrf_v2**, one of our own datasets.
>
> **The narrow lane that MIGHT survive** (needs verification, not assertion): the specific
> *factorization* — a per-**sensor** free-text embedding (device + placement + modality) shared across
> that sensor's channels, a trivial fixed intra-sensor channel role, an arbitrary **set** of sensors,
> with the whole thing feeding a **non-parametric evidence engine** rather than a parametric head. No
> single found paper clearly does all of that, but the margin is thin. See `TEXT_CONDITIONING.md`.

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

> ⚠️ **This paragraph's final sentence ("We show that...") describes an experiment that has not
> been run** — see the evidence-status banner at the top. As written it would be an unsupported
> claim in a submission. Either run §3 first, or replace that sentence with the enrollment result
> we do have.

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
