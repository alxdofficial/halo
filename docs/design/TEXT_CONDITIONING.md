# Text conditioning — factoring channel text into per-sensor identity

> **Status: design, 2026-07-22.** A refactor of how the tokenizer injects acquisition-config
> identity. It changes the *factorization*, not the pipeline: same filterbank, same transformer,
> same frozen SBERT. Extends `MOTIVATION.md` (the config-side thesis) and the encoder in
> `model/tokenizer/`. Prior-art caveat: see the MOTIVATION §2b reality check — this lane is crowded
> and the factorization below is the narrow differentiation we still have to defend.

## 1. The idea, in one paragraph

An IMU sensor has a **fixed intra-sensor format**: 6 channels = accelerometer (x,y,z) then gyroscope
(x,y,z), or 3 channels for accel-only. *Which axis a channel is* is trivial and positional — it needs
no rich language ("gyro y" is a constant). The **load-bearing** identity is at the **sensor** level:
*what device this is and where it sits on the body* — "smartwatch on the left wrist", "smartphone in
the right pocket", "accelerometer-only earbud". So the language conditioning should attach **once per
sensor** and be **shared by that sensor's channels**, while the within-sensor role is a small fixed
embedding. The model then makes **no assumption about how many sensors there are or in what order** —
only that each sensor is of a known channel format. Arbitrarily many or few sensors compose as a set.

## 2. What the current code actually does (verified against `model/tokenizer/`)

Today the text is **per-channel**, and the sensor-level information is **duplicated across the six
channel strings**:

```
stream_channel_descriptions('wisdm','watch_wrist') ->
  'accelerometer x-axis worn at the wrist (watch); includes gravity'
  'accelerometer y-axis worn at the wrist (watch); includes gravity'
  'accelerometer z-axis worn at the wrist (watch); includes gravity'
  'gyroscope x-axis worn at the wrist (watch)'
  'gyroscope y-axis worn at the wrist (watch)'
  'gyroscope z-axis worn at the wrist (watch)'
```

Pipeline (files):
1. **`filterbank.py` · `PhysicalFilterbankTokenizer`** — each native-rate patch of each channel →
   one `d_model` token via a physical-Hz constant-Q filterbank. Output `sensor_tokens (B,P,C,d)`.
   Channels are independent here; there is **no sensor grouping** — C is a flat channel axis.
2. **`channel_text.py` · `TokenTextEncoder`** — frozen SBERT, cached, per unique string →
   token-level embeddings `(C,S,384)`.
3. **`channel_text.py` · `ChannelTextFusion`** — pools each **channel's** text (learned-query
   cross-attention) to one `(B,C,D)` channel embedding, then **gated broadcast** onto every patch:
   `fused = sensor_tokens + sigmoid(Wq·sensor + Wc·channel) * channel`. So identity enters as a
   per-channel additive, gated bias, constant over patches.
4. **`transformer.py` · `DualBranchTransformer`** — factorized attention: **temporal** (patches
   within a channel, physical-time RoPE) then **cross-channel** (channels within a patch, mask-aware).
   Channels carry **no positional index** — identity is entirely the fused text, which is what makes
   channel count/order free.
5. **`encoder.py` · `SetTokenizerEncoder`** — orchestrates tokenize → encode_texts → encode, masks
   for A1 *before* fusion (so the [MASK] token still gets its channel's identity), pools to
   `per_patch (B,P,d)` and `pooled (B,d)` with channel/patch masks respected.

**So the tokenizer is already a text-keyed set over channels.** The proposed change is not "add
language conditioning" — it is **re-factor the identity so it is keyed at the sensor level**, which is
where placement/device/modality actually live, instead of being redundantly stamped on each channel.

## 3. Why the refactor is worth doing (beyond tidiness)

1. **Separates the contribution from the table-stakes.** Intra-sensor channel order is preprocessing;
   sensor identity is the claim. Factoring makes each visible and independently ablatable, and stops a
   reviewer conflating them.
2. **Removes redundant text work.** Six near-identical strings per sensor collapse to one sensor
   description + a tiny fixed role code. Fewer unique SBERT strings per batch.
3. **Makes the set structure explicit.** "Arbitrary number of sensors, each a fixed channel block"
   is the honest scope statement, and it is what the pitch should claim (not "no assumptions at all,"
   which a reviewer breaks with "you assume acc-then-gyro").
4. **Gives the acc-only case a clean home.** A 3-channel sensor is not "missing" channels — its
   sensor description carries the modality ("accelerometer-only …") and the gyro role slots are simply
   absent from that sensor's block, no zero-pad-and-mask needed at the identity level.

## 4. The scope statement (write it exactly this way in the paper)

> HALO assumes each IMU sensor has a **known channel format** (accelerometer ± gyroscope, in a fixed
> intra-sensor order). It makes **no assumption about the number of sensors, their ordering, or their
> on-body placement**; each sensor's device/placement/modality is supplied as free text and enters as
> a per-sensor identity shared by that sensor's channels. Unseen placements/devices arrive as unseen
> *descriptions* and compose through language.

## 5. How to fold it in — smallest change that is honest

The current architecture already supports this with a **localized change**, because identity is
already additive and channel positions are already index-free. Two implementable options:

**Option A — sensor-grouped text (minimal, recommended first).**
Keep the per-channel token axis `C`, but:
- Change `stream_channel_descriptions` to emit, per channel, `sensor_text ⊕ role_text` where
  `sensor_text` is the shared per-sensor string ("smartwatch, left wrist, accelerometer+gyroscope")
  and `role_text` is a tiny constant ("accel x"). Simplest: build the fused identity as
  **`sensor_embedding + role_embedding`**, where `sensor_embedding` is the SBERT pooling of the
  sensor string (shared across its 6 channels, computed once) and `role_embedding` is a small
  learned `nn.Embedding(6, d)` indexed by intra-sensor role.
- This is a change to `ChannelTextFusion` (or a thin wrapper): pool **sensor** text once per sensor,
  add the role code, then the existing gated broadcast is unchanged.
- A new input is needed: a `sensor_id (B,C)` grouping that says which channels belong to which
  sensor (so the shared sensor embedding is broadcast to the right channel block). For single-sensor
  streams (most of the corpus) this is trivially all-zeros; it matters for the simultaneous-stream
  multi-sensor training discussed in `ENROLLMENT_BY_DEMONSTRATION.md`.

**Option B — a true sensor token (larger, later).**
Add a per-sensor summary token (its identity = sensor SBERT embedding) that participates in
cross-channel attention as an extra "channel," and let the 6 real channels carry only role codes.
This is closer to the conceptual model but changes the attention shape and pooling; defer until A is
validated.

**Identity-at-init / do-no-harm:** initialise `role_embedding` small (or zero) and keep the fusion
gate init unchanged, so Option A at init ≈ the current model with sensor text broadcast — no
regression before the role code learns anything.

## 6. What must be verified before/after (do not skip — see the pose lesson)

1. **Novelty daylight.** Close-read GOAT (categorical vs free-text position?), oneHAR, and
   ActivityNarrated to confirm the per-sensor-factorization + evidence-engine combination is not
   already covered. If it is, this is a refactor for cleanliness, not a contribution.
2. **Does it help or is it neutral?** Ablate: current per-channel text vs Option A sensor+role, on
   the same held-out configs. Neutral-but-cleaner is an acceptable outcome; worse is not.
3. **Does the multi-sensor path actually exist in the data?** Option A's `sensor_id` grouping is only
   exercised by simultaneous multi-sensor streams — verify those are time-aligned (the open
   assumption from `ENROLLMENT_BY_DEMONSTRATION.md` §8), else the multi-sensor claim is untested.

## 7. Relationship to the rest of the project

This sits on the **input/config axis** (`MOTIVATION.md`), which the literature (UniMTS ablation:
config −16.7 F1 vs text −5.5) says is the expensive axis. It does **not** by itself make the model
usable (`POSITIONING.md`: usable accuracy is the bar, not beating harnet by 3 points). It is a
prerequisite cleanliness change so that, when the evidence engine appends new configs at test time,
the thing it appends is a **sensor identity** it can generalize over — not six redundant channel
strings.
