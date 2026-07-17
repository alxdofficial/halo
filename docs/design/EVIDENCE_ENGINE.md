# HALO as an Evidence Engine — working design

> **Status:** working design, NOT finalized. This is a *separate line of work* from any
> conventional (softmax/cosine-classifier) training path. It reuses the **tokenizer**; the
> **model mechanism and training harness are different** and live in their own tree (see
> "Code organization" below). Name "evidence" is a placeholder — rename freely.
>
> Captured from a design discussion (2026-07-13/14). Prose is deliberately about *why*, not
> code; nothing here is committed as final.

## 0. The shift
From an **activity classifier** (emit one label) to a **human-activity foundation model**
(emit a *supported, calibrated, extensible analysis*). At inference on a dataset it has
never seen, the model behaves like: *"I'm streaming an unknown signal; it resembles these
past labeled things with these confidences; here is the evidence for/against each candidate;
I may abstain."* The structured analysis — not the top-1 label — is the deliverable, and it
survives even when the label is wrong.

## 1. Thesis: salient, NOT invariant (config-conditional)
The field chases **invariance** (augment/normalize away orientation, rate, placement). But
invariance destroys information — an orientation-invariant feature like `|accel|` is robust
*because* it threw away the direction that discriminates activities. **We want features that
stay class-discriminative across heterogeneity because the model is TOLD the config (via
free-form channel text) and normalizes *conditionally*** — not blindly projecting the
nuisance out. Precise objective: same activity across two sensors/placements/times maps
*close* **given each config's description**, while *keeping* the discriminative structure.

Differentiators: vs **UniMTS** (blind orientation-invariance) → config-conditional salience;
vs **ZeroHAR** (fixed attribute schema) → emergent physical primitives + free-form text;
vs **ConSE** baselines (heuristic convex blend of seen labels) → a learned, calibrated,
abstaining, structured evidence decoder.

## 2. Architecture (concern flow)
```
signal ──▶ TOKENIZER (shared, physical filterbank + extensions)  ──▶ multi-view PRIMITIVES
                                                                        (band × channel × cross-channel)
PRIMITIVES ──▶ (config-conditional) ENCODER ──▶ query primitives
query ──▶ ARCHETYPAL MEMORY (curated, non-parametric, evolving) ──▶ retrieved evidence
{retrieved evidence + candidate text hypotheses} ──▶ EVIDENCE DECODER ──▶ ANALYSIS
ANALYSIS = per-candidate {evidence-for, evidence-against, analogues, calibrated confidence, ABSTAIN}
```
The **feature unit is a primitive, not a patch.** A patch is a time-box; the *feature* is
"this band's energy on this channel," "this cross-channel phase pattern," "this cadence."
A never-seen activity is then a **novel combination of familiar primitives** → retrieval
*composes* instead of matching whole embeddings.

## 3. Two evidence views (unified)
- **View 1 — retrieval/analogy:** "resembles labeled memory X, with confidence." (data-driven;
  Matching/Prototypical Nets, kNN-LM/RETRO; a *learned, calibrated* generalization of ConSE.)
- **View 2 — hypothesis/abductive:** "for Y to be true I'd expect A,B,C; seeing them raises Y,
  contradicting them lowers Y." (model-driven; energy-based / analysis-by-synthesis /
  predictive-coding; concrete instantiation = **Evidential Deep Learning** = Dirichlet evidence
  per class → confidence + "I don't know".)
- **Unification:** both are *evidence sources feeding one decoder* that **accumulates across
  patches** into per-candidate beliefs + abstention. Streaming = sequential evidence
  accumulation (Bayesian / drift-diffusion).

## 4. Archetypal memory (non-parametric, curated, evolving)
Confirmed non-parametric: stored real features, *not* free-parameter vectors; a **learnable
GATE** decides what to keep/evict, everything **soft** (differentiable).

**Bootstrapping scheme (proposed):** start with a *vast* bank (≈ every instance), soft-attend
over it, use attention telemetry to surface archetypes; as training proceeds, newer instances
carry *better* features (encoder improved) and roll in alongside retained archetypes; total
size bounded, feature *quality* rises over time.

**Scale reframe:** at our corpus size (~262k windows × 512-d ≈ 0.5 GB; per-primitive ~2–3 GB)
VRAM is **not** the binding constraint — you could hold the whole corpus. Election is a
*quality* decision (clean retrieval neighborhood), not a survival one.

**Three landmines + fixes (this is a curated MoCo queue / DND, with MoCo's known pitfall):**
1. **Representation drift (the big one).** Stored embeddings go stale as the encoder trains;
   matching a v50 query against v5 vectors compares across encoder generations. Fix: store the
   **signal + cached embedding, re-encode (lazily)**, or a **momentum/slow encoder**, or
   periodic re-encode. *Storing primitives (more stable, cheaper to re-encode) mitigates this.*
2. **Attention-mass ≠ utility → head-class collapse.** Most-attended = most-frequent; rare
   classes starve (we already have 0/2-support classes), and generic "hub" entries get kept.
   Fix: **per-class/per-primitive coverage quotas** + a **discriminability-weighted utility**
   ("did retrieving you move the prediction toward truth"), + **diversity** (anti-redundancy).
   Decay telemetry (recent, not cumulative — cumulative mixes encoder eras).
3. **Full softmax is diffuse + slow.** Fix: **soft attention over a hard-retrieved top-k**
   (ANN → k → softmax over k). The *blend* stays differentiable (gradients to query, k values,
   encoder); only *which-k* is non-differentiable, which is fine — we learn an encoder that
   puts good neighbors nearby, not "which neighbor to pick" (kNN-LM / Neural Episodic Control).

### 4.1 Patch vs primitive (memory atom granularity) — build patch-level first
- **Patch-level atom:** pool the patch's features into one embedding + one clean label. Holistic
  retrieval ("your window looks like this walking window"). Simple → the MVP (≈ learned ConSE).
- **Primitive-level atom:** store/retrieve the *parts* (a primitive = a named feature component
  *within* a patch — a band energy, the eigen-ratio, cadence, …). Parts-based retrieval ("your window
  has a familiar vertical-rhythm part + arm-swing part") → **compositional** recognition of a
  never-seen activity as a novel *combination* of familiar parts. Richer, but a primitive has a
  *diffuse* label (a rhythm appears in many activities), so it needs the constellation-decoder (§6).
- **Mechanism for patch→primitive: learned subspace heads (multi-vector / ColBERT-style).** Project
  the patch vector into H sub-vectors (separate projections, not a hard dim-split); each is a primitive
  that retrieves from its own memory partition; the decoder aggregates per-head evidence. **CATCH:**
  splitting alone yields *entangled, redundant* heads (no free disentanglement) — and specialization is
  load-bearing (entangled heads make primitive-level == patch-level with extra steps). Force it via
  (a) cross-head **decorrelation** (VICReg/Barlow-style), (b) **independent per-head memory + contribution**
  (a head with no match → "aspect unfamiliar" → feeds novelty/abstain), and — the anchor —
  (c) **ground a subset of heads in the physical primitives** (supervise head_i to predict cadence /
  eigen-ratio / coherence). Mix **grounded heads** (interpretable — keep the analysis readable) +
  **free heads** (decorrelated — capture what we didn't hand-design).

### 4.2 Evidence head — concrete mechanism + training pipeline (the MVP, made real)
The "evidence engine" concretely = **a retrieval head over a labeled memory that emits per-candidate
EVIDENCE (not logits) and can ABSTAIN.** Reduces to ConSE as an ablation (fix τ, uniform weights, no
head, argmax). *This is more than a classifier precisely because a classifier is what the baselines are.*

**Forward pass** (query rep `z`, candidate labels `{c}` as text):
```
1. top-k neighbors of z in memory M={(z_i,label_i)} by a learned metric:  s_i = <g(z), g(z_i)>
2. per-candidate evidence (the language bridge does the "unseen label" work):
       e_c = Σ_i softmax(s/τ)_i · relu(<t(label_i), t(c)>)      # neighbor votes ∝ query-sim × label-text-sim
3. density gate ρ(z)=mean neighbor sim → ê_c = ρ·e_c            # DAEDL: far from memory → low evidence
4. Dirichlet head: α_c = ê_c + 1;  belief b_c = α_c/Σα;  total S = Σê_c;  uncertainty u = C/Σα
5. predict argmax b_c  if u < θ  else ABSTAIN
6. analysis = { e_c, b_c, top-k analogues (z_i,label_i,s_i), u }   # the "useful even when wrong" payload
```
Learned = `g, t, τ, density-gate, head` (SMALL); **memory `z_i` are FIXED (non-parametric)** → cheap to
train, no convergence nightmare (that risk lives entirely in Pipeline A).

**Training pipeline (Phase 2, Pipeline A FROZEN):**
```
STEP 0  encode train windows w/ frozen A → memory M (reps + label + config + subject)
STEP 1  episodic loop:
   • sample query q (true y, subject subj); M' = M − subj's entries      # subject-disjoint, no self-retrieval leak
   • with prob p_novel: ALSO drop class y from M'                        # SIMULATED NOVELTY (the abstain trick)
   • candidates = deploy vocab; forward evidence head → {e_c}, α, belief, u
   • loss: normal → evidential (Sensoy: grow evidence for y, KL non-y evidence → 0)
           novelty → abstain (drive total S low / uncertainty u high)
   • backprop g,t,τ,gate,head only (memory reps stop-grad)
STEP 2  validate on HELD-OUT configs/datasets: macro-F1 + OSCR/AUROC/FPR95 + ECE + analysis-consistency
```
**Key trick:** class-holdout is the *only* way to train abstention — you manufacture novelty so the model
learns "nothing here supports any candidate → abstain." MVP = frozen A + fixed memory (prototypes) + this
small head; full engine adds curated/evolving memory (§4), top-down hypothesis + structured evidence (§6),
online write-on-GT.

**Voting mechanism (precise):** a neighbor does NOT run a per-neighbor MLP over labels. It carries one
**text** label; the **shared** `t` kernel spreads that label's weight across *all* candidates by text
similarity (`relu⟨t(label_i), t(c)⟩`), scaled by the neighbor's query-similarity weight. Candidates are a
**runtime argument** = the target task's label list (train: ~86-way global vocab; eval: the dataset's small
set). Two distinct "open" axes: **open-vocabulary** (novel candidate *label* still scored via text sim —
the voting kernel) vs **open-set novelty** (no candidate supported → vacuity → abstain — a separate path).
Text geometry is **load-bearing**: transfer is only as good as `t` places candidate labels vs seen labels
→ the M6 lever if weak = fine-tune the `t` adapter and/or add the sensor↔text alignment term (A2 ablation).

**Abstain reads VACUITY, not entropy.** `u = C/Σα` measures *total* evidence, so it fires on "nothing
matched" (density-gated to ~0). It does NOT fire on *conflict* (two known classes both strongly supported →
belief looks split but `Σα` is high → confidently ambiguous → answers). Threshold the total (`u`), never the
belief entropy.

**Learnable vs frozen vs calibrated (the whole Phase-2 surface):**
- **Learned (small):** `g` (query/metric projection), `t`-**adapter only** (base text LM FROZEN), `τ`
  (retrieval temperature, clamped), the **density-gate calibration** (the density *signal* ρ=mean-sim is
  computed; a tiny monotonic map is learned), the evidential head.
- **Frozen / non-parametric:** the base text LM, **all of Pipeline A (the patch vectors)**, and the memory
  `z_i` (stop-grad). Gradient from the loss flows into `t`-adapter, `g`, `τ`, density-calib — and **stops at
  frozen A** (it does NOT reach the patch feature vector; that freeze is what keeps the memory drift-free).
- **Calibrated, not learned:** the abstain threshold **`θ` is set on VALIDATION** (an operating point, not a
  weight — learning it risks always/never-abstain collapse). Other HPs (`k`, `p_novel`, loss weights, the
  Dirichlet `+1` prior) are swept/structural. **No hand-tuned constant sits in the forward decision beyond
  `θ`.**

**Loss (two regimes, selected by class-holdout):**
```
KNOWN (true y in memory):   L = EDL_CE(α, y)              # grow evidence for y (Sensoy expected-CE)
                              + λ · KL( Dir(α̃) ‖ Dir(1) )   # α̃ = wrong-class evidence → uniform prior; λ annealed↑
NOVEL (y held out):         L = KL( Dir(α) ‖ Dir(1) )      # drive ALL evidence → vacuous ⇒ u→1 ⇒ abstain
```
The KL-to-uniform term does **double duty**: zeroes *wrong-class* evidence on knowns, *all* evidence on
novelties. **No separate confidence loss** — calibration comes from (i) the evidential form, (ii) the
density gate (the EDL-mirage fix), (iii) post-hoc ECE + `θ` on val; an explicit calibration penalty is an
M6 *ablation* if calibration is weak, not a default.

**Retrieval softness is one dial (`k`, `τ`, `τ`-anneal, full-soft@train / ANN@infer).** Selection (which-k)
is hard/non-diff *by design* — we don't learn a selection policy, we learn an encoder that places good
neighbors nearby (kNN-LM / NEC); the *weighting* (softmax over the k, gate, Dirichlet) is fully
differentiable. Our corpus fits in VRAM (~0.5 GB) so **full-soft attention at train time is available** for
exact gradients; default = generous-k soft + annealed `τ` (soft→sharp), ANN top-k at inference. Avoid
small-k + cold-`τ` from step 1 (sharpest = hardest to train).

**Sequencing (why NOT end-to-end from scratch):** train Pipeline A **first** (Phase 1), freeze, then B.
Joint-from-scratch has three failure modes — (1) **cold-start**: the retrieval/evidence loss is meaningless
until the space is already good; (2) **drift**: A moving vs a snapshot memory = matching across encoder
generations; (3) **collapse**: embedding-collapse (everything retrieves everything) + abstain-collapse
(cheapest novelty-loss minimum = always abstain). End-to-end is a legitimate *later* polish, but only
**warm-started from both trained phases + a momentum/slow memory encoder** — never from scratch.

## 5. "Useful even when wrong" — two testable properties (the foundation-model claim + its eval)
1. **Analysis-consistency (same-label ⇒ same-analysis).** The evidence/analysis object should
   cluster by *true label* even when top-1 is wrong. A metric-learning objective *on the
   analysis itself*, and a **novel evaluation** decoupled from accuracy (clustering/retrieval
   quality of the analysis vector).
2. **GT-coherence / the "aha" + online write.** When the true label is revealed it should be
   **evidence-consistent** (high evidence-rank, not orthogonal) — measured as GT evidence-rank /
   "surprise." The online version *is* the emergence: on seeing GT, **write (primitives, label)
   into the bank**; the "now it makes sense" is real iff that write (a) reduces future error on
   similar samples and (b) slots in coherently. Same machinery as deployment-time learning.
   (Caveat: "aha" as a feeling is not a metric; GT evidence-rank + online improvement are.)

### 5.1 HAR world model: latent forward-dynamics (cheap add; unifies representation + abstention)
Supervisor suggestion — a "HAR world model." Scope it honestly as a **latent forward-dynamics /
next-latent-state predictor** (predict future patch latents from the past, JEPA-style, in embedding
space), **NOT** an action-conditioned RL world model (HAR has no agent-actions to condition on; skip
rollouts / long-horizon generation / planning — not needed).
- **Cheap:** it's the research-recommended JEPA latent-masked objective (CHARM) extended along the
  TEMPORAL axis — same masked-prediction SSL machinery with a causal/temporal mask instead of a channel
  mask, reusing the causal physical-time (RoPE) encoder we already plan/port. Just a predictor head + a
  mask schedule added to Pipeline A's Phase-1 pretraining.
- **Earns its place (not a bolt-on):** the **prediction error = surprise = novelty signal.** A model
  that predicts motion dynamics is *surprised* when reality deviates (an activity it has no dynamics for)
  → that surprise feeds Pipeline B's abstention/density gate directly (stacks with the density-aware
  Dirichlet). So the world model is the *source* of "I don't know": learn dynamics → deviation → surprise
  → abstain + flag novelty. Unifies the representation objective with open-set, and is a **fresher framing**
  than the saturated label-interface (do a quick "sensor/IMU world model" novelty-check before leaning on it).

### 5.2 Training objectives — cut to the ELITE 3 (discipline over a menu)
Risk: a pile of overlapping fancy objectives = a convergence project that yields a *compromise*
representation that then underperforms downstream. So **Pipeline A Phase-1 = exactly 3**:
1. **Masked spatio-temporal latent prediction** (JEPA/CHARM: mask *channels AND time*, predict in latent
   space) — the generative workhorse; **folds in** the masked-channel relational model *and* the world
   model (masking the future = the dynamics/surprise objective). One objective, two axes.
2. **Config-conditional supervised contrastive** — the discriminative workhorse; gives discrimination +
   the heterogeneity-robustness thesis + the retrieval metric structure Pipeline B needs.
3. **Physical-primitive grounding** (predict cadence / eigen-ratio from the rep) — cheap; keeps features
   interpretable + physically anchored; = the "grounded heads" idea.
**Why trust it:** #1+#2 is CrossHAR's *proven* masked+contrastive recipe (we watched it converge and give
usable HAR features — 35.7 mean). **DEFERRED to ablations** (where the convergence risk lives, so they
don't gate v1): the **equivariance / augmentation-response operator** (Siamese/BYOL-predictor of Δ *under*
the transform = equivariance not invariance), **sparse feature-space reconstruction** (bank-as-dictionary,
sparse code = evidence; feature-space because invariance kills raw reconstruction), and analysis-consistency
as a *separate* loss (overlaps the contrastive). **Augmentations:** the one non-redundant geometric aug is
the **physically-correct time-warp** (`accel×1/α², gyro×1/α`) — rotation/translation/uniform-scale commute
with `d²/dt²`, so the integrate→transform→differentiate detour reduces them to the SO(3)/gain augs we have.
The implemented stack (`data/scripts/augmentations.py`) already provides the nuisance/heterogeneity axes:
`RateCfg` (P3, anti-aliased resample → rate-invariance, co-varies the Hz channel-text), `TimeWarpCfg`
(cadence), `ChannelDropoutCfg` (P4, whole-group drop → masked-channel objective), rotation/gravity/gain.

#### 5.2.1 A1 — masked spatio-temporal latent prediction (concrete spec)
Data shapes (harmonised corpus): **6 s window · 60 Hz · 360 samples · 6 canonical channels** `[acc xyz, gyro
xyz]` (pad+mask for accel-only sources). Tokenizer patch is defined in **seconds**, so the token grid is
`T × C` where `T = window_s / patch_s`.
- **Patch-duration is a multi-scale augmentation axis, NOT a fixed HP.** The current `patch_seconds=1.5`
  gives only `T=4` — too coarse for a temporal/world-model mask. Instead **sample `patch_seconds` per
  batch** from a range (`{0.5, 0.75, 1.0, 1.5}` s → `T∈{12,8,6,4}`; 2.0 s/T=3 dropped after the
  objective-health audit — too coarse to mask, and it collapses native-short windows to one
  un-maskable patch). Multi-resolution robustness falls
  out, and the short-patch batches are where rich temporal masking happens.
- **Rate ≠ patch count (precise distinction).** Resampling changes *samples-per-patch* but NOT `T`
  (patches are in seconds), so `RateCfg` stresses the filterbank's **physical-Hz invariance**; it does
  *not* give more tokens to mask. The token-count knob is `patch_seconds`. Keep the two axes separate.
- **Masking = a RATIO over the resulting variable grid** (not a fixed token count), with a **floor on
  `T`** so a long-patch draw can't make the temporal mask degenerate. Two structured streams:
  (a) **channel/modality mask** — whole channels, **biased toward dropping the gyro triplet** at ~the
  corpus accel-only fraction (matches the real deployment shift) + occasional single-channel drops;
  (b) **temporal mask** — MAE/JEPA block mask (**random-block** variant → representation; **causal/future**
  variant → the §5.1 world-model / surprise objective). Prediction is in **latent space**, so high ratios
  are fine — start ~50 % masked, ablate up.
- **Attention:** `T·C ≈ 60` tokens → **full self-attention over the flattened grid** (no need for axial).
- **Positional encoding:** **time = RoPE keyed to physical Δt (seconds), never patch index** — this is what
  absorbs both variable rate *and* variable patch duration with zero special-casing. **Channels = a
  text-keyed SET (no positional index)** — identity comes from the channel's free-form-text content
  embedding + modality tag, which is what makes channel count/order free; this **replaces** the crude 2-way
  `modality_embedding`/`[0,0,0,1,1,1]` buffer in the current experiment code.
- **Batch constraint to design around:** the tokenizer requires a **single patch-length (in samples) per
  batch** (`patch_length = round(rate × patch_seconds)`), so rate and patch-duration **cannot both vary
  freely per-sample** — **bucket batches by `(rate, patch_seconds)`** (or pad+mask patches to a common
  grid). The current experiment runs the fixed harmonised 60 Hz grid precisely to sidestep this, so the
  minimal experiment ↔ full multi-rate/multi-scale design has a **gap not yet closed**.
- **Build discipline (M2):** pick the ranges, then **draw a batch of augmented samples and visually inspect**
  that the rate/patch-duration/channel-drop ranges are sane *before* committing; wire the bucketed sampler.

#### 5.2.2 A2 — config-conditional supervised contrastive (concrete spec)
- **Sensor↔sensor SupCon at the WINDOW (pooled) level** (labels are per-window; A1 is the patch-level loss).
- **Config-conditional:** same-activity windows from *different placements* are **positives** (nuisance
  invariance), but the placement/sensor config is fed as **input text** so the model factors placement out
  *because it's told*, not blindly (the UniMTS failure mode).
- **CLIP-style sensor↔text alignment is NOT required for the MVP.** In the evidence head, retrieval is
  sensor→sensor (`⟨g(z), g(z_i)⟩`) and unseen-label transfer is **text→text** (`⟨t(label_i), t(c)⟩`), so
  the two spaces never have to be jointly aligned — A2 only needs the sensor space to be a good
  same-activity metric, which SupCon delivers. A sensor↔label-text term is an **optional ablation**, added
  only if M6 shows unseen-label transfer is weak; keeping it out keeps the retrieval geometry SupCon-owned.

#### 5.2.3 A3 — physical-primitive grounding (concrete spec)
Auxiliary **regression** of DSP-computed physical targets from the representation (MaskFeat-style
feature-prediction; see provenance below). A cheap **grounding rail**, low loss weight, near-zero
convergence risk.
- **Primitives (v1 core):** **cadence** (dominant locomotion rate, Hz; window-level; predicted in
  **log-Hz**) + **per-band eigen-ratios** — eigenvalues `λ₁≥λ₂≥λ₃` of the per-band 3-axis accel spatial
  covariance → **linearity `(λ₁−λ₂)/λ₁`, planarity `(λ₂−λ₃)/λ₁`, isotropy `λ₃/λ₁`** (per band/patch).
  Easy adds: accel↔gyro **coherence**, relative **spectral shape**. DC/gravity-tilt is already an explicit
  tokenizer feature → predict a derived **tilt angle** or drop (avoid redundancy).
- **Why these:** eigenvalues are **rotation-invariant** (`C→RCRᵀ` leaves them fixed); the **ratios cancel
  gain** (`C→g²C`); cadence-in-Hz is rotation- and gain-invariant. i.e. invariant to exactly the nuisances
  we kill, while still describing motion geometry — the "salient-not-invariant" thesis in one feature.
- **Objective:** per-primitive Huber/MSE (simplex-aware for the eigen-ratio triple), **weighted small**;
  levels: cadence ← pooled window, eigen-ratios/coherence ← per band/patch.
- **Augmentation-aware target rule (REQUIRED — corrects the earlier "cache once offline"):** compute the
  target on the **exact augmented view the encoder consumes** — never predict a clean-signal primitive from
  an augmented input. Because aug params are known, **transform the cached target analytically**
  (time-warp α → `cadence ×= 1/α`; rotation/gain/rate → invariant → nothing to do) and only recompute for
  the messy augs (magnitude-warp, jitter). This makes nuisance augs act as **free invariance regularizers**
  and physical augs (time-warp) teach **genuine sensitivity**.
- **Validity mask, re-derived on the augmented view (per primitive, per sample):** channel-drop → coherence
  **undefined → mask**; sub-Nyquist downsample or jitter-killed periodicity → cadence **mask** (else the
  head learns to hallucinate a step rate on stationary activities). **M0-confirmed:** the cadence mask
  needs a **motion-energy floor** (std |acc| ≥ ~0.03 g) *in addition to* autocorr strength — static windows
  autocorrelate on drift and fabricate cadences otherwise. Also M0-confirmed: raw autocorrelation has
  **octave ambiguity** (walking locks to stride, running to step) — usable as a grounding target, but the
  estimator should be made octave-aware at M1 (the research-pass "adaptive cadence estimator" risk).
- **Selection criterion (this is the real answer to "aug-robust"):** admit a primitive **only if its
  response to *every* augmentation is invariant OR analytically known.** Eigen-ratios & cadence pass;
  coherence passes with a channel-drop mask; anything that changes non-analytically is disqualified as a
  target. Heterogeneity-robust *by construction*.
- **Verify before committing:** **plot** eigen-ratios & cadence across activities (walk/run/sit/stairs) and
  confirm they separate; if not, the bands are too narrow — widen them.
- **Provenance / novelty (honest — this is the deliberately un-novel rail):** the *method* = **MaskFeat**
  (Wei et al., CVPR 2022 — mask & predict a hand-crafted feature, HOG there, physical primitives here); the
  *quantities* are textbook — eigen-ratio linearity/planarity/sphericity are **structure-tensor / DTI shape
  descriptors** (Westin-style; Demantké-style point-cloud dimensionality), cadence = classic
  autocorrelation gait estimation, coherence = Welch cross-spectrum, orientation/gain-invariant
  covariance features are standard HAR. **No single ingredient is novel** (consistent with the research
  verdict) — novelty budget stays on the conjunction (config-conditional evidential memory + abstain).
  **TODO before citing:** log in `references/` and run a verification pass to pin canonical citations. See
  [[halo-references-folder]].

## 6. Structured positive/negative, located evidence (the frontier — high ceiling, high risk)
"A,B,C present at locations A,B,C **and** D,E,F absent ⇒ Y" — a structured energy / factor
graph over *located* features with **inhibitory** (negative) terms. Negative evidence is
underused in HAR and is what gives clean mutual-exclusion + abstention. Fork: let it **emerge**
from attention with signed weights + a coherence penalty (buildable, less controllable) vs make
it **explicit** (neuro-symbolic; expressive but combinatorial, easy to become a hand-wavy graph
that won't train). **Prototype the emergent version first.**

## 7. Tokenizer direction (shared substrate — brainstorm)
Current: physical-Hz **filterbank** (per-channel spectral, rate-invariant) + signed DC/gravity.
Good, but for *fine-grained, heterogeneity-salient primitives within one patch* it's per-channel,
amplitude-absolute, single-resolution, and monolithic. Candidate extensions, grouped by the
robustness they buy (each must be **ablatable**; physics-grounded hand-crafting is *on-thesis*
because it's more heterogeneity-robust than learned filters):

- **A. Cross-channel relational (biggest salience win — "patterns formed by channels"):**
  per-band **coherence/phase** between channels (amplitude-normalized → gain-robust; encodes
  movement plane/direction); inter-channel **covariance/correlation** structure;
  **orientation-normalized projection** (within-patch PCA of the accel triad → dominant motion
  plane → project → mounting-angle/placement-robust axes = *earned*, config-conditional invariance).
- **B. Amplitude-relative / scale-free (gain & g-vs-m/s² robust):** **relative band energies**
  (band/total); spectral **shape** descriptors (centroid, bandwidth, flatness, roll-off = motion
  "timbre"). Keep absolute DC/gravity too (both views available).
- **C. Multi-resolution / transients:** **wavelet** or multi-window STFT (footfall transient +
  gait rhythm together); **jerk** (d/dt accel) spectral energy (movement sharpness).
- **D. Periodicity / cadence (very discriminative + sensor-robust):** autocorrelation fundamental
  period + harmonic ratios (steps/sec, physical Hz); periodicity strength (regular vs irregular).
- **E. Emit STRUCTURED primitives, not a monolithic patch vector** — a named set the memory
  stores and the decoder attends over → compositional retrieval.
- Guardrails: prune redundancy (coherence/cov/corr overlap — pick minimal); optional small
  **gated learned residual** on top of the physical bank (adapt without losing robustness);
  optional **config-conditional normalization** via channel text (powerful, couples tokenizer
  to language — keep optional).

## 8. Open design forks (to resolve before building)
1. **The load-bearing loss:** write the "pull/push *conditioned on config* + analysis-consistency"
   objective cleanly. If it's coherent, §2–§6 are engineering around it.
2. **Utility signal for the memory:** task-supervised ("retrieval helped predict truth") vs
   self-consistency ("retrieval pulled same-label together"). Lean self-consistency-weighted blend
   (more foundation-model), but this is the real fork.
3. **Structured evidence:** emergent-inhibition (first) vs explicit neuro-symbolic (stretch).
4. **Memory freshness:** re-encode-signal vs momentum-encoder vs periodic-refresh.

## 9. Code organization (keep separate from conventional training)
```
model/tokenizer/     SHARED — port the legacy PhysicalFilterbankTokenizer + the §7 extensions.
model/evidence/      NEW — primitive encoder, archetypal memory, evidence decoder.
training/evidence/    NEW — the retrieval/evidence training harness (its own loop, NOT a
                     conventional classifier trainer). See training/evidence/README.md.
docs/design/EVIDENCE_ENGINE.md   this file (design/rationale).
docs/design/EVIDENCE_ENGINE_BUILD_PLAN.md   the milestoned build plan (how + order + gates).
```
The tokenizer is the only shared component; the mechanism + harness are self-contained so this
line never conflates with a standard softmax/cosine training path.

## 10. Novelty triangulation (honest)
Individually known: Matching/Prototypical Nets, kNN-LM/RETRO, Evidential Deep Learning, OpenMax
(open-set), ConSE, MoCo (memory queue). **The claim is the combination tied to free-form language
conditioning + heterogeneity-salient physical primitives + the analysis-artifact/consistency
evaluation (usefulness ≠ accuracy).** Each piece must be ablated; the one-sentence hook is
*config-conditional salient features feeding a curated-memory evidence engine that produces a
consistent, abstaining analysis — turning HAR from closed classification into an analysis substrate.*
