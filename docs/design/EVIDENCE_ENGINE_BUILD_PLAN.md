# Evidence Engine — Build Plan

> Implementation roadmap for the design in [`EVIDENCE_ENGINE.md`](./EVIDENCE_ENGINE.md).
> This doc is the **how + order + gates**, not the rationale. Status: proposed; not started.
> `model/` and `training/` are empty in this repo — this is greenfield, but the **legacy tree
> (`/home/alex/code/HALO/legacy_code`) has reusable pieces** to port (noted per milestone).

## Guiding principles
1. **Cheap, decisive experiments before big builds.** Every design fork gets an empirical gate
   *before* we commit code to it. Numbers over whiteboard.
2. **Decision gates, not a waterfall.** Each milestone ends with a go/no-go criterion. If a gate
   fails, we stop and re-decide rather than building on sand.
3. **Config-conditional everywhere.** Channel *text* identifies every channel; nothing is keyed
   by position or count. This is what buys variable-channel + heterogeneity robustness.
4. **Everything ablatable.** Each added primitive / learnable component ships behind a flag so we
   can prove it earns its place on a held-out **config** (not just held-out labels).
5. **MVP spine first, frontier later** (see last section).

## Two pipelines (top-level decomposition)
The build is **two pipelines** with a clean seam between them; they can be developed and tested
largely independently.

**Pipeline A — Representation ("the tokenizer", broadly).** *Everything that turns raw signal into a
heterogeneity-salient representation:* time-domain **preprocessing** (gravity-align), **cross-channel
relational/causal learning** (masked-channel set model → residuals), the **learnable frequency
domain** (fixed physical filterbank + constrained-learnable scattering/SincNet), and the
**primitives**. This pipeline is itself *learnable and trained* — not a fixed transform.

**Pipeline B — Evidence (memory + prediction).** Maintain the curated non-parametric **archetypal
memory**, treat retrieved + hypothesis signals as **evidence**, and **predict** an analysis
(evidential, abstaining).

**Milestone → pipeline:** A = M0–M3 · B = M4–M5 · M6 (eval) spans both · M7 is B.

**The seam (pin it early):** Pipeline A emits, per patch, `{query representation vector(s) + the
structured primitives + each channel's text id}`; Pipeline B consumes that + a memory of past
A-representations & labels → analysis. Fix this contract first so each side can be built against a
stub of the other.

**Training regime — two phases, matching the split:** **Phase 1** pretrains Pipeline A on the
**ELITE 3** objectives (see EVIDENCE_ENGINE.md §5.2): (1) masked spatio-temporal latent prediction
[folds in masked-channel + the "HAR world model"], (2) config-conditional supervised contrastive,
(3) physical-primitive grounding — deliberately NOT the full menu (equivariance-operator /
sparse-reconstruction = deferred ablations). **Phase 2** trains the small **evidence head** on frozen
A via episodic retrieval + class-holdout abstention (EVIDENCE_ENGINE.md §4.2). Phase 1 is
validated by the robustness probe + consistency metrics *before B exists*. **Phase 2** builds
Pipeline B on top (A frozen initially; optional joint fine-tune later). De-risks: prove the
representation is good before spending effort on the evidence machinery.

## File layout (grouped by pipeline)
```
PIPELINE A — Representation ("the tokenizer": learnable + trained)
  model/tokenizer/    preprocess.py (gravity-align) · filterbank.py · scattering.py
                      primitives.py · channel_set.py (masked-channel set model) · encoder.py
  training/tokenizer/ losses_repr.py (masked-channel SSL + salient-contrastive + consistency)
                      pretrain.py · probe_robustness.py (M0) · README.md
PIPELINE B — Evidence (memory + prediction)
  model/evidence/     memory.py · decoder.py · config.py
  training/evidence/  train.py (evidence/decoder training on A's representations) · README.md
SHARED / SEAM
  eval/               evidence_adapter.py · analysis_metrics.py
  docs/design/        EVIDENCE_ENGINE.md · EVIDENCE_ENGINE_BUILD_PLAN.md (this)
```

---

## M0 — Robustness probe (decide the front end empirically)  ·  ~0.5 day
**Goal:** adjudicate which preprocessing/primitives/transforms actually deliver invariance to the
nuisances we care about, before designing the tokenizer around guesses.
**Build:** `training/tokenizer/probe_robustness.py` — load real grids; apply synthetic **gain**,
**SO(3) rotation**, **yaw-only rotation**, and mild **time-warp/resample**; compute candidate
features **before vs after** and report an invariance score (e.g. cosine / relative-L2 drift) per
feature family:
- raw per-channel band energy (baseline — expected fragile)
- gravity-aligned per-channel band energy
- per-band accel-triad **eigenvalue ratios** (expected rotation+gain invariant)
- accel↔gyro **coherence**, **relative spectral shape**, **cadence**
- (if quick) a fixed **scattering** first-order coefficient vs a plain STFT bin under time-warp.
**Gate:** a ranked table of "invariance vs discriminability" per feature. **Keep only families that
are both invariant to the nuisance AND still separate activities.** This fixes the M1 primitive set,
**and** doubles as the A3-target check (§5.2.3): **plot eigen-ratios & cadence across activities
(walk/run/sit/stairs) and confirm they visibly separate** before admitting them as grounding targets —
if not, widen the bands. Also record each candidate's aug-response so only **invariant-or-analytic**
primitives (the A3 selection criterion) survive.
**Resolves:** §7 tokenizer forks; the scattering-vs-STFT deformation question (partly).

**RESULT (2026-07-17 — gate PASSED; `training/tokenizer/outputs/m0_probe/REPORT.md`, 1040 windows,
4 streams pocket/wrist/waist/pocket):**
- **The thesis in numbers:** `raw_band_energy` wins within-dataset (0.855 kNN-BA) but the fully
  rotation-invariant `grav_band_energy` (gravity-aligned, vertical + horizontal-pooled = yaw-killed)
  wins cross-dataset (**0.502 vs 0.461**) — raw's edge is config-specific shortcut info that does not
  transfer. Invariance verified: raw drifts under SO(3)/yaw (0.956/0.982 cos); grav/eigen/coherence/
  shape/cadence are exactly invariant to gain+rotation (1.000/0.000).
- **Primitive set fixed for M1:** gravity-aligned band energy (vert/horiz-pool) · eigen-ratios ·
  spectral shape · cadence · coherence (+ DC/tilt from the existing tokenizer). `raw_band_energy`
  EXCLUDED (fragile as predicted; keep only as ablation baseline).
- **Per-dim efficiency:** cadence = 0.394 xds-BA from 2 dims; eigen-ratios weak alone (0.212) but
  visually separate jog/run crisply (linearity ~0.95 tight) → grounding targets, not standalone.
- **Naive concat does NOT fuse** (`invariant_union` 0.461 ≤ grav alone 0.502 — weak families dilute
  kNN distance) → **fusion is the learned encoder's job; M2/M3 empirically justified.**
- **Two A3 caveats confirmed empirically:** (1) cadence validity needs a **motion-energy floor**
  (std |acc| ≥ 0.03 g), not just autocorr strength — static windows otherwise fabricate cadences;
  (2) **octave ambiguity** (walking locks to stride ~0.95 Hz, running to step ~2.5 Hz) — the
  research-pass "adaptive cadence estimator" risk is real; grounding target OK, refine estimator at M1.
- Corpus accel is unit-canonicalized to **g** (|gravity|≈1.0) — thresholds must be in g-units.

## M1 — Front end: preprocessing + primitive tokenizer  ·  ~3–4 days
**Goal:** a `signal → structured primitives` front end that is rate-, gain-, orientation-robust and
channel-count-agnostic.
**Build:**
- `model/tokenizer/preprocess.py` — **gravity-align** (low-pass → gravity vector → rotate so "down"
  is canonical; **rotate accel AND gyro by the same R** — joint, physical). Flag: on/off. Document
  it's a *partial* frame (yaw-free) — complemented by the invariant primitives.
- `model/tokenizer/filterbank.py` — **port** the legacy `PhysicalFilterbankTokenizer` (physical-Hz,
  rate-invariant) + signed **DC/gravity** feature. This is the fixed backbone.
- `model/tokenizer/scattering.py` — optional **scattering / SincNet** frontend behind a flag
  (`fixed | sincnet | scattering | free-conv`), so M0/later ablations can compare. Default = fixed
  until an ablation earns the switch.
- `model/tokenizer/primitives.py` — the **structured** primitive set that survived M0, computed over
  **text-identified channel groups** (accel triad, etc.), degrading gracefully when channels are
  missing: per-channel band energy, per-band **eigen-ratios**, **coherence**, **cadence**, **relative
  shape**, **DC**. Output is a *named dict of primitives*, not a monolithic vector.
**Gate:** unit tests asserting the M0 invariances hold on real data (rotation/gain drift below
threshold), missing-channel paths don't crash and mask correctly, and rate-invariance across
20/50/100 Hz resamples. **Port + tests green.**
**Reuse:** legacy filterbank, DC feature, SO(3) rotation util (for tests).

**RESULT (2026-07-17 — gate PASSED, 139/139 tests green):** built `model/tokenizer/`
{`filterbank.py` (verbatim port + its 21-test legacy suite → `tests/test_filterbank.py`),
`preprocess.py` (patching port + NEW joint gravity-align, g-unit thresholds, yaw-free documented),
`primitives.py` (M0-surviving set as named `Primitive(values, valid)` dict; octave-aware cadence
with motion floor; text-identified triads; graceful degradation), `scattering.py` (frontend flag:
fixed | sincnet implemented; scattering deferred; free_conv deliberately refused)}. M1 gate tests in
`tests/test_tokenizer_m1.py` (23): real-data rotation invariance (cos > 0.99 on motionsense walking),
gain/rotation invariance per family, 20/50/100 Hz rate-invariance (shared-observable-band energies
cos > 0.999; full tokens only compared where all bands observable — Nyquist-mask features
*deliberately* differ across rates), missing-channel masking, cadence motion-floor + octave rule
(recovers 2 Hz step from stride-dominant autocorr), eigen-ratios on the simplex. Runtime env:
`legacy_code/.venv/bin/python` (torch 2.9). NOTE: cadence in synthetic tests must bounce ALONG
gravity — perpendicular components enter |acc| only at second order (physics, not a bug).

## M2 — The load-bearing loss (does the theory cohere?)  ·  ~2–3 days · **HARD GATE**
**Goal:** write and sanity-check the objective *before* the full model, because if it doesn't cohere,
nothing downstream matters. Concrete per-objective spec lives in **EVIDENCE_ENGINE.md §5.2.1–5.2.3**;
this milestone implements the **ELITE 3** (A1/A2/A3) as `training/tokenizer/` + `training/evidence/`
losses.
**Build:** `training/evidence/losses.py`:
- **A1 — masked spatio-temporal latent prediction** (§5.2.1): token grid `T×C`, `T=window_s/patch_s`.
  **`patch_seconds` is a per-batch multi-scale augmentation axis** (~`{0.5,0.75,1.0,1.5,2.0}`s → `T≈3–12`),
  **not** a fixed HP — and note **rate resample ≠ patch count** (rate stresses filterbank Hz-invariance;
  `patch_seconds` is the token-count knob). Mask = **ratio** over the variable grid (floor `T`), two
  structured streams: whole-**channel** mask biased to dropping the **gyro triplet** + **temporal** block
  mask (random-block + causal/future = the world-model variant). Latent-space target → ~50 % start.
  **Batch-aware:** the tokenizer needs one patch-length/batch → **bucket batches by `(rate, patch_seconds)`**
  (or pad+mask to a common grid). **Before committing ranges: draw a batch of augmented samples and
  visually inspect** rate/patch-duration/channel-drop are sane.
- **A2 — config-conditional supervised contrastive** (§5.2.2): **window-level sensor↔sensor SupCon**,
  same-activity-across-configs pull together, **conditioned on the channel-text** (not blind-invariant).
  **No CLIP-style sensor↔text term** (label transfer is text→text in the evidence head) — that's an
  optional ablation only if M6 shows weak unseen-label transfer.
- **A3 — physical-primitive grounding** (§5.2.3): small MLP head regresses **cadence (log-Hz, window)** +
  **per-band eigen-ratios (linearity/planarity/isotropy, per band/patch)**, weighted small. **Targets
  computed on the augmented view** (analytic transform where known: time-warp → `cadence×1/α`; rotation/
  gain/rate invariant), **validity-masked** (channel-drop→coherence, sub-Nyquist→cadence). Admit a
  primitive **only if aug-response is invariant or analytic** (selection criterion).
- **Analysis-consistency:** deferred to an ablation (overlaps A2; see §5.2 deferred list).
- **Evidential/abstention** term (Dirichlet evidence; detail in M5/§4.2).
**Positional encoding (shared, decided):** **time = RoPE physical-Δt** (absorbs variable rate *and* patch
duration); **channels = text-keyed set, no positional index** (replaces the crude 2-way modality embedding).
Full self-attention over the flattened `T·C≈60` tokens (no axial needed).
**Gate (HARD):** on a *tiny* real+synthetic setup with a frozen M1 front end + a trivial encoder,
show the salient-contrastive loss (a) trains, (b) pulls same-activity/different-config together and
different-activity apart on held-out configs, and (c) A3 primitive heads learn without dominating.
**If the loss is incoherent or collapses, STOP and redesign — do not build M3+.**
**Resolves:** open fork #1 (the load-bearing loss) and #2 (task- vs self-consistency utility — decide
the blend here).

**RESULT (2026-07-18 — HARD GATE PASSED on run 3; `training/tokenizer/outputs/m2_gate/`):**
Tiny CPU setup (frozen M1 front end + 2-layer trivial encoder, M0 4-stream data, pamap2/wrist held
out entirely = hardest config transfer). **Held-out-config kNN-BA 0.366 vs 0.284 handcrafted
grav_band_energy on the same split (+29% rel), monotone 0.178→0.366 over 600 steps and still
rising; all 3 losses train (a1 2.08→0.07, supcon 4.05→2.89, a3 0.27→0.03); no collapse (eff-rank
13.6, growing).** Losses live in `training/tokenizer/losses_repr.py` (unit-tested; tests caught a
0×-inf SupCon NaN + a non-complementary mask coin). The gate itself needed 3 runs — each FAIL was a
HARNESS flaw, each a real lesson for Phase-1:
1. **Contrastive must see the CLEAN view** (two forwards: masked for A1, clean for A2/A3) — SupCon
   computed from the masked forward fights mask noise.
2. **Config-dropout (p≈0.2) trains the UNKNOWN-config token** — deployment IS an unseen config; an
   untrained fallback token distorts held-out eval. Keep this in the real system.
3. **Gravity-align is part of the front end, ALWAYS** (run 2→3: 0.238→0.366) — skipping the
   canonicalization asks the encoder to relearn M0's winning transform from scratch.
Also from the M2 visual inspection: time_warp used an unconstrained cubic → clip SATURATION
edge-held a dead flat tail (~25% of window) and could locally reverse time — fixed with monotone
PCHIP in `data/scripts/augmentations.py`.
M2 simplifications to resolve at M3/Phase-1: per-stream config token (→ real channel-text),
fixed patch_seconds=1.0 (→ multi-scale bucketed sampler), SO(3)+gain only (→ full aug stack).

## M3 — Config-conditional set encoder (channels as a text-keyed set)  ·  ~4–5 days
**Goal:** an encoder over the primitive set that is **permutation- and count-invariant** and
**physical-time-aware** — solving variable channels by construction.
**Build:**
- `model/evidence/channel_set.py` — each channel = token{primitives + text embedding of its
  description}. **Masked-channel modeling**: randomly drop channels, predict them from the rest via
  cross-channel attention (this is Idea 3; it *also* trains variable-channel robustness + yields the
  relational **residual** feature).
- `model/evidence/encoder.py` — set-transformer over channel tokens + **physical-time** positional
  encoding (**port** legacy RoPE-physical-time encoder), producing per-patch query features. No LSTM
  (fights variable channels); time is physical, not index.
**Gate:** trained with the M2 SSL pretext, show (a) masked-channel prediction works and residuals are
informative, (b) the encoder consumes 3/6/9/12-channel inputs unchanged, (c) held-out-config
same-activity clustering improves vs the M1 primitives alone.
**Reuse:** legacy RoPE-physical-time encoder; channel-text/`ChannelTextFusion` idea.

## M4 — Archetypal memory (retrieval + curation)  ·  ~4–5 days
**Goal:** the curated non-parametric evidence store.
**Build:** `model/evidence/memory.py`, staged to de-risk:
- **M4a:** frozen bank of encoded training primitives; retrieval as **one softness dial**
  (`k, τ, τ-anneal, full-soft@train / ANN@infer`) — differentiable blend over the k, which-k non-diff by
  design (we learn placement, not a selection policy; corpus fits VRAM so full-soft@train is available).
  Voting = the **shared `t` text kernel** (not a per-neighbor MLP); candidates = runtime label list.
  Prediction = confidence-weighted labels of neighbors (a learned ConSE). Establishes the retrieval baseline.
- **M4b:** **drift management** — store signal + cached embedding, lazy re-encode (or momentum
  encoder). Verify retrieval quality doesn't degrade across training epochs.
- **M4c:** **curation gate** — bounded bank with **discriminability + coverage** utility (per-class
  quotas so rare classes survive; anti-redundancy), telemetry **decayed/recent**, soft throughout.
**Gate:** retrieval-only zero-shot beats the ConSE baselines on held-out configs; ablate M4a→c to show
drift-mgmt and curation each help; confirm rare classes retained.
**Resolves:** memory-freshness fork (#4).

## M5 — Evidence decoder + evidential/abstention head  ·  ~3–4 days
**Goal:** turn retrieved + hypothesis evidence into the **analysis** (not just argmax).
**Build:** `model/evidence/decoder.py` — cross-attention over {retrieved neighbors + candidate text
hypotheses}, **accumulating across patches**; an **evidential (Dirichlet)** head → per-candidate
{evidence-for, evidence-against, calibrated confidence, **abstain**}. Output = the structured analysis.
**Loss (EVIDENCE_ENGINE.md §4.2, two regimes via class-holdout):** KNOWN → `EDL_CE(α,y) + λ·KL(wrong→Dir(1))`
(λ annealed↑); NOVEL → `KL(Dir(α)‖Dir(1))` (drive vacuous → abstain). No separate confidence loss
(calibration from EDL + density gate + post-hoc). Learnable surface = `{g, t-adapter, τ, density-calib,
head}`; base LM + Pipeline A + memory `z_i` frozen/stop-grad (gradient stops at A → drift-free).
**Abstain `θ` calibrated on VAL, not learned.**
**Gate:** calibration (ECE — are confidences honest?), abstention reads **vacuity** not entropy (low total
evidence → abstain; conflict between knowns → still answers), per-candidate evidence non-degenerate.
Zero-shot macro-F1 ≥ M4 (decoder shouldn't hurt).

## M6 — Analysis eval + HALO adapter into the existing harness  ·  ~2–3 days
**Goal:** measure the foundation-model claim and slot into the current eval.
**Build:**
- `eval/analysis_metrics.py` — **analysis-consistency** (same-label⇒same-analysis clustering),
  **GT evidence-rank** ("aha" coherence), **OSCR/AUROC/FPR95** (open-set), alongside macro-F1.
- `eval/evidence_adapter.py` — register HALO-as-evidence in the existing baseline eval harness
  (subject-disjoint, humanized labels, provenance-stamped) so it sits in the same table.
**Gate:** full run on the 8 held-out datasets; the analysis-consistency + OSCR numbers exist and are
sane; results carry provenance.

## M7 — Structured positive/negative evidence (FRONTIER — only if M1–M6 hold)  ·  research
**Goal:** located, inhibitory co-occurrence ("A,B present & D,E absent ⇒ Y").
**Build:** start with **emergent** signed-attention + a coherence penalty in the decoder (buildable);
treat explicit neuro-symbolic structure as a stretch.
**Gate:** does negative evidence measurably improve mutual-exclusion / abstention over M5? If not,
shelve — do not force explicit structure.
**Resolves:** fork #3.

---

## MVP spine vs full vision
- **MVP spine (the paper):** M0 → M1 → **M2 (hard gate)** → M3 → **M4a/b** → **M6**. That is:
  config-conditional salient primitives + text-keyed set encoder + retrieval evidence + the
  analysis-consistency & open-set eval. If this beats the ConSE/cosine baselines on cross-config
  zero-shot **and** shows a consistent, abstaining analysis, that's a submission on its own.
- **Full vision (extends the paper):** M4c curation, M5 evidential decoder, M7 structured evidence,
  and online/deployment learning (write-on-GT) — each is an ablation/section, not a prerequisite.

## Risks & kill criteria (honest)
- **M2 loss incoherent / collapses** → the whole thesis is in question; stop and redesign. This is the
  single most important gate.
- **Salient primitives don't beat raw features on held-out config (M1/M3)** → the "salient not
  invariant" claim isn't earning its keep; fall back to a simpler config-conditional encoder.
- **Memory drift unfixable / curation collapses to head classes (M4)** → fall back to frozen-bank
  retrieval (M4a) and drop the "evolving memory" claim rather than ship a broken one.
- **Scattering/SincNet lose to fixed on held-out config** → keep it fixed; don't add learnability we
  can't justify.
- **Structured evidence (M7) won't train** → shelve; the MVP doesn't depend on it.

## Immediate next action
Run **M0** (the robustness probe) on a few real grids — it's cheap, decisive, and fixes the M1
primitive set with numbers instead of arguments.
