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
**Build:** `training/evidence/probe_robustness.py` — load real grids; apply synthetic **gain**,
**SO(3) rotation**, **yaw-only rotation**, and mild **time-warp/resample**; compute candidate
features **before vs after** and report an invariance score (e.g. cosine / relative-L2 drift) per
feature family:
- raw per-channel band energy (baseline — expected fragile)
- gravity-aligned per-channel band energy
- per-band accel-triad **eigenvalue ratios** (expected rotation+gain invariant)
- accel↔gyro **coherence**, **relative spectral shape**, **cadence**
- (if quick) a fixed **scattering** first-order coefficient vs a plain STFT bin under time-warp.
**Gate:** a ranked table of "invariance vs discriminability" per feature. **Keep only families that
are both invariant to the nuisance AND still separate activities.** This fixes the M1 primitive set.
**Resolves:** §7 tokenizer forks; the scattering-vs-STFT deformation question (partly).

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

## M2 — The load-bearing loss (does the theory cohere?)  ·  ~2–3 days · **HARD GATE**
**Goal:** write and sanity-check the objective *before* the full model, because if it doesn't cohere,
nothing downstream matters.
**Build:** `training/evidence/losses.py`:
- **Config-conditional salient contrastive:** same-activity-across-configs pull together / different
  push apart, **conditioned on the channel-text** (not blind-invariant). Precise form to pin here.
- **Analysis-consistency:** same-*label* samples produce same *analysis* vector (metric loss on the
  analysis, decoupled from label correctness).
- **Masked-channel prediction** (SSL pretext; see M3): predict masked channels from present ones.
- **Evidential/abstention** term (Dirichlet evidence; deferred detail to M5).
**Gate (HARD):** on a *tiny* real+synthetic setup with a frozen M1 front end + a trivial encoder,
show the salient-contrastive loss (a) trains, (b) pulls same-activity/different-config together and
different-activity apart on held-out configs, and (c) the analysis-consistency term doesn't collapse
representations. **If the loss is incoherent or collapses, STOP and redesign — do not build M3+.**
**Resolves:** open fork #1 (the load-bearing loss) and #2 (task- vs self-consistency utility — decide
the blend here).

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
- **M4a:** frozen bank of encoded training primitives; **ANN top-k + soft attention** retrieval
  (differentiable blend over the k; which-k non-diff, fine). Prediction = confidence-weighted labels
  of neighbors (a learned ConSE). Establishes the retrieval-evidence baseline.
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
**Gate:** calibration (are confidences honest?), abstention behaves (low evidence → abstain), and the
per-candidate evidence is non-degenerate. Zero-shot macro-F1 ≥ M4 (decoder shouldn't hurt).

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
