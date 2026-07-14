# HALO Evidence-Engine — Research Synthesis & Decision Report

> Produced by a 7-facet research workflow (Consensus + Perplexity/web), 2026-07-14. Companion to
> `EVIDENCE_ENGINE.md` (design) + `EVIDENCE_ENGINE_BUILD_PLAN.md` (plan). Verdicts are the workflow's;
> treat as decision input, not gospel — but the citations are real and the novelty read is blunt.

## 1. TL;DR
- **Corroborated, yes — heavily.** Every building block of both pipelines has strong, citable precedent:
  config-conditioning-beats-invariance (Chorus, ZeroHAR, Sztyler), fixed-physical front ends matching
  learned ones (EfficientLEAF), covariance/eigen primitives (Riemannian-BCI canon), masked-channel
  relational SSL (channel-masking, RCSMR, STMAE), retrieval memory for zero-shot (TS-RAG, TimeRAF),
  evidential open-set (Sensoy EDL, DEAR), compositional zero-shot HAR (Al-Naser, MoPFormer). De-risks the
  engineering but means **no single ingredient is novel.**
- **Novelty is crowded on the two headline claims we lean on most** — "one language interface for unseen
  labels" is essentially *taken* (SensorLM, UniMTS, ZeroHAR, IMUZero, TENT); "text-keyed channel set" was
  independently published twice in the last year (CHARM, "Time to Embed"). The **defensible seam is a
  conjunction no one holds**: config-*conditional* (not invariant) physical primitives + non-parametric
  evidential memory + calibrated open-set **abstain**, evaluated on the **unseen-acquisition-config** axis.
- **Highest-value improvements to adopt now:** (1) **Density-aware evidential decoder** — gate the Dirichlet
  with ANN retrieval density (DAEDL/DIP-EDL); fixes EDL's "uncertainty is a mirage" problem AND unifies
  Pipelines A+B for free. (2) **JEPA latent-space masked-channel prediction** instead of raw reconstruction
  (CHARM), keep a light raw head for physical grounding. (3) **Report coupled/operational open-set metrics**
  (OpenAUC, AUGRC, OOSA validation-set thresholds) — answers our own results-integrity flags.

## 2. Corroborated (strongest citation each)
- **Config-conditioning over invariance (the thesis).** *Chorus (2025)* — learns how context alters the
  signal, critiques DG for oversimplifying context, +20.2% in unseen contexts. + *CANet (CVPR 2023)*,
  *Xiong two-branch (IMWUT 2025, HSIC)*. **Cite as precedent, not our invention.**
- **Gravity-align / orientation-robust-without-discarding-geometry.** *Yurtman & Barshan (2017/18)* — cut
  orientation drop 31.8%→4.7% via Earth-frame + differential quaternion rotations.
- **Fixed physical front end ≈/> learned convs.** *EfficientLEAF (2022)* — learnable filterbanks don't
  reliably beat a fixed mel filterbank. Strongest external argument for our front end — **cite as a known
  result + warning, not a contribution.**
- **Learnable-filter → fixed-relational-primitive pipeline.** *EEGminer (2021)* — near one-to-one blueprint
  for Pipeline A (learnable filters → fixed magnitude + connectivity head). Best template AND biggest
  Pipeline-A threat.
- **Covariance eigen-ratios + reference alignment.** *Zanini (2018)* affine re-centering, *Tang (2021)*
  rotation alignment — validates eigen-ratio primitive AND align-to-config-reference stance.
- **Masked-channel relational SSL.** *"Channel masking" (2023)*, *RCSMR/MonoSelfPAB (2024)*, *STMAE (2023)*.
- **Retrieval memory improves zero-shot on frozen backbone.** *TS-RAG (2025)* +6.84%. + Instance-Discrim
  memory bank (2018), Proto/Matching Nets, PatchCore coreset (2022), deep-kNN OOD (2022).
- **Evidential open-set + abstain.** *Sensoy EDL (2018)*, *DEAR (ICCV 2021, Dirichlet open-set action)*,
  *Zhao GKDE (2020, vacuity-vs-dissonance)*, *Cortese (2026, open-set-HAR-on-IMU protocol)*.
- **Compositional primitives → unseen classes.** *Al-Naser (2018)*, *MoPFormer (NeurIPS 2025)*.
  Multi-vector/late-interaction (*ColBERT→ColPali*) — **never applied to IMU, genuine white-space.**

## 3. Novelty check (blunt)
| Headline claim | Status | Crispest surviving differentiator |
|---|---|---|
| **Config-conditional *salient* (not invariant) features** | **Partially taken** (Chorus owns philosophy on IMU; ZeroHAR injects sensor-type/axis/position text) | **free-form/open-vocab** config text for **unseen** sensors + **variable channel count**; primitives **invariant-by-construction to gain/orientation** *while* conditioning; ZeroHAR is fixed-vocab + can't abstain. Reframe as **open-config + robust-yet-conditional mechanism**, not "we use sensor context." |
| **Language-keyed masked-channel set model** | **Crowded / near-preempted** (CHARM 2026, "Time to Embed" 2025) | Differentiator is the **output**: masked-channel *residual decoded into named physical primitives* + coupling to evidence engine. **DROP first-use claim of channel-text; cite CHARM as concurrent.** |
| **Curated-memory evidence engine with abstention** | **Genuinely open as a conjunction** | **Lead with this.** No one combines learned evidential Dirichlet + calibrated abstain + differentiable non-parametric memory + config-conditioning. vs ZARA: trained/deployable-without-LLM, calibrated, abstains. vs TS-RAG: open-set *classification* w/ per-candidate evidence, not forecasting. |
| **Compositional subspace / multi-vector primitives** | **Open on the sensing side** | First ColBERT-style MaxSim over **physically-grounded** primitive vectors (measured from signal, not LLM text attributes → valid under novel sensor physics). Must show via ablation vs LLM-text attributes. |

**DROP / reframe:**
- **DROP** "one language interface for unseen **labels**" as a contribution — saturated (SensorLM, 59.7M hrs). Table stakes.
- **DROP** "constrained front end matches learned convs" as a *finding* — accepted in audio (EfficientLEAF). Reframe as cross-**config** transfer (audio never tested).
- **REFRAME** "we generalize ConSE" — make the reduction concrete/testable (HALO → ConSE when memory=seen-class prototypes, evidence=softmax, abstain off) or it's a reviewer target.
- **REFRAME** "calibrated confidence from the Dirichlet" — "EDL uncertainty is a mirage" (Shen, NeurIPS 2024) undercuts bare-Dirichlet claims. Ground confidence in **retrieval density + physical primitives**; validate empirically (ECE, reliability, AUGRC), never assert from EDL theory.

## 3b. Competitors to benchmark / cite
- **Zero-shot label+config (beat head-to-head):** UniMTS (NeurIPS 2024, invariance camp), ZeroHAR (AAAI-25, closest config-conditioning threat), SensorLM (2025, scale incumbent), IMUZero (IMWUT 2025).
- **Retrieval+evidence (Pipeline B foils):** ZARA (ACL-26 — beat on calibration/abstain/unseen-config even if it wins raw F1), TS-RAG/TimeRAF (own "retrieval-augmented zero-shot" branding, forecasting).
- **Invariance foils for the thesis ablation:** ContrastSense (IMWUT 2024, cleanest domain-invariant), CrossHAR (IMWUT 2024), Wonderwall (IMWUT 2026).
- **Open-set/OSR baselines:** Cortese head-mounted-IMU open-set (2026, adopt protocol), GHOST/PostMax, deep-kNN OOD, calibrated stacking (Wang ToSN 2023).
- **Front-end bake-off arms:** LEAF/EfficientLEAF, WaveHAR/MRWA, TSF (TPAMI 2026), NormWear CWT, MoPFormer.
- **Channel-interface threats to cite:** CHARM, "Time to Embed", NormWear (count-agnostic vs our identity-conditional — contrast explicitly).

## 4. Top improvements to adopt (ranked)
1. **Density-aware evidential decoder** (DAEDL 2024 / DIP-EDL 2026) → **Pipeline B / M4-M5.** Gate Dirichlet by ANN top-k retrieval density → vacuity grows far from memory. Fixes EDL overconfidence, unifies A+B. **Do first.**
2. **JEPA latent-space masked-channel prediction** (CHARM 2026) → **Pipeline A / M1-M2.** Predict masked channel in embedding space + light raw head for physical units. Pair with dual-space reconstruction for grounding.
3. **Coupled + operational open-set metrics** (OpenAUC 2022, AUGRC 2024, OOSA 2024) → **eval / M6.** OSCR/OpenAUC + AUGRC; reject threshold on validation+surrogate-unknowns, **never test.**
4. **Config-conditional covariance re-centering** (Zanini 2018 / Tang 2021) → **Pipeline A / M2.** Running Riemannian barycenter per channel-text config bucket; affine-transport SPD before eigen-ratios. Unsupervised.
5. **Learnable retrieval mixer + trainable retriever** (TS-RAG ARM 2025 / TimeRAF 2024) → **Pipeline B / M4.** Per-neighbor gating fed by similarity; backprop through query/key (stop-grad MoCo re-index).
6. **Compose-outliers for abstain head** (COMO 2025) + **feasibility pruning** (Mancini OW-CZSL 2021) → **Pipeline B / M5.** Synthesize unseen (label×config) primitive-sets as negatives to *supervise* abstain; prune physically-infeasible (label,config) pairs.
7. **HSIC cross-head decorrelation** (Xiong 2025) → **Pipeline A / M3.** HSIC independence between config-factor and activity-factor heads.
8. **Explicitly train dropped-channel scenarios + accuracy-vs-#channels curve** (STMAE/RobustHAR) → **A/M2+eval.** The evidence reviewers will demand for variable-channel.
9. **Product-key memory indexing** (Lample 2019 / Memory Layers 2024) → **B/M5.** Sub-linear top-k as write-on-GT grows; greedy k-center coreset (PatchCore) for coverage.
10. **ColBERT MaxSim over multi-space primitive heads** (ColPali 2025) → **B/M4.** Per-family metric (Wasserstein spectral shape, cosine eigen-ratio), fused. **First on IMU = concrete novelty.**

## 5. Gaps / risks (we may be underestimating)
- **EDL uncertainty is theoretically unreliable** (Shen NeurIPS 2024 — non-vanishing with infinite data; really energy-based OOD). Bare-Dirichlet "calibrated confidence" = reviewer kill-shot → density-grounding (#1) + empirical calibration only.
- **The label-side thesis is already gone** (SensorLM/UniMTS/ZeroHAR/CHARM). Lead with coupling + unseen-config + abstain, not the interface.
- **No canonical open-set IMU benchmark** — opportunity + risk; must standardize (DAGHAR/HARBench + OOSA thresholds).
- **Cadence primitive placement-robust only with the right estimator** — naive autocorrelation degrades 73-87% across positions vs 98.9% Sylvester-criterion adaptive (van Oeveren 2018). Don't ship raw autocorrelation.
- **Invariance-vs-conditioning risks being rhetorical** — UniMTS/ContrastSense get strong numbers *via* invariance. **THE make-or-break ablation:** a concrete held-out placement/orientation case where conditioning beats the same backbone trained with a ContrastSense-style domain-invariant loss. If not, thesis unsupported.
- **ZARA may beat us on raw closed-set F1** (2.53× macro-F1, training-free). Pre-design where we win: calibration, abstention, unseen-config, deployability-without-LLM.
- **From-scratch tokenizer is a real risk** — de-risk with physics-simulated (skeleton→IMU) pretraining before real-data refinement.
- **CHARM/"Time to Embed" concurrency** → honest related-work scoping paragraph or reviewers flag overclaiming.

## 6. Reading list
**Thesis / invariance-vs-conditioning**
- Chorus: Harmonizing Context and Sensing Signals (Zhang et al., 2025) — https://consensus.app/papers/details/2e981cf802b9552eabeed893234b1eda/
- ZeroHAR (AAAI-25) — https://ojs.aaai.org/index.php/AAAI/article/view/33762
- UniMTS (NeurIPS 2024) — https://arxiv.org/abs/2410.19818
- ContrastSense (IMWUT 2024) — https://dl.acm.org/doi/10.1145/3699744

**Front end / primitives (Pipeline A)**
- EfficientLEAF: a Learnable Frontend of Questionable Use (2022) — https://arxiv.org/pdf/2207.05508
- EEGminer (Ludwig et al., 2021) — https://consensus.app/papers/details/7e9ff5511bd6550597e9bad5298f513f/
- Bruna & Mallat, Invariant Scattering Convolution Networks (2012) — https://arxiv.org/pdf/1203.1513
- Riemannian transfer / SPD re-centering (Zanini et al., 2018) — https://consensus.app/papers/details/992c883a1a8a5fe8a12af1fc6667f3b8/

**Channel-set / masked SSL — novelty threats**
- CHARM: Giving Sensors a Voice — Multimodal JEPA (2026) — https://arxiv.org/abs/2605.31580
- Time to Embed: channel descriptions unlock TSFMs (2025) — https://arxiv.org/abs/2505.14543
- "Channel masking" SSL-HAR (2023) — https://consensus.app/papers/details/20974afa313750009a4fed77d6b362b2/
- Moirai any-variate attention (2024) — https://arxiv.org/abs/2402.02592

**Memory / retrieval / evidential (Pipeline B)**
- TS-RAG (2025) — https://arxiv.org/pdf/2503.07649
- ZARA: Evidence-Grounded LLM Agents for Motion Time-Series (ACL-26) — https://arxiv.org/abs/2508.04038
- Sensoy et al., Evidential Deep Learning (2018) — https://consensus.app/papers/details/b9d44d6b5554566f89388b8f679d17ca/
- DEAR: Evidential DL for Open Set Action Recognition (ICCV 2021) — https://arxiv.org/abs/2107.10161
- DAEDL: Density-Aware Evidential DL (2024) — https://consensus.app/papers/details/b8b88ccb09b151c7a76c8fea36329460/
- "Are UQ Capabilities of EDL a Mirage?" (Shen et al., NeurIPS 2024) — https://consensus.app/papers/details/77fec22a4424596a8586863ff7116ae7/

**Compositional / multi-vector**
- MoPFormer: Motion-Primitive Transformer (NeurIPS 2025) — https://arxiv.org/abs/2505.20744
- ColPali: Vision-Language Late Interaction (ICLR 2025) — https://proceedings.iclr.cc/paper_files/paper/2025/file/99e9e141aafc314f76b0ca3dd66898b3-Paper-Conference.pdf

**Eval**
- OpenAUC / AUGRC eval-flaws (Wang 2022 / Traub 2024) — https://consensus.app/papers/details/061676420c605cbca285846f84aa10b1/
- DAGHAR leakage-free benchmark (Scientific Data 2024) — https://consensus.app/papers/details/910e901c900f5613b9917dd9e8e6cd78/
