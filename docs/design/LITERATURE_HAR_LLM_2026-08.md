# Literature notes — what the HAR / sensor-FM field is actually working on

Source: lab reading sheet "LLM for CPS-IoT System" (37 rows), filtered to sensing-relevant entries,
plus three papers reached by citation-chasing from them. PDFs + extracted text in
`references/literature/<slug>/`. PDFs are gitignored (`references/**/*.pdf`), notes are not.

Read 2026-08-11. Every claim below is from the paper's own text unless marked *[my read]*.

---

## 0. The one paper that matters most for us

### Haresamudram, Beedu, Rabbi, Saha, Essa, Plötz — **"Limitations in Employing Natural Language
### Supervision for Sensor-Based HAR — And Ways to Overcome Them"**, AAAI 2025 (arXiv 2408.12023)

`references/literature/nl_supervision_limits/`

**This is our negative result, published a year before us, with a bigger study.**

- Pre-trains CLIP-style natural-language supervision (NLS) on Capture-24, then zero-shot predicts on
  six target datasets (HHAR, Myogym, MobiAct, MotionSense, PAMAP2, MHEALTH).
- Headline: zero-shot NLS is **"drastically worse by around 30-40%"** than plain end-to-end
  supervised training and than SimCLR self-supervision. Their Figure 1 is that gap.
- They name exactly two causes:
  1. **Sensor heterogeneity** — "differences in data distributions due to hardware constraints and
     settings such as gain, data and signal processing, differences in sampling rates — even if the
     sensor locations and activities are the same." Cites Stisen 2015.
  2. **Lack of rich, diverse text descriptions** — HAR datasets have "only a handful of activity
     labels", "a far cry from the 400M image-text pairs in the original CLIP paper."
- Their remedies:
  - Adapt some layers of the pre-trained net on target data with **as little as 4 minutes of
    data per activity** → substantial recovery.
  - Generate additional prompts via LLMs; incorporate external knowledge → substantial improvement.
  - Encoder ablations: simple conv encoders beat ResNets; BERT/RoBERTa/CLIP-text beat DistilBERT.
- Their own conclusion is hedged: *"While sensor-language modeling does not outperform
  state-of-the-art supervised and self-supervised training for some datasets, its additional
  capabilities like recognizing unseen activities, and performing cross-modal search, are clearly
  advantageous."*

**Bearing on us** *[my read]*:
- Our §5c finding that the language bridge delivers ~1.2 points over chance is **confirmatory, not
  novel**. The "language interfaces for wearable sensing don't work" headline is taken.
- Their diagnosis stops where ours starts. They attribute the failure to heterogeneity + description
  poverty — both *data* explanations. They do not test whether the model is using the text at all.
  Our alias arm (nonsense labels beat real ones by 11 points on the trained decoder) is a
  *mechanistic* finding they have no instrument for.
- Their accepted remedy — 4 min of labeled target data per activity — **is enrollment**. The field
  has already conceded pure zero-shot and moved to "a little target data." That is exactly our
  k-curve. Good news for problem choice, bad news for novelty of the framing.

---

## 1. The scale route — Google, and it is closed to us

### LSM-1 — "Scaling Wearable Foundation Models", ICLR 2025 (`lsm1/`)
- 40M hours, >165,000 people; HR, HRV, accel, EDA, skin temp, altimeter at **per-minute** resolution.
- Establishes scaling laws for imputation / interpolation / extrapolation across time and sensor
  modalities. Activity recognition appears as a *downstream sample-efficiency probe*, not the goal.

### LSM-2 — "Learning from Incomplete Wearable Sensor Data" (arXiv 2506.05321) (`lsm2/`)
- Adaptive and Inherited Masking (AIM). Learnable mask token represents **both** "inherited"
  missingness (real gaps in the raw data) and artificial MAE masking.
- Motivating stat: **"our dataset exhibits pervasive missingness: 0% of records are complete."**
- Explicitly rejects imputation: it "risks introducing biases that can propagate to downstream
  models."
- Notes prior wearable FMs dodged missingness by using short context windows (<60 s, 2.56 s, 10 s)
  and filtering incomplete instances out — but clinically relevant patterns (circadian rhythm, HRV,
  daily activity profiles) need day-long windows, which always contain gaps.
- Claims first work to do representation learning directly on incomplete wearable data.

### SensorFM — "Towards a General Intelligence and Interface for Wearable Health Data"
### (arXiv 2605.22759, v3 2026-07) (`sensorfm/`)
- **>1 trillion minutes**, 5 million participants, ViT-1D masked autoencoder, 34 one-minute
  aggregate features from PPG / accel / EDA / skin temp / altimeter.
- 35 health prediction tasks (cardiovascular, metabolic, sleep, mental health, lifestyle, demographics).
- Joint scaling of capacity + data gives systematic improvement; unlocks label-efficient few-shot.
- Deploys "a classroom of LLM agents" to search the space of downstream predictive heads.
- Validated with 1,860 clinician ratings via a Personal Health Agent.
- ~40 authors, Google Research + DeepMind.

**Bearing on us** *[my read]*:
- Any contribution that reduces to "our wearable foundation model is better" is dead. We cannot
  compete on scale and the gap is five orders of magnitude in data.
- **But note what this route is not doing.** SensorFM operates on *minute-aggregate features* for
  *health outcomes*. It is not fine-grained motion recognition from raw high-rate IMU, it does not
  span heterogeneous body placements, and it has no open label vocabulary. LSM's HAR is a probe.
  The fine-grained / heterogeneous-placement / raw-IMU / open-vocabulary corner is genuinely vacant
  — not because it is solved, but because the big labs are optimising a different objective.
- LSM-2's missingness is the *closest cousin* to our observability idea and is importantly
  **different**: theirs is data physically absent (battery, dropout, artifact); ours is data present
  but unable to resolve a distinction. Nobody in this set models the second.

---

## 2. Language-centered HAR

### LanHAR — "Language-centered Human Activity Recognition" (arXiv 2410.00003) (`lanhar/`)
- Problem statement is verbatim ours: *"variations in activity patterns, device types, and sensor
  placements create distribution gaps across datasets."*
- Method: LLM generates **semantic interpretations of both sensor readings and activity labels**;
  align the two in language space; iterative re-generation (KL-guided) to control hallucination;
  two-stage training distils to a lightweight mobile sensor encoder.
- Evaluated on 4 datasets (HHAR, UCI, MotionSense, Shoaib) cross-dataset + a "new activity" protocol
  (train on 4 common activities, test on the target's other activities).
- Reported F1s sit in the ~0.3–0.65 band. Their own stated limitations: LLM inference is slow, and
  *"they remain relatively small in scale with a limited number of activities."*

### SensorLLM (EMNLP 2025, arXiv 2410.10624) (`sensorllm/`)
- Two stages: (1) Sensor-Language Alignment — auto-generated *trend descriptions* ("0.0-0.13s:
  downward; 0.13-0.14s: stable...") with special tokens marking channel boundaries; (2) Task-Aware
  Tuning for HAR classification.
- Names four obstacles to feeding sensors to LLMs: numerical tokenisation, context length,
  multivariate structure, prompt engineering.
- **Stated limitation, directly relevant:** *"relying on a fixed-class classifier may constrain
  adaptability to new activity categories and does not fully leverage the reasoning potential of
  LLMs."* They explicitly defer generative/zero-shot to future work.

### HARGPT — "Are LLMs Zero-Shot Human Activity Recognizers?" (arXiv 2403.02727) (`hargpt/`)
- Feeds *raw* IMU numbers to GPT-4 with role-play + chain-of-thought prompting.
- Claims **~80% average accuracy on unseen data**, beating traditional ML and deep baselines.
- Caveats *[my read]*: two datasets, four coarse classes (bicycling / walking / sitting-standing /
  stairs), 6-page paper. The result is in direct tension with Haresamudram's systematic finding.
  Treat as an existence proof on easy, well-separated classes, not a general capability.

**Bearing on us** *[my read]*: the language-centered cluster is crowded, its results are modest
(0.3–0.65 F1 on 4-dataset cross-dataset), and the strongest systematic study in the set says the
approach underperforms. Being "the language interface paper" is a bad position in 2026.

---

## 3. LLM-as-reasoner over sensor/time-series data

### Towards Time-Series Reasoning with LLMs (arXiv 2409.11376, Stanford/Apple/UIUC) (`ts_reasoning_llm/`)
- Lightweight TS encoder on top of an LLM, then CoT-augmented fine-tuning to produce reasoning paths.
- Frames three necessary steps: **perception → contextualisation → deductive reasoning**, and
  hypothesises existing TS-MLLMs suffer a **perception bottleneck**.
- Beats GPT-4o on zero-shot reasoning tasks; latent representation reflects slope, frequency.

### TimeSeriesExam (arXiv 2410.14752, CMU Auton Lab) (`timeseriesexam/`)
- 700+ procedurally generated MCQs over 104 templates, refined with Item Response Theory.
- Five categories: pattern recognition, noise understanding, similarity analysis, anomaly detection,
  causality analysis.
- Result: closed-source (GPT-4, Gemini) understand *simple* concepts much better than open models;
  **all models struggle with causality**.

### IoT-LLM (arXiv 2410.02429, NTU; Patterns 2025) (`iot_llm/`)
- LLMs "often produce outputs that defy physical laws" on physical-world reasoning.
- Three-step framework: format IoT data for LLMs, IoT-oriented RAG for in-context knowledge, CoT +
  role prompting to activate commonsense.
- 5-task benchmark; GPT-4o-mini +49.4% average over prior methods.

### Using LLMs for Late Multimodal Sensor Fusion for Activity Recognition (arXiv 2509.10729, Apple) (`llm_late_fusion/`)
- LLM performs **late fusion** over outputs of modality-specific audio and motion models — no shared
  embedding space, no task-specific training.
- 12-class zero-/one-shot F1 significantly above chance on an Ego4D subset.
- Motivation is explicitly ours-adjacent: avoids needing aligned training data to learn a joint space.

**Bearing on us** *[my read]*: "perception bottleneck" is the field's own name for what we measured
as encoder purity 0.68. And the Apple late-fusion result is a cheap alternative to the joint-space
approach we (and LanHAR, and IMU2CLIP) took — worth citing as the rival design.

---

## 4. Structure, segmentation, and the window assumption

### Game of LLMs (HASCA @ UbiComp 2024, arXiv 2406.13777) (`game_of_llms/`)
- Opens with the assumption we've been circling: *"A popular analysis procedure used by the community
  assumes an optimal window length... in the scenario of smart homes, where activities are of varying
  duration and frequency, the assumption of a constant sized window does not hold."*
- Proposes **structural constructs** — "the underlying unique sub-units or components that either
  constitute an activity or are relevant to it" — discovered by prompting GPT-4 / Gemini.
- Argues constructs especially help **short-duration and infrequent activities**.
- 6-page workshop paper, smart-home ambient sensors (not IMU), no full recognition system.

**Bearing on us** *[my read]*: the fixed-window critique is *stated* in the literature but not
*solved*, and not at all for wearables. That is an opening for the multi-resolution / promptable-
extent direction, and a citation that establishes the problem is recognised rather than invented.

---

## 5. Physics / calibration

### Transformer IMU Calibrator (TIC), SIGGRAPH 2025 / TOG (arXiv 2506.10580) (`tic_imu_calibrator/`)
- Breaks the "absolute static assumption" in IMU calibration: coordinate drift `R_G'G` and
  sensor-body measurement offset `R_BS` are usually assumed constant for a whole session.
- Relaxes to: (i) matrices change negligibly within a short window; (ii) movements are diverse within
  that window. Transformer estimates both from a short history of orientations + accelerations.
- Adds a **calibration trigger based on IMU rotation diversity** — only calibrate when the data
  actually constrains the estimate.
- Claims first *implicit* calibration (no T-pose, no heading reset) and first long-term accurate
  sparse-IMU mocap.

**Bearing on us** *[my read]*: this is the closest published thing to "estimate the deployment
configuration from the signal rather than trusting the declaration." It is from the graphics
community, targets orientation matrices for motion capture, and has no notion of retrieval or
labels — but the **calibration trigger** is precisely the "only trust this estimate when the data
supports it" logic we want in an admissibility gate. Cite it; do not claim the idea is unowned.

---

## 6. QA / interaction

### SensorChat, IMWUT 2025 (arXiv 2502.02883) (`sensorchat/`)
- First end-to-end QA over **long-duration, high-frequency** sensor history (days), vs prior systems
  limited to ~1 minute of data or daily-aggregate metrics.
- Three-stage pipeline: LLM question decomposition → sensor data query → LLM answer assembly.
- Handles quantitative (numeric precision) and qualitative (subjective inference) questions; up to
  93% higher accuracy than SOTA on quantitative. Runs on edge after quantisation.

**Bearing on us** *[my read]*: the retrieval-over-a-long-history framing is ours, aimed at QA rather
than recognition. If we go the promptable-detection route, this is the neighbouring system.

---

## 7. Synthesis — the field's implicit consensus

**Accepted as real problems (safe to motivate with):**
- Sensor/device/placement heterogeneity. Universally named, nobody claims to have solved it.
  (LanHAR, Haresamudram, Stisen 2015 as the canonical citation.)
- Label scarcity → label-efficient / few-shot learning is the accepted response.
- Missingness and incomplete records (LSM-2: "0% of records are complete").
- Fixed-window assumptions fail in free-living (Game of LLMs) — named, not solved.
- Perception bottleneck: models don't extract enough from the signal before reasoning.

**Demonstrated possible:**
- Scaling works — for *health outcomes* from *minute-aggregate* features (SensorFM, 35 tasks).
- Missingness-aware SSL without imputation (LSM-2).
- Small amounts of target data close most of the zero-shot gap — ~4 min/activity (Haresamudram).
- LLMs can do coarse HAR from raw numbers with CoT on easy, well-separated classes (HARGPT).
- LLM late fusion across modalities with no joint embedding space (Apple).
- Dynamic, implicit calibration from a short history of diverse motion (TIC).

**Shown to be too much to ask (as of now):**
- Plug-and-play zero-shot HAR through a language interface — 30-40% below supervised (Haresamudram).
- LLMs reasoning about complex TS concepts, especially **causality** (TimeSeriesExam: all models).
- LLMs obeying physical law in sensory reasoning without scaffolding (IoT-LLM).
- Fixed-class classifiers adapting to new activity categories (SensorLLM's own stated limitation).
- Cross-placement generalisation — *nobody in this set even attempts it.*

**Not addressed by anyone here** *[my read]*:
- **Resolvability / observability of a concept given a configuration.** LSM-2 models data that is
  *absent*. Nobody models data that is *present but insufficient* — the wrist that cannot witness
  jumping jacks. There is no equivalence-class or abstention output anywhere in this set.
- Retrieval-with-enrollment as the deployment story (closest: Haresamudram's 4-min adaptation, which
  is fine-tuning, not enrollment; and Tip-Adapter-style caches, absent from this literature).
- Within-recording sampling as a leakage unit for few-shot HAR evaluation.

---

## 8. Positioning consequences for HALO

1. **Drop "language interfaces don't work" as a headline.** Haresamudram AAAI'25 owns it. Keep our
   version as a *sharper mechanistic* result (the alias inversion, the four-mechanism ceiling) that
   their data-level diagnosis cannot produce — a section, not a paper.
2. **Do not compete on scale.** SensorFM is 1e12 minutes / 5M users / 40 authors. Any framing that
   invites that comparison loses.
3. **The vacant corner is fine-grained, heterogeneous-placement, raw-IMU, open-vocabulary
   recognition with an explicit account of what the configuration can resolve.** Google is not there;
   the language-HAR cluster is there but has conceded zero-shot; nobody models resolvability.
4. **Cite Game of LLMs and TIC as evidence our problems are recognised**, not invented — window
   assumption and signal-derived configuration respectively.
5. **The field's remedy is already enrollment.** That validates the k-curve as the right axis and
   means our enrollment results need to beat "4 minutes of fine-tuning", which is the real baseline
   the literature will demand. We have never run that comparison.

## 9. Read but judged not relevant

Rows in the sheet that are LLM-general rather than sensing: DeepSeek-V3, Less Is More (x2), PENCIL,
Tina, Certaindex, AlpacaFarm, AI PERSONA, Chain of Code, MCTS-assisted reasoning, Seed Diffusion,
QwenLong-L1.5, Verbalizable Representations, Analog Foundation Models, Efficient Long-Context,
VGGT, DreamVLA, CASIT, LLMind, CPS-LLM, Mobile Edge Intelligence survey, LLM-enabled CPS survey,
Debating with Persuasive LLMs, LLM-Powered Mobile Task Automation, Toward Cognitive Supersensing
(visual cognition, not sensing).
