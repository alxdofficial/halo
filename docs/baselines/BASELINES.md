# Baseline roster, contracts, and comparison design

The authoritative, paper-ready reference for **which baselines we compare against, how each is run, and how
we report results.** Every contract below was **verified against the actual paper + code** (2026-07-12), and re-verified
against each upstream repo's preprocessing on **2026-08-22** — see §3b for the two corrections that
sweep produced.
Companion docs: `BASELINE_FAIRNESS_POLICY.md` (the treatment contract), `MOTIVATION.md` (why HALO),
`AUGMENTATIONS.md` (the conditioning experiment).

## 1. Roster

| Baseline | Paper | Weights released? | **Verdict** |
|---|---|---|---|
| **CrossHAR** | Hong et al., IMWUT 2024 | **No** (training code only) | **self-pretrain on our corpus** |
| **LiMU-BERT** | Xu et al., SenSys 2021 | **No** (SSL recipe, ~62K params) | **self-pretrain on our corpus** |
| **harnet / ssl-wearables** | Yuan et al., npj Digit. Med. 2024 | **Yes** (`harnet5/10/30`) | **frozen (released)** |
| **UniMTS** | Zhang et al., NeurIPS 2024 | **Yes** (274 MB ckpt) | **frozen (released)** |
| **NormWear** | Luo et al., arXiv 2412.09758 | **Yes** (194M + frozen 1.1B LLM) | **frozen (released)** |
| **ImageBind** | Girdhar et al., CVPR 2023 | **Yes** (`imagebind_huge`, 4.5 GB) | **frozen (released)** |

All six are wired as auto-registered `baselines/<name>/adapter.py` adapters; the four frozen run
zero-shot as-is, the two self-trained self-pretrain on our corpus first (see §2). ImageBind uses only
its **IMU + text** towers (no images at inference) — a cosine-tier baseline like UniMTS, and the
"generic multimodal binding" reference floor.

## 2. Why each is frozen or self-trained (verified rationale)

**Self-pretrained (CrossHAR, LiMU-BERT).** These ship **no pretrained weights at all** — the repos are
training code for a self-supervised *recipe* (CrossHAR: masked + contrastive; LiMU-BERT: masked
reconstruction, ~62K params). The *faithful* use of an SSL method is to pretrain it on your own unlabeled
data, so we self-pretrain both on **our** training corpus. This also makes them leakage-free (their
released example checkpoints, where they exist, were pretrained on datasets that overlap our eval sets)
and gives a genuine **same-data, same-protocol** comparison. They are small, so this is cheap.

The locked expanded-corpus recipes are:

```bash
python -m baselines.crosshar.train --recipe full --gpu
python -m baselines.limubert.train --recipe full --gpu
```

Both use the 18-source `EXPANDED_PHASE_A_TRAIN_DATASETS` roster with a 20,000-window
per-stream cap. CrossHAR keeps its published batch 512, paired-axis augmentation,
masked reconstruction, and final contrastive phase. LiMU-BERT keeps its published
batch 128 and masked-reconstruction objective. The epoch counts preserve the number
of examples seen by the validated 12-source runs after distributing those examples
over the expanded corpus; literal paper epoch counts are not comparable because the
pooled corpus is much larger than either paper's original pretraining set. Each new
checkpoint carries a machine-readable corpus and input-contract stamp, and the
adapters reject stale or mismatched checkpoints.

**Frozen (harnet, UniMTS, NormWear).** These are **released foundation-model products** whose power *is*
large-scale pretraining we cannot and should not reproduce:
- **harnet** — pretrained on **UK-Biobank, ~700k person-days**; the paper explicitly shows downstream F1
  rises log-linearly with the number of pretraining subjects, i.e. the value is the **scale, not the
  pretext technique**. Retraining it on our small corpus would delete its contribution.
- **UniMTS** — pretrained on **HumanML3D mocap → physics-simulated IMU + GPT-3.5 text**, aligned to a
  frozen CLIP text tower; usable **zero-shot off-the-shelf** (cosine similarity, no per-dataset head).
- **NormWear** — a **194M** channel-independent encoder + a **frozen 1.1B Clinical-TinyLlama** text head,
  pretrained on ~15k signal-hours of multi-sensor *physiological* data (ECG/PPG/EEG dominate; IMU is a
  minor share). It is the "general FM applied to IMU" reference, not a HAR specialist.

Retraining any of these three from scratch on our corpus would be **unfaithful** (it discards the
pretraining that defines the method) and invites the reviewer rebuttal *"that is not the real model."*

## 3. The channel / rate defense — pre-answering "you only gave UniMTS 3 channels"

**Each model receives the *most* its own trained contract can accept — never fewer.** The asymmetry in
channel count and rate is dictated by **each model's own published design**, not by us starving it:

| Baseline | Channels it gets | Rate | Why — this is *its* contract, not our choice |
|---|---|---|---|
| **CrossHAR** | **6** (acc+gyro) | 20 Hz | its published input; matches HALO's 6-ch |
| **LiMU-BERT** | **6** (acc+gyro) | 20 Hz | its published input; matches HALO's 6-ch |
| **harnet** | **3** (acc only) | 30 Hz | its released weights are **accelerometer-only** (UK-Biobank has no gyro); the frozen conv kernels are 3-channel — a 4th channel is architecturally impossible without retraining |
| **UniMTS** | **3** (acc only) | 20 Hz | its **released checkpoint is accelerometer-only** (verified from the checkpoint tensors); giving it gyro would require re-pretraining, which discards its simulation-based pretraining |
| **NormWear** | **6** (acc+gyro) | ~65 Hz | channel-independent → we give it our **full** acc+gyro; it is *not* starved |
| **HALO (ours)** | 6 (or 3 in parity) | native | accepts variable channels by design |

So the honest statement for the paper: *"Each baseline is fed the maximal input its released model accepts.
harnet and UniMTS are accelerometer-only **because their published, released checkpoints were pretrained
on accelerometer-only data** — feeding them gyroscope channels is not possible without retraining and
discarding their pretraining. NormWear, being channel-independent, receives the full 6-channel input."*

**The airtight control — the parity row.** To kill "you gave HALO more channels/rate," we report a
**HALO parity row run on the 3-channel accelerometer-only input at each frozen baseline's rate.** If HALO
still wins at 3-ch accel-only, the advantage is architecture + language, not the extra gyro. This parity
row is the load-bearing defense; the "each model gets its published max" statement is the framing around it.

### 3b. Two contract corrections — 2026-08-22 audit

A full sweep of the eval path against each upstream repo
([`../results/EVAL_HARNESS_AUDIT_20260822.md`](../results/EVAL_HARNESS_AUDIT_20260822.md)) found the
scoring code sound but two places where a baseline was **not used as its developers intended**. Both
are now fixed in code, and both corrections favour the baseline — which is the honest direction.

| # | Baseline | What was wrong | Status |
|---|---|---|---|
| F1 | **LiMU-BERT** | Upstream `Preprocess4Normalization` divides accel by 9.8 because *its* datasets store m/s²; the model is designed to see accel ≈ 1 in **g**. Our grids **already store g**, and both our SSL pretrain and the adapter divided again — the backbone saw accel ≈ 0.10 against gyro O(0.1–0.5), about **10× off the designed accel:gyro balance**. | Both divisions removed. **Its backbone must be re-pretrained**; until then an adapter guard refuses the old-convention checkpoint rather than emitting a valid-looking wrong number. Every published LiMU-BERT figure predates the fix. |
| F2 | **UniMTS** | Upstream evaluates with **enriched** label text — each class's whole `label_dictionary` synonym list joined into one string (`data.py`: `' '.join(labels)`). We passed the bare class name, while our own `halo_evidence` row already ensembled E=8 paraphrases. | `encode_labels` now averages the shared **train-only** paraphrase pool (E=8, variant 0 = the raw label) through UniMTS's *own* CLIP tower. Affects zero-shot only — enrollment methods read no label text. |

Rule this establishes: **an input-contract claim in this file is only trustworthy once it has been
checked against the upstream repo's own preprocessing code**, not against its paper text alone.

## 4. Evaluation & reporting design

### 4a. What the published models put above their encoders

The baseline papers do not support the claim that every successful HAR model needs an elaborate
classification engine. Most of their novelty and parameter capacity is in representation learning;
their final decision rules are usually small. There are two important exceptions: CrossHAR and
LiMU-BERT use modest sequence models for supervised downstream adaptation, and NormWear learns a
query-conditioned fusion module before its simple final distance comparison.

| Baseline | Native published downstream decision rule | Where the model's main contribution lies | Our matched enrollment comparison |
|---|---|---|---|
| **harnet** | `feature -> Linear(512) -> ReLU -> Linear(classes)` (`EvaClassifier`); the paper either freezes the trunk or fine-tunes it | Large-scale accelerometer SSL and the ResNet feature extractor | Frozen encoder + 1-NN is primary; prototype and fitted heads are secondary controls |
| **CrossHAR** | A one-layer Transformer over the pretrained sequence, mean pooling, then one linear class layer | Cross-dataset augmentation and hierarchical masked/contrastive pretraining | Frozen encoder + 1-NN is primary; our global-vocabulary ConSE probe is a separate k=0 bridge |
| **LiMU-BERT** | Default `LIMU-GRU`: two small GRU stages followed by one linear class layer; the paper also compares alternate task heads | BERT-style masked IMU representation learning | Frozen encoder + 1-NN is primary; our global-vocabulary ConSE probe is a separate k=0 bridge |
| **UniMTS** | Zero-shot: normalized IMU/text dot product with a learned temperature. Few/full-shot: one `Linear(512, classes)` layer, with the encoder frozen or fine-tuned | Simulated all-joint IMU pretraining and motion-text alignment | Frozen encoder + 1-NN is the matched k>=1 comparison; native cosine is used at k=0 |
| **NormWear** | Query-conditioned MSiTF fusion, then minimum L1 distance to candidate text embeddings | Channel-independent wearable pretraining and signal/text alignment | Frozen fused representation + 1-NN is matched at k>=1; native L1 text matching is used at k=0 |
| **ImageBind** | Zero-shot: normalized IMU/text similarity; standard downstream evaluation uses linear probes | Large-scale cross-modal binding in a shared embedding space | Frozen encoder + 1-NN is matched at k>=1; native cosine is used at k=0 |

The implementation sources are the official
[harnet](https://github.com/OxWearables/ssl-wearables),
[CrossHAR](https://github.com/kingdomrush2/CrossHAR),
[LiMU-BERT](https://github.com/dapowan/LIMU-BERT-Public),
[UniMTS](https://github.com/xiyuanzh/UniMTS),
[NormWear](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear), and
[ImageBind](https://github.com/facebookresearch/ImageBind) repositories. This table describes each
paper's native head; it must not be confused with our controlled readout comparison, which applies
the same 1-NN, prototype, ridge, and fitted-linear protocols to every frozen representation.

Three complementary comparisons — each answers a *different* question, and the design mirrors what all
six baseline papers themselves do (technique attribution via within-method same-data ablation, never by
equalizing pretraining data across foundation models):

1. **Frozen-SOTA bar** — harnet, UniMTS, NormWear run **frozen** on our held-out eval sets, each through
   its own `adapter.py` (its exact channels/rate/window/normalization). Question: *are we competitive with
   the real published SOTA?* (Different pretraining data is accepted, as in every FM paper.)
2. **Same-data method row** — CrossHAR + LiMU-BERT **self-pretrained on our corpus/protocol**. Question:
   *at equal data + protocol, does our architecture + interface win?*
3. **Within-HALO causal ablation** — the **told-vs-not-told** conditioning experiment (+ a HALO
   no-conditioning / random control, mirroring UniMTS's "Random"). Question: *is the win caused by the
   language technique, not the data?* **This is the causal proof** — data is held fixed, only the
   descriptor is toggled (`MOTIVATION.md` §3, `AUGMENTATIONS.md`).

Closed-vocab baselines (CrossHAR, LiMU-BERT, harnet) reach the open-set eval via the **ConSE bridge**;
text-native ones (UniMTS, NormWear) use their own text tower (UniMTS since 2026-08-22 over the shared train-only E=8 paraphrase ensemble, per §3b). Metric: **macro-F1**, subject-disjoint,
zero-shot vs each dataset's own label strings.

## 5. Completeness of the frozen-SOTA set (verified survey, 2026-07-12)

The current set spans the three foundation-model schools cleanly — large-scale **accel SSL** (harnet),
**language-aligned mocap-sim** (UniMTS), **multimodal physiological FM** (NormWear) — plus the two
small-SSL floors (CrossHAR, LiMU-BERT). A verified landscape survey found it **nearly comprehensive**,
with **one gap a reviewer will flag**.

**ADD:**
- **ImageBind** (Meta, CVPR 2023) — **DONE ✓ (integrated as the 4th frozen adapter).** The canonical
  multimodal foundation model with a released IMU tower (accel+gyro) + text zero-shot, and — decisively —
  **both UniMTS and NormWear benchmark against it.** Frozen `imagebind_huge`; cosine tier (IMU + text
  towers, no images). Its IMU was trained on Ego4D head-mounted single-location, so it scores low on
  phone/watch HAR (our protocol: macro-F1 15.2% on motionsense; ~12.5% acc in UniMTS Table 1) — that is
  the *point*: the "generic multimodal binding fails on phone/watch HAR" reference floor.
- **PRIMUS** (Nokia Bell Labs, ICASSP 2025) — **optional.** Released IMU-SSL encoder (Zenodo weights,
  Ego4D, accel+gyro 50 Hz). Downstream use is **few-shot/linear-probe, not zero-shot** → belongs in the
  SSL tier alongside CrossHAR/LiMU-BERT, only if we want a fresh 2024–25 IMU-SSL point. Else cite.

**SKIP (with reviewer-ready reasons):**
- **IMU2CLIP** — code only, **no released weights** (UniMTS re-pretrained it); Ego4D; already reported by UniMTS. Cite.
- **SensorLM (Google, NeurIPS 2025) / SensorLLM (EMNLP 2025) / LLaSA** — sensor-language models; **no distributable weights** (private Fitbit/Pixel data) and/or captioning-QA, not clean open-set classifiers, and slow. Cite as concurrent SOTA. *(Also the works to check for the "free-language channel-conditioning is unique to HALO" claim — see §6.)*
- **AURA-MFM** — preprint, no release, smartglasses. Cite.
- **HARGPT / IMUGPT / LLaVA** — prompt-only LLM/VLM; **already in UniMTS's zero-shot table and beaten by it**, so including UniMTS subsumes this tier. Skip the run.
- **Generic TS FMs — Chronos, MOIRAI, TimesFM** — forecasting; cannot do zero-shot open-set classification (no label-semantic head). Skip.
- **MOMENT, GPT4TS, UniTS, OTiS** — classify only with a **trained head** (not zero-shot/open-set) and are heavy. Skip. *(Rationale upgraded from "too slow" to "generic TS FM, not zero-shot/open-set-capable" — more defensible.)*

**Verdict:** the **3 frozen + 2 self-trained** set is defensible; adding **ImageBind** makes it
reviewer-proof. No general-purpose TS FM and no LLM-prompting method belongs in the *run* set (UniMTS
subsumes the zero-shot LLM/CLIP tier; TS FMs can't do open-set zero-shot HAR). Reference survey:
"Foundation Models Defining A New Era in Sensor-based HAR" (arXiv:2604.02711).
