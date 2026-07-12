# Baseline roster, contracts, and comparison design

The authoritative, paper-ready reference for **which baselines we compare against, how each is run, and how
we report results.** Every contract below was **verified against the actual paper + code** (2026-07-12).
Companion docs: `BASELINE_FAIRNESS_POLICY.md` (the treatment contract), `MOTIVATION.md` (why HALO),
`AUGMENTATIONS.md` (the conditioning experiment).

## 1. Roster

| Baseline | Paper | Weights released? | **Verdict** |
|---|---|---|---|
| **CrossHAR** | Hong et al., IMWUT 2024 | **No** (training code only) | **self-pretrain on our corpus** |
| **LiMU-BERT** | Xu et al., SenSys 2021 | **No** (SSL recipe, ~62K params) | **self-pretrain on our corpus** |
| **harnet / ssl-wearables** | Yuan et al., npj Digit. Med. 2024 | **Yes** (`harnet5/10/30`) | **frozen (released)** |
| **UniMTS** | Zhang et al., NeurIPS 2024 | **Yes** (274 MB ckpt) | **frozen (released)** |
| **NormWear** | Tang et al., arXiv 2412.09758 | **Yes** (194M + frozen 1.1B LLM) | **frozen (released)** |

## 2. Why each is frozen or self-trained (verified rationale)

**Self-pretrained (CrossHAR, LiMU-BERT).** These ship **no pretrained weights at all** — the repos are
training code for a self-supervised *recipe* (CrossHAR: masked + contrastive; LiMU-BERT: masked
reconstruction, ~62K params). The *faithful* use of an SSL method is to pretrain it on your own unlabeled
data, so we self-pretrain both on **our** training corpus. This also makes them leakage-free (their
released example checkpoints, where they exist, were pretrained on datasets that overlap our eval sets)
and gives a genuine **same-data, same-protocol** comparison. They are small, so this is cheap.

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

## 4. Evaluation & reporting design

Three complementary comparisons — each answers a *different* question, and the design mirrors what all
five baseline papers themselves do (technique attribution via within-method same-data ablation, never by
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
text-native ones (UniMTS, NormWear) use their own text tower. Metric: **macro-F1**, subject-disjoint,
zero-shot vs each dataset's own label strings.

## 5. Completeness of the frozen-SOTA set

Current frozen set spans three archetypes: large-scale **accel SSL** (harnet), **language-aligned
mocap-sim** (UniMTS), **multimodal physiological FM** (NormWear). Candidates to consider adding are
tracked by a verified landscape survey (see the session notes / `[[halo-baseline-verified-contracts]]`);
the leading candidate is **ImageBind** (frozen multimodal binding with an IMU tower + text — used as a
baseline by *both* UniMTS and NormWear). This section is updated once the survey lands.
