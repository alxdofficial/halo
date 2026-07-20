# Data scaling plan (2026-07-20)

Written because the measured evidence says HALO is **data-limited, not parameter-limited**: a 7.17 M
encoder, ~50 epochs, retrieval purity plateaued at 0.68 with fine-grained classes at ~0 F1.

## 0. The real baseline (measured, replaces earlier estimates)

| quantity | value |
|---|---|
| train windows | 306,666 (+38,167 val), 93 labels, 20 streams |
| materialised native-grid hours | **547.3 h** |
| hours actually **reachable in training** (after `MAX_PER_STREAM=20_000`) | **290.0 h** |
| harnet / UK-Biobank pretraining | ~1.7 × 10⁷ h |
| ratio | **~58,000×** |

> Corrects `EVIDENCE_ENGINE_TIER2.md` §0.1, which estimated "≈10³ h assuming ~10 s windows". The
> measured figure is **290 h reachable / 547 h materialised**. The ~4-orders-of-magnitude framing stands.

## 1. Which objectives can consume UNLABELED data (verified in code)

- **A1 masked-latent — self-supervised.** Targets are the frozen tokenizer's own output. **No labels.**
- **A3 primitive grounding — self-supervised.** Targets computed analytically in the collate. **No labels.**
- **A2 config-conditional SupCon — labels ARE the objective.** `losses_repr.py:289`:
  `positives = (labels.unsqueeze(0) == labels.unsqueeze(1))`.

**The blocker is the sampler, not the loss.** `BalancedBatchSampler` is built entirely on `label_id`
(`pretrain_data.py:143,177-193`): it groups by label and excludes labels with <8 windows, so an
unlabeled window **cannot enter a batch at all**. The draw is also *source-balanced*, so a 100× larger
single dataset does not get proportionally more gradient — it only reduces repetition in its existing
slots. (Note this for step 1: naively raising a cap will not shift the gradient mix.)

⇒ Unlabeled data helps (2 of 3 objectives, including the representation driver), but **requires an
unlabeled branch**: a second sampler ignoring `label_id`, a `labels is None` guard, and a mix ratio.
**~1–2 days.** A middle path worth trying: run A2 in **SimCLR mode** on unlabeled data (two augmented
views of one window as positives) — arguably *more* on-thesis, since two rate/rotation configs of the
same motion is exactly the config-conditional framing.

## 2. ⚡ The cheapest win is already on disk: we use ~1% of Capture-24

**Verified:** `data/datasets/capture24/sessions/` holds **13,120 converted sessions, 6.4 GB, 151
subjects** — the complete corpus (full Capture-24 is 3,883 h total / **2,562 h annotated**, CC BY 4.0,
distributed as one 6.4 GB zip; our size matches, so we hold 100% of it).

Two caps throttle it:
1. `capture24/metadata.json`: `"max_hours_per_class": 25` → 10 classes × 25 h = **250 h materialised**.
2. `pretrain_data.py:53`: `MAX_PER_STREAM = 20_000` → **~33 h actually reaches training**, out of 2,562 h.

The cap exists for a real but narrow reason: `build_grids.py:121` concatenates all sessions in RAM, so a
full build OOMs. The fix is a **chunked/appending grid writer**, not less data — grids are already read
`mmap_mode="r"` (`grid_io.py:47`), so training-time memory is not the constraint.

This is **labelled, free-living, wrist, 100 Hz, gravity-present, Axivity AX3 — the same device and wear
protocol as UK Biobank**, and it feeds A1, A2 *and* A3. Unlocking it is a ~8–10× corpus increase for
**zero acquisition cost and zero licence risk**.

## 3. UK Biobank: not on the critical path

**Applications are paused.** Verbatim from the UKB access page: *"Applications are currently paused
whilst necessary changes are made to the UK Biobank Research Analysis Platform… We intend to accept new
applications in late 2026."* The RAP was shut in April 2026 after participant data was found offered for
sale; download exemptions are frozen (*"not accepting any requests for exemptions at this time"*).

Even reopened, the economics are hostile: cost is *not* the barrier (raw `.cwa` field 90001 is Tier 1,
£3,000/3 yr, or £500 student tier) — **egress is**. Individual-level data cannot be downloaded from RAP;
an exemption re-prices to Tier 3 £9,000, needs proof of ~£50k unavoidable cloud cost, and goes to a
quarterly committee. Derived fields don't substitute: field 90004 is **5-second-epoch ENMO**, single
channel, **gravity subtracted**, 20 Hz low-passed — a 6 s window is 1.2 samples, useless for a filterbank
tokenizer whose whole point is per-axis structure plus the gravity DC term.

The only viable pattern is **train inside RAP, export weights** — exactly what OxWearables did for
harnet. **Verdict: ≥12-month side project, different compute story. Do not schedule around it.**

## 4. Ranked acquisition targets

### Tier 0 — open, raw, gravity-present, no application
| # | source | scale | placement | rate | access |
|---|---|---|---|---|---|
| 1 | **Capture-24 (unlock local)** | **2,562 annotated h**, 151 subj | dominant wrist (AX3) | 100 Hz | **already on disk** |
| 2 | **NHANES 2011–2014 PAX80** | **14,693 subj × 7 d ≈ 2.47 M h** | non-dominant wrist | **80 Hz** | open CDC download, **no DUA**; ~2.2 TB |
| 3 | **PAAWS** (Release 2, Jun 2026) | **6,450 h free-living / 237 subj** + 718 h lab @ 21 locations | wrists, waist, thighs, ankles, chest, phone | 80–100 Hz | direct download |
| 4 | **ExtraSensory** | 60 users, 308k labelled 20 s examples | phone + **watch** | 25/40 Hz | free — **use `http://`, HTTPS refuses** |

NHANES is the volume play (~5,000× current corpus, zero paperwork); Inertia-1 ships **MIT-licensed
NHANES preprocessing** we can reuse. PAAWS is the best *labelled* prize — second-by-second video-annotated
free-living at 6 simultaneous placements, essentially unmined.

### Tier 1 — gated, long lead, start the emails now
**HUNT4** (**821,700 unlabelled h**, thigh + lower back, 50 Hz — best reward/effort of anything gated;
`kontakt@hunt.ntnu.no`), **NAKO** (440,000 person-days, **hip** — the placement our wrist-heavy corpus
most lacks), China Kadoorie (22,511 × 7 d, wrist).

### Tier 2 — egocentric IMU **with language** (on-thesis for the evidence engine)
**Nymeria** (264 subj, **300 h**, head + **both wrists**, plus **310,500 narration sentences / 8.64 M
words**) — the single best fit for HALO's language interface. Ego-Exo4D (221 h). Ego4D only if needed
(just 836 of 3,670 h have IMU, documented as uncalibrated).

### Tier 3 — placement diversity, cheap
**RealWorld-HAR** (15 subj × **7 simultaneous positions** — we already have it converted as an *eval* set
at waist only, so the other 6 positions are free), **OxWalk** (wrist+hip at 100 *and* 25 Hz — a free
sampling-rate ablation), RealDisp, HARTH/HAR70+ (already downloaded), WEAR 2024.

### Explicitly NOT usable
NHANES 2003–2006 PAXRAW (1-min **uniaxial counts**), All of Us Fitbit (minute summaries), MESA/NSRR/OAI
(30 s counts), UKB 5 s ENMO. Also **Google SensorFM/SensorLM/LSM-2 and Apple AHMS are not public — and
none of them see a waveform** (all train on per-minute engineered summaries). That is a *differentiation
point for our paper*, not a scale gap to apologise for.
⚠️ **Inertia-1's headline "18.2 M hours" is 86% UK Biobank served as 0.2 Hz gravity-removed
single-channel ENMO.** Only its NHANES portion (~2.47 M h) is real triaxial. Expect it cited against us;
the rebuttal is that most of it is not IMU data.

## 5. Synthetic IMU from mocap: NOT a scale multiplier

The arithmetic ends the argument before quality does. **AMASS ≈ 40 h. HumanML3D = 28.59 h. Motion-X ≈
144 h.** The *entire* text-motion mocap universe is ~150–200 h — **less than our corpus reaches once
Capture-24 is unlocked.** UniMTS pretrained on HumanML3D alone (28.6 h). No large simulated corpus is
downloadable (AMASS licence forbids redistributing derivatives), so we'd burn compute regenerating it.

The decisive negative result comes from Doherty/Yuan's own group — Darwish et al., *"Motion Capture is
Not the Target Domain"* (arXiv 2602.11064), using the exact UniMTS framework: purely synthetic
pretraining scores **0.246–0.274 zero-shot macro-F1 vs 0.307 for real**, and scaling synthetic **8×
makes it worse** (0.212). Failure modes are worst exactly where we are strongest: mocap contains **no
sedentary behaviour at all** — no sleep, no desk sitting, no non-wear — i.e. free-living, where HALO
currently ranks #1, is structurally absent.

**Where it IS useful:** synthetic is the only source with ground-truth **placement, orientation and
gravity by construction** — precisely our acquisition-config axis. Use it as a **controlled probe /
config-conditioning augmentation, never as corpus.**
Two correctness notes if we ever generate it: (a) sample the virtual sensor from the SMPL **surface mesh
with placement offsets**, not the joint centre (UniMTS and IMUGPT both sample joints — an unforced
error); (b) **the field has two incompatible gravity conventions** — TransPose/DIP/PIP are gravity-free,
UniMTS-via-IMUSim is gravity-inclusive. Since UniMTS is one of our baselines, this is a **live
correctness question about how we interpret it**.

## 6. Recommended sequence

**Step 1 — unlock what we already own (~1 week, zero risk). DO THIS FIRST.**
1. Make `build_grids` write **incrementally** instead of `np.concatenate` (`build_grids.py:121`). 1–2 d.
2. Raise `max_hours_per_class` in `capture24/metadata.json`; make `MAX_PER_STREAM` per-*dataset* and
   source-balance-aware (a naive raise won't shift the gradient mix — see §1). 1 d.
3. Add the 6 unused RealWorld placements + HARTH/HAR70+ for placement diversity. 1 d.

⇒ **290 h → ~2,800 h (≈10×)**, all labelled, all three objectives fed, no new licences. Re-tune the step
budget (30k steps was set for a 96k-window corpus and is already flagged stale) and **re-measure
retrieval purity. This is the experiment that tells us whether data moves the 0.68 ceiling** — run it
before anything below.

**Step 2 — unlabeled branch + NHANES (~3–4 weeks).** Add the unlabeled sampler + `labels=None` guard
(+ optional SimCLR-mode A2). Convert a **subject subset** of NHANES (~500 of 14,693 ≈ 84,000 h) rather
than the full 2.2 TB — already 30× Step 1 and settles the scaling question cheaply. Verify static
|acc| ≈ 1 g. ExtraSensory in parallel (labelled, so it also feeds A2).

**Step 3 — long-lead, start now, payoff in months.** Email HUNT4; apply to NAKO; pull PAAWS Release 2 and
Nymeria. **Do not** open a UKB application until it reopens, and scope it as "train in RAP, export
weights" if we ever do.

## 7. Corroboration + licence warnings (second independent survey)

A second survey agreed on every major point (Capture-24 held in full; NHANES as the volume play; UKB
paused verbatim; HUNT4 by email; PAAWS; ExtraSensory HTTP-only). Extra verified detail:

- **NHANES PAX80 file-level:** 2011-12 `PAX80_G` = 6,917 participants (~1.04 TB compressed);
  2013-14 `PAX80_H` = 7,776 (~1.17 TB). ActiGraph GT3X+, non-dominant wrist (~99% compliance), 80 Hz,
  units **g**, range ±6.006, **static gravity retained**; one `.tar.bz2` per participant on CDC FTP,
  scriptable. Ages 3–5 restricted to the NCHS RDC. Bonus children's sample: NNYFS 2012 `Y_PAX80`.
  The two surveys' totals differ (~1–2 M h vs ~2.47 M h) — order-of-magnitude consistent either way.
- **Ignore** NHANES `PAXMIN`/`PAXHR`/`PAXDAY` (MIMS minute/hour/day summaries) — go straight to PAX80.
- **harnet has larger siblings:** ssl-wearables ships **harnet10 and harnet30**, not just harnet5.
  Free, no UKB application. Worth adding as baselines and/or as retrieval-control backbones.
- **HARTH** (UCI 779) is the open, labelled window into the HUNT4 world: 22 subj free-living, dual AX3
  thigh + lower back, 50 Hz, **units g**, frame-by-frame video annotation. Direct download.
- **MotionSense gravity is reconstructible** — it ships gravity-removed `userAcceleration` *plus*
  separate `gravity.xyz`; total = sum. Relevant since it is one of our eval sets.
- **OxWalk** records wrist+hip at **100 Hz and 25 Hz simultaneously** — a free, controlled
  rate-invariance ablation for our core claim. 290 MB, CC BY 4.0.
- **Nymeria** detail: 264 participants, 300 h, Aria (2 IMUs) + **miniAria wristbands** + XSens 17-IMU
  full-body suit, all in one coordinate frame, with **230 h carrying natural-language descriptions**.

⚠️ **Licence hazards — check before any corpus release:**
- **WEAR is CC BY-NC-SA**: non-commercial **and** share-alike.
- **MMAct**'s licence is **revocable on breach** — risky to build on.
- **ssl-wearables weights** are academic-research-only, not commercial.
- **Ego4D IMU is documented as defective**: uncalibrated with "likely measurement bias", missing
  samples, non-monotonic/out-of-range timestamps, IMU absent from many components, ambiguous units.
  **Skip, or only after heavy QC.**

### 7b. Additional verified detail + what was NOT surveyed

- **SHL ships both conventions**: `Accelerometer` in m/s² (**gravity included**) *and* `Linear
  acceleration` (**gravity removed**) as separate channels, 4 positions (hand/torso/hips/bag), 2,812 h
  but only **3 subjects**. We dropped it for subject diversity — but the raw+linear pairing is a **free
  gravity ablation** if we ever want one.
- **Ego-Exo4D**: 740 wearers / 1,286 h, Aria 800 Hz + 1000 Hz IMUs. A prior **Ego4D licence does not
  carry over** — it must be signed separately.
- **Opportunity**: only 4 users, but 7 IMUs + 12 body-worn 3-D accelerometers, 4-level annotation.
  An eval set for fine-grained gestures, not pretraining fuel.
- **Useful index to close gaps cheaply**: [Awesome-IMU-Sensing](https://github.com/rh20624/Awesome-IMU-Sensing),
  a maintained catalogue (it surfaced PAAWS and **OctoNet** — 41 subjects, 10+ modalities, 2025).

⚠️ **This survey is NOT exhaustive.** The web-search budget was exhausted; the following were assigned
but **never researched** — do not assume they were rejected: MESA/NSRR actigraphy (raw vs epoch is the
key open question), All of Us Fitbit granularity, Whitehall II, Osteoarthritis Initiative, China
Kadoorie, Netherlands Lifelines, Pelotas birth cohorts, Fenland/EPIC-Norfolk, ALSPAC, Canadian Health
Measures Survey, DOMINO (2022), Wear-ME, DHARMA, Skoda, Daphnet, LARa, SisFall/MobiAct family, and
HuggingFace-hosted IMU corpora. Prior expectation: most large cohort studies release **epoch/count
summaries, not waveform**, which would disqualify them regardless of access friction — but that is a
prior, not a finding. Worth one focused follow-up session; **do not block Step 1 on it.**

## 8. Uncertainty flags
- UKB reopening date / incident cause are secondary sources; **the pause itself is confirmed verbatim**.
- PAAWS licence unconfirmed (403); gravity convention for PAAWS, Nymeria/Aria, DOMINO is **inferred from
  hardware, not documented** — probe a static clip before trusting.
- NAKO / China Kadoorie cost and raw-export policy unpublished.
- arXiv 2602.11064 and 2607.06617 are 2026 preprints surfaced by search — read directly before citing.
  Inertia-1 overlaps our tokenizer redesign and ablates placement/rate/window; worth a close read.
- Whether UniMTS's pretraining signal actually carries a DC gravity component is **unverified** and
  changes how we interpret it as a baseline.
