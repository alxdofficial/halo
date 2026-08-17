# Dataset expansion under the rehabilitation-tracking framing (implemented 2026-08-12)

> **Status:** ten sources were acquired, converted, gridded, and audited. Six are in the expanded
> Phase-A recipe: DSADS, Forth-TRACE, Opportunity, REALDISP, MM-Fit, and PHYTMO. KneE-PAD is
> converted but sits in `OPTIONAL_PHASE_A_DATASETS` — only 4.7% of its trials reach a six-second
> window — so it is an explicit opt-in stress study, not part of the default corpus (§6, §10).
> Monipar, SPAR, and Upper-Limb-Use remain held out and are part of the default evaluation roster.
> [`../design/DATA_SCALING_PLAN.md`](../design/DATA_SCALING_PLAN.md) owns the experiment rule and
> preserves the original 12-source corpus as `--corpus matched`.

## 1. Why

Phase B commits to a rehabilitation-tracking motivation (`../design/PHASE_B_TRAINING_INTENT.md` §2).
The current roster was chosen for zero-shot cross-dataset comparison and cannot measure that story.
Three failures were measured on 2026-08-08 rather than assumed:

| measured | consequence |
|---|---|
| **35 of 93** labels exist on exactly one stream; **0 of 688** enrolled executions in a 24-step probe came from a paired second stream | cross-configuration enrollment is effectively untrainable |
| Real same-subject enrollment was feasible for only **34 of 67** training labels in the 2026-08-08 probe | **Resolved in the trainer:** `same_subject_enrollment` now samples only feasible labels and uses distinct executions from the same real subject; augmentation may add a shared persona but never substitutes a different person. Coverage remains limited to those feasible labels. |
| Dev eval roster (`motionsense`, `realworld`, `shoaib`) has **0 unseen concepts** (21/21 already in the 93-label vocab); same-subject support ceiling is **k≤2** on motionsense (24 subj), **k≤2** on realworld (3 subj), and **k=0** — no curve at all — on shoaib, inclusivehar, ut_complex | the capability the paper claims cannot be measured on the development roster |

`usc_had` (k≤4, 14 subjects, 5 unseen concepts) is the only legacy set with a real same-subject
curve, and it sits behind the test seal.

**Verification convention below:** ✔ = source fetched directly on 2026-08-08. *(paper)* = taken from
the dataset's own documentation, not independently checked.

## 2. Training — already local, zero download

Highest return per unit of work: converter changes only.

| Dataset | Action | Currently ingested | Available in raw | Sensors / rate | Why |
|---|---|---|---|---|---|
| `pamap2` | ingest extra placements | `watch_wrist`, 3,186 win ✔ | hand + **chest** + **ankle** *(paper)* | acc+gyro, 100 Hz | simultaneous multi-placement for labels already in vocab |
| `mhealth` | ingest extra placements | `watch_wrist`, 1,104 win ✔ | **chest** + R wrist + **L ankle** *(paper)* | acc+gyro, 50 Hz | same |
| `harmes` | ingest second wrist | `watch_wrist`, 18,576 win ✔ | dual-wrist *(paper)* | acc+gyro, 50 Hz | bilateral pair; harmes supplies many fine-grained hand labels |
| `extrasensory` | **policy decision** — see §6 | gridded, **excluded from TRAIN** ✔ | 3 streams: hand 39,502 · pocket 62,691 · wrist 556,744 ✔ | acc only, 50 Hz | 659k free-living windows across 3 placements |
| `hmog` | **policy decision** — see §6 | gridded, excluded ✔ | 191,535 win, phone_hand ✔ | acc+gyro, 100 Hz | in-hand phone placement we barely have |
| `nhanes` | **policy decision** — see §6 | gridded, excluded ✔ | 115,179 win, watch_wrist ✔ | acc only, 80 Hz | scale + demographic breadth |
| `harth` | investigate | converted, **no native grid** ✔ | unknown | — | already converted; free if the grid builds |
| `hapt` | **exclude** | 2,112 win ✔ | — | — | same 30 people as `uci_har` — known leakage trap |
| `mobiact` | **broken** | **0 windows** ✔ | — | — | grid is empty; converter needs repair first |

## 3. Training — new acquisitions

| # | Dataset | Source | Quantity | Placements | Sensors / rate | License | Why | Ver. |
|---|---|---|---|---|---|---|---|---|
| **T1** | **REALDISP** | [UCI #305](https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset), 2.5 GB | 17 subj × 33 activities × ~20 reps | **9** (calves, thighs, back, lower+upper arms) | acc+gyro+mag+quat, 50 Hz | open | Ships **ideal / self-placement / induced-displacement** conditions — the input-side thesis as ground truth instead of synthetic augmentation. Physio-adjacent vocabulary (trunk twist, lateral/frontal arm elevation, shoulder rotation, forward stretching). Fixes all three gaps at once | ✔ |
| **T2** | **DSADS** | UCI Daily & Sports Activities | 8 subj × 19 activities × 60 segments | 5 (torso, arms, legs) | acc+gyro+mag, 25 Hz | open | multi-config **and** many executions per subject | *(paper)* |
| **T3** | **FORTH-TRACE** | [GitHub](https://github.com/spl-icsforth/FORTH_TRACE_DATASET) | 15 subj × 16 activities | 5 — **L wrist, R wrist**, torso, R thigh, L ankle | acc+gyro+mag | open | bilateral wrist pair; 9 postural **transitions**, which the corpus is thin on | ✔ |
| **T4** | **CAPPIMU** | [GitHub](https://github.com/quiqi/CAPPIMU) → Google Drive | 30 subj × 21 activities | **9** (head, arms, wrists, chest, pocket, shin) | acc+gyro+mag, 50 Hz | check | household vocabulary (slicing, ironing, folding, window cleaning) at 9 placements; overlaps harmes/xrf_v2 labels | ✔ |
| **T5** | **Opportunity** | UCI | 4 subj; drill runs = **20 reps × 17 gestures** | 7 IMU + 12 accel | acc+gyro, 30 Hz | open | textbook repeat-execution structure; only 4 subjects | *(paper)* |
| **T6** | **Gym Gesture IMU** | [IEEE DataPort](https://ieee-dataport.org/documents/gym-gesture-classification-using-imu-sensor-dataset) | 10 athletes × 5 exercises × 30 reps = 1,500 reps | 1 (wrist) | acc+gyro, 100 Hz | open, free login | **explicit per-repetition ids**; clean arm-exercise vocabulary | *(listing)* |

## 4. Evaluation — new acquisitions

| # | Dataset | Source | Quantity | Device | Sensors / rate | License | Role | Ver. |
|---|---|---|---|---|---|---|---|---|
| **E1** | **SPAR** | [GitHub](https://github.com/dmbee/SPAR-dataset) | 20 subj × 7 shoulder physio exercises × 2 sides × **20 reps** (280 CSVs) | **Apple Watch 2/3** | acc (g) + gyro (rad/s), 50 Hz | GPL-3.0 | **Primary enrollment testbed.** k=8 same-subject curve feasible; 7 genuinely unseen concepts; **L/R is a free cross-placement axis** | ✔ 280 files |
| **E2** | **SPARS9x** | [IEEE DataPort](https://ieee-dataport.org/open-access/shoulder-physiotherapy-activity-recognition-9-axis-dataset), DOI 10.21227/cx5v-vw46 | 20 subj × 10 exercises × 20 reps/shoulder + **~3 h unlabeled ADL per subject** | **Huawei Watch 2** | 9-axis + HR, 50 Hz | Open Access, free IEEE login | E1 plus a real **truth-absent / open-set population** for the confidence stage; includes isometric holds | ✔ |
| **E3** | **Monipar** | [Zenodo 8104853](https://zenodo.org/records/8104853), 35 MB | **21 Parkinson's patients + 7 controls** × 8 exercises, **repeated weekly** | off-the-shelf smartwatch | **acc only**, 50 Hz | check | the only **across-session** structure found: enrol week 1, recognize week N. MDS-UPDRS scores; remote vs supervised subgroups | ✔ |
| **E4** | **MM-Fit** | [mmfit.github.io](https://mmfit.github.io/), 800+ min | 10 exercises + null, ~20 sessions, rep-annotated | **2 smartwatches + 2 phones + earbud**, time-synced | acc+gyro (+mag on phones), 90–500 Hz | check | **cross-configuration arm** — the same repetition on 5 devices at once | ✔ |
| **E5** | **PHYTMO** | [Zenodo 6319979](https://zenodo.org/records/6319979), 3.7 GB | 30 subj **aged 20–70** × (6 exercises + 3 gait variants) × 2 series × ≥8 reps | 4 NGIMU on limbs | acc+gyro+mag, 100 Hz | CC-BY | age spread + **correct/incorrect execution labels**; research IMUs, not consumer | ✔ |
| **E6** | **KneE-PAD** | [Zenodo 12112951](https://zenodo.org/records/12112951) | **31 knee-pathology patients** × 3 exercises + 2 wrong variants each × ~10 trials (2,086 files) | 8 Delsys Trigno on leg muscles | acc+gyro, 148.15 Hz | CC-BY | real pathology, unsupervised in-clinic; placement does not match phone/watch deployment | ✔ |
| **E7** | **upper-limb-use** | [GitHub](https://github.com/biorehab/upper-limb-use-assessment), ~172 MB | 10 controls + **5 hemiparetic** × 15 ADLs | 2 wrist bands (affected / unaffected) | acc+gyro+mag, 50 Hz | check | stroke patients, bilateral; labels are functional-use, not exercise identity | ✔ |

**Implemented split:** the established development datasets remain MotionSense, RealWorld, and
Shoaib. The sealed Phase-B test roster is InclusiveHAR, USC-HAD, TNDA-HAR, UT-Complex, Monipar,
SPAR, and Upper-Limb-Use. MM-Fit, PHYTMO, and KneE-PAD were promoted to expanded training and are no
longer valid unseen-dataset tests for that arm.

## 5. Considered and rejected

| Dataset | Reason |
|---|---|
| GAITEX (Zenodo 15729055, 17.7 GB) | **CC-BY-NC-ND** — the no-derivatives clause blocks a derived corpus |
| Comprehensive IMU Dataset (17 IMUs, Figshare) | non-commercial licence; signals in **world frame**, conflicts with the device-frame + gravity-DC contract |
| Meshed IMU Garment (396 IMUs) | impractical |
| REHAB24-6 (Zenodo 13305826) | video + mocap only, **no IMU** |
| UCI Physical Therapy Exercises (#730) | 5 subjects |
| SIMUL (32 subj, 6 IMUs) | walking only |
| FDA smartphone-gait | gait only — but note 5 phone *orientations* × 2 placements; a possible controlled-orientation testbed for `MOTIVATION.md` §3 |
| motionsense / realworld / shoaib / usc_had / tnda_har / ut_complex / inclusivehar | existing eval sets — **never train on these** |

## 6. Resolution of the frozen-corpus experimental rule

`../design/DATA_SCALING_PLAN.md` deliberately keeps ExtraSensory, NHANES and H-MOG **out** of the
primary corpus so that the technique comparison argues method-over-data. They are already converted
and gridded; their absence is policy, not an oversight.

The implementation preserves both coherent options as named recipes:

1. `pretrain --corpus matched` keeps the frozen 12-source technique arm.
2. `pretrain --corpus expanded` selects the 18-source design-of-record. A comparison against this arm
   is corpus-matched only after each retrained baseline uses the same 18 sources. KneE-PAD is an
   explicit short-window/clinical-stress ablation because only 4.7% of its trials reach six seconds.

The resolved dataset tuple is serialized in every checkpoint. Do not mix results across the two
recipes in one headline table without identifying the corpus.

## 7. Constraints for whoever implements this

- **Channel contract.** Canonical is 6-channel `(acc_x,y,z, gyro_x,y,z)`, accel in **g**, gyro rad/s
  or deg/s converted, **device frame, gravity present**. Magnetometer and quaternion channels are
  dropped. Accel-only sources run through the channel mask as 3-channel streams.
- **Rate floor.** The filterbank analyses 0.3–15 Hz with a 0.9 Nyquist margin. DSADS at 25 Hz yields
  ~11 Hz usable, which is fine (`wisdm` is already 20 Hz).
- **Every training addition forces a Phase-A retrain and a memory-bank rebuild.** Batch them into one
  cycle rather than one dataset at a time.
- **Evaluation additions never touch Phase A** and can be onboarded concurrently.
- **Repetition-level vs session-level enrollment.** SPAR/SPARS9x give many repetitions *within one
  session*; Monipar gives sessions *across weeks*. These are different claims, and
  `eval_enrollment.py` already distinguishes them via `execution_ids` granularity and the
  `window_level_ids` gate. Decide before any k-curve is drawn.
- **Open decision:** `extrasensory` cannot be both a training source and a held-out test set
  (task #36 proposed the latter).

## 8. Disk estimate

Measured on 2026-08-08: `data/` is **116 GB**; native grids total **32.7 GB** across 2,728,738
windows. Grid cost follows a clean rule, verified against every existing stream:

```
grid bytes per window  ≈  window_seconds × rate_hz × 6 channels × 4 bytes
   100 Hz → 14.4 kB    80 Hz → 11.5 kB    50 Hz → 7.2 kB    25 Hz → 3.6 kB    20 Hz → 2.9 kB
```

All six canonical channel slots are stored regardless of the mask, so accel-only sources cost the
same per window as six-channel ones.

| Item | Download | Converted | Grids | Total |
|---|---:|---:|---:|---:|
| **§2 re-conversions** (pamap2, mhealth, harmes extra placements) | 0 | ~2.5 GB | ~0.3 GB | **~3 GB** |
| §2 `extrasensory` / `hmog` / `nhanes` promotion | 0 | 0 | 0 | **0** (already gridded) |
| T1 REALDISP | 2.5 GB | ~3 GB | ~0.4 GB | **~6 GB** |
| T2 DSADS | ~1 GB | ~0.4 GB | ~0.15 GB | **~1.5 GB** |
| T3 FORTH-TRACE | ~0.5 GB | ~0.3 GB | ~0.2 GB | **~1 GB** |
| T4 CAPPIMU | ~7 GB | ~3 GB | ~0.8 GB | **~11 GB** |
| T5 Opportunity | ~0.3 GB | ~0.4 GB | ~0.1 GB | **~1 GB** |
| T6 Gym Gesture | <0.1 GB | <0.1 GB | <0.1 GB | **~0.2 GB** |
| **Training subtotal** | | | | **~24 GB** |
| E1 SPAR | <0.1 GB | <0.1 GB | ~0.03 GB | **~0.1 GB** |
| E2 SPARS9x | ~6.5 GB | ~2 GB | ~0.3 GB | **~9 GB** |
| E3 Monipar | 0.04 GB | ~0.1 GB | <0.05 GB | **~0.2 GB** |
| E4 MM-Fit (inertial only, **not** the 39 GB video) | ~2.5 GB | ~1 GB | ~0.6 GB | **~4 GB** |
| E5 PHYTMO | 3.7 GB | ~1 GB | ~0.35 GB | **~5 GB** |
| E6 KneE-PAD | ~2 GB | ~0.8 GB | ~0.1 GB | **~3 GB** |
| E7 upper-limb-use | 0.2 GB | ~0.2 GB | ~0.05 GB | **~0.5 GB** |
| **Evaluation subtotal** | | | | **~22 GB** |
| Memory-bank rebuild (transient: keep old + new) | | | | **+2.3 GB** |
| **Everything** | | | | **≈ 48 GB** |
| **+30% headroom for conversion scratch** | | | | **≈ 62 GB** |

**Minimal path** — §2 re-conversions + T1 + E1 + E3, i.e. the items that fix the measured failures
with the least acquisition: **≈ 10 GB**.

The memory bank stays roughly constant at ~2.3 GB regardless of corpus growth: `ARCHIVE_BUDGET_WINDOWS`
caps it at 250,000 source windows and the current build is 248,351, so the balancing policy — which
has never yet bound — takes over as soon as anything is added.

Available on this machine at the time of writing: **1.2 TB free of 1.9 TB**. Disk is not a constraint
for any option here.

## 8b. Convention: dataset fitness is per-test, not global

A dataset is kept if it is generally good. If it turns out to be a poor fit for one *specific* test,
nothing is forced onto it — the limitation is documented, that cell is reported as unsupported, and
the dataset keeps serving the tests it does fit.

This is already the code's behaviour and should stay that way:
`build_paired_enrollment_plans` returns `status: above_paired_support_ceiling` (or
`unverified_window_level_execution_ids`) for cells it cannot honestly support, rather than
substituting windows, redefining an execution, or shrinking the cohort to make a number appear.

The worked example is SPAR (Section 9): it cannot support a same-subject enrollment curve at all,
and it remains a valuable evaluation set for unseen concepts on a consumer watch, for cross-subject
enrollment at k=0..8, and for left/right placement transfer. One failed test is not a verdict on the
dataset.

Corollary for writing: a table cell that reads "unsupported" is a result. Do not fill it, and do not
drop the dataset to make the table look complete.

## 9. Acquisition log — measured on the real downloads (2026-08-08)

All ten sources were downloaded, unpacked, and inspected. **No converters written yet; nothing is
integrated into training.** Raw archives live in `data/datasets/<name>/downloads/`.

| Dataset | On disk | Structure verified | Units verified |
|---|---:|---|---|
| realdisp | 2.5 GB → 6.6 GB unpacked | 46 `.log`: 17 ideal + 17 self + 12 mutual (3 subj × 4 configs); 120 cols = ts_s, ts_us, 9 sensors × 13 ch [acc,gyro,mag,quat], label col 120 | **50.000 Hz exactly**; accel **m/s², gravity present** (median ‖a‖ 9.90 LLA / 10.01 RLA / 10.53 back → ÷9.81 for g); gyro already **rad/s** (p99 ≈ 4–7) |
| spar | 1.7 GB (after deleting 6.9 GB of unused `lbp_csv` derived features) | **280 CSVs = 20 subjects (S1–S20) × 7 exercises × L/R**; columns `ax,ay,az,wx,wy,wz` | 50 Hz, accel **g**, gyro **rad/s** — matches our contract with no conversion |
| monipar | 34 MB | `.mat` cell arrays **subject × week**: supervised 6×8 (46 filled), remote 15×8 (72), control 7×9 (56) = **174 trials / 28 subjects / 20.3 h**; 5 cols = ts_ms, acc x/y/z, label 0–8; all 9 labels in every trial | 50 Hz, accel **m/s²**, **no gyro** (3-channel stream) |
| phytmo | 3.5 GB | archive not yet extracted | — |
| dsads | 163 MB → 425 MB | **9,120 `.txt`** = 19 activities × 8 subjects × 60 segments | — |
| opportunity | 293 MB → 860 MB | 24 `.dat` (S1–S4 × ADL1–5 + Drill) | — |
| kneepad | 277 MB → 1.5 GB | `dataset/Subject_NN/<class 0–8>/Trial_N/{emg,imu}.npy`, 4,172 arrays | — |
| mmfit | 1.7 GB → 3.6 GB | `mm-fit/w00…w20/`, 281 `.npy` named `w19_eb_l_acc.npy`, `w19_sp_r_gyr.npy` … (device × modality per workout) | — |
| forth_trace | 422 MB | 75 CSVs = 15 participants × 5 devices (`partX/partXdevN.csv`) | — |
| upper_limb_use | 271 MB | `control/data/{left,right}.csv`, `patient/data/{affected,unaffected}.csv` | — |

### ⚠ Correction to §4 — SPAR does **not** give a k=8 same-subject enrollment curve

Measured over all 280 files: **3.39 h total**, median file **42 s**, i.e. **~2.3 s per repetition**,
yielding only **1,898 six-second non-overlapping windows** in the entire dataset (median 7 per
subject/exercise/side; per subject 562 s across all 14 files).

All 20 repetitions of one exercise sit inside a **single continuous ~42 s bout**. So:

- treating a repetition as an "execution" puts enrollment and query **~2 s apart** — this is the
  adjacent-window leakage that `eval_enrollment`'s `window_level_ids` gate exists to refuse;
- treating a file as an execution gives **k = 1** per (subject, exercise, side), with left and right
  shoulder being different concepts rather than repeats.

SPAR remains valuable for what it uniquely offers — 7 genuinely unseen physiotherapy concepts on a
**consumer Apple Watch**, 20 subjects, and a free L/R cross-placement axis — but it measures
*within-session* binding, not enrollment that survives a session change. §4's "k=8 becomes trivially
feasible" was wrong and is withdrawn.

**Monipar is therefore the only verified source of genuine across-session enrollment**: up to 8–9
independent weekly sessions per subject, one week apart, on an off-the-shelf smartwatch, with real
patients. Its cost is that it is accelerometer-only. It should be promoted from a secondary arm to
the primary same-subject enrollment testbed.

This is likely a general property of exercise datasets rather than a SPAR defect: one bout per
subject per exercise is the standard protocol, and "repetitions" are within-bout. PHYTMO's two
correct series per exercise is the next-best structure and gives k = 1 across series.

> **⚠ SUPERSEDED IN PART by [`DATASET_EXPANSION_AUDIT_2026-08-11.md`](DATASET_EXPANSION_AUDIT_2026-08-11.md).**
> Section 10's window counts are pre-fix, and its `exec/(subj,label)` column counts contiguous label
> blocks rather than independent recordings — an overcount of up to 10.2x (opportunity), which makes
> the enrollment picture it paints wrong for four of the ten sources. Four converters were rewritten
> and their grids rebuilt on 2026-08-11. Read the audit first.

## 10. Conversion + verification results (2026-08-09)

All ten downloaded sources are **converted, gridded and verified**. Six are in
`EXPANDED_PHASE_A_TRAIN_DATASETS`; `dsads` and `opportunity` retain `role="stress"` to describe their
off-deployment placements, but the default grid builder and expanded trainer include them
deliberately. KneE-PAD is an explicit opt-in stress study. The remaining three are held-out
evaluation datasets.

Measured off the built `native` grids by `data/scripts/debug/sweep_new_datasets.py`:

| dataset | streams | windows | h/stream | labels | subj | rate | quiescent \|acc\| (g) | exec/(subj,label) med/max | share ≥2 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| spar | 2 | 1,898 | 1.58 | 7 | 20 | 50 | 1.005–1.009 | 1/1 | 0.00 |
| monipar | 1 | 12,079 | 20.13 | 9 | 28 | 50 | 0.974 | **7/9** | **1.00** |
| realdisp | 9 | 50,886 | 9.42 | 33 | 17 | 50 | 0.996–1.006 | 2/12 | 0.82 |
| forth_trace | 5 | 10,410 | 3.47 | 11 | 13 | 51.2 | 0.993–1.021 | 1/7 | 0.28 |
| dsads | 5 | 35,480 | 11.83 | 19 | 8 | 25 | 0.997–1.002 | 2/15 | 0.56 |
| opportunity | 5 | 10,835 | 3.61 | 4 | 4 | 30 | 1.000–1.005 | 30/234 | 1.00 |
| phytmo | 8 | 24,684 | 5.14 | 14 | 30 | 100 | 0.966–1.014 | 2/2 | 0.99 |
| mmfit | 4 | 6,280 | 2.62 | 10 | 21 workouts (10 people) | 100 | 1.006–1.022 | 3/4 | 0.99 |
| kneepad | 8 | 1,344 | 0.28 | 5 | 28 | 148.15 | 1.010–1.034 | 1/2 | 0.27 |
| upper_limb_use | 4 | 1,124 | 0.47 | 14 | 10 | 50 | 1.000–1.071 | 1/2 | 0.07 |

Every stream: accelerometer canonicalizes to **≈1 g at rest** with gravity present, gyroscope p99.9
sits in the rad/s range, no non-finite samples, no flat real channels, no data in padded slots, and
gravity alignment + the training tokenizer + the Phase-A augmenter all return finite output.

**Cross-placement simultaneity is exact.** For all seven co-located sources
(realdisp, forth_trace, dsads, opportunity, phytmo, mmfit, kneepad) window *i* of every placement
carries the same label and subject as window *i* of every other: agreement **1.0000**. That is what
makes a support example enrolled on one placement and a query drawn from another a genuine
change-of-configuration pair.

### The enrollment picture, now measured rather than assumed

Section 9 withdrew SPAR's k=8 claim. The replacement is not a guess:

- **monipar is confirmed as the across-session testbed** — median **7** independent executions per
  (subject, exercise), max 9, and **100%** of (subject, exercise) pairs have ≥2. Those executions
  are weekly visits, so k up to ~6 is a real enrol-now-recognize-later curve on real patients.
- **realdisp** gives median 2 / max 12 (ideal, self, mutual4–7 are separate recordings).
- **mmfit** gives median 3 / max 4 (sets within a workout) *and* the 4-device cross-config axis.
- **phytmo** gives exactly 2 (the two series), i.e. k=1.
- **spar, kneepad, upper_limb_use, forth_trace** give median 1 — **unsupported** for a same-subject
  curve, which is a result, not a gap to fill (§8b).

### Defects found in the sources, and what was done

| source | defect (measured) | response |
|---|---|---|
| realdisp | `subject6_self`, `subject13_self`, `subject15_mutual4` are **placeholders** — all 117 sensor columns identically 0.0 on every row, with complete label tracks | logs skipped; those subjects lose their ideal-vs-self pair. Caught by the grid sweep at 6.6% zero windows per stream, not by the converter |
| realdisp | every other log contains one **backwards clock jump** of 1300–2100 s | rate read from the median step, not the span; blocks straddling a reset dropped |
| monipar | README says 50 Hz; the real clock is **bimodal 49.87–52.85 Hz** | each trial anti-alias resampled to a true 50 Hz — a 5.7% shift would move the tremor bands this dataset exists to measure |
| forth_trace | `part4` remains at **0.147** node-to-node label agreement; part8's earlier 0.13 was caused by a 1,828 s dropout | part8 retained after gap splitting (0.974); only part4 excluded |
| forth_trace | gyroscope is **deg/s**, undocumented | converted to rad/s, with a guard that fails if a future release ships rad/s |
| dsads | **8.1%** of the distributed 5-second segment joins are genuine discontinuities (median join is indistinguishable from an interior step, so the ordering IS temporal) | sessions split at those joins; no window straddles a gap |
| dsads | UCI's block names (T/RA/LA/RL/LL) read as torso/arms/legs; the **paper** puts the units on the chest, both **wrists** and the outer sides of both **knees** | placement text corrected — HALO conditions on it, and "the right arm" would have mis-described the only two watch-like placements here |
| opportunity | gyroscope unit documented as "unknown" | resolved by measurement (forearm p99 during labelled walking = 3.15 rad/s after ÷1000; as deg/s that is physically impossible) |
| opportunity | gesture instances are **median 2.6 s** against a 6 s window | Locomotion track used; the gesture cell reported unsupported |
| phytmo | 8 of 4,520 CSVs contain all-NaN dropped-packet rows, all from subject C02's left-knee trials; the left-shin unit is **40% missing** | isolated single-sample holes interpolated; 4 trials with excessive dropout skipped rather than interpolated |
| kneepad | only **98 of 2,086** trials (4.7%) reach a 6 s window; sensors are on muscle bellies | converted and `role="stress"`, but removed from default Phase A; explicit opt-in only |

### Not acquired

`T4 CAPPIMU`, `T6 Gym Gesture IMU` and `E2 SPARS9x` from §3–§4 were never downloaded. T4 is behind a
Google Drive link and T6/E2 behind an IEEE DataPort login. They are also the three that matter least
right now: T4 and T6 are *training* additions, which §6's frozen-corpus decision has to settle first,
and E2 duplicates SPAR's within-session structure — the thing §9 showed does not support enrollment.

### Reference material

`references/datasets/<name>/` now holds `citation.json` + `SOURCE.txt` for all ten, with `paper.pdf`
for eight. REALDISP has no paper PDF (the UbiComp'12 paper is paywalled with no open version) but
ships `documentation.pdf` — the release's own Dataset Manual, which is the document the converter was
actually verified against. Monipar, FORTH-TRACE and KneE-PAD also keep their release documentation
alongside the paper. Every `citation.json` `notes` field records what was checked against the paper
and every discrepancy found.
