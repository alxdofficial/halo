# Dataset roles for movement monitoring

> **Verified inventory, 2026-09-02.** Dataset role is checkpoint-dependent. A source consumed by a
> HALO checkpoint during representation training may be used for development but not to claim
> unseen-dataset generalization for that checkpoint.

## 1. Readiness terms

- **Ready:** the local or inspected source preserves the timeline, physical units, subject/session
  identity, and annotations required by the task.
- **Reconstruct:** the source is suitable, but the existing converted sessions have discarded
  background, gaps, or original recording order. Build an application-specific source loader.
- **Conditional:** suitability depends on checkpoint provenance, a unit decision, or a complete
  release audit.
- **Blocked:** a known data or converter defect makes the current representation invalid.
- **Oracle only:** useful external measurements exist, but the files do not match HALO's raw
  accelerometer/gyroscope interface.

Publication suitability and file compatibility are separate checks. A paper can describe exactly
the desired experiment while the released files omit timestamps, physical units, or raw IMU.

## 2. Core evaluation matrix

The active protocols use a small task-specific set. Sources are selected by annotation contract,
not by whichever one gives the highest score, and each source is reported separately.

| dataset | task coverage | reason for inclusion | current gate |
|---|---|---|---|
| **C-MHAD** | Task 1 evaluation; Task 3 evaluation | synchronized video, continuous background, event intervals, and wrist/waist six-axis views | native-clock adapter and frozen manifests verified |
| **OpenPack** | Task 1 evaluation; Task 3 train/development | occupational wrist IMU with bounded operations and complete timelines | adapter and frozen manifests verified; its role differs by task |
| **WEAR** | Task 3 evaluation | independent outdoor cohort with natural continuous activity/NULL | fixed-rate adapter verified; one missing left-arm stream remains explicit |
| **OCA** | Task 3 evaluation | repeated industrial assembly in long recordings with NULL | native-rate four-stream adapter and clock-gap split verified |
| **MoniPar** | Task 2 evaluation | repeated watch protocols and clinician scores across weeks | reviewed bounded-event adapter and sealed Task-2 comparisons verified |
| **KneE-PAD** | Task 2 evaluation | correct and released incorrect rehabilitation executions | reviewed adapter; reported as a research-placement, within-visit stress cell |

Each task owns its own cohort because the unit and leakage constraints differ. Task 1 trains on the
`synth_wrist_v1` corpus and evaluates C-MHAD plus OpenPack. Task 2 trains on HARMES, CrossFit, and
their pre-materialised variants, then evaluates MoniPar plus KneE-PAD. Task 3 trains/develops on
OpenPack, CrossFit, AIDLAB-HAR, and RecoFit, then evaluates C-MHAD, WEAR, and OCA. An arm trained on
OpenPack cannot call an OpenPack result unseen-dataset transfer.

Reporting contract:

- publish one result table per dataset and task;
- keep C-MHAD wrist and waist and OCA arm/chest conditions distinct;
- aggregate uncertainty over subjects, not windows or overlapping patches;
- show unsupported model/dataset cells as `N/A` with the reason; and
- do not headline an average across datasets with incompatible metrics or deployment conditions.

## 3. Current held-out sources

| dataset | measured local structure | primary use | readiness and limitation |
|---|---|---|---|
| **MoniPar** | 174 complete weekly protocols, 28 subjects, nine labels; session duration 405-445 s | Task 2 evaluation | **Signal and event adapter verified.** Released 50 Hz gravity-present watch acceleration is exposed in g with weekly subject/session identity. The seven single-run protocol items yield 1,207 primary bounded executions; multi-run `postural_transition` and `resting` are excluded from the primary protocol. Per-visit bradykinesia and rest-tremor metadata are converted in the canonical cache. |
| **SPAR** | 280 exercise bouts, 20 subjects, seven labels; median 42 s | Task-1 alignment and Task-2 within-bout repeatability | **Conditional.** Apple Watch acceleration and gyroscope are appropriate, but about 20 repetitions share one bout and are not independent sessions. |
| **Upper Limb Use** | 1,042 converted sessions, 15 subjects, 15 ADLs | bilateral wrist Task-2 exploration | **Blocked.** 598 sessions contain under 50 rows and many contain only two samples. Reconstruct and validate the source timeline before use. |
| **KneE-PAD** | 2,084 trials, 31 subjects, nine released variant labels; median 3.79 s | known-difference Task 2 | **Ready.** Correct trials and released incorrect variants form 13,768 sealed comparisons. The 148.15 Hz muscle-belly research IMUs make this a cross-placement stress cell, not consumer-watch evidence. |

MoniPar is used only for Task 2. After requiring four stable clinician-scored reference visits and
a later scored query, 20 comparisons remain (13 accepted and 7 changed). This is a deliberately
small clinician-rated feasibility cell, not complete clinical validation.

### 3.1 Additional longitudinal sources audited for Task 2

| dataset | actual release | role and gate |
|---|---|---|
| **ALAMEDA PD** | open CC BY 4.0; a 4.8 GB ZIP of 100 Hz wrist acceleration and clinical tables | conditional exploratory source, not a primary Task-2 cohort. The actual archive has 13 sensor subject codes and 22 campaigns, not four campaigns for each of the advertised 11 contributors; one complete campaign is duplicated across two identities, placement is unknown for two campaigns, and several campaigns lack a nearby clinical visit. After calibration, placement, duplication, and 14-day clinical-linkage checks, only participants 4 and 11 retain at least two same-placement campaigns with nearby clinical visits. |
| **COPS** | public OSF release; 66 participants, about six days each, bilateral 100 Hz wrist acceleration and hourly symptom diaries; 47.87 GB compressed | secondary short-term state-fluctuation source. All 66 participant archives are locally retained, match the OSF byte inventory, and pass ZIP integrity checks. Stream nested hourly files without expanding the projected roughly 297 GiB CSV corpus. The OSF node has no declared data license, so clarify reuse terms before publication. |
| **REHAB-120** | open CC BY 4.0 data; 120 baseline and 109 discharge assessments plus rehabilitation exercises | blocked: the release does not expose defensible baseline/discharge identity linkage, exercise arrays are derived orientation/flex signals, and one released NPY is corrupt. In the assessment IMU files, 2,454/5,994 acceleration arrays are all-zero placeholders and 262 reach about +/-16 although the paper declares m/s2 from a +/-2 g accelerometer; units and saturation handling require author clarification. |
| **WATCH-PD** | 82 early untreated Parkinson's participants and 50 controls with repeated standardized visits over 12 months | best gated external validation candidate; access requires a steering-committee proposal. |

ALAMEDA and COPS are explicitly `stream` adapters. MoniPar is a materialized canonical cache because
its 174 short weekly visits are the reviewed Task-2 evaluation unit. The tracked
Task-2 audit is `applications/motion_monitoring/data/inspection/task2_audit.json`; it verifies sampled
rates, units, finite masks, placements, clock monotonicity, annotation semantics, and corpus coverage.

Cross-sectional subjects cannot be ordered as simulated future versions of one person for the
primary Task-2 evaluation. They may train a population prior, but only repeated same-person
measurements support a longitudinal claim.

## 4. Development sources consumed by expanded HALO pretraining

| dataset | useful source structure | application requirement |
|---|---|---|
| **PHYTMO** | 30 subjects, correct/incorrect therapy series, optical reference | development for Task 2; series contain repeated movements but are not one execution per file |
| **MM-Fit** | 21 full workouts, phones, two watches, earbud, RGB-D and pose | possible future complete-timeline source for Tasks 1 and 3; current sessions are per-set excerpts |
| **Opportunity** | repeated household recordings with interaction and NULL tracks in the source | reconstruct complete runs and fine annotation tracks; current converter retains only four locomotion labels |
| **REALDISP** | ideal, self, and induced placement conditions | Task-2 remounting/nuisance development |
| **HARMES** | 72 raw dominant-wrist ADL timelines with event logs | Task-2 accepted-repeat training; the reviewed application adapter keeps inter-event gaps and 2,398 bounded executions |
| **XRF V2** | synchronized wrist, pocket, glasses, and ear views | useful multi-device development; reconstruct physical scenes rather than treating six views as six events |
| **Capture24** | 151 free-living wrist days and more than 2,500 converted hours | reconstruct source days for background and scale tests; coarse-label segments are not natural event boundaries |

These sources can select duration ranges, debug task construction, and train small task heads. They
cannot provide an unseen-source result for an encoder that saw them during pretraining.

## 5. Accessible continuous and video sources

### 5.1 C-MHAD

**Status: selective release imported and verified.** All 240 inertial CSV files and 24 annotation
workbooks were inspected. C-MHAD provides two-minute streams from 12 subjects,
synchronized 50 Hz wrist or waist six-axis IMU and 15 fps video, actions of interest among arbitrary
actions, and manually inspected event intervals.

The CSV files contain two metadata rows, millisecond timestamps, acceleration in m/s2, and angular
velocity in degree/s. Convert to g and rad/s. Across the release, files contain 5,948-6,101 rows,
last 118.87-121.97 seconds, and measure 50.027 Hz; no non-finite values, non-increasing timestamps, or
invalid annotation intervals were found. The source reports 30-40 missing initial IMU samples due to
Bluetooth delay. Preserve each file's measured timestamps and
align it to video/annotations; do not force a nominal row count or interpret source-recommended zero
padding as measured stillness. The authors' loader and the official page differ by two metadata rows
in the offset convention, equivalent to 40 ms at 50 Hz. Treat this as boundary uncertainty and
report a +/-2-sample sensitivity check for any boundary-error result.

**Role:** strongest sealed Task-1 source after a complete manifest audit and an exact-event Task-3
control. Report wrist gestures
and waist transitions separately.

### 5.2 WEAR

**Status: selective release imported and verified.** All 24 inertial CSV files and 29 annotation
JSON files were inspected. The published cohort has 22 participants performing 18 outdoor exercises
with four limb accelerometers and synchronized egocentric video.

The inspected raw file is fixed-rate 50 Hz, gravity-present acceleration in g with an explicit
`null` label, and no timestamp column. Derive time from row order and the documented rate. Use one arm
watch for the consumer-wearable primary result; leg watches may be secondary only.

The 24 recording files represent 22 people. The paper establishes that `sbj_18` and `sbj_19` are
second sessions of `sbj_0` and `sbj_14`, but the available documentation does not prove which repeat
maps to which original. Keep all four in one leakage group for splitting, but do not create
same-person pairs until the directional mapping is confirmed. `sbj_20` through `sbj_23` are four
new test participants. In `sbj_10.csv`, all three left-arm axes are absent for 51,392 rows. The
primary right-arm stream is complete; any multi-sensor result must use a channel mask rather than
zero filling or interpolation.

**Role:** real continuous Task-1 false-alarm evaluation and coarse Task-3 recurrence control.

### 5.3 OCA

**Status: complete release adapter implemented and verified.** The 33.22 MB official ZIP contains twelve
continuous CSV sessions, four six-axis IMUs, millisecond timestamps, six work phases, NULL, metadata,
and official train/validation/test partitions. Training and validation share P1/P2; only the P3/P4
test partition is subject-disjoint. It covers roughly 282 assembly cycles from five participants.

The files use two native clock regimes near 20 Hz and 27 Hz. `P0-R0.csv` contains an 81-second
timestamp gap and must be split. Median acceleration norm is about 9.8, consistent with m/s2. The
paper identifies the IMU as a BNO055, and every released gyroscope value is exactly quantized at
1/16 degree/s, the device's degree/s output scale. Convert acceleration by `/ 9.80665` and gyroscope
by `* pi / 180`. Two sensors are on upper arms and two are on a vest/chest; some sessions use active
arm support.

**Role:** primary public occupational Task-3 benchmark and a secondary Task-1 transfer source. Report
arm-support conditions separately and do not present it as a consumer-watch deployment.

### 5.4 GAITEX

**Status: oracle only.** The release contains repeated gait and rehabilitation variants, optical
motion capture, RGB video, manual repetition times, and Xsens-derived orientation. Its files called
raw IMU data contain orientation quaternions, not raw accelerometer and gyroscope channels required by
HALO. It can validate orientation, phase, or biomechanical interpretations through a separate
adapter, but it is not a native HALO test set.

### 5.5 RecGym

**Status: blocked.** The official 103 MB UCI ZIP downloaded on 2026-08-28 has a valid-looking central
directory but invalid file offsets and cannot extract `RecGym.csv`. An accessible mirror exposes
subject, placement, session, normalized acceleration/gyroscope values, capacitance, and workout, but
no timestamps or documented inverse physical scaling. Do not use it for HALO or physical
measurements until the original sequence and units are recovered from the authors or repository.

## 6. Repetition and recurrence sources

This section distinguishes four annotation contracts that must not be treated as interchangeable:

- **event interval:** start and end of each action occurrence;
- **repetition start:** a point annotation for each repetition, without its complete extent;
- **set interval plus count:** a bounded group and its total count, without individual occurrences;
- **sequence count:** a total count without all temporal locations.

### 6.1 OpenPack

**Status: complete 5.20 GiB non-RGB release imported and verified.** OpenPack contains more than 53
hours from 16 distinct people. The inspected release has 20,264 work-operation and 53,760 fine-action
rows, slightly more than the paper's 20,129 and 52,529 counts. Four nominal-30 Hz IMUs cover both
wrists and upper arms; the release also provides depth, LiDAR,
IoT, operation, action, box-cycle, order, and NULL annotations. Sessions contain repeated packaging
cycles and include procedural variation, irregular situations, and a rushed condition.

The `box` identity links fine actions to one completed packaging cycle. The adapter's
`packing_cycle` interval is derived as the minimum and maximum source operation boundaries for a box;
it is not a source-provided event interval. OpenPack can therefore train event matching at the
fine-action level and test hierarchical grouping into operations or cycles.
The 21 released identifiers represent 16 people: five `U02xx` identifiers are later recordings of
documented `U01xx` people. Collapse those aliases before any subject split. The 416 IMU files are
finite with gravity-present acceleration near 1 g, but their timestamps measure 30.30-33.33 Hz rather
than one exact rate. Preserve timestamps, split or mask two 1.26/5.88-second gaps in
`U0108.zip:atr/atr01/S0400.csv`, and exclude one zero-duration `Bend Flap` action row. Preserve action, operation,
and box annotation levels separately and report the active placement. The non-RGB release is CC
BY-NC-SA 4.0; RGB has separate terms.

**Role:** strongest verified occupational Task-1/Task-3 source. It may be a training source or an
official held-out-subject benchmark, but the same arm cannot call its test split an unseen dataset.

### 6.2 Bodyweight Exercise Segmentation

**Status: complete 136.5 MB archive and schemas inspected; physical units blocked.** The CC BY 4.0
release contains 13 continuous workouts from nine participants, 431 minutes, twelve exercises plus
no-exercise, exact set/rest intervals, and 4,756 repetition-start markers. Four six-axis IMUs are on
the left/right wrists and ankles.

The raw JSON uses irregular timestamps with duplicated samples and observed rates around 83-91 Hz.
Acceleration values appear integer-scaled around 1,000 per g, but neither acceleration nor gyroscope
conversion is documented sufficiently for defensible HALO ingestion. Obtain the thesis, acquisition
configuration, source code, or author confirmation before conversion. One participant contributes
five of the thirteen sessions, so subject-balanced splitting is mandatory.

**Role:** excellent controlled repetition-boundary source after units are resolved;
secondary rather than sole publication evidence because it accompanies a 2026 master's thesis rather
than a peer-reviewed dataset paper.

### 6.3 CrossFit exercise repetitions

**Status: complete selected release acquired and verified.** The paper reports 54 participants, ten
exercises, about 230 minutes, and 5,461 repetition starts. Fifty participants completed the
constrained workout represented by the released non-NULL arrays; the paper's 54-person count covers
the wider study. The inspected release contains 446 non-NULL exercise arrays and exactly 5,461
non-NULL repetition arrays. It also contains seven NULL bouts and
nine pseudo-repetition arrays under `Null`; those are background, not repetitions. Six non-NULL
fragments are only 0.08-0.16 seconds and must be excluded or separately justified. Accelerometer and
gyroscope rows are finite; one recording has missing orientation rows, which does not affect HALO's
IMU adapter. The release participant map exposes 57 codes, including seven NULL-only codes, so a
split must use the released mapping and must not equate code count with the paper's distinct-person
count. Arrays are interpolated to approximately 100 Hz by the authors' preprocessing code, and
repetition starts were marked using watch vibration. The released repetition slices run from one
machine-paced vibration cue to the next; they are useful same-motion training excerpts, not observed
natural repetition boundaries and not valid natural boundary ground truth. The public release has no
clear data licence. Obtain rights-holder confirmation before a publication result depends on it.

**Role:** strongest verified subject-diverse controlled source for Task-1/Task-3 same-motion pair
training and held-out-exercise evaluation. It is structured exercise data rather than natural
continuous occupational background.

### 6.4 AIDLAB-HAR

**Status: complete 3.29 MB release imported and verified.** AIDLAB-HAR contains 180 EDF recordings,
130 event CSV files, 13 exercises, and three background-like activities. Its event files provide
series intervals and 1,486 repetition-marker fiducial windows. Those marker windows are not complete
repetition intervals. The 90 `SUBxx` values are source codes, not globally unique participant IDs,
so this release cannot support a subject-disjoint claim without an external identity map. The public
signals are 50 Hz chest acceleration in g and orientation quaternions; raw gyroscope is absent.
The v3 archive is a corrected redistribution on Aidlab's production S3 endpoint, but the public page
does not establish the provenance of the corrections or an explicit data licence. Confirm the
rights holder and disclose who produced v3 before a publication result depends on it.

**Role:** small control for series boundaries and repetition anchors, not a complete repetition-
segmentation benchmark. Use through a declared acceleration-only adapter and do not report a
participant-disjoint split from the released `SUBxx` codes.

### 6.5 RecoFit

**Status: selected 1.57 GB continuous file acquired and verified.** The complete repository spans
more than 200 participants, but the selected `multionly` file contains 94 unique subject IDs and 126
complete visits, matching the RecoFit training cohort. It provides right-forearm 50 Hz acceleration
in g and gyroscope in degree/s, non-exercise background, exercise-set intervals, and total repetition
counts. It does not annotate every repetition's timestamp. The selected continuous file is sufficient
to reconstruct the isolated exercise data, so the duplicate `singleonly` file is intentionally not
downloaded.

**Role:** large-scale weak supervision for set consistency, counting, and background rejection. It
cannot directly supply event-level positive pairs without first inferring and validating individual
occurrences.

### 6.6 CaRa and DWC/ExRAC

**Status: official repositories and annotation contracts inspected; complete local import pending.**
CaRa contains ten subjects, fifty activities, more than 1,700 sequences, and approximately 35,787
repetitions from a dominant-hand smartwatch at 100 Hz. Its labels primarily provide sequence counts.
DWC/ExRAC contains 37 subjects, fifty actions, 1,502 entries, and 49,258 repetitions from a Galaxy
Watch 4; every sequence has a total count and the first three exemplars are localized through spoken
cues.

**Role:** weakly supervised recurrence/counting and Task-1 exemplar-conditioned controls. Neither
release supplies complete start/end boundaries for every occurrence.

## 7. Task-specific choices

The exact distinction between action instances, bouts, sets, counts, and fiducials is owned by
[`ANNOTATION_INVENTORY.md`](ANNOTATION_INVENTORY.md). The former generic event-proposal stage is only
an optional runtime baseline documented in
[`MOTION_PROPOSAL_BASELINE.md`](../methods/MOTION_PROPOSAL_BASELINE.md).

### Task 1: arbitrary task detection

- Use independent reference and query executions, not two windows from one bout.
- Train on `synth_wrist_v1`, whose references and inserted targets come from independent CrossFit
  wrist executions with configuration-matched real backgrounds.
- Evaluate C-MHAD and OpenPack under paired same-subject and cross-subject enrollment.

Owner: [`TASK1_ARBITRARY_DETECTION.md`](../tasks/TASK1_ARBITRARY_DETECTION.md).

### Task 2: change quantification

- Use HARMES and CrossFit accepted executions plus deterministic raw-level variants to train the
  change ruler.
- Use MoniPar clinician-scored single-run protocol items as the small between-week evaluation cell.
- Use KneE-PAD correct and released incorrect trials as a separate within-visit stress cell.
- Use a quality-controlled ALAMEDA subset for exploratory long-term personal state trends only after its
  identity/linkage manifest is verified, and COPS only as a short-term fluctuation stress test.
- Treat MJFF Levodopa Response as a gated future extension. DUO-GAIT is not part of V1 because the
  accessible archive remained incomplete.
- Do not use Upper Limb Use until the short-session defect is resolved.
- Treat GAITEX as a separate biomechanical/orientation oracle.

Owner: [`TASK2_CHANGE_QUANTIFICATION.md`](../tasks/TASK2_CHANGE_QUANTIFICATION.md).

### Task 3: recurrent motion discovery

- Train the pairwise same-motion metric from exact intervals in OpenPack, exercise/repetition arrays
  in CrossFit, AIDLAB-HAR series plus fiducial anchors, and Bodyweight after its unit blocker is
  resolved. Keep each annotation contract explicit rather than treating all four as exact intervals.
- Use RecoFit, MM-Fit, CaRa, and DWC/ExRAC only in explicitly weakly supervised arms.
- Use OpenPack official held-out subjects for occupational within-source evaluation and OCA for
  cross-dataset occupational transfer.
- Use WEAR and C-MHAD as complementary continuous recurrence controls.
- Reconstruct MM-Fit, Opportunity, HARMES, and Capture24 source timelines for development.
- Use training labels only as arbitrary event-equivalence identities. Hide all evaluation labels from
  matching and clustering, exposing them only for scoring.

Owner: [`TASK3_RECURRENT_MOTION_DISCOVERY.md`](../tasks/TASK3_RECURRENT_MOTION_DISCOVERY.md).

## 8. Prospective collection gap

No verified public source provides all four application requirements with a consumer phone or watch:
independent longitudinal sessions, precise event boundaries, controlled execution differences, and
long occupational background. A focused prospective collection should include:

- an engineering pilot of 10-20 participants followed by a relevant patient or worker cohort;
- a consumer wrist watch and, where justified, a phone in a natural pocket/waist position;
- 4-8 movements across at least three independently remounted sessions;
- several clinician- or ergonomist-approved references and controlled deviations;
- 30-60 minutes or more of natural target-absent background per participant;
- synchronized video and independent boundary review; and
- load, context, and task metadata for any clinical or ergonomic interpretation.

Public-data experiments should determine the final placement, recording duration, and sample-size
requirements before collection.

## 9. Primary sources

- [MoniPar public record](https://zenodo.org/records/8104853)
- [C-MHAD publication](https://doi.org/10.3390/s20102905) and
  [official dataset page](https://personal.utdallas.edu/~kehtar/C-MHAD.html)
- [WEAR publication](https://doi.org/10.1145/3699776) and
  [official dataset page](https://mariusbock.github.io/wear/)
- [OCA dataset record](https://doi.org/10.24406/fordatis/121) and
  [publication](https://doi.org/10.1109/INDIN51773.2022.9976078)
- [MM-Fit official dataset page](https://mmfit.github.io/)
- [OpenPack official dataset page](https://open-pack.github.io/) and
  [publication](https://doi.org/10.1109/PerCom59722.2024.10494448)
- [Bodyweight Exercise Segmentation release](https://doi.org/10.6084/m9.figshare.32756517.v1)
- [CrossFit repetition publication](https://doi.org/10.3390/s19030714)
- [RecoFit official repository](https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors)
- [CaRa official repository](https://github.com/bbvisual/CaRaCount) and
  [publication](https://doi.org/10.1109/TPAMI.2025.3548131)
- [DWC/ExRAC official repository](https://github.com/cvlab-stonybrook/ExRAC)
- [AIDLAB-HAR publication](https://doi.org/10.3390/s24123891)
- [GAITEX publication](https://doi.org/10.1038/s41597-025-06439-x)
- Local converter and sensor decisions: [`DATA_HETEROGENEITY.md`](DATA_HETEROGENEITY.md)
