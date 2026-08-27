# Dataset roles for movement monitoring

> **Application inventory, 2026-08-27.** Dataset role is checkpoint-dependent. A source consumed by
> a HALO checkpoint during self-supervised pretraining may be used for development but not to claim
> unseen-dataset generalization for that checkpoint.

## Primary held-out sources

| dataset | subjects and structure | sensors | task role | limitation |
|---|---|---|---|---|
| **MoniPar** | 21 Parkinson's patients and 7 controls, repeated weekly exercises for up to 9 weeks | consumer-watch accelerometer, 50 Hz | cross-session Task 1; longitudinal Task 2; association with MDS-UPDRS-derived measurements | early-stage cohort, severity imbalance, accel only |
| **SPAR** | 20 subjects, seven shoulder exercises, left/right watch, about 20 repetitions in each bout | Apple Watch accel+gyro, 50 Hz | within-bout Task 1 and phase alignment | repetitions share one continuous bout, so they do not establish cross-session generalization |
| **Upper Limb Use** | 10 controls and 5 hemiparetic participants performing 15 ADLs | bilateral wrist accel+gyro, 50 Hz | affected/unaffected and bilateral Task-2 exploration | no longitudinal exercise-quality protocol |
| **KneE-PAD** | 31 people with knee pathology, three exercises with clinically grouped error variants | eight research IMUs plus sEMG, 148.15 Hz | known-difference Task 2 if checkpoint provenance confirms exclusion | non-consumer placements; most trials are shorter than six seconds |

MoniPar is the strongest existing source because its sessions are genuinely separated by weeks and
its publication links sensor features to clinical motor measurements. SPAR is valuable for event
boundaries and repeated shoulder motion, but adjacent repetitions must not be relabeled as independent
enrollment sessions.

## Development sources already consumed by expanded HALO pretraining

| dataset | useful property | permitted use on current expanded checkpoint |
|---|---|---|
| **PHYTMO** | correct/incorrect therapy executions, repeated series, optical reference | algorithm development and visualization only |
| **MM-Fit** | continuous workouts, repetition annotations, synchronized watch/phone/ear devices | Task-1 and Task-3 development; cross-device debugging |
| **Opportunity** | repeated short object-interaction and locomotion patterns | motif-discovery development |
| **REALDISP** | repeated activities under ideal and displaced sensor placement | remounting and placement robustness development |
| **HARMES** | fine-grained repeated wrist kitchen and bathroom motions | occupational motif development |
| **XRF V2** | synchronized wrist, pocket, glasses, and ear streams | heterogeneous-device development |

These datasets can expose implementation failures and help select duration ranges on development
subjects. They cannot close the final unseen-data claim for a HALO encoder that saw them in Phase A.

## Required structure by task

### Task 1

- one or more independent reference executions;
- a later continuous recording with precise event intervals;
- target-absent intervals;
- same-subject hard negatives; and
- real source-recording ids so adjacent windows cannot cross the split.

### Task 2

- repeated executions of the same task;
- several accepted baseline repetitions;
- independent remounting or session variation;
- known perturbations, clinician scores, or physical reference measurements; and
- enough subjects for within-person and between-person effects to be separated.

### Task 3

- long continuous occupational or household recordings;
- repeated actions with event boundaries hidden from the algorithm;
- substantial unstructured and target-absent background;
- recurrence across separated intervals rather than duplicate sensor buffers; and
- human-reviewable context for deciding whether a motif is meaningful work.

## Prospective collection gap

No current public source provides all three application requirements using only a consumer watch or
phone: independent longitudinal sessions, annotated execution differences, and long occupational
background. A focused prospective collection should therefore include:

- 10-20 participants for an engineering pilot, followed by a real patient or worker cohort for an
  applied claim;
- 4-8 movements recorded across at least three sessions;
- independent watch removal and remounting;
- several clinician-approved references and controlled deviations;
- at least 30-60 minutes of target-absent or natural background per participant;
- synchronized video and review annotations; and
- explicit load/context metadata for any occupational risk interpretation.

The initial public-data experiments should determine sensor placement, recording duration, and
sample-size requirements before this collection is finalized.

## Provenance sources

- [MoniPar publication](https://pubmed.ncbi.nlm.nih.gov/38148984/)
- [PHYTMO publication](https://pubmed.ncbi.nlm.nih.gov/35661743/)
- [KneE-PAD publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC11992047/)
- Local converter and sensor decisions: [`DATA_HETEROGENEITY.md`](DATA_HETEROGENEITY.md)
