# Task 1: arbitrary task detection

> **Design of record, 2026-08-29.** This document owns the Task-1 data construction, matching,
> training, and evaluation protocol on the `application-motion-monitoring` branch. Shared motivation,
> representation contracts, and reporting rules remain in the documents linked from
> [`docs/README.md`](../README.md).

## 1. Question and deployment contract

Given one or a few bounded recordings of a movement, determine whether independent executions occur
in a later continuous IMU recording and return their time intervals.

The movement name is not an input. It can be a known rehabilitation exercise or an arbitrary
personal task. A deployment consists of:

1. recording one or more reference executions;
2. trimming and validating those references;
3. monitoring a later continuous recording;
4. proposing intervals that contain coherent motion;
5. matching each proposal or timeline subsequence against the references; and
6. returning event times, scores, and aligned reference evidence for review.

IMU data cannot establish intent. An automatically proposed interval is a motion event until a user
or clinician confirms that it is the intended task.

## 2. Episode representation

The canonical training query is **120 physical seconds**, independent of sampling rate. This is long
enough to contain targets, hard distractors, and meaningful target-absent background while producing
only about 120 positions for an encoder with one-second patches. Longer deployment recordings are
processed in overlapping 120-second blocks and detections are merged in physical time.

Reference duration remains variable. References are padded only within a batch and carry an honest
valid-time mask. The common episode schema is:

```text
reference:
    one independently recorded execution, variable duration
query:
    120 s continuous or constructed timeline
targets:
    zero or more [start_sec, end_sec] intervals
metadata:
    subject, session, source recording, placement, channels, rate, gravity state
masks:
    valid samples/patches, padding, and artificial-join guard intervals
```

The 120-second choice is an initial engineering default, not a scientific claim. A 60-second arm may
be profiled for end-to-end training throughput, but all methods in a comparison receive the same
physical query duration.

## 3. Data pools

### 3.1 Reference and positive executions

The main training condition uses two different physical executions of the same source action:

```text
enrollment = enrollment_augmentation(reference execution A1)
inserted target = insertion_augmentation(independent execution A2)
```

`A1` and `A2` must have different execution identifiers. Different views from synchronized sensors
are not independent executions. Adjacent windows or repetitions from one unsegmented bout are also
not independent sessions.

Preferred development sources are HARMES event segments, XRF V2 action intervals, Opportunity and
FORTH-TRACE transitions, SP-SW-HAR transitions, UniMiB bounded actions, MM-Fit exercise sets,
PHYTMO exercise series, KU-HAR trials, and WISDM trials. A set or series is called a bout unless the
source supplies individual repetition boundaries.

Episode conditions are kept separate:

- same person, different execution;
- same person, different session or remounting;
- different person, same source action;
- target absent; and
- similar-action hard negative.

### 3.2 Continuous background

Preferred background comes from real continuous captures: Capture24 free-living data, Opportunity
NULL intervals, gaps between HARMES events, NFI-FARED `no activity`, between-set MM-Fit intervals,
and background intervals in XRF V2. Source converters that currently discard NULL or inter-event
samples are insufficient for Task 1; the application loader must preserve those source timelines.

A query should come from one physical recording whenever possible. If several recordings must be
joined, all parts must have compatible modality, placement, channel semantics, gravity state, and
sampling clock. Artificial joins receive a guard interval that cannot be a target or contribute to
the loss. A discontinuity must never become a shortcut for detecting an insertion.

### 3.3 Distractors

Each constructed query includes ordinary background and, where available, actions with similar
duration, frequency, body placement, or movement family. Enrolled distractors prevent the mere
presence of structured movement from identifying the target. Target-absent queries are mandatory.

## 4. Construction rules

The preferred order of evidence is:

1. a naturally continuous recording with native event intervals;
2. a continuous recording into which an independent execution is inserted;
3. compatible real background and action recordings joined into a timeline; and
4. an augmented copy of one execution, used only as a smoke test.

Selection and augmentation are distinct. Choosing another subject or session supplies independent
evidence; it is not augmentation. Permitted initial transformations are bounded retiming, enrollment
boundary error, measured sensor noise, and session-level remounting. Transformations that affect only
the inserted interval can watermark the answer. Rotation, scale, and noise therefore apply to the
whole query session, while enrollment receives an independent session-level transform. Retiming can
remain local because speed variation is part of the target hypothesis.

For insertion, an equal-duration background interval is replaced rather than numerically adding two
IMU signals. Targets are never placed across source gaps or joins. A configurable guard region around
an artificial boundary is excluded from boundary and detection losses.

## 5. Matching and learning arms

Every encoder exports normalized, timestamped patch embeddings through the shared `MotionSequence`
interface. The first matcher is non-parametric:

```text
reference and query embeddings
    -> pairwise cosine cost
    -> constrained subsequence dynamic time warping
    -> duration/warp penalty
    -> event proposals
    -> temporal non-maximum suppression
    -> development-calibrated threshold
```

Subsequence DTW is the primary floor because it searches a long recording without requiring a
pre-cut candidate and permits bounded local speed differences. Kernel subsequence search provides a
faster approximation when deployment recordings become long. Soft-DTW is reserved for the learned
arm because it makes the alignment cost differentiable; it is not required by the initial system.

The comparison has three levels for HALO and each compatible released encoder:

| arm | encoder | task mechanism | interpretation |
|---|---|---|---|
| frozen-direct | frozen | normalization + cosine subsequence DTW | information already present in the representation |
| frozen-calibrated | frozen | the same matcher plus a tiny fitted metric or score calibration | ease of adaptation |
| end-to-end | trainable | the same interface with differentiable alignment/ranking | best task-specific system |

The smallest learned extension is a shared linear projection or diagonal feature weighting before
alignment. It is introduced only after the frozen-direct floor is measured. End-to-end training uses
independent positive executions, target-absent queries, hard negatives, and a differentiable
soft-DTW or ranking objective. Fine-tuning creates a task-specific encoder checkpoint; the original
frozen checkpoint remains shared by all application tasks.

## 6. Action proposals and boundaries

Task matching and action proposal are evaluated separately. The proposal ladder is:

1. user-supplied enrollment boundaries, followed by automatic trimming;
2. motion energy with hysteresis;
3. multivariate change-point refinement;
4. recurrence or cross-reference consistency; and
5. an optional temporal action-localization head trained with explicit background.

The initial deployment uses user-guided enrollment and automatic trimming. Automatic monitoring can
use a cheap proposal stage, but the full subsequence matcher must also be evaluated on an unfiltered
timeline so proposal errors are visible. A proposed interval is rejected as an enrollment reference
when repeated demonstrations do not contain a stable alignable pattern.

## 7. Dataset roles

### 7.1 Development

Existing expanded-pretraining sources may be used to construct episodes, choose duration ranges,
debug boundaries, and fit task heads. They cannot support an unseen-dataset claim for a HALO
checkpoint that consumed them during representation training.

CrossFit supplies the strongest subject-diverse controlled repetition source for same-motion
matching. OpenPack supplies occupational fine-action intervals and repeated work cycles. AIDLAB-HAR
provides series boundaries and repetition-marker anchors, while Bodyweight becomes useful after its
unresolved raw IMU scaling is repaired. RecoFit, MM-Fit, CaRa, and DWC/ExRAC provide set/count or exemplar/count
supervision and remain weak-supervision arms rather than event-level ground truth.

### 7.2 Primary sealed evaluation: C-MHAD

C-MHAD is the closest public match to the deployment protocol: 12 subjects, 240 two-minute streams,
five wrist gestures, seven waist transitions, 50 Hz accelerometer and gyroscope, arbitrary actions of
non-interest, synchronized video, and manually inspected start/end times at 10 ms resolution. One
annotated occurrence can be enrolled and later streams searched without synthesizing the target.

The wrist-gesture and waist-transition protocols are reported separately. Development thresholds
must be fixed before C-MHAD is read for scoring. Source: Wei, Chopada, and Kehtarnavaz, *Sensors*
2020, DOI [10.3390/s20102905](https://doi.org/10.3390/s20102905).

A source-file audit on 2026-08-28 confirmed that each sampled CSV has two metadata rows followed by
timestamped acceleration in m/s2 and angular velocity in degree/s. Convert those units explicitly.
The source reports 30-40 missing IMU samples near the start of a stream due to Bluetooth delay. Use
timestamps and the synchronized-video alignment to preserve that missing interval; do not create a
motion signature by blindly zero-padding it.

### 7.3 Secondary evaluations

- **WEAR:** continuous outdoor workouts, smartwatch acceleration, synchronized egocentric video,
  explicit NULL periods, and interval annotations. Use one wrist stream for the primary consumer-
  wearable comparison. A file-level check confirmed fixed-rate 50 Hz, gravity-present acceleration
  in g, four limb placements, and no raw timestamp column; preserve row order and derive time from
  the documented rate. DOI [10.1145/3699776](https://doi.org/10.1145/3699776).
- **MoniPar:** same-person exercise protocols repeated across weeks. It tests longitudinal
  enrollment but supplies exercise blocks rather than video-verified individual repetitions.
- **OCA:** approximately 282 industrial assembly cycles, six action phases, and NULL
  labels. It is an occupational transfer result, not the primary consumer-device result. Dataset
  record: [10.24406/fordatis/121](https://doi.org/10.24406/fordatis/121). Its verified files contain
  both approximately 20 Hz and 27 Hz sessions, plus an 81-second timestamp gap in `P0-R0.csv`.
  Split gaps, preserve each session's timestamps rather than imposing one global rate, convert
  acceleration from m/s2 to g, and convert BNO055 degree/s values to rad/s.

SPAR contains about 20 repetitions in each exercise bout but does not provide independent session or
verified per-repetition intervals in the local representation. It is a boundary-discovery stress
test, not the primary detection benchmark.

## 8. Split and leakage rules

- Training, development, and sealed evaluation are split before window or episode creation.
- Reference and query occurrences are distinct physical executions.
- Subject and source-recording identifiers are retained in every episode.
- Simultaneous sensor views cannot cross the reference/query boundary as independent evidence.
- Thresholds, warp limits, projection weights, and checkpoint selection use development data only.
- C-MHAD and WEAR are not used for task training if they serve as sealed tests.
- Checkpoint provenance is audited per encoder; possible upstream overlap is reported or the affected
  dataset is removed from the unseen-data comparison.
- Strict arbitrary-action results hold out activity families as well as datasets. C-MHAD wrist
  gestures provide a useful novel-motion condition; its common posture transitions are reported as
  a separate generic-motion condition.

## 9. Metrics and required controls

Primary metrics are event average precision and recall at a predeclared false-alarms-per-hour
operating point. Also report event F1, onset/offset error, temporal IoU, count error, target-absent
false alarms, and recall versus query/reference duration ratio. Uncertainty is computed over subjects,
not windows.

Required controls are raw-signal DTW, engineered physical features with the same matcher, every
frozen encoder with the same matcher, an oracle-boundary arm, and each automatic proposal method.
Oracle boundaries isolate representation and matching quality; automatic boundaries measure the
deployable system.

## 10. Required visualizations

1. a full query timeline with the match-score curve, threshold, detections, and true intervals;
2. a reference-query similarity matrix with the selected alignment path;
3. event recall versus false alarms per hour at each reference count;
4. target-present and target-absent score distributions by subject and dataset;
5. recall versus execution-speed and duration ratio; and
6. a review strip of true detections, hard negatives, false alarms, and misses.

Every detection shown to a user links to the enrolled execution and its aligned segment. A pooled
accuracy bar cannot explain whether a failure came from boundaries, speed variation, or a similar
distractor.

## 11. First milestone

1. Inventory independent executions, continuous background, and event boundaries in the local
   corpus.
2. Export or compute one-second physical feature sequences.
3. Run cosine subsequence DTW on natural MoniPar cross-week pairs and synthetic independent-trial
   episodes.
4. Measure runtime, positive/hard-negative separation, target-absent scores, and boundary recovery.
5. Acquire and verify the complete C-MHAD release before freezing its manifest as the primary test.
6. Only then export frozen HALO and baseline representations through the common adapter.

Preliminary probes establish feasibility and defaults. They are not promoted application results.

## 12. Preliminary feasibility probe

> **Diagnostic only, 2026-08-28.** Reproduce with
> `/home/alex/code/HALO/legacy_code/.venv/bin/python -m applications.motion_monitoring.task1.preliminary_probe --max-cases 350`.
> The probe uses
> fixed physical accelerometer features on CPU; it does not evaluate HALO or a baseline encoder.

The converted corpus contains enough bounded action material to construct development episodes, but
not every converter retains a continuous timeline:

| source | locally measured structure | Task-1 implication |
|---|---|---|
| HARMES | 7,018 event segments, 38.75 h | many repeatable actions; original inter-event timeline still needed |
| XRF V2 | 5,435 physical events across 32,610 sensor views, 74.78 h | strong multi-device events; current sessions do not reconstruct scenes |
| Opportunity | 3,620 labeled blocks grouped into 24 recordings, 6.50 h | continuous recordings are reconstructable, but explicit NULL is absent from converted labels |
| MM-Fit | 555 sets grouped into 21 workouts, 3.09 h | useful natural workout structure; sets are not individual repetition annotations |
| Capture24 | 13,120 coarse segments, 2,560.38 h | abundant free-living background; current segments need source-day reconstruction |
| MoniPar | 174 complete weekly protocols from 28 subjects, 20.29 h | strongest local natural cross-session probe |
| USC-HAD | 840 trials from 14 subjects, 12 actions, five trials each | strong isolated independent-execution control |
| MotionSense | 360 repeated trials from 24 subjects | useful phone-based independent-trial control |

No inspected converted source exposes an explicit `NULL`, `background`, or `no_activity` label in
its session label map. This does not mean background is absent from the raw datasets. It means the
Task-1 loader must recover source timelines and preserved gaps instead of treating the action-only
session store as a complete monitoring recording. SP-SW-HAR is particularly unsuitable in its
current application form because its converted sessions are one-second windows.

For the numerical probe, one exercise block from a MoniPar week was used as the reference and the
next available week from the same subject was searched. Across 350 deterministic cases spanning
22-24 subjects per exercise:

| method | full-session center accuracy | mean temporal IoU | target score better than paired target-absent 120 s crop | recall at 95% negative specificity |
|---|---:|---:|---:|---:|
| physical features + subsequence DTW | 68.57% | 0.598 | 94.86% | 75.14% |
| pooled physical features + cosine | 67.14% | 0.550 | 96.29% | 75.43% |

The CPU probe processed 12.4 cases/s. The result supports three limited conclusions:

1. fixed 120-second queries are computationally and statistically workable;
2. simple features already separate many target-present and target-absent crops, so raw and physical
   controls will be meaningful rather than trivial; and
3. temporal alignment modestly improves boundary overlap but does not yet improve calibrated
   presence detection over pooling. Finger tapping and chair-rise intervals are the clearest hard
   cases, so encoder comparisons and better boundary constraints remain necessary.

All 240 selectively downloaded C-MHAD inertial streams and 24 annotation workbooks, all 24 WEAR
inertial files and 29 annotation JSON files, and OCA's complete 33 MB release were inspected on
2026-08-29. This establishes file-level compatibility, not a frozen test manifest. The tracked
inspection records durations, interval validity, rates, identity aliases, missing channels, and OCA's
clock gap. A separate immutable manifest must still be frozen before sealed evaluation.

## 13. Research basis

- Li and Hu, *Head Gesture Recognition Combining Activity Detection and Dynamic Time Warping*,
  Journal of Imaging 2024, [DOI 10.3390/jimaging10050123](https://doi.org/10.3390/jimaging10050123).
- Candelieri et al., *Efficient Kernel-Based Subsequence Search for Data-Driven Prediction and
  Decision Making*, Sensors 2019, [DOI 10.3390/s19235192](https://doi.org/10.3390/s19235192).
- Cuturi and Blondel, *Soft-DTW: a Differentiable Loss Function for Time-Series*, ICML 2017,
  [arXiv 1703.01541](https://arxiv.org/abs/1703.01541).
- Wei, Chopada, and Kehtarnavaz, *C-MHAD: Continuous Multimodal Human Action Dataset of Simultaneous
  Video and Inertial Sensing*, Sensors 2020,
  [DOI 10.3390/s20102905](https://doi.org/10.3390/s20102905).
- Bock et al., *WEAR: An Outdoor Sports Dataset for Wearable and Egocentric Activity Recognition*,
  IMWUT 2024, [DOI 10.1145/3699776](https://doi.org/10.1145/3699776).
