# Task 0: event proposal and segmentation

> **Design of record, 2026-08-30.** Task 0 is deliberately a small, off-the-shelf front end.
> It proposes observable motion intervals for Tasks 1-3; it is not a new activity recognizer.

## 1. Question and scope

Given a continuous IMU recording, where are the bounded intervals of coherent human motion?

```text
continuous native-rate IMU
    -> invalid-data mask
    -> physical motion evidence
    -> change points and hysteresis intervals
    -> candidate events with confidence
```

The output is a list of start times, end times, proposal scores, boundary confidence, and source
provenance. It does not assign an activity name and cannot establish that a movement was intentional.
Stillness, incidental movement, and deliberate work can only be separated to the extent that their
sensor dynamics differ. A Task-1 reference, recurrence in Task 3, synchronized video, or human review
provides the missing semantic evidence.

## 2. Why an established method is sufficient

Two mature method families already address the required boundary problem:

1. multivariate change-point detection partitions a timeline when its local signal distribution
   changes; and
2. temporal action localization predicts variable-duration events and background over a complete
   timeline.

[PELT](https://doi.org/10.1080/01621459.2012.737745) is a standard penalized change-point method that
does not require a predefined event count and has a maintained implementation in
[`ruptures`](https://github.com/deepcharles/ruptures/blob/master/docs/user-guide/detection/pelt.md).
Greedy Gaussian Segmentation (GGS) has direct wearable-HAR evidence, but it requires a chosen number
of segments and assumes approximately Gaussian, independent samples. Learned inertial temporal
action localization has also improved coherent segmentation and NULL handling, but it requires
supervised interval labels and adds another trained model.

The primary Task-0 implementation is therefore physical motion evidence, hysteresis, and PELT.
Energy-only hysteresis is the simple floor, GGS is an HAR-specific comparison, and released temporal
localization is a supervised comparison rather than a prerequisite. Task 0 is not a claimed modeling
contribution.

## 3. Output contract

Every proposal contains:

- source recording, subject, session, and sensor-stream identifiers;
- start and end times in physical seconds;
- the valid-sample fraction and any gap or clipping flags;
- a motion-evidence score;
- start- and end-boundary confidence;
- the physical features that triggered the proposal;
- links to aligned raw signal and `MotionSequence` patches; and
- an explicit `uncertain` state when the operating threshold is not met confidently.

Overlapping proposals from synchronized sensor streams remain linked. They are not counted as
independent events.

## 4. Primary off-the-shelf pipeline

### 4.1 Signal-quality screen

Before event proposal, split at hard recording boundaries, clock reversals, and long timestamp gaps.
Mark non-finite values, constant runs, saturation, and missing channels. Never interpolate through a
gap large enough to hide an event boundary. Fixed-rate files without timestamps retain row order and
derive physical time only from the source-specified rate.

### 4.2 Physical evidence

Compute a compact evidence sequence in short overlapping windows using native timestamps:

- root-mean-square dynamic acceleration magnitude after a low-pass gravity estimate; and
- root-mean-square angular-speed magnitude when gyroscope data exist.

Jerk, spectral concentration, periodicity, and cross-axis covariance change are optional ablations,
not parts of the primary detector. They remain only if a development-set ablation improves boundary
or false-proposal metrics enough to justify the added parameters.

Estimate the slowly varying gravity component with a low-pass filter only to obtain dynamic motion
evidence. Retain the gravity-present signal for the encoder and for posture-sensitive downstream
analysis. Fit robust feature centers and scales on development recordings and freeze them before
sealed evaluation. Per-recording median/MAD normalization may be reported as an unlabeled online-
adaptation arm, but it is not silently applied in the primary protocol because it changes the
operating point based on each test recording's motion prevalence.

### 4.3 Initial intervals

Use two thresholds with hysteresis:

- a higher threshold starts an interval;
- a lower threshold keeps it active;
- a minimum duration removes spikes; and
- a short-gap rule joins fragments separated by brief pauses.

Thresholds, minimum duration, merge gap, PELT penalty, `min_size`, and `jump` are selected on
development subjects in physical-time units and then frozen. A false-proposals-per-hour target is
used only when the development source exhaustively labels relevant background. No parameter is
fitted to a sealed test recording's labels.

### 4.4 Boundary refinement

Run `ruptures.Pelt(model="l2")` over the compact evidence sequence in each quality-contiguous block.
Use the hysteresis intervals to associate nearby change points with rough starts and ends. Search
only a bounded margin around each rough boundary, then select the nearest change point supported by
the combined evidence. This keeps the method conventional and prevents a distant unrelated change
from replacing the event boundary. Compare GGS only through a tested reference implementation.

Each boundary's confidence combines change magnitude, agreement across features, and stability under
small changes to window duration and stride. Threshold-sensitive intervals are retained as uncertain
rather than silently discarded.

### 4.5 Proposal merging

Merge intervals only when their gap is short and the evidence remains above the continuation floor.
Keep nested or adjacent intervals when the data support two scales; Task 3 may need a repeated
sub-action while Task 1 may need the complete exercise. Final non-overlap is a downstream decision,
not an irreversible Task-0 assumption.

## 5. Comparison arms

| arm | input | fitted component | purpose |
|---|---|---|---|
| energy floor | raw physical summaries | development thresholds | simplest deployable baseline |
| PELT + hysteresis | compact physical summaries | penalty and thresholds | primary Task-0 method |
| frozen-latent change points | HALO or released encoder timeline | penalty and thresholds | test whether the representation sharpens boundaries |
| temporal localization | raw or frozen latent timeline | small released TAL head | learned upper comparison |
| video-privileged localization | IMU at deployment, video during training/annotation | same small head | test privileged supervision |

The learned arm should begin from a maintained implementation such as the inertial TAL pipeline of
Bock et al., which includes ActionFormer-style configurations. It must not be described as
unsupervised because event/background labels train it.

## 6. Training and development data

The statistical floors require only development thresholds. The learned comparison requires complete
timelines with positive intervals and explicit background.

Training examples must preserve the source recording whenever possible. Synthetic timelines are
permitted for development when source converters expose only isolated actions:

1. draw real background from one compatible recording;
2. replace, rather than add to, an interval with an independently recorded action;
3. mask a guard region around every artificial join;
4. include target-absent timelines and incidental movement;
5. use several event durations and numbers of events; and
6. split subjects and source recordings before construction.

Synthetic events teach mechanics but cannot establish real-world boundary accuracy. The primary
evaluation remains naturally continuous, human- or source-annotated data.

## 7. Preprocessing and augmentation

The default statistical method uses no training augmentation. For a learned comparison, only
physically motivated nuisance transformations are allowed:

- one session-wide SO(3) rotation applied consistently to every axis of every vector sensor;
- measured sensor noise or small calibration-scale error;
- bounded sample loss with an honest missing-data mask;
- small annotation-boundary jitter; and
- a mild whole-session clock perturbation when the model consumes physical timestamps.

Do not apply a transformation only to positive intervals, because that creates an insertion
watermark. Do not use local time warping as a Task-0 nuisance: duration and temporal boundaries are
the quantities being detected.

## 8. Dataset roles and verified compatibility

| source | verified structure | role | required handling |
|---|---|---|---|
| **OpenPack** | 53,760 fine-action rows in the inspected release, nested within operations and repeated box cycles, plus NULL and synchronized occupational sensors | largest occupational development source or official held-out-subject test | collapse five identity aliases, preserve measured 30.30-33.33 Hz timestamps, handle two clock gaps and one zero-duration action, and keep action/operation/box levels separate |
| **Bodyweight Exercise Segmentation** | continuous workouts with exact set/rest intervals and 4,756 repetition-start points | controlled repetition-onset development after source repair | raw integer acceleration/gyroscope scale is unresolved; point starts do not supply complete repetition end boundaries |
| **AIDLAB-HAR** | series intervals and repetition-marker fiducial windows for 13 exercises plus three background-like activities | small proposal and anchor control | marker windows are not full repetitions, `SUBxx` codes are not global participant IDs, and the chest stream is acceleration/orientation rather than standard six-axis watch input |
| **C-MHAD** | 240 two-minute streams; synchronized 50 Hz six-axis IMU and 15 fps video; manually inspected intervals | primary sealed Task-0/1 evaluation | parse two metadata rows; convert m/s2 to g and degree/s to rad/s; align by timestamp and known missing initial Bluetooth samples rather than fabricating motion with zero padding |
| **WEAR** | continuous 50 Hz four-watch acceleration, explicit `null`, THUMOS-style intervals, synchronized egocentric video | primary learned-development or sealed evaluation source | use one arm watch for the consumer-wearable result; acceleration is gravity-present in g; row order supplies time because the raw CSV has no timestamp column |
| **OCA** | twelve continuous CSV sessions, four six-axis IMUs, and six actions plus NULL | occupational evaluation and Task-3 development | split timestamp gaps; preserve per-session native rates near 20 or 27 Hz; convert m/s2 to g and BNO055 degree/s to rad/s; report research upper-arm/chest placement |
| **MM-Fit** | synchronized phones, watches, earbud, RGB-D, pose, workout sets and repetition information | development only for current expanded HALO checkpoint | reconstruct complete workouts from source files; converted set sessions are not continuous recordings |
| **MoniPar** | 174 complete weekly exercise protocols, 28 subjects, 50 Hz watch acceleration | longitudinal development and Task-2 linkage | exercise blocks are source intervals, not individual repetition boundaries |

File-level checks on 2026-08-29 covered all selectively downloaded C-MHAD and WEAR files and the
complete OCA release. OCA contains both approximately 20 Hz and 27 Hz sessions, and `P0-R0.csv`
contains an 81.296-second timestamp gap. A loader that assumes one global rate or one uninterrupted
session would be invalid. Its gyroscope values are exactly quantized at the BNO055's 1/16 degree/s
scale, so the adapter converts them by `* pi / 180`. C-MHAD streams contain 5,948-6,101 rows over 118.87-121.97 seconds rather
than one nominal row count. WEAR's 24 files represent 22 people, and `sbj_10.csv` is missing 51,392
rows on every left-arm axis. Both releases require explicit file/timestamp and missing-channel
manifests rather than count-based assumptions.

## 9. Evaluation

Primary metrics are:

- event average precision across temporal-IoU thresholds;
- event recall at fixed false-proposals-per-hour operating points where background is exhaustively annotated;
- start and end boundary absolute error;
- boundary F1 within declared time tolerances;
- over-segmentation and under-segmentation rates;
- proposal stability under small analysis-window and stride changes; and
- wall time, peak memory, and energy where measurable per recording hour.

Report subject-level bootstrap confidence intervals and per-dataset results. Score NULL/stillness,
incidental motion, sensor defects, and annotated events separately only when those annotations
exist. On C-MHAD, the released labels mark actions of interest rather than every coherent movement;
an unmatched proposal is therefore unscored or manually reviewed, not automatically a false
positive. Use C-MHAD primarily for annotated-event recall and boundary quality. Report false
proposals per hour only on datasets whose background annotation is verified as exhaustive. Frame
accuracy is secondary because background prevalence can make it misleading.

Task 1 must be evaluated both with Task-0 proposals and by direct full-timeline matching. Task 3 must
include a direct timeline motif baseline. These controls expose proposal-stage false negatives.

## 10. Required visualizations

1. a complete recording timeline with raw motion evidence, proposed intervals, and ground truth;
2. zoomed boundary panels showing feature changes and confidence;
3. event-recall versus false-proposals-per-hour curves only for exhaustively annotated background;
4. boundary-error distributions per event duration and dataset;
5. an over-/under-segmentation confusion summary; and
6. a review strip of high-confidence true events, false proposals, and misses.

The plots must show physical time and actual signal evidence. A single aggregate F1 score is not
sufficient to understand whether the detector is usable.

## 11. First implementation milestone

1. Pin and test `ruptures` in the project environment, then implement quality masks, development-
   fitted physical evidence, hysteresis, and PELT refinement. It is not installed as of 2026-08-28.
2. Reconstruct one development timeline each from MM-Fit and MoniPar.
3. Build file-level loaders for C-MHAD and WEAR without reading their test labels during tuning.
4. Freeze thresholds on development subjects.
5. Report proposal curves and direct-downstream controls before adding a learned head or GGS arm.

## 12. Research basis

- Killick, Fearnhead, and Eckley, *Optimal Detection of Changepoints With a Linear Computational
  Cost*, JASA 2012, [DOI 10.1080/01621459.2012.737745](https://doi.org/10.1080/01621459.2012.737745).
- Truong, Oudre, and Vayatis, *Selective Review of Offline Change Point Detection Methods*, Signal
  Processing 2020; maintained implementation in
  [`ruptures`](https://github.com/deepcharles/ruptures/blob/master/docs/user-guide/detection/pelt.md).
- Bock, Moeller, and van Laerhoven, *Temporal Action Localization for Inertial-based Human Activity
  Recognition*, IMWUT 2024, [DOI 10.1145/3699770](https://doi.org/10.1145/3699770).
- Li et al., *Applying Multivariate Segmentation Methods to Human Activity Recognition From Wearable
  Sensors' Data*, JMIR mHealth 2019, [DOI 10.2196/11201](https://doi.org/10.2196/11201).
- Aminikhanghahi and Cook, *Enhancing Activity Recognition Using CPD-Based Activity Segmentation*,
  Pervasive and Mobile Computing 2019,
  [DOI 10.1016/j.pmcj.2019.01.004](https://doi.org/10.1016/j.pmcj.2019.01.004).
- Alhammad and Al-Dossari, *Dynamic Segmentation for Physical Activity Recognition Using a Single
  Wearable Sensor*, Applied Sciences 2021, [DOI 10.3390/app11062633](https://doi.org/10.3390/app11062633).
- Bock et al., *WEAR: An Outdoor Sports Dataset for Wearable and Egocentric Activity Recognition*,
  IMWUT 2024, [DOI 10.1145/3699776](https://doi.org/10.1145/3699776).
- Wei, Chopada, and Kehtarnavaz, *C-MHAD*, Sensors 2020,
  [DOI 10.3390/s20102905](https://doi.org/10.3390/s20102905).
