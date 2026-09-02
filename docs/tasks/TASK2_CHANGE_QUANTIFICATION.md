# Task 2: change quantification

> **Design of record, 2026-08-29.** Task 2 compares confirmed executions of the same movement. Its
> first version learns ordinary personal variation, then tracks persistent deviation through
> alignment and measurement; it is not a clinical diagnosis model.

**Implementation status, 2026-08-31.** The shared timestamped encoder export, bounded-execution
episode contract, set-conditioned metric head, phase-normalized latent residual, accepted-query and
changed-query objectives, within-reference-set ranking, masked unit-scaled target regression,
encoder/head gradient telemetry, and robust personal joint-variation fit are mechanically
implemented. Short synthetic and real-cache smokes pass. The longitudinal state smoother, complete
physical measurements, reportable task manifests, long training, and external validation remain
outstanding. Episodes are constrained to one dataset-scoped subject identity and compatible sensor
configuration. Bounded executions with internal missing-patch gaps are rejected rather than
interpolated across.

## 1. Question and deployment contract

Given several accepted executions of one task and a later confirmed execution, what changed, where
in the movement did it change, and is the change larger than ordinary repetition and remounting
variation?

```text
accepted baseline executions + later confirmed execution
    -> temporal alignment
    -> latent and physical residuals over movement phase
    -> comparison with the personal noise envelope
    -> interpretable change report with uncertainty
```

Task 2 receives independently bounded executions from source annotations, user-guided recordings, or
Task-1 detections. It does not require or evaluate a generic motion-proposal stage.

The system reports **difference and persistent drift**. It may report improvement, deterioration,
quality, fatigue, or clinical severity only when an external measure defines that direction. This
distinction is central: a stable change in strategy can be real without being clinically bad.

## 2. Research rationale

Wearable rehabilitation studies commonly summarize speed, intensity, consistency, control, and
smoothness. Such measurements can detect performance differences, but reviews also find limited
clinical validation and substantial variation in protocols. A useful system must therefore report
reliability and minimum detectable change, not just correlation or class separation.

Smoothness requires particular care. Spectral arc length (SPARC) and log dimensionless jerk have
well-defined interpretations when computed on suitable kinematic signals. Integrating consumer
accelerometer data to estimate velocity introduces drift. The initial implementation therefore
applies SPARC directly to angular velocity when gyroscope data are available, reports acceleration-
based jerk as an explicitly different measurement, and does not infer joint angles or position from
a single uncalibrated watch.

Task 2 is not presented as a standalone novel covariance or trend algorithm. Its contribution is as
the longitudinal measurement case study in the complete detect/compare/discover system and as a
stress test of representation geometry: useful embeddings should suppress accepted personal and
acquisition variation while retaining controlled or externally validated execution change. The
statistical components remain deliberately established and auditable.

## 3. Input and output

### 3.1 Input

- at least three accepted baseline executions where possible;
- one or more later executions confirmed to be the same task;
- source timestamps, valid-channel masks, placement, gravity state, and session identity;
- timestamped encoder patches and aligned raw IMU; and
- optional clinician scores, known execution variants, video, or motion-capture measurements.

One reference is enough to compute a distance but not enough to estimate normal personal variation.
Such a result is marked `reference_limited`.

### 3.2 Output

- total and phase-specific duration change;
- latent-shape residual over normalized movement phase;
- acceleration and angular-velocity intensity change;
- frequency-content and cadence change;
- smoothness and within-execution consistency change;
- standardized deviation from the accepted personal baseline;
- test-retest and remounting uncertainty;
- nearest historical executions and their alignment paths; and
- an optional externally validated direction or severity estimate.

## 4. Primary non-learned algorithm

### 4.1 Build the accepted baseline

Quality-control the accepted executions first. Reject corrupt traces and flag, rather than silently
drop, configuration mismatches. Compute all pairwise constrained DTW distances and choose the medoid
execution as the initial reference. With enough examples, compute a DTW barycenter for display while
retaining the real medoid as inspectable evidence.

The baseline distribution is formed from leave-one-execution-out comparisons among accepted
executions. It therefore measures ordinary execution variation rather than treating one arbitrary
demonstration as exact truth.

### 4.2 Align once in latent time

Use constrained dynamic time warping on normalized patch embeddings. The path is monotonic and has a
development-selected maximum warp. Alignment maps each execution to movement phase from 0 to 1.

Duration is recorded before alignment and reported separately. This prevents DTW from making a slow
execution appear unchanged merely because its shape can be warped onto the reference.

For robustness, compare three alignment inputs:

1. raw magnitude and physical features;
2. frozen HALO embeddings; and
3. each released encoder's temporal embeddings.

### 4.3 Compute phase-local latent residuals

For every matched pair on the alignment path, compute cosine distance between embeddings. Aggregate
those distances in a fixed number of movement-phase bins. Report the median residual, upper-tail
residual, and contiguous high-residual regions. This gives both a global difference and an answer to
"where did the execution change?"

Do not collapse this curve into one learned score in the initial system. The curve is the evidence
from which later validated scores can be built.

### 4.4 Compute interpretable physical measurements

All quantities use physical seconds and preserve sensor masks:

- total duration and phase durations;
- repetition count and cadence when repeated cycles can be established;
- median, peak, and integrated dynamic acceleration magnitude;
- median, peak, and integrated angular-velocity magnitude;
- energy in fixed physical frequency bands and dominant periodic frequency;
- SPARC on gyroscope angular velocity when sampling and duration are adequate;
- acceleration log dimensionless jerk, labeled `LDLJ-A` and not equated with velocity-based LDLJ;
- pause fraction and submovement count; and
- within-execution cycle variability when multiple repetitions occur.

Axis-specific values are shown only when orientation is known or gravity-frame alignment is valid.
Magnitude and physical-frequency summaries remain the placement-robust default.

### 4.5 Convert change into a personal deviation

Form one fixed feature vector per aligned execution comparison. It contains phase-local latent
residuals and separately measured physical changes. Fit the personal model separately for one
person, task, and compatible acquisition setup.

Estimate a robust center and per-feature scale from accepted executions. Then fit the covariance of
the standardized residuals using oracle-approximating shrinkage toward an identity covariance. This
captures ordinary correlated variation without estimating a singular high-dimensional covariance
from a handful of examples. With fewer than three accepted executions, use the diagonal fallback and
mark the result `reference_limited`.

The scalar joint deviation is the dimension-normalized Mahalanobis distance. Per-feature residuals
remain available for interpretation:

```text
standardized feature = (current - baseline_median)
                       / max(1.4826 * baseline_MAD, measurement_floor)

joint deviation = sqrt(standardized_features * precision * standardized_features
                       / number_of_features)
```

The measurement floor comes from test-retest and remounting data, not an arbitrary epsilon. Report
the raw unit, percent change, and standardized change together. A change is marked as exceeding
ordinary variation only when its confidence interval also exceeds the predeclared minimum detectable
change.

The implementation is `applications.motion_monitoring.task2.personal.fit_personal_variation`. This
deployment adaptation uses accepted reference executions only; it is separate from global neural
training.

### 4.6 Estimate persistent longitudinal change

Apply a robust state-space smoother or exponentially weighted trend model to the sequence of
per-execution deviations. This is the proposed personal low-pass filter: it operates across repeated
executions or sessions, not within the raw IMU trace. It suppresses isolated poor repetitions while
preserving a sustained change in latent shape, duration, cadence, intensity, or smoothness.

The trend model receives measurement uncertainty and session/remounting indicators. It reports the
estimated personal state, uncertainty band, change-point probability, and persistence duration. Its
time constant is selected on development subjects and translated into executions or days so it has
an interpretable meaning.

Raw physical trends remain parallel outputs. The encoder is useful only if its latent trajectory
detects validated changes more reliably, earlier, or with fewer false alarms than those physical
features alone.

## 5. Lightweight learned arm

The application model includes a small global metric because an encoder trained for activity
classification is not assumed to preserve the within-activity differences required here. The
encoder embeds every recording independently and in parallel. Immediately before scoring, a small
role-aware head contextualizes the accepted reference executions, then lets the query attend to that
reference set and optional same-person, same-configuration recordings of other tasks. The reference
baseline never attends to the query, so an unusual query cannot redefine normality for itself.

The head retains the transparent cosine distance to the mean accepted phase trajectory and learns a
bounded correction from the contextualized evidence. Reference, query, and personal-context tokens
have distinct role embeddings. No subject identifier is supplied to the network. A fixed cosine
metric and the robust personal statistical model remain mandatory floors.

Train it with episodic classification and ordinal constraints:

- a held-out accepted query should score below a known changed query under the exact same reference
  set;
- same-person, cross-session accepted repetitions are preferred positives;
- ordinary remounting and measured device noise are nuisance controls; and
- activity families, subjects, and datasets used for evaluation remain held out.

The head may improve the latent alignment metric, but it does not replace the physical report. A
single opaque quality regressor is deferred until clinician labels support one.

The learned target is nuisance-tolerant change, not disease identity. Stable accepted repetitions
teach the model what should remain close; known execution variants or externally measured
longitudinal changes teach which differences should remain observable.

For every released baseline encoder, freeze the encoder and train the same head separately. Report
HALO with the encoder frozen under that matched protocol, then report a distinct end-to-end HALO arm
whose task loss updates a task-specific encoder copy. Do not compare a fitted HALO head against an
unadapted baseline embedding.

## 6. Training-data construction

### 6.1 Real development episodes

Construct episodes containing three to eight accepted reference executions, one independent query,
and optional same-person recordings of other tasks as personal context. Split by subject and source
session before drawing episodes. The query and every transformed descendant of it must be absent from
its reference set. The reference-set size varies during development so deployment is not tied to one
enrollment count.

Useful local development sources are:

- **PHYTMO:** paired correct/incorrect therapy exercises and repeated series;
- **KneE-PAD:** correct and two clinically motivated incorrect variants of knee exercises;
- **REALDISP:** ideal, self, and induced sensor displacement as nuisance controls;
- **SPAR:** repeated shoulder exercises within a bout for phase alignment and repeatability; and
- **MM-Fit:** repeated exercise sets with synchronized pose for development-only phase checks.

PHYTMO, REALDISP, and MM-Fit were consumed by the expanded HALO checkpoint and therefore cannot be
used to claim unseen-source generalization for that checkpoint.

### 6.2 Synthetic sensitivity tests

Task 2 does not need a continuous recording. Independent bounded executions with the same declared
action are sufficient. Synthetic modifications can therefore provide both development training and
a held-out controlled evaluation in which direction and severity are known exactly:

- bounded whole-execution retiming;
- phase-local amplitude scaling;
- a phase-local pause or extra submovement;
- boundary truncation and extension; and
- session-wide orientation/remounting and measured noise as nuisances.

Known-change and nuisance transformations must be labeled separately. A model should respond to the
former and remain stable to the latter. Split base subjects, actions, sessions, and source recordings
before generating transformations; otherwise transformed copies leak across train and evaluation.
Evaluation seeds, severity ranges, and transformation compositions are frozen in a task manifest.

The controlled synthetic benchmark is a primary engineering test of sensitivity and invariance. It
is not by itself evidence of clinical or longitudinal usefulness. At least one real repeated-session,
known-variant, or externally measured evaluation remains required for that applied claim.

### 6.3 Training mixture

Construct balanced batches over source subjects and tasks rather than over the number of generated
variants. Whenever possible, an accepted and changed query share the exact same accepted reference
set so the ranking loss asks a well-defined personal question. Each batch contains:

- untouched independent accepted-query episodes;
- changed-query episodes with a declared direction and severity;
- nuisance queries such as remounting or measured device noise that remain accepted;
- optional same-person, same-configuration recordings of other tasks as role-marked context; and
- unlabeled real episodes used only for personal baseline fitting or unsupervised reporting.

Do not apply a change only to one class in a way that leaves an augmentation watermark. Hold out
complete base executions before generating either training or evaluation variants.

### 6.4 No-label mode

When no correct/incorrect labels exist, fit nothing beyond robust baseline statistics. The system can
still quantify longitudinal change and estimate whether it exceeds test-retest variation. It cannot
learn which direction is desirable.

### 6.5 Video and privileged supervision

When synchronized video exists, use a dedicated temporal-localization or pose pipeline to propose
event phase and visible kinematics. A VLM may provide coarse movement descriptions and flag examples
for review, but it is not precise ground truth for boundary timing, joint angle, or clinical quality.
Any sealed-test direction label must come from the source protocol, a clinician, a validated physical
measurement, or independent human review. Deployment remains IMU-only unless a result is explicitly
reported as multimodal.

## 7. Evaluation datasets and compatibility

| source | evaluation question | current status |
|---|---|---|
| **PHYTMO** | can the method distinguish source-declared correct from deliberately incorrect rehabilitation exercise execution? | primary controlled Task-2 development source: 30 subjects, two correct and two incorrect series per task, four 100 Hz IMUs, and motion-capture reference. |
| **ALAMEDA PD** | do wrist-motion summaries and latent state change across repeated free-living campaigns, and do those changes agree with repeated clinical measures? | accessible but conditional. The 4.8 GB ZIP passed its published MD5 and its Parquet data are finite 100 Hz acceleration in g with real timestamps. However, the archive has 13 sensor subject codes and only 22 campaigns, one complete campaign is byte-identical under participants 14 and 16, two participant-16 campaigns have unknown placement, two early campaigns fail device calibration, and several campaigns have no nearby clinical visit. All 32 daily files from the four selected same-placement campaigns have a documented visit within 14 days; their actual file dates, rather than campaign-folder names, are used for this join. It has no bounded repeated-task labels, so use it only as an exploratory free-living trend source unless the authors clarify the release. |
| **COPS** | can a personal state model distinguish persistent hourly tremor or kinesia changes from ordinary within-day movement variability? | useful secondary source: 66 participants, bilateral 100 Hz wrist acceleration, hourly symptom diaries, and 393.8 usable participant-days. All 66 compressed participant archives are locally retained, match the OSF byte inventory, and pass ZIP integrity checks. The verified participant sample has monotonic 100 Hz timestamps, finite acceleration in g, and the documented diary schema. It measures week-scale state fluctuation, not long-term disease progression. Stream the nested hourly files rather than expanding roughly 297 GiB of CSV. The OSF node is public but declares no data license, so clarify reuse terms before publication. |
| **REHAB-120** | can standardized stroke movements expose real recovery over a two-week intervention? | blocked despite strong clinical design. The release has 120 baseline and 109 discharge assessments but does not expose a defensible subject linkage between them; exercise files contain derived angles/flex signals rather than HALO-compatible IMU channels, and one released NPY is corrupt. Of 5,994 assessment sensor-1 arrays, 2,454 have all-zero acceleration and 262 reach about +/-16, while the paper declares m/s2 from a +/-2 g accelerometer. Seek identity, unit, and sentinel/saturation clarification before use. |
| **KneE-PAD** | do measures distinguish correct from two real incorrect variants? | usable if checkpoint provenance confirms exclusion; most local trials are under six seconds and use research placements, so the adapter must support honest short sequences |
| **SPAR** | are phase and physical measures stable across repeated shoulder motions? | usable for within-bout repeatability, not independent longitudinal generalization |
| **Upper Limb Use** | do bilateral measures expose affected/unaffected differences in ADLs? | blocked pending converter/source-timeline repair: 598 of 1,042 local sessions contain under 50 samples and many contain only two samples |
| **GAITEX** | can orientation/biomechanical reference data validate phase and known gait/exercise variants? | not a native HALO six-axis source: released IMU tables contain Xsens orientation quaternions rather than raw acceleration/gyroscope; use only through a separate orientation adapter or as an external oracle |

MoniPar is deliberately excluded from Task 2 because its sample labels are contiguous protocol states,
not independently bounded repetitions. It remains a valid natural cross-week source for Task 1. No inspected public
source simultaneously provides consumer-watch six-axis IMU, independent remounting, several
longitudinal sessions, deliberate execution errors, and sealed video/clinical ground truth. A small
prospective collection remains necessary for a strong applied claim.

Cross-subject ordering is never a substitute for longitudinal time. Other subjects may be used to
learn a population severity direction or initialize a nuisance model, but all primary trend and
change-point claims must be evaluated on repeated measurements from the same person, task,
device family, placement, channel set, and gravity convention.

### 7.1 Implemented data and encoder checks

The application adapters now expose selected ALAMEDA daily free-living campaigns and bilateral COPS
hourly recordings. ALAMEDA and COPS deliberately expose no
action events: their clinical or diary linkage is state supervision, not execution-boundary truth.
Both are read in place rather than expanded into a second cache.

The author-released HARNet, UniMTS, and NormWear encoders, plus optional ImageBind, pass finite
representation probes on these real streams. The same encoders also pass one-step Task-2 head fits
with nonzero finite head gradients. These checks prove mechanical compatibility only. They do not
close the remaining scientific gate: controlled PHYTMO/KneE-PAD and longitudinal ALAMEDA/COPS
development/test episode manifests must be frozen before a reportable fit.

## 8. Evaluation metrics

### 8.1 Reliability and measurement error

- intraclass correlation with the declared ICC form and confidence interval;
- within-subject coefficient of variation;
- standard error of measurement;
- minimum detectable change at 95% confidence;
- Bland-Altman bias and limits of agreement; and
- failure rate due to alignment, missing sensors, or insufficient duration.

Report within-session, cross-session, and remounting conditions separately.

### 8.2 Sensitivity to known differences

- sensitivity to known variants, specificity on accepted executions, balanced accuracy, accepted-
  execution false-alarm rate, and subject-level effect size at a development-fixed threshold;
- AUROC as a secondary threshold-free ranking diagnostic;
- Spearman association for ordinal clinician scores;
- mixed-effects estimates for repeated longitudinal scores;
- phase-local AUPRC or temporal IoU when changed phases are annotated; and
- monotonic response to held-out controlled speed, amplitude, and pause changes.

For longitudinal use, additionally report change-point delay, false alerts per subject-month where
the source duration supports it, trend correlation with repeated external measurements, and the
ability to distinguish a persistent drift from one isolated abnormal repetition.

### 8.3 Comparison against alternatives

Use identical events and alignment constraints for raw signal, physical features, HALO, and released
encoders. The main question is whether a representation improves reliability and sensitivity beyond
the physical controls, not whether it separates ordinary activity labels.

All confidence intervals resample subjects. Sessions and repetitions from one subject are not
treated as independent people.

## 9. Required visualizations

1. baseline and current traces aligned over movement phase, with a baseline variability envelope;
2. a phase-by-feature heatmap showing where latent and physical differences occur;
3. raw-unit and standardized-change dot plots with the minimum detectable change marked;
4. longitudinal small multiples showing the personal baseline, per-execution deviations, smoothed
   state, uncertainty, and externally measured outcomes for each participant;
5. Bland-Altman plots for repeatability;
6. the nearest historical execution and its alignment path; and
7. example video clips only where consent and dataset terms permit them.

Avoid radar charts and a single unlabeled quality score. Both obscure units and uncertainty.

## 10. First implementation milestone

1. Implement medoid selection, constrained latent DTW, and phase-bin residuals.
2. Add duration, magnitude, physical-frequency, gyroscope SPARC, and LDLJ-A measurements.
3. Estimate a personal baseline envelope with leave-one-execution-out controls.
4. Add the robust longitudinal state and test persistent-versus-isolated synthetic changes.
5. Validate sensitivity on PHYTMO and nuisance stability on REALDISP development subjects.
6. Freeze all choices, then evaluate MoniPar longitudinal sessions and eligible held-out sources.
7. Add a learned projection only if a specific fixed-metric failure remains.

## 11. Research basis

- Chen et al., *Shrinkage Algorithms for MMSE Covariance Estimation*, IEEE Transactions on Signal
  Processing 2010, [DOI 10.1109/TSP.2010.2053029](https://doi.org/10.1109/TSP.2010.2053029).
- Komaris et al., *Unsupervised IMU-based evaluation of at-home exercise programmes*, BMC Sports
  Science, Medicine and Rehabilitation 2022,
  [DOI 10.1186/s13102-022-00417-1](https://doi.org/10.1186/s13102-022-00417-1).
- O'Reilly et al., *Wearable Inertial Sensor Systems for Lower Limb Exercise Detection and
  Evaluation: A Systematic Review*, Sports Medicine 2018,
  [DOI 10.1007/s40279-018-0878-4](https://doi.org/10.1007/s40279-018-0878-4).
- Swain et al., *Wearable Technology for Assessing Movement Quality During Functional Tasks: A
  Systematic Review*, Sports Medicine 2023,
  [DOI 10.1007/s40279-023-01905-1](https://doi.org/10.1007/s40279-023-01905-1).
- Balasubramanian et al., *On the Analysis of Movement Smoothness*, IEEE Transactions on Biomedical
  Engineering 2012, [DOI 10.1109/TBME.2011.2179545](https://doi.org/10.1109/TBME.2011.2179545).
- Melendez-Calderon et al., *Estimating Movement Smoothness From Inertial Measurement Units*,
  Frontiers in Bioengineering and Biotechnology 2020,
  [DOI 10.3389/fbioe.2020.558771](https://doi.org/10.3389/fbioe.2020.558771).
- Jain et al., *Action Quality Assessment Using Siamese Network-Based Deep Metric Learning*, IEEE
  TCSVT 2020, [DOI 10.1109/TCSVT.2020.3017727](https://doi.org/10.1109/TCSVT.2020.3017727).
- Dal Farra et al., *Test-Retest Reliability and Minimal Detectable Changes for Wearable
  Sensor-Derived Gait Stability, Symmetry, and Smoothness*, Sensors 2025,
  [DOI 10.3390/s25061764](https://doi.org/10.3390/s25061764).
- Papagiannakis et al., *The ALAMEDA Data Collection Protocol*, Healthcare 2023,
  [DOI 10.3390/healthcare11192656](https://doi.org/10.3390/healthcare11192656); public data record
  [DOI 10.5281/zenodo.15769959](https://doi.org/10.5281/zenodo.15769959).
- Nesser et al., *Continuous observation of Parkinsonian symptoms using symptom diaries and
  wearable accelerometry*, Scientific Data 2026,
  [DOI 10.1038/s41597-026-06999-6](https://doi.org/10.1038/s41597-026-06999-6).
- Czech et al., *Improved measurement of disease progression in people living with early
  Parkinson's disease using digital health technologies*, Communications Medicine 2024,
  [DOI 10.1038/s43856-024-00481-3](https://doi.org/10.1038/s43856-024-00481-3).
- Lv et al., *A wearable sensor-based kinematic dataset collected under standardized rehabilitation
  tasks from 120 post-stroke patients*, Scientific Data 2026,
  [DOI 10.1038/s41597-026-07802-2](https://doi.org/10.1038/s41597-026-07802-2).
