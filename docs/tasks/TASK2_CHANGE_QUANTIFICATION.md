# Task 2: change quantification

> **Design of record, 2026-08-29.** Task 2 compares confirmed executions of the same movement. Its
> first version learns ordinary personal variation, then tracks persistent deviation through
> alignment and measurement; it is not a clinical diagnosis model.

**Implementation status, 2026-08-30.** The shared timestamped encoder export, bounded-execution
pair contract, phase-normalized latent residual, accepted-variation and known-change objectives,
masked unit-scaled target regression, and encoder/head gradient telemetry are mechanically
implemented. Short real-cache smokes pass. Personal longitudinal state fitting, complete physical
measurements, task manifests, long training, and external validation remain outstanding.

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

For each scalar or phase bin, estimate the accepted baseline median and median absolute deviation.
The standardized deviation is:

```text
deviation = (current - baseline_median) / max(1.4826 * baseline_MAD, measurement_floor)
```

The measurement floor comes from test-retest and remounting data, not an arbitrary epsilon. Report
the raw unit, percent change, and standardized change together. A change is marked as exceeding
ordinary variation only when its confidence interval also exceeds the predeclared minimum detectable
change.

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

## 5. Optional lightweight learned arm

Learning is added only if fixed alignment cannot separate known changes from nuisance variation. The
smallest head is a normalized diagonal weighting or linear projection applied to patch embeddings
before the same DTW and reporting pipeline.

Train it with pairwise or ordinal constraints:

- independent accepted repetitions should be closer than known incorrect or perturbed executions;
- same-person, cross-session accepted repetitions are preferred positives;
- ordinary remounting and measured device noise are nuisance controls; and
- activity families, subjects, and datasets used for evaluation remain held out.

The head may improve the latent alignment metric, but it does not replace the physical report. A
single opaque quality regressor is deferred until clinician labels support one.

The learned target is nuisance-tolerant change, not disease identity. Stable accepted repetitions
teach the model what should remain close; known execution variants or externally measured
longitudinal changes teach which differences should remain observable.

## 6. Training-data construction

### 6.1 Real development pairs

Construct tuples containing a baseline set, an independent accepted execution, and, where available,
a known variant. Split by subject and source session before drawing tuples.

Useful local development sources are:

- **PHYTMO:** paired correct/incorrect therapy exercises and repeated series;
- **KneE-PAD:** correct and two clinically motivated incorrect variants of knee exercises;
- **REALDISP:** ideal, self, and induced sensor displacement as nuisance controls;
- **SPAR:** repeated shoulder exercises within a bout for phase alignment and repeatability; and
- **MM-Fit:** repeated exercise sets with synchronized pose for development-only phase checks.

PHYTMO, REALDISP, and MM-Fit were consumed by the expanded HALO checkpoint and therefore cannot be
used to claim unseen-source generalization for that checkpoint.

### 6.2 Synthetic sensitivity tests

Synthetic modifications test whether a metric responds in the intended direction:

- bounded whole-execution retiming;
- phase-local amplitude scaling;
- a phase-local pause or extra submovement;
- boundary truncation and extension; and
- session-wide orientation/remounting and measured noise as nuisances.

Known-change and nuisance transformations must be labeled separately. A model should respond to the
former and remain stable to the latter. Synthetic changes are not called clinical impairment and are
not the primary test result.

### 6.3 No-label mode

When no correct/incorrect labels exist, fit nothing beyond robust baseline statistics. The system can
still quantify longitudinal change and estimate whether it exceeds test-retest variation. It cannot
learn which direction is desirable.

### 6.4 Video and privileged supervision

When synchronized video exists, use a dedicated temporal-localization or pose pipeline to propose
event phase and visible kinematics. A VLM may provide coarse movement descriptions and flag examples
for review, but it is not precise ground truth for boundary timing, joint angle, or clinical quality.
Any sealed-test direction label must come from the source protocol, a clinician, a validated physical
measurement, or independent human review. Deployment remains IMU-only unless a result is explicitly
reported as multimodal.

## 7. Evaluation datasets and compatibility

| source | evaluation question | current status |
|---|---|---|
| **MoniPar** | does a watch-based movement signature change across weeks and associate with neurologist-reviewed MDS-UPDRS exercise severity? | strongest local longitudinal signal source; 21 PD and 7 controls. Severity MAT files exist separately but are not exposed by the current converter; implement and audit subject/week alignment before fitting this association. |
| **KneE-PAD** | do measures distinguish correct from two real incorrect variants? | usable if checkpoint provenance confirms exclusion; most local trials are under six seconds and use research placements, so the adapter must support honest short sequences |
| **SPAR** | are phase and physical measures stable across repeated shoulder motions? | usable for within-bout repeatability, not independent longitudinal generalization |
| **Upper Limb Use** | do bilateral measures expose affected/unaffected differences in ADLs? | blocked pending converter/source-timeline repair: 598 of 1,042 local sessions contain under 50 samples and many contain only two samples |
| **GAITEX** | can orientation/biomechanical reference data validate phase and known gait/exercise variants? | not a native HALO six-axis source: released IMU tables contain Xsens orientation quaternions rather than raw acceleration/gyroscope; use only through a separate orientation adapter or as an external oracle |

After its severity adapter is verified, MoniPar is suitable for real longitudinal association but
not a balanced severity benchmark: the
published cohort omits the most severe class and is concentrated in low severity. No inspected public
source simultaneously provides consumer-watch six-axis IMU, independent remounting, several
longitudinal sessions, deliberate execution errors, and sealed video/clinical ground truth. A small
prospective collection remains necessary for a strong applied claim.

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

- AUROC and subject-level effect size for accepted versus known variants;
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
