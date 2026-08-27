# Evaluation protocol

> **Protocol plan, 2026-08-27.** No application result is reportable until its artifact records the
> dataset, subjects, sessions, references, query intervals, encoder checkpoint, adapter, thresholds,
> and protocol fingerprint defined here.

## 1. Shared data rules

1. Split by real subject and source recording before creating windows.
2. A reference and evaluated occurrence must be independent executions. Adjacent windows from one
   bout do not become independent examples.
3. Development data selects thresholds and hyperparameters; sealed test data only measures them.
4. Test activities may be named for scoring but their names do not enter Tasks 1-3.
5. Synthetic insertion uses an independently recorded occurrence when possible. Copying the query
   excerpt, even after augmentation, is a smoke test only.
6. Report same-session, cross-session, remounting, cross-device, and cross-subject conditions
   separately. Do not average away the deployment condition.
7. A HALO checkpoint used for an unseen-dataset claim must not have consumed that dataset during
   self-supervised pretraining.

## 2. Representation comparison

All methods receive the same source recordings and event boundaries. Compare:

- raw acceleration/gyroscope DTW or matrix-profile matching;
- interpretable physical features with the same temporal matcher;
- HALO patch embeddings;
- author-released HARNet, UniMTS, NormWear, and ImageBind representations where their temporal
  interfaces support the task; and
- a small Siamese metric head only as a later, explicitly trained arm.

Different upstream corpora are part of the released model. This comparison identifies the best
application component; it does not by itself attribute a gain to architecture. Report checkpoint,
model size, accepted sensors, temporal granularity, latency, memory, and energy where measurable.

## 3. Task 1: arbitrary activity detection

### Cohort construction

- Enroll one or more executions from one session.
- Search an independent continuous session from the same subject for the primary result.
- Include target-absent sessions and hard negatives from related movements.
- Keep reference count fixed within a comparison and report the full reference-count curve only as
  a secondary analysis.

### Metrics

- event average precision;
- recall at fixed false alarms per hour;
- event F1 at the development-selected threshold;
- onset and offset error;
- count error;
- detection latency for streaming-compatible variants; and
- degradation across session, remounting, placement, and device changes.

Window accuracy is not a primary metric because background dominates continuous recordings.

## 4. Task 2: activity difference quantification

Evaluate three properties separately.

### Reliability

- within-session and cross-session test-retest error;
- intraclass correlation where enough repeated sessions exist;
- sensitivity to device remounting and ordinary repetition variation; and
- the minimum change distinguishable from that noise floor.

### Known execution differences

- AUROC and effect size for correct versus independently recorded incorrect variants;
- ordinal association with clinician severity or quality scores;
- sensitivity to controlled changes in speed, range, and amplitude; and
- phase localization agreement when a deviation interval is annotated.

### Longitudinal association

- within-subject association between movement change and clinical measurements;
- mixed-effects or repeated-measures analysis rather than treating sessions as independent people;
- confidence intervals over subjects; and
- explicit separation of cross-sectional subject differences from within-person change.

Latent distance is never labeled "quality" solely because it separates activity classes.

## 5. Task 3: recurrent motion discovery

Activity annotations are hidden from the discovery algorithm and used only for scoring.

### Metrics

- event-level precision and recall for repeated ground-truth actions;
- occurrence-count error;
- motif coverage: fraction of genuinely repeated events assigned to a coherent cluster;
- cluster purity and fragmentation, reported together rather than purity alone;
- false motif rate on target-absent or low-structure intervals;
- boundary error and stability under small duration/stride changes;
- recurrence ranking quality for the top motifs shown to a reviewer; and
- computational cost per hour of recording.

Evaluate complete recurrence thresholds. A threshold such as ten occurrences per day is an
application filter, not the only operating point.

### Human review study

For the applied occupational result, an ergonomist or informed reviewer receives representative
clips and timelines for the top motif clusters and marks each as meaningful, duplicate, noise, or
uncertain. Measure review time, inter-rater agreement, and the fraction of cumulative motion exposure
covered by confirmed motifs.

## 6. Existing data roles

| dataset | allowed role on current checkpoints | main value | limitation |
|---|---|---|---|
| MoniPar | held-out application evaluation | weekly consumer-watch sessions and MDS-UPDRS-derived exercises | early-stage Parkinson cohort; severity imbalance |
| SPAR | held-out detection evaluation | repeated shoulder motions from Apple Watch | repetitions share one bout; not cross-session |
| Upper Limb Use | held-out exploratory evaluation | bilateral wrist data from controls and hemiparetic patients | not longitudinal exercise-quality data |
| PHYTMO | development only for current HALO checkpoint | repeated correct/incorrect therapy executions and optical reference | consumed by expanded HALO pretraining |
| MM-Fit | development only for current HALO checkpoint | continuous rep annotations and synchronized devices | consumed by expanded HALO pretraining |
| KneE-PAD | conditional held-out stress evaluation if checkpoint provenance confirms exclusion | real knee-pathology execution errors | research body placements and mostly sub-6-second trials |

A publishable evaluation will probably require a small prospective collection with independent
remounting, several sessions, target-absent intervals, clinician-approved references, controlled
deviations, and video ground truth.

## 7. Statistical and reporting rules

- The unit of uncertainty is the subject, not the window.
- Report per-subject and per-dataset results before pooled summaries.
- Predeclare primary metrics and operating points.
- Include confidence intervals and all failed/unsupported cells.
- Do not choose the best checkpoint or threshold on the test cohort.
- Publish target prevalence and recording hours so false-alarm rates are interpretable.
- Store generated tables and figures under a versioned application result directory and summarize
  only the promoted protocol in `docs/results/RESULTS.md`.
