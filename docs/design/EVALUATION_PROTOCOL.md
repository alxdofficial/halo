# Evaluation protocol

> **Protocol plan, 2026-08-31.** No application result is reportable until its artifact records the
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
8. Encoder and downstream checkpoint selection uses development subjects only. The chosen artifact
   is frozen before any sealed test score is computed.

## 2. Representation comparison

All methods receive the same source recordings and event boundaries. Compare:

- raw acceleration/gyroscope DTW or matrix-profile matching;
- interpretable physical features with the same temporal matcher;
- HALO patch embeddings;
- the primary author-released HARNet, UniMTS, and NormWear representations where their temporal
  interfaces support the task;
- ImageBind only as an optional generic multimodal appendix control; and
- the same small Task-3 pairwise metric trained separately for each frozen representation.

The primary external comparison contains only author-released pretrained checkpoints. Historical
CrossHAR and LiMU-BERT backbones pretrained in this repository may appear only as supplementary
diagnostics because their corpus, schedule, augmentation, and checkpoint choices are ours. Every
adapter must disclose whether it exports native temporal features or obtains a timeline by sliding a
model that natively returns only one pooled vector.

Different upstream corpora are part of the released model. This comparison identifies the best
application component; it does not by itself attribute a gain to architecture. Report checkpoint,
model size, accepted sensors, temporal granularity, latency, memory, and energy where measurable.

## 3. Task 1: arbitrary task detection

Task 1 enrolls independent executions, searches later continuous recordings, and reports event
average precision, false alarms per hour, boundary error, and count error. The complete cohort,
episode, complete-timeline decoding, and metric protocol is owned by
[`TASK1_ARBITRARY_DETECTION.md`](../tasks/TASK1_ARBITRARY_DETECTION.md). Window accuracy is not a
primary metric because background dominates continuous recordings.

The primary result always searches the complete timeline. The optional physical motion-proposal
baseline may additionally report runtime, event recall, and downstream accuracy to quantify whether
screening is worthwhile.

## 4. Task 2: activity difference quantification

The complete cohort, baseline, alignment, metric, and visualization protocol is owned by
[`TASK2_CHANGE_QUANTIFICATION.md`](../tasks/TASK2_CHANGE_QUANTIFICATION.md).

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

The complete motif-search, hidden-label scoring, and human-review protocol is owned by
[`TASK3_RECURRENT_MOTION_DISCOVERY.md`](../tasks/TASK3_RECURRENT_MOTION_DISCOVERY.md).

Training annotations define arbitrary same/different event identities. Evaluation identities are
held out completely; their annotations are hidden from matching and clustering and used only for
scoring. Report oracle source-interval matching and complete-timeline multiscale discovery
separately.

### Metrics

- event-level precision and recall for repeated ground-truth actions;
- same-motion pair AUROC and AUPRC before clustering;
- occurrence-count error;
- motif coverage: fraction of genuinely repeated events assigned to a coherent cluster;
- cluster purity and fragmentation, reported together rather than purity alone;
- false motif rate on target-absent or low-structure intervals;
- boundary error and stability under small duration/stride changes;
- recurrence ranking quality for the top motifs shown to a reviewer; and
- computational cost per hour of recording.

Report unseen-subject/known-identity, unseen-subject/held-out-identity, and unseen-dataset conditions
separately. Cross-dataset pairs are not assigned negative targets merely because annotation
vocabularies differ.

Evaluate complete recurrence thresholds. A threshold such as ten occurrences per day is an
application filter, not the only operating point.

### Human review study

For the applied occupational result, an ergonomist or informed reviewer receives representative
clips and timelines for the top motif clusters and marks each as meaningful, duplicate, noise, or
uncertain. Measure review time, inter-rater agreement, and the fraction of cumulative motion exposure
covered by confirmed motifs.

## 6. Existing data roles

The verified, checkpoint-dependent inventory is owned by
[`APPLICATION_DATASETS.md`](../data/APPLICATION_DATASETS.md). In particular, Upper Limb Use is
currently blocked by short converted artifacts; C-MHAD, WEAR, and OCA have verified raw-timeline
adapters and are frozen as test-only members of `COHORT_V1`. Dataset readiness must be copied into each
result artifact so a later source or converter change cannot silently alter the cohort.

The core public evaluation matrix is intentionally small:

| dataset | primary role | complementary role |
|---|---|---|
| **C-MHAD** | sealed Task-1 demonstrated-action detection | Task-3 event recurrence and wrist-versus-waist control |
| **WEAR** | long continuous Task-1 false-alarm validation | coarse Task-3 activity-bout control |
| **MoniPar** | longitudinal Task-2 change quantification | natural cross-week Task-1 condition |
| **OCA** | occupational Task-3 recurrent-motion discovery | occupational Task-1 transfer |

This is one shared four-dataset study, not four unrelated benchmark collections. C-MHAD, WEAR, and
OCA remain non-reportable until task-specific episode manifests and operating points are frozen;
their recording membership is fixed in `COHORT_V1`, and their loaders and physical contracts are
implemented and verified. UniMTS/OCA is additionally unsupported until
upper-arm placement is mapped explicitly to the released skeleton contract.

OpenPack is the priority Task-3 metric-training and held-out-identity source. Its import and
provenance checks are complete; it enters a result only after the exact identity manifest is frozen,
and never serves as an unseen-dataset test for an arm that trained its matcher on OpenPack.

A prospective consumer-watch collection is warranted only if the public datasets leave a material
deployment condition unmeasured. Such a collection would include
independent remounting, several sessions, target-absent intervals, clinician- or ergonomist-approved
references, controlled deviations, and video ground truth.

The exact temporal annotation contract of every source is recorded in
[`ANNOTATION_INVENTORY.md`](../data/ANNOTATION_INVENTORY.md).

## 7. Statistical and reporting rules

- The unit of uncertainty is the subject, not the window.
- Report one primary table per dataset and task condition, with subject-level confidence intervals.
- Do not use a pooled cross-dataset score as the headline: the datasets measure different tasks,
  placements, populations, durations, and annotation contracts.
- A compact cross-dataset figure may show standardized effect sizes or ranks, but it must preserve
  each dataset as a separate point and must not replace the per-dataset tables.
- Predeclare primary metrics and operating points.
- Include confidence intervals and all failed/unsupported cells.
- Do not choose the best checkpoint or threshold on the test cohort.
- Publish target prevalence and recording hours so false-alarm rates are interpretable.
- Store generated tables and figures under a versioned application result directory and summarize
  only the promoted protocol in `docs/results/RESULTS.md`.
