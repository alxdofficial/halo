# Evaluation protocol

> **Protocol plan, 2026-08-30.** No application result is reportable until its artifact records the
> dataset, subjects, sessions, references, query intervals, encoder checkpoint, adapter, thresholds,
> and protocol fingerprint defined here.

## 1. Shared data rules

1. Split by real subject and source recording before creating windows.
2. A reference and evaluated occurrence must be independent executions. Adjacent windows from one
   bout do not become independent examples.
3. Development data selects thresholds and hyperparameters; sealed test data only measures them.
4. Test activities may be named for scoring but their names do not enter Tasks 0-3.
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

## 3. Task 0: event proposal and segmentation

Task 0 receives complete timelines with event boundaries hidden. It reports:

- event average precision across temporal-IoU thresholds;
- event recall at fixed false-proposal rates where background is exhaustively annotated;
- false proposals per recording hour where background is exhaustively annotated;
- start/end boundary error and boundary F1;
- over-segmentation and under-segmentation rates;
- stability under small duration and stride changes; and
- computational cost per recording hour.

Score near-still background, incidental movement, sensor artifacts, and coherent annotated events
separately only where the source supports those distinctions. An unmatched proposal is not a false
positive when the source labels only target actions and leaves other real movement unannotated.
C-MHAD therefore supports annotated-target recall and boundary metrics, but not naive false
proposals per hour. A VLM may generate training pseudo-labels, but sealed evaluation boundaries and
labels require source-provided or human-verified ground truth. Also report Task-1 and Task-3 results
with oracle intervals, Task-0 proposals, and direct-timeline controls so the downstream cost of
proposal errors is visible.
The complete contract is owned by
[`TASK0_EVENT_SEGMENTATION.md`](../tasks/TASK0_EVENT_SEGMENTATION.md).

## 4. Task 1: arbitrary task detection

Task 1 enrolls independent executions, searches later continuous recordings, and reports event
average precision, false alarms per hour, boundary error, and count error. The complete cohort,
episode, action-proposal, and metric protocol is owned by
[`TASK1_ARBITRARY_DETECTION.md`](../tasks/TASK1_ARBITRARY_DETECTION.md). Window accuracy is not a
primary metric because background dominates continuous recordings.

Report both Task-0-proposal search and direct full-timeline search so proposal misses are not
misattributed to the Task-1 matcher.

## 5. Task 2: activity difference quantification

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

## 6. Task 3: recurrent motion discovery

The complete motif-search, hidden-label scoring, and human-review protocol is owned by
[`TASK3_RECURRENT_MOTION_DISCOVERY.md`](../tasks/TASK3_RECURRENT_MOTION_DISCOVERY.md).

Training annotations define arbitrary same/different event identities. Evaluation identities are
held out completely; their annotations are hidden from matching and clustering and used only for
scoring. Report oracle source intervals and Task-0 proposals separately.

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

## 7. Existing data roles

The verified, checkpoint-dependent inventory is owned by
[`APPLICATION_DATASETS.md`](../data/APPLICATION_DATASETS.md). In particular, Upper Limb Use is
currently blocked by short converted artifacts; C-MHAD, WEAR, and OCA have passed only the file-level
checks recorded there and still need frozen manifests. Dataset readiness must be copied into each
result artifact so a later source or converter change cannot silently alter the cohort.

The core public evaluation matrix is intentionally small:

| dataset | primary role | complementary role |
|---|---|---|
| **C-MHAD** | sealed Task-0 segmentation and Task-1 demonstrated-action detection | wrist-versus-waist condition |
| **WEAR** | external continuous Task-0/Task-1 validation | Task-3 recurrence control |
| **MoniPar** | longitudinal Task-2 change quantification | natural cross-week Task-1 condition |
| **OCA** | occupational Task-3 recurrent-motion discovery | occupational Task-0/Task-1 transfer |

This is one shared four-dataset study, not four unrelated benchmark collections. C-MHAD, WEAR, and
OCA are not reportable until their pending loader/manifest checks in `APPLICATION_DATASETS.md` are
closed. UniMTS/OCA is additionally unsupported until upper-arm placement is mapped explicitly to
the released skeleton contract.

OpenPack is the priority Task-3 metric-training and held-out-identity source. It enters the final
evaluation matrix only after import and provenance checks, and never serves as an unseen-dataset test
for an arm that trained its matcher on OpenPack.

A prospective consumer-watch collection is warranted only if the public datasets leave a material
deployment condition unmeasured. Such a collection would include
independent remounting, several sessions, target-absent intervals, clinician- or ergonomist-approved
references, controlled deviations, and video ground truth.

## 8. Statistical and reporting rules

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
