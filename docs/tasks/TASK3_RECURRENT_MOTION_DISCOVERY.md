# Task 3: recurrent motion discovery

> **Design of record, 2026-08-29.** Task 3 learns a transferable definition of "the same recurring
> motion" from datasets with bounded, labeled events. Deployment requires neither an enrolled
> reference nor an activity vocabulary. Classical motif search remains a non-learned control.

**Implementation status, 2026-08-31.** Complete-timeline collation, dense physical-time multiscale
candidates, exact-event overlap targets, scope-local arbitrary identities, balanced pair sampling,
same-motion metric learning, recurrence graph decoding, temporal consolidation, metrics, and
encoder/head gradient telemetry are mechanically implemented. Short real-cache smokes pass.
Operating-point calibration, immutable manifests, long training, and human-review artifacts remain
outstanding.
Candidate construction is tested to be invariant to heterogeneous batch padding. Events clipped at
a crop boundary cannot be presented as exact supervision: non-exhaustive crops ignore them, while
exhaustive supervision fails closed.

## 1. Question and deployment contract

Given a long unlabeled IMU recording, which temporally structured motions recur, where do they occur,
and how much cumulative exposure do they represent?

```text
continuous recording
    -> one timestamped base-patch encoding
    -> pooled multiscale temporal candidates
    -> learned pairwise same-motion scores
    -> temporal consolidation of overlapping candidates
    -> thresholded recurrence graph
    -> recurring motion clusters and bouts
    -> recurrence and exposure ranking
    -> human confirmation, rejection, or naming
```

The system returns **motion motifs**, not activities, intent, or ergonomic hazards. A reviewer
supplies meaning and decides whether a motif warrants monitoring. A confirmed motif can become a
Task-1 reference, and Task 2 can track its later execution.

## 2. Training formulation

### 2.1 Labels are arbitrary equivalence identities

Training annotations define only whether two bounded events are instances of the same source action.
Their text is not embedded and the numeric identity has no meaning. Initially scope an identity as:

```text
(dataset, annotation track, compatible sensor configuration, action identity)
```

Two events with the same identity form a positive pair. Two explicitly different identities within
the same annotation system form a negative pair. Events from different datasets are not assumed to
be equal or different unless their annotation equivalence has been verified. This avoids treating
shared words as physical equivalence or treating different naming conventions as negatives. A
compatible configuration has the same device family, placement, channel set, and gravity convention.
Device model and native rate may vary as declared nuisance factors within that family; a phone and a
watch are never interchangeable. Synchronized sensor views of one physical execution share one
instance identifier and therefore cannot masquerade as independent positive executions.

The deployment test holds out complete identities. Success therefore means that the learned matching
rule transfers to motions whose training labels and semantic names were never seen.

### 2.2 Dense multiscale candidates and boundary supervision

Encode each complete recording once at the encoder's declared base stride. Pool adjacent patch
embeddings over a small fixed set of physical durations to produce a temporal pyramid. Each candidate
stores its start, end, duration, scale, embedding, recording, subject, session, and sensor
configuration.

Source event intervals supervise candidates without being supplied at deployment:

- candidates with sufficient overlap with the same event identity are positives;
- candidates assigned to explicitly different events are negatives;
- partial overlaps near boundaries are ignored until a boundary-tolerant target is defined; and
- unlabeled intervals are negatives only when the annotation track is exhaustive.

Candidate durations are selected from development event-duration quantiles and then frozen. This is
how the system addresses unknown event duration; it does not stretch source labels or assume one
correct scale. Events shorter than the compared encoder's temporal support are reported in a
separate resolution-limited stratum. Candidates overlapping conflicting source events remain
unassigned.

Source intervals additionally provide an oracle-event matching control, but the primary deployed
condition searches the complete timeline. No generic motion-proposal recall can cap discovery.

### 2.3 Pair construction

Split subjects and source sessions before constructing events or pairs. A batch contains balanced
positive and negative pairs:

- positives are independent executions with the same identity, preferably from different sets,
  sessions, or subjects;
- ordinary positives include natural changes in duration, amplitude, and execution style;
- negatives are different labeled actions from the same subject, session, and configuration where
  possible;
- similar actions are useful hard negatives once the initial random-negative model is stable;
- simultaneous sensor views of one execution are linked views, not independent events; and
- augmented copies are robustness examples, not substitutes for real positive executions.

Do not sample repetitions from one set across train and validation. Their strong temporal and subject
correlation would make generalization appear better than it is.

### 2.4 Minimal learned matcher

The primary learned component is deliberately small:

1. a frozen encoder exports a normalized sequence of timestamped patch embeddings for each event;
2. an optional normalized diagonal weighting or linear projection is applied to each patch;
3. a constrained temporal matcher compares the two sequences while allowing bounded speed change;
4. an affine calibration maps the resulting distance to a same-motion probability; and
5. balanced binary cross-entropy trains the projection and calibration.

Use constrained soft-DTW when gradients through alignment are required. Report ordinary constrained
DTW and pooled cosine matching as simpler controls. Fine-tune the encoder end to end only after the
frozen-encoder comparison identifies a representation limitation.

Train this small component separately for every frozen released encoder. HALO is additionally
reported with a task-specific end-to-end arm. The current implementation provides projected
pooled-candidate affinity; temporal DTW re-scoring remains a required control and must not be
described as implemented training behavior until it is connected to the complete candidate path.

The first loss is simply:

```text
target = 1  when the two events have the same source identity
target = 0  when their source identities are explicitly different
loss   = balanced binary cross-entropy(match_probability, target)
```

No separate variance or anti-collapse objective is required: mapping every event to one vector would
misclassify all negative pairs. Monitor positive and negative score distributions to verify this in
practice.

## 3. Deployment grouping

### 3.1 Candidate comparison and temporal decoding

Use pooled cosine similarity as a fast screen, then apply the exact temporal matcher only to plausible
pairs. Do not compare a candidate with temporally overlapping copies of itself. Multiple scales and
nearby starts will generate duplicate descriptions of one event; consolidate them with a declared
temporal non-maximum suppression, Soft-NMS, or weighted interval-selection rule before counting
occurrences. Keep direct variable-length matrix-profile search over raw or frozen features as a
non-learned literature baseline.

### 3.2 Recurrence graph

Create an undirected edge when a pair's calibrated same-motion probability passes a threshold chosen
on development subjects and held-out identities. Mutual-nearest-neighbor edges are the initial
guard against weak one-way chaining. Connected components are the simplest clusterer; compare
HDBSCAN or Leiden only if measured fragmentation or chaining requires them.

Low-confidence events remain unassigned. The number of clusters is never supplied from hidden test
labels.

### 3.3 Recurrence count and bout granularity

Three decisions remain separate:

1. **identity:** the learned matcher decides which temporal candidates represent the same motion;
2. **recurrence acceptance:** a user-selected `min_occurrences` controls which clusters are shown;
3. **bout grouping:** nearby occurrences from one cluster may be grouped using a maximum temporal gap
   or a requested number of repetitions per group.

The recurrence count is therefore a review and presentation parameter, not part of semantic cluster
identity. This prevents a choice such as "show actions repeated ten times" from changing what the
model considers the same action.

### 3.4 Output contract

Each cluster contains:

- an artifact-scoped cluster identifier;
- all non-overlapping occurrence intervals and pair scores;
- a real medoid and several diverse examples;
- duration, cadence, and inter-occurrence-gap distributions;
- recurrence count and cumulative exposure time;
- temporal dispersion across the recording or shift;
- within-cluster compactness and nearest-negative separation;
- proposal, threshold, and boundary stability;
- source sensor configurations and missing-data summaries; and
- reviewer status: `unreviewed`, `confirmed`, `duplicate`, `noise`, or `uncertain`.

## 4. Data construction and dataset roles

### 4.1 Exact or near-exact event supervision

| source | useful annotation | intended role | current limitation |
|---|---|---|---|
| **OpenPack** | 53,760 fine-action and 20,264 operation rows in the inspected release, box-cycle identities, NULL, and synchronized occupational sensors | strongest occupational metric-learning source and official held-out-subject evaluation | one workflow and 16 people; collapse five identity aliases and filter one zero-duration action before splitting |
| **Bodyweight Exercise Segmentation** | exact set/rest intervals and 4,756 repetition-start markers across 13 continuous workouts | controlled repetition training and boundary development | raw integer acceleration/gyroscope scaling is not documented well enough for HALO ingestion; verify units first |
| **CrossFit** | 5,461 non-NULL repetition arrays across 10 exercises; the paper reports 54 participants | strongest subject-diverse repetition training source | structured workouts, six fragments under 0.2 seconds, and a 57-code release map that must not be treated as 57 distinct people |
| **AIDLAB-HAR** | series intervals and 1,486 repetition-marker fiducial windows for 13 exercises plus three background-like activities | small proposal and temporal-anchor control | marker windows are not full repetition intervals, `SUBxx` codes are not global participant IDs, and released chest data lack raw gyroscope |
| **C-MHAD** | video-verified intervals in continuous two-minute recordings | sealed arbitrary-event and recurrence control | short recordings and only 12 subjects |

### 4.2 Weak event or count supervision

| source | useful annotation | valid use | limitation |
|---|---|---|---|
| **RecoFit** | 126 complete visits from 94 subjects in the selected continuous file, exercise-set intervals, total repetition counts, and natural non-exercise background | set-level consistency, count, and weak localization | no individual repetition timestamps |
| **MM-Fit** | set intervals, repetition counts, synchronized video and pose, and multiple consumer devices | development timelines and privileged boundary derivation | consumed by current HALO pretraining; current converter stores set excerpts |
| **CaRa** | 50 activities and sequence-level counts over about 35,787 repetitions | weak recurrence/count supervision and held-out-class tests | no complete occurrence boundaries |
| **DWC/ExRAC** | sequence counts and three localized spoken exemplars | exemplar-conditioned counting and weak localization | not every occurrence is bounded |
| **PHYTMO** | repeated correct/incorrect therapy series with optical reference | development of execution variation and Task-2 linkage | series-level rather than per-repetition events; consumed by HALO pretraining |

Count-only datasets cannot train the exact event-pair loss without latent occurrence inference. Keep
them in a separate weak-supervision arm rather than pretending that a sequence count supplies event
boundaries.

### 4.3 Continuous transfer and final evaluation

| source | evaluation purpose | limitation |
|---|---|---|
| **OpenPack** | known-action/unseen-subject occupational recurrence and hierarchical action-to-box-cycle grouping | not an unseen dataset if used for matcher training |
| **OCA** | unseen industrial assembly transfer with repeated phases and NULL | five subjects, research placements, mixed native rates, and one clock gap; convert BNO055 degree/s to rad/s |
| **WEAR** | continuous outdoor recurrence, background rejection, and video-assisted review | acceleration only and activity intervals rather than exact internal repetition boundaries |
| **CrossFit** | controlled held-out-subject and held-out-exercise repetition matching | not representative of an occupational shift |

A strong open-world experiment trains on controlled workout identities and evaluates completely
unseen occupational identities. OpenPack can additionally test hierarchical granularity because it
contains fine actions, operations, and repeated box cycles.

The central access, provenance, and converter status for every source is owned by
[`APPLICATION_DATASETS.md`](../data/APPLICATION_DATASETS.md).

## 5. Preprocessing and augmentation

The first metric model uses clean independent executions. Preserve native timestamps, hard gaps,
physical units, gravity state, sensor placement, and valid-channel masks.

Introduce one nuisance family at a time only after the clean model is measured:

- one physically consistent session-wide rotation for co-located vector sensors;
- mild whole-event retiming while retaining original duration as metadata;
- measured device noise or calibration scale; and
- honest channel loss with corresponding masks.

Do not transform only inserted events in a synthetic timeline, numerically add two IMU recordings,
or manufacture every positive by augmenting one buffer. These shortcuts create detectable artifacts
instead of realistic execution variability.

## 6. Evaluation protocol

Freeze the matcher and threshold before scoring each test condition. Labels are hidden from grouping
and exposed only to score the frozen output.

### 6.1 Required generalization splits

1. unseen subjects with action identities represented in training;
2. unseen subjects and complete held-out action identities;
3. an unseen dataset and domain; and
4. where supported, different sessions, remountings, or compatible sensor configurations.

### 6.2 Pair and clustering metrics

- occurrence precision, recall, and F1 after complete-timeline decoding;
- false motif occurrences per recording hour and occurrence-count error;
- B-cubed precision, recall, and F1;
- pairwise cluster precision and recall;
- cluster purity and mean fragments per true motif reported together;
- same-motion pair AUROC, AUPRC, score separation, and equal-error rate as secondary affinity
  diagnostics;
- adjusted Rand index as a secondary closed-annotation summary;
- repeated-event coverage;
- boundary temporal IoU and start/end error after dense temporal decoding;
- stability under small threshold and proposal-boundary changes; and
- runtime and peak memory per recording hour.

Report oracle-event matching and complete-timeline discovery side by side. This separates metric and
clustering quality from dense candidate generation and temporal decoding.

### 6.3 Human review

For each recording, show a fixed-budget list of top clusters with the medoid, diverse examples,
timeline positions, count, and cumulative duration. A reviewer marks each cluster as meaningful
work, duplicate, incidental motion, noise, or uncertain. Measure review time, inter-rater agreement,
precision among reviewed clusters, and cumulative exposure covered.

## 7. Known difficulties and required safeguards

- **Nested granularity:** an atomic reach, a packing operation, and one completed box can all recur.
  Preserve containment and report the chosen evaluation level.
- **Graph chaining:** a generic reach can weakly connect distinct actions. Monitor cluster diameter,
  nearest-negative separation, and mutual-neighbor connectivity.
- **Heterogeneous background:** NULL is an outlier region, not one coherent class.
- **Duration variation:** speed, fatigue, and pauses alter length. Use bounded temporal alignment and
  report duration separately.
- **Placement and observability:** do not force wrist, ankle, chest, and phone motions to match when
  their body dynamics are genuinely different. Stratify or condition comparisons.
- **Annotation mismatch:** repetition starts, full intervals, set intervals, and counts provide
  different supervision and must remain distinct.
- **Unannotated legitimate motifs:** an algorithm may discover real repeated motion absent from the
  source labels. Video or human review is required before counting it as a false positive.
- **Dominant periodic activity:** walking can dominate recurrence. Keep it visible but rank motifs by
  both recurrence and the declared review context.
- **Identity leakage:** repetitions from one set or simultaneous sensor views are strongly
  correlated and cannot cross splits as independent examples.

## 8. First implementation milestone

1. Build a deterministic event manifest from one exact-boundary training source, retaining subject,
   session, identity, and configuration.
2. Export complete frozen HALO and released-baseline `MotionSequence` timelines.
3. Build the shared base-stride temporal pyramid and interval-overlap targets.
4. Implement pooled-cosine, constrained-DTW, and matrix-profile floors.
5. Train the balanced pairwise matcher and record positive/negative score telemetry.
6. Calibrate recurrence and temporal-consolidation thresholds on held-out subjects and identities.
7. Implement mutual-neighbor components, unassigned candidates, grouping, and review filtering.
8. Evaluate oracle source intervals and complete-timeline discovery separately.
9. Run a cross-domain experiment from controlled exercise training to OCA or held-out OpenPack
   occupational actions.

## 9. Research basis

- Yeh et al., *Matrix Profile I: All Pairs Similarity Joins for Time Series*, ICDM 2016,
  [DOI 10.1109/ICDM.2016.0179](https://doi.org/10.1109/ICDM.2016.0179).
- Linardi et al., *Matrix Profile Goes MAD: Variable-Length Motif and Discord Discovery in Data
  Series*, Data Mining and Knowledge Discovery 2020,
  [DOI 10.1007/s10618-020-00685-w](https://doi.org/10.1007/s10618-020-00685-w).
- Alaee et al., *Time Series Motifs Discovery Under DTW*, ICDM 2020,
  [DOI 10.1109/ICDM50108.2020.00099](https://doi.org/10.1109/ICDM50108.2020.00099).
- Xia et al., *Robust Unsupervised Factory Activity Recognition with Body-worn Accelerometer Using
  Temporal Structure of Multiple Sensor Data Motifs*, IMWUT 2020,
  [DOI 10.1145/3411836](https://doi.org/10.1145/3411836).
- Bock et al., *SCAN: Learning to Classify Wearable Sensor Data Without Labels*, HASCA 2022,
  [DOI 10.1145/3544794.3558477](https://doi.org/10.1145/3544794.3558477).
- Soro et al., *Recognition and Repetition Counting for Local Muscular Endurance Exercises in
  Exercise-Based Rehabilitation*, Sensors 2019,
  [DOI 10.3390/s19030714](https://doi.org/10.3390/s19030714).
- Yoshimura et al., *OpenPack: A Large-scale Dataset for Recognizing Packaging Works in IoT-enabled
  Logistic Environments*, PerCom 2024,
  [DOI 10.1109/PerCom59722.2024.10494448](https://doi.org/10.1109/PerCom59722.2024.10494448).
