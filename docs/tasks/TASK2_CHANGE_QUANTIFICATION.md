# Task 2: personal change quantification

> **Design of record, 2026-09-02.** This document describes the implemented Task 2 protocol. It
> replaces earlier plans for DTW, persistent-trend modelling, DUO-GAIT, and a test-set-derived
> threshold. Those features are not part of version 1.

## 1. Question

Task 2 assumes that the action boundary and action identity are already known. Given several
accepted executions of one action by one person, and a later execution of that action, it asks:

**How far has the later execution moved outside this person's ordinary variation?**

The output is a continuous deviation score and a personal reference-only operating limit. It is
not a diagnosis, a disease-progression estimate, or a generic changed/not-changed classifier.

## 2. Data of record

| role | dataset | unit | configuration | use |
|---|---|---|---|---|
| train | HARMES | bounded dominant-wrist ADL execution | watch, acceleration + gyroscope | accepted repeats and raw-level training variants |
| train | CrossFit | bounded wrist exercise repetition | watch, acceleration + gyroscope | accepted repeats and raw-level training variants |
| evaluation | MoniPar | weekly bounded protocol execution | wrist watch, acceleration only | clinician-scored between-week change |
| evaluation | KneE-PAD | bounded rehabilitation exercise trial | eight muscle-belly IMUs, acceleration + gyroscope | correct versus released incorrect execution variants |

The generated `task2_modified_v1` cache contains deterministic variants of the two training
sources. Its current cache has 47,118 recordings. `TASK2_TEST_V1.json` freezes 13,788 comparisons:
20 MoniPar and 13,768 KneE-PAD. The small MoniPar cell must be reported as such; it is evidence of
mechanical feasibility, not clinical validation.

DUO-GAIT is not part of version 1. Repeated archive failures left only an incomplete local copy,
and substituting a different longitudinal source would silently change the intended endpoint.

## 3. Comparison contract

Every comparison contains:

1. four accepted reference executions;
2. one query execution;
3. one person, action, source dataset, and sensor compatibility key; and
4. independently bounded encoder input for every execution.

The compatibility key fixes device family, placement, channel set, and gravity convention.
Sampling rate is deliberately not part of the key because each encoder adapter owns its native-rate
handling. The event is cropped in raw sensor space before encoding. Therefore a representation of a
short reference cannot use context from the surrounding recording.

Official cross-encoder comparisons use `build_common_task2_units.py`. It retains only training
executions and sealed evaluation units available to every compared encoder, requires equal temporal
stride, and records the manifest and encoder provenance.

## 4. Training corpus

The clean HARMES and CrossFit executions are the accepted examples. A derived cache adds two kinds
of raw-signal variants before encoding:

- **Acquisition nuisances**, which remain accepted: remount rotation, sensor noise, and boundary
  jitter.
- **Physical modifications**, which are treated as differences: whole-execution retiming,
  phase-local amplitude reduction, reduced range, 4-6 Hz tremor, and an inserted pause.

Each physical root is tracked. A query cannot also be a reference, and two generated descendants
of one physical execution cannot masquerade as independent examples. Both the native six-channel
view and an acceleration-only view are materialised so the model sees the channel family used by
MoniPar. Transform type is not a label shortcut: most modified examples also receive an acquisition
nuisance.

Each training reference set contributes:

- two accepted queries from the same person and action;
- two modified queries from the same person and action; and
- one accepted execution of the same action from another person.

Different-day accepted repeats are sampled more often where the source contains them. Batches do
not mix source datasets or sensor compatibility keys. Dataset weights keep the HARMES cross-session
signal visible despite the larger generated corpus.

## 5. Learned ruler

The upstream encoder produces timestamped patch embeddings. `ChangeRuler` then:

1. optionally projects or reweights embedding dimensions;
2. linearly resamples each bounded execution to eight normalized movement-phase bins;
3. contextualises the accepted reference set with one small self-attention layer;
4. lets the query phase bins attend to the reference bins; and
5. adds a learned residual refinement before computing cosine distance to the reference prototype.

The final refinement is initialized to zero. At initialization, the learned arm exactly matches the
transparent normalized-phase cosine floor, while gradients still reach the refinement weights. As
training moves it away from zero, both deployment queries and leave-one-reference-out calibration
queries pass through the same learned path.

The training objective is a severity-scaled ranking loss. An accepted query must be closer to the
personal prototype than a modified or other-person query by a margin. A small pull term tightens
accepted queries. There is no change classifier, target-value regression, or learned global
threshold.

## 6. Deployment score

For one person/action/configuration cell:

1. each accepted reference is held out in turn and scored against the others;
2. those leave-one-out phase-residual vectors fit a robust center and scale;
3. an OAS-shrunk covariance estimates their joint variation;
4. the query receives a standardized Mahalanobis-style joint deviation; and
5. the personal operating limit is `mean(leave-one-out deviations) + 1.96 * SD`.

At least four accepted references are required. With fewer references, a deviation may still be
computed but the operating limit is reported as unavailable. The personal limit is not called an
MDC: unlike measurement-science MDC, it is not `1.96 * sqrt(2) * SEM`.

The same report is produced for the untrained floor by omitting the learned ruler. A mandatory raw
physical control uses duration and available acceleration/gyroscope magnitude, variability,
dynamic RMS, and jerk summaries under the same personal-reference protocol.

## 7. Evaluation

Results are reported separately per source **and per declared analysis cell**; cells are never
pooled, because they use different reference rules to answer different questions.

- **MoniPar, `between_week_reliability` (479 comparisons, 21 subjects):** references are the
  earliest four weekly visits, scored or not, and every later visit is an accepted query. A weekly
  repeat of the same task by the same person is accepted by construction, so this cell needs no
  clinical label and never infers one. It is the protocol's only measurement of the between-week
  noise floor, since KneE-PAD is a single visit; restricting it to clinician-scored visits would
  discard every healthy control and every remote patient, which is 178 of 196 series.
- **MoniPar, `clinician_rated_change` (20 comparisons, 4 subjects):** references are the earliest
  four clinician-scored visits sharing one released MDS-UPDRS bradykinesia score. Only later scored
  visits are queries. A score difference of at least one point defines the changed set; at least two
  points is a predeclared stricter subgroup. Unscored visits are excluded rather than labelled by
  assumption.
- **KneE-PAD, `known_difference` (13,768 comparisons, 31 subjects):** references are the earliest four released correct trials. Remaining correct trials
  are accepted queries and released incorrect variants are changed queries. This is a within-visit,
  cross-placement stress test, not longitudinal evidence.

Primary outputs are accepted-repeat false-alarm rate and within-person/action/configuration AUROC
between accepted and changed deviations. Reliability is calculated separately by action and stream
from named occasions shared across subjects; groups without enough aligned occasions are reported
as unsupported. ICC estimates are not clipped. SEM and true MDC95 are measurement summaries, not
the deployed personal threshold. Subject bootstrap intervals resample people, never executions.

## 8. Reproducible run order

1. Rebuild source and generated caches when their provenance changes:

   ```bash
   /home/alex/code/HALO/legacy_code/.venv/bin/python \
     -m applications.motion_monitoring.data.build_cache --force task2_modified_v1
   ```

2. Freeze the cohort and manifests:

   ```bash
   /home/alex/code/HALO/legacy_code/.venv/bin/python \
     -m applications.motion_monitoring.task2.build_manifests_v1
   ```

3. For each encoder, build a bounded representation cache with `build_representations.py`, using
   `TASK2_TEST_V1.json` plus bounded kinds for HARMES, CrossFit, and `task2_modified_v1`.
4. Freeze the shared encoder intersection with `build_common_task2_units.py`.
5. Run `task2/train_full.py` for the learned arm and `task2/evaluate_v1.py` both with and without
   `--head-directory`.

All runners fail closed on cohort, manifest, representation, common-unit, and checkpoint
provenance. The sealed evaluation manifest is never used to fit the ruler or its operating point.

## 9. Scope and limitations

- MoniPar's clinician-rated cell has only 20 eligible comparisons over 4 subjects after requiring a
  stable four-visit scored reference period, so responsiveness there is severely underpowered. Its
  reliability cell is far broader at 479 comparisons over 21 subjects, but answers only the noise
  question.
- No cell measures induced physiological change. That was DUO-GAIT's role before it was dropped for
  repeated download failure, and it must not be filled by substituting a free-living state source.
- KneE-PAD uses research muscle-belly sensors and one visit, so it cannot establish watch-based
  longitudinal performance.
- Generated changes test whether a ruler can learn the intended geometry; they are not substitutes
  for natural physiological progression.
- Normalized phase uses linear interpolation, not DTW. Persistent trend filtering and
  change-point detection are future extensions.
- The physical control is intentionally simple and auditable. A learned gain must be reported
  against both that control and the frozen-embedding cosine floor.

## References

- Goldsack et al., V3 framework, *npj Digital Medicine* 2020,
  [DOI 10.1038/s41746-020-0260-4](https://doi.org/10.1038/s41746-020-0260-4).
- Ratitch et al., reliability evaluation, *Digital Biomarkers* 2023,
  [DOI 10.1159/000531054](https://doi.org/10.1159/000531054).
- Kasnesis et al., KneE-PAD, *Scientific Data* 2025,
  [DOI 10.1038/s41597-025-04963-4](https://doi.org/10.1038/s41597-025-04963-4).
- Papagiannakis et al., ALAMEDA/MoniPar context, *Healthcare* 2023,
  [DOI 10.3390/healthcare11192656](https://doi.org/10.3390/healthcare11192656).
