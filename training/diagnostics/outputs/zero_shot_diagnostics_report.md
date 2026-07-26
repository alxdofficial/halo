# Zero-Shot Heterogeneity Diagnostic Report

Date: 2026-07-25

## Scope and protocol

- Protocol v4: 93-label vocabulary (`05853ac157dd03ad`) and subject split
  `b2030909040a1dce`.
- Full evaluation: 8 registered models x 7 primary held-out datasets = 56 complete cells.
- Metrics: macro-F1 and subject-stratified 95% confidence intervals. TNDA-HAR has one
  anonymous subject, so its interval is explicitly degenerate.
- Controlled diagnostics: up to 400 identical windows per stream under rate, channel,
  SO(3) orientation, and gravity perturbations. These are stress tests, not leaderboard
  scores.
- Placement: the only row-aligned multi-placement pair is XRF-V2 left wrist vs left
  pocket. XRF-V2 is in the training corpus, so this is an in-corpus control and is not a
  held-out placement estimate. HALO Evidence may contain the exact query windows.

## Full evaluation

| Model | Mean macro-F1 | Actual evaluation time |
|---|---:|---:|
| HARNet, corpus-matched control | 49.3 | 79.8 s |
| HARNet, legacy/off-the-shelf corpus | 45.7 | 39.1 s |
| HALO Evidence, fixed+MR checkpoint | 42.9 | 59.0 s |
| CrossHAR | 42.8 | 44.0 s |
| HALO ConSE, fixed+MR control | 38.7 | 169.5 s including head refit |
| UniMTS | 34.7 | 65.3 s |
| HALO ConSE, native checkpoint | 34.4 | 100.8 s including head refit |
| LiMU-BERT | 32.2 | 41.2 s |
| ImageBind | 11.4 | 62.3 s |
| NormWear | 5.1 | 709.2 s |

The official eight-row table, including all per-dataset confidence intervals, is in
`current_protocol_table.md`. The fixed+MR HALO and matched HARNet controls are in
`halo_fixed_mr_table.md` and `harnet_matched_table.md`.

## Main findings

1. The zero-shot bridge is the dominant HALO failure.

   On the native HALO checkpoint, a subject-disjoint supervised linear probe reaches
   84.8 mean macro-F1 on the six datasets with at least three subjects. The current
   zero-shot ConSE row reaches 32.7 on those same cells: a 52.1-point mean gap.
   Per-dataset gaps range from 37.0 to 66.3 points.

2. The newer fixed+multiresolution representation is materially better.

   With the same 93-label ConSE bridge and the same 20-stream head-fit corpus, fixed+MR
   scores 38.7 versus 34.4 for the native checkpoint. On the same fixed+MR checkpoint,
   untrained evidence retrieval scores 42.9, a further 4.2-point gain attributable to
   the transfer mechanism rather than a different encoder.

3. Novel labels remain a hard failure.

   The internal label-novelty diagnostic contains 12 dataset-label rows and 208 queries
   whose canonical labels are absent from the Phase-A label set. Accuracy is 0% for every
   one. The affected labels are `jumping_up`, `walking_right`, `walking_forward`,
   `running_forward`, `walking_left`, `biking`, `talking`, `eating`,
   `drinking_coffee`, `ramp_descent`, `ramp_ascent`, and `smoking`.

4. HALO fixed+MR is comparatively stable, but not invariant.

   Its aggregate F1 retention is 1.012 for rate, 0.973 for gyro removal, 0.937 for
   orientation, and 0.921 for gravity removal. Prediction consistency is lower for
   channel removal (0.736), showing that a stable aggregate F1 can still hide label
   swaps.

5. The evidence mechanism improves task performance but still reacts to geometry.

   HALO Evidence retention is 1.008 for rate, 1.008 for channel removal, 0.894 for
   orientation, and 0.921 for gravity removal. Channel prediction consistency is only
   0.741 despite unchanged aggregate F1; orientation costs 4.4 macro-F1.

6. Gravity handling is the largest systematic weakness for accel-only baselines.

   Gravity-removal retention is 0.595 for corpus-matched HARNet and 0.413 for UniMTS.
   LiMU-BERT is also weak at 0.566. HARNet and UniMTS show exact channel-axis retention
   because they consume acceleration only; this is architectural insensitivity to gyro,
   not evidence that they handle arbitrary missing channels.

7. CrossHAR and LiMU-BERT are sensitive to channel/orientation shifts.

   CrossHAR retains 0.816 under gyro removal and 0.846 under rotation. LiMU-BERT retains
   0.881 and 0.767, respectively, with prediction consistency near 0.5.

8. Near-floor models cannot support robustness claims.

   ImageBind averages 11.4 macro-F1 and NormWear 5.1. Perturbations sometimes improve
   their scores by chance while predictions remain poor. The diagnostic therefore reports
   mean absolute F1 movement and ratio-of-mean F1; mean per-stream ratios are explicitly
   marked unstable and near-floor streams are counted.

## Fairness controls

- HARNet legacy and matched are separate rows. The legacy row uses its historical
  nine-dataset, uncapped head-fit corpus. The matched control uses the HALO roster and
  20,000-window cap, but necessarily excludes KU-HAR and XRF AirPods because HARNet
  requires gravity-present acceleration. Matching increases mean F1 from 45.7 to 49.3.
- CrossHAR and LiMU-BERT backbones were self-pretrained on the original nine-dataset
  corpus; their current 93-label heads use the expanded head-fit corpus. This is disclosed
  and is not identical backbone pretraining exposure.
- Plain HALO and HALO Evidence originally pointed to different Phase-A checkpoints.
  Dedicated fixed+MR ConSE results now isolate the bridge comparison.
- XRF-V2 placement results are not valid held-out robustness estimates. They remain useful
  only as same-instant, subject-disjoint in-corpus controls.

## Harness changes

- Added a model-agnostic heterogeneity harness and deterministic, label-covering subsets.
- Replaced silent synthetic fallback in the HALO diagnostic with explicit `--synthetic`.
- Replaced hard-coded 59-label ConSE values in the ceiling probe with current result cells.
- Added validation-selected subject-disjoint ceiling probes.
- Added current corpus, checkpoint, embedding-path, and exact-bank fingerprints to evidence
  artifacts; rebuilt the bank with 203,929 vectors, 20 configurations, and all 93 labels.
- Fixed UniMTS package-import collision and changed its resampling to anti-aliased
  `resample_poly`.
- Added result-cell advisory locks, atomic writes, adapter/timing metadata, and run
  provenance for new evaluations.
- Corrected near-zero retention aggregation and marked in-corpus placement contamination.
- Removed stale hard-coded result values from active evidence scripts and adapter docs.

## Remaining limitations

- No held-out dataset provides row-aligned multi-placement grids. A publication-grade
  physical placement claim needs such data or a clearly different estimand.
- TNDA-HAR cannot support subject-disjoint confidence intervals, subject variability, or a
  ceiling probe because all rows have subject `unknown`.
- ConSE heads were fitted with one fixed seed. Final paper numbers should quantify head-fit
  seed variability or justify the single deterministic fit.
- The rebuilt bank invalidates the older learned evidence head and decoder. Only the
  untrained evidence mechanism is current; learned evidence artifacts must be retrained.
- Controlled perturbations isolate mechanisms but are not substitutes for natural
  cross-device and cross-placement test sets.
- The immutable `current_protocol_results/` snapshot should be used for this report. A
  separately launched legacy protocol-v3 CPU process was still active during the audit and
  could overwrite files in `eval/results/`; current runners now lock cells, but that older
  process predates the lock.

## Verdict

The diagnostic and current-protocol evaluation infrastructure is ready for quantitative
analysis. The results are not yet publication-final: select and document the primary
fairness row, rerun learned evidence artifacts against the rebuilt bank, address head-fit
seed uncertainty, and avoid making held-out placement claims from XRF-V2.
