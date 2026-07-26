# Phase-A data scaling plan (authoritative, 2026-07-26)

This replaces the historical acquisition survey. It records what is actually accessible, implemented,
materialised, and allowed into each experiment.

## Experimental rule

Keep two claims separate:

1. **Technique comparison:** the frozen 12-source corpus is the default for HALO and corpus-matched
   baselines. ExtraSensory and NHANES are absent. This is the arm used to argue that the method, rather
   than extra data, improves results.
2. **Data-scaling ablation:** explicitly add ExtraSensory and a versioned, bounded NHANES subset. This
   tests whether more free-living variation raises HALO's ceiling. It must be reported as an expanded
   data result, never substituted for the matched technique comparison.

Optional datasets require both `build_grids --dataset ...` and an explicit `pretrain --datasets ...`
roster. Missing requested grids fail fast. The exact roster and subset manifest are persisted.

**The scaling arm also shifts composition, so read a null result carefully.** Both optional sources
are accelerometer-only wrist free-living, so adding them dilutes capture24 but also cuts the
multi-placement, gyro-bearing share. "More free-living data did not help" and "the batch got less
heterogeneous" are not distinguishable from the headline number alone; report the per-source val
breakdown alongside it.

Draw shares must be measured from `TemperatureSampler`, not computed as `n^α / Σn^α`. The α=0.5
weights are only half the story: `placement_pair_fraction=0.1` reserves ~10% of every batch for
*verified simultaneous* events, which exist **only** in nfi_fared and xrf_v2, so those two are
oversampled on top of their temperature weight. Measured over 10 batches of 512 on the primary
corpus:

| dataset | pair_fraction=0 | **actual (0.1)** |
|---|---:|---:|
| capture24 | 47.4% | **43.3%** |
| nfi_fared | 6.5% | **13.4%** ← 2× |
| wisdm | 10.5% | 9.6% |
| xrf_v2 | 8.1% | **9.2%** |
| harmes | 5.8% | 4.8% |
| uci_har | 4.1% | 3.1% |
| others | — | ≤4% each |

nfi_fared has 12,263 eligible paired events against xrf_v2's 5,124, so the quota lands mostly on
nfi_fared. An earlier revision of this table quoted the analytic `n^α` shares (capture24 33.5%,
extrasensory 22.0%) and was wrong on two counts: it ignored the pair quota and mixed the expanded
roster into a primary-corpus claim.

## Measured local corpus

| Corpus | Streams | Materialised windows | Materialised hours | Train / val at the default data seed 20260718 | Semantic labels |
|---|---:|---:|---:|---:|---:|
| Primary 12 sources | 20 | 1,729,885 | 2,858.34 | 1,542,518 / 186,269 (95 implausible + 1,003 duplicate dropped) | 93 |
| Expanded (+ExtraSensory + NHANES pilot) | 24 | 2,504,001 | 4,148.53 | 2,212,911 / 256,247 (95 implausible + 34,748 duplicate dropped) | 93 + reserved `__unlabeled__` |

The seed is `pretrain.py`'s `data_seed` default; an earlier revision of this table quoted 20260726,
which no default command reproduces. Materialised counts are what `build_grids` wrote; the train/val
counts are what `CorpusIndex` admits after the quality screens below.

### Quality screens applied at index time

Both screens cache window indices to `data/quality/` and are applied by `CorpusIndex` **and** by
`build_memory` (which reads grids directly, so it must exclude the same windows the encoder never
trained on).

- `scan_implausible` — accelerometer beyond ±16 g or gyroscope beyond ±2000 dps, i.e. outside any
  consumer full-scale range. 95 windows corpus-wide (kuhar 91, wisdm 4). The accelerometer half was
  added after an audit found accel-only streams were skipped entirely; it currently catches nothing
  (peak observed |accel| is 8.0 g on wisdm) but the streams are no longer unscreened.
- `scan_duplicates` — byte-identical repeated windows, i.e. a device re-emitting a stale buffer
  rather than sampling. 34,814 windows corpus-wide: **extrasensory/watch_wrist 18,421 (3.3%)**,
  nhanes 15,324 (13.3%), unimib_shar 940 (8.0%), kuhar 75, motionsense 34, xrf_v2/airpods_ear 20.
  Where a duplicate group carries one label, one member is kept; where members disagree, the whole
  group is dropped because the label is unknowable (97 such groups on ExtraSensory's wrist, 5 on
  xrf_v2's ear stream).

  Inspecting the groups turned up three pre-existing source properties worth recording, all
  independent of the new datasets. unimib_shar's 443 groups are same-subject, same-label repeated
  trial exports of real fall motion. motionsense's 34 are near-static sitting. **kuhar's 75 groups
  are same-label but 20 of them span *different subjects*** — byte-identical vigorous motion
  attributed to two people, which would otherwise put the same window on both sides of a
  subject-disjoint split. Dropping the repeat closes that leak. xrf_v2's ear stream contributes 5
  all-zero groups: the gravity-removed earbud reported nothing and the label varied, so all 20
  windows go.

Window count is not converted to hours by multiplying by six: UCI HAR has 2.56 s pre-windowed records
and SP-SW-HAR uses 1 s windows. Hours above use each grid's actual samples/rate.

Capture-24 is now uncapped: 1,530,792 native windows / 2,551.32 h. Its full harmonised baseline view
was also rebuilt (1,530,795 windows), so HALO and compatible baselines no longer see different
Capture-24 subsets.

## Access decisions

| Candidate | Access test | Real bytes inspected | Decision |
|---|---|---|---|
| Capture-24 | Already local, full 6.4 GB source release | All 151 subjects; random raw/session/grid samples | **Enabled in primary, uncapped** |
| ExtraSensory | Official HTTP archives return 200 with stable sizes; 7.54 GB already local | Full phone, watch, labels, and author CV/platform split | **Enabled as optional labelled Phase A** |
| NHANES PAX80_G | Live CDC index exposes 6,917 archives; individual HEAD/download works | Eight deterministic participants (1.37 GB compressed), 24 spread hours each | **Enabled as optional label-free Phase A pilot** |
| PAAWS Release 2 | Official website and GitHub parser are public, but Northeastern collection returns HTTP 403 from this machine | Official sample file only; no released participant archive | **Not integrated** |

The PAAWS sample confirms ActiGraph CSV headers, 80 Hz acceleration in g, 100 Hz IMU in the lab, and
interval annotation tables. If access becomes possible, begin with free-living accelerometer-only
left/right wrist, right waist, and phone-back streams. Do not use thigh/ankle/chest placements for the
phone/watch deployment claim. A sample file cannot establish archive naming, missingness, subject
grouping, release-wide units, or download reproducibility, so no converter is committed yet.

## Source-specific plumbing

### Capture-24

- Dominant-wrist Axivity acceleration, 100 Hz, g, gravity present; accelerometer only.
- All 13,120 converted sessions are included. The two-pass writer avoids an in-RAM concatenate.
- Native and harmonised grids use the same sessions. Gyroscope slots are zero-padded and masked.
- Local source: `references/datasets/capture24/paper.pdf`.

### ExtraSensory

- Streams: explicit phone-in-pocket, explicit phone-in-hand, and Pebble watch-at-wrist.
- Phone unit conversion uses the authors' 26-Android / 34-iPhone subject split. Android is m/s2;
  iPhone is g. This avoids misclassifying one corrupted Android user's low-magnitude files.
- Phone timestamps are used directly; observed clocks vary by device. Measured over 1,500 sampled
  raw examples the per-example clock runs p01 30.3 / p50 34.1 / p75 49.6 / p99 119.2 Hz — the paper's
  40 Hz is nominal, and 60.7% of examples are delivered below it. `STREAM_SOURCE_RATE_HZ` therefore
  declares 30 Hz, the observed floor (min per-subject median 30.1 Hz), so no band above 15 Hz is ever
  claimed observable. That is deliberately conservative: it also discards real 15-20 Hz content on
  the faster examples. Storing a per-window source rate would recover it and is the natural next
  step, since the converter already computes each example's true clock and then throws it away.
- Watch files are 25 Hz (measured: exactly 25.0) and milli-g (measured: median raw ‖a‖ 1020, no file
  anywhere near the ÷1000 decision threshold). Each example is independently resampled to 50 Hz.
- **Source defect — the Pebble re-emits stale buffers.** 3.3% of wrist windows are byte-identical
  repeats, verified in the raw archive rather than inferred: subject `0A986513` has one group of 178
  identical recordings spanning 10,620 s, and `3600D531` is 87.7% duplicates. 29 of 56 subjects are
  affected. This is upstream, not a converter bug — the phone archives have zero duplicate files.
  Because the phone kept labelling those minutes independently, 36.9% of duplicate pairs carry
  contradictory labels (lying↔sitting 4,413 pairs, lying↔standing 1,110). `scan_duplicates` removes
  them; anything reading grids outside `CorpusIndex` must apply that screen too.
- Examples are truncated to complete 6 s blocks before same-subject/stream/activity aggregation, so
  grid windows never cross raw example boundaries.
- One unambiguous observable movement is required. A single stair direction supersedes a co-occurring
  generic walking tag; both directions or conflicting movements are dropped. Phone bag/table/unknown
  placements are pruned rather than assigned false deployment text.
- Materialised: 62,691 pocket + 39,502 hand + 556,744 wrist windows = 1,098.23 h.
- Local source: `references/datasets/extrasensory/paper.pdf`.

### NHANES PAX80_G

- ActiGraph GT3X+, non-dominant wrist, 80 Hz, calibrated g, gravity present; accelerometer only.
- Fetching requires a positive deterministic subject count or explicit SEQN list. There is no
  full-corpus mode: the official 2011-2012 release is about 1.04 TB compressed.
- The pilot has eight hash-selected participants and up to 24 hours per participant, selected across
  the wear interval rather than taking only the first day.
- Released QC intervals, non-finite records, and clock gaps are hard window boundaries.
- There are no activity diaries. `__unlabeled__` may feed label-free Phase A objectives, but is excluded
  from global semantic vocabulary, validation, label-text probes, and Phase B.
- Materialised: 115,179 windows / 191.97 h. One subject lost 21 six-second windows to QC/gap handling.
- **The pilot is mostly stillness, and no non-wear detector is applied.** 43.0% of windows have
  max-axis std below 0.003 g and 13.3% are byte-identical repeats of a motionless posture (all with
  std ≈ 3e-6 — benign, unlike ExtraSensory's). The published QC file only covers sensor malfunction
  (~0.26% of the release), not off-body time, so the usual Choi/Troiano non-wear step is absent.
  `scan_duplicates` removes the exact repeats; the remaining near-static mass is real free-living
  sedentary/sleep time and is left in deliberately.
- **Eight subjects is few for the weight it carries.** Under the default α=0.5 temperature sampler
  the pilot draws 9.2% of every batch from 8 people (capture24 draws 33.5% from 151). Widen the
  subject count before reading anything into a NHANES-driven result.
- Local sources: `references/datasets/nhanes/procedures_manual.pdf` and
  `references/datasets/nhanes/pax80_g_documentation.html`.

## Augmentation and conditioning compatibility

All optional streams are accelerometer-only. Their native grids use the standard six-slot layout
`[acc_xyz, gyro_xyz]`; absent gyro channels are zero and masked. Real end-to-end samples verified that
signal/rate/crop/rotation/gravity/text augmentations preserve those zeros and masks through both
independent views and multi-resolution collate.

Configuration text is distinct:

- ExtraSensory: phone + hand, phone + trouser pocket, watch + wrist.
- NHANES: watch + non-dominant wrist.

Acquisition-rate metadata remains distinct from storage rate: ExtraSensory watch is observed at 25 Hz
although stored at 50 Hz; phone observability uses a conservative 30 Hz source floor; NHANES remains
80 Hz. This prevents the filterbank from claiming unobservable high-frequency bands.

## Sampling implications

The default temperature sampler uses `P(dataset) proportional to n_dataset^0.5`, not raw proportional
sampling. Expected primary shares are:

| Dataset | Primary draw share |
|---|---:|
| Capture-24 | 48.65% |
| WISDM | 10.42% |
| XRF V2 | 7.33% |
| NFI-FARED | 6.40% |
| HARMES | 5.36% |
| all other primary datasets combined | 21.84% |

Expanded shares put Capture-24 at 33.49%, ExtraSensory at 21.97%, and NHANES at 9.19%. Within
ExtraSensory, raw window-proportional drawing makes the watch 84.5% of that dataset's samples. Thus the
expanded run is intentionally a free-living wrist-heavy scale experiment, not a balanced placement
experiment. Per-source and per-stream telemetry should be inspected before changing the sampler.

At batch 512 and 30,000 steps, training draws 15.36 million windows with replacement: about 8.9 raw
corpus equivalents for primary and 6.1 for expanded, but exposure differs by source. To preserve
approximately the primary run's Capture-24 exposure in the expanded arm requires about 45,000 steps.
Report both data roster and optimizer steps; “epochs” alone is not well-defined under temperature
sampling.

## Launch sequence

1. Run the matched 12-source Phase A at the frozen 30,000-step recipe.
2. Run the 14-source expanded pilot at the same 30,000 steps as a fixed-compute ablation.
3. If the pilot moves transfer/retrieval metrics, run a predeclared approximately 45,000-step
   exposure-matched arm.
4. Only then expand NHANES subject count. Prefer more subjects over more hours per subject; 64 subjects
   at 24 selected hours would add roughly 1,536 h while remaining a modest download.
5. Do not delay the current Phase A for another dataset. The corpus is large enough to test the model;
   its remaining weakness is wrist/config imbalance and scale relative to industrial foundation
   corpora, not a lack of trainable signal.

## Verification

- 84 focused converter/grid/policy/data-loader tests pass.
- Full expanded-roster CPU smoke completed one optimization step with finite A1, VICReg, TF-C,
  placement, EMA, and A3 losses. ExtraSensory and NHANES both appeared in per-source A1 telemetry;
  13 of 14 total sources appeared in that single stochastic batch.
- The optional-only smoke also completed; NHANES correctly has no validation or semantic probe rows.
- No GPU training was launched.

Primary online sources:

- ExtraSensory: http://extrasensory.ucsd.edu/
- NHANES PAX80_G documentation:
  https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAX80_G.htm
- Capture-24: https://www.nature.com/articles/s41597-024-03960-3
- PAAWS release status and formats: https://www.paawsstudy.org/data.html
