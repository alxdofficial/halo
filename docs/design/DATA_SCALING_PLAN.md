# Phase-A data scaling plan (authoritative, updated 2026-08-12)

This replaces the historical acquisition survey. It records what is actually accessible, implemented,
materialised, and allowed into each experiment.

## Experimental rule

Keep two claims separate:

1. **Technique comparison:** preserve the original 12-source recipe as `pretrain --corpus matched`
   (with a separately rebuilt, corpus-fitted sensor-bias artifact).
   This is the arm used to argue that the method, rather than extra data, improves results against
   corpus-matched baselines.
2. **Expanded design-of-record:** `pretrain --corpus expanded` is the current default. It adds
   DSADS, Forth-TRACE, Opportunity, REALDISP, MM-Fit, and PHYTMO for real placement and execution
   variation. This arm must be called expanded, and a baseline comparison against it is
   corpus-matched only after the baseline has been refitted on the same 18 sources. KneE-PAD remains
   an explicit short-window/clinical-stress ablation rather than part of this default.
3. **Additional scale ablation:** ExtraSensory, bounded NHANES, and H-MOG remain opt-in through an
   explicit `--datasets` roster. They must never enter either named recipe implicitly.

Optional datasets require both `build_grids --dataset ...` and an explicit `pretrain --datasets ...`
roster. Missing requested grids fail fast. Both named recipes resolve to explicit tuples that are
persisted in checkpoints, so a later default-roster change cannot alter an artifact rebuild.

**The scaling arm also shifts composition, so read a null result carefully.** The two optional
free-living sources, ExtraSensory and NHANES, are accelerometer-only wrist data, so adding them
dilutes capture24 but also cuts the
multi-placement, gyro-bearing share. "More free-living data did not help" and "the batch got less
heterogeneous" are not distinguishable from the headline number alone; report the per-source val
breakdown alongside it.

Phase A no longer reserves synchronous event pairs. Draw shares therefore follow the documented
dataset `n^0.25` distribution with a 25% cap, followed by square-root subject tempering. Measure the
realized finite-batch shares in `corpus_audit.py`; do not mix the optional expanded roster into a
primary-corpus claim.

## Measured local corpus

| Corpus | Streams | Materialised windows | Materialised hours | Train / val at the default data seed 20260718 | Semantic labels |
|---|---:|---:|---:|---:|---:|
| Matched 12 sources | 20 | 1,783,208 | 2,899.13 | 1,589,481 / 192,617 (101 implausible + 1,009 duplicate dropped) | 93 |
| Expanded 18 sources (default) | 56 | 1,963,606 | 3,166.12 | 1,744,926 / 217,554 (102 implausible + 1,024 duplicate dropped) | 166 |
| Expanded + optional ExtraSensory/NHANES/H-MOG | 61 | 2,929,257 | 4,775.54 | 2,592,013 / 302,373 (102 implausible + 34,769 duplicate dropped) | 166 + reserved `__unlabeled__` |

The seed is `pretrain.py`'s `data_seed` default; an earlier revision of this table quoted 20260726,
which no default command reproduces. Materialised counts are what `build_grids` wrote; the train/val
counts are what `CorpusIndex` admits after the quality screens below.

### Quality screens applied at index time

Both screens cache window indices to `data/quality/` and are applied by `CorpusIndex` **and** by
`build_memory` (which reads grids directly, so it must exclude the same windows the encoder never
trained on).

- `scan_implausible` — accelerometer beyond ±16 g or gyroscope beyond ±2000 dps, i.e. outside any
  consumer full-scale range. 102 windows in the expanded corpus (KU-HAR 96, WISDM 4, PAMAP2 1,
  PHYTMO 1). The accelerometer
  half was added after an audit found accel-only streams were skipped entirely; the PHYTMO shin hit
  confirms that branch is live.
- `scan_duplicates` — byte-identical repeated windows, i.e. a device re-emitting a stale buffer
  rather than sampling. 34,835 windows across all built native grids: **extrasensory/watch_wrist
  18,421 (3.3%)**, nhanes 15,324 (13.3%), unimib_shar 940 (8.0%), kuhar 79, motionsense 34,
  xrf_v2/airpods_ear 22, and 3 in each DSADS stream.
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

Capture-24 is now uncapped: 1,543,573 native contexts / 2,560.38 h. Its full harmonised baseline view
was also rebuilt (1,530,795 windows), so HALO and compatible baselines no longer see different
Capture-24 subsets.

## Access decisions

| Candidate | Access test | Real bytes inspected | Decision |
|---|---|---|---|
| Capture-24 | Already local, full 6.4 GB source release | All 151 subjects; random raw/session/grid samples | **Enabled in primary, uncapped** |
| ExtraSensory | Official HTTP archives return 200 with stable sizes; 7.54 GB already local | Full phone, watch, labels, and author CV/platform split | **Enabled as optional labelled Phase A** |
| NHANES PAX80_G | Live CDC index exposes 6,917 archives; individual HEAD/download works | Eight deterministic participants (1.37 GB compressed), 24 spread hours each | **Enabled as optional label-free Phase A pilot** |
| H-MOG | Official Box archive returns 200; 6,132,356,276 bytes with published MD5 | All 100 nested participant archives, bundled schema, real sessions, converter and native grid | **Enabled as optional labelled Phase A** |
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

### H-MOG

- Samsung Galaxy S4 held in the hand during reading, writing, and map-navigation tasks.
- Co-located accelerometer (m/s², gravity present) and gyroscope (rad/s), nominally 100 Hz.
- The bundled data dictionary defines `Gesture_scenario` 1/2 as sitting/walking. Conversion checks
  that field against the independent odd/even `TaskID` protocol code before assigning labels.
- Sensor streams are synchronized on monotonic `EventTime`. Gaps over 250 ms split the signal; each
  continuous block is truncated to complete six-second windows, so no grid window crosses a gap.
- The official files repeat some `ActivityID` metadata rows. Protocol fields must agree before
  deduplication. One outer archive ID typo (`207969` -> unique internal ID `207696`) is retained as an
  explicit manifest alias for subject-disjoint splitting.
- Materialized: 191,535 windows / 319.23 h from 100 participants, 154.41 h sitting and 164.82 h
  walking. Full-grid screens found no physical-rail violations and no byte-identical duplicates.
- Local sources: `references/datasets/hmog/{paper.pdf,data_description.pdf}`.

## Augmentation and conditioning compatibility

ExtraSensory and NHANES are accelerometer-only; H-MOG carries real co-located accelerometer and
gyroscope. Every native grid uses the standard six-slot layout `[acc_xyz, gyro_xyz]`; absent gyro
channels are zero and masked. Real end-to-end samples verified that signal/rate/crop/rotation/gravity/
text augmentations preserve zeros and masks through both independent views and multi-resolution
collate, and the H-MOG smoke exercised all six real channels.

Configuration text is distinct:

- ExtraSensory: phone + hand, phone + trouser pocket, watch + wrist.
- NHANES: watch + non-dominant wrist.
- H-MOG: Samsung Galaxy S4 phone + hand.

Acquisition-rate metadata remains distinct from storage rate: ExtraSensory watch is observed at 25 Hz
although stored at 50 Hz; phone observability uses a conservative 30 Hz source floor; NHANES remains
80 Hz; H-MOG remains at its native 100 Hz. This prevents the filterbank from claiming unobservable
high-frequency bands.

## Sampling implications

The default temperature sampler uses `P(dataset) proportional to n_dataset^0.25` with a 25% ceiling,
not raw proportional sampling. Expected shares for the 18-source design-of-record are:

| Dataset | Expanded draw share |
|---|---:|
| Capture-24 | 15.91% |
| WISDM | 7.35% |
| REALDISP | 6.84% |
| DSADS | 6.28% |
| XRF V2 | 6.14% |
| NFI-FARED | 5.83% |
| PHYTMO | 5.70% |
| all other datasets combined | 45.95% |

The six default direct-converted additions receive their measured temperature-sampler share recorded
by the launch preflight. KneE-PAD is opt-in: only 4.7% of its trials reach six seconds, so including it
in the default recipe caused repeated exposure to a tiny muscle-belly stress corpus. Capture-24 still
contains most admitted rows without defining the representation. Per-source and per-stream telemetry
should be inspected before changing the sampler.

The production batch-1,024 / 7,500-step recipe draws 7.68 million windows with replacement: the same
sample budget as the measured batch-256 / 30,000-step reference and about 4.4 aggregate
expanded-corpus equivalents. Exposure still differs by source. Report the data roster, sampled-window
budget, batch size, and optimizer steps; “epochs” alone is not well-defined under temperature sampling.

## Launch sequence

1. Run the expanded 18-source Phase A at the sample-matched 1,024 / 7,500 production recipe.
2. Retain or rerun the matched 12-source arm when making a technique-only baseline claim.
3. Only then consider the optional ExtraSensory/NHANES/H-MOG scale arm or the KneE-PAD short/stress
   study. Prefer more NHANES subjects
   over more hours per subject; 64 subjects
   at 24 selected hours would add roughly 1,536 h while remaining a modest download.
4. Do not delay the current Phase A for another dataset. The corpus is large enough to test the model;
   its remaining weakness is wrist/config imbalance and scale relative to industrial foundation
   corpora, not a lack of trainable signal.

## Verification

The earlier four-objective smoke record was superseded when Phase A was consolidated. Current
verification is maintained in `training/tokenizer/README.md`: a launch must pass the objective-health
diagnostic, gradient check, full tests, and a consolidated JEPA + augmentation-VICReg CPU smoke. Optional-only
runs still require NHANES to have no semantic validation or Phase-B rows. No GPU training is implied
by these checks.

Primary online sources:

- ExtraSensory: http://extrasensory.ucsd.edu/
- NHANES PAX80_G documentation:
  https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAX80_G.htm
- H-MOG: https://hmog-dataset.github.io/hmog/
- Capture-24: https://www.nature.com/articles/s41597-024-03960-3
- PAAWS release status and formats: https://www.paawsstudy.org/data.html
