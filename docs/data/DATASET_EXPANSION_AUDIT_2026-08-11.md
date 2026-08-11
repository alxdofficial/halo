# Second-opinion audit of the ten new datasets (2026-08-11)

Independent re-measurement of the NO-GO audit recorded against
[`DATASET_EXPANSION_2026-08.md`](DATASET_EXPANSION_2026-08.md) §10. Every number below was
re-derived from the raw downloads and the built grids, not carried over.

**Bottom line: the NO-GO verdict held, and the *reason list* needed two corrections and six
additions.** Nothing here is in Phase-A training or the Phase-B rosters (verified, §4), so none of it
was contaminating anything.

> **Status 2026-08-11: every finding below is FIXED.** Five converters were changed and their
> sessions and grids rebuilt; the enrollment leakage unit, the evaluation quality screen and the
> per-stream candidate vocabulary were rebuilt in the plumbing. See §6 for the resolution log and
> what it measurably changed. 443 tests pass.

---

## 1. Verdict on each prior finding

| # | prior finding | verdict |
|---|---|---|
| 1 | FORTH-TRACE timestamps unusable for `np.interp` | **confirmed**, severity re-scoped (§1.1) |
| 2 | MM-Fit fabricates phone/earbud data across gaps | **confirmed** (§1.2) |
| 3 | MM-Fit workout ≠ subject | **confirmed**, with the paper quote and split ids (§1.3) |
| 4 | Opportunity long-hole splitting emits the hole | **confirmed exactly** — 14 windows/placement, 70 total (§1.4) |
| 5 | DSADS gap detection causes label-dependent loss | **half wrong** — the count is exact, the causal story is not (§1.5) |
| 6 | Candidate sets ignore per-stream observability | **confirmed**, with per-stream counts (§1.6) |
| 7 | Additions not committed | **confirmed** — 10 dataset dirs + 8 reference dirs untracked |
| — | monipar "more-affected wrist" text | **confirmed**, and it is 34% of windows, not 7/28 subjects (§1.7) |
| — | upper_limb_use loses shared event identity | **confirmed, and understated** — it is a leak, not just lost metadata (§2.2) |
| — | non-integer rate rounding 51.2→51, 148.148→148 | **confirmed**, impact quantified as negligible (§1.8) |
| — | `sweep_new_datasets.py` calls `gravity_align` wrongly | **confirmed** — 51 of 51 streams, error swallowed, `aligned_finite: true` reported on the *unaligned* tensor |
| — | 4 datasets lack `eval_labels.json` | **confirmed** — forth_trace, opportunity, dsads, realdisp |

### 1.1 FORTH-TRACE — confirmed, two distinct defects, severity re-scoped

The release serialises the millisecond clock to **6 significant figures**, so resolution degrades as
elapsed time grows. Past `t = 1,000,000 ms` the quantum is 100 ms while the node keeps emitting
51.18 samples/s:

```
part0dev1  t≥1e6: 1946 samples, diffs ∈ {0 ms ×1565, 100 ms ×380}
           values 1000000, 1000000, 1000000, 1000100, 1000100, …
```

That is 3.66% of every ~1040 s recording. Separately, **all five nodes of `part3` carry a genuinely
non-monotonic clock** from t = 1.6 s onward (947–2,350 backwards steps each, e.g.
`1586.6, 1625.7, 1606.2, 1645.2` — two interleaved sub-sequences emitted out of order).
`np.interp` is undefined for non-increasing `xp`, so part3's grids are not merely low-passed, they
are wrong. Long gaps confirmed: 6.81, 10.20, 10.55, 10.64, 13.09, 16.83, 16.84, 36.78, 63.34 s, plus
a 1828 s gap in `part8dev5` (already excluded on label agreement).

**Severity correction.** The "20–50% high-frequency loss" figure is measured mostly *above* the
filterbank's band. Measured interp/raw band-power ratios, whole file:

| band | part0dev1 | part4dev1 | part3dev5 |
|---|---:|---:|---:|
| 0.3–3 Hz | 1.00 | 1.00 | 1.01 |
| 3–8 Hz | 0.96 | 0.96 | 0.98 |
| **8–15 Hz** | **0.86** | **0.89** | **0.82** |
| 15–25 Hz (out of band) | 0.68 | 0.89 | 0.52 |

In the tail region alone, 8–15 Hz drops to 0.47 / 1.74 / 0.29. So: real, in-band, worth fixing —
but "20–50%" describes the out-of-band figure. The blocker status rests on part3, not on the tail.

**Fix as applied (§5).** A first attempt derived time purely from the row index. That is wrong here:
`part9` and `part11` drop samples *silently*, without recording a gap, so an index-derived timeline
shifts every label after the drop and node agreement collapsed (13 → 8 usable participants). The
insight that resolves it is that the stamps lose *resolution* but not *accuracy* — a value quantised
to 100 ms is still within ~5 samples of the truth, which is useless for interpolation and perfectly
adequate for **placement**. So each row is scattered to its nearest integer slot on an ideal 51.2 Hz
timeline, rows sharing a quantum are spread evenly across it, and no sample value is ever altered.

### 1.2 MM-Fit — confirmed

Per-device totals over all 21 workouts:

| device | samples | duplicate ts | gaps > 0.5 s | max gap | mean rate |
|---|---:|---:|---:|---:|---:|
| sw_l acc | 5,087,619 | 20,945 | 1 | **74.47 s** | 103.0 Hz |
| sw_r acc | 5,142,870 | 27,535 | 1 | 7.83 s | 104.1 Hz |
| sp_r acc | 10,482,011 | **1,052,285** | 55 | 2.17 s | 212.6 Hz |
| eb_l acc | 4,175,310 | **1,163,923** | **601** | 3.66 s | 84.0 Hz |

The earbud duplicates 27.9% of its timestamps (packetised delivery) and has 601 sub-second-plus
gaps; `np.interp` draws a straight line through every one. **Additional to the prior report: the
left smartwatch has a 74.5 s hole** that also gets interpolated. The 212 Hz → 100 Hz phone decimation
has no anti-alias filter, which is a second-order issue next to the gaps.

### 1.3 MM-Fit subject identity — confirmed, with the source

The paper is explicit (§3.1): *"Ten subjects participated in the data collection; two participants
carried out six workout sessions each, one participated in two sessions, and the remaining
participants carried out one workout session each."* (2×6 + 1×2 + 7×1 = 21 ✓.)

It also gives a reproducible split (§5.1.2): train `(1,2,3,4,6,7,8,16,17,18)`, validation
`(14,15,19)`, seen-subject test `(9,10,11)`, **cross-subject test `(0,5,12,13,20)`**.

So `convert.py`'s `subject_note` — *"MM-Fit … does not release a workout-to-participant map, so each
workout is treated as its own subject"* — is half true and materially misleading. No full map is
released, but the paper states 21 workouts come from 10 people and names a person-disjoint split.
**Consequence: the only honest cross-subject cell on MM-Fit is the paper's split boundary.** Because
the full map is unrecoverable, any *other* workout-disjoint partition may put one person on both
sides — including cross-subject enrollment, which could enrol and query the same person.

### 1.4 Opportunity — confirmed exactly

`cuts = sorted({0, len(table)} | {edge for hole in long_holes for edge in hole})` puts both hole
edges in the cut list, so the middle piece **is** the hole and is emitted like any other piece. Its
sensor rows were interpolated at line 168; its label track is real, so it produces sessions.

Measured: exactly **one** such hole dataset-wide — `S2-ADL1`, 220.9 s long, containing 165.1 s of
labelled samples. `data/quality/duplicate_windows.json` independently caught it:
`opportunity/{back,left_lower_arm,left_upper_arm,right_lower_arm,right_upper_arm}` = 14 windows
each = **70 stream-windows**, matching the prior report.

### 1.5 DSADS — the number is exact, the diagnosis is half wrong

Reproduced the loss exactly: **504 of 7,600** expected windows per placement, with the same
per-class breakdown (a05 194, a06 122, a08 53).

But the implied conclusion — that this is spurious loss biasing the class distribution — does not
survive the right test. For each segment join I computed the **percentile of the join step within
that recording's interior-step distribution**. Continuous data should average 0.50:

| activity | cuts @3× | mean join percentile | windows lost |
|---|---:|---:|---:|
| a05 ascending_stairs | 254 | **0.923** | 194 |
| a06 descending_stairs | 160 | **0.862** | 122 |
| a08 moving_in_an_elevator | 88 | 0.507 | 53 |
| a18 jumping | 44 | 0.573 | 26 |
| a09 walking_in_a_parking_lot | 28 | 0.511 | 16 |
| a01 sitting | 17 | 0.528 | 12 |
| *(all others)* | ≤23 | 0.49–0.55 | ≤15 |

**a05 and a06 really are discontinuous** — more than half their joins sit above the 92nd/86th
percentile of their own interior steps, which is not what a continuous recording looks like. The
physical reading is obvious: you cannot ascend stairs continuously for five minutes, so those
recordings are bouts. Splitting them is *correct*, and the resulting class imbalance is a property
of the protocol, not an artifact.

The mechanism critique still lands, just on different activities. `a08` (mean percentile **0.507** —
statistically indistinguishable from continuous) still loses 88 joins and 53 windows, because the
threshold is `3 × median interior step` and a quiet activity's median interior step is sensor noise.
Roughly **312 of the 726 cuts are false positives costing ~190 windows**, and none of those is on
stairs.

**Fix:** replace the fixed 3× ratio with a within-recording percentile test (join step above, say,
p99.9 of interior steps). That keeps all of a05/a06's real cuts and removes nearly all the rest —
measured, a p99.99 rule fires 13 times dataset-wide vs 726.

### 1.6 Per-stream candidate observability — confirmed

`eval/data.load_eval_labels` reads one `eval_labels.json` per **dataset**; there is no per-stream
vocabulary. Measured observed-label counts:

| dataset | vocab | per-stream observed |
|---|---:|---|
| phytmo | 20 | arm/forearm ×4: **6** · shin/thigh ×4: **14** |
| upper_limb_use | 15 | control ×2: 14 · patient_affected: **8** · patient_unaffected: **9** |
| kneepad | 9 | all 8 streams: **5** |
| forth_trace | *(none)* | 11 of 16 activities survive 6 s windowing |

A phytmo arm stream is therefore scored against 14 candidates its acquisition configuration can
never record.

### 1.7 monipar placement text — confirmed, larger than stated

`deployment_policy.py:289` sets the conditioning text to `"the more-affected wrist"` for the whole
stream. Subjects are `hc01–hc07` (healthy controls), `rem*`, `sup*`. The controls contribute
**4,069 of 12,079 windows = 33.7%**, and for them the text is simply false — the note two lines
below even says so ("healthy controls wore it on the dominant hand"). Since HALO conditions on this
text, a third of monipar's windows carry a wrong acquisition description.

### 1.8 Non-integer rate rounding — confirmed, negligible

`assemble.py:68` does `int(round(src_hz))`, so 51.2 → 51 and 148.148 → 148. Measured on the built
harmonised grids: forth_trace native is 2,082 windows, harmonised is **2,083** — the extra window is
the 0.39% time stretch made visible. kneepad's error is 0.10%.

In frequency terms a 2.00 Hz cadence reads as 1.992 Hz. That is far inside any filterbank band, so
this is a correctness wart rather than a measurement problem. Worth fixing with
`Fraction(src_hz).limit_denominator()`; not worth blocking on.

---

## 2. Not in the prior audit

### 2.1 [high] Execution ids count label blocks, not recordings

**This is the most consequential finding, and it invalidates part of `DATASET_EXPANSION_2026-08.md`
§10's headline table.** Every converter mints a new session (hence execution) id for each contiguous
label block. Two "executions" of the same (subject, label) are therefore often two blocks of *one
continuous recording*, seconds apart — precisely the adjacent-window regime §9 rejected for SPAR.

Comparing executions per (subject, label) against **distinct source recordings** per (subject, label):

| dataset | exec/(s,l) med/max | recordings/(s,l) med/max | inflation | what an "execution" really is |
|---|---:|---:|---:|---|
| **monipar** | 7 / 9 | **7 / 9** | **×1.0** | one weekly visit — genuinely independent |
| **realdisp** | 2 / 12 | **2 / 6** | ×1.1 | ideal / self / mutual4–7 — separate recordings |
| phytmo | 2 / 2 | 2 / 2 | ×1.0 | the two series — separate source files |
| mmfit | 3 / 4 | 1 / 1 | ×3.0 | the three sets, same workout |
| dsads | 2 / 15 | **1 / 1** | ×3.2 | split pieces of one 5-min bout |
| forth_trace | 1 / 7 | **1 / 1** | ×1.7 | blocks of one 1040 s recording |
| opportunity | **30 / 234** | **6 / 6** | **×10.2** | blocks within one of 6 ADL runs |

So the honest enrollment picture is narrower than §10 states:

- **monipar** is the only genuine across-session source. Confirmed, unchanged.
- **realdisp** is the only other across-recording source — but its k > 1 is *intrinsically a
  configuration change* (ideal vs self-placed vs displaced), so its curve conflates enrollment with
  placement transfer. That may be a feature; it is not a plain same-subject curve.
- **phytmo** is unaffected: each series is its own source file with its own clock, so its two
  executions per (subject, label) are real. (An earlier draft of this table put it at ×2.0; that was
  an artifact of grouping its recordings by subject.)
- **mmfit** supports within-session, multi-set enrollment only. Its converter argues this
  explicitly and defensibly (sets are minutes apart, unlike SPAR's within-bout reps) — but all
  three sets live in one workout, so no across-session claim is available.
- **dsads / forth_trace / opportunity** support no honest same-subject curve at all, for the same
  reason SPAR does not.

`eval_enrollment`'s guard does not catch this: `window_level_ids` fires at
`singleton_share > 0.95`, and the measured shares are opportunity 0.66, kneepad 0.47,
upper_limb_use 0.38, forth_trace 0.20. The gate tests for *window-level* ids; the defect here is
*within-recording block-level* ids, which the gate is blind to.

**Fix:** carry a recording id through conversion separately from the block id, and make the
enrollment planner group on the recording, not the session. Then re-derive §10's table.

### 2.2 [high] upper_limb_use split ids are a cross-configuration leak

`convert.py:130` iterates the four source CSVs independently, so the left- and right-wrist bands —
worn **simultaneously during the same ADL** — get different execution ids
(`…_c01_left_wrist_buttoning_a_shirt_00` vs `…_c01_right_wrist_…_00`).

This is not merely lost metadata. Everywhere else with simultaneous placements (realdisp, dsads,
mmfit, phytmo, opportunity, forth_trace) the placement is *absent* from the execution id, so window
*i* of two placements shares an execution and cross-placement enrollment on the same instant is
correctly refused. For upper_limb_use it is **permitted**: enrol the left band, query the right
band, same second of the same recording. That would read as cross-configuration transfer.

(SPAR also puts the placement in the id, but there it is right — left and right shoulder bouts are
sequential and genuinely different physical events.)

### 2.3 [medium] The quality-exclusion caches are training-only

`training/tokenizer/pretrain_data.py` and `training/evidence/build_memory.py` both load
`duplicate_windows.json` + `implausible_windows.json`. **`eval/data.py` does not** — no reference
anywhere in `eval/`, and `eval_enrollment` draws its support and query windows through
`eval.data.load_eval_stream`.

This is live today, not hypothetical: `motionsense/phone_front_pocket` is a **Phase-B dev stream**
and carries 34 flagged duplicate windows of 4,534. It is small (0.75%) but it is currently inside
the enrollment evaluation. For the new datasets it matters more — Opportunity's 70 synthetic
hole-windows are excluded from training and the memory bank, and served to evaluation.

### 2.4 [medium] phytmo and kneepad vocabularies are text-degenerate

HALO scores candidates from label *text*. Ten of phytmo's 20 labels are exact
`X` / `X_performed_incorrectly` pairs; six of kneepad's nine are `squat` / `squat_with_…` and
`walking` / `walking_with_…` variants. Under a language-conditioned scorer those pairs are close to
indistinguishable **by construction**, independent of how good the encoder is.

This is a different problem from §1.6 (observability) and needs a different answer: score these
datasets as *exercise identity* (phytmo 10-way, kneepad 3-way) and report correct-vs-incorrect
execution as a separate binary cell, rather than as one flat 20-way / 9-way vocabulary.

### 2.5 [low] SPAR converter docstring contradicts its own behaviour

Lines 15–17 state *"NO events.json is written"*; line 192 writes one. The **behaviour** is right
(one file = one execution, placement-scoped, honestly yielding k = 1), so this is a stale comment on
an otherwise carefully documented file. Separately, the emitted ids get a doubled dataset prefix:
`spar:spar:s01:bent_over_row:left_wrist:0`.

---

## 3. Confirmed clean

- All 51 streams load; shapes, finiteness, canonical channel order, zero padded slots, masks,
  gravity magnitude and gyro ranges all pass, as previously reported.
- Cross-placement simultaneity: window *i* of every placement carries the same label and subject,
  agreement 1.0000, for all seven co-located sources.
- 51 streams × totals reproduce exactly: **155,936 stream-windows**.
- **433 tests pass** (the prior audit ran a 115-test subset).

One caveat on the prior "zero subject overlap" line: true at the id level, but for MM-Fit
workout-disjoint is not person-disjoint (§1.3), so it should not be stated unqualified.

---

## 4. Containment — none of this is in training

Verified directly rather than assumed:

- `training.tokenizer.pretrain_data.TRAIN_DATASETS` — the 12 primary sources; none of the ten.
- `OPTIONAL_PHASE_A_DATASETS` — extrasensory, nhanes, hmog only.
- `deployment_policy.PRIMARY_EVAL_DATASETS` — the 7 legacy eval sets only.
- `policy.PHASE_B_DEV_DATASETS` / `PHASE_B_TEST_DATASETS` — unchanged.
- `build_memory.py:265` takes its roster from the **checkpoint's recorded** `train_datasets`, not
  from the current constant, so a roster edit cannot retroactively pull new data into an existing
  bank.

The ten are reachable only through an explicit `--datasets` argument.

---

## 5. What was changed

### Converters (sessions and grids rebuilt)

| file | change |
|---|---|
| `forth_trace/convert.py` | The released stamp column is no longer an interpolation abscissa. Each row is placed on an ideal 51.2 Hz timeline at the nearest integer slot; rows sharing a quantised stamp are spread evenly across their quantum rather than de-tied onto consecutive slots (de-tying skipped a slot every time 100 ms failed to divide 19.53 ms and shredded the recording into ~19,000 one-slot holes). Empty runs longer than 1 s are real dropouts and split the recording; shorter ones hold the preceding sample. **No resampling happens at all** — the emitted signal is the raw sample values. |
| `mmfit/convert.py` | Grid instants that any of the four devices did not observe are marked unobserved, and a labelled set overlapping one is dropped whole rather than shipped as an interpolated straight line. `subject_note` now states the paper's 10-participant structure and records its reproducible split under `paper_splits`. |
| `opportunity/convert.py` | The long-hole cut list no longer emits the hole itself as a piece. |
| `dsads/convert.py` | Segment-join discontinuity is a percentile within the recording's own interior-step distribution, not a fixed 3× the median. |
| `upper_limb_use/convert.py` | Gains `recording_id`, which is what closes the cross-configuration leak (§2.2). |
| six converters | Gain a `recording_id(session_id)` rule and write `recordings.json`. |

### Plumbing

| file | change |
|---|---|
| `eval/data.py` | `execution_ids` is now the recording, composed from `recordings.json` + `events.json`; the old value is kept as `block_ids` and `execution_granularity` says which is in force. The duplicate/implausible caches are applied, with `quality_screen` reporting `applied` / `unavailable: <reason>` so a silent empty screen is impossible. `load_eval_labels(dataset, stream)` honours a per-stream protocol vocabulary. |
| `build_grids.py` | Session directories absent from `labels.json` are orphans from an earlier converter run and are skipped, with a count printed. Without this, MM-Fit's 61 dropped sets would have survived in the grid and defeated their own fix. |
| `assembly/assemble.py` | Resample ratio from `Fraction.limit_denominator`, not `round`: 51.2 Hz is now exactly 75/64 rather than 51→60. |
| `eval_enrollment.py` | The artifact records `query_execution_granularity` / `support_execution_granularity` per protocol cell. |
| `deployment_policy.py` | monipar placement text is "the wrist"; the affected-side detail moved to the note. |
| `debug/sweep_new_datasets.py` | `gravity_align` is called with its real signature and allowed to raise. |
| `curate/build_recording_maps.py` | New. Re-applies each converter's own `recording_id` to the sessions already on disk, so an existing conversion gains the map without re-reading the raw archives. |
| `eval_labels.json` ×4 | Written for forth_trace, opportunity, dsads, realdisp from each manifest's activity list. |
| `eval_labels.json` phytmo | Per-stream protocol vocabulary (arm/forearm 6, shin/thigh 14). kneepad and upper_limb_use document why they are deliberately NOT restricted. |
| tests | Six new: recording grouping and its nesting invariant, the no-map case, the quality screen and its unavailable path, per-stream vocabulary, and the orphan-session guard. |

## 6. What it measurably changed

| dataset | windows/stream before | after | why |
|---|---:|---:|---|
| opportunity | 2,167 | **2,153** | the 14 fabricated hole-windows per placement are gone |
| mmfit | 1,799 | **1,570** | 61 labelled sets overlapped an acquisition gap |
| dsads | 7,096 | **7,563** | ~312 false-positive splits recovered; a05/a06's real cuts kept |
| forth_trace | 2,082 | **2,068** | one more participant usable, 1,082 s of unobserved time dropped |

Corpus total: **51 streams, 157,215 stream-windows** (was 155,936).

Independent confirmations:

- **Opportunity has disappeared from `duplicate_windows.json`.** It previously contributed 14
  near-constant duplicates per placement; the total cached duplicates fell by exactly 70. The
  fabricated windows are gone at source, not merely screened.
- **FORTH-TRACE keeps 14 of 15 participants**, up from 13, at a *higher* minimum node-label
  agreement (0.84 vs 0.80). `part8`, previously excluded at 0.13, passes once its 1,828 s dropout is
  excluded instead of interpolated across; only `part4` (0.147) still fails, reproducing the original
  measurement.
- **The sweep's gravity check is real for the first time**: 0 `gravity_align_error`, 82 streams
  aligned, all finite. Previously all 51 raised `TypeError`, were swallowed, and reported success.
- **DSADS cuts are now where the physics says they are**: ascending stairs 37 and descending stairs
  14 of the remaining splits, `moving_around_in_an_elevator` down from 88 cuts to 2.

## 7. Still open

- **Committing.** The ten converter directories and their references remain untracked.
- **The ZS-XD table's alignment is unscreened.** `non_harmonised` grids have no
  duplicate/implausible cache, so `load_eval_stream` reports `unavailable` there rather than
  silently passing. Building the caches for that alignment would move published baseline numbers,
  so it is a deliberate decision rather than a fix to slip in.
- **§2.4 reporting structure.** phytmo and kneepad still carry flat text-degenerate vocabularies;
  splitting exercise identity from correct-vs-incorrect execution needs a harness change, and the
  `eval_labels.json` notes say so rather than faking it by dropping the incorrect variants.
- **MM-Fit's paper split is recorded but not enforced.** `paper_splits` is in the manifest; nothing
  yet restricts cross-subject enrollment to that boundary.

## 8. Original recommended order of work

Nothing below is required before the next Phase-B run; the ten are inert.

1. **Commit the converters, configs and references.** Untracked converters are the only finding that
   costs something every day it stands.
2. **§2.1 recording ids.** Cheapest high-value fix, and it changes which datasets we are allowed to
   claim enrollment curves on. Do this before any k-curve is drawn on the new sources.
3. **§2.2 upper_limb_use** — one-line change to the session key, then rebuild that one dataset.
4. **§1.1 FORTH-TRACE** and **§1.2 MM-Fit** converter rewrites + grid rebuilds. Both are blockers for
   using those datasets at all; neither blocks anything else.
5. **§1.4 Opportunity** — exclude the hole piece from `cuts`; ~3 lines.
6. **§1.5 DSADS** — percentile join test; keeps a05/a06 cuts, drops ~312 false positives.
7. **§2.3 / §1.6 / §2.4 evaluation plumbing** — apply the exclusion caches in `eval/data.py`,
   add per-stream candidate vocabularies, restructure phytmo/kneepad scoring.
8. **§1.7 monipar text**, **§1.8 rate rounding**, **§2.5 SPAR docstring**, the
   `sweep_new_datasets.py` `gravity_align` signature, and the four missing `eval_labels.json`.
