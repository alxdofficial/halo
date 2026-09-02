# Task 1 spec: reference resolution and semantics

> **Design spec, 2026-09-01; §§A/1–4 implemented 2026-09-01; §C split redesign +
> synthetic training corpus decided 2026-09-02 and is the current implementation queue
> (§E items 6–12).** Owner for implementation: the Task-1 pipeline. This document resolves
> the sub-patch-reference problem and the long-reference problem measured on the
> 2026-09-01 data probe, and the set-level-label problem measured 2026-09-02; link it from
> `TASK1_ARBITRARY_DETECTION.md` §6 when convenient.
>
> Priority: §A (one short execution) and §§1–4 (stride grid + snapping) are **mandatory**
> and independent of each other. §B (occupancy readout) and §5 (adaptive patch scale) are
> optional arms, evaluated on development data before adoption.

## A. Reference semantics: one short execution (mandatory)

A reference is **one bounded execution of the movement, not a set, bout, or session.**
The 2026-09-01 probe showed set-length references (RecoFit median 50 patches, max 485)
can never satisfy the DTW feasibility rule inside a 60 s crop — those episodes silently
contribute zero loss-valid positions — and 44 % of RecoFit targets covered > 80 % of the
crop, degenerating detection into whole-crop classification.

Per-source enrollment rule (recorded in the task-manifest protocol block):

| source | source annotation | single-execution reference |
|---|---|---|
| OpenPack | fine action | already one execution — unchanged |
| C-MHAD | action instance | already one execution — unchanged |
| OCA | sample-label run (median 3.8 s) | unchanged; runs > 20 s enroll a leading sub-interval |
| AIDLAB-HAR | repetition fiducial | fiducial + context floor (§2) |
| CrossFit | exercise bout (~33 s, many reps) | **one author-provided repetition slice** (~2–4 s); note rep ends are machine-cue boundaries, so treat end-boundary supervision as weak |
| RecoFit | set interval + count | **exemplar prefix**: `set_duration / repetition_count` = period; enroll ~1–3 periods, bounded to [2 s, 20 s] |
| WEAR | activity bout (median 48 s, periodic) | leading excerpt of the bout, bounded to [10 s, 15 s] |

- Cap: `max_reference_seconds = 20` (global), enforced after the per-source rule.
- Train: the enrolled sub-interval is drawn with randomized phase inside the source
  annotation (per sample). Development/test: one deterministic seeded draw per unit.
- Consequence: reference lengths land in ~2–30 grid positions everywhere; the ⌈M/2⌉
  feasibility rule stops binding; detecting a set/bout becomes multiple detections of one
  movement, which the count-error metric already measures.
- Add a **loud-fail counter**: any episode whose reference yields zero feasible endpoints
  is counted and reported in training telemetry and the manifest audit (never silent).
- Stride note: with repetition-slice references, CrossFit moves to the short-event bucket
  (0.5 s stride), superseding the 1.0 s initial assignment below. After single-execution
  extraction, RecoFit references are also short (~2–6 s; see §A.1), so its stride tier
  should be revisited at the same time.

### A.1 Reference audit (how "one execution" is verified, not assumed)

Annotation kinds are classified (per `ANNOTATION_INVENTORY.md`) as **discrete-execution**
(OpenPack, C-MHAD, OCA), **repetition-container** (CrossFit bouts, RecoFit sets), or
**periodic-continuous** (WEAR). The extraction rules in the table above apply per class.
Every extracted reference then passes a manifest-time audit:

1. **Duration gate:** references outside [0.3 s, 20 s] are rejected with a recorded
   reason (a "single execution" outside that range indicates a container annotation).
2. **Embedding self-similarity periodicity check** (runs on cached embeddings; no raw
   signal needed): mean off-diagonal cosine vs lag over the candidate interval. A
   discrete-execution reference must show no internal period; a container's estimated
   period must agree with its metadata estimate (`set_duration / count` for RecoFit)
   within tolerance or a 2x harmonic. Disagreement → **loud-fail the unit** (recorded,
   excluded), never silently guess.
3. Per-unit provenance recorded in the manifest: extraction rule, metadata period,
   embedding period, audit outcome.

Measured validation (2026-09-01, 120 RecoFit sets, HALO cache at 1 s stride): 97 % of
sets show detectable internal periodicity (confirming they are multi-rep containers and
the detector works); arithmetic period median 2.05 s (p10 1.4, p90 2.7); agreement with
the embedding lag 66 % — limited primarily by the 1 s lag resolution, which cannot
resolve 1–2.5 s periods. **The periodicity audit therefore requires the fine-stride
cache (§1) to be fully effective**; at 1 s stride it serves as a coarse validator only.
Phase misalignment of the extracted period window is acceptable for RecoFit's role
(targets are set intervals, and any full period is a valid template of the cycle).

### A.2 Targets must be execution-level too (measured 2026-09-02; blocks retraining)

§A made references one execution but left **targets** as the source annotation. On
container sources that is a set/bout 10–13x longer than the reference, so a correct
single-execution detection cannot reach IoU 0.5 and the endpoint loss labels only the
bout end as positive although every execution end inside it is a true match.

Measured on the development manifest (HALO PB-04, 1 s stride, dev threshold 0.1436):

| source | ref median | target median | targets > 2x ref | direct F1 | with coalescing (0.5x ref) |
|---|---|---|---|---|---|
| RecoFit (55 % of dev, 44 % of train units) | 3.3 s | 46 s | 96 % | 0.000 | 0.023 |
| CrossFit (13 % of dev, 12 % of train) | 3.0 s | 33 s | 100 % | 0.000 | 0.124 |
| OpenPack | 1.6 s | 1.5 s | 13 % (duration variability, not multi-execution) | 0.038 | 0.049 |
| AIDLAB-HAR | 0.4 s | 0.4 s | 0 % | 0.038 | — |
| C-MHAD (test) | 2.1 s | 2.1 s | 7 % | — | — |
| OCA (test) | 3.8 s | 3.8 s | 9 % (long single ADL runs) | — | — |
| WEAR (test) | 12 s | 45 s | 82 % | — | 0.121 (§D.4) |

Pooled dev direct F1 fell 0.068 → 0.011 for this reason; the dev-selected threshold is
therefore chosen on a near-degenerate objective. **Affected: RecoFit, CrossFit (train/dev)
and WEAR (test). Unaffected: OpenPack, AIDLAB-HAR, C-MHAD, OCA.**

**Decision (2026-09-02): do not repair these labels.** The RecoFit per-rep segmenter and
WEAR occupancy scoring that were drafted here are retired. Instead the split is redesigned
so that every evaluation source is a natural long recording with clean per-execution
labels, and training is synthesized from exact single-execution donors — see §C. One
measurement from the drafting survives because it is reused there: CrossFit's release
ships every repetition as a standalone array (`np_reps_data`, median 2.5 s); on 300 sets
those arrays tile the parent set contiguously (tail after the last rep median 0.16 s) with
near-constant within-set length (CV median 0.004), only 152/3658 are bit-exact slices of
the parent, and the filename `repetition_index` is **not temporal order** (Squats_231: reps
at 3.0 s strides, index order 0,3,4,…,10,1,2). Donors are therefore taken from the rep
arrays directly; their offsets inside the parent are never needed.

## B. Occupancy readout (optional arm, evaluate before adopting)

The current dense per-patch score answers "does an occurrence **end** here?" (endpoint
DTW). A parallel readout from the same soft-DTW tables — a forward–backward pass giving
each query patch the posterior of being **covered** by a feasible alignment — answers
"is the reference being performed here?" It is order-aware (unlike naive per-patch max
cosine, which is a bag-of-moments detector and must not be used), denser in supervision,
and free of the feasibility cliff for clipped occurrences. Risks: adjacent occurrences
merge (OCA counting) and boundaries blur. Decision rule: implement as a second head from
the existing tables, compare against the endpoint head on development data (presence
AUROC, count error, boundary error); adopt as primary only if it wins there. A related
optional DP tweak: allow partial-reference alignment at crop edges (free begin/end on the
reference axis only at query-crop boundaries) so clipped occurrences supervise their
visible part.

## Problem (measured)

At the current shared 1.0 s stride, enrolled reference events shorter than ~2 s produce 0–1
usable embedding positions:

- AIDLAB-HAR fiducials (median 0.40 s): **35 %** of sampled training episodes rejected
  ("reference event contains no valid embedding patches"); **97 %** of survivors have a
  1-position reference.
- OpenPack fine actions: 34 % of events < 1 s; ~33 % of episodes get 1-position references.
- C-MHAD (sealed test): **31 % of units** have 1–2 s references → 1 position.

A 1-position reference degenerates subsequence DTW into nearest-patch cosine retrieval — a
different, weaker task — and its score distribution is pooled into the same calibrated
threshold as multi-position DTW scores, degrading both strata. Rejected units are silent
data loss.

## Fix

### 1. Shared fine-stride grid (all encoders)

- Stride menu: **{0.25 s, 0.5 s, 1.0 s}**. One value per dataset, chosen from the dataset's
  event-duration distribution on development data, recorded in the task-manifest `protocol`
  block, identical for every encoder on that dataset.
- Initial assignment: 0.25 s for C-MHAD, OpenPack, AIDLAB-HAR; 0.5 s for OCA; 1.0 s for
  WEAR, CrossFit, RecoFit (long events; no benefit).
- Receptive fields are untouched: each baseline keeps its published window (harnet 10 s,
  UniMTS 10 s, NormWear 6 s) and preprocessing; HALO keeps 1 s patches in the shared arm.
  `localization_intervals` already converts overlapping supports to midpoint cells.
- Cache impact: one representation cache per (encoder, stride tier). Provenance must record
  the stride; `CachedMotionSequenceDataset` consumers must refuse a stride that disagrees
  with the manifest protocol block.

### 2. Reference snapping (grid-quantized enrollment)

Replace exact-interval patch-center selection in `_trim_reference` /
`task1.episodes` with:

1. Snap the enrolled event interval to the **nearest grid boundary** on each side.
   Rounding up pads a sliver (< 1 step) of real surrounding signal; rounding down discards
   a sliver of the event edge. Both are bounded by half a step and are treated as
   enrollment boundary error (already a licensed augmentation).
2. **Minimum floor: 2 grid positions.** If nearest-snapping yields < 2 positions, extend
   with real surrounding context up to the floor, subject to:
   - total added context ≤ 0.5 s beyond the true event;
   - the added context is drawn **asymmetrically at random** (left/right split randomized)
     so no fixed event-position artifact is learnable;
   - the extension must stay inside the same quality-contiguous valid run (never cross an
     invalid patch or recording edge; if impossible, reject the unit with the existing
     recorded-reason mechanism).
3. **Slivers are discarded, events never are.** No unit is dropped for being long; the
   long-reference problem is out of scope here (see the separate crop/enrollment-length
   issue in the probe notes).

### 3. Determinism split

- **Training:** re-draw the snap jitter and context split every time a unit is sampled
  (cheap — index selection over cached embeddings). Enrollment boundary error becomes an
  augmentation.
- **Development/test:** one deterministic draw per unit, seeded by
  `sha256(seed, dataset, unit identity)` in the manifest-construction style, so sealed
  evaluation is reproducible bit-for-bit.

### 4. Calibration strata

Keep per-stratum threshold calibration by reference position count (buckets: 2–3, 4–15,
16+; the 1-position bucket should be empty after this fix — if not, it is reported as the
separate nearest-patch condition already required by `TASK1_ARBITRARY_DETECTION.md` §6).
Never pool score distributions across buckets into one threshold.

### 5. HALO-only adaptive-patch arm (optional, separately labeled)

HALO's variable patch length permits a second scale (e.g., 0.5 s patches) selected by
reference-duration *bucket* (rule fixed on development data). This is a **separately
labeled arm** ("HALO + adaptive scale"), never the shared-protocol row. Baselines cannot
follow (fixed input lengths); that asymmetry is a capability claim and is reported as
such, alongside the like-for-like shared-protocol HALO row and a degraded-HALO attribution
ablation (fixed-rate resampling mimicking baseline contracts) if the capability arm wins.

## C. Split redesign and synthetic training corpus (decided 2026-09-02)

Principle: **reserve the best natural data for evaluation; manufacture training data whose
labels are exact by construction.** Fixing set-level annotations (§A.2) would have meant
training on boundaries of unknown accuracy and evaluating with an invented tolerance.
Fidelity demands on the synthesis are modest because the learnable part is a linear
projection over frozen embeddings — it learns a metric, not physics — but the controls in
§C.4 are mandatory because a spliced corpus can be solved by detecting the splice.

### C.1 Sensor-config families and the new split

| source | placement | channels | natural long recording | per-execution labels | new role |
|---|---|---|---|---|---|
| C-MHAD | right wrist | acc+gyro | yes (8 h, 12 subj) | exact | **test** |
| OpenPack | right wrist | acc+gyro | yes — most natural (54 h, 16 subj) | per operation | **dev + test, subject-disjoint** (moves out of train) |
| OCA | upper arm / chest | acc+gyro | yes (6 h, 5 subj) | yes | **test** |
| AIDLAB-HAR | chest | acc only | short sessions | per rep | **dev** (chest family) |
| CrossFit | wrist | acc+gyro | no (sets only) | exact rep clips: 5,455 across 10 exercises, 57 subj | **donor bank** (train only) |
| RecoFit | right forearm | acc+gyro | 79 h sessions, 94 subj, 100 % annotated (2,147 `Non-Exercise` segments = 29 h, median 34 s) | set-level only | **background + distractor bank** (train only) |
| WEAR | both wrists | **acc only** | yes (19 h) | bout-level only | acc-only background only; never mixed into the 6-channel family; no longer a target |

- The wrist-IMU family (C-MHAD, OpenPack, CrossFit, RecoFit) is the primary arm. OCA and
  AIDLAB stay as evaluation/dev in their own families; a training corpus for those
  families is out of scope until the wrist arm is measured.
- OpenPack dev/test split: 4 of 16 subjects to development, 12 to test, chosen by the
  existing seeded leakage-group assignment; the split is recorded in `COHORT_V1`'s
  successor and the fingerprint changes. Calibration (threshold, NMS) is done on natural
  dev only — a threshold chosen on synthetic queries does not transfer.
- Nothing synthetic ever enters development or test.

### C.2 Synthetic query construction (wrist family)

One training unit = (reference, query, per-execution targets, present flag), built as:

1. **Background:** a contiguous RecoFit segment of 60–120 s (Non-Exercise, or an
   exercise set of a *different* label, or a mix across a session), or a CrossFit set of a
   different exercise. Background labels are known, so every background region is either
   neutral or a labelled distractor.
2. **Reference:** one CrossFit rep clip of exercise X from subject A (or a RecoFit
   exemplar prefix), passed through §A's snapping/floor unchanged.
3. **Inserted positives:** 0–4 rep clips of exercise X that are **different executions,
   mostly from subjects other than A**, each augmented independently (§C.3), inserted at
   distinct low-motion points. Their inserted extents are the targets — exact by
   construction. `target_present = False` units insert none.
4. **Inserted negatives:** rep clips of exercises ≠ X inserted by the *same procedure*
   in the same or other units, so that "was spliced" is uninformative. Every unit contains
   at least one inserted segment, positive or not.
5. **Seams:** crossfade (0.2–0.4 s) at both ends; the donor's low-pass gravity vector is
   rotated onto the background's local gravity before insertion; guard intervals around
   each seam are excluded from the endpoint loss (`guard_intervals_sec` / `loss_valid`).
6. **Config matching:** donor and background share placement family, channel set, and
   sample rate (resample donor to the background rate); acc-only backgrounds may only
   receive acc-only donors.
7. Synthesis is raw-level, deterministic under a recorded seed, materialized as a finite
   corpus (target ≈ 40 h of queries) and **encoded once per encoder** into a
   representation cache tagged `condition: "synthetic"` — so baselines with 6–10 s
   receptive fields see the seams the way they would in deployment. Embedding-level
   cut-and-paste is not used.

### C.3 Augmentation (applied to donors; session-level transforms to the whole query)

Time-warp ±15 %, amplitude scale 0.8–1.2, small axis rotation (≤ 15°) applied
consistently to every inserted segment of a unit plus the background (never only the
inserted segment — no watermarking), additive sensor noise at background noise level.

### C.4 Controls (must pass before any trained number is reported)

1. **Splice-leak control:** train the head on the synthetic corpus with a *random*
   reference (wrong exercise). Its event F1 on synthetic dev must be at chance; if it is
   not, the seams are a feature and the recipe is rejected.
2. **Natural-dev gate:** train on synthetic, calibrate and measure on natural dev
   (OpenPack-dev + AIDLAB). The learned head must beat the untrained direct floor there
   before anything runs on test. This is the only success criterion.
3. **Reference-identity control:** an arm where inserted positives are the reference's
   own clip (augmented) must score *higher* on synthetic dev and *not* on natural dev —
   confirming the cross-execution rule matters.
4. Per-unit provenance: donor recording ids, background recording id, insertion
   intervals, augmentation parameters, seed — all persisted.

### C.5 Retired by this decision

RecoFit as a target source; WEAR as a target source (with it §D.1 block queries and
§D.4 coalescing, which remain implemented but unused); the RecoFit periodicity segmenter;
WEAR occupancy scoring; the 2026-09-01 CrossFit-only "synthetic workout" construction
(subsumed by §C.2, which keeps its join/guard/augmentation rules).

## D. Protocol-v2 decisions (decided 2026-09-01)

1. **WEAR block queries (retired 2026-09-02 with WEAR leaving the target set; kept implemented).** A WEAR Task-1 query unit is a deterministic, non-overlapping
   **10-minute block** of a session (last partial block kept if ≥ 5 min), not the whole
   session. Rationale: sessions are ~50 min containing all 18 labels, leaving only 2
   target-absent units in the whole test manifest; median bout 48 s (p90 99 s) means a
   10-minute block holds ~5–8 bouts, so most labels are naturally absent per block —
   hundreds of natural target-absent units, no synthesis. Targets are intervals inside
   the block; references remain cross-recording. This is a manifest protocol revision
   (new fingerprint) and must land before anything is promoted.
2. **OCA target-absent condition: N/A by domain.** Cyclic assembly means every recording
   (and any work-containing block) contains all six phases; "enrolled phase absent" does
   not exist in this domain. Report the cell as unsupported with this reason; OCA's
   weight rests on count error and FA/h. Do not fabricate absent units.
3. **Multi-reference enrollment (k > 1).** Fusion rule: score each candidate against every
   enrolled reference independently, take the **minimum score** per candidate, then one
   NMS pass over the fused candidates. Start at k ∈ {1, 3}; k > 1 references must be
   independent executions under the same compatibility key, chosen by the existing
   deterministic seeded selection. Report k=1 and k=3 as separate columns.
4. **Bout coalescing (implemented 2026-09-01; unused since 2026-09-02, see §C.5).** An excerpt reference from a
   periodic activity legitimately matches repeatedly through one long bout; scoring the
   repetitions separately fails IoU>=0.5 against the full-bout target by construction
   (measured: WEAR direct F1 collapsed to 0.011 under excerpt references). Accepted
   matches whose gap is at most **0.5x the enrollment excerpt duration** chain into one
   detection (minimum member score). The gap is derived from the enrollment rule — for
   WEAR's 12 s excerpts that is 6.0 s — declared in the protocol, never tuned on test.
   Applies only to datasets whose Task-1 targets are periodic bouts (currently WEAR).
   Measured effect (stride-1.0 cache, dev-calibrated threshold): WEAR direct F1
   0.011 -> 0.121, precision 0.008 -> 0.128, FA/h 11.4 -> 3.6.

## E. Implementation queue (mechanical; in order)

Status 2026-09-02 (evening): items 1–5 are the 2026-09-01 fixes (implemented, measured);
items 6–9 are BUILT (see per-item notes), 10–12 are in flight. Nothing is committed.

1. **DONE** — numba-compiled DTW in `task1/matcher.py`; bit-identical, ~1000x.
2. **DONE** — grid snapping, 2-position floor, §A single-execution references with
   duration + raw-validity gates, loud-fail recording. Measured: training-episode
   rejections 35 %→0 % (AIDLAB), 10 %→0 % (OpenPack); 1-position references 0 %.
3. **DONE (partly retired)** — schema-v2 manifests, WEAR blocks, fine-stride HALO caches
   (0.25 s / 0.5 s). Per-stride evaluation wiring still open (item 10).
4. **DONE** — subject cluster-bootstrap CI95, reference-position and target-duration
   strata, per-dataset rejection recording in `task1/full_evaluation.py`.
5. **DONE (prior work)** — Task-0 calibration parity.
6. **DONE** — Cohort/split rebuild (§C.1): `manifests/COHORT_TASK1_V2.json`
   (`cohort_task1_v2`, fingerprint `1561aa29…`) with declared roles
   (`train_only` / `development_only` / `split_evaluation` / `evaluation`); OpenPack dev 4 /
   test 12 subjects; `TASK1_{TRAIN,DEVELOPMENT,TEST}_V2.json` from
   `task1/build_manifests_v2.py` (train 4569 units, dev 850, test 4017). `COHORT_V1` and the
   Task-3 manifests are untouched.
7. **DONE** — `data/adapters/synth_wrist_v1.py::load_donor_bank` (5427 CrossFit rep clips,
   50 subjects, 10 exercises, 1–8 s, fully valid) and `load_background_index` (RecoFit
   sessions with set/junk intervals; low-motion placement scored at synthesis time).
8. **DONE** — `synth_wrist_v1` is a `derived` canonical cache (registry policy `derived`,
   provenance = generator sha + source-cache provenance): 1603 recordings, 40.0 h,
   94 background subjects, 2246 primary + 1086 distractor inserts, every insert with the
   §C.4.4 provenance. Encoded once for HALO PB-04 into
   `artifacts/representations/app_v2_halo_pb04_synth` (stride 1.0); representation caches
   now bind per-dataset raw-cache fingerprints and union across cohorts
   (`representation_cache.open_representations`).
   Manifest-level reference rule: `reference_identity: donor` (never the target's donor clip;
   different donor subject where possible — verified 0/0 violations).
   Known deviation: 567/1603 recordings ended with zero primary inserts (config weight 20 %)
   because placement can fail when RecoFit sets dominate a window; the metadata records
   what was actually inserted.
9. **DONE (built; results pending)** — `task1/controls_v2.py` runs the splice-leak control
   (wrong-exercise references; chance = untrained matcher under the same wrong references)
   and the reference-identity control (same-donor-clip references paired against the
   cross-clip rule on identical units) on a background-subject hold-out inside the
   synthetic corpus; the natural-dev gate is in `train_full.py` (`natural_dev_gate`).
10. Per-stride cache selection in evaluation/training (one cache dir per stride tier);
    baseline fine-stride caches for the new test set.
11. Retrain the head on the synthetic corpus; report natural-dev gate, then test —
    together with the dev-only checkpoint A/B (item 12).
12. **Checkpoint A/B (dev only)** — long_4h vs PB-04 on development sources using
    `app_v1_halo_long4h_dev`; the 2026-09-01 C-MHAD event-AUC probe (PB-04 0.508 = chance
    vs UniMTS 0.688) is diagnostic only, never selection.
13. Open/optional: §D.3 k=3 fusion; §B occupancy readout; §5 HALO adaptive-scale arm.

Measured context for prioritization (2026-09-01, HALO direct, current data): C-MHAD
transferred-threshold F1 0.106 vs **oracle-threshold F1 0.110** — threshold transfer is
not the bottleneck; the matching floor is, and its leading measured cause is the
reference-resolution stratum (31 % of C-MHAD units are 1-position references). Training
telemetry: endpoint objective learns (train endpoint F1 0 → 0.32) but development event
F1 is flat (0.068 → 0.072) — objective/evaluation misalignment, not optimizer failure;
revisit the learned arm only after items 1–3 land.

## Performance prerequisite

Stride 0.25 s multiplies DTW cost-matrix work ~16× versus 1.0 s. The current
`task1/matcher.py` hard-DTW is a Python-level loop and is already the slow path.
**Vectorize the DTW recurrence (anti-diagonal or compiled) before enabling the 0.25 s
tier**, or the full evaluation becomes impractical. The soft-DTW training path has the
same scaling; the 60 s training crop bounds it, but budget before launching long runs
(size runs first).

## Acceptance checks (before/after, on the probe script)

1. Reference-rejection rate for AIDLAB-HAR ≤ 5 % (from 35 %) and OpenPack ≤ 2 %.
2. 1-position reference fraction ≈ 0 in train and C-MHAD dev-side units.
3. Position-count histograms per dataset recorded in the manifest audit.
4. Score-distribution overlap between strata measured (report, no target).
5. No change to any baseline's published window or preprocessing; caches carry stride in
   provenance and loaders enforce it.

## Files touched (expected)

- `applications/motion_monitoring/build_representations.py` (per-dataset stride)
- `applications/motion_monitoring/representation_cache.py` (stride in provenance + check)
- `applications/motion_monitoring/task1/episodes.py` (`_trim_reference` → snap + floor +
  jitter; deterministic eval draw)
- `applications/motion_monitoring/evaluation_manifests.py` (protocol block: grid per
  dataset; stratum definitions)
- `applications/motion_monitoring/task1/train_full.py` (per-sample jitter; per-stratum
  calibration)
- `applications/motion_monitoring/task1/matcher.py` (vectorization prerequisite)
