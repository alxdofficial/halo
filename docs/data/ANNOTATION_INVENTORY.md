# Temporal annotation inventory

> **Inventory of record, 2026-08-30.** This document states what temporal supervision each source
> can legitimately provide. The machine-readable inventory is
> [`ANNOTATION_INVENTORY.json`](../../applications/motion_monitoring/data/ANNOTATION_INVENTORY.json).

## Why this inventory exists

The three application tasks require different evidence. An exercise-set interval, a repetition
count, a marker near a repetition, and a complete action start/end interval are not interchangeable.
The inventory records both the original source capability and the form currently exposed by HALO.

Regenerate measured application-cache counts with:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.data.annotation_inventory
```

## Capability matrix

| source | current form | exact instance start/end | continuous source timeline | primary use |
|---|---|---:|---:|---|
| **C-MHAD** | continuous two-minute recordings with video-verified action intervals | yes | yes | sealed Task-1 and Task-3 event evaluation |
| **OpenPack** | continuous work sessions with fine-action, operation, and box-cycle intervals | yes | yes | Task-1/3 training and occupational evaluation |
| **OCA** | continuous sample-labeled assembly sessions | yes, at assembly-phase level | yes | occupational transfer and recurrence evaluation |
| **XRF V2** | Phase-A excerpts; raw video-aligned scenes remain local | yes in source | yes | reconstruct for Task-1/3 training, including `pouring_water` |
| **CrossFit** | parent exercise arrays and author-provided repetition excerpts | yes | limited workout sequence | controlled repetition training |
| **HARMES** | Phase-A excerpts; raw wrist recording and event logs remain local | yes in source | yes | reconstruct wrist-ADL Task-1/3 training timelines |
| **WEAR** | continuous outdoor activity and NULL intervals | activity bouts only | yes | long-duration false-alarm and coarse-bout evaluation |
| **RecoFit** | continuous exercise visits with set intervals and counts | no | yes | background, set matching, and count supervision |
| **AIDLAB-HAR** | series intervals and short repetition-marker windows | no | yes, within short recordings | boundary and temporal-anchor control |
| **MM-Fit** | Phase-A set excerpts; full workouts recoverable | no, sets plus counts | yes in source | weak supervision and video/pose development |
| **MoniPar** | continuous weekly exercise protocol | exercise blocks only | yes | Task-2 longitudinal association after severity audit |
| **PHYTMO** | bounded correct/incorrect therapy series | series only | no | Task-2 known-change development |
| **KneE-PAD** | short correct and incorrect exercise trials | trial extent only | no | Task-2 known-variant evaluation candidate |
| **SPAR** | shoulder-exercise bouts containing repetitions | bout only | no | Task-2 repeatability, not localization |

## Measured application caches

The generated JSON measures the real converted payload rather than copying publication counts. At
the current revision it records:

| source | recordings | events | temporal annotation kinds |
|---|---:|---:|---|
| C-MHAD | 240 | 2,039 | action instances |
| OpenPack | 105 | 75,040 | fine actions, operations, box cycles |
| OCA | 13 | 4,618 | sample-label runs |
| CrossFit | 5,923 | 5,917 | exercise sequences and repetition excerpts |
| WEAR | 24 | 1,460 | activity bouts and background |
| RecoFit | 126 | 4,814 | sets, background, and source-junk intervals |
| AIDLAB-HAR | 180 | 1,616 | series and repetition fiducials |

Counts at nested annotation levels must not be added as though they were independent events. For
example, one OpenPack time span can simultaneously belong to a fine action, an operation, and a box
cycle.

### Cohort boundary and resolution audit

The 2026-08-31 audit loaded every `COHORT_V1` recording and checked every interval against its real
sensor clocks. All events are finite and contained in at least one source stream. Three same-track
overlaps require task-manifest policy rather than silent repair: two conflicting C-MHAD test events
and one 35 ms OpenPack fine-action boundary overlap.

Temporal resolution is a material eligibility condition. OpenPack contains 17,849 fine actions under
one second, and OCA contains 1,429 sample-label runs under one second. A one-second representation
cannot claim fine boundary localization on those intervals. Task manifests must report event-duration
strata, mark intervals unsupported by an encoder's temporal receptive field, and provide a common
resolution-supported comparison. Source intervals are never lengthened to make them eligible.

## Task-specific admission rules

### Task 1

Training may use either natural continuous recordings with exact target intervals or synthetic
timelines assembled from independent bounded executions and compatible real background. Primary
evaluation requires a natural continuous timeline and real action-instance intervals. Set counts or
fiducial markers are not sufficient event ground truth.

Reference/query construction must retain the source interval exactly. The primary sequence-matching
stratum requires at least two valid representation positions in the reference; single-position
events are reported separately as a nearest-patch condition. Conflicting overlapping source events
are excluded from primary event scoring unless an independent annotation audit resolves them.

### Task 2

The critical evidence is not background annotation. It is repeated bounded executions plus an
external change variable, such as session time, an accepted/altered execution condition, or a
clinical measurement. Long timelines are useful for end-to-end evaluation but are not a prerequisite
for the aligned change measurement itself.

The frozen seven-source cohort is sufficient for global metric development and a controlled
synthetic benchmark, but not yet for a strong real longitudinal claim. The measured within-person,
same-action recording support is concentrated in OpenPack: its training partition has 345 fine-action
subject/action groups with at least three recordings. AIDLAB-HAR and CrossFit have one stored
recording per subject/action; RecoFit has at most two sets per subject/action. MoniPar's repeated
weekly protocols therefore remain the priority real repeated-session adapter.

Synthetic Task-2 splits operate on disjoint bounded executions and must split subjects, source
recordings, sessions, and actions before generating changes. They do not require a continuous source
timeline. Synthetic evaluation establishes controlled sensitivity and nuisance rejection, but does
not replace real known-variant or longitudinal validation.

### Task 3

Training requires arbitrary event identities and exact intervals where available. Deployment uses a
dense multiscale search over the full timeline; source intervals are hidden from discovery and used
only for its loss or evaluation. Set/count datasets remain explicitly weak supervision.

The multiscale duration set is selected from development event-duration distributions and frozen
before test. Candidates intersecting conflicting source events or ambiguous partial overlaps receive
no pair target. Oracle source-interval matching and complete-timeline discovery remain separate.

## Open adapter work

1. Reconstruct XRF V2 scene timelines directly from the retained HDF5 signals and temporal-action
   JSON. Do not use the Phase-A excerpts, reset clocks, or six-second minimum.
2. Reconstruct HARMES full wrist timelines with their event logs and clock correction while retaining
   inter-event background and hard sensor gaps.
3. Reconstruct MM-Fit full workouts only if its set/count/video supervision is needed after the exact
   event sources are measured.
4. Expose and audit MoniPar severity/session alignment before using it for a Task-2 association.
