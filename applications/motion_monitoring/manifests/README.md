# Frozen application manifests

`COHORT_V1.json` is the authoritative recording-level cohort for the seven application sources with
canonical raw-timeline caches. It binds exact cache membership and provenance, excludes CrossFit's
derived repetition copies, and keeps each canonical subject or conservative identity-linkage group
inside one split. The deterministic development selection favors source annotation coverage and
keeps labels seen in only one leakage group in training.

Rebuild it only as an explicit protocol revision:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.data.build_manifest
```

The file's embedded fingerprint changes if a source cache, adapter output, split seed, role, or
recording assignment changes. Generated representation caches must record that fingerprint.

HARMES and MoniPar are not yet members because they still use the older representation-corpus
storage rather than the application `RawRecording` cache contract. Their integration requires a
separate reviewed adapter revision; they must not be appended as unvalidated index lists.

`TASK{1,3}_{TRAIN,DEVELOPMENT,TEST}_V1.json` freezes the exact units used to fit task heads,
select one global operating threshold per encoder/readout, and report the three sealed test
datasets separately. Task 1 contains enrollment/query pairs; Task 3 contains complete
recording-stream timelines. The test totals are 2,612 Task-1 trials and 340 Task-3 timelines.

## Task-1 V2 (2026-09-02, `docs/tasks/TASK1_REFERENCE_RESOLUTION_SPEC.md` section C)

`COHORT_TASK1_V2.json` (`cohort_task1_v2`) is Task 1's own cohort with declared per-dataset roles;
`COHORT_V1.json` stays the Task-3 cohort and is untouched. Roles:

| dataset          | role                | split                                   |
|------------------|---------------------|-----------------------------------------|
| `synth_wrist_v1` | `train_only`        | train (synthetic; never calibrates)     |
| `aidlab_har`     | `development_only`  | development                             |
| `openpack`       | `split_evaluation`  | development 4 subjects / test 12        |
| `c_mhad`, `oca`  | `evaluation`        | test                                    |

`TASK1_{TRAIN,DEVELOPMENT,TEST}_V2.json` are built from it by
`python -m applications.motion_monitoring.task1.build_manifests_v2`. Synthetic references are
chosen by donor identity (`reference_identity: donor`): never the same donor clip as a target,
and a different donor subject wherever one exists. RecoFit, CrossFit and WEAR are no longer
Task-1 targets (they are the synthesis background and donor sources instead).

`synth_wrist_v1` is a `derived` canonical cache: deterministic raw-level synthesis from the
CrossFit and RecoFit caches (`data/adapters/synth_wrist_v1.py`, ≈40 h). Its provenance is the
generator's sha plus the two source caches' provenance, so `build_cache --force synth_wrist_v1`
reproduces it bit-for-bit. Representation caches record per-dataset raw-cache fingerprints and
may be opened under any cohort whose raw caches match (`open_representations`), so a natural
cache built under `COHORT_V1` and a synthetic cache built under `COHORT_TASK1_V2` serve one
manifest together.

Runners: `task1/train_full.py` (multi-cache `--representations`, natural-dev gate),
`task1/evaluate_v2.py` (dev calibration → test), `task1/controls_v2.py` (section C.4 controls).

`TASK2_TEST_V1.json` is intentionally empty. None of the current sealed datasets provides
same-person, same-task bounded executions with accepted-versus-changed truth. The manifest records
that blocker so synthetic or clinical-state proxies cannot silently be reported as the real test.

Validate source identities and representation-cache coverage before a run:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.evaluation_readiness \
  --representation harnet=/path/to/harnet_cache \
  --representation unimts=/path/to/unimts_cache
```
