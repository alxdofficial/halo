# Frozen application manifests

These files freeze data membership and evaluation units. They are protocol artifacts, not mutable
training outputs. A manifest fingerprint changes when source-cache provenance, cohort membership,
or unit construction changes.

## Active protocols

| task | cohort | train | development | test |
|---|---|---|---|---|
| Task 1 | `COHORT_TASK1_V2.json` | `TASK1_TRAIN_V2.json` | none; synthetic training subjects provide the threshold hold-out | `TASK1_TEST_V2.json` |
| Task 2 | `COHORT_TASK2_V1.json` | `TASK2_TRAIN_V1.json` | none; the operating limit is fitted from each person's references | `TASK2_TEST_V1.json` |
| Task 3 | `COHORT_TASK3_V2.json` | `TASK3_TRAIN_V2.json` | none; the threshold comes from held-out training identities under a false-edge budget | `TASK3_TEST_V2.json` |

Files from Task 1 V1 were removed. Do not recreate or consume them.

## Task 1

Task 1 V2 uses `synth_wrist_v1` for training and C-MHAD plus OpenPack for sealed evaluation.
Reference/query comparisons are configuration-matched and contain paired same-subject and
cross-subject enrollment conditions. Rebuild with:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.task1.build_manifests_v2
```

The query timeline must be represented in full. Every enrolled reference must also be encoded as
an independently bounded input so it cannot use context outside its declared interval. Official
cross-encoder runs first freeze the shared eligible units with `build_common_task1_units.py`.

## Task 2

Task 2 trains from HARMES, CrossFit, and the generated `task2_modified_v1` variants. Its sealed
evaluation contains clinician-scored MoniPar comparisons and correct-versus-incorrect KneE-PAD
comparisons. Rebuild with:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.task2.build_manifests_v1
```

All Task 2 executions are encoded independently at their source boundaries. Official
cross-encoder runs freeze the common train-execution and test-unit intersection with
`build_common_task2_units.py`. The train/evaluate runners require that file and verify its hashes.
See `docs/tasks/TASK2_CHANGE_QUANTIFICATION.md` for the complete protocol and commands.

## Task 3

`COHORT_TASK3_V2.json` and the Task 3 V2 manifests are the active recurrence-discovery protocol.
The V1 Task-3 manifests (`TASK3_TRAIN_V1`, `TASK3_DEVELOPMENT_V1`, `TASK3_TEST_V1`) are RETIRED and
must not be consumed: their evaluation sources, C-MHAD and WEAR, carry a median of one occurrence
per identity and so cannot measure recurrence at all, and their development split has been replaced
by an a-priori operating point. `COHORT_V1.json` itself stays, because other builders still default
to it.
They use complete recording-stream timelines and retain a distinct development split for model
selection.

## Representation rules

- Representation caches carry the cohort fingerprint, raw-cache fingerprints, encoder provenance,
  patch duration, and temporal stride.
- A cache with stale provenance must be rebuilt, not relabelled.
- Official cross-encoder Task 1 and Task 2 comparisons require equal stride and a frozen common-unit
  intersection.
- Missing official units are errors. Pilot `--limit` caches are not valid result inputs.
