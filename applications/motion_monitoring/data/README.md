# Movement-monitoring datasets

This directory owns data used by the three application tasks. It is intentionally separate from
`data/datasets`, which is the HALO representation-training corpus.

- `SOURCE_INVENTORY.json` is the single acquisition and role contract.
- `splits.py` defines the mandatory subject/linkage grouping validator for task manifests. A
  manifest is invalid if a canonical subject or conservative unresolved-identity group crosses
  train, development, and test assignments.
- `../manifests/COHORT_V1.json`, `COHORT_TASK1_V2.json`, and `COHORT_TASK2_V1.json` freeze
  task-specific recording membership. Their embedded fingerprints, rather than copied values in
  prose, are authoritative.
- `PAYLOAD_CHECKSUMS.json` freezes the accepted file set, byte sizes, and SHA-256 digests.
- `CORPUS_SUMMARY.json` records measured effective hours, event counts, and known balance caveats.
- `acquire.py` downloads only the source modalities listed in that contract.
- `inspect_sources.py` validates downloaded files and writes `inspection/summary.json`.
- `inspection/loader_profiles.json` records real cold/warm and multi-process measurements.
- `adapters/registry.py` is the common lazy-loading entry point for acquired sources.
- `verify_adapters.py` checks real decoded recordings against the common physical contract.
- `sources/<dataset>/downloads` and `sources/<dataset>/raw` are local, ignored payloads.

Do not copy an existing HALO dataset into this tree. The inventory records reusable existing
sources and their provenance; application loaders should read those sources in place.

The acquisition gate is deliberately conservative: direct public access, no required video,
normally at most 5 GiB compressed per source, and a CPU-only conversion expected to finish within
30 minutes. OpenPack is the one documented near-threshold exception because its 5.20 GiB subject
archives supply the only large occupational source with nested action, operation, and box-cycle
annotations.

## Active study

| task | fitting sources | evaluation sources |
|---|---|---|
| Task 1 V2 | `synth_wrist_v1` derived from independent CrossFit wrist executions | C-MHAD, OpenPack |
| Task 2 V1 | HARMES, CrossFit, `task2_modified_v1` | MoniPar, KneE-PAD |
| Task 3 V1 | OpenPack, CrossFit, AIDLAB-HAR, RecoFit | C-MHAD, WEAR, OCA |

Each task has its own cohort and leakage contract. A dataset may therefore have different roles in
different tasks, but one result must obey one signed task manifest.
An evaluation source must not be used to fit the task head, thresholds, preprocessing choices, or
stopping rule for the result that names it as held out. Within-source subject splits are separate
development experiments and must not be described as unseen-dataset transfer.

Acquisition refuses any file not present in the tracked checksum contract. Changing a release is an
explicit review operation, not an automatic consequence of an upstream listing changing. The
empirical acquisition inspection is tracked in `inspection/summary.json`. All seven new sources
also have lazy raw-timeline adapters. They preserve each source clock and sampling rate, express
acceleration in g and gyroscope in rad/s, expose missing values through boolean masks, split hard
clock gaps, and retain source annotation semantics in event metadata. They do not resample, window,
or synthesize task examples; those choices belong to each task pipeline.

Use the registry rather than importing a source module directly:

```python
from applications.motion_monitoring.data.adapters.registry import iter_recordings

for recording in iter_recordings("openpack", limit=8):
    ...
```

Run the real-payload adapter check with the scientific environment:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.data.verify_adapters --limit 8
```

Adapters decode one recording at a time. Task datasets must shard work explicitly when using
multiple loader processes; an iterable copied unchanged into several workers would otherwise repeat
the same source sequence. For repeated training passes, build the lossless map-style cache once:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.data.build_cache \
  openpack crossfit aidlab_har recofit c_mhad wear oca \
  harmes monipar kneepad synth_wrist_v1 task2_modified_v1 --workers 4
```

The generated `sources/<dataset>/processed/canonical_v1` directories are ignored by Git. They retain
native clocks, values, masks, stream metadata, and events in independently memory-mappable records;
they do not resample, window, impute, or create task labels. `CachedRecordingDataset` is map-style,
so a standard distributed or random sampler can assign disjoint indices to loader workers without
re-reading a source archive or duplicating an epoch. Cache metadata binds the raw payload tree,
adapter source, and shared data contract; stale caches fail at load time and must be rebuilt.

Build or validate the cohort only after all canonical caches are current:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.data.build_manifest
```

`COHORT_V1` remains the Task 3 cohort. Task 1 and Task 2 are rebuilt with their dedicated manifest
builders because their units and duplicate-view rules differ. CrossFit repetition excerpts are
excluded from Task 3 but retained for Task 2, where the repetition itself is the unit.

Frozen encoder exports use the shared representation cache and retain patch timestamps, masks, and
physical summaries:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python \
  -m applications.motion_monitoring.build_representations \
  --manifest applications/motion_monitoring/manifests/COHORT_V1.json \
  --checkpoint <frozen-checkpoint.pt> \
  --output applications/motion_monitoring/artifacts/representations/<encoder>
```

The builder refuses trainable encoders and binds each cache to both the cohort fingerprint and the
checkpoint digest. Application artifacts are local and ignored; the manifest and generation code
are versioned.

## External-source rule

Every acquired source must have `references/datasets/<dataset>/SOURCE.txt` and `citation.json` with
the publication, dataset page, release scope, licence status, and any interpretation caveats. A
readable local paper or official protocol page is required for integration review. These source
documents may remain ignored for copyright reasons; the durable tracked locator is the citation
metadata and stable URL.
