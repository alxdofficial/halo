# Movement-monitoring datasets

This directory owns data used by the three application tasks. It is intentionally separate from
`data/datasets`, which is the HALO representation-training corpus.

- `SOURCE_INVENTORY.json` is the single acquisition and role contract.
- `splits.py` defines the mandatory subject/linkage grouping validator for task manifests. A
  manifest is invalid if a canonical subject or conservative unresolved-identity group crosses
  train, development, and test assignments.
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

## Minimum viable study

The frozen target is five training/development sources and four evaluation sources:

| use | sources |
|---|---|
| training/development | OpenPack, CrossFit, AIDLAB-HAR, RecoFit, and existing HARMES |
| evaluation | C-MHAD, WEAR, OCA, and existing MoniPar |

Only seven sources are new. HARMES and MoniPar remain in `data/datasets` and are read in place.
The verified selective acquisitions total 6,310 files and 8,676,140,820 bytes (8.080 GiB).
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
  openpack crossfit aidlab_har recofit c_mhad wear oca --workers 4
```

The generated `sources/<dataset>/processed/canonical_v1` directories are ignored by Git. They retain
native clocks, values, masks, stream metadata, and events in independently memory-mappable records;
they do not resample, window, impute, or create task labels. `CachedRecordingDataset` is map-style,
so a standard distributed or random sampler can assign disjoint indices to loader workers without
re-reading a source archive or duplicating an epoch. Cache metadata binds the raw payload tree,
adapter source, and shared data contract; stale caches fail at load time and must be rebuilt.

## External-source rule

Every acquired source must have `references/datasets/<dataset>/SOURCE.txt` and `citation.json` with
the publication, dataset page, release scope, licence status, and any interpretation caveats. A
readable local paper or official protocol page is required for integration review. These source
documents may remain ignored for copyright reasons; the durable tracked locator is the citation
metadata and stable URL.
