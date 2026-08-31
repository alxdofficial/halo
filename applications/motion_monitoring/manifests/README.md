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
