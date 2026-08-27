# data/datasets

One subfolder per dataset. Each `<name>/` holds the **dataset-specific** pieces:

- downloads / upstream source (**gitignored** — regenerated from the converter)
- converter / preprocessing script(s) that turn the raw source into per-subject sessions
- metadata, channel descriptions, and per-dataset notes (e.g. which device/placement/channels we keep,
  gravity state, sampling rate, any known data-quality caveats)

Shared, **cross-dataset** logic (unit/gravity canonicalization, the device/channel-selection policy,
harmonised-vs-raw assembly, augmentations, the setup-all entry point) lives in [`../scripts`](../scripts),
not here.

The old generic-HAR train/evaluation roster is no longer the application protocol. Dataset roles for
demonstrated-action detection, movement comparison, and recurrent-motion discovery are defined in
[`../../docs/data/APPLICATION_DATASETS.md`](../../docs/data/APPLICATION_DATASETS.md). In particular,
an encoder pretraining source is not automatically an independent application test source.

The gridded training corpus and old held-out HAR roster remain available for representation training
and historical reproduction. Complete converted sessions, timestamps, gaps, subject identity, and
execution provenance are the authoritative inputs for the new application tasks; six-second grids
must not be mistaken for complete recordings.

New data sources still require a locally readable publication or official protocol under
[`../../references/datasets`](../../references/datasets), verified acquisition metadata, and an
explicit task role before they enter an application experiment.
