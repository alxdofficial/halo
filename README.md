# HALO

HALO is a research system for **personalized movement monitoring from wearable IMU data**. It uses
pretrained temporal representations from phones, smartwatches, and compatible consumer wearables to
support three application tasks:

1. detect an arbitrary demonstrated movement in later continuous recordings;
2. quantify how two executions of the same movement differ; and
3. discover frequently recurring motion motifs in unlabeled occupational recordings.

The intended workflow is **demonstrate or discover, detect, compare, and track**. Rehabilitation is
the primary application. Occupational monitoring is scoped to repetitive-motion exposure and drift;
the system does not infer intent, fatigue, injury, or clinical improvement without external ground
truth.

Read the active design in this order:

1. [`docs/design/MOTIVATION.md`](docs/design/MOTIVATION.md)
2. [`docs/design/RESEARCH_TASKS.md`](docs/design/RESEARCH_TASKS.md)
3. [`docs/design/DESIGN_OF_RECORD.md`](docs/design/DESIGN_OF_RECORD.md)
4. [`docs/design/EVALUATION_PROTOCOL.md`](docs/design/EVALUATION_PROTOCOL.md)
5. [`docs/design/IMPLEMENTATION_PLAN.md`](docs/design/IMPLEMENTATION_PLAN.md)

## Current status

This branch, `application-motion-monitoring`, records the application pivot agreed on 2026-08-27.
The encoder, data converters, released-checkpoint baseline adapters, and prior evaluation machinery
already exist. The common `MotionSequence` export and the three application evaluators are planned
but not yet implemented.

The previous zero-shot, k-curve, and retrieve-mix-vote research remains recoverable from Git at the
branch point, commit `32267b6`. It is not the design of record on this branch.

## Technical foundation

- **Data:** converters preserve subjects, sessions, timestamps, sensor units, placement, sampling
  rate, gravity state, and channel validity. The new tasks must consume whole session timelines rather
  than treating six-second training grids as independent recordings.
- **HALO encoder:** physical-time frontend plus temporal patch embeddings and explicit acquisition
  metadata.
- **External representations:** author-released HARNet, UniMTS, NormWear, and ImageBind checkpoints,
  used frozen through faithful adapters.
- **Initial downstream methods:** raw-signal DTW, physical-feature alignment, latent subsequence
  alignment, and matrix-profile-style motif discovery. Learned metric heads come later only if the
  frozen floors identify a representation limitation.

## Layout

```text
baselines/            # author-released checkpoint adapters and publications
data/
  datasets/           # downloads, converters, manifests, and channel descriptions
  scripts/            # curation, units, assembly, quality checks, and EDA
model/
  tokenizer/          # HALO representation encoder and frontends
  evidence/           # historical classification experiments; not the active application design
training/
  tokenizer/          # optional HALO representation pretraining and diagnostics
  evidence/           # historical Phase-B trainers retained for reproducibility
  diagnostics/        # representation diagnostics
eval/                 # prior HAR evaluation plus future shared application protocol code
docs/                 # active motivation, task, design, data, baseline, and result records
tests/                # regression tests
```

Application code will live under `applications/motion_monitoring/` so it does not inherit candidate-
label or Phase-B assumptions from the previous evaluation harness.

## Development

Use the project interpreter for torch, scipy, pandas, and h5py:

```bash
/home/alex/code/HALO/legacy_code/.venv/bin/python -m pytest tests -q
```

Raw datasets, checkpoints, caches, and generated run artifacts remain gitignored. Design decisions
and promoted result summaries are tracked so switching branches restores the corresponding research
program without duplicating stale documents.
