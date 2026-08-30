# HALO

HALO is a research system for **personalized movement monitoring from wearable IMU data**. It uses
pretrained temporal representations from phones, smartwatches, and compatible consumer wearables to
support four application tasks:

0. propose coherent motion events and estimate their start and end boundaries;
1. detect an arbitrary demonstrated movement in later continuous recordings;
2. quantify how two executions of the same movement differ; and
3. cluster and discover frequently recurring motion motifs in unlabeled occupational recordings.

The intended workflow is **propose, demonstrate or discover, detect, compare, and track**. Rehabilitation is
the primary application. Occupational monitoring is scoped to repetitive-motion exposure and drift;
the system does not infer intent, fatigue, injury, or clinical improvement without external ground
truth.

The contribution is evaluated as one end-to-end application system. The four tasks are small linked
operations, not separate papers, and Task 0 deliberately uses an established statistical proposal
method rather than claiming a new segmentation model.

Read the active design in this order:

1. [`docs/design/MOTIVATION.md`](docs/design/MOTIVATION.md)
2. [`docs/design/RESEARCH_TASKS.md`](docs/design/RESEARCH_TASKS.md)
3. [`docs/tasks/TASK0_EVENT_SEGMENTATION.md`](docs/tasks/TASK0_EVENT_SEGMENTATION.md)
4. [`docs/tasks/TASK1_ARBITRARY_DETECTION.md`](docs/tasks/TASK1_ARBITRARY_DETECTION.md)
5. [`docs/tasks/TASK2_CHANGE_QUANTIFICATION.md`](docs/tasks/TASK2_CHANGE_QUANTIFICATION.md)
6. [`docs/tasks/TASK3_RECURRENT_MOTION_DISCOVERY.md`](docs/tasks/TASK3_RECURRENT_MOTION_DISCOVERY.md)
7. [`docs/design/DESIGN_OF_RECORD.md`](docs/design/DESIGN_OF_RECORD.md)
8. [`docs/design/ENCODER_HYPOTHESES.md`](docs/design/ENCODER_HYPOTHESES.md)
9. [`docs/design/EVALUATION_PROTOCOL.md`](docs/design/EVALUATION_PROTOCOL.md)
10. [`docs/design/IMPLEMENTATION_PLAN.md`](docs/design/IMPLEMENTATION_PLAN.md)

## Current status

The application pivot agreed on 2026-08-27 is now the design of record on `main`. The encoder, data
converters, released-checkpoint baseline adapters, application source acquisition, and seven
lossless raw-timeline adapters exist. Task 0 now has a tested native-time statistical detector,
development-only calibration, and interval evaluation package. Its full calibration and held-out
visual audit remain open. The common `MotionSequence` export and Tasks 1-3 remain planned.

The previous zero-shot, k-curve, and retrieve-mix-vote research remains recoverable from Git at
commit `32267b6` and branch `archive/pre-application-main-20260830`. It is not the design of record.

## Technical foundation

- **Data:** converters preserve subjects, sessions, timestamps, sensor units, placement, sampling
  rate, gravity state, and channel validity. The new tasks must consume whole session timelines rather
  than treating six-second training grids as independent recordings.
- **HALO encoder:** physical-time frontend plus temporal patch embeddings and explicit acquisition
  metadata.
- **External representations:** a minimal primary roster of author-released HARNet, UniMTS, and
  NormWear checkpoints, used frozen through faithful adapters; ImageBind remains an optional generic
  multimodal control.
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
