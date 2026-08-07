# HALO

**H**eterogeneity-**A**ware **L**anguage-aligned IMU model for **O**pen-set HAR.

HALO is an IMU foundation model for **real-world phone/watch human activity recognition** under
heterogeneous acquisition. A channel-independent, rate-invariant tokenizer produces per-patch
embeddings that survive changes in sampling rate, channel set, and sensor placement; activities are
then recognized **zero-shot** from natural-language label text — no per-dataset classifier.

Training is two phases:

- **Phase A** — label-free representation pretraining (JEPA + augmentation VICReg). Activity labels
  never enter the loss. See [`training/tokenizer/README.md`](training/tokenizer/README.md).
- **Phase B** — the evidence engine: retrieval over a memory bank of Phase-A patches, candidate-set
  prediction, then separate reject-confidence calibration. See
  [`docs/design/PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md`](docs/design/PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md).

This repo is a **clean rebuild** of the v2 work. It carries only the current, verified design; the
prior tree lives beside it as `legacy_code/` (not part of this repo) and is mined for reference only.

## Design pillars

- **Deployment-scoped data.** The primary corpus uses phone, watch, and bounded consumer-wearable
  placements: pockets/waist, wrist/forearm, lower back, smart glasses/head, and earbud/ear. Each
  stream contains accelerometer and trustworthy co-located gyroscope when available; accel-only
  streams are explicitly masked. ECG, magnetometer, orientation, and unrelated body rigs are pruned.
  See `data/scripts/curate/deployment_policy.py`.
- **Two dataset versions.** From the curated stream we build a **harmonised** view (fixed 6-channel
  `[acc_xyz, gyro_xyz]` canonical order, zero-pad + validity mask) and a **non-harmonised** view
  (native 3/6-channel width). See `data/scripts/assembly/baseline_view.py`.
- **Unit canonicalization.** Accelerometer values use `g` via a single source of truth
  (`data/scripts/curate/accel_units.py`); iOS `userAcceleration` is rebuilt as `userAcc + gravity`
  when a gravity vector exists. Gravity-removed streams (KU-HAR and XRF AirPods) are explicitly
  described and masked from gravity-dependent behavior rather than being silently treated as total
  acceleration.
- **Tiered, faithful baseline comparison.** Heterogeneity is compared as a stack — **T0** base model,
  **T1** rate, **T2** channels/placement, **T3** open-set labels — with an explicit faithfulness
  contract for what may/may not be done to a baseline. See
  [`docs/baselines/BASELINE_FAIRNESS_POLICY.md`](docs/baselines/BASELINE_FAIRNESS_POLICY.md).

## Layout

Organized by concern (top-level folders, not a single Python package):

```
baselines/            # one subfolder per baseline: citation + paper, cloned repo (gitignored), adapter.py
data/
  datasets/           # one subfolder per dataset: downloads (gitignored), converter, metadata, channel descriptions
  scripts/            # shared cross-dataset logic (imported as data.scripts.*), grouped by stage
    curate/             # deployment_policy.py (device/channel/placement), accel_units.py (unit → g)
    assembly/           # baseline_view.py (harmonised vs non-harmonised), assemble.py
    labels/             # canonical vocabulary + global label mapping
model/
  tokenizer/          # filterbank + encoder + text conditioning (Phase A)
  evidence/           # retrieval, evidence head, decoder, confidence (Phase B)
training/
  tokenizer/          # Phase-A pretraining (JEPA + augmentation VICReg) + probes
  evidence/           # Phase-B memory bank, episodic trainer/evaluator
  diagnostics/        # cross-cutting analyses
eval/                 # zero-shot / few-shot scoring, protocol stamping, table assembly
experiments/          # isolated representation studies with their own configs and outputs
docs/                 # design / data / baselines — see docs/README.md for the reading order
tests/                # regression tests (green)
```

Each folder has a short README describing exactly what belongs in it.

## Status

All five layers below are **built and running**; the open questions are empirical, not structural.

1. **Data** — 12 training datasets (20 streams, native-rate grids) + 7 held-out zero-shot test sets;
   93-label canonical vocabulary. `docs/data/DATA_PIPELINE.md`.
2. **Model** — physical-Hz filterbank tokenizer + config-conditional dual-branch encoder.
3. **Phase A** — label-free JEPA + augmentation-VICReg pretraining.
4. **Phase B** — memory bank + per-patch retrieval + candidate-CE predictor + confidence calibration.
5. **Baselines + eval** — 8 models scored on 7 test sets under a stamped protocol;
   `python -m eval.assemble_table`.

⚠️ **Before citing any number, read `docs/design/EVIDENCE_ENGINE_FINDINGS.md`.** Several headline
claims have been retracted, and that doc — not this one — is the authoritative empirical position.

## Development

Tests run against the interpreter in `legacy_code/.venv` (this tree has no `.venv` of its own):

```bash
/path/to/legacy_code/.venv/bin/python -m pytest tests -q
```

Data, checkpoints, and vendored baseline repos are gitignored and regenerated from source.
