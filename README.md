# HALO

**H**eterogeneity-**A**ware **L**anguage-aligned IMU model for **O**pen-set HAR.

HALO is a language-aligned IMU foundation model for **real-world phone/watch human activity
recognition**. A channel-independent, rate-invariant tokenizer produces per-patch embeddings that
are contrastively aligned to natural-language activity labels, so activities are recognized
**zero-shot** by similarity to label text — no per-dataset classifier — across heterogeneous
sampling rates, channel sets, and sensor placements.

This repo is a **clean rebuild** of the v2 work. It carries only the current, verified design; the
prior tree lives beside it as `legacy_code/` (not part of this repo) and is mined for reference only.

## Design pillars

- **Deployment-scoped data.** We support what a phone (pocket / waist / thigh) or a watch
  (wrist / arm) actually records: one physical device per sample, accelerometer + co-located
  gyroscope, gravity-present, in `g`. Placements and modalities we don't ship (ankle/chest/torso,
  ECG, magnetometer, orientation) are pruned. See `halo/data/deployment_policy.py`.
- **Two dataset versions.** From the curated stream we build a **harmonised** view (fixed 6-channel
  `[acc_xyz, gyro_xyz]` canonical order, zero-pad + validity mask) and a **non-harmonised** view
  (native 3/6-channel width). See `halo/baselines/baseline_view.py`.
- **Unit canonicalization.** One convention — accelerometer in `g`, gravity present — via a single
  source of truth (`halo/data/accel_units.py`); iOS `userAcceleration` is rebuilt as
  `userAcc + gravity`, and the only unavoidable gravity-removed set is disclosed, never faked.
- **Tiered, faithful baseline comparison.** Heterogeneity is compared as a stack — **T0** base model,
  **T1** rate, **T2** channels/placement, **T3** open-set labels — with an explicit faithfulness
  contract for what may/may not be done to a baseline. See
  [`docs/BASELINE_FAIRNESS_POLICY.md`](docs/BASELINE_FAIRNESS_POLICY.md).

## Layout

```
halo/
  data/
    deployment_policy.py   # phone/watch device-stream curation (channel/placement selection)
    accel_units.py         # unit + gravity canonicalization (single source of truth)
  baselines/
    baseline_view.py       # harmonised / non-harmonised model-input views
docs/
  BASELINE_FAIRNESS_POLICY.md   # tiered comparison + faithfulness contract (design of record)
tests/                     # regression tests for the above (green)
```

## Status

**Landed (this repo):** the data-curation + baseline-view foundation and the fairness policy, with
passing tests. This is the layer every model and every baseline reads from, so it comes first.

**Roadmap (rebuilding into this repo):**
1. Dataset build: `deployment_policy` over raw sessions → windowed **harmonised** + **non-harmonised**
   grids (60 Hz + native), per dataset. *(A few converters need re-runs first — e.g. PAMAP2 wrist
   gyro/±16 g — tracked in the build.)*
2. Model: physical-Hz filterbank tokenizer + dual-branch encoder + per-patch language-alignment head.
3. Training: symmetric InfoNCE alignment to SBERT label text, channel-text + augmentation curriculum.
4. Evaluation: zero-shot cross-dataset (macro-F1), subject-disjoint few-shot, tiered ablations.
5. Baselines: CrossHAR / LiMU-BERT / DeepConvLSTM (we train) + ssl-wearables / UniMTS / NormWear
   (frozen), each per its faithfulness contract, via the ConSE bridge or native text tower.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Data, checkpoints, and vendored baseline repos are gitignored and regenerated from source.
