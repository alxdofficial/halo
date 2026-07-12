# Data pipeline — source → grids

How raw datasets become the harmonised / non-harmonised corpus. Every stage is one module in
`data/scripts/` (see also `DATA_HETEROGENEITY.md` for the per-dataset rationale).

## Stages

| # | Stage | Module | Output |
|---|---|---|---|
| 1 | **Convert** | `data/datasets/<ds>/convert.py` | `data/datasets/<ds>/sessions/*.parquet` + `manifest.json` + `labels.json` *(gitignored)* |
| 2 | **Curate** | `curate/deployment_policy.py` | one phone/watch device stream, acc(+gyro), gravity reconstructed |
| 3 | **Unit** | `curate/accel_units.py` | accelerometer → g |
| 4 | **Assemble** | `assembly/assemble.py` | (resample →) window → harmonised / non-harmonised views (`Grid`) |
| 5 | **Orchestrate** | `build_grids.py` | per-stream grids under `data/datasets/<ds>/grids/<alignment>/<stream>/` |

Harmonised = 60 Hz, fixed 6-ch `[acc,gyro]` + mask, **canonical** labels. Non-harmonised = native rate,
native 3/6-ch, native labels. `placement_strict` → phones only (drops the watch datasets).

## Run

```bash
python -m data.scripts.setup_all_datasets                 # (1) download + convert → sessions  [needs source access]
python -m data.scripts.build_grids                        # (2–5) harmonised (phone+watch) + non-harmonised
python -m data.scripts.build_grids --placement-strict     #       harmonised-strict (phones only)
python -m data.scripts.labels.build_global_label_mapping  # canonical ConSE vocabulary → data/labels/global_labels.json
```

## Status

- **Stages 2–5 are complete and unit-tested** (on synthetic sessions): curation, unit→g, gravity
  reconstruction, anti-aliased 60 Hz resample, harmonised/non-harmonised views, `placement_strict`,
  canonical labels, and the `build_grids` accumulation core. `56` tests green.
- **Running end-to-end needs the source downloads (stage 1)** — no sessions are materialized in the
  repo, so the disk I/O in `build_grids.iter_sessions` / the converters has not been executed here.

## Converter alignment — gated on the downloads

`deployment_policy` names the exact source columns each stream needs. Before stage 1 output will
curate cleanly, a few converters must emit those names (verify against real converter output):

| Dataset | deployment_policy needs | converter action |
|---|---|---|
| **pamap2** | `hand_acc16_*`, `hand_gyro_*` | export the wrist IMU's ±16 g accel **and gyro** (raw has them; current export kept only `hand_acc6`/mag/ori) |
| **uci_har** | `total_acc_*`, `body_gyro_*` | export `total_acc` (gravity present) + `body_gyro` under those names |
| **wisdm** | `phone_accel_*`, `phone_gyro_*`, `watch_accel_*`, `watch_gyro_*` | name the phone stream `phone_accel`/`phone_gyro` (currently bare `acc`/`gyro`); also merge the disjoint accel/gyro rows or keep gyro optional |
| **unimib_shar** | `acc_x/y/z` | accelerometer-only; the source subject-map was lost — a re-download is needed to restore subject ids for splits |

The remaining 11 datasets are expected to match (their `deployment_policy` sources use names the
converters already emit), but confirm once sessions exist.
