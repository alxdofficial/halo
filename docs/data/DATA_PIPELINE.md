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
python -m data.scripts.download_datasets                  # (1) download raw → data/datasets/<ds>/downloads/
python -m data.datasets.<ds>.convert                      #     convert one dataset → sessions/ + labels.json + manifest.json
python -m data.scripts.build_grids                        # (2–5) harmonised (phone+watch) + non-harmonised, all datasets
python -m data.scripts.build_grids --dataset hhar wisdm   #       ...or just some datasets
python -m data.scripts.labels.build_global_label_mapping  # canonical ConSE vocabulary → data/labels/global_labels.json
```

`placement_strict` (phones only, "harmonised-strict") is not a separate build — it is the phone
subset of the harmonised grids, selected at training time via `deployment_streams(placement_strict=True)`.

## Status — run end-to-end and verified (2026-07-12)

**11 datasets converted + verified end-to-end** on real downloads (harmonised 60 Hz 6-ch `[acc,gyro]` /
non-harmonised native): motionsense, hapt, uci_har, pamap2, wisdm, mhealth, realworld, hhar, kuhar,
unimib_shar (+ shoaib/inclusivehar/capture24 in progress). Verification invariants that passed:
accelerometer median magnitude ≈ 1 g where gravity is present (uci_har 1.02, pamap2 1.00, hhar 0.99,
realworld 1.00, …) and ≈ 0 where gravity is removed (kuhar 0.074); correct channel count + mask
(acc-only sets show `mask=[T,T,T,F,F,F]`); 60 Hz harmonised; canonical labels; subjects present.

**Provenance + downloader:** `data/scripts/download_datasets.py` encodes the verified sources (direct
UCI/uni-mannheim URLs + Kaggle slugs). Gated: **mobiact** (Kaggle returns 403 until you accept the
dataset terms on kaggle.com); shoaib/capture24/inclusivehar download via the scriptable URLs in the
downloader notes. `harth` is downloaded but `role='stress'` (thigh/lower-back), so it is not in the
primary `build_grids` output.

**Pre-windowed datasets** (`metadata.json: pre_windowed: true`): uci_har (128-sample / 2.56 s segments)
and unimib_shar (151-sample). These ship as fixed short segments too short for the 6 s corpus window,
so `build_grids` treats each distributed segment as exactly one window. (kuhar uses its *continuous*
`1.Raw_time_domian_data`, so it is windowed normally.) unimib's upstream Kaggle CSV lost the
subject-map, so its splits collapse to a single pseudo-subject (documented, acceptable for a train set).

## Converter contract (the recipe every converter follows)

Run as `python -m data.datasets.<ds>.convert` from the repo root. Each converter: (a) reads raw from
`data/datasets/<ds>/downloads/`, writes to `data/datasets/<ds>/` (`DS_DIR = Path(__file__).resolve().parent`);
(b) emits **raw whole-recording sessions** `sessions/<id>/data.parquet` (build_grids does the 6 s
windowing — converters must NOT pre-window); (c) includes a `subject`
column for subject-disjoint splits and writes per-session activity to `labels.json`; (d) emits the exact
source column names `deployment_policy` selects for that dataset's stream. All 14 non-gated converters
now satisfy this and are verified against real output.
