# Data pipeline — source → grids

How raw datasets become the gridded corpus. Every stage is one module in `data/scripts/` (see also
`DATA_HETEROGENEITY.md` for the per-dataset rationale).

## Stages

| # | Stage | Module | Output |
|---|---|---|---|
| 1 | **Convert** | `data/datasets/<ds>/convert.py` | `data/datasets/<ds>/sessions/*.parquet` + `manifest.json` + `labels.json` *(gitignored)* |
| 2 | **Curate** | `curate/deployment_policy.py` | one phone/watch device stream, acc(+gyro), gravity reconstructed |
| 3 | **Unit** | `curate/accel_units.py` | accelerometer → g |
| 4 | **Assemble** | `assembly/assemble.py` | (resample →) window → one view per alignment (`Grid`) |
| 5 | **Orchestrate** | `build_grids.py` | per-stream grids under `data/datasets/<ds>/grids/<alignment>/<stream>/` |
| 6 | **Screen** | `scan_implausible.py`, `scan_duplicates.py` | per-alignment exclusion caches in `data/quality/` |

**Three alignments** are built (`build_grids._ALIGNMENTS`), not two:

| alignment | rate | channels | labels | who consumes it |
|---|---|---|---|---|
| `native` | native (20/50/100 Hz) | 6-ch `[acc,gyro]` + mask | canonical | **Phase A and Phase B** — the rate-invariant filterbank trains here |

Native grids retain all recording samples. They use non-overlapping contexts of at most six seconds;
the final shorter context is right-padded on disk and accompanied by `lengths.npy`. HALO loaders slice
to that valid length before augmentation and form fixed one-second patches, including one honest final
short patch. The baseline grid regimes remain full-window only.
| `harmonised` | 60 Hz | 6-ch `[acc,gyro]` + mask | canonical | layout-locked baselines that need a fixed rate |
| `non_harmonised` | native | native 3/6-ch | native | the default `run_baselines` scoring alignment |

`placement_strict` → phones only (drops the watch datasets).

Stage 6 is **not optional**. `CorpusIndex`, `build_memory`, `resolvability`, `sensor_bias`, `eval/data.py`
and `run_baselines` all load these caches with `require=True`: a missing or stale cache is a hard
failure, because silently readmitting byte-identical stale-buffer windows would corrupt both training
and scoring. Rebuild them after every grid rebuild, once per alignment you intend to use.

## Run

```bash
python -m data.scripts.download_datasets                  # (1) download raw → data/datasets/<ds>/downloads/
python -m data.datasets.<ds>.convert                      #     convert one dataset → sessions/ + labels.json + manifest.json
python -m data.scripts.build_grids                        # (2–5) all three alignments; default roster =
                                                          #       expanded Phase-A train ∪ primary eval
python -m data.scripts.build_grids --dataset hhar wisdm   #       ...or just some datasets
python -m data.scripts.build_grids --alignment native     #       ...or just one alignment

python -m data.scripts.scan_implausible                   # (6) REQUIRED — native exclusion cache
python -m data.scripts.scan_duplicates                    # (6) REQUIRED — native exclusion cache
python -m data.scripts.scan_implausible --alignment non_harmonised   # needed by run_baselines
python -m data.scripts.scan_duplicates  --alignment non_harmonised   # needed by run_baselines

python -m data.scripts.labels.build_global_label_mapping  # canonical ConSE vocabulary → data/labels/global_labels.json
python -m data.scripts.curate.sensor_bias --build         # frozen per-sensor physics; Phase A fails closed without it
```

`build_grids` with no `--dataset` no longer means "every dataset on disk": it resolves
`EXPANDED_PHASE_A_TRAIN_DATASETS ∪ PRIMARY_EVAL_DATASETS` via `build_stream_specs()`, so a clean
rebuild reproduces the trainer's roster instead of depending on grids left behind by an earlier
one-off build. Optional scale sources (`extrasensory`, `nhanes`, `hmog`, `kneepad`) must be named
explicitly.

`placement_strict` (phones only, "harmonised-strict") is not a separate build — it is the phone
subset of the harmonised grids, selected at training time via `deployment_streams(placement_strict=True)`.

## Historical status snapshot (2026-07-12)

This section records the earlier converter milestone and is not the current Phase-A roster. The
authoritative live roster, stream semantics, and measured corpus properties are in
`DATA_HETEROGENEITY.md` and `training/tokenizer/README.md`.

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
source column names `deployment_policy` selects for that dataset's stream. All 35 converters under
`data/datasets/*/convert.py` satisfy this and are verified against real output.

One deliberate exception to (b): the 2026-08 multi-placement sources (realdisp, forth_trace, dsads,
opportunity, phytmo, mmfit, kneepad, monipar, spar, upper_limb_use) write **every placement of one
recording into a single frame** with a per-placement column prefix. That is what makes window *i* of
each placement the same physical instant, which the paired-contrast measurement in
`training/evidence/resolvability.py` depends on. Copy this pattern for any future multi-sensor source.
