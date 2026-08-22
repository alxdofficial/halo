# Results

> ## ⚠️ SUPERSEDED — 2026-08-22
> Current tables: [`ADAPTATION_TABLE_20260822.md`](ADAPTATION_TABLE_20260822.md) (headline) and
> [`RESULTS.md`](RESULTS.md) (index, headline inlined). The HALO row here predates the compact
> evidence engine.

## 1. Zero-shot

No labelled examples. Macro F1, mean over datasets.

| model | ordinary | specialized novel |
|---|---:|---:|
| UniMTS | 32.70 | **19.24** |
| CrossHAR | **37.01** | 10.88 |
| HARNet | 33.82 | 11.40 |
| HALO (ours) | 27.43 | 12.08 |
| LIMU-BERT | 27.60 | 9.11 |
| ImageBind | 11.38 | 8.15 |
| NormWear | 5.08 | 3.58 |

Datasets per regime: ordinary 4, specialized_novel 3

## 2. Label efficiency

Frozen encoder, `linear_head` fitted on k novel labelled examples per class. Macro F1, mean over datasets.

### ordinary

| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO (ours) | **56.54** | **61.47** | **67.22** | **67.47** | 64.51 |
| UniMTS | 54.81 | 59.92 | 65.49 | 66.69 | **65.05** |
| LIMU-BERT | 54.26 | 61.00 | 64.80 | 64.65 | 60.29 |
| CrossHAR | 51.94 | 57.46 | 63.02 | 63.53 | 61.37 |
| HARNet | 51.35 | 56.63 | 61.56 | 62.77 | 59.73 |
| ImageBind | 45.17 | 53.12 | 58.48 | 58.98 | 55.94 |
| NormWear | 35.81 | 42.43 | 46.81 | 46.76 | 44.99 |

### specialized novel

| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| UniMTS | **38.95** | 39.38 | **55.23** | 61.04 | 65.17 |
| HARNet | 34.72 | 37.54 | 54.40 | **61.34** | **66.37** |
| HALO (ours) | 35.89 | **39.79** | 50.00 | 55.08 | 58.29 |
| CrossHAR | 32.20 | 36.35 | 46.69 | 49.07 | 51.12 |
| ImageBind | 29.19 | 32.91 | 40.36 | 44.98 | 48.53 |
| LIMU-BERT | 28.95 | 31.12 | 38.29 | 39.40 | 39.96 |
| NormWear | 25.31 | 25.94 | 33.89 | 36.36 | 38.19 |

## 3. Adaptation mechanism

HALO only. The native retrieval/enrollment mechanism against heads fitted on the same frozen features, same cells. Macro F1.

| mechanism | k=1 | k=2 | k=4 | k=8 |
|---|---:|---:|---:|---:|
| native retrieval | 41.34 | 53.09 | 50.74 | 61.54 |
| prototype | **68.27** | **71.56** | 70.02 | 74.03 |
| ridge | 67.91 | 71.20 | **71.07** | **75.12** |
| identity | 52.87 | 62.67 | 60.70 | 72.72 |

Sanity controls (not competitors):

| control | k=1 | k=2 | k=4 | k=8 |
|---|---:|---:|---:|---:|
| support removed | **26.57** | **26.57** | **16.11** | **16.11** |
| support labels shuffled | 12.79 | 8.84 | 6.24 | 3.58 |

16 matched cells over motionsense, realworld, shoaib. Italic rows are sanity controls, not competitors.

### 3b. Mechanism ablations

Phase-B held-out-concept validation (a different protocol from 3a: held-out concepts and subjects from the training corpus, deployment top-k rule). Macro F1 by support count.

| arm | k=0 | k=1 | k=2 | k=4 | mean |
|---|---:|---:|---:|---:|---:|
| evidence mixer (adopted) | 0.5116 | 0.5034 | 0.4972 | 0.5007 | 0.5032 |
| + semantic text refinement | 0.5146 | 0.5157 | 0.5042 | 0.5047 | 0.5098 |
| mixer, post-trunk rows | 0.5351 | 0.5217 | 0.5112 | 0.5309 | 0.5247 |
| mixer, candidate-blind form only | 0.3579 | 0.4792 | 0.4711 | 0.4898 | 0.4495 |
| mixer, scrambled label vocabulary | 0.3445 | 0.3615 | 0.3520 | 0.3983 | 0.3641 |
| NO mixer, encoder fine-tuned | 0.3907 | 0.4232 | 0.4255 | 0.4557 | 0.4238 |
| NO mixer, nothing trained | 0.3876 | 0.4075 | 0.4447 | 0.4435 | 0.4208 |
