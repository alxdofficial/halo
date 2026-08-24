# Results

## 1. Zero-shot

No labelled examples. Macro F1, mean over datasets.

| model | ordinary | specialized novel |
|---|---:|---:|
| UniMTS | 31.98 | **17.37** |
| CrossHAR | **37.70** | 11.22 |
| HARNet | 33.82 | 11.40 |
| LIMU-BERT | 30.60 | 10.27 |
| HALO (ours, compact engine) | 20.72 | 9.72 |
| ImageBind | 11.38 | 8.15 |
| NormWear | 5.08 | 3.58 |

Datasets per regime: ordinary 4, specialized_novel 3

## 2. Label efficiency

`k` is the number of independent enrolled executions per candidate. HALO is shown with both its native evidence engine and one-nearest-neighbor over the same learned representation. Every external model uses the same one-nearest-neighbor rule, which requires no fitting and sees only the enrolled support executions. Macro F1, mean over datasets.

### ordinary

| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| LIMU-BERT / 1-NN | **56.91** | **61.95** | **65.24** | **64.89** | 61.55 |
| UniMTS / 1-NN | 50.69 | 56.20 | 61.01 | 62.22 | **62.68** |
| HALO (ours, native engine) | 53.99 | 57.82 | 61.68 | 60.81 | 58.48 |
| HALO (ours, 1-NN) | 53.32 | 56.83 | 60.33 | 60.01 | 57.97 |
| CrossHAR / 1-NN | 50.54 | 54.43 | 59.58 | 58.70 | 57.39 |
| HARNet / 1-NN | 47.34 | 50.51 | 53.20 | 52.66 | 50.69 |
| ImageBind / 1-NN | 43.02 | 49.06 | 53.22 | 53.19 | 51.02 |
| NormWear / 1-NN | 26.26 | 29.56 | 32.92 | 35.35 | 37.11 |

### specialized novel

| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO (ours, native engine) | **38.28** | **38.71** | **51.74** | **55.78** | **58.22** |
| HALO (ours, 1-NN) | 36.78 | 37.88 | 50.00 | 53.25 | 55.35 |
| UniMTS / 1-NN | 37.02 | 36.71 | 49.12 | 52.77 | 55.05 |
| HARNet / 1-NN | 30.83 | 31.84 | 43.78 | 47.62 | 50.98 |
| LIMU-BERT / 1-NN | 30.58 | 33.83 | 40.28 | 42.78 | 43.97 |
| CrossHAR / 1-NN | 28.32 | 32.36 | 40.73 | 43.04 | 45.44 |
| ImageBind / 1-NN | 27.29 | 29.59 | 35.43 | 38.29 | 40.56 |
| NormWear / 1-NN | 18.63 | 20.20 | 25.61 | 28.21 | 31.06 |

## 3. Additional matched readouts

These comparisons apply the same readout to each model's exposed latent representation. Each enrolled execution contributes one equally weighted, normalized pooled vector. Prototype forms class centroids from enrolled supports only and never accesses query examples. `linear_head` fits only a linear classifier on those supports. Ridge is retained as an additional diagnostic.

### ordinary: prototype

| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| LIMU-BERT | **56.91** | **60.19** | **63.84** | **63.06** | **59.18** |
| HALO (ours, compact engine) | 53.32 | 55.50 | 58.10 | 57.56 | 55.74 |
| CrossHAR | 50.54 | 54.04 | 58.17 | 56.69 | 55.61 |
| UniMTS | 50.69 | 52.27 | 55.79 | 55.99 | 54.62 |
| HARNet | 47.34 | 49.52 | 52.93 | 52.39 | 50.40 |
| ImageBind | 43.02 | 47.12 | 48.70 | 46.39 | 44.54 |
| NormWear | 26.26 | 27.17 | 28.82 | 28.80 | 27.83 |

### ordinary: linear_head

| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| LIMU-BERT | **55.25** | **61.85** | **66.31** | **66.50** | **64.12** |
| HALO (ours, compact engine) | 52.92 | 57.81 | 62.67 | 62.29 | 61.34 |
| UniMTS | 50.99 | 55.99 | 61.60 | 63.13 | 62.40 |
| CrossHAR | 50.70 | 55.46 | 61.41 | 61.26 | 60.25 |
| HARNet | 48.02 | 52.12 | 56.98 | 57.66 | 57.20 |
| ImageBind | 41.93 | 48.45 | 52.87 | 54.07 | 53.67 |
| NormWear | 29.10 | 35.50 | 40.64 | 42.29 | 43.43 |

### ordinary: ridge

| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO (ours, compact engine) | **52.83** | **56.09** | **60.18** | **60.80** | **58.07** |
| LIMU-BERT | 50.23 | 53.46 | 58.41 | 57.85 | 55.06 |
| UniMTS | 47.66 | 49.84 | 53.62 | 54.67 | 54.16 |
| HARNet | 46.73 | 49.16 | 53.55 | 53.97 | 52.73 |
| ImageBind | 42.92 | 48.16 | 51.75 | 51.71 | 49.71 |
| CrossHAR | 42.99 | 45.84 | 50.32 | 50.62 | 50.38 |
| NormWear | 20.68 | 19.99 | 20.76 | 21.59 | 23.89 |

### specialized novel: prototype

| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO (ours, compact engine) | 36.78 | **37.06** | **49.16** | **51.11** | **52.78** |
| UniMTS | **37.02** | 36.49 | 47.87 | 50.19 | 50.91 |
| HARNet | 30.83 | 31.55 | 41.69 | 44.19 | 45.91 |
| CrossHAR | 28.32 | 32.46 | 40.97 | 43.38 | 44.74 |
| LIMU-BERT | 30.58 | 33.36 | 38.76 | 40.38 | 41.66 |
| ImageBind | 27.29 | 29.84 | 34.15 | 36.04 | 37.15 |
| NormWear | 18.63 | 18.14 | 21.50 | 21.76 | 21.05 |

### specialized novel: linear_head

| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO (ours, compact engine) | **37.41** | **38.49** | **52.87** | **57.85** | **61.70** |
| UniMTS | 37.09 | 37.54 | 51.06 | 55.91 | 59.41 |
| HARNet | 32.00 | 34.38 | 47.84 | 54.18 | 59.40 |
| LIMU-BERT | 31.33 | 35.10 | 42.92 | 46.03 | 47.62 |
| CrossHAR | 29.33 | 33.94 | 42.95 | 46.43 | 49.56 |
| ImageBind | 27.54 | 30.44 | 37.42 | 41.37 | 44.47 |
| NormWear | 20.67 | 22.92 | 31.19 | 34.97 | 37.73 |

### specialized novel: ridge

| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO (ours, compact engine) | **35.61** | **36.14** | **49.17** | **52.85** | **57.74** |
| UniMTS | 34.39 | 34.78 | 46.71 | 50.28 | 52.85 |
| HARNet | 30.12 | 31.66 | 43.82 | 48.89 | 54.03 |
| CrossHAR | 26.77 | 31.03 | 40.46 | 43.31 | 45.79 |
| ImageBind | 26.24 | 29.85 | 35.15 | 39.35 | 43.00 |
| LIMU-BERT | 27.56 | 30.75 | 36.07 | 38.25 | 40.39 |
| NormWear | 14.68 | 14.24 | 18.43 | 19.35 | 19.67 |
