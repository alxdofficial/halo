# Results

## 1. Zero-shot

No labelled examples. Macro F1, mean over datasets.

| model | ordinary | specialized novel |
|---|---:|---:|
| UniMTS | 31.98 | **17.37** |
| CrossHAR | **37.70** | 11.22 |
| HARNet | 33.82 | 11.40 |
| LIMU-BERT | 30.60 | 10.27 |
| HALO (ours) | 26.99 | 11.32 |
| ImageBind | 11.38 | 8.15 |
| NormWear | 5.08 | 3.58 |

Datasets per regime: ordinary 4, specialized_novel 3

## 2. Label efficiency

`k` is the number of independent enrolled executions per candidate. HALO is shown with its retrieve-mix-vote mechanism in addition to the same three non-gradient readouts used for every representation: one-nearest-neighbor, support prototypes, and closed-form ridge regression. All readouts see only the enrolled support executions. Macro F1, mean over datasets.

### ordinary

| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO / retrieve-mix-vote | 48.86 | 51.50 | 55.18 | 53.59 | 51.58 |
| HALO / 1-NN | 53.72 | 57.91 | 60.85 | 60.62 | 58.74 |
| HALO / prototype | 53.72 | 57.82 | 58.82 | 59.01 | 55.22 |
| HALO / ridge | 52.53 | 56.47 | 57.93 | 58.19 | 55.78 |
| LIMU-BERT / 1-NN | 56.91 | **61.95** | **65.24** | **64.89** | 61.55 |
| LIMU-BERT / prototype | **56.91** | 60.19 | 63.84 | 63.06 | 59.18 |
| LIMU-BERT / ridge | 50.23 | 53.46 | 58.41 | 57.85 | 55.06 |
| UniMTS / 1-NN | 50.69 | 56.20 | 61.01 | 62.22 | **62.68** |
| UniMTS / prototype | 50.69 | 52.27 | 55.79 | 55.99 | 54.62 |
| UniMTS / ridge | 47.66 | 49.84 | 53.62 | 54.67 | 54.16 |
| CrossHAR / 1-NN | 50.54 | 54.43 | 59.58 | 58.70 | 57.39 |
| CrossHAR / prototype | 50.54 | 54.04 | 58.17 | 56.69 | 55.61 |
| CrossHAR / ridge | 42.99 | 45.84 | 50.32 | 50.62 | 50.38 |
| HARNet / 1-NN | 47.34 | 50.51 | 53.20 | 52.66 | 50.69 |
| HARNet / prototype | 47.34 | 49.52 | 52.93 | 52.39 | 50.40 |
| HARNet / ridge | 46.73 | 49.16 | 53.55 | 53.97 | 52.73 |
| ImageBind / 1-NN | 43.02 | 49.06 | 53.22 | 53.19 | 51.02 |
| ImageBind / prototype | 43.02 | 47.12 | 48.70 | 46.39 | 44.54 |
| ImageBind / ridge | 42.92 | 48.16 | 51.75 | 51.71 | 49.71 |
| NormWear / 1-NN | 26.26 | 29.56 | 32.92 | 35.35 | 37.11 |
| NormWear / prototype | 26.26 | 27.17 | 28.82 | 28.80 | 27.83 |
| NormWear / ridge | 20.68 | 19.99 | 20.76 | 21.59 | 23.89 |

### specialized novel

| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO / retrieve-mix-vote | 31.18 | 31.38 | 36.18 | 37.48 | 41.83 |
| HALO / 1-NN | 36.01 | 36.18 | 48.16 | 52.07 | 54.85 |
| HALO / prototype | 36.01 | 35.86 | 45.16 | 46.98 | 48.49 |
| HALO / ridge | 35.65 | 35.21 | 46.70 | 51.37 | **56.41** |
| LIMU-BERT / 1-NN | 30.58 | 33.83 | 40.28 | 42.78 | 43.97 |
| LIMU-BERT / prototype | 30.58 | 33.36 | 38.76 | 40.38 | 41.66 |
| LIMU-BERT / ridge | 27.56 | 30.75 | 36.07 | 38.25 | 40.39 |
| UniMTS / 1-NN | **37.02** | **36.71** | **49.12** | **52.77** | 55.05 |
| UniMTS / prototype | **37.02** | 36.49 | 47.87 | 50.19 | 50.91 |
| UniMTS / ridge | 34.39 | 34.78 | 46.71 | 50.28 | 52.85 |
| CrossHAR / 1-NN | 28.32 | 32.36 | 40.73 | 43.04 | 45.44 |
| CrossHAR / prototype | 28.32 | 32.46 | 40.97 | 43.38 | 44.74 |
| CrossHAR / ridge | 26.77 | 31.03 | 40.46 | 43.31 | 45.79 |
| HARNet / 1-NN | 30.83 | 31.84 | 43.78 | 47.62 | 50.98 |
| HARNet / prototype | 30.83 | 31.55 | 41.69 | 44.19 | 45.91 |
| HARNet / ridge | 30.12 | 31.66 | 43.82 | 48.89 | 54.03 |
| ImageBind / 1-NN | 27.29 | 29.59 | 35.43 | 38.29 | 40.56 |
| ImageBind / prototype | 27.29 | 29.84 | 34.15 | 36.04 | 37.15 |
| ImageBind / ridge | 26.24 | 29.85 | 35.15 | 39.35 | 43.00 |
| NormWear / 1-NN | 18.63 | 20.20 | 25.61 | 28.21 | 31.06 |
| NormWear / prototype | 18.63 | 18.14 | 21.50 | 21.76 | 21.05 |
| NormWear / ridge | 14.68 | 14.24 | 18.43 | 19.35 | 19.67 |

## 3. Per-dataset performance

These tables use the same protocol as the aggregate results. Values are macro F1 averaged over seeds within each held-out dataset.

### Native zero-shot by dataset

| model | Inclusive-HAR | MoniPar | SPAR | TNDA-HAR | USC-HAD | UT Complex | Upper Limb Use |
|---|---:|---:|---:|---:|---:|---:|---:|
| HALO (ours) | 23.21 | 15.19 | 15.35 | 46.21 | 14.64 | 23.88 | 3.43 |
| LIMU-BERT | **36.10** | 11.05 | 9.57 | 44.42 | 22.89 | 19.00 | 10.20 |
| UniMTS | 30.53 | **22.22** | **22.21** | 45.12 | 24.17 | 28.09 | 7.68 |
| CrossHAR | 30.56 | 11.97 | 9.79 | **52.27** | **35.14** | **32.85** | **11.90** |
| HARNet | 23.15 | 13.10 | 17.39 | 51.07 | 32.35 | 28.72 | 3.70 |
| ImageBind | 18.84 | 5.69 | 11.78 | 13.02 | 6.02 | 7.64 | 6.97 |
| NormWear | 8.60 | 0.76 | 6.10 | 5.11 | 4.82 | 1.80 | 3.87 |

### ordinary enrollment

#### Inclusive-HAR

| model / readout | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO / retrieve-mix-vote | 30.19 | 32.53 | 33.66 | 32.85 | 35.42 |
| HALO / 1-NN | 34.90 | 37.09 | 38.24 | 41.96 | 43.85 |
| HALO / prototype | 34.90 | 37.22 | 38.38 | 42.82 | 43.94 |
| HALO / ridge | 34.65 | 38.14 | 38.89 | 42.71 | 44.83 |
| LIMU-BERT / 1-NN | 31.56 | 36.14 | 37.75 | 38.78 | 39.18 |
| LIMU-BERT / prototype | 31.56 | 33.03 | 35.90 | 38.74 | 41.13 |
| LIMU-BERT / ridge | 31.15 | 32.64 | 35.69 | 37.79 | 38.16 |
| UniMTS / 1-NN | **35.40** | **38.63** | **41.93** | **44.24** | **49.04** |
| UniMTS / prototype | **35.40** | 35.55 | 38.63 | 41.59 | 43.70 |
| UniMTS / ridge | 34.80 | 35.42 | 39.52 | 42.52 | 44.96 |
| CrossHAR / 1-NN | 32.83 | 33.85 | 36.81 | 37.61 | 39.97 |
| CrossHAR / prototype | 32.83 | 34.09 | 37.36 | 38.89 | 40.01 |
| CrossHAR / ridge | 31.47 | 32.43 | 35.35 | 36.82 | 39.29 |
| HARNet / 1-NN | 33.04 | 34.37 | 34.26 | 34.48 | 35.85 |
| HARNet / prototype | 33.04 | 34.14 | 35.46 | 36.68 | 37.98 |
| HARNet / ridge | 32.65 | 34.39 | 36.02 | 37.49 | 40.26 |
| ImageBind / 1-NN | 23.15 | 27.60 | 29.12 | 32.69 | 35.58 |
| ImageBind / prototype | 23.15 | 25.01 | 27.20 | 28.25 | 34.14 |
| ImageBind / ridge | 23.06 | 26.20 | 29.07 | 29.64 | 32.95 |
| NormWear / 1-NN | 25.94 | 26.52 | 27.57 | 29.30 | 30.66 |
| NormWear / prototype | 25.94 | 25.76 | 26.42 | 26.95 | 26.06 |
| NormWear / ridge | 25.12 | 24.66 | 24.85 | 24.50 | 23.02 |

#### TNDA-HAR

| model / readout | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO / retrieve-mix-vote | 59.21 | 64.28 | 69.00 | 72.15 | 74.86 |
| HALO / 1-NN | 59.07 | 63.91 | 68.69 | 72.12 | 76.34 |
| HALO / prototype | 59.07 | 64.53 | 63.77 | 67.73 | 68.56 |
| HALO / ridge | 59.05 | 63.48 | 62.40 | 68.04 | 70.63 |
| LIMU-BERT / 1-NN | **60.59** | **66.34** | **70.99** | 75.64 | 79.96 |
| LIMU-BERT / prototype | **60.59** | 63.49 | 70.24 | 73.26 | 74.64 |
| LIMU-BERT / ridge | 54.04 | 59.68 | 69.54 | 73.73 | 75.17 |
| UniMTS / 1-NN | 56.84 | 63.99 | 70.94 | **76.82** | **80.00** |
| UniMTS / prototype | 56.86 | 57.12 | 63.18 | 65.70 | 65.34 |
| UniMTS / ridge | 52.02 | 53.20 | 58.05 | 61.63 | 63.15 |
| CrossHAR / 1-NN | 56.60 | 61.27 | 70.85 | 74.96 | 78.81 |
| CrossHAR / prototype | 56.60 | 60.87 | 67.94 | 72.45 | 77.61 |
| CrossHAR / ridge | 49.19 | 50.48 | 56.54 | 62.14 | 67.84 |
| HARNet / 1-NN | 53.50 | 56.34 | 59.47 | 64.21 | 68.08 |
| HARNet / prototype | 53.50 | 54.17 | 60.57 | 64.86 | 66.63 |
| HARNet / ridge | 54.43 | 54.22 | 61.15 | 65.99 | 69.23 |
| ImageBind / 1-NN | 47.61 | 55.88 | 60.20 | 66.37 | 72.21 |
| ImageBind / prototype | 47.61 | 55.03 | 54.07 | 54.35 | 55.78 |
| ImageBind / ridge | 45.88 | 54.77 | 57.69 | 63.60 | 68.87 |
| NormWear / 1-NN | 29.35 | 35.20 | 41.37 | 47.70 | 53.53 |
| NormWear / prototype | 29.35 | 30.09 | 34.81 | 37.45 | 37.40 |
| NormWear / ridge | 24.35 | 21.88 | 24.45 | 29.88 | 30.75 |

#### USC-HAD

| model / readout | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO / retrieve-mix-vote | 48.58 | 49.43 | 52.61 | 40.76 | 44.46 |
| HALO / 1-NN | 56.26 | 60.81 | 63.75 | 53.18 | 56.03 |
| HALO / prototype | 56.26 | 59.50 | 61.07 | 52.06 | 53.15 |
| HALO / ridge | 54.31 | 56.88 | 59.70 | 48.78 | 51.89 |
| LIMU-BERT / 1-NN | 66.63 | 70.47 | **73.36** | **62.85** | **65.51** |
| LIMU-BERT / prototype | **66.63** | **70.54** | 72.82 | 62.02 | 61.78 |
| LIMU-BERT / ridge | 56.56 | 58.59 | 62.07 | 50.11 | 51.85 |
| UniMTS / 1-NN | 53.62 | 59.62 | 63.45 | 56.99 | 58.99 |
| UniMTS / prototype | 53.62 | 56.78 | 59.33 | 52.46 | 54.82 |
| UniMTS / ridge | 50.97 | 54.36 | 57.05 | 51.86 | 54.38 |
| CrossHAR / 1-NN | 54.81 | 59.43 | 63.25 | 50.86 | 53.38 |
| CrossHAR / prototype | 54.81 | 58.95 | 62.01 | 49.04 | 49.20 |
| CrossHAR / ridge | 48.16 | 52.22 | 55.54 | 42.51 | 44.02 |
| HARNet / 1-NN | 48.72 | 52.26 | 55.78 | 45.51 | 48.14 |
| HARNet / prototype | 48.72 | 52.03 | 54.79 | 44.56 | 46.60 |
| HARNet / ridge | 47.71 | 51.08 | 54.46 | 45.71 | 48.71 |
| ImageBind / 1-NN | 50.67 | 56.16 | 59.79 | 43.71 | 45.26 |
| ImageBind / prototype | 50.67 | 53.87 | 56.29 | 42.83 | 43.71 |
| ImageBind / ridge | 50.81 | 54.07 | 57.58 | 45.12 | 47.30 |
| NormWear / 1-NN | 23.14 | 25.60 | 27.70 | 24.76 | 27.14 |
| NormWear / prototype | 23.14 | 24.48 | 24.81 | 20.76 | 20.03 |
| NormWear / ridge | 18.30 | 19.61 | 20.63 | 17.71 | 17.91 |

#### UT Complex

| model / readout | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO / retrieve-mix-vote | 57.45 | 59.74 | 65.47 | 68.60 | n/a |
| HALO / 1-NN | 64.65 | 69.83 | 72.70 | 75.21 | n/a |
| HALO / prototype | 64.65 | 70.01 | 72.05 | 73.42 | n/a |
| HALO / ridge | 62.12 | 67.37 | 70.76 | 73.24 | n/a |
| LIMU-BERT / 1-NN | **68.87** | **74.85** | **78.87** | **82.28** | n/a |
| LIMU-BERT / prototype | **68.87** | 73.71 | 76.39 | 78.21 | n/a |
| LIMU-BERT / ridge | 59.16 | 62.94 | 66.32 | 69.78 | n/a |
| UniMTS / 1-NN | 56.89 | 62.55 | 67.70 | 70.82 | n/a |
| UniMTS / prototype | 56.89 | 59.63 | 62.03 | 64.20 | n/a |
| UniMTS / ridge | 52.83 | 56.37 | 59.87 | 62.66 | n/a |
| CrossHAR / 1-NN | 57.93 | 63.19 | 67.41 | 71.39 | n/a |
| CrossHAR / prototype | 57.93 | 62.23 | 65.36 | 66.38 | n/a |
| CrossHAR / ridge | 43.15 | 48.21 | 53.86 | 61.03 | n/a |
| HARNet / 1-NN | 54.09 | 59.06 | 63.30 | 66.45 | n/a |
| HARNet / prototype | 54.09 | 57.75 | 60.89 | 63.48 | n/a |
| HARNet / ridge | 52.15 | 56.94 | 62.57 | 66.69 | n/a |
| ImageBind / 1-NN | 50.64 | 56.60 | 63.77 | 69.99 | n/a |
| ImageBind / prototype | 50.64 | 54.56 | 57.23 | 60.14 | n/a |
| ImageBind / ridge | 51.95 | 57.60 | 62.64 | 68.49 | n/a |
| NormWear / 1-NN | 26.59 | 30.92 | 35.05 | 39.64 | n/a |
| NormWear / prototype | 26.59 | 28.35 | 29.24 | 30.06 | n/a |
| NormWear / ridge | 14.96 | 13.79 | 13.11 | 14.27 | n/a |

### specialized novel enrollment

#### MoniPar

| model / readout | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO / retrieve-mix-vote | 34.59 | 38.95 | 32.82 | 34.12 | 36.99 |
| HALO / 1-NN | 38.40 | 41.80 | 38.09 | 41.08 | 43.47 |
| HALO / prototype | 38.40 | 41.26 | 37.11 | 38.43 | 39.34 |
| HALO / ridge | 36.43 | 38.14 | 36.36 | 39.57 | 43.61 |
| LIMU-BERT / 1-NN | 36.63 | 41.18 | 33.75 | 37.73 | 39.34 |
| LIMU-BERT / prototype | 36.63 | 38.85 | 30.53 | 33.26 | 35.45 |
| LIMU-BERT / ridge | 30.35 | 31.80 | 26.12 | 29.44 | 32.31 |
| UniMTS / 1-NN | **43.31** | 45.65 | 41.32 | 42.65 | 43.00 |
| UniMTS / prototype | **43.31** | **46.43** | **42.74** | 43.58 | 42.92 |
| UniMTS / ridge | 38.84 | 42.36 | 40.44 | 42.62 | 43.99 |
| CrossHAR / 1-NN | 38.16 | 42.25 | 38.35 | 40.83 | 42.74 |
| CrossHAR / prototype | 38.16 | 42.37 | 40.05 | 42.08 | 43.01 |
| CrossHAR / ridge | 36.94 | 42.48 | 40.58 | 42.83 | 44.85 |
| HARNet / 1-NN | 34.65 | 40.09 | 39.89 | 44.27 | 47.46 |
| HARNet / prototype | 34.65 | 38.73 | 37.23 | 39.18 | 40.65 |
| HARNet / ridge | 34.47 | 40.72 | 41.70 | **46.97** | **51.30** |
| ImageBind / 1-NN | 28.89 | 32.91 | 26.57 | 29.75 | 32.59 |
| ImageBind / prototype | 28.89 | 30.20 | 24.18 | 25.76 | 26.82 |
| ImageBind / ridge | 26.86 | 30.14 | 24.70 | 29.37 | 32.83 |
| NormWear / 1-NN | 23.22 | 26.55 | 25.00 | 27.62 | 30.29 |
| NormWear / prototype | 23.22 | 23.88 | 21.06 | 22.30 | 22.03 |
| NormWear / ridge | 16.70 | 17.74 | 17.80 | 20.36 | 21.90 |

#### SPAR

| model / readout | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO / retrieve-mix-vote | 42.52 | 39.71 | 39.54 | 40.85 | 46.67 |
| HALO / 1-NN | 52.02 | **52.48** | **58.23** | 63.05 | 66.23 |
| HALO / prototype | 52.02 | 49.90 | 53.21 | 55.53 | 57.64 |
| HALO / ridge | 52.73 | 50.97 | 57.04 | **63.18** | **69.20** |
| LIMU-BERT / 1-NN | 35.63 | 45.39 | 46.81 | 47.83 | 48.59 |
| LIMU-BERT / prototype | 35.63 | 45.22 | 46.99 | 47.51 | 47.86 |
| LIMU-BERT / ridge | 35.34 | 44.50 | 46.02 | 47.07 | 48.46 |
| UniMTS / 1-NN | **52.86** | 50.65 | 56.92 | 62.89 | 67.11 |
| UniMTS / prototype | **52.86** | 48.67 | 53.00 | 56.81 | 58.89 |
| UniMTS / ridge | 51.25 | 48.30 | 52.99 | 57.94 | 61.72 |
| CrossHAR / 1-NN | 31.83 | 39.05 | 43.10 | 45.25 | 48.14 |
| CrossHAR / prototype | 31.83 | 38.11 | 41.89 | 44.67 | 46.48 |
| CrossHAR / ridge | 30.60 | 36.67 | 40.35 | 43.80 | 46.74 |
| HARNet / 1-NN | 45.09 | 42.97 | 47.67 | 50.98 | 54.51 |
| HARNet / prototype | 45.09 | 42.42 | 46.15 | 49.20 | 51.17 |
| HARNet / ridge | 43.88 | 41.72 | 45.95 | 50.80 | 56.77 |
| ImageBind / 1-NN | 36.58 | 41.74 | 44.29 | 46.82 | 48.53 |
| ImageBind / prototype | 36.58 | 41.98 | 44.13 | 46.31 | 47.48 |
| ImageBind / ridge | 35.94 | 42.17 | 45.61 | 49.32 | 53.16 |
| NormWear / 1-NN | 22.69 | 23.82 | 26.22 | 28.80 | 31.82 |
| NormWear / prototype | 22.69 | 21.53 | 21.95 | 21.23 | 20.07 |
| NormWear / ridge | 20.30 | 18.86 | 19.06 | 18.35 | 17.45 |

#### Upper Limb Use

| model / readout | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO / retrieve-mix-vote | 16.42 | 15.47 | n/a | n/a | n/a |
| HALO / 1-NN | 17.61 | 14.25 | n/a | n/a | n/a |
| HALO / prototype | 17.61 | 16.43 | n/a | n/a | n/a |
| HALO / ridge | 17.80 | 16.51 | n/a | n/a | n/a |
| LIMU-BERT / 1-NN | **19.49** | 14.93 | n/a | n/a | n/a |
| LIMU-BERT / prototype | **19.49** | 16.02 | n/a | n/a | n/a |
| LIMU-BERT / ridge | 16.98 | 15.95 | n/a | n/a | n/a |
| UniMTS / 1-NN | 14.90 | 13.83 | n/a | n/a | n/a |
| UniMTS / prototype | 14.90 | 14.36 | n/a | n/a | n/a |
| UniMTS / ridge | 13.07 | 13.68 | n/a | n/a | n/a |
| CrossHAR / 1-NN | 14.96 | 15.78 | n/a | n/a | n/a |
| CrossHAR / prototype | 14.96 | 16.89 | n/a | n/a | n/a |
| CrossHAR / ridge | 12.79 | 13.94 | n/a | n/a | n/a |
| HARNet / 1-NN | 12.75 | 12.47 | n/a | n/a | n/a |
| HARNet / prototype | 12.75 | 13.51 | n/a | n/a | n/a |
| HARNet / ridge | 12.02 | 12.54 | n/a | n/a | n/a |
| ImageBind / 1-NN | 16.39 | 14.14 | n/a | n/a | n/a |
| ImageBind / prototype | 16.39 | **17.34** | n/a | n/a | n/a |
| ImageBind / ridge | 15.92 | 17.24 | n/a | n/a | n/a |
| NormWear / 1-NN | 10.00 | 10.24 | n/a | n/a | n/a |
| NormWear / prototype | 10.00 | 9.02 | n/a | n/a | n/a |
| NormWear / ridge | 7.04 | 6.12 | n/a | n/a | n/a |
