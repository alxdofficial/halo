# Results

## 1. Zero-shot

No labelled examples. Macro F1, equally averaged over all held-out datasets.

| model | all datasets |
|---|---:|
| UniMTS | **25.72** |
| HALO (ours) | 22.37 |
| ImageBind | 10.00 |
| NormWear | 4.44 |

Held-out datasets: 7.

## 2. Label efficiency

`k` is the number of independent enrolled executions per candidate. HALO is shown with its retrieve-mix-vote mechanism in addition to the same three non-gradient readouts used for every representation: one-nearest-neighbor, support prototypes, and closed-form ridge regression. All readouts see only the enrolled support executions. Macro F1, mean over datasets.

| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO / retrieve-mix-vote | 44.22 | 45.67 | 52.48 | 53.02 | 53.38 |
| HALO / 1-NN | **45.44** | 47.31 | 55.70 | 56.47 | 56.53 |
| HALO / prototype | **45.44** | 47.02 | 54.09 | 54.04 | 52.42 |
| HALO / ridge | 44.26 | 46.08 | 54.33 | 55.19 | 55.56 |
| UniMTS / 1-NN | 44.83 | **47.85** | **57.04** | **59.07** | **59.63** |
| UniMTS / prototype | 44.83 | 45.51 | 53.15 | 54.05 | 53.14 |
| UniMTS / ridge | 41.97 | 43.38 | 51.32 | 53.21 | 53.64 |
| HARNet / 1-NN | 40.26 | 42.51 | 50.06 | 50.98 | 50.81 |
| HARNet / prototype | 40.26 | 41.82 | 49.18 | 49.66 | 48.61 |
| HARNet / ridge | 39.61 | 41.66 | 50.31 | 52.27 | 53.25 |
| ImageBind / 1-NN | 36.28 | 40.72 | 47.29 | 48.22 | 46.83 |
| ImageBind / prototype | 36.28 | 39.71 | 43.85 | 42.94 | 41.59 |
| ImageBind / ridge | 35.77 | 40.31 | 46.22 | 47.59 | 47.02 |
| NormWear / 1-NN | 22.99 | 25.55 | 30.49 | 32.97 | 34.69 |
| NormWear / prototype | 22.99 | 23.30 | 26.38 | 26.46 | 25.12 |
| NormWear / ridge | 18.11 | 17.53 | 19.98 | 20.84 | 22.21 |

## 3. Per-dataset performance

These tables use the same protocol as the aggregate results. Values are macro F1 averaged over seeds within each held-out dataset.

### Native zero-shot by dataset

| model | Inclusive-HAR | MoniPar | SPAR | TNDA-HAR | USC-HAD | UT Complex | Upper Limb Use |
|---|---:|---:|---:|---:|---:|---:|---:|
| HALO (ours) | 27.08 | 14.63 | 10.98 | 37.44 | **26.04** | **34.92** | 5.50 |
| UniMTS | **30.53** | **22.22** | **22.21** | **45.12** | 24.17 | 28.09 | **7.68** |
| ImageBind | 18.84 | 5.69 | 11.78 | 13.02 | 6.02 | 7.64 | 6.97 |
| NormWear | 8.60 | 0.76 | 6.10 | 5.11 | 4.82 | 1.80 | 3.87 |

### Enrollment by dataset

#### Inclusive-HAR

| model / readout | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO / retrieve-mix-vote | 32.19 | 34.59 | 34.60 | 39.74 | 44.90 |
| HALO / 1-NN | 33.24 | 33.50 | 36.10 | 39.36 | 43.26 |
| HALO / prototype | 33.24 | 34.21 | 35.01 | 38.08 | 38.77 |
| HALO / ridge | 33.05 | 34.13 | 36.39 | 39.43 | 43.30 |
| UniMTS / 1-NN | **35.40** | **38.63** | **41.93** | **44.24** | **49.04** |
| UniMTS / prototype | **35.40** | 35.55 | 38.63 | 41.59 | 43.70 |
| UniMTS / ridge | 34.80 | 35.42 | 39.52 | 42.52 | 44.96 |
| HARNet / 1-NN | 33.04 | 34.37 | 34.26 | 34.48 | 35.85 |
| HARNet / prototype | 33.04 | 34.14 | 35.46 | 36.68 | 37.98 |
| HARNet / ridge | 32.65 | 34.39 | 36.02 | 37.49 | 40.26 |
| ImageBind / 1-NN | 23.15 | 27.60 | 29.12 | 32.69 | 35.58 |
| ImageBind / prototype | 23.15 | 25.01 | 27.20 | 28.25 | 34.14 |
| ImageBind / ridge | 23.06 | 26.20 | 29.07 | 29.64 | 32.95 |
| NormWear / 1-NN | 25.94 | 26.52 | 27.57 | 29.30 | 30.66 |
| NormWear / prototype | 25.94 | 25.76 | 26.42 | 26.95 | 26.06 |
| NormWear / ridge | 25.12 | 24.66 | 24.85 | 24.50 | 23.02 |

#### MoniPar

| model / readout | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO / retrieve-mix-vote | 34.77 | 38.12 | 32.94 | 35.77 | 38.46 |
| HALO / 1-NN | 37.18 | 40.91 | 37.19 | 39.83 | 42.27 |
| HALO / prototype | 37.18 | 39.58 | 36.14 | 37.06 | 37.78 |
| HALO / ridge | 35.22 | 38.34 | 35.36 | 38.62 | 42.39 |
| UniMTS / 1-NN | **43.31** | 45.65 | 41.32 | 42.65 | 43.00 |
| UniMTS / prototype | **43.31** | **46.43** | **42.74** | 43.58 | 42.92 |
| UniMTS / ridge | 38.84 | 42.36 | 40.44 | 42.62 | 43.99 |
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
| HALO / retrieve-mix-vote | 57.51 | 57.23 | 60.16 | 63.85 | 66.45 |
| HALO / 1-NN | **57.95** | **57.60** | 62.43 | 66.93 | 71.71 |
| HALO / prototype | **57.95** | 56.36 | 61.31 | 64.82 | 67.29 |
| HALO / ridge | 56.84 | 57.32 | **63.74** | **70.15** | **74.32** |
| UniMTS / 1-NN | 52.86 | 50.65 | 56.92 | 62.89 | 67.11 |
| UniMTS / prototype | 52.86 | 48.67 | 53.00 | 56.81 | 58.89 |
| UniMTS / ridge | 51.25 | 48.30 | 52.99 | 57.94 | 61.72 |
| HARNet / 1-NN | 45.09 | 42.97 | 47.67 | 50.98 | 54.51 |
| HARNet / prototype | 45.09 | 42.42 | 46.15 | 49.20 | 51.17 |
| HARNet / ridge | 43.88 | 41.72 | 45.95 | 50.80 | 56.77 |
| ImageBind / 1-NN | 36.58 | 41.74 | 44.29 | 46.82 | 48.53 |
| ImageBind / prototype | 36.58 | 41.98 | 44.13 | 46.31 | 47.48 |
| ImageBind / ridge | 35.94 | 42.17 | 45.61 | 49.32 | 53.16 |
| NormWear / 1-NN | 22.69 | 23.82 | 26.22 | 28.80 | 31.82 |
| NormWear / prototype | 22.69 | 21.53 | 21.95 | 21.23 | 20.07 |
| NormWear / ridge | 20.30 | 18.86 | 19.06 | 18.35 | 17.45 |

#### TNDA-HAR

| model / readout | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO / retrieve-mix-vote | **57.74** | 61.91 | 66.34 | 69.46 | 72.85 |
| HALO / 1-NN | 57.15 | 61.94 | 66.25 | 69.37 | 72.88 |
| HALO / prototype | 57.15 | 60.87 | 65.43 | 66.60 | 67.81 |
| HALO / ridge | 56.88 | 60.10 | 66.67 | 68.25 | 70.90 |
| UniMTS / 1-NN | 56.84 | **63.99** | **70.94** | **76.82** | **80.00** |
| UniMTS / prototype | 56.86 | 57.12 | 63.18 | 65.70 | 65.34 |
| UniMTS / ridge | 52.02 | 53.20 | 58.05 | 61.63 | 63.15 |
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
| HALO / retrieve-mix-vote | 51.94 | 51.80 | 53.73 | 40.58 | 44.23 |
| HALO / 1-NN | **56.06** | **60.22** | 63.27 | 51.09 | 52.55 |
| HALO / prototype | **56.06** | 59.31 | 60.53 | 49.65 | 50.46 |
| HALO / ridge | 50.80 | 53.53 | 56.32 | 43.57 | 46.86 |
| UniMTS / 1-NN | 53.62 | 59.62 | **63.45** | **56.99** | **58.99** |
| UniMTS / prototype | 53.62 | 56.78 | 59.33 | 52.46 | 54.82 |
| UniMTS / ridge | 50.97 | 54.36 | 57.05 | 51.86 | 54.38 |
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
| HALO / retrieve-mix-vote | 60.15 | 62.41 | 67.12 | 68.70 | n/a |
| HALO / 1-NN | 59.80 | **64.07** | **68.95** | **72.24** | n/a |
| HALO / prototype | 59.80 | 63.08 | 66.12 | 68.04 | n/a |
| HALO / ridge | **60.21** | 63.69 | 67.52 | 71.15 | n/a |
| UniMTS / 1-NN | 56.89 | 62.55 | 67.70 | 70.82 | n/a |
| UniMTS / prototype | 56.89 | 59.63 | 62.03 | 64.20 | n/a |
| UniMTS / ridge | 52.83 | 56.37 | 59.87 | 62.66 | n/a |
| HARNet / 1-NN | 54.09 | 59.06 | 63.30 | 66.45 | n/a |
| HARNet / prototype | 54.09 | 57.75 | 60.89 | 63.48 | n/a |
| HARNet / ridge | 52.15 | 56.94 | 62.57 | 66.69 | n/a |
| ImageBind / 1-NN | 50.64 | 56.60 | 63.77 | 69.99 | n/a |
| ImageBind / prototype | 50.64 | 54.56 | 57.23 | 60.14 | n/a |
| ImageBind / ridge | 51.95 | 57.60 | 62.64 | 68.49 | n/a |
| NormWear / 1-NN | 26.59 | 30.92 | 35.05 | 39.64 | n/a |
| NormWear / prototype | 26.59 | 28.35 | 29.24 | 30.06 | n/a |
| NormWear / ridge | 14.96 | 13.79 | 13.11 | 14.27 | n/a |

#### Upper Limb Use

| model / readout | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO / retrieve-mix-vote | 15.24 | 13.62 | n/a | n/a | n/a |
| HALO / 1-NN | 16.73 | 12.93 | n/a | n/a | n/a |
| HALO / prototype | 16.73 | 15.70 | n/a | n/a | n/a |
| HALO / ridge | **16.81** | 15.46 | n/a | n/a | n/a |
| UniMTS / 1-NN | 14.90 | 13.83 | n/a | n/a | n/a |
| UniMTS / prototype | 14.90 | 14.36 | n/a | n/a | n/a |
| UniMTS / ridge | 13.07 | 13.68 | n/a | n/a | n/a |
| HARNet / 1-NN | 12.75 | 12.47 | n/a | n/a | n/a |
| HARNet / prototype | 12.75 | 13.51 | n/a | n/a | n/a |
| HARNet / ridge | 12.02 | 12.54 | n/a | n/a | n/a |
| ImageBind / 1-NN | 16.39 | 14.14 | n/a | n/a | n/a |
| ImageBind / prototype | 16.39 | **17.34** | n/a | n/a | n/a |
| ImageBind / ridge | 15.92 | 17.24 | n/a | n/a | n/a |
| NormWear / 1-NN | 10.00 | 10.24 | n/a | n/a | n/a |
| NormWear / prototype | 10.00 | 9.02 | n/a | n/a | n/a |
| NormWear / ridge | 7.04 | 6.12 | n/a | n/a | n/a |
