# Matched adaptation results

`k` is the number of independent enrolled executions per candidate. External-model
1-NN, prototype, and ridge controls use one equally weighted pooled vector per enrolled
execution and require no gradient fitting. HALO's retrieve-mix-vote mechanism instead
consumes the enrolled executions' patch/sensor rows.

## Semantic zero-shot

| regime | model | method | k | macro F1 | datasets |
|---|---|---:|---:|---:|---:|
| ordinary | halo_compact | zero_shot | 0 | 29.31 | 4 |
| ordinary | imagebind | zero_shot | 0 | 11.38 | 4 |
| ordinary | normwear | zero_shot | 0 | 5.08 | 4 |
| ordinary | unimts | zero_shot | 0 | 31.98 | 4 |
| specialized_novel | halo_compact | zero_shot | 0 | 7.68 | 3 |
| specialized_novel | imagebind | zero_shot | 0 | 8.15 | 3 |
| specialized_novel | normwear | zero_shot | 0 | 3.58 | 3 |
| specialized_novel | unimts | zero_shot | 0 | 17.37 | 3 |

## Coherent adaptation comparison

| regime | model | method | k | macro F1 | datasets |
|---|---|---:|---:|---:|---:|
| ordinary | halo_compact | retrieve-mix-vote | 1 | 41.26 | 4 |
| ordinary | halo_compact | retrieve-mix-vote | 2 | 43.41 | 4 |
| ordinary | halo_compact | retrieve-mix-vote | 4 | 45.25 | 4 |
| ordinary | halo_compact | retrieve-mix-vote | 8 | 44.42 | 4 |
| ordinary | halo_compact | retrieve-mix-vote | 16 | 39.76 | 3 |
| ordinary | halo_compact | 1-NN | 1 | 51.87 | 4 |
| ordinary | halo_compact | 1-NN | 2 | 55.33 | 4 |
| ordinary | halo_compact | 1-NN | 4 | 58.79 | 4 |
| ordinary | halo_compact | 1-NN | 8 | 58.22 | 4 |
| ordinary | halo_compact | 1-NN | 16 | 55.63 | 3 |
| ordinary | halo_compact | prototype | 1 | 51.87 | 4 |
| ordinary | halo_compact | prototype | 2 | 54.22 | 4 |
| ordinary | halo_compact | prototype | 4 | 57.11 | 4 |
| ordinary | halo_compact | prototype | 8 | 55.84 | 4 |
| ordinary | halo_compact | prototype | 16 | 53.09 | 3 |
| ordinary | halo_compact | ridge | 1 | 50.70 | 4 |
| ordinary | halo_compact | ridge | 2 | 53.61 | 4 |
| ordinary | halo_compact | ridge | 4 | 57.15 | 4 |
| ordinary | halo_compact | ridge | 8 | 57.11 | 4 |
| ordinary | halo_compact | ridge | 16 | 53.79 | 3 |
| ordinary | harnet | 1-NN | 1 | 47.34 | 4 |
| ordinary | harnet | 1-NN | 2 | 50.51 | 4 |
| ordinary | harnet | 1-NN | 4 | 53.20 | 4 |
| ordinary | harnet | 1-NN | 8 | 52.66 | 4 |
| ordinary | harnet | 1-NN | 16 | 50.69 | 3 |
| ordinary | harnet | prototype | 1 | 47.34 | 4 |
| ordinary | harnet | prototype | 2 | 49.52 | 4 |
| ordinary | harnet | prototype | 4 | 52.93 | 4 |
| ordinary | harnet | prototype | 8 | 52.39 | 4 |
| ordinary | harnet | prototype | 16 | 50.40 | 3 |
| ordinary | harnet | ridge | 1 | 46.73 | 4 |
| ordinary | harnet | ridge | 2 | 49.16 | 4 |
| ordinary | harnet | ridge | 4 | 53.55 | 4 |
| ordinary | harnet | ridge | 8 | 53.97 | 4 |
| ordinary | harnet | ridge | 16 | 52.73 | 3 |
| ordinary | imagebind | 1-NN | 1 | 43.02 | 4 |
| ordinary | imagebind | 1-NN | 2 | 49.06 | 4 |
| ordinary | imagebind | 1-NN | 4 | 53.22 | 4 |
| ordinary | imagebind | 1-NN | 8 | 53.19 | 4 |
| ordinary | imagebind | 1-NN | 16 | 51.02 | 3 |
| ordinary | imagebind | prototype | 1 | 43.02 | 4 |
| ordinary | imagebind | prototype | 2 | 47.12 | 4 |
| ordinary | imagebind | prototype | 4 | 48.70 | 4 |
| ordinary | imagebind | prototype | 8 | 46.39 | 4 |
| ordinary | imagebind | prototype | 16 | 44.54 | 3 |
| ordinary | imagebind | ridge | 1 | 42.92 | 4 |
| ordinary | imagebind | ridge | 2 | 48.16 | 4 |
| ordinary | imagebind | ridge | 4 | 51.75 | 4 |
| ordinary | imagebind | ridge | 8 | 51.71 | 4 |
| ordinary | imagebind | ridge | 16 | 49.71 | 3 |
| ordinary | normwear | 1-NN | 1 | 26.26 | 4 |
| ordinary | normwear | 1-NN | 2 | 29.56 | 4 |
| ordinary | normwear | 1-NN | 4 | 32.92 | 4 |
| ordinary | normwear | 1-NN | 8 | 35.35 | 4 |
| ordinary | normwear | 1-NN | 16 | 37.11 | 3 |
| ordinary | normwear | prototype | 1 | 26.26 | 4 |
| ordinary | normwear | prototype | 2 | 27.17 | 4 |
| ordinary | normwear | prototype | 4 | 28.82 | 4 |
| ordinary | normwear | prototype | 8 | 28.80 | 4 |
| ordinary | normwear | prototype | 16 | 27.83 | 3 |
| ordinary | normwear | ridge | 1 | 20.68 | 4 |
| ordinary | normwear | ridge | 2 | 19.99 | 4 |
| ordinary | normwear | ridge | 4 | 20.76 | 4 |
| ordinary | normwear | ridge | 8 | 21.59 | 4 |
| ordinary | normwear | ridge | 16 | 23.89 | 3 |
| ordinary | unimts | 1-NN | 1 | 50.69 | 4 |
| ordinary | unimts | 1-NN | 2 | 56.20 | 4 |
| ordinary | unimts | 1-NN | 4 | 61.01 | 4 |
| ordinary | unimts | 1-NN | 8 | 62.22 | 4 |
| ordinary | unimts | 1-NN | 16 | 62.68 | 3 |
| ordinary | unimts | prototype | 1 | 50.69 | 4 |
| ordinary | unimts | prototype | 2 | 52.27 | 4 |
| ordinary | unimts | prototype | 4 | 55.79 | 4 |
| ordinary | unimts | prototype | 8 | 55.99 | 4 |
| ordinary | unimts | prototype | 16 | 54.62 | 3 |
| ordinary | unimts | ridge | 1 | 47.66 | 4 |
| ordinary | unimts | ridge | 2 | 49.84 | 4 |
| ordinary | unimts | ridge | 4 | 53.62 | 4 |
| ordinary | unimts | ridge | 8 | 54.67 | 4 |
| ordinary | unimts | ridge | 16 | 54.16 | 3 |
| specialized_novel | halo_compact | retrieve-mix-vote | 1 | 22.83 | 3 |
| specialized_novel | halo_compact | retrieve-mix-vote | 2 | 22.01 | 3 |
| specialized_novel | halo_compact | retrieve-mix-vote | 4 | 26.50 | 2 |
| specialized_novel | halo_compact | retrieve-mix-vote | 8 | 32.18 | 2 |
| specialized_novel | halo_compact | retrieve-mix-vote | 16 | 37.23 | 2 |
| specialized_novel | halo_compact | 1-NN | 1 | 35.21 | 3 |
| specialized_novel | halo_compact | 1-NN | 2 | 35.41 | 3 |
| specialized_novel | halo_compact | 1-NN | 4 | 46.69 | 2 |
| specialized_novel | halo_compact | 1-NN | 8 | 50.32 | 2 |
| specialized_novel | halo_compact | 1-NN | 16 | 53.56 | 2 |
| specialized_novel | halo_compact | prototype | 1 | 35.21 | 3 |
| specialized_novel | halo_compact | prototype | 2 | 34.09 | 3 |
| specialized_novel | halo_compact | prototype | 4 | 45.11 | 2 |
| specialized_novel | halo_compact | prototype | 8 | 47.24 | 2 |
| specialized_novel | halo_compact | prototype | 16 | 48.27 | 2 |
| specialized_novel | halo_compact | ridge | 1 | 34.38 | 3 |
| specialized_novel | halo_compact | ridge | 2 | 33.95 | 3 |
| specialized_novel | halo_compact | ridge | 4 | 46.02 | 2 |
| specialized_novel | halo_compact | ridge | 8 | 50.44 | 2 |
| specialized_novel | halo_compact | ridge | 16 | 54.87 | 2 |
| specialized_novel | harnet | 1-NN | 1 | 30.83 | 3 |
| specialized_novel | harnet | 1-NN | 2 | 31.84 | 3 |
| specialized_novel | harnet | 1-NN | 4 | 43.78 | 2 |
| specialized_novel | harnet | 1-NN | 8 | 47.62 | 2 |
| specialized_novel | harnet | 1-NN | 16 | 50.98 | 2 |
| specialized_novel | harnet | prototype | 1 | 30.83 | 3 |
| specialized_novel | harnet | prototype | 2 | 31.55 | 3 |
| specialized_novel | harnet | prototype | 4 | 41.69 | 2 |
| specialized_novel | harnet | prototype | 8 | 44.19 | 2 |
| specialized_novel | harnet | prototype | 16 | 45.91 | 2 |
| specialized_novel | harnet | ridge | 1 | 30.12 | 3 |
| specialized_novel | harnet | ridge | 2 | 31.66 | 3 |
| specialized_novel | harnet | ridge | 4 | 43.82 | 2 |
| specialized_novel | harnet | ridge | 8 | 48.89 | 2 |
| specialized_novel | harnet | ridge | 16 | 54.03 | 2 |
| specialized_novel | imagebind | 1-NN | 1 | 27.29 | 3 |
| specialized_novel | imagebind | 1-NN | 2 | 29.59 | 3 |
| specialized_novel | imagebind | 1-NN | 4 | 35.43 | 2 |
| specialized_novel | imagebind | 1-NN | 8 | 38.29 | 2 |
| specialized_novel | imagebind | 1-NN | 16 | 40.56 | 2 |
| specialized_novel | imagebind | prototype | 1 | 27.29 | 3 |
| specialized_novel | imagebind | prototype | 2 | 29.84 | 3 |
| specialized_novel | imagebind | prototype | 4 | 34.15 | 2 |
| specialized_novel | imagebind | prototype | 8 | 36.04 | 2 |
| specialized_novel | imagebind | prototype | 16 | 37.15 | 2 |
| specialized_novel | imagebind | ridge | 1 | 26.24 | 3 |
| specialized_novel | imagebind | ridge | 2 | 29.85 | 3 |
| specialized_novel | imagebind | ridge | 4 | 35.15 | 2 |
| specialized_novel | imagebind | ridge | 8 | 39.35 | 2 |
| specialized_novel | imagebind | ridge | 16 | 43.00 | 2 |
| specialized_novel | normwear | 1-NN | 1 | 18.63 | 3 |
| specialized_novel | normwear | 1-NN | 2 | 20.20 | 3 |
| specialized_novel | normwear | 1-NN | 4 | 25.61 | 2 |
| specialized_novel | normwear | 1-NN | 8 | 28.21 | 2 |
| specialized_novel | normwear | 1-NN | 16 | 31.06 | 2 |
| specialized_novel | normwear | prototype | 1 | 18.63 | 3 |
| specialized_novel | normwear | prototype | 2 | 18.14 | 3 |
| specialized_novel | normwear | prototype | 4 | 21.50 | 2 |
| specialized_novel | normwear | prototype | 8 | 21.76 | 2 |
| specialized_novel | normwear | prototype | 16 | 21.05 | 2 |
| specialized_novel | normwear | ridge | 1 | 14.68 | 3 |
| specialized_novel | normwear | ridge | 2 | 14.24 | 3 |
| specialized_novel | normwear | ridge | 4 | 18.43 | 2 |
| specialized_novel | normwear | ridge | 8 | 19.35 | 2 |
| specialized_novel | normwear | ridge | 16 | 19.67 | 2 |
| specialized_novel | unimts | 1-NN | 1 | 37.02 | 3 |
| specialized_novel | unimts | 1-NN | 2 | 36.71 | 3 |
| specialized_novel | unimts | 1-NN | 4 | 49.12 | 2 |
| specialized_novel | unimts | 1-NN | 8 | 52.77 | 2 |
| specialized_novel | unimts | 1-NN | 16 | 55.05 | 2 |
| specialized_novel | unimts | prototype | 1 | 37.02 | 3 |
| specialized_novel | unimts | prototype | 2 | 36.49 | 3 |
| specialized_novel | unimts | prototype | 4 | 47.87 | 2 |
| specialized_novel | unimts | prototype | 8 | 50.19 | 2 |
| specialized_novel | unimts | prototype | 16 | 50.91 | 2 |
| specialized_novel | unimts | ridge | 1 | 34.39 | 3 |
| specialized_novel | unimts | ridge | 2 | 34.78 | 3 |
| specialized_novel | unimts | ridge | 4 | 46.71 | 2 |
| specialized_novel | unimts | ridge | 8 | 50.28 | 2 |
| specialized_novel | unimts | ridge | 16 | 52.85 | 2 |

## Random-label binding

| regime | model | method | k | macro F1 | datasets |
|---|---|---:|---:|---:|---:|
| - | - | - | - | - | - |
