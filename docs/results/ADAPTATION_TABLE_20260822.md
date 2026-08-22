# HALO compact engine vs baselines across k — matched adaptation protocol

**Generated** 2026-08-22 from `eval/adaptation_results/{v1_d85761d, halo_compact_20260822}` ·
manifest `adaptation_v1` (61 cells, 7 datasets, 5 seeds, execution-disjoint support/query,
fingerprint-verified identical across all rows) · **halo_compact** =
`training/tokenizer/outputs/long_4h_20260821/best.pt` (compact evidence engine, step 35k,
1,010,790 trainable params (learned scorer off), adapter `baselines/halo_compact`), baseline rows from `v1_d85761d`.
Macro-F1, coherent labels, dataset-macro aggregation. **Bold = best in column.**

How each row is scored:
* `nearest / prototype / ridge / linear_head` — the standard adaptation methods on each model's
  FROZEN window features (identical fitting code, identical support/query draws). halo_compact's
  features are its pooled per-(patch,sensor) retrieval rows (d=128, the smallest in the table).
* `zero_shot` — each model's own native rule: ConSE bridge for closed-vocab baselines, native
  text cosine for text-aligned ones, and for halo_compact the engine's deployed mechanism
  (rows → cosine top-64 over a frozen 512-window stratified training-corpus bank → evidence
  mixer → text vote). No heads fit.

21 manifest cells are `insufficient_independent_executions` for every model alike.

## ordinary · zero_shot
| model | k=0 |
|---|---:|
| halo_compact | 36.95 |
| harnet | 33.82 |
| unimts | 32.70 |
| crosshar | **37.01** |
| limubert | 27.60 |
| imagebind | 11.38 |
| normwear | 5.08 |

## ordinary · nearest · coherent
| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo_compact | **57.14** | **61.18** | 64.93 | 64.09 | 61.08 |
| harnet | 48.64 | 51.98 | 55.03 | 54.91 | 52.58 |
| unimts | 53.76 | 59.01 | 63.72 | 64.16 | **63.27** |
| crosshar | 51.77 | 56.35 | 61.25 | 60.24 | 57.71 |
| limubert | 54.39 | 61.05 | **65.33** | **65.00** | 60.89 |
| imagebind | 45.29 | 51.70 | 55.91 | 55.36 | 51.74 |
| normwear | 30.86 | 35.55 | 39.63 | 40.57 | 39.55 |

## ordinary · prototype · coherent
| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo_compact | **56.24** | **59.29** | **62.64** | **61.71** | **58.58** |
| harnet | 47.34 | 49.51 | 52.95 | 52.49 | 50.40 |
| unimts | 50.69 | 52.30 | 55.49 | 55.63 | 54.19 |
| crosshar | 50.95 | 54.17 | 57.64 | 56.86 | 54.51 |
| limubert | 53.21 | 58.25 | 61.13 | 60.22 | 57.39 |
| imagebind | 43.02 | 47.07 | 48.76 | 46.25 | 44.30 |
| normwear | 26.26 | 27.31 | 28.84 | 29.08 | 28.20 |

## ordinary · ridge · coherent
| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo_compact | **55.04** | **59.39** | **64.24** | **64.44** | **60.61** |
| harnet | 48.21 | 52.05 | 57.48 | 58.86 | 55.47 |
| unimts | 47.31 | 51.47 | 56.61 | 58.85 | 56.89 |
| crosshar | 45.04 | 49.70 | 54.98 | 55.34 | 53.62 |
| limubert | 42.61 | 49.16 | 54.61 | 56.47 | 54.26 |
| imagebind | 43.38 | 50.49 | 54.79 | 53.62 | 48.79 |
| normwear | 22.13 | 23.03 | 25.29 | 27.21 | 23.52 |

## ordinary · linear_head · coherent
| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo_compact | **57.41** | **62.76** | **66.54** | 66.42 | 63.28 |
| harnet | 51.35 | 56.63 | 61.56 | 62.77 | 59.73 |
| unimts | 54.81 | 59.92 | 65.49 | **66.69** | **65.05** |
| crosshar | 51.94 | 57.46 | 63.02 | 63.53 | 61.37 |
| limubert | 54.26 | 61.00 | 64.80 | 64.65 | 60.29 |
| imagebind | 45.17 | 53.12 | 58.48 | 58.98 | 55.94 |
| normwear | 35.81 | 42.43 | 46.81 | 46.76 | 44.99 |

## specialized_novel · zero_shot
| model | k=0 |
|---|---:|
| halo_compact | 8.75 |
| harnet | 11.40 |
| unimts | **19.24** |
| crosshar | 10.88 |
| limubert | 9.11 |
| imagebind | 8.15 |
| normwear | 3.58 |

## specialized_novel · nearest · coherent
| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo_compact | **43.23** | **44.71** | **58.66** | **62.08** | **64.48** |
| harnet | 32.62 | 33.84 | 46.38 | 51.00 | 54.58 |
| unimts | 38.84 | 39.34 | 52.94 | 57.86 | 61.18 |
| crosshar | 31.65 | 35.31 | 44.38 | 47.28 | 50.10 |
| limubert | 31.99 | 34.62 | 42.42 | 46.77 | 49.25 |
| imagebind | 28.60 | 31.75 | 37.62 | 41.51 | 43.94 |
| normwear | 23.60 | 24.74 | 32.19 | 35.32 | 38.00 |

## specialized_novel · prototype · coherent
| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo_compact | **43.06** | **43.73** | **58.05** | **61.13** | **63.00** |
| harnet | 30.83 | 31.55 | 41.79 | 44.30 | 46.03 |
| unimts | 37.02 | 36.11 | 47.82 | 50.65 | 51.56 |
| crosshar | 31.27 | 35.00 | 44.41 | 46.40 | 47.60 |
| limubert | 29.90 | 31.85 | 37.07 | 38.45 | 39.14 |
| imagebind | 27.29 | 29.50 | 34.27 | 36.29 | 37.28 |
| normwear | 18.63 | 18.20 | 21.87 | 22.11 | 21.45 |

## specialized_novel · ridge · coherent
| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo_compact | **39.45** | **41.72** | **57.14** | **61.11** | **64.31** |
| harnet | 29.85 | 32.63 | 47.36 | 53.25 | 58.69 |
| unimts | 32.41 | 32.76 | 48.30 | 53.82 | 58.04 |
| crosshar | 26.43 | 29.98 | 41.27 | 44.33 | 47.02 |
| limubert | 16.84 | 19.75 | 27.91 | 31.50 | 35.63 |
| imagebind | 26.16 | 30.02 | 36.73 | 40.56 | 43.15 |
| normwear | 11.75 | 12.64 | 14.57 | 15.96 | 17.48 |

## specialized_novel · linear_head · coherent
| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo_compact | **43.36** | **46.01** | **61.44** | **66.17** | **69.46** |
| harnet | 34.72 | 37.54 | 54.40 | 61.34 | 66.37 |
| unimts | 38.95 | 39.38 | 55.23 | 61.04 | 65.17 |
| crosshar | 32.20 | 36.35 | 46.69 | 49.07 | 51.12 |
| limubert | 28.95 | 31.12 | 38.29 | 39.40 | 39.96 |
| imagebind | 29.19 | 32.91 | 40.36 | 44.98 | 48.53 |
| normwear | 25.31 | 25.94 | 33.89 | 36.36 | 38.19 |

## zero_shot (k=0) per dataset — the mechanism each model ships with

| model | inclusivehar | tnda_har | usc_had | ut_complex | **ord mean** | monipar | spar | upper_limb_use | **spec mean** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| halo_compact | 31.05 | 54.84 | 33.32 | 28.62 | **36.95** | 16.98 | 5.05 | 4.21 | **8.75** |
| harnet | 23.15 | 51.07 | 32.35 | 28.72 | **33.82** | 13.10 | 17.39 | 3.70 | **11.40** |
| unimts | 27.03 | 48.64 | 26.83 | 28.31 | **32.70** | 23.24 | 24.74 | 9.75 | **19.24** |
| crosshar | 25.49 | 54.59 | 29.02 | 38.96 | **37.01** | 11.83 | 6.39 | 14.43 | **10.88** |
| limubert | 28.69 | 44.41 | 21.24 | 16.06 | **27.60** | 10.29 | 8.04 | 9.01 | **9.11** |
| imagebind | 18.84 | 13.02 | 6.02 | 7.64 | **11.38** | 5.69 | 11.78 | 6.97 | **8.15** |
| normwear | 8.60 | 5.11 | 4.82 | 1.80 | **5.08** | 0.76 | 6.10 | 3.87 | **3.58** |

## Reading

* **Enrollment (the project's claim): halo_compact leads 35 of 40 method×k columns**, including
  every specialized_novel column at every k — the clinical/rehab regime where names carry no
  usable semantics and the representation must do the work. It does this at d=128, the smallest
  feature dimension in the table (baselines 512–2048). The five it does not win are all
  `ordinary` at high k: nearest k=4/8/16 (limubert 65.33/65.00, unimts 63.27) and linear_head
  k=8/16 (unimts 66.69/65.05) — i.e. everyday activities with many labelled examples, where a
  bigger frozen feature has room to be fitted and the semantic bridge no longer matters.
* **Zero-shot ordinary is competitive** (36.95, second by 0.06 to crosshar), through the native
  engine mechanism with no fitted head — the baselines' zero-shot rows all require a head fit on
  the training corpus (ConSE tier) or a text tower.
* **Zero-shot specialized_novel is weak (8.75)** — expected and honest: the engine's bank holds
  no clinical motions and their names don't project onto the training vocabulary. This is
  exactly the case enrollment exists for, and one enrolled example already moves 8.75 → 43.
* Caveats: the audit's limubert scale finding (its rows here predate the fix and may understate
  it); unimts rows predate the label-text ensemble fix (zero-shot only; enrollment methods don't
  read label text). The old `halo` (Phase-A) row is superseded by halo_compact and omitted.
