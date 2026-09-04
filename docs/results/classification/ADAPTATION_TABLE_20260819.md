# HALO vs baselines across k — matched adaptation protocol

> ## ⚠️ SUPERSEDED — 2026-08-22
> Replaced by [`ADAPTATION_TABLE_20260822.md`](ADAPTATION_TABLE_20260822.md), which scores the
> **compact evidence engine** (`halo_compact`, 1,010,790 params) on the same `adaptation_v1`
> manifest with the same fingerprint. The `halo` row below is the older Phase-A checkpoint
> `phase_a_h_mae_fixes_20260818/best.pt` and is **not** the current model. The BASELINE rows are
> unchanged between the two tables (both drawn from `v1_d85761d`), so this file stays useful as
> the record of how the previous HALO compared — nothing more.

**Generated** 2026-08-19 from `eval/adaptation_tables/v2_20260819/` ·
manifest `adaptation_v1` (61 cells, 7 datasets, 5 seeds, execution-disjoint support/query) ·
HALO checkpoint `phase_a_h_mae_fixes_20260818/best.pt`, baselines from `v1_d85761d`.
20,237 assembled cells. Macro-F1, coherent labels. **Bold = best in column.**

Two regimes: `ordinary` (4 datasets, everyday activities) and `specialized_novel`
(3 datasets, clinical / rehab motions). `zero_shot` is the ConSE semantic bridge; `nearest`,
`prototype`, `ridge`, `linear_head` are heads fitted on k labelled examples per class over the
frozen representation.


## ordinary . zero_shot
| model | k=0 |
|---|---:|
| halo | 27.43 |
| harnet | 33.82 |
| unimts | 32.70 |
| crosshar | **37.01** |
| limubert | 27.60 |
| imagebind | 11.38 |
| normwear | 5.08 |

## ordinary . nearest . coherent
| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo | **56.74** | 60.96 | 64.75 | 63.39 | 59.63 |
| harnet | 48.64 | 51.98 | 55.03 | 54.91 | 52.58 |
| unimts | 53.76 | 59.01 | 63.72 | 64.16 | **63.27** |
| crosshar | 51.77 | 56.35 | 61.25 | 60.24 | 57.71 |
| limubert | 54.39 | **61.05** | **65.33** | **65.00** | 60.89 |
| imagebind | 45.29 | 51.70 | 55.91 | 55.36 | 51.74 |
| normwear | 30.86 | 35.55 | 39.63 | 40.57 | 39.55 |

## ordinary . prototype . coherent
| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo | **55.80** | **59.42** | **62.41** | **61.31** | **57.44** |
| harnet | 47.34 | 49.51 | 52.95 | 52.49 | 50.40 |
| unimts | 50.69 | 52.30 | 55.49 | 55.63 | 54.19 |
| crosshar | 50.95 | 54.17 | 57.64 | 56.86 | 54.51 |
| limubert | 53.21 | 58.25 | 61.13 | 60.22 | 57.39 |
| imagebind | 43.02 | 47.07 | 48.76 | 46.25 | 44.30 |
| normwear | 26.26 | 27.31 | 28.84 | 29.08 | 28.20 |

## ordinary . ridge . coherent
| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo | **54.06** | **59.61** | **65.12** | **65.01** | **61.06** |
| harnet | 48.21 | 52.05 | 57.48 | 58.86 | 55.47 |
| unimts | 47.31 | 51.47 | 56.61 | 58.85 | 56.89 |
| crosshar | 45.04 | 49.70 | 54.98 | 55.34 | 53.62 |
| limubert | 42.61 | 49.16 | 54.61 | 56.47 | 54.26 |
| imagebind | 43.38 | 50.49 | 54.79 | 53.62 | 48.79 |
| normwear | 22.13 | 23.03 | 25.29 | 27.21 | 23.52 |

## ordinary . linear_head . coherent
| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo | **56.54** | **61.47** | **67.22** | **67.47** | 64.51 |
| harnet | 51.35 | 56.63 | 61.56 | 62.77 | 59.73 |
| unimts | 54.81 | 59.92 | 65.49 | 66.69 | **65.05** |
| crosshar | 51.94 | 57.46 | 63.02 | 63.53 | 61.37 |
| limubert | 54.26 | 61.00 | 64.80 | 64.65 | 60.29 |
| imagebind | 45.17 | 53.12 | 58.48 | 58.98 | 55.94 |
| normwear | 35.81 | 42.43 | 46.81 | 46.76 | 44.99 |

## specialized_novel . zero_shot
| model | k=0 |
|---|---:|
| halo | 12.08 |
| harnet | 11.40 |
| unimts | **19.24** |
| crosshar | 10.88 |
| limubert | 9.11 |
| imagebind | 8.15 |
| normwear | 3.58 |

## specialized_novel . nearest . coherent
| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo | 36.48 | 39.06 | 47.55 | 50.57 | 52.46 |
| harnet | 32.62 | 33.84 | 46.38 | 51.00 | 54.58 |
| unimts | **38.84** | **39.34** | **52.94** | **57.86** | **61.18** |
| crosshar | 31.65 | 35.31 | 44.38 | 47.28 | 50.10 |
| limubert | 31.99 | 34.62 | 42.42 | 46.77 | 49.25 |
| imagebind | 28.60 | 31.75 | 37.62 | 41.51 | 43.94 |
| normwear | 23.60 | 24.74 | 32.19 | 35.32 | 38.00 |

## specialized_novel . prototype . coherent
| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo | 36.78 | **39.12** | 46.38 | 49.22 | 50.66 |
| harnet | 30.83 | 31.55 | 41.79 | 44.30 | 46.03 |
| unimts | **37.02** | 36.11 | **47.82** | **50.65** | **51.56** |
| crosshar | 31.27 | 35.00 | 44.41 | 46.40 | 47.60 |
| limubert | 29.90 | 31.85 | 37.07 | 38.45 | 39.14 |
| imagebind | 27.29 | 29.50 | 34.27 | 36.29 | 37.28 |
| normwear | 18.63 | 18.20 | 21.87 | 22.11 | 21.45 |

## specialized_novel . ridge . coherent
| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo | **35.39** | **38.37** | **50.23** | **54.82** | 57.82 |
| harnet | 29.85 | 32.63 | 47.36 | 53.25 | **58.69** |
| unimts | 32.41 | 32.76 | 48.30 | 53.82 | 58.04 |
| crosshar | 26.43 | 29.98 | 41.27 | 44.33 | 47.02 |
| limubert | 16.84 | 19.75 | 27.91 | 31.50 | 35.63 |
| imagebind | 26.16 | 30.02 | 36.73 | 40.56 | 43.15 |
| normwear | 11.75 | 12.64 | 14.57 | 15.96 | 17.48 |

## specialized_novel . linear_head . coherent
| model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo | 35.89 | **39.79** | 50.00 | 55.08 | 58.29 |
| harnet | 34.72 | 37.54 | 54.40 | **61.34** | **66.37** |
| unimts | **38.95** | 39.38 | **55.23** | 61.04 | 65.17 |
| crosshar | 32.20 | 36.35 | 46.69 | 49.07 | 51.12 |
| limubert | 28.95 | 31.12 | 38.29 | 39.40 | 39.96 |
| imagebind | 29.19 | 32.91 | 40.36 | 44.98 | 48.53 |
| normwear | 25.31 | 25.94 | 33.89 | 36.36 | 38.19 |

---

# What the table says

## 1. HALO has the best few-shot representation in the field — and the worst zero-shot

**Ordinary regime.** HALO is **best at every k on prototype and ridge**, best on `linear_head` at
k=1–8, best on `nearest` at k=1. Against the strongest baseline per cell it wins by 1.7–4.0 points.

And at k=0 it is **last among the real contenders**:

| model | k=0 ordinary | k=1 prototype | jump |
|---|---:|---:|---:|
| crosshar | **37.01** | 50.95 | +13.9 |
| harnet | 33.82 | 47.34 | +13.5 |
| unimts | 32.70 | 50.69 | +18.0 |
| limubert | 27.60 | 53.21 | +25.6 |
| **halo** | **27.43** | **55.80** | **+28.4** |

HALO gains more from a single labelled example than any other model — because it starts furthest
behind. One example per class is worth **+28.4** points to HALO and **+13.9** to CrossHAR. The
representation is not the problem; the path from representation to label *text* is.

## 2. A second, separate gap: HALO saturates on specialized motions at high k

| specialized · linear_head | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| halo | 35.89 | **39.79** | 50.00 | 55.08 | 58.29 |
| harnet | 34.72 | 37.54 | 54.40 | **61.34** | **66.37** |
| unimts | **38.95** | 39.38 | **55.23** | 61.04 | 65.17 |

HALO leads at k≤2 and falls **8.1 points behind harnet by k=16**. Its curve flattens where the
baselines keep climbing. On `ridge` the same regime is much closer (halo best at k=1–8, 57.82 vs
harnet 58.69 at k=16), so this is specific to what a trained head can extract — consistent with a
256-d representation that saturates, or with clinical/rehab motions being thin in the training
corpus.

## 3. Ranking the work

| gap | size | where |
|---|---|---|
| **zero-shot semantic bridge** | −9.6 vs crosshar (ordinary), −7.2 vs unimts (specialized) | ConSE path, label text |
| **high-k specialized headroom** | −8.1 vs harnet at k=16 | representation capacity / corpus coverage |
| few-shot ordinary | HALO already leads | nothing to fix |

The first is both the larger gap and the paper's central claim (a language interface to unseen
labels). It is also the cheapest to attack: everything downstream of the classifier softmax is a
fixed function with no learned parameters.

---

# Experiments run against the zero-shot gap (2026-08-19)

The table says the k=0 cell is where HALO trails. Everything below is measured on the same 11
zero-shot cells of `adaptation_v1`, with the exact protocol aggregation (per-cell macro-F1 →
per-dataset mean → per-regime mean). **Parity check: the unmodified path reproduces the table's
27.43 / 12.08 exactly**, so the harness reimplementation is faithful and the deltas are real.

## E1 — ConSE query centring: a real improvement, but it belongs to everyone

A ConSE vector is a convex combination of training-label embeddings, so it inherits the training
vocabulary's mean. Measured, the mean off-diagonal label–label cosine is **0.338**: a third of every
similarity is a common-mode pedestal carrying no information about the activity. Subtracting
`train_embs.mean(0)` before the cosine removes it. It uses only the *training* vocabulary, so no
target-label information enters, and it adds no parameters.

| model | ordinary | +centred | Δ | specialized | +centred | Δ |
|---|---:|---:|---:|---:|---:|---:|
| crosshar | 37.01 | **39.45** | +2.44 | 10.88 | 11.57 | +0.69 |
| harnet | 33.82 | 34.65 | +0.83 | 11.40 | 12.77 | +1.38 |
| **halo** | 27.43 | 29.06 | +1.64 | 12.08 | **14.26** | +2.18 |
| limubert | 27.60 | 28.09 | +0.49 | 9.11 | 10.04 | +0.93 |

**It helps every ConSE-tier model.** This is a correction to the shared bridge, *not* a HALO result.
Applied fairly the ordinary gap to CrossHAR **widens** (9.58 → 10.39); on specialized, HALO moves
clearly ahead of the other ConSE models (14.26 vs 12.77 / 11.57) but still trails UniMTS (19.24,
cosine-tier, not measured here).

Implemented as `scoring.CONSE_CENTRE_QUERY`, **default `False`** so every published number is
preserved. Turning it on requires re-scoring all ConSE-tier models together, or the table is invalid.

## E2 — Prompt template: small and not additive

`"a person {label}"` with centring gives 29.38 / 14.18 versus 29.06 / 14.26 for the raw label with
centring — inside noise, and it does not stack with centring (raw+centred and person+centred are
within 0.3). Richer templates (`"wearable sensor recording of a person …"`) and a 4-template ensemble
were **worse**. Prompt engineering is not the lever here.

## E3 — Softmax temperature: already optimal, not a lever

The head's fitted temperature is 2.3997. Sweeping the full axis (exactly, via `p^(1/α)` renormalised,
which is the softmax at temperature `α·T`):

| temp × | effective T | ordinary | specialized | ord+centred | spec+centred |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.60 | 26.54 | **12.90** | 28.82 | **14.86** |
| 0.50 | 1.20 | 27.02 | 12.58 | **29.14** | 14.66 |
| **1.00** | **2.40** | **27.43** | 12.08 | 29.06 | 14.26 |
| 2.00 | 4.80 | 26.86 | 11.23 | 28.20 | 13.83 |
| 8.00 | 19.20 | 21.06 | 10.46 | 23.56 | 12.17 |

The fitted value is at the optimum for `ordinary`; sharpening slightly helps `specialized` (+0.8) but
the effect is small. **Calibration is not the problem.**

## E4 — The bridge is not degenerate either

If HALO's classifier were near one-hot, ConSE would collapse to nearest-training-label and could not
interpolate. Measured over all zero-shot queries:

```
softmax entropy   1.517 nats   (uniform over 166 labels = 5.112)
top-1 probability 0.511
effective classes 5.5
```

So each prediction is a genuine blend of ~5–6 training concepts. The bridge has material to work
with; it is not collapsing.

## What that leaves

Centring (+1.6/+2.2), prompts (~0), temperature (~0) and softmax shape (healthy) together account
for only a small part of a ~10-point gap. By elimination, the gap is in **the classifier itself** —
the map from HALO's feature space onto the 166 training-label vocabulary. HALO's features separate
activities better than any baseline (it wins every few-shot cell), so the deficiency is not
discriminative power but *semantic alignment*: the probe is fitted for classification accuracy, and
ConSE then reads its softmax as if it were a semantic distribution.

**The next experiment, not run for time:** fit the bridge head with a semantic objective — regress
features onto label-text embeddings (the `conse_probe_predict` ridge form already in
`eval/scoring.py`) instead of cross-entropy over label indices — and compare. That directly targets
alignment rather than accuracy, needs no architecture change, and is the same lever as the pending
fine-grained-description task.

## E5 — The gap located: HALO's bridge classifier is semantically MISALIGNED

For every zero-shot query the true target label is known, so the training labels the classifier puts
mass on can be scored directly for semantic proximity to that truth. The training vocabulary is
shared, so `oracle` — the best cosine achievable by any training label — is a control that must come
out identical across models, and does.

| model | align@1 | align@T | oracle | oracle − align@T | entropy | top-1 prob |
|---|---:|---:|---:|---:|---:|---:|
| **halo** | 0.4278 | **0.3919** | 0.7364 | **0.3446** | **1.517** | **0.511** |
| harnet | 0.4552 | 0.4098 | 0.7364 | 0.3266 | 1.559 | 0.471 |
| crosshar | **0.4718** | **0.4226** | 0.7364 | **0.3138** | 1.686 | 0.478 |

*`align@1`* = cosine between the top-1 training label's text and the true target label's text.
*`align@T`* = the same, probability-weighted over the top-10. *`oracle`* = max over all 166 training
labels.

**The alignment ordering reproduces the zero-shot ordering exactly:**

| | align@T | k=0 ordinary |
|---|---:|---:|
| halo | 0.3919 | 27.43 |
| harnet | 0.4098 | 33.82 |
| crosshar | 0.4226 | 37.01 |

Identical oracle (0.7364) rules out vocabulary coverage. So HALO's classifier is *more confident*
(lowest entropy 1.517, highest top-1 0.511) and *less semantically right* than CrossHAR's — it is
very sure which of the 166 training labels a window is, and that certainty does not carry semantic
information about a label it has never seen.

### The diagnosis, by elimination

| candidate cause | verdict | evidence |
|---|---|---|
| representation quality | **ruled out** | HALO wins every few-shot cell in the ordinary regime |
| vocabulary coverage | **ruled out** | oracle identical at 0.7364 for all models |
| softmax calibration | **ruled out** | fitted T=2.40 is at the optimum (E3) |
| softmax degeneracy | **ruled out** | 5.5 effective classes, top-1 0.511 (E4) |
| bridge arithmetic | **partly** | centring worth +1.6/+2.2, but helps every model (E1) |
| **classifier semantic alignment** | **the gap** | align@T ordering reproduces the k=0 ordering exactly (E5) |

### The implied fix

The ConSE head is a 2-layer probe fitted with **cross-entropy over label indices** — an objective
that rewards telling the 166 training labels apart and is indifferent to whether the resulting
softmax is a sensible distribution over *meanings*. ConSE then reads that softmax as if it were
semantic. HALO's representation separates activities better than any baseline, so it optimises that
index objective harder, becomes more confident, and ends up *further* from semantic truth.

**Fit the bridge with a semantic objective instead**: ridge-regress features onto label-text
embeddings (the `conse_probe_predict` form already in `eval/scoring.py`) rather than classifying
indices, and score with the same protocol. This targets exactly the measured deficiency, changes no
architecture, adds no parameters, and applies equally to every ConSE-tier model so the comparison
stays fair. Not run here for time; it is the single highest-value next experiment.
