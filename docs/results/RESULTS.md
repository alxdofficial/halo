# Current Best Results

> Last updated 2026-08-24. `PB-04-SET-SCALAR-1NN` is the promoted HALO checkpoint because it gives
> the strongest overall zero-shot result among the HALO checkpoints. Its deployed
> enrollment readout is 1-NN; retrieve-mix-vote is retained and reported as an auxiliary ablation.

**Result set:** `PB-04-SET-SCALAR-1NN` selected checkpoint. See the
[`Phase-B Version Registry`](PHASE_B_TRAINING_STATUS.md) for the exact checkpoint hash and for how it
differs from the previous `PB-03-PAIRWISE-1NN` model.

## Protocol

The promoted checkpoint is
`training/tokenizer/outputs/e2e_pb04_fixed_filterbank_35k_20260824/best.pt`. It was trained end to
end from random initialization for 35,000 steps and selected at step 10,000 using the predeclared
development metric. Training used partial-enrollment episodes with candidate counts 2, 4, 8, and
16. Each episode queried up to four labels and enrolled queried labels and distractors independently.

External evaluation uses the sealed `adaptation_v1` manifest: seven held-out datasets, five seeds,
execution-disjoint support and query sets, and no test-set training. Here `k` has the standard N-way
k-shot meaning: every candidate receives exactly `k` independent enrolled executions. The query
cohort and candidate roster remain fixed across k. Macro F1 is averaged within each dataset and then
equally across datasets.

The strict assembler validated 7,997 coherent-label cells. Generated aggregate and per-dataset
tables are in
[`../../eval/adaptation_tables/e2e_set_scalar_1nn_35k_20260824_best_full/headline_tables.md`](../../eval/adaptation_tables/e2e_set_scalar_1nn_35k_20260824_best_full/headline_tables.md).

## Zero-Shot Recognition

No labelled target-dataset execution is available at k=0. Each model uses its native zero-shot
mechanism.

| model | all held-out datasets |
|---|---:|
| CrossHAR | **26.35** |
| UniMTS | 25.72 |
| HARNet | 24.21 |
| LIMU-BERT | 21.89 |
| **HALO PB-04** | 20.27 |
| ImageBind | 10.00 |
| NormWear | 4.44 |

HALO is not competitive at k=0. The current design should therefore be described as an enrollment
adaptation system, not as a leading semantic zero-shot classifier.

## Label-Efficient Adaptation

HALO retrieve-mix-vote inference keeps individual six-second recording rows, combines enrollment with its
fixed 512-row corpus memory, applies the learned scalar correction to every query-memory pair, and
chooses the corrected nearest candidate. For every encoder, the table also reports the same three
non-gradient enrollment rules: pooled-execution 1-NN, support prototypes, and closed-form ridge.
Each sees only the enrolled support executions. Linear-head fitting is excluded from the main table.

| model / readout | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| HALO / retrieve-mix-vote | 41.28 | 42.87 | 48.85 | 48.22 | 47.68 |
| **HALO / 1-NN** | **46.13** | 48.59 | 56.62 | 57.77 | 57.19 |
| HALO / prototype | **46.13** | 48.41 | 54.27 | 55.00 | 52.53 |
| HALO / ridge | 45.30 | 47.35 | 54.19 | 55.92 | 56.03 |
| LIMU-BERT / 1-NN | 45.63 | **49.90** | 56.92 | 57.52 | 54.52 |
| LIMU-BERT / prototype | 45.63 | 48.69 | 55.48 | 55.50 | 52.17 |
| LIMU-BERT / ridge | 40.51 | 43.73 | 50.96 | 51.32 | 49.19 |
| UniMTS / 1-NN | 44.83 | 47.85 | **57.04** | **59.07** | **59.63** |
| UniMTS / prototype | 44.83 | 45.51 | 53.15 | 54.05 | 53.14 |
| UniMTS / ridge | 41.97 | 43.38 | 51.32 | 53.21 | 53.64 |
| CrossHAR / 1-NN | 41.02 | 44.97 | 53.30 | 53.48 | 52.61 |
| CrossHAR / prototype | 41.02 | 44.79 | 52.43 | 52.25 | 51.26 |
| CrossHAR / ridge | 36.04 | 39.49 | 47.03 | 48.19 | 48.55 |
| HARNet / 1-NN | 40.26 | 42.51 | 50.06 | 50.98 | 50.81 |
| HARNet / prototype | 40.26 | 41.82 | 49.18 | 49.66 | 48.61 |
| HARNet / ridge | 39.61 | 41.66 | 50.31 | 52.27 | 53.25 |
| ImageBind / 1-NN | 36.28 | 40.72 | 47.29 | 48.22 | 46.83 |
| ImageBind / prototype | 36.28 | 39.71 | 43.85 | 42.94 | 41.59 |
| ImageBind / ridge | 35.77 | 40.31 | 46.22 | 47.59 | 47.02 |
| NormWear / 1-NN | 22.99 | 25.55 | 30.49 | 32.97 | 34.69 |
| NormWear / prototype | 22.99 | 23.30 | 26.38 | 26.46 | 25.12 |
| NormWear / ridge | 18.11 | 17.53 | 19.98 | 20.84 | 22.21 |

Seven datasets contribute at k=1 and k=2, six at k=4 and k=8, and five at k=16 because the sealed
protocol never reuses an execution to manufacture a larger support set.

![Primary adaptation curves](figures/e2e_set_scalar_1nn_35k_20260824_best/primary_adaptation_curves.png)

PB-04's 1-NN readout is the promoted enrollment mechanism. Retrieve-mix-vote is below 1-NN at every
k and is reported to make that negative result explicit. The per-dataset tables below show where
the aggregate differences originate.

## Per-Dataset Performance

The complete per-dataset report contains native zero-shot results and separate `retrieve-mix-vote`,
1-NN, prototype, and ridge curves for every model on all seven held-out datasets:
[`headline_tables.md`](../../eval/adaptation_tables/e2e_set_scalar_1nn_35k_20260824_best_full/headline_tables.md#3-per-dataset-performance).
The promoted HALO mechanism is summarized below; `n/a` means the sealed protocol could not form the
requested support count without reusing an execution.

| dataset | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|---:|
| Inclusive-HAR | 23.21 | 34.90 | 37.09 | 38.24 | 41.96 | 43.85 |
| MoniPar | 15.19 | 38.40 | 41.80 | 38.09 | 41.08 | 43.47 |
| SPAR | 15.35 | 52.02 | 52.48 | 58.23 | 63.05 | 66.23 |
| TNDA-HAR | 46.21 | 59.07 | 63.91 | 68.69 | 72.12 | 76.34 |
| Upper Limb Use | 3.43 | 17.61 | 14.25 | n/a | n/a | n/a |
| USC-HAD | 14.64 | 56.26 | 60.81 | 63.75 | 53.18 | 56.03 |
| UT Complex | 23.88 | 64.65 | 69.83 | 72.70 | 75.21 | n/a |

## Checkpoint And Readout Selection

PB-04 trained for 35,000 steps in 4,314 seconds. The predeclared development metric selected step
10,000. Both that checkpoint and the final step were evaluated on the same sealed manifest. Values
below are mean macro F1; positive-k columns average k=1,2,4,8,16.

| checkpoint / readout | k=0 | mean k>0 |
|---|---:|---:|
| PB-03 retrieve-mix-vote | 16.01 | **54.55** |
| PB-03 1-NN | - | 53.30 |
| PB-04 best retrieve-mix-vote | 20.27 | 45.78 |
| PB-04 best 1-NN | - | 53.26 |
| PB-04 last retrieve-mix-vote | **20.30** | 38.65 |
| PB-04 last 1-NN | - | 52.30 |

PB-04 is promoted for its stronger zero-shot performance over PB-03. Its selected encoder and PB-03
are effectively tied under the overall 1-NN enrollment average (53.26 versus 53.30). The learned
mechanism is far below its own 1-NN control, and late training makes that gap larger. Therefore the
promoted model is PB-04 with 1-NN enrollment, not PB-04 retrieve-mix-vote. Full tables are in
[`e2e_set_scalar_1nn_35k_20260824_best_full/headline_tables.md`](../../eval/adaptation_tables/e2e_set_scalar_1nn_35k_20260824_best_full/headline_tables.md).

## Learned Readout Finding

At every k, PB-04 retrieve-mix-vote is below PB-04 1-NN. This is consistent
with the prior PB-03 decomposition, where most of the apparent native gain came from retaining
individual recording rows rather than from learned score correction. No current experiment shows
that a learned Phase-B mixer improves a matched nearest-neighbor decision.

## Training Health

PB-04 completed 35,000 finite steps in 71.9 minutes. Development macro F1 rose from 0.2010 before
training to 0.3642 at step 10,000, then ended at 0.3353. External evaluation confirms that the
selected checkpoint is preferable: the final checkpoint loses 5.5-8.9 F1 on the aggregate
retrieve-mix-vote curve. Encoder-only 1-NN changes much less, so late training primarily overfits
the learned correction.

## Current Conclusion

PB-04 is the current best HALO checkpoint because it provides the strongest zero-shot result. For
enrollment adaptation, its supported mechanism is simple 1-NN over the learned representation. The
encoder and append-only enrollment memory are supported by the experiments; a learned Phase-B mixer
is not. The next clean experiment is end-to-end episodic encoder training with a differentiable
soft-nearest-neighbor objective and exact hard 1-NN inference, with no mixer, vote head, or learned
reranker.
