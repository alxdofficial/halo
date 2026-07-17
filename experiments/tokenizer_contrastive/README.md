# Contrastive Tokenizer Robustness Study

This experiment tests whether the existing HALO V2 physical-Hz tokenizer can
learn activity-level and geometric robustness without training the complete
classification/language-alignment model.

## Scope

The physical-Hz filterbank and band centers remain fixed. Training updates:

- the tokenizer's per-channel linear projection;
- an accelerometer/gyroscope modality embedding;
- patch averaging that preserves six ordered channel tokens, followed by a
  minimal representation MLP;
- a contrastive projection head that is discarded for downstream use.

It does not yet add gravity-relative vertical/horizontal features. That is a
separate controlled variant after this experiment establishes whether paired
SO(3) views improve the current tokenizer.

The six channel tokens are concatenated, not averaged across xyz. Axis averaging
would make summed frequency energy rotation-invariant before training and would
invalidate the comparison.

## Data and positives

The experiment consumes generated `harmonised` grids and defaults to the primary
training datasets. Dataset-scoped subjects are assigned to disjoint train,
validation, and test splits across all streams. Each batch selects multiple
canonical activities and spreads each activity's windows across available
datasets.

Two independent views are generated for every window. In supervised mode, all
views with the same canonical activity are positives and different activities
are negatives. In instance mode, only two views of the same window are
positives. Rotation is a shared proper SO(3) transform for co-located
accelerometer and gyroscope triads.

## Commands

CPU smoke test:

```bash
python -m experiments.tokenizer_contrastive.train \
  --smoke --run-name smoke --device cpu
```

An existing non-empty run directory is never overwritten unless `--overwrite`
is supplied explicitly. Any field can be configured by passing a JSON file with
the same schema as a generated `config.json`:

```bash
python -m experiments.tokenizer_contrastive.train \
  --config my_experiment.json --run-name configured_run --device cuda
```

Controlled runs on a remote GPU:

```bash
# No geometric augmentation.
python -m experiments.tokenizer_contrastive.train \
  --run-name axis_no_rotation --rotation-probability 0 --device cuda

# Paired random rotations.
python -m experiments.tokenizer_contrastive.train \
  --run-name axis_so3 --rotation-probability 1 --device cuda
```

Real full-corpus runs use all primary training windows and derive every label
represented in the subject-disjoint training split:

```bash
# Matched control.
python -m experiments.tokenizer_contrastive.train \
  --full-train-corpus --run-name full_axis_no_rotation \
  --rotation-probability 0 --device cuda

# Geometric contrastive run.
python -m experiments.tokenizer_contrastive.train \
  --full-train-corpus --run-name full_axis_so3 \
  --rotation-probability 1 --device cuda
```

The full preset removes the per-domain/label cap and uses a coverage-balanced
epoch sampler. Every indexed training window is consumed at least once per
epoch; filler samples maintain the required number of classes and positives in
tail batches. Evaluation datasets remain excluded.

Evaluate a checkpoint:

```bash
python -m experiments.tokenizer_contrastive.evaluate \
  --checkpoint experiments/tokenizer_contrastive/outputs/axis_so3/best.pt \
  --split test --device cuda
```

The default device is CPU so this experiment never takes a shared GPU unless
`--device cuda` or `--device auto` is explicitly supplied.

## Outputs

Each run writes under `outputs/<run-name>/`:

```text
config.json
index_summary.json
tokenizer_source.json
metrics.jsonl
initial.pt
best.pt
last.pt
summary.json
evaluation_test.json
```

Checkpoints record and validate the exact SHA-256 hash of the sibling tokenizer
source. Until HALO V2 is ported into this clean repository, `--legacy-root`
selects that sibling repository. Both training and evaluation accept the option,
so a checkpoint can move to a pod with a different filesystem path while still
rejecting tokenizer source drift.

Every run evaluates and saves `initial.pt` before the first optimizer step. This
provides the matched epoch-zero robustness and retrieval baseline. Evaluation
randomness is reseeded per epoch so controlled runs receive reproducible stress
views.

## Interpretation

Validation and test views use two independent full SO(3) rotations even when a
checkpoint was trained with rotation disabled. The primary diagnostics are
paired-view cosine similarity for both the reusable representation and the
disposable contrastive projection under this fixed stress test. Activity
retrieval tests whether invariance was learned without collapsing activity
classes. Cross-domain retrieval restricts nearest-neighbour candidates to other
datasets/streams. Retrieval excludes every repeated copy of the query recording.
Embedding standard deviation and effective rank are collapse indicators.
Same-activity, different-activity, and paired-over-different similarity margins
prevent uniformly high cosine similarity from being mistaken for useful
invariance.

The first comparison must keep data, seed, optimizer, and training budget fixed:

1. `axis_no_rotation`: current tokenizer without SO(3) views.
2. `axis_so3`: current tokenizer with SO(3) positive pairs.
3. A later gravity-relative tokenizer variant under the same SO(3) protocol.
