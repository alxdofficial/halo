# Phase-A tokenizer pretraining

Phase A trains the sensor representation consumed by the Phase-B evidence engine. The training
loss is activity-label-free; labels are read only by validation probes.

## Objectives

The live recipe has exactly two objectives:

| objective | supervision | default weight |
|---|---|---:|
| `jepa` | masked student predicts clean contextual tokens from an EMA teacher | 1.0 |
| `relation` | VICReg over two augmentations of every window, plus verified cross-placement agreement | 1.0 |

The relation objective contains two separately reduced positive-relation types:

1. **Augmentation relation:** available for every sampled window. Published VICReg MSE invariance,
   variance, and covariance terms provide the universal objective and collapse protection.
2. **Cross-placement relation:** available only for explicit simultaneous events. A scale-free cosine
   term is weighted by `cross_placement_weight=0.1`; its batch-window quota is 20%. Sparse placement
   pairs do not estimate their own covariance matrix and cannot dilute the universal relation.

The supported progression is:

```text
R0: jepa_weight=0, cross_placement_weight=0  (augmentation relation only)
R1: jepa_weight=0, cross_placement_weight>0  (unified relation)
R2: jepa_weight>0, cross_placement_weight>0  (final Phase-A Lite recipe)
```

Cadence/eigen primitives remain diagnostic probes in `probe_robustness.py`; they are not training
targets. Physical-feature reconstruction, SimCLR, SupCon, TF-C, separate A4, mask compensation,
label-balanced sampling, and automatic objective-weight calibration have been removed.

## JEPA masking

The student sees one physical-time mask shared across short and long resolutions. Masked temporal
attention is isolated by resolution, so a resolution is excluded when masking would leave it no
visible token. This prevents impossible one-token prediction targets. Very short sessions can
therefore receive no JEPA gradient on a draw, but still participate fully in relation learning.

Short and long resolution losses are reduced independently and averaged. Partial tail patches are
weighted by represented duration, preventing the more numerous short tokens from dominating.

## Data and sampling

The default corpus is the 12 native-rate training datasets in `TRAIN_DATASETS`. The sampler is
hierarchical and label-free:

```text
P(dataset) proportional to n_dataset^0.25, capped at 25%
P(subject | dataset) proportional to n_subject^0.5
P(window | subject) uniform
```

When cross-placement learning is enabled, 20% of batch windows are reserved for verified event
pairs from NFI-FARED, XRF-v2, and SP-SW-HAR. Every reserved window still receives the universal
augmentation relation. WISDM is excluded from cross-placement pairing because its phone and watch
timestamps use unrelated uptime clocks.

The native acquisition rate is passed separately from the stored array rate. This prevents an
upsampled 25 Hz signal from being treated as though it contains genuine 25 Hz spectral content.

## Defaults

| setting | value |
|---|---:|
| encoder | d=256, 6 layers, 8 heads |
| frontend | fixed physical filterbank |
| temporal grids | simultaneous short + long |
| conditioning | factored channel-role + sensor identity |
| batch / steps | 256 / 30,000 |
| LR / warmup / weight decay | 4.2e-4 / 1,000 / 0.05 |
| gradient clip | 1.0 |
| JEPA EMA decay | 0.996 |
| validation | subject-disjoint kNN BA + ConSE probe |

The constrained-learnable frontend and single-resolution path remain active tokenizer ablations:

```bash
python -m training.tokenizer.pretrain --frontend fixed --device cuda \
  --out training/tokenizer/outputs/fixed
python -m training.tokenizer.pretrain --frontend learnable --device cuda \
  --out training/tokenizer/outputs/learnable
python -m training.tokenizer.pretrain --no-multiresolution --device cuda \
  --out training/tokenizer/outputs/single_resolution
```

R0/R1 controls:

```bash
python -m training.tokenizer.pretrain --jepa-weight 0 --cross-placement-weight 0 ...
python -m training.tokenizer.pretrain --jepa-weight 0 ...
```

## Launch

```bash
python -m data.scripts.scan_implausible
python -m data.scripts.scan_duplicates
python -m data.scripts.build_grids --alignment native
python -m training.tokenizer.objective_health
python -m training.tokenizer.grad_check
python -m training.tokenizer.pretrain --device cuda \
  --out training/tokenizer/outputs/<run>
```

The parser defaults to CPU, so `--device cuda` is required for a real run. Checkpoints contain the
encoder, JEPA predictor and teacher, relation projector, optimizer/scheduler/scaler state, complete
configuration, RNG state, corpus fingerprint, and provenance. Old four-objective checkpoints cannot
be resumed under the consolidated objective semantics; inference code can still load their encoder.

## Optional scale data

ExtraSensory, a bounded NHANES subset, and H-MOG are wired but opt-in. NHANES contributes only to
label-free Phase A and is excluded from semantic validation and Phase B. Pass the explicit dataset
roster with `--datasets`; the roster is persisted in `run_config.json`. Dataset publications and
protocol documents are under `references/datasets/`.
