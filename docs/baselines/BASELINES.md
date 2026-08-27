# Representation and signal baselines

The application study compares movement representations, not native HAR classifiers. Every method
feeds the same Task-1 matcher, Task-2 comparison pipeline, and Task-3 motif evaluator where its
temporal output permits that use.

## Required non-learned controls

| control | purpose |
|---|---|
| raw IMU subsequence DTW | establishes whether a pretrained representation beats direct waveform matching |
| magnitude/gravity-aligned DTW | tests simple orientation robustness without a learned encoder |
| physical-feature sequence | compares duration, intensity, frequency, smoothness, and stability directly |
| raw or feature matrix profile | established floor for recurrent subsequence discovery |

A pretrained encoder is not useful merely because it beats another encoder. It must improve on these
simple, interpretable application methods or provide materially better robustness or efficiency.

## Released checkpoint roster

| encoder | released representation | application use |
|---|---|---|
| **HARNet / ssl-wearables** | accelerometer encoder pretrained on UK Biobank | frozen temporal/sliding-window representation |
| **UniMTS** | released universal time-series checkpoint | frozen IMU representation through its official preprocessing |
| **NormWear** | released channel-independent wearable checkpoint | frozen variable-channel representation |
| **ImageBind** | released IMU tower | frozen IMU representation where temporal granularity is adequate |
| **HALO** | project checkpoints | frozen physical-time patch representation and controlled ablations |

CrossHAR and LiMU-BERT remain excluded because the repository's usable backbones were pretrained by
us rather than released by their authors. That would reintroduce choices about corpus, schedule,
augmentation, and checkpoint selection.

## What is shared

- source recordings and real event boundaries;
- subject/session splits and application manifests;
- reference counts and target-absent intervals;
- matching, alignment, event-merging, threshold-selection, and motif-scoring code;
- task metrics and subject-level uncertainty; and
- physical-time windows and strides whenever an encoder lacks a native temporal sequence.

Each adapter preserves its checkpoint's documented units, rate, channels, normalization, and shape.
An encoder is never modified to accept information absent from its released architecture.

## What the comparison means

The primary comparison asks which available frozen representation is the best component for the
application. Different upstream datasets and model sizes are therefore documented rather than
artificially matched. A result does **not** prove that one architecture is intrinsically superior.

HALO architecture claims require within-HALO controls using the same data and optimization, such as
fixed filterbank versus continuous kernels, acquisition conditioning versus neutral metadata, or
pooled versus temporal patch output.

## Primary sources

- [HARNet / ssl-wearables](https://github.com/OxWearables/ssl-wearables)
- [UniMTS](https://github.com/xiyuanzh/UniMTS)
- [NormWear](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear)
- [ImageBind](https://github.com/facebookresearch/ImageBind)

The detailed treatment contract is in
[`BASELINE_FAIRNESS_POLICY.md`](BASELINE_FAIRNESS_POLICY.md).
