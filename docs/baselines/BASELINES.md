# Baseline roster and comparison design

This is the paper-ready reference for which external models are reported and how they are used.
The governing rule is intentionally simple:

> Report an external model only when its authors released a pretrained checkpoint that we can use
> without reproducing its pretraining.

This removes variation from locally chosen pretraining data, schedules, augmentations, and
checkpoint selection. The detailed treatment contract is in
[`BASELINE_FAIRNESS_POLICY.md`](BASELINE_FAIRNESS_POLICY.md).

## Active roster

| baseline | publication | released checkpoint | use in this study |
|---|---|---|---|
| **HARNet / ssl-wearables** | Yuan et al., npj Digital Medicine 2024 | `harnet5/10/30` | frozen encoder; matched enrollment readouts |
| **UniMTS** | Zhang et al., NeurIPS 2024 | released 274 MB checkpoint | native zero-shot; frozen encoder for enrollment |
| **NormWear** | Luo et al., arXiv 2412.09758 | released 194M checkpoint and frozen language model | native zero-shot; frozen encoder for enrollment |
| **ImageBind** | Girdhar et al., CVPR 2023 | `imagebind_huge` | native IMU-text zero-shot; frozen encoder for enrollment |

No external backbone in the active comparison is pretrained or fine-tuned by us.

CrossHAR and LiMU-BERT are excluded from the paper comparison because their authors did not release
the pretrained checkpoints needed here. Their code and old artifacts remain in the repository for
auditability, but their locally pretrained backbones are not inputs to current tables or figures.

## Input contracts

Each checkpoint receives the strongest input allowed by its released architecture. Differences in
rate or channel count are properties of the published checkpoint, not choices made to weaken a
baseline.

| model | channels | rate and window | reason |
|---|---:|---|---|
| **HARNet** | 3-axis accelerometer | 30 Hz; 5 s crop | released UK Biobank weights are accelerometer-only |
| **UniMTS** | 3-axis accelerometer | model-native resampling and 200-sample input | released checkpoint is accelerometer-only and uses its SMPL placement interface |
| **NormWear** | available accelerometer and gyroscope channels | 65 Hz | released model is channel-independent and accepts variable real channels |
| **ImageBind** | 3-axis accelerometer and 3-axis gyroscope | its released IMU preprocessing contract | released IMU tower accepts both sensors |
| **HALO** | available accelerometer and gyroscope channels | native sampling rate; physical-time windows | variable-rate and variable-channel handling is part of the model |

The adapters preserve each model's published normalization, resampling, masking, and channel order.
Evaluation datasets and support/query episodes are otherwise identical across models.

## What is compared

### Zero-shot recognition

At `k=0`, a model must predict a held-out dataset's meaningful candidate labels without target
support. The headline table contains only models with a native open-vocabulary decision rule:

- HALO: its current language-aware zero-shot path.
- UniMTS: normalized IMU/text similarity through its released text tower.
- NormWear: its released signal/text comparison.
- ImageBind: normalized similarity between its released IMU and text towers.

HARNet has no native open-vocabulary classifier. A ConSE adapter can be fitted for diagnostics, but
that introduces an additional trained component and is therefore omitted from the paper's zero-shot
table.

### Enrollment adaptation

For `k >= 1`, each model uses its released frozen representation and exactly the same enrolled
executions. The common readouts are:

- one-nearest-neighbor;
- normalized class prototypes; and
- closed-form ridge regression.

These readouts require no encoder update. HALO's retrieve-mix-vote mechanism is shown separately as
a mechanism ablation. The current supported HALO deployment rule is 1-NN because the learned
mechanism has not surpassed that matched control.

### Within-HALO attribution

Changes to HALO's tokenizer, conditioning, training objective, or readout are evaluated with the
same data and manifest. These within-model ablations, rather than locally retraining external
models, provide the controlled evidence for attributing gains to a HALO design choice.

## Reporting rules

1. Use the sealed support/query manifest for every model.
2. Keep test queries out of training, tuning, checkpoint selection, and support construction.
3. Average seeds within each dataset, then weight held-out datasets equally.
4. Report macro F1 and per-dataset results.
5. Identify the exact released checkpoint, adapter contract, and HALO checkpoint.
6. Do not present a locally trained bridge as a baseline's native zero-shot mechanism.
7. Do not restore CrossHAR or LiMU-BERT to paper tables unless author-released compatible weights
   become available and pass the same provenance checks.

## Primary sources

- [HARNet / ssl-wearables](https://github.com/OxWearables/ssl-wearables)
- [UniMTS](https://github.com/xiyuanzh/UniMTS)
- [NormWear](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear)
- [ImageBind](https://github.com/facebookresearch/ImageBind)

Local copies of baseline publications are stored with the other baseline literature under the
repository's publication folders.
