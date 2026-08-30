# Augmentation policy for movement monitoring

> **Application policy, 2026-08-30.** The first Task-1, Task-2, and Task-3 experiments use clean
> recorded signals. Augmentations are enabled one family at a time only for the learned metric
> arm and never manufacture the only positive examples.

## Why the default is clean

The application needs to measure real execution difference. Aggressive invariance can erase the
same orientation, intensity, timing, or frequency changes Task 2 is supposed to expose. Motif
discovery can also cluster augmentation artifacts instead of recurrent human movement.

Raw-signal, physical-feature, and frozen-representation floors therefore receive no stochastic
augmentation.

## Available transformation families

The implementation in `data/scripts/augmentations.py` contains transformations for rotation,
jitter/noise, amplitude scale, rate/time resampling, gravity removal, channel dropout, and text
description perturbation. Their existence does not make them part of the application recipe.

## Allowed use in metric learning

| transformation | initial role | constraint |
|---|---|---|
| small shared 3D rotation | remounting robustness | rotate co-located accelerometer and gyroscope with the same SO(3) transform; retain orientation-sensitive controls |
| mild time scaling | speed robustness | preserve physical timing metadata and report real duration separately |
| mild sensor noise | device robustness stress | calibrate against measured device noise; never use arbitrary jitter as the default |
| channel dropout | missing-modality robustness | update signal, masks, and descriptions consistently; never write into padded channels |
| gravity removal | declared configuration stress only | do not treat gravity-present and gravity-removed views as identical without telling the model |

Text paraphrasing and arbitrary label aliases are not relevant to Tasks 1-3 because movement names do
not enter the core matcher.

## Positive-pair rule

The preferred positive is the same movement performed independently in another execution or
session. An augmented view of one excerpt is a robustness pair, not evidence that the representation
captures real human repetition variability. Batches and reports distinguish these sources.

## Required ablation

Any enabled family is compared against the clean metric model using identical real positive pairs.
Report Task-1 false alarms, Task-2 sensitivity to known execution changes, and remounting robustness.
An augmentation is rejected if its invariance gain removes meaningful execution-change signal.
