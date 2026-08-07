# Phase-A tokenizer pretraining

Phase A trains the sensor representation consumed by the Phase-B evidence engine. The training
loss is activity-label-free; labels are read only by validation probes.

## Objectives

The live recipe has exactly two objectives:

| objective | supervision | default weight |
|---|---|---:|
| `jepa` | masked student predicts clean contextual tokens from an EMA teacher | 1.0 initially |
| `vicreg` | VICReg over two independent augmentations of every window | 1.0 initially |

VICReg's published MSE invariance, variance, and covariance terms provide augmentation robustness,
collapse prevention, and redundancy reduction for every sampled window. Phase A no longer contains
a sparse synchronous cross-placement objective or a paired-source sampling quota. Cross-placement
transfer remains an evaluation axis and Phase B can still use event metadata from the grids.

The supported objective controls are:

```text
default: jepa_weight>0, vicreg_weight>0  (JEPA + augmentation VICReg)
control: jepa_weight=0, vicreg_weight>0  (VICReg only)
```

Cadence/eigen primitives remain diagnostic probes in `probe_robustness.py`; they are not training
targets. Physical-feature reconstruction, SimCLR, SupCon, TF-C, separate A4, mask compensation,
label-balanced sampling, and continuously adaptive objective weighting have been removed.

## Objective-gradient calibration

Equal scalar loss weights do not imply equal influence on the shared encoder: VICReg includes its
published 25/25/1 internal coefficients while JEPA is a bounded cosine loss. The default CLI recipe
trains through warmup with the initial weights, measures 50 post-warmup batches,
applies one solved scalarization, freezes it, and continues:

```bash
python -m training.tokenizer.pretrain --device cuda \
  --calibrate-objectives-at 2000 \
  --objective-calibration-mode apply \
  --out training/tokenizer/outputs/phase_a
```

`--calibrate-objectives-at 2000 --objective-calibration-mode apply` is the real-run default. It is
written explicitly above so launch records remain self-explanatory. Smoke tests disable calibration
unless requested, and `--jepa-weight 0` disables it for the VICReg-only control. A deliberately
uncalibrated JEPA+VICReg run must pass `--calibrate-objectives-at 0`.

Calibration sets JEPA to a 45% gradient-norm share against VICReg. A common scale preserves the
pre-calibration median combined
encoder-gradient norm, avoiding an accidental LR/clipping change. The solved weights take effect on
the step *after* calibration and never change again. This avoids applying a large post-warmup JEPA
coefficient while its initialization-time gradient is still transiently large.

The run writes `objective_calibration.json`, logs an `objective_calibration_applied` event, and
rewrites `run_config.json` with the frozen resolved coefficients. `--resume` always hydrates those
coefficients as immutable trajectory state; changing them starts a new run. The report
includes all per-batch norms, pairwise gradient cosines, sampled-source shares, configuration,
corpus fingerprint, and provenance.

For a report-only throw-away pilot, omit application explicitly:

```bash
python -m training.tokenizer.pretrain --device cuda \
  --calibrate-objectives-at 2000 \
  --objective-calibration-mode report \
  --out training/tokenizer/outputs/objective_calibration
```

Report mode stops immediately after writing the recommendation and emits no validation checkpoint.
Keep `--steps` equal to the planned full run (normally 30,000), even though report mode exits at the
calibration step: `steps` defines the cosine LR and EMA schedules, so shortening it to 2,000 would
measure a different, already-decayed training trajectory.

Ordinary runs continue to log weighted top-level encoder-gradient norms and JEPA/VICReg cosine at
step 1 and every 500 steps. This is a deterministic one-time schedule, not a continuously adaptive
optimizer. Nearby 0.5x/1x/2x JEPA-to-VICReg ratio arms should be selected using the fixed transfer
and robustness probes rather than gradient equality alone.

For a bounded stability monitor, `--steps 30000 --stop-after 4000` checkpoints at step 4,000 while
retaining the 30,000-step LR and EMA schedules. Resume that checkpoint without `--stop-after` to
continue the same trajectory.

## JEPA masking

Default `--jepa-mask-mode per_resolution`: **one contiguous block per temporal grid, drawn
independently**, and the masked student attends **across** resolutions. Inferring a hidden
fine-grained token from a visible coarse summary of the same seconds (or the reverse) is treated as
legitimate inference, not leakage. Blocks stay contiguous in each grid's own physical-time order —
scattered i.i.d. masking degenerates for quasi-stationary motion, where an adjacent 0.5 s token is
nearly a copy.

Three properties follow, all measured on 24 live batches:

| | `shared_interval` (control) | `per_resolution` (default) |
|---|---:|---:|
| realised mask fraction (nominal 0.5) | 0.558 short / 0.674 long | **0.50 / 0.50** (median) |
| windows with temporal masking in BOTH grids — long-window sources | 94–98% | **100%** |
| … `unimib_shar` (3.0 s) | 44.1% | **100%** |
| … `uci_har` (2.56 s) | 24.8% | **100%** |
| … `sp_sw_har` (1.0 s) | **0.0%** | **86.5%** |
| masked long tokens with every contained short token visible | 25.7% (inert — attention isolated) | **16.0%** (live, by design) |

The third row is the reason for the change. Under isolation a grid holding a single token could
never be masked — hiding it left that grid no visible context — so a guard removed the mask and the
window silently fell back to a cross-channel task. That made short-window sources (~19% of draws)
train on a *different objective* than the rest of the corpus while `zero_supervision_fraction`
still read 0.0. With cross-resolution attention the other grid supplies the context, so every
source trains on the same objective. `sp_sw_har` stops at 86.5% rather than 100% because a window
that reduces to one token per grid cannot mask both and keep anything visible.

Dropping isolation also removes a train/inference mismatch: JEPA previously never trained the
cross-resolution attention that evaluation uses.

The cost, accepted deliberately: at equal `mask_ratio_time` this is the **easier** task — it masks
0.50 of each grid where the interval scheme realised 0.558/0.674, and it hands the student the
other grid. JEPA is already the fast-converging half of the recipe. **~0.6 holds realised
difficulty roughly level; 0.75 is where MAE's ablation peaks.** The default stays 0.5 so the mode
change is a single variable.

`--jepa-mask-mode shared_interval` restores the historical arm (one physical interval masked in
both grids, resolution-isolated attention) that every completed 30k run used. It requires
multiresolution.

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

No source-specific quota overrides this distribution. Synchronous streams are sampled as ordinary
windows because VICReg uses only two augmentations of the same row and has no negative-pair mining.

At 30,000 steps and batch 256, sampling is with replacement and is not equivalent to equal corpus
epochs. The median expected draws per available training row range from roughly 1.3 for Capture-24
to 297 for mHealth; Capture-24 still supplies about 7.7 million opportunities while smaller sources
are deliberately revisited. This is the intended dataset/subject-temperature policy, and raw-corpus
epoch counts must not be used to describe source exposure.

The native acquisition rate is passed separately from the stored array rate. This prevents an
upsampled 25 Hz signal from being treated as though it contains genuine 25 Hz spectral content.

## Defaults

| setting | value |
|---|---:|
| encoder | d=256, 6 layers, 8 heads (head dim 32) |
| VICReg expander | 256 -> 256 -> 128 |
| JEPA temporal mask ratio | 0.5 nominal (~0.6 realised) |
| frontend | fixed physical filterbank |
| temporal grids | simultaneous short + long |
| conditioning | factored channel-role + sensor identity |
| batch / steps | 256 / 30,000 |
| LR / warmup / weight decay | 3e-4 / 1,000 / 0.05 |
| gradient clip | 1.0 |
| JEPA EMA decay | 0.996 |
| validation | subject-disjoint, label/stream-covered kNN + ConSE probes |
| objective calibration | 50 batches ending at step 2,000, apply once |
| RTX 4090 loader | 12 workers (override with `--num-workers`) |

`--lr`, `--warmup-steps`, `--weight-decay`, and `--grad-clip` expose the optimizer controls for
attributable pilot sweeps; `--lr 4.2e-4` retains the former batch-512-scaled comparison arm.
`best.pt` is selected by kNN recall
macro-averaged over observed `(label, stream)` cells rather than a source-dominated global average.

The constrained-learnable frontend and single-resolution path remain active tokenizer ablations:

```bash
python -m training.tokenizer.pretrain --frontend fixed --device cuda \
  --out training/tokenizer/outputs/fixed
python -m training.tokenizer.pretrain --frontend learnable --device cuda \
  --out training/tokenizer/outputs/learnable
python -m training.tokenizer.pretrain --no-multiresolution --device cuda \
  --out training/tokenizer/outputs/single_resolution
```

Sizing arms opened against published precedent, all **off by default** and unmeasured on our data
(three literature transfers already failed at our scale, so each is a knob, not an adoption):

```bash
--num-heads 4              # head dim 32 -> 64, the usual transformer range, same parameter count
--mask-ratio-time 0.75     # BEiT 40 / MAE 75 / data2vec-2.0 ~80 / I-JEPA 0.7-1.0 vs our 0.5
--jepa-ema-schedule cosine # BYOL ramp 0.996 -> 1.0 instead of 0.996 held for all 30k steps
--vicreg-proj-hidden / --vicreg-proj-dim   # VICReg wants the expander WIDER than d_model
```

Objective control:

```bash
python -m training.tokenizer.pretrain --jepa-weight 0 ...  # augmentation VICReg only
```

## Launch

```bash
python -m data.scripts.build_grids --alignment native
python -m data.scripts.scan_implausible
python -m data.scripts.scan_duplicates
python -m training.tokenizer.objective_health
python -m training.tokenizer.grad_check
python -m training.tokenizer.pretrain --device cuda \
  --compile --num-workers 12 \
  --calibrate-objectives-at 2000 --objective-calibration-mode apply \
  --out training/tokenizer/outputs/<run>
```

On the local RTX 4090, transformer-only dynamic compilation reduces the steady production loop from
about 180 ms to 153 ms per update end-to-end. A cold process pays roughly 35–73 seconds once, plus a
small one-time validation graph compile; the 30,000-step run amortizes this comfortably. The old
whole-encoder compile path was slower because ragged unique-text shapes caused specialization churn;
`--compile` now leaves text conditioning eager and compiles only the dual-branch transformer. Batch
256 is throughput-optimal in the measured range: batch 320 gained only 2.7% windows/s and batch 384
regressed, while changing optimization statistics. FP16 was marginally faster than BF16. Do not use
the old advice to increase batch merely because the 24 GB card has free VRAM.

Quality scans are fingerprinted to the current native grids and training fails closed when either
cache is missing or stale. Grid construction must therefore precede both scans.

The parser defaults to CPU, so `--device cuda` is required for a real run. Checkpoints contain the
encoder, JEPA predictor and teacher, VICReg projector, optimizer/scheduler/scaler state, complete
configuration, CPU/CUDA RNG state, corpus fingerprint, source provenance, and the Python/Torch/
NumPy/SciPy/CUDA/cuDNN/GPU runtime identity. Applied-calibration weights
are hydrated from the checkpoint on resume. Every run also writes `source.patch` plus
`source_provenance.json`; checkpoints carry its SHA-256 so a resume under different source is
rejected. Resume with the original launch settings and `--resume <checkpoint>`; calibration flags do
not need to be repeated. `--force` clears all known run metadata as well as logs/checkpoints. Old four-objective checkpoints cannot
be resumed under the consolidated objective semantics; inference code can still load their encoder.

## Optional scale data

ExtraSensory, a bounded NHANES subset, and H-MOG are wired but opt-in. NHANES contributes only to
label-free Phase A and is excluded from semantic validation and Phase B. Pass the explicit dataset
roster with `--datasets`; the roster is persisted in `run_config.json`. Dataset publications and
protocol documents are under `references/datasets/`.
