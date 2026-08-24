# Phase-A tokenizer pretraining

Phase A trains the sensor representation consumed by the Phase-B evidence engine. The training
loss is activity-label-free; labels are read only by validation probes.

## Objectives

The live recipe has exactly two top-level objectives:

| objective | supervision | default weight |
|---|---|---:|
| `jepa` | masked student predicts clean contextual sensor tokens from an EMA teacher | 1.0 initially |
| `vicreg` | VICReg collapse control on pooled embeddings and the sensor rows used by Phase B | 1.0 initially |

VICReg's published MSE invariance, variance, and covariance terms provide collapse prevention and
redundancy reduction for every sampled window. Half of its default value acts directly on the
sensor-isolated retrieval rows; the other half acts on the pooled contextual representation through
the projector. The clean reference uses identical physical views. Rotation, rate, and modality
dropout are explicit one-at-a-time ablations. Phase A no longer contains
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
  --calibrate-objectives-at 500 \
  --objective-calibration-mode apply \
  --out training/tokenizer/outputs/phase_a
```

`--calibrate-objectives-at 500 --objective-calibration-mode apply` is the real-run default. It is
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
  --calibrate-objectives-at 500 \
  --objective-calibration-mode report \
  --out training/tokenizer/outputs/objective_calibration
```

Report mode stops immediately after writing the recommendation and emits no validation checkpoint.
Keep `--steps` equal to the planned full run (normally 7,500), even though report mode exits at the
calibration step: `steps` defines the cosine LR and EMA schedules, so shortening it to 500 would
measure a different, already-decayed training trajectory.

Ordinary runs continue to log weighted top-level encoder-gradient norms and JEPA/VICReg cosine at
step 1 and every 500 steps. This is a deterministic one-time schedule, not a continuously adaptive
optimizer. Nearby 0.5x/1x/2x JEPA-to-VICReg ratio arms should be selected using the fixed transfer
and robustness probes rather than gradient equality alone.

For a bounded stability monitor, `--steps 7500 --stop-after 1000` checkpoints at step 1,000 while
retaining the 7,500-step LR and EMA schedules. Resume that checkpoint without `--stop-after` to
continue the same trajectory.

## JEPA masking

The canonical encoder folds each accelerometer or gyroscope xyz triad into one sensor token. JEPA
uses two active signal masking strategies inside one objective:

1. A contiguous temporal block hides sensor tokens over part of the observation.
2. A whole sensor may be hidden only when another sensor at the same placement remains visible.
Descriptor reconstruction is disabled in the reference recipe (`descriptor_weight=0`). Its code is
retained only as an explicit ablation because it adds a distinct retrieval target and previously made
the supposedly simple JEPA term harder to interpret. Padded patches and absent sensors are never
targets. There is no causal/future-tail branch.

The reference collate uses fixed one-second patches. A recording's final partial six-second context and
its final partial one-second patch are retained with honest lengths. Session pooling weights patches by
the physical duration they represent, so a short tail cannot count as a complete second.

## Data and sampling

The design-of-record default is the **expanded 18-source native-rate corpus**: 56 streams,
1,744,926 admitted training windows, 217,554 subject-disjoint validation windows, and 166 canonical
activity names at data seed 20260718. The original 12-source recipe remains available as
`--corpus matched`; use it for a technique-only comparison against baselines fitted on that corpus.
The offline `sensor_bias.json` artifact is bank metadata only. It does not enter Phase-A training and
therefore does not need to be rebuilt when the Phase-A corpus changes.
The sampler is hierarchical and label-free:

```text
P(dataset) proportional to n_dataset^0.25, capped at 25%
P(subject | dataset) proportional to n_subject^0.5
P(window | subject) uniform
```

No source-specific quota overrides this distribution. Synchronous streams are sampled as ordinary
windows because VICReg uses only two augmentations of the same row and has no negative-pair mining.

At 7,500 steps and batch 1,024, sampling draws the same 7.68 million windows as the measured
30,000-step batch-256 reference. It is with replacement and is not equivalent to equal corpus
epochs. Capture-24 supplies 15.3% of expected draws despite containing most admitted rows;
expected draws per available row range from 0.71 to 550 after subject tempering (median 0.87). The six
direct-converted additions collectively supply 32.7% of draws. This is the intended
dataset/subject-temperature policy, and raw-corpus epoch counts must not be used to describe source
exposure.

The native acquisition rate is passed separately from the stored array rate. This prevents an
upsampled 25 Hz signal from being treated as though it contains genuine 25 Hz spectral content.

## Defaults

| setting | value |
|---|---:|
| encoder | d=256, **3 layers**, 8 heads (head dim 32) -- `--num-layers`; see the depth note below |
| VICReg expander | 256 -> 256 -> 128 |
| JEPA temporal mask ratio | 0.5 of physically ordered tokens, up to rounding |
| frontend | fixed physical filterbank |
| temporal grid | one-second patches in non-overlapping contexts of at most six seconds |
| token unit | one token per modality triad (accelerometer or gyroscope) |
| conditioning | frozen sensor text descriptor through a gated learnable projection; measured sensor statistics are bank-only metadata |
| batch / steps | 1,024 / 7,500 (7.68M sampled windows) |
| LR / warmup / weight decay | 6e-4 / 250 / 0.10 |
| gradient clip | 1.0 |
| CUDA precision | FP16 autocast + dynamic loss scaling; FP32 master weights/statistics |
| JEPA EMA decay | 0.984095744256 (`0.996^4`, preserving half-life in examples) |
| validation | internal subject-disjoint probes; `best.pt` selected every 500-2,000 steps on dataset-macro kNN over MotionSense, RealWorld and Shoaib. ExtraSensory wrist/hand are encoded and REPORTED but excluded from the scalar (measured anti-correlated with training). Posture canaries reported per source. |
| objective calibration | 50 batches ending at step 500, apply once |
| RTX 4090 loader | 12 workers (override with `--num-workers`) |

`--lr`, `--warmup-steps`, `--weight-decay`, and `--grad-clip` expose the optimizer controls for
attributable pilot sweeps. The batch-1024 defaults use conservative square-root LR scaling; weight
decay and EMA momentum are adjusted so their cumulative effect remains approximately constant per
sample relative to the batch-256 reference.
Ten completed 7,500/15,000-step batch-1,024 arms now exist (2026-08-18); the recipe is no longer
merely smoke-tested. Their results, and the reference points every future arm is measured against,
are in [`docs/results/PHASE_A_RECOVERY_20260818.md`](../../docs/results/PHASE_A_RECOVERY_20260818.md).

**Trunk depth: 3, and this is the only setting the 2026-08-18 sweep established.** Main effect of
depth 3 + jitter + scale over the 6-layer no-augmentation control was **+0.0157** on held-out
development transfer, above the 0.012 screening noise floor. It also removed the decay every
6-layer arm showed: those peaked by step 1,000-4,000 and lost 3-6 points, while the depth-3 arms
held their value for the full 7,500 steps. Probing every depth shows why -- activity information
peaks at depth 1-3 and loses 4-8 points by depth 6 in every trained arm, while a random-init trunk
is flat.

**Calibrated reference points** (same metric, same seed):

| | development transfer |
|---|---:|
| random-init encoder, same architecture | 0.8012 |
| best 2026-08-18 arm (`h_mae_fixes`) | 0.8468 |
| old-good 4k checkpoint | **0.8577** -- still unbeaten |
| rejected `phase_a_fixed_1s_rotation_20260817` | 0.7161 (below random init) |

Phase-A SSL is therefore worth about +0.057 over a random trunk; the fixed physical filterbank
supplies the rest. Treat "beats the random-init floor" as necessary, not sufficient.
`best.pt` is selected by a frozen readout of the actual sensor rows Phase B stores, evaluated on the
three development datasets above. The sealed test roster is never used for checkpoint selection.

The constrained-learnable frontend and multi-resolution path remain active tokenizer ablations:

```bash
python -m training.tokenizer.pretrain --frontend fixed --device cuda \
  --out training/tokenizer/outputs/fixed
python -m training.tokenizer.pretrain --frontend learnable --device cuda \
  --out training/tokenizer/outputs/learnable
python -m training.tokenizer.pretrain --multiresolution --device cuda \
  --out training/tokenizer/outputs/multiresolution
```

Sizing arms opened against published precedent, all **off by default** and unmeasured on our data
(three literature transfers already failed at our scale, so each is a knob, not an adoption):

```bash
--num-heads 4              # head dim 32 -> 64, the usual transformer range, same parameter count
--mask-ratio-time 0.75     # BEiT 40 / MAE 75 / data2vec-2.0 ~80 / I-JEPA 0.7-1.0 vs our 0.5
--jepa-ema-schedule cosine # BYOL-style ramp from the batch-adjusted base momentum to 1.0
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
python -m data.scripts.curate.sensor_bias --build
python -m training.tokenizer.objective_health
python -m training.tokenizer.grad_check
python -m training.tokenizer.pretrain --device cuda \
  --corpus expanded \
  --num-workers 12 \
  --calibrate-objectives-at 500 --objective-calibration-mode apply \
  --out training/tokenizer/outputs/<run>
```

Use `--corpus matched` for the frozen 12-source comparison arm. Every checkpoint serializes its
resolved dataset tuple, so a future change to the module default cannot silently change a memory-bank
rebuild. `--datasets ...` remains the explicit custom/scale-arm escape hatch.

CUDA launches enable transformer-only dynamic compilation by default; pass `--no-compile` only for
debugging. Profiling on the local RTX 4090 measured about 11.6 updates/s and 11,900 windows/s at batch
1,024, with 4.75 GiB peak allocated VRAM. The 7,500-step sample-matched run therefore projects to
roughly 12–13 minutes including cold compilation, recurring probes, and checkpoint I/O. The old
whole-encoder compile path was slower because ragged unique-text shapes caused specialization churn;
`--compile` leaves text conditioning eager and compiles only the dual-branch transformer. FP16 and
BF16 had indistinguishable measured throughput, so FP16 remains the design-of-record precision.

Multi-resolution remains a batch-512 ablation: its 12–22 patches per window use 7.67 GiB at batch 512,
and batch 1,024 would force the collate to discard finer resolution pairs. `--multiresolution`
therefore selects a sample-matched 512 / 15,000 schedule with the corresponding LR, warmup, decay,
EMA, and calibration defaults. An explicit multi-resolution batch above 512 is rejected.

This is mixed precision rather than a blanket `model.half()` conversion. Linear projections and
attention matrix products use FP16 Tensor Cores. Model parameters, AdamW state, the filterbank FFT and
spectral-energy reduction, JEPA cosine normalization, VICReg mean/variance/covariance, gradient norms,
clipping, and validation kNN/ridge calculations remain FP32. Dynamic `GradScaler` starts at 2^14,
unscales before clipping, and skips the EMA/scheduler update whenever an optimizer step overflows.
The 500-step production-path check had zero skipped updates and held scale 2^14 throughout; live
telemetry records the scale, dtype, and skipped-update counts.
Validation deliberately bypasses the training compile hook: both 161-label support/query subsets take
about 1.8 seconds eagerly, whereas compiling their one-off shapes added roughly 35 seconds of cold
startup. Their neural path still uses FP16 autocast, and kNN/ridge scoring remains FP32.

Quality scans are fingerprinted to the current grids and stored separately by alignment. Phase-A
training fails closed when either native cache is missing or stale; model-agnostic evaluation uses
the independently generated `--alignment non_harmonised` caches. The sensor-bias artifact is built
from training subjects only and records
its dataset roster, data-split seed, and validation-subject hash. Sensor-mode training fails closed
if any of these differ or if a requested training sensor is absent. Grid construction and quality
scans must therefore precede the sensor-bias build.

## Live monitoring

The trainer appends structured scalar records every 50 updates. A separate CPU-only monitor can
refresh a human summary, a machine-readable health snapshot, and the dashboard once per minute
without importing Torch or touching the training GPU:

```bash
python -m training.tokenizer.monitor_training \
  --run-dir training/tokenizer/outputs/<run> --render --watch 60
```

This atomically writes `health.txt`, `health.json`, and `telemetry.png`. A periodically waking agent
should run the same command once without `--watch`, then read `health.json`; its `status` is
`green`, `warning`, or `critical`, and every alert has a stable code and explanation. The monitor
checks heartbeat age, finite inputs/losses/gradients, JEPA target coverage, AMP update skips,
representation collapse signals, objective-gradient balance, severe clipping, throughput changes,
realized versus target dataset shares, validation regression, and checkpoint presence. Alerts are
deliberately compound or sustained where a single noisy batch would otherwise cause false alarms.

The dashboard shows weighted objective losses, VICReg components, positive-pair margins, module and
per-objective encoder gradients, encoder/projector/EMA-teacher health, the actual label/stream
checkpoint-selection metric, source and augmentation realization, throughput, VRAM, ETA, AMP skips,
and input validity. Descriptor retrieval telemetry appears only when that ablation is enabled. Validation remains
every 1,000 steps; the minute monitor does not run an extra model probe and therefore adds no training
compute.

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
roster with `--datasets`; the roster is persisted in `run_config.json`. Before such a run, rebuild
`sensor_bias.json` with the identical ordered roster and `--data-seed`. Dataset publications and
protocol documents are under `references/datasets/`.

## Phase B: compact end-to-end episodic training

`pretrain_episodic.py` trains the current recording-level evidence reranker through its deployment
rule. Eight independent episodes share one encoder forward but retain separate fixed-size memory
banks, candidate sets, queries, and support. The temporal sensor encoder produces one vector per
six-second recording. Raw cosine retrieves 64 rows. Candidate, query, and retrieved-evidence tokens
share one unordered attention layer, whose only output is one bounded scalar correction per evidence
row before corrected nearest-neighbor classification. It cannot refine vectors, emit candidate
logits, or vote. Inputs are clean by default; `--augment` is an explicit ablation. The single design
of record is `docs/design/COMPACT_EVIDENCE_ENGINE.md`.
