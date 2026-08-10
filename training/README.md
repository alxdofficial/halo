# training

Two sequential phases. Phase A learns the representation without activity labels; Phase B learns to
use retrieved evidence on top of a frozen Phase-A encoder.

## `tokenizer/` — Phase A (label-free representation pretraining)

Two universal objectives: **JEPA** (a masked student predicts an EMA teacher's clean contextual
tokens) + **VICReg** over two augmentations of every window. Activity labels never enter the loss — only the
validation probes. Full recipe, defaults, and ablation arms: [`tokenizer/README.md`](tokenizer/README.md).

Entry points: `pretrain.py` (train), `pretrain_data.py` (corpus + temperature sampler),
`losses_repr.py` (objectives), `objective_health.py` / `grad_check.py` (pre-launch checks),
`eval_transfer.py` / `probe_robustness.py` / `probe_ceiling.py` (probes).

## `evidence/` — Phase B (memory + evidence prediction)

Builds a patch-level memory bank from a frozen Phase-A checkpoint, retrieves per query patch, and
trains one candidate-set predictor objective. No `UNKNOWN` candidate is introduced. A separate
frozen-predictor confidence experiment exists but is parked and is not part of the current launch.

Entry points: `build_memory.py` (bank), `train_patch_decoder.py` (candidate-CE predictor),
`eval_enrollment.py` (zero/full/partial support, same/cross-configuration evaluation, and matched
controls), and the parked `train_patch_confidence.py` calibration experiment.

## `diagnostics/` — cross-cutting analyses

Baseline heterogeneity and zero-shot difficulty reports; artifacts under `diagnostics/outputs/`.

Phase A is activity-label-free. Phase B may read labels attached to retrieved memory examples and
the runtime candidate vocabulary.
