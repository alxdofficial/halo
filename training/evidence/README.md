# Phase-B Evidence System

The active Phase-B design is closed-form sensor-patch retrieval with a small admissibility gate. It
does not train the parked relational decoder. The motivation and behavioral contract are in
[`docs/design/PHASE_B_TRAINING_INTENT.md`](../../docs/design/PHASE_B_TRAINING_INTENT.md).

## Prerequisite

Use the completed Phase-A checkpoint with `token_granularity='sensor'`. The current selected source is
`training/tokenizer/outputs/phase_a_fixed_1s_rotation_20260817/best.pt` at step 27,000. The older
`phase_a_headline/best.pt` checkpoint is channel-granular and cannot produce a valid Phase-B bank.

## Build Artifacts

Set the Phase-A checkpoint once:

```bash
export HALO_CKPT=training/tokenizer/outputs/phase_a_fixed_1s_rotation_20260817/best.pt
```

Build the schema-5 memory bank:

```bash
python -m training.evidence.build_memory \
  --checkpoint "$HALO_CKPT" \
  --sensor-rows \
  --device cuda \
  --out training/evidence/outputs/memory_bank.pt
```

Measure train-only, per-sensor resolvability:

```bash
python -m training.evidence.resolvability \
  --build \
  --checkpoint "$HALO_CKPT" \
  --device cuda \
  --out training/evidence/outputs/resolvability.json
```

Fit the gate and bind it to the exact bank:

```bash
python -m training.evidence.gate_predictor \
  --fit \
  --rank 8 \
  --bank training/evidence/outputs/memory_bank.pt \
  --out training/evidence/outputs/admissibility_gate.pt
```

Rank 8 is the current default. The held-out study below still reports ranks 1, 2, 4, and 8 so the
capacity choice remains an explicit ablation rather than an unexamined constant.

The fit command refuses a resolvability table containing Phase-B development or test datasets. The
evaluation command refuses legacy stream-level tables, schema-3 banks, unbound gates, mismatched
checkpoints, changed embedding paths, and malformed sensor foreign keys.

The files currently present at the default output paths are historical and are expected to fail
these guards. Rebuild all three in the order above. A successful current build replaces the schema-3
93-label bank with a schema-5 bank under the current 166-label vocabulary.

## Gate Generalization

Run every held-out study into one artifact before external evaluation:

```bash
python -m training.evidence.gate_extrapolation \
  --split all \
  --ranks 1 2 4 8 \
  --out training/evidence/outputs/gate_extrapolation.json
```

The table used to fit the gate is not an independent quality measure. Interpret held-out skill
against the per-concept baseline, not only against a global constant. Separate invocations without
distinct `--out` paths overwrite the same JSON, so do not run the four splits sequentially into the
default path.

## Enrollment Evaluation

Run the development roster first:

```bash
python -m training.evidence.eval_enrollment \
  --checkpoint "$HALO_CKPT" \
  --bank training/evidence/outputs/memory_bank.pt \
  --predictor training/evidence/outputs/admissibility_gate.pt \
  --device cuda
```

Evaluate arbitrary names separately:

```bash
python -m training.evidence.eval_enrollment \
  --checkpoint "$HALO_CKPT" \
  --bank training/evidence/outputs/memory_bank.pt \
  --predictor training/evidence/outputs/admissibility_gate.pt \
  --device cuda \
  --random-aliases
```

Use `--protocol-role test` only after development choices are fixed. Evaluation reports k = 0, 1,
2, 4, and 8; full and partial enrollment; support removal; cyclic support-label shuffling; the same
retrieval rule with admissibility disabled; prototypes; ridge heads; and genuine subject and
configuration relations where the dataset supports them.

### Frozen HARNet representation control

To test whether enrollment performance is limited by HALO's representation rather than the evidence
engine, score the official frozen HARNet-5 trunk on the same nested execution support plans:

```bash
python -m training.evidence.eval_harnet_enrollment \
  --device cuda \
  --protocol-role dev \
  --support 1 2 4 8
```

This control fits no HARNet classifier and does not use HALO's evidence engine. It reports nearest
support, normalized prototype, and deterministic ridge curves from identical support/query windows.
The legacy versus corpus-matched HARNet distinction does not apply here because that distinction
changes only the fitted ConSE head; the frozen released trunk is identical.

`--gate-top-k` controls the number of returned rows per query patch and candidate; the default is 64.
The evaluator currently constructs the searched population from 16 source windows per corpus label.
These are different quantities, and both are written to the result JSON. `sensor_bias` similarity
remains disabled in the active predictor.

## Stage 2: Optional Gate Refinement

Run Stage 2 only after the warm-start gate shows useful held-out extrapolation and external Stage-1
performance. It freezes Phase A, memory features, cosine ranking, and voting, and updates only the
small admissibility gate:

```bash
python -m training.evidence.train_admissibility_gate \
  --bank training/evidence/outputs/memory_bank.pt \
  --gate training/evidence/outputs/admissibility_gate.pt \
  --out training/evidence/outputs/admissibility_stage2 \
  --device cuda
```

Each training episode varies the candidate count, partial enrollment, support count, and coherent
label paraphrases. Query and support windows come from distinct recorded executions. Candidate
labels are removed from ordinary corpus memory; only the episode's selected supports are restored
with explicit candidate bindings. Training uses a fully soft distribution over every physically
compatible candidate-row choice in a bounded, label-balanced working memory, so rows outside
deployment top-k still receive gradient. Validation applies top-k to the same continuous adjusted
score used by training.

The implementation keeps the immutable source rows on the GPU, embeds each sensor description once,
and evaluates all query windows in an episode together. These are exact execution optimizations: the
episode distribution, FP32 score, admissibility equation, and loss are unchanged. Cheap telemetry is
written every 20 steps; the default full validation cadence is every 500 steps.

The loss is candidate cross-entropy plus a small replay penalty on the train-only resolvability
measurements. The replay term is an anchor on the meaning of admissibility, not a second prediction
task. Telemetry reports gradient norms, gate spread, parameter drift, retrieval entropy, effective
row count, and top-k validation against the same predictor with admissibility disabled.
Use external Phase-B development datasets, not internal training loss or fitted table cells, to
select whether the refined artifact replaces Stage 1.

```bash
python -m training.evidence.eval_enrollment \
  --checkpoint "$HALO_CKPT" \
  --bank training/evidence/outputs/memory_bank.pt \
  --predictor training/evidence/outputs/admissibility_stage2/best.pt \
  --protocol-role dev \
  --device cuda \
  --out training/evidence/outputs/admissibility_stage2/eval_dev.json
```

Do not use `train_patch_decoder.py` for gate refinement; that file reproduces the parked relational-
decoder experiments.

Default enrollment outputs include the predictor mode in the filename, for example
`eval_enrollment_dev_admissibility_gate.json`. This prevents a parked relational run from silently
overwriting a current-design result.

## Archived Relational Experiments

`train_patch_decoder.py`, the learned subspace retriever, the relational decoder, their telemetry,
and their old checkpoints remain for reproducing earlier results. They are not the default Phase-B
model and are not prerequisites for admissibility evaluation. Confidence calibration is also parked.
