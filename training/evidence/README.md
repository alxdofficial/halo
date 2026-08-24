# Phase-B Evidence System

The active Phase-B model is the compact end-to-end evidence engine described in
[`docs/design/COMPACT_EVIDENCE_ENGINE.md`](../../docs/design/COMPACT_EVIDENCE_ENGINE.md). It does
not require a prebuilt memory-bank artifact, a resolvability table, or an admissibility-gate fit.

## End-to-end experiment

The active representation experiment trains the compact encoder and scalar evidence reranker
together. `C` is the complete decision roster, while four labels per episode supply four query
executions each. The other candidates are distractors that compete in the same cross-entropy loss
without requiring additional sensor encodes.

```bash
python -m training.tokenizer.pretrain_episodic \
  --random-init \
  --phase-b-regime unified \
  --steps 35000 \
  --warmup-steps 500 \
  --episodes-per-step 8 \
  --candidate-counts 8 16 32 64 \
  --query-labels-per-episode 4 \
  --queries-per-candidate 4 \
  --bank-windows 512 \
  --top-k 64 \
  --val-every 1000 \
  --val-episodes 32 \
  --num-workers 8 \
  --out training/tokenizer/outputs/<run-name>-e2e
```

Evaluation does not sample this training roster. The primary external protocol uses every eligible
label in each test dataset as its fixed candidate set.

## Specialized Phase-B heads

The clean experiment loads one selected encoder checkpoint and trains two small scalar rerankers
against that same frozen representation. This prevents one head's objective from changing the
features seen by the other.

```bash
# Semantic zero-shot head: k=0 only.
python -m training.tokenizer.pretrain_episodic \
  --checkpoint training/tokenizer/outputs/<phase-a-run>/best.pt \
  --freeze-encoder \
  --phase-b-regime zero-shot \
  --steps 35000 \
  --warmup-steps 500 \
  --episodes-per-step 8 \
  --queries-per-candidate 4 \
  --bank-windows 512 \
  --top-k 64 \
  --val-every 1000 \
  --val-episodes 32 \
  --num-workers 8 \
  --out training/tokenizer/outputs/<run-name>-zero-shot

# Enrollment head: k=1/2/4/8/16 only, from the identical encoder checkpoint.
python -m training.tokenizer.pretrain_episodic \
  --checkpoint training/tokenizer/outputs/<phase-a-run>/best.pt \
  --freeze-encoder \
  --phase-b-regime enrollment \
  --steps 35000 \
  --warmup-steps 500 \
  --episodes-per-step 8 \
  --queries-per-candidate 4 \
  --bank-windows 512 \
  --top-k 64 \
  --val-every 1000 \
  --val-episodes 32 \
  --num-workers 8 \
  --out training/tokenizer/outputs/<run-name>-enrollment
```

Current defaults use clean one-second patches, four independent query executions for each of four
queried labels, eight independent episodes per optimizer step, a
512-window memory, full-bank voting, and a global top-64 shortlist for scalar evidence rescoring.
Candidate rosters are 8/16/32/64. The zero-shot head sees only k=0; the enrollment head sees exact k
values 1/2/4/8/16. Random aliases and signal augmentation are disabled unless explicitly requested.
`--phase-b-regime unified` remains an explicit matched control, not the recommended final pair.

Use `--profile-steps 12` for a bounded real-corpus GPU profile and `--smoke` for a three-step
integration check. Training telemetry is appended to `log.jsonl`; `best.pt` is selected on the
held-out-concept coherent k-curve, and each validation record includes the same-checkpoint retrieval
semantic-vote baseline, the enrolled-1NN reference wherever it is defined, and learned-minus-control
margins.

## Frozen encoder comparison

The matched HARNet and UniMTS controls use the same episodes and evidence engine but keep their
released backbones frozen. Their commands and completed results are in
[`docs/results/ENCODER_COMPARISON_20260822.md`](../../docs/results/ENCODER_COMPARISON_20260822.md).

## Historical pipelines

The following modules reproduce superseded experiments and are not prerequisites for the active
model:

- `build_memory.py`, `resolvability.py`, and `gate_predictor.py`: fitted admissibility-gate study;
- `train_admissibility_gate.py`: optional historical gate refinement;
- `train_patch_decoder.py` and `train_patch_confidence.py`: parked relational decoder and confidence
  experiments.

Their persisted artifacts must not be passed to `pretrain_episodic.py` or described as the current
HALO Phase-B model.
