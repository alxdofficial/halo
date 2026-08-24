# Phase-B evidence system

The active Phase-B implementation is the recording-level contextual scalar reranker described in
[`docs/design/COMPACT_EVIDENCE_ENGINE.md`](../../docs/design/COMPACT_EVIDENCE_ENGINE.md).

It requires no prebuilt memory artifact, resolvability table, or admissibility-gate fit. The trainer
builds independent episode memories directly from the Phase-A corpus and optimizes the encoder and
reranker together. The only retrieval policy is a fixed 64-row cosine shortlist.

## End-to-end training

```bash
python -m training.tokenizer.pretrain_episodic \
  --random-init \
  --phase-b-regime unified \
  --steps 35000 \
  --episodes-per-step 8 \
  --candidate-counts 2 4 8 16 \
  --query-labels-per-episode 4 \
  --queries-per-candidate 4 \
  --max-support 16 \
  --bank-windows 512 \
  --val-every 1000 \
  --val-episodes 32 \
  --num-workers 8 \
  --out training/tokenizer/outputs/<run-name>
```

Current defaults use clean one-second patches, eight independent episodes per optimizer step, 16
query recordings per episode, and one recording row for every six-second query or memory window.
Candidate rosters are 2/4/8/16 and support counts are 0/1/2/4/8/16. Random aliases and signal
augmentation are disabled unless explicitly requested.

Use `--profile-steps 12` for a bounded real-corpus GPU profile and `--smoke` for a three-step
integration test. Telemetry is appended to `log.jsonl`. Validation reports the learned curve, raw
nearest-neighbor curve, enrolled-1NN curve where defined, and their margins. `best.pt` is selected on
the held-out-concept coherent k-curve.

## Frozen encoder controls

`--checkpoint ... --freeze-encoder` trains only the reranker on one selected HALO encoder. The
`--encoder-comparison` experiment can use released HARNet or UniMTS encoders with the same episode
protocol. These are controls, not separate HALO designs.

## Historical code

`build_memory.py`, `resolvability.py`, `gate_predictor.py`, `train_admissibility_gate.py`,
`train_patch_decoder.py`, and `train_patch_confidence.py` reproduce earlier experiments. They are not
imported by `pretrain_episodic.py` and their artifacts are incompatible with current reranker
checkpoints. The previous active voting model is preserved at Git tag
`phaseb-vector8-vote-20260824`.
