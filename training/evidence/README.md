# training/evidence/ — Pipeline B (Evidence: memory + prediction)

**Pipeline B** is a retrieval/evidence mechanism trained on top of frozen Phase-A representations.
Its live predictor has one objective: candidate-set cross-entropy on answerable episodes. A second,
frozen-predictor stage calibrates reject confidence from correctness and truth-absent episodes.

- **Design:** see [`docs/design/EVIDENCE_ENGINE.md`](../../docs/design/EVIDENCE_ENGINE.md).
- **Build plan:** see [`docs/design/EVIDENCE_ENGINE_BUILD_PLAN.md`](../../docs/design/EVIDENCE_ENGINE_BUILD_PLAN.md).
- **Shared with the rest of HALO:** the tokenizer (`model/tokenizer/`, physical filterbank +
  extensions). Everything else here — the archetypal memory, the evidence decoder, the training
  loop — is bespoke to this approach.
- **Status:** the consolidated predictor and confidence stages are implemented and have CPU smokes.
  Earlier pooled-window, EDL, auxiliary-loss, and duplicate multi-subspace training paths were
  removed. Whether the new predictor beats its identity control is an empirical gate, not a claim.

Nothing here should import or be imported by a conventional classifier trainer; the only shared
dependency is the tokenizer.

Run the real sequence after Phase A finishes:

```bash
python -m training.evidence.build_memory --device cuda
python -m training.evidence.train_patch_decoder --device cuda
python -m training.evidence.train_patch_confidence --device cuda
python -m training.evidence.eval_patch_decoder --device cuda \
  --confidence training/evidence/outputs/patch_evidence_confidence.pt
```
