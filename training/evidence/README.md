# training/evidence/ — Pipeline B (Evidence: memory + prediction)

**Pipeline B** of the evidence engine — a *retrieval / evidence-accumulation* mechanism (archetypal
memory → evidence → evidential/abstaining prediction), trained on top of Pipeline A's representations
(`training/tokenizer/`). Deliberately **kept separate from any conventional softmax/cosine-classifier
training path** so the two never conflate.

- **Design:** see [`docs/design/EVIDENCE_ENGINE.md`](../../docs/design/EVIDENCE_ENGINE.md).
- **Build plan:** see [`docs/design/EVIDENCE_ENGINE_BUILD_PLAN.md`](../../docs/design/EVIDENCE_ENGINE_BUILD_PLAN.md).
- **Shared with the rest of HALO:** the tokenizer (`model/tokenizer/`, physical filterbank +
  extensions). Everything else here — the archetypal memory, the evidence decoder, the training
  loop — is bespoke to this approach.
- **Status:** implemented — memory bank, per-patch retrieval with learned subspaces, candidate-aware
  decoder, confidence/EDL, and the episodic trainer/evaluator all exist and have CPU smokes. What is
  *not* settled is whether training the decoder helps: the untrained retrieval + text-ensemble
  configuration remains the strongest evidence arm, and the trained decoder has been measured
  net-negative against its control. See `docs/design/EVIDENCE_ENGINE_FINDINGS.md` for the current
  position and the retracted claims.

Nothing here should import or be imported by a conventional classifier trainer; the only shared
dependency is the tokenizer.
