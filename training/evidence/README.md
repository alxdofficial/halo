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
- **Status:** design only; no implementation yet. The design is not finalized (see the open
  forks in the design doc). Name "evidence" is a placeholder.

Nothing here should import or be imported by a conventional classifier trainer; the only shared
dependency is the tokenizer.
