# training/evidence/ — evidence-engine harness (separate line of work)

This directory is reserved for the **evidence-engine** training harness — a *retrieval /
evidence-accumulation* mechanism, deliberately **kept separate from any conventional
softmax/cosine-classifier training path** so the two never conflate.

- **Design:** see [`docs/design/EVIDENCE_ENGINE.md`](../../docs/design/EVIDENCE_ENGINE.md).
- **Shared with the rest of HALO:** the tokenizer (`model/tokenizer/`, physical filterbank +
  extensions). Everything else here — the archetypal memory, the evidence decoder, the training
  loop — is bespoke to this approach.
- **Status:** design only; no implementation yet. The design is not finalized (see the open
  forks in the design doc). Name "evidence" is a placeholder.

Nothing here should import or be imported by a conventional classifier trainer; the only shared
dependency is the tokenizer.
