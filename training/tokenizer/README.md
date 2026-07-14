# training/tokenizer/ — Pipeline A (Representation) harness

**Pipeline A** of the evidence engine: the *learnable representation pipeline* — everything that
turns raw signal into a heterogeneity-salient representation. Not a fixed transform; it is trained.

Covers: time-domain **preprocessing** (gravity-align), **cross-channel relational/causal learning**
(masked-channel set model → residuals), the **learnable frequency domain** (fixed physical
filterbank + constrained-learnable scattering/SincNet), and the **primitives**.

- **Design / plan:** [`docs/design/EVIDENCE_ENGINE.md`](../../docs/design/EVIDENCE_ENGINE.md) ·
  [`docs/design/EVIDENCE_ENGINE_BUILD_PLAN.md`](../../docs/design/EVIDENCE_ENGINE_BUILD_PLAN.md).
- **Phase 1 training:** masked-channel SSL + config-conditional salient-contrastive +
  analysis-consistency (M0–M3). Validated by the robustness probe *before* Pipeline B exists.
- **Model components:** `model/tokenizer/`.
- **Seam to Pipeline B:** emits `{query representation + structured primitives + per-channel text id}`.

Status: design only; no implementation yet.
