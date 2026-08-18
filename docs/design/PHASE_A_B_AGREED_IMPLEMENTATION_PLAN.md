# Phase-A Implementation Record and Phase-B Handoff

Status: the sensor-granularity Phase-A run completed on 2026-08-17, but it is now a historical failed
arm: independent SO(3) invariance removed gravity-frame signal, retrieval rows remained
cross-sensor-contextual, and checkpoint selection used an internal source probe. Do not build the
next Phase-B bank from `phase_a_fixed_1s_rotation_20260817/best.pt`. The replacement clean recipe is
implemented and awaits training. The older
`training/tokenizer/outputs/phase_a_headline/best.pt` checkpoint is channel-granular and historical.

This document records the final Phase-A recipe and the artifact contract at the boundary between
phases. It intentionally does not duplicate the Phase-B design. The canonical Phase-B motivation,
episode policy, objective, and validation gates live in `PHASE_B_TRAINING_INTENT.md`; runnable
commands live in `training/evidence/README.md`.

## Implementation Map

Phase A:

- objectives and train loop: `training/tokenizer/losses_repr.py` and `pretrain.py`;
- label-free hierarchical data/sampling: `training/tokenizer/pretrain_data.py`;
- tokenizer and encoder: `model/tokenizer/`;
- executable documentation: `training/tokenizer/README.md`;
- focused tests: `tests/test_losses_repr.py`, `test_pretrain_data.py`, and `test_build_grids.py`.

The CPU integration smoke is:

```bash
python -m training.tokenizer.pretrain --smoke --steps 2 \
  --out /tmp/halo_phase_a_smoke --force
```

Phase B consumes the Phase-A patch representation through:

- memory construction and provenance: `training/evidence/build_memory.py` and `bank_guard.py`;
- train-only per-sensor measurement: `training/evidence/resolvability.py`;
- admissibility and sensor-patch retrieval: `training/evidence/admissibility_gate.py`,
  `gate_predictor.py`, and `admissible_retrieval.py`;
- enrollment evaluation: `training/evidence/eval_enrollment.py`.

The learned subspace retriever, relational decoder, and `train_patch_decoder.py` are parked
reproduction paths. They are not the current Phase-B handoff.

## Final Phase-A Objective

The live label-free objective is exactly:

```text
JEPA masked contextual prediction
+ augmentation VICReg on every sampled window
```

### JEPA

- A clean EMA teacher emits contextual token targets with no gradient.
- The masked student predicts teacher latents at valid masked positions through a dedicated
  predictor.
- One randomly located contiguous block is drawn independently in every resolution.
- Resolutions can contextualize one another, and at least one valid token remains visible globally.
- Each resolution is reduced independently; partial tail patches are duration weighted.
- The teacher updates only after an optimizer step and is checkpointed and restored.

### VICReg collapse control

- The clean reference uses identical physical views. Transformations are controlled ablations.
- VICReg runs in float32 both on projected pooled embeddings and directly on the sensor-isolated
  retrieval rows used by Phase B.
- Its invariance, variance, and covariance terms provide a label-free relation objective without
  defining other windows as negatives.
- Post-warmup JEPA/VICReg encoder-gradient geometry is measured once; solved scalar weights remain
  fixed rather than being continuously auto-balanced.

Cadence/eigen primitives remain diagnostics rather than objectives. Physical reconstruction,
SimCLR, SupCon, TF-C, separate multistream relation losses, causal/future-tail masking, mask
compensation, objective alternation, and objective auto-calibration were removed from the live
trainer rather than retained as dormant flags.

## Frontend, Sampling, and Health

- Fixed and mildly learnable filterbank arms use the same objectives and data path.
- Phase A is activity-label-free. Labels may be retained as metadata for diagnostics but do not
  determine sampling positives or enter either loss.
- Dataset sampling mass is proportional to `n^0.25`, capped at 25% per dataset, then split across
  subjects proportional to `n_subject^0.5`.
- Training logs per-objective gradient norms, VICReg components, minimum feature standard deviation,
  effective rank, embedding norms, covariance statistics, and positive-view similarity.
- Non-finite values and clear representation collapse are explicit failures.

## Native-Grid Event Provenance

The event migration completed on 2026-07-25:

- all 29 materialized native grids carry explicit event identities;
- all 20 Phase-A streams were finite and metadata-consistent at that audit;
- XRF V2 contributes 5,794 events shared across six placements;
- NFI-FARED contributes 13,260 events shared across two placements;
- the subject-disjoint training split contains 17,387 paired events and 55,270 paired windows.

Phase A does not sample by event identity. Phase B uses event identity to group synchronous
placements and prevent query/support leakage.

## Phase-B Handoff Contract

Every Phase-B artifact must bind to:

- the exact Phase-A checkpoint and tokenizer configuration;
- corpus and subject-split fingerprints;
- schema-5 per-sensor bank layout, stored descriptor semantics, source-row provenance, and sensor
  embedding-path probe;
- candidate-text encoder/vocabulary provenance;
- structural-metadata and event-pair policy.

Existing memory banks are obsolete for the current design because their rows came from the old
cross-sensor path. Rebuild only after the replacement Phase-A checkpoint passes development transfer,
rank, provenance, and robustness gates. From that point onward, use
`PHASE_B_TRAINING_INTENT.md` for the training contract and
`training/evidence/README.md` for commands. This handoff document should not acquire another copy of
the Phase-B recipe.
