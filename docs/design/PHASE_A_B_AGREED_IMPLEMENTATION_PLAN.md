# Phase-A Implementation Record and Phase-B Handoff

Status: Phase A completed on 2026-08-07. The selected checkpoint is
`training/tokenizer/outputs/phase_a_headline/best.pt` at step 27,000 (`val_ba=0.288435`).

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
- patch retrieval and evidence decoder: `model/evidence/patch_retrieval.py` and
  `relational_decoder.py`;
- episodic predictor training: `training/evidence/train_patch_decoder.py`. Confidence calibration
  is parked and its separate `train_patch_confidence.py` experiment is not part of the active
  Phase-B claim.

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

### Augmentation VICReg

- Two independently augmented views are available for every sampled window.
- VICReg runs in float32 on raw projector outputs.
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
- patch-bank schema and source-row provenance;
- candidate-text encoder/vocabulary provenance;
- structural-metadata and event-pair policy.

The existing pre-redesign memory bank is obsolete because it has no current patch table or
source-row provenance. Rebuild it from the selected Phase-A checkpoint before the first real
Phase-B run. From that point onward, use `PHASE_B_TRAINING_INTENT.md` for the training contract and
`training/evidence/README.md` for commands. This handoff document should not acquire another copy of
the Phase-B recipe.
