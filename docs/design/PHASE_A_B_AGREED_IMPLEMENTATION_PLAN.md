# Agreed Phase-A and Phase-B Implementation Plan

Status: Phase A completed on 2026-08-07. The selected checkpoint is
`training/tokenizer/outputs/phase_a_headline/best.pt` at step 27,000 (`val_ba=0.288435`). Phase B was
consolidated on 2026-08-07 to one candidate-CE predictor objective followed by a separate frozen-
predictor confidence-calibration stage. Earlier pooled-window, EDL, auxiliary-loss, and duplicate
multi-subspace implementations were removed.

The adaptation-focused episode redesign was implemented on 2026-08-08. Its motivation and exact
live recipe are in `PHASE_B_TRAINING_INTENT.md`.

## Implementation map and launch gates

Phase A:

- objectives and train loop: `training/tokenizer/{losses_repr,pretrain}.py`;
- label-free hierarchical data/sampling: `training/tokenizer/pretrain_data.py`;
- tests: `tests/{test_losses_repr,test_pretrain_data,test_build_grids}.py`;
- CPU integration smoke passed:
  `python -m training.tokenizer.pretrain --smoke --steps 2 --out /tmp/halo_phase_a_smoke --force`.

Phase B:

- schema-v3 pooled + source-aware patch bank: `training/evidence/{build_memory,bank_guard}.py` and
  `training/tokenizer/eval_transfer.py`;
- candidate-aware decoder, confidence, learned EMA subspaces:
  `model/evidence/{decoder,confidence,patch_retrieval}.py`;
- episodic plumbing/trainer/calibrator/evaluator:
  `training/evidence/{patch_episodes,train_patch_decoder,train_patch_confidence,eval_patch_decoder}.py`;
- canonical motion families: `data/labels/activity_families.json`;
- tests: `tests/{test_eval_transfer_detailed,test_evidence_patch_pipeline}.py`;
- CPU integration smokes passed:
  `python -m training.evidence.train_patch_decoder --smoke` and
  `python -m training.evidence.train_patch_confidence --smoke`.

Native-grid event migration completed on 2026-07-25:

- all 29 materialized native grids carry explicit event IDs;
- all 20 Phase-A streams are finite and metadata-consistent;
- XRF V2 contributes 5,794 events shared across six placements;
- NFI-FARED contributes 13,260 events shared across two placements;
- the subject-disjoint training split contains 17,387 paired events / 55,270 paired windows.

The XRF V2 and NFI-FARED converters still write event identities because Phase B uses them for bank
guarding and episode construction. Phase A neither requires nor samples by those identities.

Before a real Phase-B run, rebuild the bank with the selected frozen Phase-A checkpoint, then train
and evaluate the patch arm:

```bash
python -m training.evidence.build_memory --device cuda
python -m training.evidence.train_patch_decoder --device cuda
python -m training.evidence.train_patch_confidence --device cuda
python -m training.evidence.eval_patch_decoder --device cuda \
  --confidence training/evidence/outputs/patch_evidence_confidence.pt
python -m training.evidence.eval_enrollment --device cuda
```

`eval_patch_decoder.py` reports the identity evidence-decoder control beside the trained predictor.
There is no second pooled-window learned trainer.

Patch retrieval uses a fixed-size, EMA-projected active index for tractable independent
query-patch/subspace lookup. The index samples 16 source windows per label hierarchically across
configuration and subject and retains every valid patch grid from those windows. This is an
implementation policy, not a user-facing hyperparameter. In fine-tuning mode its EMA-tokenizer keys
are refreshed in deterministic shards.

Corpus scale is controlled without deleting windows. Phase A assigns dataset mass proportional to
`n^0.25`, caps any dataset share at 25%, and splits each dataset's mass across subjects
proportional to `n_subject^0.5`. There is no source-specific pair quota. Phase B uses one global,
label/configuration-balanced archive budget instead of independently tunable stream and label caps;
its patch-episode queries are uniform over selected labels and configurations with square-root
subject tempering.

Phase A remains activity-label-free. Phase B may use labels attached to retrieved memory examples and
the runtime candidate vocabulary.

## Phase A: label-free representation pretraining

The consolidated objective is:

```text
JEPA masked contextual prediction
+ augmentation VICReg for every window
```

### Augmentation VICReg

- Use float32 VICReg over two independently augmented views of every sampled window.
- Use raw projector outputs for VICReg's MSE invariance, variance, and covariance terms.
- Measure post-warmup JEPA and VICReg encoder-gradient geometry once, then freeze the solved scalar
  weights. Do not continuously adapt weights or alternate objectives.

### Masked targets

- Use only an EMA teacher with no gradients. The clean teacher emits contextual token targets; the masked
  student predicts them at valid masked positions through a dedicated predictor.
- Update the teacher only after an optimizer step and save/restore it in checkpoints.
- Draw one randomly located contiguous block independently in each resolution. Let resolutions
  contextualize each other and keep at least one real token globally visible.
- Reduce each resolution independently and duration-weight partial tail patches.

### Frontends, sampling, and health

- Run the fixed and mildly learnable filterbank arms through exactly the same objective and data path.
- Keep source-temperature sampling for the label-free recipe.
- Log per-objective gradient norms, VICReg component values, minimum feature standard deviation,
  effective rank, embedding norms, covariance statistics, and positive similarity by pair type.
- Warn or fail on non-finite values and clear representation collapse.

Cadence/eigen primitives are diagnostics, not objectives. Physical reconstruction, SimCLR, SupCon,
TF-C, separate A4, mask compensation, balanced sampling, and objective auto-calibration were deleted
from the live trainer rather than retained as dormant switches.

## Phase B: patch evidence with candidate hypotheses

### Memory and retrieval

- Preserve pooled window vectors as the frozen T2.0 control.
- Add a versioned patch table containing each valid patch vector and its label, subject, dataset/config,
  source window, physical event, sensor group, center time, duration, resolution, and validity.
- Encode each sensor stream separately and combine sensor patches into a session-level query set.
- Retrieve separately for every query patch while excluding the same window/event and subject in
  training.
- Cap retrieved contributions per source window and label.
- Aggregate evidence with physical-duration normalization within each resolution, then combine
  resolutions without giving the denser short-patch grid an automatic vote-count advantage.

### Learned subspaces

- Use learned projections, never hard contiguous dimension slices.
- Give each patch subspace its own retrieval and evidence contribution.
- Use an EMA copy of the subspace projectors for memory indexing and rebuild projected indexes
  periodically; online projectors score the retrieved items.
- Do not add subspace-specific auxiliary losses. Head contribution and overlap are telemetry; the
  candidate prediction objective decides whether specialized retrieval is useful.

### Decoder token set

- Use learned structural roles `QUERY`, `EVIDENCE`, `PROVIDED_SUPPORT`, and `CANDIDATE`.
- Candidate tokens contain frozen label-text content plus the candidate role.
- First contextualize query/evidence patches. Candidate tokens then use permutation-equivariant
  self-attention and cross-attention over the fixed physical evidence states.
- Candidate order and evidence order receive no arbitrary sequence positions.
- Always pass structural metadata: role, source-window membership, sensor membership, patch center,
  patch duration, resolution, and validity masks.
- Treat explicit acquisition text/rate/gravity reinjection as an ablation because Phase A already
  conditions on those values.

### Prediction and confidence

- Emit nonnegative evidence per candidate with patch, sensor, subspace, and retrieved-example
  attribution.
- Do not introduce an `UNKNOWN` candidate.
- Select among allowed labels from normalized candidate evidence.
- After predictor training, freeze it and estimate reject confidence separately from absolute evidence, retrieval density, patch/sensor
  agreement, normalized vote entropy, and the leading-candidate margin.
- Train confidence to predict `correct AND answerable` on a mixture of truth-present and truth-absent
  episodes, then evaluate ECE, AUROC, and risk/coverage. Confidence never changes candidate logits.

### Episodic training

Use the following fixed recipe:

- exactly balanced cycle over semantic zero-support, ordinary few-support, cross-subject/cross-config
  few-support, and same-subject enrollment regimes;
- four or eight candidate labels;
- exactly 1, 2, 4, or 8 support event/window units per candidate in supported episodes;
- coherent labels or episode-local random aliases, with aliases forbidden at zero support;
- remove ordinary memory rows of every candidate concept and restore only selected support;
- apply persistent virtual-subject style before independent acquisition augmentation;
- random, language-near, motion-family-near, and physically confusable distractors;
- predictor stage: truth-present classification episodes only, optimized by candidate cross-entropy;
- confidence stage: the predictor is frozen; adaptation-distributed truth-present and coherent
  truth-absent episodes train only BCE.

The query window and verified event are always excluded from support. Cross-subject episodes also
exclude query subjects and configurations when selecting support. Reserve complete label families
from decoder training for model selection so candidate-text reasoning is tested on unseen concepts.
One support example means one verified physical event, including all of its synchronous placements;
unpaired data uses one source window.
Labels without enough independent executions remain in lower-support episodes but are excluded from
infeasible high-support strata.

The numerical predictor uses hard top-k retrieval. A label/window/resolution/duration-balanced soft
all-memory route is attached only in backward with a straight-through estimator, so non-selected
rows teach the retrieval projection without changing inference behavior.

## Compatibility, provenance, and gates

- Record all objective modes, coefficients, EMA settings, event-pair policy, memory schema, and
  structural-metadata policy in run artifacts.
- Bind every Phase-B artifact to the exact Phase-A checkpoint, corpus, bank fingerprint, patch schema,
  and candidate-text vocabulary.
- Keep controls for implicit versus explicit acquisition metadata, frozen versus EMA-fine-tuned
  tokenizer, fixed versus learnable Phase-A frontend, JEPA+VICReg versus VICReg-only, and the
  identity evidence decoder. Retrieval subspaces are part of the fixed Phase-B recipe rather than a
  dormant no-subspace branch.
- Required tests cover VICReg collapse prevention, EMA stop-gradient/update/restore, event-pair
  integrity, patch-bank provenance, leakage exclusions, candidate/evidence permutations, metadata
  masking, episodic support budgets, confidence behavior, and checkpoint guards.
## Phase B configuration and end-to-end tokenizer fine-tuning

Phase B exposes only orthogonal concepts:

1. **Candidate labels** are task input, not a memory hyperparameter.
2. **Support** (`zero`, `1`, `2`, `4`, `8` per candidate) is an episode condition. The standard
   training recipe samples a fixed mixture and validation reports every condition separately.
3. **Evidence budget** is the sole retrieval-capacity knob. Retrieval K and per-window/per-label
   contribution limits are derived from it.
4. **Tokenizer mode** is either `frozen` or `ema_finetune`.

The archive budget, rotating active-index construction, retrieval subspaces, heterogeneity mixture, and
non-truth memory population are fixed recipe policies. They are persisted in artifacts and logged,
but are not independently tunable CLI flags. This makes contradictory states (for example asking
for eight supports after excluding the truth label, or setting a per-label cap above the total
evidence budget) unrepresentable.

In `ema_finetune` mode the complete memory bank remains detached. Every bank window stores its
deployment stream and original native-grid row. A detached EMA tokenizer supplies global
selection keys. After hard top-K selection and bounded evidence assembly, the raw query and selected
source windows are reloaded and passed through the online tokenizer. Query/evidence similarities
are recomputed differentiably inside this selected set, so candidate cross-entropy updates the
online tokenizer, retriever, and evidence decoder without retaining a graph for the full bank.
EMA keys are refreshed in deterministic shards, and inference uses the EMA tokenizer.
