# Agreed Phase-A and Phase-B Implementation Plan

Status: Phase A completed on 2026-08-07. The selected checkpoint is
`training/tokenizer/outputs/phase_a_headline/best.pt` at step 27,000 (`val_ba=0.288435`). Phase B was
consolidated on 2026-08-07 to one candidate-CE predictor objective followed by a separate frozen-
predictor confidence-calibration stage. Earlier pooled-window, EDL, auxiliary-loss, and duplicate
multi-subspace implementations were removed.

## Implementation map and launch gates

Phase A:

- objectives and train loop: `training/tokenizer/{losses_repr,pretrain}.py`;
- label-free hierarchical data/sampling: `training/tokenizer/pretrain_data.py`;
- tests: `tests/{test_losses_repr,test_pretrain_data,test_build_grids}.py`;
- CPU integration smoke passed:
  `python -m training.tokenizer.pretrain --smoke --steps 2 --out /tmp/halo_phase_a_smoke --force`.

Phase B:

- schema-v2 pooled + patch bank: `training/evidence/{build_memory,bank_guard}.py` and
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
```

`eval_patch_decoder.py` reports the identity evidence-decoder control beside the trained predictor.
There is no second pooled-window learned trainer.

Patch retrieval uses a bounded EMA-projected coreset by default (`--index-per-label 256`) for
tractable independent query-patch/subspace lookup. Each label draw is stratified across config and
resolution, and the coreset is resampled at index refreshes; `--index-per-label -1` is the exact
full-bank control when hardware permits. Consequently, an episodic support value of `all` means all
eligible source windows in the active index, while repeated refreshes expose the full bank over
training.

Corpus scale is controlled without deleting windows. Phase A assigns dataset mass proportional to
`n^0.25`, caps any dataset share at 25%, and splits each dataset's mass across subjects
proportional to `n_subject^0.5`. There is no source-specific pair quota. Phase B's 8,000-window label cap is
water-filled across configurations, and patch-episode queries are uniform over selected labels and
configurations with square-root subject tempering. Historical controls remain available through
`--sampler-alpha 0.5 --sampler-max-dataset-share 1 --sampler-subject-alpha 1`,
`build_memory --label-cap-policy random`, and `train_patch_decoder --query-balance legacy_sqrt`.

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

- Use learned structural roles `QUERY`, `EVIDENCE`, and `CANDIDATE`.
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

Vary these axes independently:

- candidate budget: approximately 10%, 25%, 50%, and 100% of the vocabulary;
- memory-label roster: approximately 25%, 50%, and 100% of eligible training labels, independent of
  the candidate roster;
- true-label memory support: 0, 1, 2, 4, 8, or all eligible examples;
- independently subsampled support for other candidate labels;
- same-config, cross-config-only, and query-config-absent memory;
- random, language-near, motion-family-near, and physically confusable distractors;
- predictor stage: truth-present classification episodes only, optimized by candidate cross-entropy;
- confidence stage: the predictor is frozen and truth-present/truth-absent episodes train only BCE.

The query window/event and subject are always excluded from memory. Reserve complete label families
from decoder training for model selection so candidate-text reasoning is tested on unseen concepts.
One support example means one verified physical event, including all of its synchronous placements;
unpaired data uses one source window.
True support is sampled with probabilities 35%, 25%, 15%, 10%, 5%, and 10% for
`0,1,2,4,8,all`; other-label support uses 5%, 10%, 15%, 20%, 20%, and 30%. Requested finite true
support must be physically realizable after leakage/configuration exclusions or the episode is
resampled, and requested versus realized support is logged.

## Compatibility, provenance, and gates

- Record all objective modes, coefficients, EMA settings, event-pair policy, memory schema, and
  structural-metadata policy in run artifacts.
- Bind every Phase-B artifact to the exact Phase-A checkpoint, corpus, bank fingerprint, patch schema,
  and candidate-text vocabulary.
- Keep controls for implicit versus explicit acquisition metadata, no-subspace versus subspace
  retrieval, fixed versus learnable frontend, JEPA+VICReg versus VICReg-only, and the identity
  evidence decoder.
- Required tests cover VICReg collapse prevention, EMA stop-gradient/update/restore, event-pair
  integrity, patch-bank provenance, leakage exclusions, candidate/evidence permutations, metadata
  masking, episodic support budgets, confidence behavior, and checkpoint guards.
