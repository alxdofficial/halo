# training/evidence/ — Pipeline B (Evidence: memory + prediction)

**Pipeline B** is a retrieval/evidence mechanism initialized from Phase-A representations. Its
predictor has one objective: candidate-set cross-entropy on answerable episodes. The default keeps
Phase A frozen; `--tokenizer-mode ema_finetune` uses detached EMA retrieval keys and re-encodes only
the selected raw query/evidence windows with gradients. The separate reject-confidence calibration
experiment is implemented but parked; it is not part of the current Phase-B launch sequence.

- **Canonical motivation and live contract:** see
  [`docs/design/PHASE_B_TRAINING_INTENT.md`](../../docs/design/PHASE_B_TRAINING_INTENT.md).
- **Phase-A artifact handoff:** see
  [`docs/design/PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md`](../../docs/design/PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md).
- **Historical research:** `docs/archive/EVIDENCE_ENGINE*.md` records earlier experiments and rejected
  branches; it is not configuration guidance.
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
python -m training.evidence.train_patch_decoder --device cuda --real-smoke
python -m training.evidence.train_patch_decoder --device cuda
python -m training.evidence.eval_patch_decoder --device cuda
python -m training.evidence.eval_enrollment --device cuda
python -m training.evidence.eval_enrollment --device cuda --random-aliases
# Run the sealed roster only after development decisions are frozen.
python -m training.evidence.eval_patch_decoder --device cuda --protocol-role test
python -m training.evidence.eval_enrollment --device cuda --protocol-role test
```

`eval_patch_decoder` is the semantic zero-shot protocol. Both evaluators default to the development
roster (`motionsense`, `realworld`, `shoaib`); `--protocol-role test` selects the sealed external
roster (`inclusivehar`, `usc_had`, `tnda_har`, `ut_complex`). Explicit `--datasets` overrides either
roster and is recorded in the result artifact.

`eval_enrollment` reports same-subject and cross-subject support curves for `k=0,1,2,4,8`; its
random-alias run starts at `k=1` and isolates example-based adaptation from help supplied by known
label semantics. Before support is appended, every base-archive row whose canonical concept is one
of the episode's candidate labels is removed; candidate concepts can enter the runtime memory only
through explicit enrollment. A curve freezes its subjects, candidate labels, and query windows at
the highest execution-supported `k`; smaller `k` values use nested prefixes of the same support set.
Unsupported points are marked rather than silently changing the evaluated population. Window-level
pseudo-event ids are rejected for same-subject adaptation. Every result includes the learned
decoder, identity decoder, support-removed, cyclically label-shuffled support, prototype, and fitted
L2 ridge-head controls, plus seen/unseen-concept and per-subject results. Enrollment summaries treat
subjects as the independent unit and include paired subject-bootstrap intervals for each control
delta. The semantic evaluation reports subject-bootstrap intervals per deployment stream.
Development, sealed-test, and explicit custom runs use separate output filenames.

The standard predictor exposes one retrieval-capacity setting:

```bash
python -m training.evidence.train_patch_decoder --device cuda --evidence-budget 64
```

Retrieval K and per-window/per-label contribution limits are derived from that budget. Candidate
labels are task input. Training cycles evenly over semantic zero-support, ordinary few-support,
cross-subject few-support, and same-subject enrollment. Supported episodes use exactly
`1`, `2`, `4`, or `8` independent event/window examples for every candidate and mix coherent labels
with episode-local neutral aliases. Candidate sets contain `4`, `8`, `12`, or `16` labels. The
archive has one global upper budget; when the source corpus is smaller no rows are discarded. Its
active label/config/subject-balanced view rotates every 100 steps.

Physical views are a fixed 50/50 recipe: stored clean frozen-encoder query/support vectors (or live
clean forwards when fine-tuning), or the full virtual-subject plus mild acquisition-augmentation
simulation. Validation evaluates every held-out
episode both ways and reports clean and augmented balanced accuracy separately.
Its fixed episodes cycle evenly across held-subject, held-configuration, and jointly held queries.

Inference remains hard top-k. Training attaches a 0.1-weighted balanced soft all-memory vote only in backward so
unselected rows teach retrieval without changing the forward result. Predictor and confidence health
telemetry is updated about once per minute in separate run directories:

```text
training/evidence/outputs/telemetry/patch_evidence_predictor/
training/evidence/outputs/telemetry/patch_evidence_confidence/
```

Each launch writes an immediate run-identified heartbeat, a run-specific JSONL history, and an atomic
`phase_b_telemetry_latest.json`. Generate a machine-readable health verdict, concise text summary,
and live plot without using the training GPU:

```bash
python -m training.evidence.monitor_training \
  --telemetry-dir training/evidence/outputs/telemetry/patch_evidence_predictor \
  --render --watch 60
```

Predictor telemetry includes candidate-count-normalized CE, training accuracy, per-curriculum-stratum
metrics, component gradients, clipping, hard/soft retrieval-gradient agreement, evidence-pool
concentration, support usage, fixed train/held-out canaries, throughput, and VRAM. Confidence telemetry
includes target balance, score saturation, BCE, AUROC/AUPRC, ECE, Brier score, and risk/coverage. Both
stages fail before the optimizer update on non-finite gradients.

Both training stages write atomic resumable state beside their output as `*.last.pt`. Resume with
the same command and `--resume <path-to-last-state>`; bank identity and trajectory-affecting options
are checked before state is restored.

The optional end-to-end experiment is:

```bash
python -m training.evidence.train_patch_decoder --device cuda \
  --tokenizer-mode ema_finetune \
  --checkpoint training/tokenizer/outputs/phase_a_headline/best.pt
```

Fine-tuning starts after a fixed decoder warm-up so the identity-initialized decoder first develops a
nontrivial physical path. Inference and confidence calibration use the saved EMA tokenizer.
