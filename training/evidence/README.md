# training/evidence/ — Pipeline B (Evidence: memory + prediction)

**Pipeline B** is a retrieval/evidence mechanism initialized from Phase-A representations. Its
predictor has one objective: candidate-set cross-entropy on answerable episodes. The default keeps
Phase A frozen; `--tokenizer-mode ema_finetune` uses detached EMA retrieval keys and re-encodes only
the selected raw query/evidence windows with gradients. A second, frozen-predictor stage calibrates
reject confidence from correctness and truth-absent episodes.

- **Canonical motivation and live contract:** see
  [`docs/design/PHASE_B_TRAINING_INTENT.md`](../../docs/design/PHASE_B_TRAINING_INTENT.md).
- **Phase-A artifact handoff:** see
  [`docs/design/PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md`](../../docs/design/PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md).
- **Historical research:** `docs/design/EVIDENCE_ENGINE*.md` records earlier experiments and rejected
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
python -m training.evidence.train_patch_confidence --device cuda
python -m training.evidence.eval_patch_decoder --device cuda \
  --confidence training/evidence/outputs/patch_evidence_confidence.pt
python -m training.evidence.eval_enrollment --device cuda
python -m training.evidence.eval_enrollment --device cuda --random-aliases
```

`eval_patch_decoder` is the semantic zero-shot protocol. `eval_enrollment` reports same-subject and
cross-subject support curves for `k=0,1,2,4,8`; its random-alias run starts at `k=1` and isolates example-based adaptation
from any help supplied by known label semantics.

The standard predictor exposes one retrieval-capacity setting:

```bash
python -m training.evidence.train_patch_decoder --device cuda --evidence-budget 64
```

Retrieval K and per-window/per-label contribution limits are derived from that budget. Candidate
labels are task input. Training cycles evenly over semantic zero-support, ordinary few-support,
cross-subject/cross-config few-support, and same-subject enrollment. Supported episodes use exactly
`1`, `2`, `4`, or `8` independent event/window examples for every candidate and mix coherent labels
with episode-local neutral aliases. Candidate sets contain `4`, `8`, `12`, or `16` labels. The
archive has one global upper budget; when the source corpus is smaller no rows are discarded. Its
active label/config/subject-balanced view rotates every 100 steps.

Physical views are a fixed 50/50 recipe: exact clean source query/support executions, or the full
virtual-subject plus mild acquisition-augmentation simulation. Validation evaluates every held-out
episode both ways and reports clean and augmented balanced accuracy separately.

Inference remains hard top-k. Training attaches a balanced soft all-memory vote only in backward so
unselected rows teach retrieval without changing the forward result. Phase-B health telemetry is
updated about once per minute in `training/evidence/outputs/telemetry/`.

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
