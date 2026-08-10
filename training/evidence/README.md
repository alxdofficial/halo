# training/evidence/ — Pipeline B (Evidence: memory + prediction)

**Pipeline B** is a retrieval/evidence mechanism initialized from Phase-A representations. Its
predictor has exactly one objective: candidate-set cross-entropy on answerable episodes. The default keeps
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
- **Status:** the relational learned-query predictor has passed unit and synthetic integration
  tests but has not yet been trained at scale. The superseded run and the defects that motivated
  this design are preserved in
  [`docs/results/PHASE_B_TRAINING_STATUS.md`](../../docs/results/PHASE_B_TRAINING_STATUS.md). Earlier
  pooled-window, EDL, auxiliary-loss, and duplicate multi-subspace training paths were removed.

Nothing here should import or be imported by a conventional classifier trainer; the only shared
dependency is the tokenizer.

Run the real sequence after Phase A finishes:

```bash
python -m training.evidence.build_memory --device cuda
python -m training.evidence.train_patch_decoder --device cuda --real-smoke
python -m training.evidence.train_patch_decoder --device cuda
python -m training.evidence.eval_enrollment --device cuda
python -m training.evidence.eval_enrollment --device cuda --random-aliases
# Run the sealed roster only after development decisions are frozen.
python -m training.evidence.eval_enrollment --device cuda --protocol-role test
```

The coherent-label `k=0` enrollment cell is the semantic zero-support protocol. The evaluator
defaults to the development roster (`motionsense`, `realworld`, `shoaib`); `--protocol-role test`
selects the sealed external roster (`inclusivehar`, `usc_had`, `tnda_har`, `ut_complex`). Explicit
`--datasets` overrides either roster and is recorded in the result artifact.

`eval_enrollment` reports same-subject and cross-subject support curves for `k=0,1,2,4,8`. Positive
coherent-label cells include both full enrollment and a fixed half-candidate partial-enrollment
condition. Where another valid dataset stream exists, it also enrolls from that configuration and
queries the primary stream. The random-alias run starts at `k=1`, remains fully enrolled, and
isolates example-based adaptation from help supplied by known label semantics. Before support is
appended, every base-archive row whose canonical concept is one
of the episode's candidate labels is removed; candidate concepts can enter the runtime memory only
through explicit enrollment. A curve freezes its subjects, candidate labels, and query windows at
the highest execution-supported `k`; smaller `k` values use nested prefixes of the same support set.
Unsupported points are marked rather than silently changing the evaluated population. Window-level
pseudo-event ids are rejected for same-subject adaptation. Every result includes the learned
decoder, identity decoder, support-removed, cyclically label-shuffled support, prototype, and fitted
L2 ridge-head controls, plus separately named Phase-A-vocabulary and Phase-B-training exposure
results, enrolled/unenrolled-candidate results, and per-subject results. Enrollment summaries treat
subjects as the independent unit and include paired subject-bootstrap intervals for each control
delta. The semantic evaluation reports subject-bootstrap intervals per deployment stream.
Development, sealed-test, and explicit custom runs use separate output filenames.

The standard predictor exposes one model-capacity setting. Its default optimizer batch is eight
independently randomized episodes with eight query executions apiece:

```bash
python -m training.evidence.train_patch_decoder --device cuda --evidence-budget 64
```

For a measured compute-shape experiment, use the explicit names
`--episodes-per-step` and `--queries-per-episode`; any positive count is accepted. There is no
ambiguous predictor `--batch` argument.

Retrieval K is derived from that budget. The final roster contains the highest-scoring unique rows
across query patches and learned subspaces; no label or window cap changes the ranking. Candidate
labels are task input. Every episode draws its own condition uniformly rather than following a
schedule: episode type (semantic zero-support, ordinary few-support, cross-subject few-support,
same-subject enrollment), `1`-`8` support examples per enrolled candidate, `2`-`16` candidate
labels, and how many of those candidates are enrolled — drawing all of them is full enrollment,
fewer is partial. One third of supported episodes relabel their candidates with episode-local
neutral aliases; those are always fully enrolled, because an alias name carries no information for
an un-enrolled candidate. Distractors are half nearest-confusable and half random in every episode.
The realized mix is reported in telemetry rather than prescribed. The
archive has one global upper budget; when the source corpus is smaller no rows are discarded. Its
active view rotates every 100 steps and caps each label at 16 windows: half the budget reserves
distinct executions from one rotating anchor subject for same-subject enrollment, while the other
half remains configuration- and subject-balanced.

Physical views are drawn 50/50 per episode (in expectation, not exactly per batch): stored clean frozen-encoder query/support vectors (or live
clean forwards when fine-tuning), or the full virtual-subject plus mild acquisition-augmentation
simulation. Validation evaluates every held-out
episode both ways and reports clean and augmented balanced accuracy separately.
Its default 39 fixed base episodes form the complete 13-recipe by three-transfer-fold Cartesian
product across held-subject, held-configuration, and jointly held queries.
Those internally subject-disjoint folds omit same-subject enrollment by construction; the external
enrollment evaluator measures genuine same-person support on source data that identifies people.

## Readout

Phase B has one readout: `model/evidence/relational_decoder.py`. It applies one self-attention
stack to `[candidate names; background label names; query patches; retrieved evidence rows]`.
Randomized episode-local coreference slots bind evidence to names without creating stable label
identities. A candidate logit is the readout on its token and nothing else — there is no closed-form
base term. The retrieval score enters as a shift-invariant relative prior on evidence-key attention:
`log_softmax(s/tau) + log(valid evidence count)`. Equal selected scores therefore add zero bias,
and a common cosine offset cannot starve candidate, label, or query tokens. This is the only
differentiable path from the candidate loss back to the retriever, since selection is a hard top-k
over frozen memory vectors. `candidate_logit_spread` in telemetry is the health signal to watch: it
collapsing to zero means the readout has become a constant predictor.

Every additive token ingredient — sensor signal, label text, role, coreference slot, source-window
group, physical time, and query-relative acquisition relations — is independently normalized to a
direction and multiplied by a learned positive scalar initialized to one before the token-level
LayerNorm. This prevents random structural embeddings from numerically drowning the sensor/text
content while preserving learnable relative importance. The seven scales and their gradients are
recorded in telemetry.

Evaluation artifacts carry separate evaluator-source and protocol fingerprints. The comparison
table refuses missing or mismatched identities even when support-free controls happen to agree.
The artifact also states which subject/configuration adaptation claims its realized cohorts can
support. In the current sealed roster, TNDA-HAR cannot provide genuine enrollment evidence and no
dataset provides sealed cross-configuration enrollment; those unsupported scopes are not headline
claims.

Retrieval is entirely learned and query driven. Learned projected query keys select hard top-k
memory rows, and the candidate loss is the only thing that trains them — it reaches the retriever
through the score's attention bias on the rows that were actually selected. Episode support identity
is never used to choose or append a forward row, and there is no auxiliary retrieval objective: an
earlier revision added a multiple-instance support-boundary loss over the whole memory, which was
introduced when the decoder structurally could not reach the retriever at all. That premise is gone,
and the objective was label-conditioned at train time and absent at evaluation.

The accepted consequence is that a support row outside the roster can never be promoted; the
retriever refines ranking within the neighbourhood it already reaches. That is the same limitation
RAG, Atlas and RePlug operate under.

## Telemetry

Predictor health telemetry is updated about once per minute. The parked confidence experiment uses
the same telemetry utility in its own directory if it is run later:

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

Predictor telemetry includes raw optimization CE plus candidate-count-normalized diagnostic CE,
training accuracy, per-curriculum-stratum metrics, component gradients, retriever/decoder gradient
ratio, clipping, evidence-role attention mass and entropy, evidence-pool concentration, assembled
support recall, raw pre-deduplication subspace overlap, fixed-canary roster churn, candidate logit
spread, throughput, and VRAM. The trainer fails before the optimizer update on non-finite gradients.

Both training stages write atomic resumable state beside their output as `*.last.pt`. Resume with
the same command and `--resume <path-to-last-state>`; bank identity and trajectory-affecting options
are checked before state is restored.

The optional end-to-end experiment is:

```bash
python -m training.evidence.train_patch_decoder --device cuda \
  --tokenizer-mode ema_finetune \
  --checkpoint training/tokenizer/outputs/phase_a_headline/best.pt
```

Fine-tuning starts after a fixed decoder warm-up so the relational decoder first develops a
nontrivial physical path. The active EMA key view is fully refreshed on the normal 100-step memory
cadence. Inference uses the saved EMA tokenizer.
