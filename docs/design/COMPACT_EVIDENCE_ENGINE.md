# Compact evidence engine

Status: **scalar evidence reranker implemented and smoke-tested; full training and held-out
evaluation are pending.**

This is the active Phase-B design. The earlier attention mixer emitted a free residual for every
candidate and performed substantially worse than one-nearest-neighbor on the same HALO features.
Those completed results remain in [`../results/RESULTS.md`](../results/RESULTS.md). The current design
keeps the useful retrieval rule fixed and gives the learned model only one narrow responsibility:
change how much each retrieved evidence row is trusted.

## Purpose

Phase B turns one shared sensor representation into a classifier that can use ordinary corpus memory
and newly enrolled labeled examples. An enrolled example is an ordinary memory row carrying an
episode-local candidate label. It has no dedicated lookup path and is not manually inserted into the
retrieved set.

The experiment asks one direct question: can contextual reasoning rescore retrieved evidence well
enough to beat the fixed retrieval vote?

## Prediction path

```
all patches and sensors in one query recording
                    |
                    v
fixed cosine score against every memory row
                    |
        +-----------+--------------------+
        |                                |
        v                                v
full-memory base vote       global top-64 evidence rows
                                         |
                         [candidate label tokens]
                         [all query row tokens]
                         [retrieved evidence row tokens]
                                         |
                              one set-attention layer
                                         |
                       one bounded scalar per evidence row
        |                                |
        +------------ fixed vote --------+
                         |
                  candidate prediction
```

Separate query recordings never attend to one another. All patches and sensors belonging to one
recording are presented together, along with its shared evidence shortlist and candidate set.

## Retrieval and memory

The default score is normalized feature cosine divided by `0.07`. There is no learned pair scorer in
the active model. The Phase-A encoder may be frozen for the clean Phase-B diagnostic or trained end
to end in a separate experiment.

The full memory remains in the differentiable vote. With an unfrozen encoder this gives every score
path gradient; with the recommended frozen-encoder diagnostic it still keeps the deployed prediction
rule honest, while only shortlisted rows pass through trainable reranker parameters. The top 64 rows
under the maximum score assigned
by any query row form one recording-level shortlist for contextual rescoring. Rows outside that
shortlist keep their original cosine score rather than disappearing.

There is deliberately no hard accelerometer/gyroscope or gravity-convention filter. Retrieval
compares learned latent vectors rather than raw physical units, and the full-memory design is meant
to leave every row reachable. Modality, gravity convention, placement, and device remain visible in
the sensor descriptions. Telemetry reports how often the shortlist crosses modality and gravity
boundaries so the behavior can be evaluated rather than assumed.

## Tokens seen by the reranker

Each item is one fused token, not a collection of separate metadata tokens:

- **Candidate token:** candidate text and a candidate role embedding.
- **Query token:** Phase-A patch/sensor vector, its sensor description, and a query role embedding.
- **Evidence token:** retrieved patch/sensor vector, sensor description, evidence label text,
  standardized cosine score, evidence role embedding, and source-recording group embedding.

Role embeddings make the three token types explicit. Recording-group embeddings tell the model
which retrieved patches and sensors came from the same source recording. Candidate order, query-row
order, and evidence-row order carry no positional meaning.

Enrolled evidence carries the exact episode-local candidate text. That shared text is the
coreference link between an enrolled row and its candidate. The new reranker does not use random
candidate-slot embeddings; removing them makes candidate permutation equivariance structural and
removes a redundant source of episode noise.

## Scalar correction

The contextualized hidden state of each evidence token passes through one shared scalar head:

```
raw_correction = shared_linear(contextualized_evidence_row)
gain           = 2.0 * sigmoid(learned_gain_logit)
correction      = gain * tanh(raw_correction)
new_score       = cosine_score + correction
```

The default gain starts at `0.05` and can never exceed `2.0`. The scalar head uses a small nonzero
initialization (`std=0.001`). Consequently the initial model is extremely close to the fixed vote,
but every input projection, role/group embedding, attention parameter, output head parameter, and
gain parameter receives gradient on the first backward pass.

The reranker cannot emit candidate logits. It can only change evidence-row scores. Candidate labels
still enter the final decision through the fixed vote, preventing the direct query-to-candidate
classifier shortcut available to the retired mixer.

## Fixed evidence vote

For each query patch/sensor row:

1. Normalize all memory-row scores with one softmax.
2. Give an enrolled row a one-hot vote for its assigned candidate.
3. Let a corpus row distribute its vote using normalized positive text similarity between its
   canonical label and the candidate labels. If no similarity is positive, use a uniform vote.
4. Sum row weight times row vote over memory.
5. Average the resulting candidate distributions over all query rows in the recording.

The output is a candidate probability distribution. Candidates compete for one finite quantity of
evidence, and the learned component cannot bypass this operation.

## Training regimes and objective

The active experiment trains two reranker checkpoints from the same frozen encoder:

- `zero-shot`: only k=0 episodes; the detached semantic full-memory vote is its reference.
- `enrollment`: only k=1/2/4/8/16 episodes; patch-level enrolled 1NN is its reference whenever the true
  candidate and at least one competing candidate have support. The semantic vote remains the
  reference for queries whose candidate has not been enrolled.

The 1NN reference uses the strongest cosine match between any query patch/sensor row and an enrolled
row for each candidate. It is an internal control for the engine's own retrieval granularity, not the
separate pooled-execution 1NN reported in the external baseline table.

This masked definition is required because enrollment is intentionally partial: a candidate with no
support has no valid 1NN prediction. Giving missing candidates a fabricated neighbour would make the
comparison easier but meaningless.

The primary loss is candidate cross-entropy. The same episode also computes the fixed retrieval
controls. A no-regression term penalizes cases where the learned path gives the true candidate a
larger loss than its detached regime reference:

```
task       = CE(reranked_prediction, target)
reference  = CE(detach(regime_reference), target)
regression = mean(max(0, task_per_query - reference_per_query))
loss       = task + regression
```

This encourages improvement but does not guarantee it. The validation report therefore carries the
learned, semantic-vote, and valid enrolled-1NN curves separately. A head is accepted only if it beats
the appropriate fixed control from the same encoder checkpoint.

## Episodes

- Eight independently constructed episodes share one encoder forward and one masked, vectorized
  evidence-engine call per optimizer step. Episodes share tensor shape only; their candidates,
  memory, retrieval, attention, vote, and loss remain independent.
- `C` is the complete label roster presented to the classifier and is sampled from 8, 16, 32, or
  64 during training. Four labels from that roster contribute four distinct query executions each;
  the remaining labels are distractors that still compete in the loss but require no sensor encode.
  This separates decision difficulty from encoder batch size and lets a label-poor sensor stream
  host a large-roster episode without pretending it recorded every distractor activity.
- The zero-shot head uses k=0; the enrollment head uses 1, 2, 4, 8, or 16 independent executions
  per enrolled candidate.
- Support and query executions are disjoint; the default split is also stream-disjoint.
- Every query in an episode shares one acquisition stream. Support is excluded from that stream,
  preventing device identity from revealing the answer.
- Memory contains 512 windows. Enrollment displaces background rather than increasing memory size.
- When `C * k` exceeds 512, random partial enrollment is capped to the number of complete k-shot
  candidate groups that fit. The requested k is never silently truncated.
- Background memory excludes the episode candidate labels.
- Random label aliases and signal augmentation remain disabled by default.

The primary external evaluation does not resample `C`: its candidate roster is the complete set of
eligible labels defined by each test dataset. A separate large-roster stress test may add fixed
distractors, but it is not substituted for the dataset-native result.

For the isolated Phase-B experiment, load the selected shared encoder checkpoint and pass
`--freeze-encoder`. End-to-end random initialization remains supported as a separate representation
experiment, but it answers a different question.

## Telemetry

Training records the fixed and learned loss and accuracy, learned-minus-base validation curves, and
prediction changes. The reranker additionally reports:

- gradient norms for its input/fusion modules, attention stack, and scalar output separately;
- correction gain, mean, spread, maximum, and correlation with the original cosine score;
- evidence-weight shift and effective evidence-row count before and after correction;
- enrolled-evidence weight and correction before/after rescoring;
- shortlist bank coverage, enrolled share, support share, modality crossing, and gravity crossing.

These measurements distinguish useful contextual rescoring from merely sharpening cosine, always
favoring enrolled rows, collapsing onto a few rows, or leaving part of the model dormant.

The active same-`C` vectorized path was profiled on the RTX 4090 with real 512-window banks:

| episodes per step | step time | episode throughput | peak allocated VRAM |
|---:|---:|---:|---:|
| 4, sequential reference | 67.4 ms | 59/s | 1.03 GiB |
| 4, vectorized | 38.8 ms | 103/s | 1.23 GiB |
| **8, vectorized default** | **55.4 ms** | **144/s** | **2.33 GiB** |
| 16, vectorized | 103.9 ms | 154/s | 4.54 GiB |

Eight is the throughput knee: it doubles independently sampled episodes relative to the previous
recipe while remaining faster per optimizer update. Sixteen consumes almost twice the time for only
about seven percent more episode throughput. The sequential path remains a debug oracle, and tests
require batched and sequential logits, fixed controls, selected rows, parameter gradients, and
query/memory feature gradients to agree.

## Current size and verification

At width 128, three temporal encoder layers, and one reranker layer:

| component | trainable parameters |
|---|---:|
| encoder front end | 244,096 |
| temporal trunk | 397,696 |
| fixed cosine scorer | 0 |
| scalar evidence reranker | 228,108 |
| **end-to-end total** | **869,900** |

With the encoder frozen, Phase B trains only the 228,108 reranker parameters. The frozen
all-MiniLM-L6-v2 text tower is reported separately and is never updated.

Verification on 2026-08-24:

- 112 focused engine, episode, data-plumbing, and gradient tests pass; the repository-wide suite
  has 740 passing tests.
- Candidate permutation equivariance is tested with the active nonzero reranker path.
- Every reranker parameter receives nonzero gradient on the first backward pass.
- Separate real-data RTX 4090 zero-shot and enrollment smoke runs completed loading, validation,
  optimizer steps, telemetry, checkpoint save, and strict checkpoint reconstruction. Their saved
  encoder state hashes are identical.
- At step 1, the mean absolute logit change was about `6e-6`; the reasoner gradient norm was finite,
  and no clipping was required.

These checks establish implementation validity, not model quality. The next result must compare the
new scalar reranker against its fixed vote and patch-level 1-NN on the unchanged development
protocol before running the sealed test suite.
