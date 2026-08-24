# Compact evidence engine

Status: **recording-level contextual scalar reranker implemented, trained, and evaluated as
PB-04. It underperforms direct 1-NN and is retained as an ablation, not the promoted readout.**

The promoted model uses the PB-04 encoder and direct 1-NN enrollment. This document records the
learned contextual scalar readout that was tested on top of that encoder. The previous patch-level
retrieve-mix-vote model is preserved by the Git tag `phaseb-vector8-vote-20260824`; its historical
results must not be combined with PB-04 results.

## Purpose

Phase B starts from a strong one-nearest-neighbor rule and asks one narrow question: can a small
learned function improve which memory recording is considered nearest? It does not learn a separate
classifier and it does not aggregate normalized votes from many rows.

The same mechanism handles two memory sources:

- the fixed training-corpus memory, whose rows carry canonical activity labels; and
- newly enrolled recordings, whose rows carry episode-local candidate identities.

## Prediction path

```
six-second query window -> encoder -> one pooled query vector
                                      |
                                      v
                         cosine against every memory window
                                      |
                          retrieve the nearest 64 rows
                                      |
                                      v
          one unordered set: candidates + query + retrieved evidence
             (signal, acquisition description, and evidence label)
                                      |
                                      v
                  one small bounded scalar correction per row
                                      |
                                      v
                         corrected nearest candidate
```

There is no evidence vote, candidate-logit residual, refined semantic vector, or hard
sensor-compatibility pool. Attention is used only to contextualize the retrieved set before one
scalar is read from each evidence row. The final rule remains corrected nearest neighbor.

## Recording rows

One query or memory row represents one six-second model window. Its signal vector is exactly the
encoder's `pooled` output, which is also used by the matched external 1NN control. Phase B does not
reconstruct this vector by averaging patch or sensor rows.

The row's acquisition description is the normalized mean of the frozen text embeddings for sensors
that are actually present. It preserves device, placement, modality, units, and gravity convention
without introducing source-specific numeric identifiers.

## Contextual scalar reranker

Raw cosine retrieves at most 64 memory recordings per query. One set-attention layer then receives:

- every candidate-label text as a candidate-role token;
- the query signal vector and acquisition description as a query-role token; and
- each retrieved signal vector, acquisition description, evidence-label text, standardized raw
  cosine score, and enrollment flag as an evidence-role token.

The set has no positional encoding, so candidate and evidence ordering cannot carry information. A
shared head reads only the contextualized evidence tokens and emits one scalar per row. It cannot
modify vectors or emit a candidate score. The correction is bounded by a hyperbolic tangent and a
learned gain, then added to raw cosine. The gain starts at `0.05` cosine units and cannot exceed
`0.5`. The output head has a small nonzero initialization, so every mixer component receives a
gradient from the first step while the initial correction remains small.

## Candidate scoring

An enrolled row is bound exactly to its episode-local candidate. A corpus row is associated with the
candidate through the cosine similarity between its canonical activity text and candidate text. This
semantic value is a non-positive score offset: an exact label match costs zero and less similar text
is penalized.

For each candidate, the forward score is the largest corrected score among compatible memory rows.
The final prediction is the candidate with the largest such score. This is corrected 1NN, not voting.

The maximum is discrete. During training, its forward value remains the exact maximum, while its
backward derivative is taken from a smooth maximum with temperature `0.05`. Consequently deployment
and training make the same decision, but every finite memory row can receive learning signal. The
smooth rule is not used to produce deployment scores.

## Baseline preservation

Every forward pass reports both learned logits and the unchanged raw-cosine logits. Non-shortlisted
rows retain encoder gradients through the all-row smooth-maximum surrogate, but only shortlisted
rows receive mixer corrections. Training uses
candidate cross-entropy plus a no-regression penalty when the learned path gives the target more loss
than its detached reference:

- k=0 uses raw corpus nearest-neighbor scoring;
- k>0 uses enrolled 1NN where the true candidate and at least one competitor have support, otherwise
  it uses the raw combined-memory result.

This penalty encourages, but cannot mathematically guarantee, improvement on unseen data. Validation
therefore reports learned-minus-raw and learned-minus-enrolled-1NN curves, and external evaluation
reports both `full_raw_1nn` and `full_reranked_1nn` from the same memory.

## Episode training

Each optimizer step contains eight independent episodes. Every episode has its own candidate roster,
queries, support bindings, distractors, and 512-recording memory. Up to four candidate labels supply
four query recordings each; remaining candidates are distractors. Candidate counts cycle through
2/4/8/16 and support counts through 0/1/2/4/8/16.

Enrollment is partial and random. Support is sampled from different acquisition streams when the
data permit it and is never manually inserted into the cosine shortlist. Corpus memory and enrolled
rows are encoded by the current encoder in end-to-end runs.
Signal augmentation and arbitrary label aliases are disabled by default.

Episodes with the same candidate count are vectorized together. Padding is limited to query and
memory row counts; masks prevent padded rows from entering retrieval or correction. On an RTX 4090,
a synthetic worst-case evidence-engine pass at the default shape (8 episodes, 16 queries, 512 memory
rows, 16 candidates, top 64) measured 7.7 ms for forward plus backward and 0.22 GiB peak allocated
memory. This excludes the encoder and data pipeline. The three-step real-corpus smoke test completed
without a non-finite loss or gradient.

Core episode plans are deterministic and cached under
`$XDG_CACHE_HOME/halo/episode_plans` (normally `~/.cache/halo/episode_plans`). The cache key includes
the exact corpus rows, split metadata, episode settings, schedules, seed, NumPy version, and planner
source, so changed inputs rebuild rather than reuse stale plans. Cold builds use eight processes; a
16,384-episode real-corpus benchmark took 14.6 seconds cold and 0.05 seconds warm.
`HALO_EPISODE_PLAN_CACHE_DIR` may relocate the machine-local cache.

## Active command

```bash
python -m training.tokenizer.pretrain_episodic \
  --random-init \
  --phase-b-regime unified \
  --steps 35000 \
  --out training/tokenizer/outputs/<run-name>
```

The defaults shown above are serialized in the checkpoint. `source.patch` in each run records the
exact source difference used for that experiment.
