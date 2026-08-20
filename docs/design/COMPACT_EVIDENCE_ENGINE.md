# The compact evidence engine

Status: **implemented end to end; model, trainer, checkpoint reload, and real-corpus smoke tested.**
Date: 2026-08-20

## What changed and why

Two constraints drove this redesign.

**Parameter parity.** HALO is compared against models whose sensor towers run from 63k (LIMU-BERT,
CrossHAR) to 5.2M (UniMTS). The previous HALO carried 6.0M learnable parameters — encoder 4.34M,
query head 263k, evidence mixer 1.39M — which is larger than every baseline it beats and larger than
several it loses to. A win at that size is a weaker claim.

**More places learnable, not bigger learnable places.** The pieces that were *stipulated* — the hard
modality/gravity compatibility filter, the fixed cosine retrieval rule, the fixed vote temperature —
were the parts that could not improve. The pieces that were *large* — a 3.16M dual-branch trunk, a
263k query head, a 132k descriptor head with no path to any loss — were the parts spending capacity
without earning it.

The redesign inverts both: every step from patch to prediction is now learned and trained end to
end, at about a fifth of the parameters.

## The pipeline

```
query window ─┐
              ├─ filterbank → sensor fold → description conditioning → temporal self-attention
memory bank ──┘                                              ↓
                                        feature (d) + description (384) per sensor row
                                                             ↓
              PairScorer( q_feat, q_desc, m_feat, m_desc ) → (Q, M) scalar scores → top-k
                                                             ↓
              EvidenceMixer: self-attention over  [ candidates | query | k retrieved rows ]
                                                             ↓
                    log evidence weights (Q, k, C)  +  refined label vectors (Q, k, 384)
                                                             ↓
              vote: softmax over rows × cosine against candidate names → (Q, C) logits
```

### 1. Feature extractor — full front end, temporal-only attention

`model/tokenizer/transformer.py::TemporalTrunk`, selected by `SetTokenizerEncoder(trunk="temporal")`.

The physical-Hz filterbank, xyz sensor fold and sensor-description conditioning are unchanged. The
trunk is now a few pre-norm layers of temporal self-attention with physical-time RoPE, GELU
feed-forward at `ffn_mult × d_model`, residual on both sub-layers, and a final norm.

The cross-sensor attention branch is gone. It was ~40% of the trunk and fused sensors at a point
where the model knows nothing about them beyond their descriptions. That fusion now happens in the
mixer, where retrieved rows from one recording share a co-membership group and sit beside their
labels and descriptions.

This buys a property the dual-branch trunk never had: **each sensor is encoded in isolation**, so a
wrist accelerometer's retrieval feature is identical whether or not a gyroscope was on the same
device. That is the invariance the cross-configuration claim wants. It also retires the
`--retrieval-depth isolated|full` question — retrieval reads the full trunk depth *and* is
sensor-isolated, where before those were alternatives.

`descriptor_prediction=False` drops the 132k descriptor head, which the reference recipe disabled
and froze on every run.

### 2. Retrieval — a learned pair scorer

`model/evidence/retrieval_scorer.py::PairScorer`.

```
score = f( query feature, query description, memory feature, memory description )
```

`H` factored bilinear heads over `[feature ; projected description]` on each side, then a small MLP
over the `H` head scores plus the raw feature and description cosines. Query and memory get separate
projections — "is this row useful evidence for that query" is not symmetric.

This replaces **both** the hard compatibility filter and the fixed cosine. The physics the filter
enforced is expressible from the inputs the scorer already has: the sensor description string carries
modality and gravity convention verbatim ("a watch accelerometer on the left wrist; gravity removed;
recorded alongside a gyroscope"). Learning it from language rather than reading it from a metadata
table is the HALO claim applied to HALO's own retrieval step, and it extends to acquisition
configurations that were never in the corpus.

Whether it *was* learned is a measurement, not an assumption: `physics_violation_rate` reports how
often the selected top-k crosses a modality or gravity boundary. It starts near 50% on random
features. Falling toward zero is evidence the descriptions carry the constraint; staying high
alongside good accuracy would be evidence the constraint mattered less than was assumed. Either is a
result.

`base_gain` (init `1/0.07`) multiplies the raw feature cosine and `residual_gain` (init 0.02) the
learned head, so **at step 0 the score is the retired cosine rule to within 2%** — later differences
are attributable to training, not to a different starting point. Neither is frozen; both are logged.

There is no `-inf` anywhere. An unreachable row is one no gradient can recover.

### 3. Mixing — one sequence per query

`model/evidence/evidence_mixer.py::EvidenceMixer`.

```
per query: [ CAND_1..CAND_C ][ QRY QDESC ][ EV_1..EV_k ][ EDESC_1..EDESC_k ][ ELBL_1..ELBL_k ]
```

Six roles and two identity channels: `slot` for coreference ("these tokens refer to the same
concept") and `group` for co-membership ("these came from the same recording"). Both are permuted
every episode so an id can only ever mean a relation, never a memorable label.

The previous mixer built one sequence per *episode* containing every query alongside the whole bank
(~1.8k tokens), which let one query's refinement depend on another query's evidence — something that
will not exist at deployment, where a query arrives alone. Per-query sequences are `C + 2 + 3k` — 202
tokens at C=8, k=64 — and cheaper: attention is quadratic in length and linear in batch.

The retrieval score enters as a standardised additive attention bias on the evidence tokens, not as
an input feature, so no later layer can launder it into a path that reaches a candidate without
passing through the row it describes.

### 4. Readout and vote

`EvidenceMixerConfig.readout` is one field with two values, and they are **alternatives, not a
feature and its extension**. Mixing is identical either way; what differs is what attention emits.

- **`"weights"`** (368,774) — a scalar per (query, row, candidate), added to the retrieval score in
  log-weight units: `log w = retrieval score + correction_gain ×
  (evidence_candidate_form + query_evidence_candidate_form)`, two low-rank bilinear forms both
  routed through an evidence row. There is deliberately **no query × candidate form**: that is
  a classifier bypassing evidence, and the channel through which an earlier decoder collapsed below
  the chance floor. The readout cosine stays at the text encoder's own geometry.
- **`"semantic"`** (401,798) — no scalar at all. Attention displaces each retrieved row's label
  vector inside the frozen text space, and the evidence weight is the retrieval score alone.

`"semantic"` is strictly more expressive per row — a 384-d displacement against a scalar — and it
subsumes reweighting: a vector pushed equidistant from every candidate contributes equally to all of
them, which is a down-weight. It also changes the *shape* of the computation, which is the reason to
measure rather than assume. With the weight shared across candidates the readout collapses:

```
logit_c = Σ_m w_m ⟨ℓ̃_qm, t_c⟩ = ⟨ Σ_m w_m ℓ̃_qm , t_c ⟩ = ⟨v_q, t_c⟩
```

The episode becomes **one vector scored against every candidate** — a CLIP-shaped model in which no
candidate can consult different evidence from any other. The rectification in `vote` is the only
thing standing in the way of that factorisation, and on this corpus's 166-label vocabulary it fires
on **0.6%** of (row, candidate) pairs (mean pairwise label cosine 0.256), so it does not rescue the
structure.

**Where this bites is k=0, not k≥1** — the reverse of what it looks like at first glance. With rows
enrolled, the mask that keeps an enrolled row out of every candidate but its own already makes the
weights candidate-dependent, so `"semantic"` recovers per-candidate evidence selection for free. With
nothing enrolled the mask is vacuous, the collapse is exact, and the arm reduces to a single pooled
query embedding. That is the zero-shot cell, which is the one this design exists to protect and the
one the retired decoder fell through.

`"weights"` is therefore the default: it is 16,513 parameters — 1.5% of the model — for a structure
that does not collapse anywhere. `"semantic"` remains a real arm to measure, not a discarded idea.

The vote is a softmax over retrieved rows times a rectified cosine against candidate names — **one
path for every row**. The old `identity = 1` special case for enrolled rows is deleted: an enrolled
row's label vector *is* its bound candidate's text, so the ordinary cosine returns exactly 1 and the
branch was numerically redundant. It also had to go, because a constant is unreachable by the
`"semantic"` readout — support strength would have had no learned control at all under that arm.
Corpus rows vote the ConSE bridge, which is the only mechanism available at k=0.

**No per-candidate parameters exist anywhere in the model.** An unseen candidate is scored by exactly
the same operation as a seen one, which makes closed-vocabulary collapse structurally impossible
rather than merely discouraged. `tests/test_engine.py` pins this by permuting the candidate set and
asserting the output permutes with it.

### 5. The memory bank

`training/tokenizer/episodic.py::BankSpec`, `build_bank_plan`.

512 recordings, the same construction in training and deployment.

- **Fixed size.** Support displaces background rather than adding to it, so retrieval always chooses
  from the same-size pool and step cost does not depend on k.
- **Stratified.** Half the draws go stream-first (uniform over acquisition configurations, then over
  that stream's labels), half label-first. Measured on a synthetic corpus where one stream holds 100×
  the windows: naive uniform-over-windows gives stream entropy **0.048** and reaches 4 of 6 streams;
  this sampler gives **0.910** and reaches all 6, at label entropy 0.985.
- **Excludes the episode's candidate labels.** Not a handicap — the deployment condition. The bank is
  drawn from *training* labels and the query carries an unseen one, so no deployed bank row can
  already be named the answer. Training against a bank that contains the answer teaches lookup inside
  a closed vocabulary and makes the k=0 cell measure something that will not exist when it matters.

Episodic variation is unchanged in spirit: support count k varies per episode, candidate count varies
2–16, enrollment is partial, and a fraction of episodes replace canonical names with neutral aliases.
Adaptation is only ever as good as it was taught.

## Parameter budget

`python -m eval.model_budget`

| size | d | trunk | mix | front end | trunk | scorer | mixer | **TOTAL** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 96 | 2 | 2 | 149,376 | 149,760 | 86,754 | 224,390 | **610,280** |
| **small (committed)** | **128** | **3** | **2** | 244,224 | 397,696 | 115,458 | 368,774 | **1,126,152** |
| medium | 160 | 3 | 2 | 361,600 | 620,000 | 144,162 | 547,974 | **1,673,736** |
| wide | 192 | 4 | 3 | 501,504 | 1,188,480 | 172,866 | 1,059,014 | **2,921,864** |

**Committed 2026-08-20: `small`.** `EngineConfig()` defaults are exactly this — d_model 128,
4 heads, FFN 256, 3 trunk layers, 2 mixer layers, top_k 64, n_groups 96, readout `"weights"`
(1,126,152; `"semantic"` is 1,159,176).

The two readouts differ by 33,024 in the mixer. The frozen text tower (all-MiniLM-L6-v2, 22,713,216)
is shared, never trained, identical at every size, and reported separately — UniMTS carries a 63.4M
frozen CLIP text tower on the same footing.

| baseline | sensor tower | frozen text |
|---|---:|---:|
| LIMU-BERT | 62,646 | — |
| CrossHAR | 62,646 | — |
| UniMTS | 5,189,956 | 63,428,097 |
| HARNet5 | not measured (checkpoint not resident) | — |
| NormWear | not measured (checkpoint not resident) | TinyLlama |
| ImageBind | 1,200,786,990 | — |
| **previous HALO** | **5,996,902** | 22,713,216 |

At `small`, HALO is **0.22× UniMTS's sensor tower** and **19% of the previous HALO**.

## Cost

Phase-B forward+backward on an RTX 4090. A 512-recording bank is about **4,900 retrieval rows**:
6-second windows at 1-second patches, ~1.6 sensors per window, and a row is one (window, patch,
sensor) triple.

| shape | ms |
|---|---:|
| one episode (Q=26), 1,000-row bank | 7.98 |
| one episode (Q=26), 4,900-row bank | 8.01 |
| four episodes fused (Q=104), k=32 | 9.35 |
| four episodes fused (Q=104), k=64 | 11.33 |

The stack is **launch-bound, not compute-bound** — a 5× larger bank costs little, and doubling k
costs 21% only on the fused path. The current trainer fuses all episode windows into one encoder
forward, which is the expensive part, then calls the engine separately for each independent episode.
Fusing variable candidate sets into one padded engine call remains a throughput optimization; it is
not required for mathematical parity and must preserve episode isolation when implemented.

### The scorer

The pair scorer is fully batched — two side projections, one `bmm` over the interaction heads, two
cosine matmuls, one pointwise MLP, and no Python loop anywhere. Profiling at Q=104 against a
4,900-row bank found the cost was not in the parallelism but in one place:

| | ms |
|---|---:|
| **pair-head MLP over (Q, M, 10)** | **0.641** |
| descriptor cosine matmul | 0.041 |
| feature cosine matmul | 0.033 |
| top-k(64) over (Q, 4900) | 0.032 |
| interaction heads (bmm) | 0.014 |
| descriptor projection | 0.028 |

The MLP was **71% of the forward pass** because it writes, reads and writes back a (104, 4900, 32)
hidden activation — ~65 MB of traffic for 45 MFLOP of arithmetic. It is bandwidth-bound, so
`torch.compile` (0.664) and a manual `addmm` chain (0.665) changed nothing, and shrinking the hidden
width would only trade capacity for bandwidth.

Running **that block alone in bfloat16** is 8.6× (0.679 → 0.079 ms). The precision argument is
bounded: its output is scaled by `residual_gain` and added to a score of order 1, the base cosine
and the final sum stay in float32, and top-64 selection is unchanged on ≥63 of 64 rows even at a
gain of 5 — 250× the initial gain. CPU stays in float32 so the test suite remains an exact
reference.

Net: scorer forward 0.897 → **0.327 ms**, forward+backward 2.228 → **1.269 ms**.

## Guarantees under test

`tests/test_engine.py`, 32 tests:

- the budget formula matches a real block; sets are read as sets (permutation equivariance);
  padded tokens cannot influence real ones; additive channels enter as directions
- the temporal trunk encodes each sensor in isolation
- the scorer starts at the retired cosine rule and reads all four of its arguments
- an enrolled row cannot vote for another candidate; a candidate with no admissible evidence scores
  zero rather than NaN
- every learnable parameter receives gradient in both committed variants — this caught a
  structurally dead output bias on the scorer (a constant on every score cancels in the vote's
  softmax and again in the mixer's standardisation)
- candidates are scored by their text, not their index
- top-k is the same size in training and evaluation; the batched interaction heads match their
  einsum reference; the bfloat16 pair head does not change what gets retrieved
- identity ids are redrawn every episode and their vocabularies are checked, not wrapped
- the bank covers streams that window counts would bury, holds its size as support grows, never
  contains the episode answer, and carries no duplicate row

## Removed

- `mixer_vote` (91 lines) in `admissible_retrieval.py`, replaced by `engine_vote` (deployment is now
  the ordinary forward pass — there is no deployment-only rule to keep in sync)
- the old `EvidenceMixer` and `tests/test_evidence_mixer.py` (surviving properties ported)
- `--mixer-pool` (top-k lives in the checkpoint's engine config), `--no-mixer` → `--no-engine`,
  `--mixer-checkpoint` → `--engine-checkpoint`
- `SensorRows` moved to `model/evidence/rows.py`: it is part of the model's input contract, not of a
  training-side scoring module

## Remaining experiments

1. **Train the compact Phase-A checkpoint.** The implementation and checkpoint reconstruction are
   tested, but a dual-trunk Phase-A checkpoint is intentionally rejected by this Phase-B trainer.
2. **Measure both readouts.** `weights` is the committed default; `semantic` remains an ablation.
3. **Fuse engine calls if profiling warrants it.** The current four calls keep episode boundaries
   explicit and correct. A padded fused call is an optimization, not a model change.

## Trainer profile (2026-08-20, `--profile-steps`)

Phase breakdown at defaults (bank 512, 4 episodes/step, bf16 autocast, full corpus, RTX 4090):

| phase | ms | share |
|---|---:|---:|
| data wait | 0.9 | 0.8% |
| encoder forward (~550 windows) | 10.7–29.0 | ~15% |
| episodes (4 × engine fwd + loss) | 33.7–45.5 | ~35% |
| backward | 40.5 | ~40% |
| optimizer | 2.1–3.3 | ~3% |
| **total** | **~100** | |

The step is **CPU-dispatch-bound**: 335 ms self-CPU vs 111 ms self-CUDA over 3 profiled steps — the
GPU idles about two-thirds of the time. Consistent with that, per-episode cost is **flat in batch
size**: 23.9 / 26.6 / 24.8 / 25.2 ms per episode at 2/4/8/16 episodes per step. Batching episodes
does not amortize, because each episode is its own sequential engine call and its own backward
subgraph. `--episodes-per-step` therefore stays 4 on optimization grounds (gradient averaging), not
speed grounds; the loader is a non-issue at 0.8%.

**Bank sampling was 24.0 ms per bank and is now 3.2 ms (7.5×)** — `np.isin` against the constant
blocked-label set ran per window draw (46% of the cost), subject lists were re-sorted per draw, and
`rng.choice` per scalar carries ~5 µs against `rng.integers`-and-index. All hoisted; the sampling
distribution is unchanged and every bank invariant (distinct, full, exclusion) holds. Plan
construction for a 24k-episode run: ~8 min → ~1.1 min.

The remaining known lever is fusing a step's episodes into one padded engine call — the episode +
backward phases are 75% of the step and mostly launch overhead, so the standalone-engine
measurements suggest roughly 2–3× is available. Deferred deliberately: it needs padded candidate
sets and block-diagonal memory handling, which is refactor risk ahead of the first real training
run rather than after it.
