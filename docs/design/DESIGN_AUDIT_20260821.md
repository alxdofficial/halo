# Design audit — compact evidence engine, 2026-08-21

Requested before any further long training. Scope: every stage of the model and trainer as they
stand on `phase-b-diagnostics`, what is *verified* (test or measurement), and what is an *open
risk*. Companion documents: `COMPACT_EVIDENCE_ENGINE.md` (architecture),
`../results/PHASE_B_DIAGNOSIS_20260820.md` (why training plateaus).

## The model as it stands today

```
window ── filterbank ── sensor fold ── descriptor conditioning ── temporal trunk (3 layers)
                                                                        │  128-d row per (patch, sensor)
retrieval    score = cos(patch_q, patch_m) / 0.07          ← FIXED, zero parameters (2026-08-21)
selection    hard top-64 per query row
mixing       set attention over [candidates | query | top-64] → scalar per (row, candidate)
vote         softmax over rows (+ null row) × rectified text cosine → logits
             vote_scope: "topk" (default) or "bank" (all rows vote; mixer corrections scattered)
```

Learnable: encoder 641,792 + mixer 364,678 = **1,006,470**. The learned pair scorer (115k) is
retained behind `PairScorerConfig(learned=True)` but off — measured at exactly +0.0000 in
isolation.

## Stage-by-stage: verified vs open

### Encoder
| property | status |
|---|---|
| sensor isolation (a row is independent of co-resident sensors) | ✅ test |
| residual branches start small (`1/sqrt(2·n_layers)` scaling) | ✅ fixed 2026-08-20, test; was \|Δx\|/\|x\| = 2.55 at layer 0 |
| representation improves under training (kNN 0.782 → 0.875) | ✅ measured |
| **cross-config activity matching flat at ×1.3 from init through 6k steps** | ⚠️ **the** open problem — the objective never rewards that axis |
| descriptor conditioning is *not* the cause of config-sorting (masking it leaves cross-config lift unchanged) | ✅ measured (inference-masking caveat noted) |

### Retrieval
| property | status |
|---|---|
| plain cosine == learned scorer on the end task | ✅ ladder (+0.0000 / +0.0085) |
| fixed path returns fp32 under autocast (feeds softmax + top-k) | ✅ fixed 2026-08-21 — found *by* making it the default |
| modality/gravity constraint | ⚠️ **unguarded**: the hard filter is gone and the learned scorer that had re-learned it is off. Cosine has no mechanism for it. `physics_violation_rate` stays in telemetry as the watchdog; violations did not measurably hurt the ladder rungs that ran without any constraint |
| infeasible top_k fails at parse time, not step 21 | ✅ guard added (k=512 OOMed a 24 GiB card; ~1,550-token sequences retained for backward) |

### Mixer
| property | status |
|---|---|
| permutation equivariance, padding isolation, per-episode identity redraw, no per-candidate parameters, both readouts fully live | ✅ tests (700) |
| attention is worth ~+0.014 (replicated, same-seed pairs); depth beyond 2 layers worth nothing | ✅ measured |
| what step 6 actually learns is enrolled-row coreference (corr +0.45), i.e. it compensates for retrieval rather than "mixing information" | ✅ measured — a design smell, not a bug: the fix is upstream |
| content is 24.3% of each token; identity channels (redrawn per episode) are half of it | ⚠️ measured, but the obvious remedy **failed**: `--identity-gain 0.3` scored 2.5σ *worse*. Flag retained, default unchanged |

### Vote
| property | status |
|---|---|
| null calibration row (k=1 enrollment stays learnable), FP32 island, finite under all-enrolled / none / ±1e±3 log-weights | ✅ tests + numerics battery |
| `vote_scope="bank"`: 100% gradient coverage (vs 55% topk on the test bank), +2.9 ms, +1.1 GiB | ✅ measured |
| **bank voting does not improve the final model** — raw best within 0.003 of topk on matched seeds; the "+0.092 paired gain" was an artifact (see Methodology) | ✅ measured, 2 seeds |

### Trainer / episodes
| property | status |
|---|---|
| provenance lift pinned at 0.0000 (`disjointness=stream` + `shared_query_stream`) | ✅ printed at startup, re-verified after the planner fix |
| **candidate counts were collapsing**: stream drawn first → uniform {2,4,8,16} became 45/30/17/8%, chance 32.8% not 22.2%, half of training binary | ✅ **fixed 2026-08-21** (count drawn first, then a capable stream); 22/23/25/30%, provenance still 0.0000, 54/56 streams still host; test added. **All previously reported selection scores sat on the inflated floor** — cross-run comparisons remain internally consistent (same floor everywhere), absolute values do not |
| validation draw pinned to `--val-seed`, decoupled from `--seed` | ✅ verified: same seed ⇒ identical step-0; residual cross-seed step-0 spread is encoder-init variance, which the paired statistic handles |
| banks are per-episode (4 × 512/step): 2,080 windows encoded to score 32 queries | ⚠️ 98% of encoder compute builds banks. Deliberate (per-episode candidate exclusion), but the ratio is worth revisiting now that VRAM is declared a non-constraint |
| episode mix (k, alias, C) is random per step, not balanced | ⚠️ by design ("randomness teaches robustness"), but a step can contain no k=0 episode; validation is schedule-pinned, so measurement is unaffected |

## Methodology rules (each one was violated once this week, at cost)

1. **Between-run noise is 0.068** if the validation draw floats with the seed; the concept holdout
   dominates. Pin `--val-seed`. Fixed.
2. **The paired gain (best − own step-0) is valid only when the step-0 function is identical across
   arms.** An arm that changes the deployed rule (e.g. `vote_scope`) changes its own baseline;
   comparing gains then rewards *starting worse*. The +0.092 for bank voting was exactly this.
   For architecture changes, compare raw scores at matched seed on the pinned draw.
3. **A `+0.0000` gain means the best checkpoint was step 0** — which can mean "crashed", not
   "ineffective". topk512's zeros were an OOM at step 21. Check the log's tail before interpreting.
4. Chance depends on the C-distribution. After the planner fix it is ~21.5%; before, 32.8%.

## Where the accumulated evidence points

Every learnable stage has now been isolated, and the ordering is stable: the step-6 readout carries
~80% of the gain; attention adds a little; learned retrieval and full-bank gradient flow add
nothing; the encoder trained jointly is neutral-to-negative; nothing moves k=0. Meanwhile the
encoder's cross-configuration matching — the capability retrieval actually needs — sits at ×1.3
untrained *and stays there through training*, because no term in the objective ever presents
"same activity, different device" as a pair to pull together.

The conclusion this audit keeps arriving at from different directions: **the bottleneck is the
learning signal's content, not the architecture, the gradients, or the optimizer.** The
architecture is now clean, tested, and cheap; further plumbing changes are unlikely to move it.
The open decision — one invariance term (conflicts with "no auxiliary losses") vs Phase-A
pretraining (conflicts with "e2e = random init") vs accepting the ceiling and reporting it — is
design-level and is the user's call.
