# Phase-B evidence mixer: the first learned component that works, and why

**Date** 2026-08-19 · **Encoder** `phase_a_h_mae_fixes_20260818/best.pt` · **Trainer**
`training/tokenizer/pretrain_episodic.py` · **Module** `model/evidence/evidence_mixer.py`

Supersedes [`PHASE_B_E2E_20260818.md`](PHASE_B_E2E_20260818.md), whose title — "four arms, and why
none of them works" — was correct for the components it tested and is now false for the model.

All numbers are hard top-k (the deployment rule, `top_k=64`), macro-F1 in label space, meaned over
the fixed support grid k = 0, 1, 2, 4. Held-out **concepts** and held-out **subjects**; 96
validation episodes per checkpoint. Measured validation std on this protocol is **0.0130**, so treat
anything under ~0.026 between two arms as unresolved.

---

## 1. The headline

| arm | what trains | hard mean | k=0 | transfer |
|---|---|---|---|---|
| **mixer** | encoder + mixer + gate | **0.5247** | **0.5351** | 0.8175 |
| encoder only | encoder + gate | 0.4093 | 0.3534 | 0.7999 |
| frozen control | nothing | 0.4053 | 0.3916 | 0.8246 |

**+0.119 over its own control**, against **+0.004** for encoder fine-tuning alone. Every previous
learned Phase-B component was worth ±0.02, and the most recent one was net-negative.

The k=0 cell is the one that matters most: held-out concepts with no support, where the only live
mechanism is the label-text bridge. That cell goes **0.3916 → 0.5351**. It is also the cell where
the previous learned decoder collapsed *below chance* (16.18 → 9.44 against a 13.77 floor), which is
why the mixer forbids any query×candidate term — see the design doc.

Transfer barely moves (0.8175 vs 0.8246). Earlier arms bought +0.018 of task for −0.054 of
representation; this buys +0.119 for −0.007. The gain and the damage have come apart.

## 2. The gain is SEMANTIC — the controlled result

The mixer's correction is a sum of three separable low-rank forms. `qm` is **candidate-blind**: its
value is identical for every candidate, so it can only sharpen retrieval uniformly. `mc` and `qmc`
can say *which candidate a row speaks to*.

| arm | forms | hard mean | **k=0** | reading |
|---|---|---|---|---|
| all forms | qm+mc+qmc | 0.5247 | 0.5351 | |
| **text only** | mc+qmc | 0.5194 | 0.5279 | `qm` is redundant — 0.005 apart, inside noise |
| | qmc | 0.4882 | 0.4948 | |
| | mc | 0.4801 | 0.4752 | |
| **candidate-blind** | qm | 0.4495 | **0.3579** | helps k≥1, pushes k=0 **below control** |
| frozen control | — | 0.4053 | 0.3916 | |
| **SCRAMBLED VOCABULARY** | qm+mc+qmc | **0.3641** | **0.3445** | **inverts the gain** |

The scramble is the load-bearing control. The mixer is handed a fixed permutation of the label
vocabulary: identical shapes, roles, token counts, forms and readout — only the words no longer mean
what they say. The +0.119 does not merely vanish, it inverts to **−0.041 below the frozen control**.

It is not a training failure. That arm reached `pair_gain` 0.0462 and applied corrections of 3.19
nats, comparable to the winning arms. It trained just as hard and learned confidently from
meaningless semantics.

**Conclusion: candidate-facing structure is worth nothing on its own; the language carries it.**
This corroborates the earlier elimination result that HALO's zero-shot deficit was classifier
semantic alignment (align@T halo 0.3919 < harnet 0.4098 < crosshar 0.4226, reproducing the zero-shot
ordering across all three models exactly).

Secondary finding, separable and clean: **few-shot gain is representational, zero-shot gain is
semantic.** Uniform reweighting (`qm`) helps where an enrolled row carries the answer by identity,
and hurts where the text bridge is the only mechanism.

## 3. Storage depth, pooling, query head

| arm | hard mean | k=0 | transfer | note |
|---|---|---|---|---|
| mixer, depth full | 0.5247 | 0.5351 | 0.8175 | |
| mixer + query head, full | 0.5036 | 0.5163 | 0.8186 | query head costs 0.021 — **inside 2σ, not established** |
| mixer + query head, isolated | 0.5012 | 0.5235 | **0.8288** | isolated ≈ full, better transfer |
| mixer + query head, pool 64 | 0.4897 | 0.4882 | 0.8172 | pool 64 costs 0.035 — outside noise |
| mixer + query head, uncapped pool | 0.5228 | 0.5198 | 0.8254 | ≈ pool 256 |

Two design claims that did **not** survive contact with the data, both of them mine:

- **`--retrieval-depth full` is not worth it.** Storing post-trunk rows makes 92.4% of the encoder
  trainable instead of 18.8%, which was the argument for it. But at the *frozen control* it costs
  1.55 points (0.4053 vs 0.4208), concentrated at k≥1, because the stored row now depends on which
  other sensors were present. With the mixer, isolated ≈ full and isolated keeps both better
  transfer and config-independence — which is the thesis. **Isolated is the recipe.**
- **The query projection head is not earning its 263,424 parameters.** The gap is inside 2σ, so this
  is a decision under uncertainty rather than a finding; it is off by default and stays measurable.

## 4. Dead path, measured rather than assumed

| | frozen/early | trained mixer |
|---|---|---|
| memory rows receiving gradient | 1.000 | **0.918** |
| gradient mass on deployment's top 64 | 0.72 | 0.68–0.91 |
| soft/hard argmax agreement | 0.98–1.00 | 0.98–1.00 |

Training votes over the whole working set, so initially *every* memory row is reached. As the mixer
sharpens the weights ~8% of rows underflow to exactly zero gradient. The dead path is real, small,
and now a logged series (`dead_path/*`) rather than an assumption.

## 5. Gradient audit — task specificity

`--grad-audit` replays each episode with the answer key rolled and reports
`1 − cos(g, g_scrambled)`. Near 0 means a module's gradient is the same whether or not the labels
are right.

| module | task specificity |
|---|---|
| encoder | 0.81 – 0.98 |
| evidence mixer | 0.79 – 0.92 (all groups) |
| query projection | 0.795 |
| **admissibility gate** | **0.156 – 0.165** |

The gate is a function of frozen text alone and never sees the signal, so most of its gradient is a
shared "push all admissibilities" term rather than task credit. That is a quantitative account of
the standing measurement that the gate trained alone *hurts* by 0.018.

Two other things the audit reports, both correct rather than bugs: the encoder's `descriptor_head`
and `mask_token` are Phase-A-only and structurally unreachable here, and the query head's first two
layers receive no gradient at step 0 because it is a zero-init residual (they unblock at step 2).

## 6. The adopted configuration

```
--mixer --retrieval-depth isolated --mixer-forms mc qmc --mixer-pool 256
        (admissibility gate on, query head off)
```

`forms`, `pool` and `depth` are now the code defaults, and the `qm` projection heads are no longer
constructed at all when the form is disabled. `--mixer` itself is still opt-in pending §7.

## 7. In flight

`best_base` (the adopted combo, never yet measured on the isolated base), two semantic-refinement
arms, `nogate`, and one seed repeat.

## 8. What this does NOT establish

Every number here is **held-out-concept validation inside the training corpus**. None of it has
touched the eval harness that produces the headline table against HARNet/CrossHAR/UniMTS, because
`admissible_retrieval.predict` has no mixer branch and `SensorRows` carries no source-window id for
the mixer's `group` channel. Closing that needs a bank field, a bank rebuild, and a pooled mixer
branch in `predict`.

Single seed on most arms. The +0.119 headline is ~9σ against the measured 0.0130 std and is not in
doubt; differences *among* mixer arms of 0.02 are not resolved.
