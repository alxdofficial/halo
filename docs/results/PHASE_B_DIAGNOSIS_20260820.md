# Why Phase-B training plateaus — diagnosis, 2026-08-20

Branch `phase-b-diagnostics`. Every number below is measured; the scripts are
`training/evidence/phase_b_autopsy.py` and `training/evidence/phase_b_bottleneck.py`.

## Summary

Two findings, and the second corrects the first.

**Mechanism.** The encoder's retrieval feature encodes **acquisition configuration, not activity**
(×7.0 lift for same-config rows; same-activity support rows sit at the 39th percentile), and
underneath that, **activity names and activity signals are nearly unrelated on this corpus**
(r = 0.11 across 105 labels), which caps any signal-based zero-shot bridge. These are measurements
on fixed checkpoints and they stand.

**Methodology — and this invalidates the first pass of arm comparisons.** Four replicates of one
identical configuration, differing only in seed, give selection scores of 0.4499 / **0.5881** /
0.4545 / 0.4519 — **between-run sd 0.068**, five times the ±0.013 within-run scatter I had been
using as the noise band. Every "this intervention does nothing" conclusion drawn against ±0.013 was
underpowered and is withdrawn.

Almost all of that variance is the **validation draw**, not training: the seed selects which
concepts are held out (16 to 23 of them across the four runs), and the step-0 baseline alone has
sd 0.073. Pairing each run against its own step-0 score removes it — the **paired gain has
sd 0.0069**, a tenfold tighter statistic, and it shows training clearly works: **+0.0656 ± 0.0035,
p = 0.0003**.

**Every future Phase-B comparison must use the paired gain, not the raw score.**

## What was ruled out, with the measurement that ruled it out

| Hypothesis | Measurement | Verdict |
|---|---|---|
| Encoder representation collapses | per-stage autopsy: effective rank 8.6 (random init) vs 7.9 (best) vs 8.1 (last); kNN label accuracy **rises** 0.782 → 0.847 → 0.875 | **No.** The encoder improves monotonically. My earlier "rank collapse 24→9" read a per-episode telemetry statistic on ~100 rows, not a property of the representation. |
| Performance declines after the peak | linear trend over the plateau, p = 0.113; the "best" at step 1750 is +2.1σ, i.e. the max of 24 draws | **No.** Flat plateau, not a decline. |
| The step budget is too small | plateau reached at step 250 = 1,000 episodes; 6,000 steps adds nothing | **No.** More steps do not help. |
| The vote is broken | with the *same* vote and *same* top-64, a perfectly semantic retriever scores **0.840** vs the model's 0.365 | **No.** The vote is fine. |
| The text vocabulary is not separable | oracle single-row ceiling 0.95–1.00 at C=2…16; held-out concepts are separable against the training vocabulary | **No.** Ample headroom. |
| Retrieval never finds useful rows | 40.5% of retrieved rows are individually sufficient, vs a 33.3% base rate | **Partly.** Better than chance, but weak. |

## What is actually wrong

**1. All of the model's accuracy comes from retrieval *selection*; the learned weighting is inert.**

| vote variant (k=0) | macro-F1 |
|---|---:|
| ignore the query entirely (uniform over the whole bank) | 0.271 |
| uniform over the model's retrieved top-64 | **0.365** |
| the model's own trained weights | 0.347 |

The trained weights are *worse* than uniform. Within the retrieved set, the score correlates with
label relevance at r = 0.028. Consistent with that, the vote is a near-uniform average —
**57.4 effective rows of 64** — and 79% of the queries in an episode receive the same prediction.

**2. Retrieval ranks by acquisition configuration, not activity.**

| property of the retrieved top-64 | in top-64 | bank base rate | lift |
|---|---:|---:|---:|
| same acquisition config as the query | 16.9% | 2.4% | **×7.0** |
| same activity, different subject and device (enrolled support) | 4.8% | 0.8% | ×1.7 |

A support row — same activity, different subject, different device — sits at the **39th percentile**
of the ranking, barely better than chance. The encoder has learned same-label clustering *within* a
configuration (kNN 0.875) and almost nothing *across* configurations, which is the only thing that
matters here.

**3. Underneath both: names and signals are nearly unrelated on this corpus.**

Across 105 labels, correlating signal-space similarity of class centroids against label-text
similarity:

```
Pearson  r = +0.114   (p = 3.3e-17)
Spearman r = +0.146   (p = 1.7e-27)
top-5 neighbour overlap between the two spaces: 19.2%   (chance 4.8%)
```

Significant, and very weak. This is a **ceiling on any signal-based ConSE bridge**, independent of
the model: zero-shot transfer works by mapping an unseen name to nearby training names and expecting
their signals to resemble the query. At r = 0.11 that inference is mostly noise. "Brushing teeth"
and "brushing hair" have near-identical names and unrelated dynamics; "walking" and "cycling" have
unrelated names and similar periodicity.

This predicts k=0 should be weak while k≥1 should not, because enrollment does not use the bridge at
all. The measured curve matches: coherent k=0 **0.324**, k=4 **0.521**, alias k=4 **0.524**.

## Interventions tried, and what they show

All at 1,500 steps, random init, held-out concepts. Noise band ±0.013.

Re-scored with the paired statistic (gain over each run's own step-0, noise sd 0.0069):

| arm | paired gain | z vs replicates | gain k=0 | gain k=4 | gain alias |
|---|---:|---:|---:|---:|---:|
| **augment** | **+0.1075** | **+6.0** | +0.0350 | +0.1615 | +0.1176 |
| augment, 3000 steps | +0.0989 | +4.8 | −0.0318 | +0.1016 | +0.1278 |
| top-k 16, 6000 steps | +0.0962 | +4.4 | +0.0665 | +0.1785 | +0.0638 |
| k=64, 6000 steps (reference) | +0.0946 | +4.2 | +0.0266 | +0.1812 | +0.0685 |
| aux 0.05 | +0.0891 | +3.4 | −0.0355 | +0.1073 | +0.1149 |
| top-k 16 | +0.0867 | +3.0 | +0.0792 | +0.1298 | +0.0581 |
| control (1500 steps) | +0.0753 | +1.4 | −0.0121 | +0.1340 | +0.0656 |
| replicate mean (n=4) | +0.0656 | 0.0 | +0.0147 | +0.0406 | +0.0959 |
| encoder LR 5e-4 | +0.0658 | +0.0 | +0.0030 | +0.1595 | +0.0433 |
| aux 0.2 | +0.0536 | −1.7 | +0.0343 | +0.0559 | +0.0722 |
| top-k 8 | +0.0522 | −1.9 | +0.0125 | +0.1229 | +0.0247 |
| aux 1.0 | +0.0506 | −2.2 | +0.0470 | +0.0546 | +0.0728 |
| top-k 8 + encoder LR | +0.0455 | −2.9 | +0.0107 | +0.1048 | +0.0285 |

Two things the first pass got wrong. **Augmentation is the best intervention tested**, not a null
one — it is what teaches invariance to acquisition configuration, which is the measured defect, and
it lands 6 sd above the replicate mean. **Longer training also helps**: every 6,000-step arm sits at
+0.095 against +0.066–0.075 for 1,500 steps, so the model was never converged at step 250; the raw
metric was simply too noisy to show it.

The raw scores that produced the withdrawn conclusion, kept for the record:

| arm | selection | coherent | alias | k=0 | k=4 | support selected |
|---|---:|---:|---:|---:|---:|---:|
| control | 0.4321 | 0.4104 | 0.4539 | 0.2851 | 0.4739 | 0.0120 |
| retrieval-alignment aux 0.05 | 0.4459 | 0.3886 | 0.5032 | 0.2617 | 0.4472 | 0.0271 |
| retrieval-alignment aux 0.2 | 0.4105 | 0.3604 | 0.4605 | 0.3315 | 0.3958 | 0.0391 |
| retrieval-alignment aux 1.0 | 0.4074 | 0.3537 | 0.4611 | 0.3442 | 0.3945 | 0.0416 |
| top-k 16 | 0.4273 | 0.4227 | 0.4320 | 0.3439 | 0.4751 | 0.0142 |
| top-k 8 | 0.3970 | 0.3795 | 0.4146 | 0.2793 | 0.4528 | 0.0150 |
| encoder LR 5e-4 | 0.4226 | 0.4136 | 0.4316 | 0.3003 | 0.4995 | 0.0100 |
| top-k 8 + encoder LR | 0.3902 | 0.3622 | 0.4183 | 0.2775 | 0.4346 | 0.0070 |

Against the *raw* score nothing cleared the band — which is exactly the underpowered comparison
described above. The aux loss remains an informative case: it **worked mechanically** —
support retrieval rose 3.5× (0.012 → 0.042) and alias rose 0.454 → 0.503 — but coherent fell by the
same amount, netting zero. It is not adopted; it stays behind `--retrieval-aux-weight 0.0` as a
diagnostic probe, since the finding is that the two objectives trade against each other rather than
that retrieval supervision is unavailable.

Post-hoc repairs that also failed: centering the evidence (0.355), sharpening the vote softmax
(0.371), top-1 row only (0.387), config-mean removal from the features (0.365, and support ranking
got *worse*, 39.4 → 43.8 percentile).

## What follows

The zero-shot cell is limited by the corpus, not the model, and no amount of Phase-B training will
change that. The honest framings are (a) report k=0 as bridge-limited and quantify the limit with
the r = 0.11 measurement, which is a result about HAR label semantics rather than a defect, and
(b) put the weight of the claim on k ≥ 1 adaptation, which does not use the bridge and already
reaches 0.52.

Signal augmentation, which targets cross-configuration invariance directly and is already
implemented (`--augment`, off by default), also fails to clear the band: 0.4289 at 1,500 steps and
0.4203 at 3,000, against 0.4321 for control. Ten interventions, none of them effective.

## The finding that reopens the question

The plateau is NOT a representation limit, because the representation never stops improving:

| checkpoint | kNN label accuracy | signal-vs-name Pearson r |
|---|---:|---:|
| random init | 0.782 | 0.031 |
| best (step 1750) | 0.847 | 0.114 |
| last (step 6000) | **0.875** | **0.160** |

Both are still climbing at step 6000 while the selection score has been flat since step 250. The
bridge is five times better than random init and getting better; the end metric does not move.

The mechanism that reconciles this is the width of the vote. The vote averages the top 64 of ~5,000
rows, and the *average label composition* of a broad set is insensitive to modest ranking
improvements — a better ranking reshuffles the set without much changing its mean. That predicts
narrower top-k should convert the improving bridge into accuracy, and that neither narrow-k nor long
training alone would show it, which is exactly what the grids found (top-k 16 gives the best k=0 of
any 1,500-step arm, 0.344 vs 0.285, but its advantage is inside the band at that budget).

`gridD` tests the compound directly: top-k 16 at 6,000 and 12,000 steps.


## Ablation ladder — which learnable stage is failing?

Working backwards from the loss, one learnable stage enabled at a time, everything else replaced by
a fixed stand-in (retrieval → plain feature cosine at fixed temperature, no parameters; mixing →
removed; encoder → random init, frozen). 1,500 steps, one seed, paired gain over each run's own
step-0 (noise sd 0.0069, so a marginal effect above ~0.014 is real).

| rung | step-0 | paired gain | k=0 | k=4 | alias | marginal |
|---|---:|---:|---:|---:|---:|---:|
| 1. readout only, no attention | 0.3391 | +0.0915 | −0.0072 | +0.0917 | +0.1139 | |
| 2. + attention | 0.3389 | +0.1068 | +0.0008 | +0.1001 | +0.1443 | **+0.0153** |
| 3. + learned retrieval | 0.3573 | **+0.1153** | −0.0033 | +0.1375 | +0.1462 | +0.0085 |
| 4. + encoder (FULL model) | 0.3573 | +0.0895 | +0.0101 | +0.1169 | +0.0878 | **−0.0258** |
| isolation: retrieval alone | 0.3578 | **+0.0000** | +0.0000 | +0.0000 | +0.0000 | |
| isolation: encoder alone | 0.3390 | +0.0329 | +0.0125 | +0.0583 | +0.0267 | |
| depth: 4 attention layers | 0.3389 | +0.1084 | +0.0020 | +0.0904 | +0.1504 | |

Four things fall out, and they answer the question directly.

**The readout carries almost all of the learning.** Rung 1 — a frozen random encoder, a fixed cosine
retrieval, no attention at all, and only the step-6 low-rank forms learnable — already gets +0.0915
of the +0.1153 the best rung reaches. Everything else in the model is fighting over the last 20%.

**Attention adds a little; depth adds nothing.** +0.0153 for two layers (2.2 sd, real but small),
and four layers is +0.1084 against two layers' +0.1068, i.e. no gain from depth.

**Learned retrieval in isolation contributes exactly nothing** (+0.0000: no validation ever beat
step-0). Its only path to the loss is a softmax that is a near-uniform average over 64 rows, and
the gradient through that is too weak to move it. Inside the full stack it is worth +0.0085, which
is inside the noise. This is the same conclusion the post-hoc analysis reached from the other side —
uniform weights over the retrieved set beat the trained weights.

**Training the encoder end to end makes things WORSE**: −0.0258, −3.7 sd, when added on top of
rung 3. The encoder in isolation helps a little (+0.0329), so it is not that encoder gradients are
useless — it is that training it jointly with the rest, at this scale and objective, is
counterproductive. That is the single most actionable result here.

**No learnable stage improves k=0.** Gains there run −0.007 to +0.013 across every rung, against
+0.09 to +0.15 for k=4 and alias. Learning buys adaptation and buys nothing at all for zero-shot,
which is exactly what the r = 0.11 name-signal measurement predicts.

## Numerical defects found by the audit

**Fixed — residual branches were not scaled at init.** On real episode tokens, layer 0's attention
update was **|Δx|/|x| = 2.55**: it overwrote the residual stream it was meant to refine, and layer 1
then sat at **99% of uniform attention entropy** with nothing left to discriminate. Scaling the two
residual output projections by `1/sqrt(2 * n_layers)` — standard practice from GPT-2 onward, and
simply missing here — brings layer 0 to 0.11. The ladder above was run with this fix in place.

**Found, configurable, not yet defaulted — the identity channels are as loud as the content.**
`ScaledSum` normalises content and the role/slot/group channels to unit norm with equal gain, so
**content is 24.3% of every token** and the slot and group ids, which are redrawn every episode, are
half of it: from attention's point of view half of each token is per-episode noise. The gains stayed
within 5% of 1.0 through training, so the model does not correct this on its own.
`EvidenceMixerConfig.identity_gain_init` (and `--identity-gain`) exposes it; 0.3 puts content at
52.6%. Default is unchanged pending the measurement.

Audited and clean: the pre-norm stack does not grow activations with depth; the attention bias is
standardised to a scale comparable to the content logits, so it neither vanishes nor saturates; the
vote is finite under all-enrolled, no-enrolled, and log-weights scaled by 1e-6 and 1e3.
