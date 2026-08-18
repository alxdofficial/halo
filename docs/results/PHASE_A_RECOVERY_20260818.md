# Phase-A recovery measurements — 2026-08-18

Measurements taken while recovering from the rejected `phase_a_fixed_1s_rotation_20260817` arm.
Everything here is measured, not projected. The Phase-B numbers are re-derived from the committed
`eval/adaptation_tables/v1_d85761d_stage2/cells.csv`; the Phase-A numbers come from
`training.tokenizer.eval_transfer.development_transfer_score`.

## 1. The random-init floor, and what Phase-A training is worth

`development_transfer_score` on the three clean-label development cohorts (MotionSense, RealWorld,
Shoaib), subject-disjoint kNN over the **sensor rows Phase B actually stores**. A 2x2 over
{random init, trained} x {isolated 1-layer retrieval row, full 6-layer token}, same weights within
each row:

| weights | isolated retrieval row | full-depth token | isolated − full |
|---|---:|---:|---:|
| random init (seed 20260718) | 0.8012 | 0.8017 | −0.001 |
| trained 4k (`phase_a_sensor_v1_20260813_v2`) | **0.8577** | 0.7863 | **+0.071** |
| trained 27k (`phase_a_fixed_1s_rotation_20260817`) | 0.7161 | 0.6765 | +0.040 |

Three readings:

1. **The isolated retrieval row is a free win on trained weights** (+0.071 / +0.040) and neutral at
   random init (−0.001). It independently validates the `retrieval_tokens` change in `d50c121`: the
   gain is real, it comes from removing cross-sensor contamination rather than added capacity, and
   it is *created by training* — the two readouts are identical before any gradient step, so the
   full-depth trunk is actively mixing away information the isolated row keeps.
2. **Phase-A SSL is worth +0.057** — the entire measured value of training, best-checkpoint-ever
   (0.8577) minus random init (0.8012). The fixed physical filterbank supplies the rest. A random
   projection roughly preserves the distances the tokenizer already encodes, which is why random
   init is such a strong baseline here.
3. **The rejected arm trained to 0.0851 BELOW random init.** Training actively destroyed the
   representation. This is a third independent confirmation of the independent-SO(3) diagnosis,
   after the per-dataset regression pattern and the one-variable pilots.

## 2. The posture canary

Static postures (`lying`/`sitting`/`standing`) are separated only by gravity direction. On the
rejected arm they collapse 2–2.5x harder than the aggregate score moves:

| canary | old-good 4k | rejected 27k | Δ |
|---|---:|---:|---:|
| posture/shoaib | 1.0000 | 0.7033 | −0.297 |
| posture/realworld | 0.7238 | 0.4848 | −0.239 |
| posture/motionsense | 0.9831 | 0.7963 | −0.187 |
| *aggregate (4-source scalar)* | *0.6516* | *0.5741* | *−0.078* |

This is now a tracked per-arm metric, not a postmortem.

## 3. ExtraSensory is anti-correlated with training

Added to the selection roster for wrist/free-living coverage, then measured across three encoders:

| encoder | ExtraSensory kNN BA |
|---|---:|
| random init | 0.307 |
| rejected 27k | 0.274 |
| old-good 4k | 0.251 |

Monotonically **decreasing** in training quality. Folding it into the selection scalar would bias
checkpoint choice toward undertrained encoders, so it is retained as a reported diagnostic
(including its wrist posture canary) and excluded from the scalar. Free-living self-reported labels
are the likely cause.

## 4. The k=8→16 "droop" is a cohort artifact, not saturation

Ordinary coherent macro-F1 appears to fall from k=8 to k=16 for HALO (51.18 → 49.32). It does not.
UT-Complex lacks 16 independent support executions and leaves the cohort. Restricted to the three
datasets present at both k (InclusiveHAR, TNDA-HAR, USC-HAD), **every model improves**:

| model / rule | k=8 (all) | k=8 (common) | k=16 | matched Δ |
|---|---:|---:|---:|---:|
| HALO identity | 53.20 | 47.65 | 50.93 | +3.28 |
| HALO learned | 51.18 | 47.24 | 49.32 | +2.08 |
| HALO ridge | 53.82 | 47.70 | 49.95 | +2.25 |
| UniMTS linear | 66.69 | 62.62 | 65.05 | +2.43 |
| LiMU-BERT linear | 64.65 | 59.63 | 60.29 | +0.66 |
| HARNet linear | 62.77 | 56.23 | 59.73 | +3.51 |

HALO's k-scaling is normal. The **level** is the deficit. Any cross-k claim must carry the
dataset-count column.

## 5. The representation gap, with semantics removed

Random-alias protocol (fresh one-to-one label aliases per episode, so no text or semantic
information can contribute), ordinary regime, matched 3-dataset cohort, identical readout
arithmetic on both sides:

| model | rule | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---:|---:|---:|---:|---:|
| UniMTS | nearest | 51.16 | 56.86 | 61.80 | 60.96 | 63.27 |
| LiMU-BERT | nearest | 51.14 | 57.41 | 60.85 | 58.62 | 60.89 |
| CrossHAR | prototype | 48.02 | 50.88 | 54.42 | 52.81 | 54.51 |
| HARNet | prototype | 45.09 | 46.76 | 50.26 | 48.81 | 50.40 |
| **HALO** | **identity** | **42.10** | **45.61** | **48.70** | **47.88** | **50.73** |
| ImageBind | prototype | 40.48 | 44.53 | 45.90 | 41.54 | 44.30 |

HALO places 5th of 6 on pure support-driven representation quality — **12 points behind LiMU-BERT
at k=4**, with no gate, no text and no semantics involved. LiMU-BERT is ~62K parameters and was
self-pretrained on *our own corpus* (`docs/baselines/BASELINES.md`), which eliminates data scale,
compute and capacity as explanations and leaves the objective and the tokenizer.

## 6. Consequence for the plan

The augmentation fleet can at best recover the old-good checkpoint (0.8577), which was already
12 points behind LiMU-BERT downstream. Two structural facts point past augmentation tuning:

- Phase-A transfer **peaks early and then decays** — the old run peaked at step 4,000 of 30,000;
  the clean arm peaks at step 1,000 of 7,500 and falls thereafter. An objective whose transfer
  optimum arrives in the first 15% of the schedule is not being helped by more of itself.
- With clean (identical) positive views, VICReg's invariance term is ~0 by construction, so it acts
  only as a whitening regulariser and **JEPA is the sole learning signal** — masked prediction of a
  slowly-updating EMA teacher's latents, the setup most prone to shortcut solutions.

The cheapest decisive next experiment is therefore an **objective** change, not a recipe change:
replace the EMA-latent JEPA target with masked reconstruction of the *filterbank analysis features*
of held-out patches. That target is a fixed physical quantity already computed every step, so it
cannot collapse, and it is the objective class LiMU-BERT uses to beat us by 12 points at 1/120th
the parameter count.
