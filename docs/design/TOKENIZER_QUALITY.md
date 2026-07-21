> ⚠️ **STALE ENCODER (2026-07-21).** This battery was run on an **11k-step** checkpoint from the
> OLD 8-dataset / 57-label / 60 Hz corpus. Current encoders are `pretrain_native` and
> `pretrain_fixed_mr` on the 12-dataset / 93-label native-rate corpus. The probe *methods* are still
> the ones we use (`training/tokenizer/eval_quality.py`); the *numbers* describe a model that no
> reported result depends on.

# Phase-1 tokenizer quality report (2026-07-18)

Comprehensive quality battery on the frozen Phase-1 encoder (`best.pt`, d256/6L, step 11k).
Probes in `training/tokenizer/eval_quality.py`; raw numbers in
`outputs/pretrain/quality_report.json`. All on **held-out** eval datasets, subject-disjoint,
kNN unless noted. **This is representation characterization — NOT the ConSE zero-shot
macro-F1 baseline comparison (that is M6).**

## Results

### 1. Discriminability (per held-out dataset)
| dataset | kNN-BA | linear-probe F1 | eff-rank (of 256) | labels |
|---|---|---|---|---|
| shoaib | 0.891 | 0.924 | 86 | 7 |
| motionsense | 0.842 | 0.895 | 91 | 6 |
| realworld | 0.744 | 0.780 | 75 | 8 |
| inclusivehar | 0.502 | 0.569 | 83 | 6 |
| **mean** | **0.745** | **0.792** | ~84 | |

Linear-probe F1 ≥ kNN everywhere → features are well **linearly separable** (not just locally
clustered). Effective rank ~75–91 → healthy, **no collapse**.

### 2. Invariance (motionsense; embedding drift + kNN with transformed query vs clean bank)
| nuisance | emb cosine | kNN (clean 0.861 → transformed) |
|---|---|---|
| rotation (SO3) | 0.979 | 0.861 → **0.861** (no loss) |
| gain (±30%) | 0.972 | 0.861 → 0.836 |
| rate (60→30 Hz) | 0.902 | 0.861 → **0.769** (−0.09) |

### 3. Learned vs handcrafted (M0 gravity-aligned band energy, same splits)
| dataset | learned kNN | handcrafted kNN | winner |
|---|---|---|---|
| motionsense | 0.842 | 0.898 | handcrafted |
| shoaib | 0.891 | 0.940 | handcrafted |
| realworld | 0.744 | 0.711 | **learned** |
| inclusivehar | 0.502 | 0.545 | handcrafted |
| **mean** | **0.745** | **0.773** | **handcrafted** |

### 4. Missing-channel robustness
mean kNN full-IMU **0.745** → accel-only **0.682** (−0.06). Graceful (67% of corpus is accel-only).

### 5. Cross-PLACEMENT transfer (wisdm phone↔watch; TRAIN data, flagged)
phone→watch **0.413**, watch→phone **0.272**.

## Interpretation — strengths & weaknesses

### Strong
- **In-domain discriminability is solid** (kNN 0.75 / linF1 0.79), linearly separable, no collapse.
- **Rotation-invariant** (cos 0.98, zero kNN loss) and **gain-robust** — the two nuisances we most
  designed for. The salient-not-fragile thesis holds on these axes.
- **Degrades gracefully to accel-only** (−6 pts), matching the deployment reality.

### Weak / honest limits
1. **Does NOT clearly beat handcrafted physical features on in-domain discriminability**
   (learned 0.745 < handcrafted 0.773 mean; learned wins only on realworld, the 8-label hardest set).
   At THIS scale/steps, the transformer + contrastive does not surpass the M0 gravity-aligned band
   energy for within-dataset activity separation. The learned model's justification therefore rests
   on the axes this probe does NOT test — **config-conditioning (channel text), a learnable retrieval
   metric for Pipeline B, and open-set/cross-config generalization** — which Pipeline B must
   demonstrate. Caveat: handcrafted band-energy is itself an *input* the encoder consumes, so this is
   "did the learned layers add value on top of the physics?" — on in-domain kNN, not yet.
2. **Cross-PLACEMENT transfer is the weakest axis** (phone→watch 0.41). This is precisely the
   config-heterogeneity the thesis targets — and the channel-text conditioning is not yet enough to
   bridge phone-pocket ↔ watch-wrist. (Partly genuine: wrist and pocket see different motion.)
3. **Rate-invariance is only partial** (kNN −0.09 at 30 Hz) — weaker than rotation/gain, despite the
   physical-Hz filterbank. Rate is the least-robust nuisance.
4. **inclusivehar (0.50)** — atypical/disability motion; a persistent fairness gap (also lowest for
   the baselines).

## Implications
- The learned tokenizer is a decent, invariant, retrieval-ready representation — but its **edge over
  handcrafted features is unproven** and must come from Pipeline B (conditioning + evidential
  retrieval + abstain), not raw discriminability.
- The two weakest axes — **cross-placement** and **rate** — are exactly what **more config-diverse
  data** fixes (new placements/devices/rates). This directly motivates the dataset-expansion work:
  we are placement/device-poor, and it shows.
- Next-run levers suggested by this: enable `time_warp` (cadence/rate diversity), add more
  placement/device diversity (data), and consider a light sensor↔text alignment term if Pipeline B's
  open-set transfer is weak.
