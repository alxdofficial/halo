# Tokenizer ablation — spectral filterbank vs learned rate-aware SSM

> **Status: Arm B is a standalone PROTOTYPE — NOT GO for training the comparison (2026-07-22).**
> An independent audit (verified here) found the prototype is not yet reachable from the training
> harness, cannot fit the intended batch in autograd, and had over-claimed its rate-invariance.
> Blockers before any comparison run — see §"Audit blockers" below:
> 1. **Not harness-integrated** — the CLI, the A1 calibration path, checkpoint save/resume/eval, and
>    the regularization hooks all assume a filterbank; Mamba needs its own path (`#2`).
> 2. **Backward graph too large** — the sequential scan retains the recurrent graph over all S steps
>    (~137 MiB saved tensors at a *tiny* config); infeasible at the real batch/`d_model` without a
>    fused/parallel scan (`#3`).
> 3. **Objective confound** — the A1 target is the *filterbank's* features, so Arm B is partly trained
>    to reproduce Arm A (`#7`).
> 4. **Rate-invariance was over-claimed** — see the corrected physics note below.
>
> The decisive head-to-head for the feature-extraction question (`POSITIONING.md` §10): is our
> physical-Hz filterbank leaving performance on the table versus a learned front end? Both arms share
> the drop-in contract `forward(patches (B,P,S,C), rate, N) -> (B,P,C,d)`, so the encoder body is
> identical — only the front end changes (`SetTokenizerEncoder(frontend=...)`).

## The two arms

| | Arm A — `frontend="fixed"` | Arm B — `frontend="mamba"` |
|---|---|---|
| method | multi-scale physical-Hz constant-Q filterbank | per-channel selective SSM on raw native-rate signal |
| rate handling | rate-invariant by construction (bands fixed in Hz) | **Δ = (1/rate)·softplus(·)** — physical step at *init*; a learned bias, NOT a guarantee |
| prior | strong handcrafted spectral prior | learned dynamics, minimal prior |
| observability | explicit Nyquist mask + resolution flag | implicit (model must infer from Δ) |
| gravity/DC | explicit signed-DC feature | preserved in raw signal (see normalisation) |
| file | `model/tokenizer/filterbank.py` | `model/tokenizer/mamba_frontend.py` |

**Arm B — the physics, corrected.** An SSM is a discretised continuous ODE `dh/dt = A h + B u`; its
step Δ is the physical time between samples. We init the base step to `1/rate`, which gives the SSM
recurrence a **rate-aware inductive bias**: on a smooth signal the same motion at 50 vs 100 Hz drives
the state at the same physical speed (measured: **~5× closer** under the correct rate on a clean 2 Hz
sinusoid, `tests/test_mamba_frontend.py`).

> ⚠️ **This is NOT "rate-invariance by construction," which was an over-claim.** Two paths in the
> frontend are *not* rate-invariant: (a) the depthwise **causal conv** has a fixed number of taps, so
> 4 taps span 0.2 s at 20 Hz but 0.04 s at 100 Hz — a rate-dependent physical window; (b) the **`D`
> skip** (`y += D·x`) is rate-independent. And the Δ multiplier is **unbounded at train time**
> (`mult_min/max` bound only the *init* — a `dt_proj.bias=10` yields a 10× multiplier), so training
> can distort the physical clock. The clean-sinusoid margin is also **seed-dependent** (2–8× over 30
> inits; ~5× at the test's seed 0 on current HEAD — an independent audit reported ~1× on a different
> state). **Measured on a real IMU window** (100→200 Hz resample): the correct-rate vs wrong-rate
> advantage is **~1.0× (negligible)** — the clean-sinusoid margin does not carry to broadband data. So rate-invariance for Arm B is a **learned target seeded by the Δ init**,
> not a structural guarantee. If we want it structural we must make the conv physical-time-aware
> (or drop it) and bound/monitor the multiplier — and if we don't, the comparison must say so and
> track the multiplier distribution by source/rate.

**Per channel, shared weights** (both arms): one tokenizer applied independently per channel, so
count/order stay free and identity comes from text downstream. Per-**sensor** SSM (cross-channel at
input) is a *follow-up* if Arm B is competitive — doing it now would change two variables (front end
AND channel grouping) and confound the comparison.

## Normalisation of the raw signal (Arm B)

The pipeline already rescales all accelerometer data to **g** (`accel_units.to_g`, m/s²→÷9.80665), so
raw values are in consistent physical units — gravity ≈ 1 g on whichever axis points down. Therefore:

- **A light, frozen, PER-MODALITY standardisation** is applied (separate μ/σ for accel vs gyro —
  they differ ~1.6× in scale in the corpus, and a single global scalar would let the larger dominate
  σ while the shared `in_proj` cannot compensate per channel; same calibration pattern as the
  filterbank: `fit_norm_stats` / `accumulate`+`finalize`). μ/σ are shared **within** each modality's
  axes, so the relative gravity direction survives. Default on; identity until calibrated.
- **NO instance normalisation.** Per-window mean removal (RevIN-style) would erase the DC/gravity
  component that separates static postures (stand/sit/lie) — the exact signal the filterbank is
  careful to keep as its signed-DC feature. Global (not per-window) standardisation removes the scale
  fingerprint while preserving the relative gravity direction. Pinned by
  `test_gravity_dc_is_preserved_not_normalized_away`.

This is *lighter* than the filterbank's elaborate DC handling precisely because `to_g` already did the
unit work — and applying the same frozen, gravity-preserving treatment to both arms keeps the
comparison fair.

## Fairness controls (so the result means something)

1. **Both get principled rate handling** — filterbank via physical Hz, SSM via Δ=1/rate. (A side-token
   rate for the SSM would be a rigged fight the filterbank wins by construction.)
2. **Same everything else** — encoder body, text conditioning, augmentations, subject splits,
   objective, optimizer, step budget, `d_model`. Swap only `frontend`.
3. **Keep Arm B small.** It is more learned capacity / less prior; our learnable-filterbank arm was
   already inert at this data scale, so over-sizing Arm B would test capacity, not method. Match
   parameter count to Arm A as closely as practical and report both counts.

## Data subset + metric suite — BUILT (2026-07-22)

**Subset** (`training/tokenizer/ablation_subset.py`), the "3-rate core": 6 train streams / 5 datasets
spanning 20/50/100 Hz × pocket/waist/wrist × acc-only+acc+gyro × gravity ±, capped ~10k/stream
(~55k windows), subject-disjoint. Held-out config = **xrf_v2** (unseen dataset; 6 simultaneous
placements — ideal for cross-config retrieval). capture24 excluded (144k acc-only would dominate).

| stream | rate | placement | chans | gravity |
|---|---|---|---|---|
| wisdm phone_pocket / watch_wrist | 20 | pocket / wrist | 6 | present |
| uci_har phone_waist | 50 | waist | 6 | present |
| unimib phone_pocket | 50 | pocket | **3** | present |
| pamap2 watch_wrist | 100 | wrist | 6 | present |
| kuhar phone_waist | 100 | waist | 6 | **removed** |

**Metrics** (`eval/tokenizer_metrics.py`, unit-tested `tests/test_tokenizer_metrics.py`): kNN purity,
cross-config retrieval (rate & placement), alignment/uniformity, effective rank, rate/placement
decodability, linear-probe/transfer BA — pure functions + `collect_embeddings`/`run_suite`.
Deliberately excludes A1/filterbank-similarity (confound #7).

**Arm A baseline** (fixed filterbank, `pretrain_fixed_mr`, on this subset — the bar Arm B must beat):

| | kNN purity | cross-rate retr. | cross-place retr. | rate-decode | place-decode | eff. rank | frontend params |
|---|---|---|---|---|---|---|---|
| val (in-dist) | **0.869** | **0.925** | 0.794 | **1.00** | **0.997** | 133.6/256 | **25.3k** |
| held-out xrf_v2 | 0.849 | — | 0.842 | — | — | — | (Mamba 194k, 7.67×) |

**Already-interesting:** the filterbank fully **leaks** rate & placement (decode 1.00 / 0.997 —
expected: Nyquist mask exposes rate, gravity survives) **yet cross-rate retrieval is 0.925** — so
config leakage is *not* harmful here; the activity signal dominates. That gives Arm B a concrete
target: *lower leakage at equal-or-better retrieval* is the win; *lower leakage that also hurts
retrieval* is not.

## Decision criterion (pre-registered)

Report internal val-kNN balanced accuracy and ZS-XD macro-F1 for both arms, with parameter counts and
wall-clock. **If Arm B does not beat Arm A on held-out configs by more than seed noise, the filterbank
stands** (with the Ravì-2016 rate-invariance-in-the-spectral-domain justification). If Arm B wins,
switch, then test per-sensor Arm B. Either outcome is publishable as a measured tokenizer choice.

## Perf caveat (Arm B)

The selective scan is currently **sequential** (exact, portable, kernel-free). The forward footprint
was fixed (per-step `(M,E,N)`, not the old `(M,S,E,N)` ~39 GB pre-materialisation), **but the backward
graph still retains the recurrence over all S steps** — measured ~137 MiB saved tensors at a *tiny*
config (B=2,P=4,S=64,C=6,d=64) vs 0.1 MiB for the filterbank. This does **not** scale to the real
batch / `d_model`, and shrinking the batch would change the SupCon objective. A fused/parallel
selective scan or the `mamba_ssm` CUDA kernel (not installed; a build step) with a memory-efficient
backward is **required before even a representative smoke run** — Mamba's efficiency claims depend on
its hardware-aware scan, not a Python recurrence.

## Audit blockers (independent review, 2026-07-22 — verified here)

Recorded so the "NO-GO for training" status is concrete and actionable. Numbers reproduced locally.

| # | severity | issue | status |
|---|---|---|---|
| 1 | blocker | `learnable=` passed twice via `build_frontend` broke the **default fixed path** (pretrain/eval). | ✅ **fixed** + regression test (`test_legacy_learnable_kwarg_still_builds_all_arms`). |
| 1b | blocker | **Silent substitution** (2nd audit): `pretrain.py` never passed `frontend=cfg.frontend`, so `cfg.frontend="mamba"` built the FIXED filterbank and stamped the checkpoint `"mamba"` — a falsely-labelled ablation. | ✅ **fixed** — routes `frontend=cfg.frontend`; the mamba path now **fails loud** (`NotImplementedError`) at model construction until its lifecycle lands, so no mislabeled run is possible (CLI or programmatic). |
| 2 | blocker | Mamba **not harness-integrated**: A1 calibration copies filterbank stats / calls `dc_mu`; the loop calls `adaptation_regularization`/`learnable`/`adaptation_summary`; `eval_transfer.build_encoder` always builds a filterbank. | ❌ open — needs a common frontend interface (calibration, regularization, save/resume/eval round-trip). CLI now accepts `mamba` but it raises until this lands. |
| 3 | blocker | **Backward graph too large** (see perf caveat). | ❌ open — needs parallel scan / kernel. |
| 4 | high | **Rate-invariance over-claimed** ("by construction", 5.6×). Real-window advantage ~1.0×; conv/skip are rate-dependent. | ✅ **claim corrected** (physics note above). Open *design* choice: make conv physical-time-aware / bound Δ, or keep the weaker "learned bias" claim. |
| 5 | high | **Δ multiplier unbounded** at train time (`mult_min/max` bound only init). | ⚠️ documented; bound-or-monitor is a design decision tied to #4. |
| 6 | high | **Per-modality norm assumes canonical channel order** (0:3 accel, 3:6 gyro); a permuted or 3-channel input misassigns/raises. Corpus always pads to canonical 6-ch so training is protected, but the "arbitrary channel count/order" contract is **false for this arm**. | ⚠️ documented; general fix = carry a modality id per channel, or require canonical layout explicitly. |
| 7 | medium | **A1 objective target is the filterbank's features**, so Arm B is partly trained to reproduce Arm A. | ❌ open — add an objective-neutral comparison (e.g. A1 off, or a frontend-agnostic target) and report A1/A2 magnitudes. |
| 8 | medium | Protocol not reproducible: subset TBD, seed count unspecified, Mamba settings absent from `PretrainConfig`/checkpoints. Params: frontend **25.3k (fixed) vs 194.3k (Mamba), 7.67×**; full encoder **+2.35%**. | ❌ open — stamp `d_state/d_inner/d_conv/Δ-bounds/pool/norm-groups/scan`, use matched multi-seed runs, disclose param gap. |

**Verdict: NO-GO for training the comparison.** The standalone prototype is sound in structure and
worth continuing (per-modality calibration correct, padding handling correct, grad flow verified), but
it is unreachable from the harness, cannot fit the intended batch in autograd, and its central
rate-invariance claim did not hold on real data. Clear #2, #3, #7 (and decide #4/#5/#6) before any
run. No trained Mamba checkpoint exists, so representation quality is still unmeasured.
