# Tokenizer ablation — spectral filterbank vs learned rate-aware SSM

> **Status: Arm B is now HARNESS-INTEGRATED (2026-07-22 pm). One perf caveat remains before a
> full-scale run; the comparison scaffolding (subset + metrics) is live.** Two independent audits
> flagged a set of blockers; all are fixed or addressed except scan throughput — see §"Audit
> blockers":
> - ✅ **Harness integration (#2)** — `--frontend mamba` builds the SSM (was silently the filterbank);
>   frontend-agnostic calibration, `adaptation_regularization`/`adaptation_summary` hooks, checkpoint
>   config, and `eval_transfer` reconstruction all done; 5 integration tests + 11 unit tests.
> - ✅ **Backward memory + throughput (#3)** — the **official fused `mamba_ssm` CUDA kernel** is now
>   wired in (`use_kernel`, auto-selected on CUDA), with the pure-PyTorch chunked-checkpointed scan as
>   the CPU/test fallback (the same dual the mamba repo ships). Verified **numerically identical**
>   (max diff 3e-8) and **9.7× faster** fwd+bwd on GPU. *A full-scale training smoke has still not
>   been run, but the perf blocker is resolved.* Requires `causal-conv1d` + `mamba-ssm` (CUDA-only).
> - ✅ **Objective neutrality (#7)** — `--a1-weight 0` runs the comparison on A2 (SupCon) + A3
>   (grounding), both frontend-agnostic, so Arm B is no longer trained to reproduce the filterbank.
> - ✅ **Δ-multiplier (#5)** — soft reg pulls the baseline toward the physical clock; monitored in
>   `adaptation_summary`. **Rate-invariance remains a learned bias, not structural** (conv/skip caveat
>   in the physics note below).
> - ⚠️ **Channel order (#6)** — Arm B requires the canonical 6-channel layout (the corpus always
>   provides it); documented constraint, not arbitrary-order like the filterbank.
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
| 1b | blocker | **Silent substitution**: `pretrain.py` never passed `frontend=cfg.frontend`, so `cfg.frontend="mamba"` built the FIXED filterbank and stamped the checkpoint `"mamba"`. | ✅ **fixed** — routes `frontend=cfg.frontend`. |
| 2 | blocker | Mamba **not harness-integrated** (calibration, regularization/logging hooks, checkpoint config, eval reconstruction all assumed a filterbank). | ✅ **fixed** — frontend-agnostic calibration (each frontend runs its own `accumulate/finalize`); `learnable`/`adaptation_regularization`/`adaptation_summary` on the SSM; `eval_transfer.build_encoder` reconstructs the actual frontend; CLI accepts `mamba`. 5 integration tests (`test_pretrain_mamba_integration.py`). |
| 3 | blocker | **Backward graph too large / slow scan.** | ✅ **fixed** — official fused **`mamba_ssm` CUDA kernel** (`use_kernel`, auto on CUDA) + pure-PyTorch checkpointed fallback (CPU/tests). Verified identical (max diff 3e-8) and **9.7× faster** fwd+bwd. Needs `causal-conv1d`+`mamba-ssm`. Full-scale training smoke not yet run. |
| 4 | high | **Rate-invariance over-claimed** ("by construction", 5.6×). Real-window advantage ~1.0×; conv/skip rate-dependent. | ✅ **claim corrected** (physics note). Making it structural (physical-time conv / bounded Δ) is an open design choice. |
| 5 | high | **Δ multiplier unbounded** at train time. | ✅ **addressed** — `adaptation_regularization` softly pulls the multiplier baseline toward the physical clock (1/rate); `adaptation_summary` monitors its distribution. |
| 6 | high | **Per-modality norm assumes canonical channel order** (0:3 accel, 3:6 gyro); a permuted/3-channel input misassigns/raises. | ⚠️ **documented constraint** — Arm B requires the canonical 6-channel layout, which the corpus always provides (pad+mask). General fix (modality id per channel) deferred; not needed for this corpus. |
| 7 | medium | **A1 objective target is the filterbank's features** → Arm B trained to reproduce Arm A. | ✅ **addressed** — `--a1-weight 0` runs an objective-neutral comparison on A2 (SupCon) + A3 (grounding), both frontend-agnostic. Report with A1 off (and optionally on, disclosed). |
| 8 | medium | Protocol repro: Mamba settings absent from `PretrainConfig`/checkpoints. Params: frontend **25.3k vs 194.3k (7.67×)**; full encoder **+2.35%**. | ✅ **config stamped** (`mamba_d_state/d_inner/d_conv/scan_chunk`). Remaining: matched multi-seed runs + disclose the param gap when reporting. |

**Verdict: reachable and integrated; one perf caveat before a full-scale run.** Arm B now trains
through the harness with objective-neutral option, bounded backward memory, monitored physical clock,
and a reproducible checkpoint. The remaining gate is **scan throughput** (Python loop) at batch 512 —
a parallel scan / kernel, or a reduced-batch run, is needed to make the comparison fast; a full-scale
training smoke has not yet been run, so no trained Mamba checkpoint exists and representation quality
is still unmeasured. The subset + metric suite (above) are live and the Arm A baseline is measured.
