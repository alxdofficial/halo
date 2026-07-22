# Tokenizer ablation — spectral filterbank vs learned rate-aware SSM

> **Status: design + Arm B built, 2026-07-22.** The decisive head-to-head for the feature-extraction
> question raised in the literature review (`POSITIONING.md` §10-adjacent): is our handcrafted
> physical-Hz filterbank leaving performance on the table versus a learned front end? Both arms share
> the drop-in contract `forward(patches (B,P,S,C), rate, N) -> (B,P,C,d)`, so the encoder body,
> text conditioning, augmentations, objective and budget are **identical** — only the front end
> changes (`SetTokenizerEncoder(frontend=...)`, via `scattering.build_frontend`).

## The two arms

| | Arm A — `frontend="fixed"` | Arm B — `frontend="mamba"` |
|---|---|---|
| method | multi-scale physical-Hz constant-Q filterbank | per-channel selective SSM on raw native-rate signal |
| rate handling | rate-invariant by construction (bands fixed in Hz) | **Δ = (1/rate)·softplus(·)** — physical discretisation step |
| prior | strong handcrafted spectral prior | learned dynamics, minimal prior |
| observability | explicit Nyquist mask + resolution flag | implicit (model must infer from Δ) |
| gravity/DC | explicit signed-DC feature | preserved in raw signal (see normalisation) |
| file | `model/tokenizer/filterbank.py` | `model/tokenizer/mamba_frontend.py` |

**Arm B — the physics.** An SSM is a discretised continuous ODE `dh/dt = A h + B u`; its step Δ is
the physical time between samples. We set the base step to `1/rate` and let the selective term be a
**dimensionless multiplier ≈1** around it. So the same motion at 20 vs 100 Hz advances the state at
the same physical speed and integrates to the same state over the same physical window —
rate-invariance by construction, the SSM analogue of the filterbank's physical-Hz bands. **Measured:**
the same 2 Hz oscillation sampled at 50 vs 100 Hz gives tokens **5.6× closer** under the correct
native rate than a wrong-rate control (`tests/test_mamba_frontend.py`). Rate is the discretisation
step itself, not a side conditioning token the model must learn to read.

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

## Data subset for the head-to-head (TO BE CONFIRMED)

Pretraining both arms on the full corpus twice is wasteful for a first read. Select a subset that:

- **spans the rate range** (20 / 50 / 100 Hz) — otherwise Arm B's rate-invariance is never exercised;
- **spans placements/devices** — so the held-out-config transfer eval is meaningful;
- is **small enough** to pretrain both arms quickly, large enough to be non-trivial.

Candidate: a handful of datasets covering all three rates and ≥3 placements, held-out-config eval on
the standard ZS-XD cells (or the internal subject-disjoint val-kNN for a fast inner loop). **Exact
subset to be chosen with the user before spending GPU.**

## Decision criterion (pre-registered)

Report internal val-kNN balanced accuracy and ZS-XD macro-F1 for both arms, with parameter counts and
wall-clock. **If Arm B does not beat Arm A on held-out configs by more than seed noise, the filterbank
stands** (with the Ravì-2016 rate-invariance-in-the-spectral-domain justification). If Arm B wins,
switch, then test per-sensor Arm B. Either outcome is publishable as a measured tokenizer choice.

## Perf caveat (Arm B)

The selective scan is currently **sequential** (exact, portable, kernel-free) — fine for tests/small
runs, O(S) Python steps. Full-corpus pretraining should first swap in an associative parallel scan or
the `mamba_ssm` CUDA kernel (not installed; a build step). The module interface is unchanged by that,
so it does not affect the comparison design — only its wall-clock.
