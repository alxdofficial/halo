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
| method | multi-scale physical-Hz constant-Q filterbank | per-channel **3-layer stacked** selective SSM on raw native-rate signal |
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

**Depth (Arm B):** the tokenizer is now a **stack of 3 residual Mamba blocks** per channel
(`mamba_n_layers`, default 3) — a single block is a weak extractor (~one adaptive filter + gate); the
stack gives hierarchical nonlinear *intra-patch* feature extraction (local oscillation → cycle →
segment) before pooling to one token. Inter-patch temporal modelling stays in the shared transformer.
A deep Mamba that *owns* the temporal modelling (over the full sequence) is a larger, separate
experiment — flagged, not built.

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

> ⚠️ **The earlier "Arm A baseline" numbers are RETRACTED as a control** (2nd tokenizer audit). They
> were run on `pretrain_fixed_mr`, which trained on the **FULL corpus INCLUDING xrf_v2** (xrf is in
> `TRAIN_DATASETS`). So its "held-out xrf" figures were only *subject*-held-out, not
> dataset/config-held-out — invalid as a transfer control. And `pretrain_fixed_mr` used A1 on / full
> corpus / 25k steps, i.e. a different protocol than the intended subset + `a1_weight=0` Arm B. **A
> valid comparison needs FRESH matched runs of BOTH arms** on the subset (xrf genuinely held out),
> same seed(s), same budget, same objective. The retracted numbers (kNN purity 0.869 val; rate/place
> decode 1.00/0.997) are kept only as a rough sanity signal that the suite runs and that the
> filterbank *does* leak config — not as the bar.
>
> Also note the metric confounds (2nd audit #7), disclosed in `eval/tokenizer_metrics.py`:
> **rate/placement decodability is confounded with dataset identity** (each rate maps to specific
> datasets, so "rate decode 1.00" may be reading dataset, not rate); purity/retrieval are now
> **macro-averaged** by label; xrf cross-placement retrieval is **paired same-session** evidence, not
> independent cross-dataset generalization. Interpret decodability alongside transfer, never alone.

## Decision criterion (pre-registered)

Report internal val-kNN balanced accuracy and ZS-XD macro-F1 for both arms, with parameter counts and
wall-clock. **If Arm B does not beat Arm A on held-out configs by more than seed noise, the filterbank
stands** (with the Ravì-2016 rate-invariance-in-the-spectral-domain justification). If Arm B wins,
switch, then test per-sensor Arm B. Either outcome is publishable as a measured tokenizer choice.

## Perf (Arm B) — resolved

The selective scan now runs on the **official `mamba_ssm` CUDA kernel** (`selective_scan_fn`), built
from source (`causal-conv1d` 1.6.2, `mamba-ssm` 2.3.2). Measured **9.7× faster** than the reference
path and **bit-identical** to it (max abs diff 3e-8 single-block, 3.6e-7 across the 3-block stack).
The kernel carries its own memory-efficient backward, so the old backward-recurrence blow-up is gone.

For CPU / no-kernel environments there is a pure-PyTorch fallback: a **chunked, gradient-checkpointed**
scan (`scan_chunk=32`) so the backward graph never materialises all S steps at once. `pretrain.py`
**fails loud** (`RuntimeError`) if `frontend="mamba"` is requested on CUDA without the kernel, or on CPU
outside a smoke run — no silent fallback to the slow path in a real run.

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
| 8 | medium | Protocol repro: Mamba settings absent from `PretrainConfig`/checkpoints. | ✅ **config stamped** (`mamba_d_state/d_inner/d_conv/scan_chunk`). Param gap corrected below (#2b); matched multi-seed runs still required. |
| 9 | blocker | **Calibration device mismatch** (found in the 2026-07-22 readiness sweep): `accumulate_norm_stats` built its weight tensor on CPU while patches/buffers were on CUDA → `RuntimeError` on the *first* real GPU run. CPU-only unit tests could not catch it. | ✅ **fixed** — every calibration factor now follows `patches.device`; caught only by an actual `--device cuda` run. |
| 10 | blocker | **Eval OOM** (readiness sweep): the per-channel scan runs `M=B·P·C` sequences at once; the val embedder's `B=256` block → `M≈24.6k`, `S=256` → multi-GB kernel intermediates → 21 GiB / OOM on a 24 GB card. Training (`B=32`) fit, so it only surfaced at validation. | ✅ **fixed** — chunk M via `forward_chunk` (each sequence is independent → **bit-exact**, verified 0.0 diff). See #1b: the default had to become *auto* to bound production memory. |

### Second GPU audit (independent review, 2026-07-23 — all verified here on-GPU)

This review correctly caught that the 2026-07-22 "GO" was **premature**: the checkpointing that was
supposed to bound training memory *never fired*, and several protocol knobs made the comparison
scientifically ambiguous. All confirmed empirically and fixed.

| # | severity | issue | status |
|---|---|---|---|
| 1b | **blocker** | **Training memory was NOT bounded.** The `forward_chunk` checkpoint gate was `seq.requires_grad`, which is **False** for raw sensor inputs → checkpoint fired **0 times** (instrumented); real-path backward peaked **21 GiB at tiny d=64/M=4224**, and the production batch OOM'd on the first kernel call. The prior "0.0 diff" check masked it by forcing `requires_grad_(True)`. | ✅ **fixed** — gate on train+grad+`any(param.requires_grad)` (`use_reentrant=False` differentiates params even when no input needs grad); **`forward_chunk` default is now auto** (`16M // (d_inner·S)`, ~2.5 GiB/chunk) since peak is *linear in chunk·d_inner·S* and a fixed 4096 OOMs at d=256. Measured real-path backward peak **~2.8 GiB** at M=45k/d=256, invariant to batch; still bit-exact. |
| 2b | high | **Arms are not capacity-matched, and the doc understated it.** At the production `d_model=256`: frontend **1,368,832 vs 25,344 (54.0×)**; full trainable **8,627,664 vs 7,284,176 (+18.4%)**. Doc still cited 7.67× / +2.35% (a smaller-width figure). | ✅ **doc corrected** (these numbers verified locally). Per the user's call (param-matching is not a priority at this absolute size), this is framed as a **method+capacity** comparison — report both counts; a param-matched filterbank control is optional. |
| 3b | high | **`--subset` did not apply the documented cap** (`DEFAULT_CAP=10_000`): it set only dataset names, so training used ~94k windows while the metric harness capped at 10k — train and eval used *different* corpora. | ✅ **fixed** — `--subset` now sets `max_per_stream=DEFAULT_CAP`; added `--max-per-stream` override. Train and eval share one corpus definition. |
| 4b | high | **Mamba SSM params got AdamW weight decay.** `A_log`, `D`, `dt_proj.bias` fell in the `wd=0.05` group; official mamba marks `A_log`/`D` `_no_weight_decay`; ~0.53× shrinkage over 30k steps corrupts SSM timescales/skip. | ✅ **fixed** — dedicated no-decay group for those params (`dt_proj.bias` too — decaying it fights the Δ-baseline regularizer). |
| 5b | high | **kNN support bank was non-deterministic.** `train_eval_loader` used `shuffle=True` with no `generator=`, so the support set differed per eval **and per arm** → matched training seeds ≠ matched validation. | ✅ **fixed** — fixed `torch.Generator(seed=cfg.seed)` on that loader; support is now identical across arms/evals. |
| 6b | high | **Resume could corrupt a run.** Only `frontend`/`multiresolution` validated → omitting `--subset`/`--a1-weight 0` on resume silently continued on the full corpus with A1 on; and `best_ba` was read from the resumed ckpt's own `val_ba`, so a resume from `last.pt` could overwrite a better `best.pt`. | ✅ **fixed** — validate `train_datasets/a1_weight/max_per_stream/d_model/num_layers` + corpus fingerprint on resume; persist and restore the running `best_ba`. |
| 7b | medium | **Narrow eval evidence:** only 10/46 val labels support cross-rate retrieval; xrf transfer probe shares 9/28 labels; xrf placement retrieval benefits from synchronized streams. | ⚠️ **disclosed** (retraction block above + `eval/tokenizer_metrics.py`). The pre-registered ZS-XD macro-F1 still lives outside the suite — run both ckpts through the HALO baseline adapter (separate cache paths) as the headline. |
| 8b | medium | **Remote repro:** local commit ahead of `origin/main`; `pyproject.toml` didn't declare `mamba-ssm`/`causal-conv1d`. | ✅ **deps added** (`[project.optional-dependencies] mamba`, pinned + build note). **Must `git push`** before a pod clones; the GO is conditional on that. |

**Revised verdict (2026-07-23): GO for a matched run, with the framing caveats disclosed.** The
memory blocker is genuinely fixed — a real default-width (`d_model=256`, batch 512) mamba training
step **plus** validation runs on the 4090 with peak recorded (`peak_gib` in the val log), and the
per-chunk auto-sizing holds the frontend backward peak at ~2.8 GiB regardless of batch. The protocol
knobs (cap, decay groups, support determinism, resume guards) are fixed so the two arms are now truly
matched except for the **deliberate** method+capacity difference (#2b), which must be reported. Before
launching: `git push`; then run ≥2 matched seeds per arm, and add the ZS-XD macro-F1 head-line (#7b).

**Verified launch recipe** (`--subset` now caps to 10k/stream; both arms objective-neutral, xrf held out):

```bash
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
$PY -m training.tokenizer.pretrain --frontend fixed --subset --a1-weight 0 --seed 0 --device cuda --out outputs/abl_fixed_s0
$PY -m training.tokenizer.pretrain --frontend mamba --subset --a1-weight 0 --seed 0 --device cuda --out outputs/abl_mamba_s0
$PY -m eval.tokenizer_metrics --checkpoint outputs/abl_fixed_s0/best.pt --out outputs/abl_fixed_s0/metrics.json
$PY -m eval.tokenizer_metrics --checkpoint outputs/abl_mamba_s0/best.pt --out outputs/abl_mamba_s0/metrics.json
# repeat with --seed 1 (and 2) for both arms; report mean±sd + both param counts + the #7b confounds
```
