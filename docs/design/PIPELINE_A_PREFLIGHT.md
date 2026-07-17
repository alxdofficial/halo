# Pipeline A — Phase-1 pre-flight audit & decisions

> Consolidated record of the pre-training audit (2026-07-18) so we can reason about the
> corpus, objectives, hyperparameters, and the bugs we fixed without re-deriving them.
> Code: `training/tokenizer/` (harness + diagnostics), `model/tokenizer/` (the encoder).
> Status: **all gates PASS; launch held on GPU availability + explicit go.**

---

## 1. Corpus (the pretraining data)

Balanced train corpus = the 8 non-eval datasets' **harmonised** grids (60 Hz · 6 s · 6 canonical
channels `[acc xyz, gyro xyz]`, pad+mask), capped at **20k windows per stream**.

| dataset | windows used | subjects | share of train |
|---|---|---|---|
| wisdm (2 streams) | 40,000 | 51 | 38 % |
| capture24 | 20,000 (of 144k) | 151 | 18 % |
| unimib_shar | 11,771 | 30 | 11 % |
| uci_har | 10,299 | 30 | 10 % |
| hhar | 9,587 | 9 | 9 % |
| kuhar | 9,333 | 89 | 9 % |
| pamap2 / mhealth | 4,290 | 19 | ~5 % |
| **total** | **~105,280** | | |

- **96k train / 11k val**, subject-disjoint, **57 labels**. Loading verified: 0 NaN / 0 Inf; all
  accel unit-canonicalized to g (|gravity|≈1; kuhar 0.057 = documented gravity-removed).
- **`hapt` DROPPED** — it is the UCI-HAR re-release (same 30 subjects/recordings, per-window
  NCC 0.98 vs uci_har), so keeping both leaked near-duplicate val windows into train.
- **wisdm at 38 %** is per-*stream* capping (phone + watch are genuinely distinct configs); kept
  for baseline parity. Per-dataset capping is an option if this proves to over-weight it.
- **Label imbalance** max/min ≈ 5015 (walking 10k … tiny fall classes). The balanced sampler
  neutralizes this at batch level; labels with **< 8 windows are excluded from anchoring** (they
  can't form real SupCon positives).

## 2. The elite-3 objectives — final config & rationale

Two forwards per step: **masked → A1**; **clean → A2 + A3** (contrastive must not fight mask noise).
Gravity-align is always applied (per window, on real-length data). A1 targets come from a **frozen,
calibrated** filterbank; the calibration excludes absent channels + phantom patches.

| knob | value | rationale |
|---|---|---|
| `patch_seconds` (multi-scale) | `{0.5, 0.75, 1.0, 1.5}` → T∈{12,8,6,4} | per-batch draw; 2.0/T=3 dropped (too coarse to mask, collapses short windows) |
| A1 mask ratio (time) | 0.5 | MAE/JEPA-style; latent-space target |
| A1 channel-event p / gyro-bias | 0.25 / 0.7 | whole-channel drops biased to the gyro triad (the real deployment shift) |
| A1 causal fraction | 0.3 | the world-model / streaming variant |
| A2 SupCon temperature | 0.1 | Khosla et al. |
| A2 batch | 32 classes × 8 = 256 | guarantees 7 positives/anchor |
| A3 weight | 0.1 | grounding **rail**, not a driver |
| gravity-aug p | 0.15 | was 0.5 → removed gravity on 52 % of windows (killed the M0 features) |
| RoPE periods | 0.5 – 600 s | finest patch ↔ session span; physical seconds, never index |
| mask_token init | N(0, 0.02) | MAE-style (was zeros = dead-symmetric) |
| optim | AdamW 3e-4, wd 0.05, warmup 1k, cosine, clip 1.0 | standard |

**Model size: `d_model=256, 6 layers, 8 heads, FFN 1024` (~7.3M).** Serves all three consumers
(tokenizer rep, A1/A2/A3 heads, evidence-engine multi-vector memory 8×32), data-appropriate for 96k
windows (well below the shortcut-prone 20M legacy), ~70 min / 20k-step run. **Committing**: the
frozen encoder's `d` sets the memory-bank vector width — changing later = retrain A + rebuild memory.

## 3. Reusable diagnostics (re-run before any future training)

Env: `legacy_code/.venv/bin/python` (has torch/scipy; system python3 does not).

| script | what it checks | gate |
|---|---|---|
| `training/tokenizer/corpus_audit.py` | volume · imbalance · loading integrity · realized aug/task distribution | manual review |
| `training/tokenizer/grad_check.py` | activation RMS per stage (fusion balance) · per-module + per-layer grad norms · dead params · frozen-text invariant | PASS/FAIL |
| `training/tokenizer/objective_health.py` | min/med/max of A1 supervised-tokens, A2 positives/anchor, A3 valid-targets/batch | PASS/FAIL |
| `training/tokenizer/aug_inspect.py` | visual before/after of each aug axis + multi-scale patch grid | eyeball |
| `training/tokenizer/m2_gate.py` | tiny-scale elite-3 transfer gate (held-out config) | PASS/FAIL |

## 4. Findings & fixes ledger

The pre-flight ran three passes: (a) my corpus/HP/profiling audit, (b) the **adversarial debug
sweep** (33 agents, 24 confirmed findings, all verified), (c) the **objective-health** check. The
sweep caught 3 HIGH bugs that (a) missed — this is why we gate on it.

### HIGH (would have crippled the run)
1. **Gravity-align was a ~96 % no-op.** `align_batch` estimated gravity on the *zero-padded* patch
   buffer (padding diluted the low-pass below the 0.5 g threshold) and rotated each patch
   independently. The M2 lesson-3 canonicalization (+0.13 BA in the gate) was dead in the pipeline.
   → Fixed: one rotation per window, on real-length data, in `MultiScaleCollate` before patchify.
2. **Phantom all-zero patches.** Rate-aug rounding left `usable < P`; with no `patch_padding_mask`,
   these entered A1 loss / pooling / RoPE as real. → collate emits `patch_padding_mask`, threaded
   through encoder + A1 + calibration.
3. **A1 trained on absent channels + phantom patches** (~36–45 % of A1 terms were "predict the
   zero-padding signature"). → `a1_mask = token_mask & channel_mask & patch_pad`.
4. **A1 zero-supervision on 14.2 % of windows** (objective-health). `make_mask_plan` was
   validity-blind, so masks landed entirely on absent/phantom tokens. → validity-aware mask plan
   (block on real patches, drops on real channels; guarantees ≥1 real masked token for T≥2). Plus
   dropped `patch_seconds=2.0`. Residual 0.76 % = genuinely un-maskable 1-patch windows.
5. **Val kNN evaluated only capture24** (stream-ordered val + 2k truncation → 8 of 56 labels
   selected best.pt). → `index.val` shuffled so any truncated subset is cross-dataset.

### MEDIUM
6. **Norm calibration folded in zero absent channels + phantom patches** (mis-scaled every band).
   → `accumulate_norm_stats` now takes `channel_mask` + `patch_mask`.
7. **uci_har ≡ hapt leak** → hapt dropped (see §1).
8. **Cadence fabricated ~4 Hz on aperiodic motion** (boundary argmax). → require an interior local
   maximum; reject boundary peaks.
9. **Gravity aug p=0.5** removed gravity on 52 % of windows → 0.15.
10. **A1 loss on absent gyro wasted budget** (the channel half of #3) — same fix.

### LOW
- Filterbank DSP forced to **fp32 under autocast** (fp16 band-energy headroom).
- Collate `patch_seconds` RNG was cloned identically across workers → now per-batch (content-seeded).
- `mask_token` zeros → N(0, 0.02).
- Text-embedding assembly dedupes to unique strings (profiling: forward 157→142 ms/step).

### Refuted (1)
- One sweep finding was refuted on verification (not a real bug).

## 5. Health snapshot at launch-readiness

- **grad_check: PASS** — sensor RMS 0.55, fusion contributes 18 % of scale (balanced), transformer
  layer grads 4.9→3.2 (no vanish/explode), mask_token grad live, **0 dead params**, text LM frozen.
- **objective_health: PASS** — A1 supervised tokens/window med 9 (0.76 % un-maskable), A2 7/7
  positives/anchor (0 starved), A3 cadence 47–89 valid/batch, eigen 256/256.
- **profiling** — 142 ms/step at d192; d256 ≈ 210 ms/step est. → ~70 min / 20k steps. Backward is
  the floor (42 %); data loading is not a bottleneck (< 1 % wait).
- **181/181 unit tests** green.

## 6. Deferred / open (not blockers)

- **Model-scale sweep** — d256 chosen by reasoning; a d192-vs-d256-vs-d384 A/B is a later ablation.
- **wisdm per-stream vs per-dataset cap** — revisit if 38 % over-weights it.
- **Run length** — 20k ≈ 48 epochs; a "final" run would extend with a retuned cosine.
- **EMA/JEPA latent teacher for A1** — current A1 target is the calibrated filterbank (MaskFeat-
  style, no-collapse). An EMA teacher is the upgrade path if A1 saturates.
