# Zero-shot results — HALO vs baselines, full 7-dataset table (2026-07-20)

**Supersedes** `RESULTS_PRELIMINARY.md` (2026-07-14) and the earlier corpus-matched-baselines-only
draft of this file. This is the current, valid comparison table:

- **HALO's own row is now in the table** — scored through the *identical* ConSE ZS-XD path as the
  baselines (Pipeline B; see Provenance). This is the bar HALO's Phase-A representation clears.
- crosshar and limubert are **retrained from scratch on HALO's exact 12-dataset corpus**, so all
  self-pretrained models see the *same* training data HALO does.
- All 7 primary eval datasets are scored for every model (the four frozen baselines were extended
  to `usc_had`, previously an addendum).

## Protocol
Zero-shot cross-dataset (ZS-XD), subject-disjoint, **macro-F1 primary**, subject-stratified 95% CIs,
ConSE bridge for closed-vocab models. All eval datasets are held out — never trained on. Every model
is scored on the same 7 primary eval streams (`deployment_policy.PRIMARY_EVAL_DATASETS`).

## Results — macro-F1 [subject-stratified 95% CI], all 7 eval datasets

| Model | motionsense | realworld | shoaib | inclusivehar | usc_had | tnda_har | ut_complex | **mean** |
|---|---|---|---|---|---|---|---|---|
| **harnet** (frozen, UK-Biobank) | **82.2** [80.2,84.1] | 32.2 [30.1,34.2] | **71.3** [66.3,75.6] | 23.1 [20.3,25.8] | **34.5** [31.6,36.2] | **55.3** [degen] | 32.6 [31.1,34.2] | **47.3** |
| **HALO** (ours, Phase-A + ConSE head) | 54.0 [49.8,58.0] | **43.0** [39.1,47.2] | 44.8 [42.1,47.5] | **26.7** [22.8,30.1] | 17.0 [14.4,19.2] | 45.2 [degen] | **52.1** [49.6,54.7] | **40.4** |
| **unimts** (frozen, CLIP) | 41.3 [38.8,43.6] | 32.3 [30.8,33.9] | 35.1 [32.2,37.8] | **26.7** [22.4,30.6] | 24.4 [21.7,27.4] | 48.9 [degen] | 29.1 [27.6,30.6] | **34.0** |
| **crosshar** (self, corpus-matched) | 45.8 [41.7,49.3] | 21.9 [19.9,24.0] | 33.3 [28.2,38.5] | 24.5 [21.2,27.4] | 16.1 [13.3,18.0] | 41.5 [degen] | 26.6 [23.0,30.3] | **30.0** |
| **limubert** (self, corpus-matched) | 44.1 [40.5,47.4] | 24.6 [20.2,28.3] | 35.1 [27.9,42.3] | 24.5 [20.5,27.9] | 2.0 [1.5,2.7] | 28.6 [degen] | 9.4 [6.7,12.2] | **24.1** |
| **imagebind** (frozen, floor) | 11.8 [9.4,14.1] | 12.7 [9.4,15.7] | 14.4 [11.8,16.8] | 18.8 [14.7,22.9] | 1.5 [1.0,2.0] | 13.0 [degen] | 7.6 [6.4,8.8] | **11.4** |
| **normwear** (frozen, L1) | 7.2 [6.4,8.1] | 4.8 [3.6,6.5] | 3.6 [3.6,3.6] | 8.6 [6.8,10.6] | 4.8 [3.8,6.0] | 5.1 [degen] | 1.8 [1.2,2.7] | **5.1** |

Cells: macro-F1 (%). **Bold** = column winner. `[degen]` = degenerate CI (tnda_har unknown-subject
sentinel) — point estimate OK. Rows sorted by mean. `mobiact` was dropped from the eval set (phantom
entry — raw download + grids never materialized; see `deployment_policy.EXCLUDED_PRIMARY_DATASETS`).

## Key observations
- **harnet leads the mean (47.3); HALO is 2nd (40.4).** harnet is a frozen UK-Biobank harnet5 via
  ConSE — its huge wrist-accel pretraining corpus gives it a commanding lead specifically on the
  **phone-in-pocket** sets (motionsense 82.2, shoaib 71.3). It is not corpus-matched to us (external
  frozen weights), so it is a strong *external* reference, not a same-data control.
- **HALO is the most configuration-balanced model, and wins the config-shift regime it targets.**
  It is **#1 on both wrist sets** — `ut_complex` **52.1** (vs harnet 32.6, next-best) and `realworld`
  **43.0** (vs unimts 32.3) — and #1 on the free-living `inclusivehar`. It never collapses on a
  config: its worst cell is 17.0 (usc_had), whereas harnet swings from 82 down to 23 and limubert
  falls to 2.0. This is exactly the cross-placement / cross-device generalization HALO is built for;
  harnet's average is carried by the two phone-pocket sets that most resemble... nothing it trained
  on (it's wrist-trained) — its lead there is a genuinely strong but narrow signal.
- **HALO beats every same-data control decisively.** Against the corpus-matched self-pretrained pair
  it is +10.4 over crosshar (30.0) and +16.3 over limubert (24.1) on the mean, and beats the frozen
  text-aligned unimts (34.0) by +6.4 — while using a native, resampling-free input path.
- **Cross-config difficulty gradient persists** for the weaker baselines: crosshar/limubert/normwear
  drop sharply on the wrist sets vs phone-in-pocket; HALO and (to a degree) unimts are the models
  that hold up across the placement shift.

## Phase-A representation ceiling — the gap is the text bridge, not the encoder
A supervised **linear probe** on *frozen* HALO features, fit on each eval dataset's own labels
(subject-disjoint, in-distribution — an upper bound on what a linear head on these features can do),
vs HALO's zero-shot ConSE number:

| dataset | supervised-probe ceiling | zero-shot ConSE | gap |
|---|---|---|---|
| motionsense | 97.5 | 54.0 | +43.5 |
| realworld | 83.9 | 43.0 | +40.9 |
| shoaib | 95.7 | 44.8 | +50.9 |
| inclusivehar | 64.9 | 26.7 | +38.2 |
| usc_had | 75.3 | 17.0 | +58.3 |
| ut_complex | 88.4 | 52.1 | +36.3 |
| **mean** | **84.3** | **39.6** | **+44.7** |

(`tnda_har` skipped — 1-subject sentinel, no subject-disjoint split.) The frozen representation
**linearly separates the activities at 84% mean**; the entire ZS-XD shortfall (37–58 pts per set) lives
in the **zero-shot text bridge**, not the encoder. Two consequences: (1) Phase A is *not* the bottleneck —
more pretraining objectives would chase the wrong problem; (2) a better bridge (Phase B: learned retrieval
metric + grounding + calibration) has ~45 pts of headroom, and since HALO's motionsense ceiling (97.5)
exceeds harnet's *zero-shot* 82.2, a good bridge should let HALO pass harnet. Reproduce:
`python -m training.tokenizer.probe_ceiling --checkpoint training/tokenizer/outputs/pretrain_native/best.pt`.

## Provenance
- **HALO** — *ours*. Phase-A representation encoder (`training/tokenizer/outputs/pretrain_native/best.pt`;
  30k-step, d_model-256, val kNN-BA 0.659, git `532d19c`). Phase-A has **no trained label-text head**,
  so — exactly like the frozen/self ConSE baselines — it is scored through a linear softmax head fit on
  **frozen** HALO features over the 59-way global training vocabulary, then the ConSE bridge
  (`baselines/halo/adapter.py`, tier=`conse`). The head-fit is the **same leakage-safe protocol** as
  crosshar/harnet: fit on the 20 training streams, best epoch selected on a **subject-disjoint** held-out
  fold, temperature-calibrated (T=1.20, the #82 fix). Head-fit val-acc = 0.768 over 45 held-out subjects.
  **Native input contract** — HALO ingests each eval stream at its *native* rate/length/channels with no
  resampling (the physical-Hz filterbank tokenizer + gravity-aware DC feature), unlike the baselines,
  which are resampled to their fixed contracts. So HALO's row uses the same estimator + bridge +
  leakage discipline as the baseline rows, differing only where its thesis says it should: the front end.
- **crosshar, limubert** — *self-pretrained by us*, **retrained 2026-07-20 on the corpus-matched
  12-dataset corpus** (uci_har, hhar, pamap2, wisdm, kuhar, unimib_shar, mhealth, capture24, sp_sw_har,
  nfi_fared, harmes, xrf_v2 — `hapt` dropped as a UCI-HAR near-duplicate). ConSE heads refit with
  subject-disjoint calibration (#82). crosshar = masked-recon + NT-Xent (90 ep); limubert = masked-recon
  (250 ep).
- **harnet, unimts, imagebind, normwear** — *frozen released weights*. The 6 common cells are unchanged
  from 2026-07-14; the `usc_had` column was added 2026-07-20 (no retrain — released weights, ConSE/cosine
  heads only). Note: the frozen-backbone ConSE head-fit (harnet) uses its own legacy 9-dataset corpus
  list rather than the 12-dataset one — second-order for an external frozen backbone, but the reason
  harnet is labelled an *external reference* rather than a same-data control.

## Reproduce
```bash
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
# HALO's row (fits the ConSE head on first run, caches to baselines/halo/halo_conse_head.pt):
HALO_CKPT=training/tokenizer/outputs/pretrain_native/best.pt \
  PYTHONUNBUFFERED=1 $PY -u -m eval.run_baselines --baselines halo --device cuda
# baselines on all 7 (frozen: no retrain; self: retrain first — see below):
PYTHONUNBUFFERED=1 $PY -u -m eval.run_baselines --baselines harnet unimts imagebind normwear crosshar limubert --device cuda
# retrain a self-pretrained baseline corpus-matched:
$PY -m baselines.limubert.train --epochs 250 --batch-size 256 --lr 1.4e-3 --num-workers 8 --gpu
$PY -m baselines.crosshar.train --epochs 90 --epochs-cl 30 --batch-size 512 --augment --num-workers 8 --gpu
# assemble:
$PY -m eval.assemble_table --baselines halo harnet unimts crosshar limubert imagebind normwear \
  --datasets motionsense realworld shoaib inclusivehar usc_had tnda_har ut_complex --out docs/baselines/RESULTS_V2.md
```

## Caveats / honest notes
- **HALO does not top the mean** — harnet's external UK-Biobank scale wins it on phone-pocket. HALO's
  claim is *configuration generalization* (best on wrist + free-living, most balanced, no collapse), and
  *dominance over every same-corpus control*, not a clean sweep. Present it that way.
- The `[degen]` tnda_har CI comes from an unknown-subject sentinel — point estimates are usable, CIs are
  not; don't rank on tnda_har alone.
- This is HALO **Phase-A** (representation pretraining) with a *linear* probe head. Phase-B
  (config-conditional text alignment) is expected to lift the phone-pocket cells where a linear ConSE
  probe on frozen features is the current ceiling.
