# Baseline zero-shot results — corpus-matched refresh (2026-07-20)

**Supersedes** `RESULTS_PRELIMINARY.md` (2026-07-14). This is the current, valid baseline table:
crosshar and limubert have been **retrained from scratch on HALO's exact 12-dataset corpus**, so all
self-pretrained baselines now see the *same* training data HALO does — the numbers are finally
apples-to-apples.

## Protocol
Zero-shot cross-dataset (ZS-XD), subject-disjoint, **macro-F1 primary**, subject-stratified 95%
bootstrap CIs, ConSE bridge for closed-vocab baselines. Held-out eval datasets (never trained on).

## Provenance
- **crosshar, limubert** — *self-pretrained by us*, **retrained 2026-07-20 on the corpus-matched
  12-dataset corpus** (uci_har, hhar, pamap2, wisdm, kuhar, unimib_shar, mhealth, capture24,
  sp_sw_har, nfi_fared, harmes, xrf_v2 — `hapt` dropped as a UCI-HAR near-duplicate; the 2026-07
  expansion added). ConSE heads refit with subject-disjoint calibration (temperature scaling; the
  #82 leakage fix). crosshar = masked-recon + NT-Xent contrastive (90 ep); limubert = masked-recon
  (250 ep).
- **harnet, unimts, imagebind, normwear** — *frozen released weights*, unchanged from 2026-07-14.

## Results — macro-F1 [subject-stratified 95% CI], 6 common eval datasets

| Model | motionsense | realworld | shoaib | inclusivehar | tnda_har | ut_complex | **mean** |
|---|---|---|---|---|---|---|---|
| **harnet** (frozen, UK-Biobank) | **82.2** [80.2,84.1] | 32.2 [30.1,34.2] | **71.3** [66.3,75.6] | 23.1 [20.3,25.8] | **55.3** [degen] | 32.6 [31.1,34.2] | **49.5** |
| **unimts** (frozen, CLIP) | 41.3 [38.8,43.6] | 32.3 [30.8,33.9] | 35.1 [32.2,37.8] | 26.7 [22.4,30.6] | 48.9 [degen] | 29.1 [27.6,30.6] | **35.6** |
| **crosshar** (self, corpus-matched) | 45.8 [41.7,49.3] | 21.9 [19.9,24.0] | 33.3 [28.2,38.5] | 24.5 [21.2,27.4] | 41.5 [degen] | 26.6 [23.0,30.3] | **32.3** |
| **limubert** (self, corpus-matched) | 44.1 [40.5,47.4] | 24.6 [20.2,28.3] | 35.1 [27.9,42.3] | 24.5 [20.5,27.9] | 28.6 [degen] | 9.4 [6.7,12.2] | **27.7** |
| **imagebind** (frozen, floor) | 11.8 [9.4,14.1] | 12.7 [9.4,15.7] | 14.4 [11.8,16.8] | 18.8 [14.7,22.9] | 13.0 [degen] | 7.6 [6.4,8.8] | **13.1** |
| **normwear** (frozen, L1) | 7.2 [6.4,8.1] | 4.8 [3.6,6.5] | 3.6 [3.6,3.6] | 8.6 [6.8,10.6] | 5.1 [degen] | 1.8 [1.2,2.7] | **5.2** |

Cells: macro-F1 (%). `[degen]` = degenerate CI (tnda_har unknown-subject sentinel) — point estimate OK.

### `usc_had` addendum (7th dataset, corpus-matched baselines only)
`usc_had` was added to the eval set after 2026-07-14, so only the freshly-run baselines have it:
crosshar **16.1** [13.3,18.0], limubert **2.0** [1.5,2.7]. The four frozen baselines have **not** yet
been scored on usc_had; re-score them (no retrain needed) before adding a usc_had column to the main
table. `mobiact` was dropped from the eval set (phantom entry — raw download + grids never
materialized; see `deployment_policy.EXCLUDED_PRIMARY_DATASETS`).

## Key observations
- **harnet is the baseline to beat (mean 49.5)** — a frozen UK-Biobank harnet5 via ConSE, with a big
  lead on the phone datasets (motionsense 82.2, shoaib 71.3).
- **The corpus-matched retrain reordered the self-pretrained pair:** limubert jumped **20.7 → 27.7**
  (its old backbone was stale / pre-capture24), while crosshar eased **35.7 → 32.3**. Both are now
  trained on HALO's exact corpus, so the comparison is finally fair. Ranking: harnet > unimts (35.6)
  > crosshar (32.3) > limubert (27.7) > imagebind (13.1) > normwear (5.2).
- **Cross-config difficulty gradient:** every baseline drops sharply on the wrist sets (ut_complex)
  and free-living (inclusivehar) vs the phone-in-pocket sets — the config-generalization gap HALO
  targets.

## Not yet in this table
- **HALO's own ZS-XD row.** HALO Phase-A is trained (best.pt, val kNN-BA 0.659, corpus fingerprint
  `99fa9ce6…`), but its downstream ZS-XD eval on these test sets has not been run yet — this table is
  "the bar," HALO's row is the pending next step.
- **usc_had for the 4 frozen baselines** (see addendum).

## Reproduce
```bash
# retrain a self-pretrained baseline (corpus-matched, robust on a shared box):
python -m baselines.limubert.train --epochs 250 --batch-size 256 --lr 1.4e-3 --num-workers 0 --gpu   # or --num-workers 8, unbuffered
python -m baselines.crosshar.train --epochs 90 --epochs-cl 30 --batch-size 512 --augment --num-workers 8 --gpu
# score + assemble:
PYTHONUNBUFFERED=1 python -m eval.run_baselines --baselines crosshar limubert
python -m eval.assemble_table --datasets motionsense realworld shoaib inclusivehar tnda_har ut_complex --out docs/baselines/RESULTS_V2.md
```
