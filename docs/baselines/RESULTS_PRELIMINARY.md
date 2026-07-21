> ⚠️ **SUPERSEDED SNAPSHOT — pre-vocabulary-fix protocol.** Every number below was produced under the stale **59-label** global vocabulary, which silently discarded 11.48% of training windows. The vocabulary is now **93** labels and all ConSE heads + the memory bank are stale and awaiting refit. Do NOT mix these numbers with post-fix ones. See `docs/design/EVIDENCE_ENGINE_FINDINGS.md` (STATUS block) for the current position.

# Baseline zero-shot results — PRELIMINARY (2026-07-14)

> ⚠️ **PRELIMINARY — not final.** Real numbers, but several audit P1 fixes are NOT yet applied
> (see caveats). Do not cite. Establishes the *bar HALO must beat*; HALO itself is not built yet.

**Protocol:** zero-shot cross-dataset (ZS-XD), subject-disjoint, **macro-F1** primary,
subject-stratified 95% bootstrap CIs, ConSE bridge for closed-vocab baselines. Held-out eval
datasets (never trained on). Backbones: crosshar + limubert **self-pretrained on our balanced 107k
corpus, current committed code** (2026-07-14 overnight run); harnet/unimts/imagebind/normwear are
frozen released weights.

| Model | motionsense | realworld | shoaib | inclusivehar | ut_complex | tnda_har | **mean** |
|---|---|---|---|---|---|---|---|
| **harnet** (frozen, UK-Biobank) | **82.2** | 32.2 | **71.3** | 23.1 | **32.6** | **55.3** | **49.5** |
| **crosshar** (self, ours) | 49.4 | 30.0 | 47.8 | 25.8 | 16.9 | 44.4 | 35.7 |
| **unimts** (frozen, CLIP) | 41.3 | **32.3** | 35.1 | **26.7** | 29.1 | 48.9 | 35.6 |
| **limubert** (self, ours) | 35.5 | 20.3 | 22.0 | 12.2 | 7.2 | 27.2 | 20.7 |
| **imagebind** (frozen, floor) | 11.8 | 12.7 | 14.4 | 18.8 | 7.6 | 13.0 | 13.1 |
| **normwear** (frozen, L1) | 7.2 | 4.8 | 3.6 | 8.6 | 1.8 | 5.1 | 5.2 |

Cells: macro-F1 (%). CIs in the per-cell JSONs; `tnda_har` CIs are **degenerate** (unknown-subject
sentinel); `ut_complex` CIs are **suspect** (fabricated 10-subject split) — point estimates OK.

## Observations
- **harnet is the strongest baseline by a clear margin** (mean 49.5) — a frozen UK-Biobank harnet5 via
  ConSE. Big lead on phone datasets (motionsense 82.2, shoaib 71.3). The bar HALO must beat.
- **crosshar (35.7) ≈ unimts (35.6)** in the middle tier — and **crosshar now clearly beats limubert
  (20.7)**, both self-pretrained on our corpus (the contrastive phase + the P0 `.train()` fix helped).
- **imagebind (13.1) and normwear (5.2)** are the disclosed floors, as expected.
- Note the **cross-config difficulty gradient**: everyone drops hard on wrist datasets (ut_complex) and
  inclusivehar — exactly the heterogeneity axis the evidence-engine thesis targets.

## Caveats (why this is PRELIMINARY)
1. ✅ Q1 label-humanization fix IS applied (fair text across cosine baselines).
2. **usc_had excluded** — gyro-in-dps bug (P1) would produce a garbage row until `np.deg2rad` fix.
3. **mobiact excluded** — no grids (source/MobiFall confusion, P1).
4. Head checkpoint selected by **window-acc, not macro-F1** (P1) — may shift crosshar/limubert/harnet.
5. **No provenance stamps** on the result JSONs (P1).
6. These are uniform **ConSE fair-probes** (mean-pool + linear head), NOT reproductions of each paper's
   downstream classifier (Q2 disclosure).
7. **No HALO row** — HALO isn't built; this is the baseline field only.

Run: `python -m eval.run_baselines --baselines harnet unimts imagebind normwear crosshar limubert
--datasets motionsense realworld shoaib inclusivehar ut_complex tnda_har --device cuda`
