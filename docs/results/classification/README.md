# Classification-era results (pre-pivot)

These documents describe the HALO classification line as it stood before the 2026-08-30 pivot to
motion-monitoring applications. They were removed from `main` by that pivot and are restored here,
unedited, as the provenance trail for numbers the IMWUT paper refers to.

**Do not edit them.** They record what was measured at the time, under the protocols named inside
them. Where a number here disagrees with a newer measurement, the newer one wins and the
disagreement belongs in the new report, not in an edit to these files.

| file | what it records |
|---|---|
| `ADAPTATION_TABLE_20260822.md` | the compact engine vs baselines across k on the `adaptation_v1` manifest |
| `ENCODER_COMPARISON_20260822.md` | frozen-encoder probe comparison |
| `EVAL_HARNESS_AUDIT_20260822.md` | the eval-harness sweep: scoring path clean, findings F1 (limubert scale) and F2 (unimts label text) |
| `PHASE_B_STEP0_CONTROL.md` | the paired step-0 control, and the finding that training pushed zero-shot below chance |
| `PHASE_B_DIAGNOSIS_20260820.md` | the plateau diagnosis: retrieval ranks by acquisition config, learned retrieval worth nothing |
| `PHASE_B_MIXER_20260819.md` | the evidence mixer result and its scrambled-vocabulary control |
| `PHASE_B_E2E_20260818.md` | first end-to-end run; encoder rank collapse from random init |
| `PHASE_A_RECOVERY_20260818.md` | Phase-A recovery run |

| `PHASE_B_TRAINING_STATUS.md`, `PHASE_B_TRAINING_INTENT.md` | Phase-B training intent and status |
| `ADAPTATION_TABLE_20260819.md` | the earlier adaptation table |
| `COMPACT_EVIDENCE_ENGINE.md` | the compact evidence engine — the direct architectural ancestor of `model/evidence/comparator.py` |
| `RESULTS.md` | the classification-era headline results (NOT `docs/results/RESULTS.md`, which is the application line) |

**A few links in these files do not resolve, on purpose.** They point at *generated* artifacts —
`eval/adaptation_tables/**/headline_tables.md` and result figures — that were produced by scripts
and were not carried onto this branch. Everything needed to read the numbers is in the prose here;
the generated originals are on `archive/pre-application-main-20260830`. Do not fabricate
replacements.

The original copies remain on branch `archive/pre-application-main-20260830`.
