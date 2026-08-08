# Archive — superseded records

> ## Nothing in this folder describes the current system. Do not cite anything here.
>
> **If you are writing the paper, or reconstructing what HALO does today, you are in the wrong
> folder.** Go to [`../README.md`](../README.md) and read the live docs it lists.

These files are kept, not deleted, because the record of what we believed and why is part of the
work: every one of them was authoritative when written, and several document claims we later
retracted. Retractions are worth preserving — they are the reason the current design looks the way
it does, and they stop the same idea being re-proposed.

Every file here carries a banner at the top naming its live replacement. If you find one that does
not, that is a bug — add the banner.

## What is here and what replaced it

| Archived | Why | Live replacement |
|---|---|---|
| `EVIDENCE_ENGINE.md` | working design; EDL, full-soft forward and the pooled trainer were rejected and deleted | `design/PHASE_B_TRAINING_INTENT.md` |
| `EVIDENCE_ENGINE_TIER2.md` | Tier-2 plan; pooled trainer, EDL, decoder multi-subspace branch and auxiliary losses removed 2026-08-07 | `design/PHASE_B_TRAINING_INTENT.md` |
| `EVIDENCE_ENGINE_FINDINGS.md` | empirical record of the pooled era. **Its STATUS block says "there are currently NO valid results" — that was true on 2026-07-21 and is false now.** | `../README.md` for results; `design/REMEDIATION_PLAN.md` for issue status |
| `LEARNABLE_TOKENIZER_ARM.md` | hypothesis **falsified** — the learnable filterbank measured inert; multiresolution did the work | `design/PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md` |
| `NATIVE_PRETRAIN_PREFLIGHT.md` | preflight for a run using the old SupCon / A3 / multi-rail recipe | `training/tokenizer/README.md` |
| `PIPELINE_A_PREFLIGHT.md` | 8-dataset / 57-label / 60 Hz audit, two corpus generations old | `design/DATA_SCALING_PLAN.md` |
| `TOKENIZER_QUALITY.md` | quality battery run on an 11k-step encoder; predates every current checkpoint | `design/PHASE_A_B_AGREED_IMPLEMENTATION_PLAN.md` |
| `RESULTS_V2.md` | results snapshot under the 59-label vocabulary | `../README.md` |
| `RESULTS_PRELIMINARY.md` | older results snapshot, same stale vocabulary | `../README.md` |
| `STORAGE_AUDIT_2026-07-26.md` | point-in-time disk audit; the numbers have already moved | — |

## The number trap

Anything here produced before the vocabulary fix used a **59-label** global vocabulary that silently
discarded 11.5% of training windows. The current protocol is **93 labels (v4)**. Numbers across that
boundary are not comparable, and several docs here quote figures that were later retracted outright
(49.5, 47.3, 40.4, 42.7, 45.1, 46.1, and the `r = −0.973` correlation).

There is exactly one current results table, generated from `eval/results/` by
`python -m eval.assemble_table`, and quoted in [`../README.md`](../README.md).
