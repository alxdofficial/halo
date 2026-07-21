# Pre-vocabulary-fix results snapshot (2026-07-20)

Frozen copy of `eval/results/` as it stood **before** the global-vocabulary fix (59 → 93 labels).

Every number here was produced under the stale 59-label vocabulary, which silently discarded
**39,413 / 343,235 training windows (11.48%)** from the memory bank and from every ConSE head-fit.
It also predates: the F1 eval-label-text fix, the F2 held-out-config hyperparameter selection, the
deterministic head seeding, the shared subject-split manifest, and the unified probe.

Kept so the pre-fix numbers stay reproducible and we can report an honest before/after.
**Do not mix these with post-fix numbers in one table.**

See `docs/design/EVIDENCE_ENGINE_FINDINGS.md` (STATUS block) and
`docs/design/REMEDIATION_PLAN.md`.
