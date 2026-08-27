# Training and diagnostics

The application-motion-monitoring branch starts from frozen representations. Training is optional
and is introduced only after non-parametric signal and representation floors are measured.

## `tokenizer/`

HALO representation pretraining, checkpoint loading, temporal representation export, and
representation-health diagnostics. The existing JEPA/VICReg trainer remains useful for controlled
within-HALO experiments, but the three application tasks do not require another pretraining run to
begin.

See [`tokenizer/README.md`](tokenizer/README.md) for the implemented encoder recipe. Its Phase-A name
is historical terminology; application code should call the output a representation checkpoint.

## `evidence/`

Historical candidate-label, memory-bank, and retrieve-mix-vote experiments. They are retained so the
previous published-result branch remains reproducible, but they are not the active downstream design.
Do not extend these trainers for the new tasks.

The application path uses sequence matching, aligned difference measurement, and motif discovery in
a new `applications/motion_monitoring/` package. If frozen representations fail, one small Siamese
metric projection may be trained and shared by all three tasks.

## `diagnostics/`

Existing representation and provenance probes remain useful. New diagnostics should measure:

- cross-session same-motion versus different-motion separation;
- remounting and device sensitivity;
- temporal embedding rank and patch diversity;
- verification calibration and false-match behavior;
- target-absent false alarms; and
- motif recurrence versus duplicate-buffer artifacts.

Activity labels may be used to score hidden-label evaluation but do not enter the core application
algorithms.
