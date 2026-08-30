# Previous zero-shot and evidence-engine design

The application design on `main` retains old implementation code where removing it would impede
reproducibility, but the corresponding active-looking documents and generated result copies have
been removed.

The complete previous documentation and promoted tables are available at the branch point:

```bash
git show 32267b6:docs/README.md
git show 32267b6:docs/results/RESULTS.md
git show 32267b6:docs/design/PHASE_B_TRAINING_INTENT.md
```

Relevant old code includes `model/evidence/`, `training/evidence/`, generic HAR evaluation under
`eval/`, and candidate-label sections of the tokenizer trainer. It is historical support code, not
the design of record for the four-task movement-monitoring system.

Switch to `archive/pre-application-main-20260830` when reproducing those experiments. Do not
restore copies of the old documents to `main`; use Git history so there is one authoritative active
version.
