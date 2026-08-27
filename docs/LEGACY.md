# Previous zero-shot and evidence-engine design

The `application-motion-monitoring` branch retains the old implementation code where removing it
would impede reproducibility, but the corresponding active-looking documents and generated result
copies have been removed.

The complete previous documentation and promoted tables are available at the branch point:

```bash
git show 32267b6:docs/README.md
git show 32267b6:docs/results/RESULTS.md
git show 32267b6:docs/design/PHASE_B_TRAINING_INTENT.md
```

Relevant old code includes `model/evidence/`, `training/evidence/`, generic HAR evaluation under
`eval/`, and candidate-label sections of the tokenizer trainer. It is historical support code, not
the design of record for the three movement-monitoring tasks.

Switch to `main` or a previous design branch when reproducing those experiments. Do not restore
copies of the old documents to this branch; use Git history so there is one authoritative version.
