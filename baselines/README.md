# baselines

One subfolder per baseline we compare against. Each `<name>/` holds everything about that baseline:

- **`citation.json`** + the paper / publication(s) that identify it
- **`repo/`** — a clone of the baseline's upstream repository (**gitignored**; re-clone from the URL in `citation.json`)
- **`adapter.py`** — the thin wrapper that runs the baseline under our protocol, honoring its input
  contract (rate / channels / open-set handling) per [`../docs/BASELINE_FAIRNESS_POLICY.md`](../docs/BASELINE_FAIRNESS_POLICY.md)

Planned baselines:

| Baseline | Weights | Open-set |
|---|---|---|
| crosshar, limubert, deepconvlstm | we train / from scratch | ConSE / few-shot |
| ssl_wearables, unimts, normwear | frozen released | ConSE / native text |

Nothing here yet — populated as each baseline is ported.
