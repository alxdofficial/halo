# baselines

One subfolder per baseline we compare against. Each `<name>/` holds everything about that baseline:

- **`citation.json`** + the paper / publication(s) that identify it
- **`repo/`** — a clone of the baseline's upstream repository (**gitignored**; re-clone from the URL in `citation.json`)
- **`adapter.py`** — the thin wrapper that runs the baseline under our protocol, honoring its input
  contract (rate / channels / open-set handling) per [`../docs/baselines/BASELINE_FAIRNESS_POLICY.md`](../docs/baselines/BASELINE_FAIRNESS_POLICY.md)

Integrated baselines (all scored under the current protocol — see [`../eval/results`](../eval/results)):

| Baseline | Weights | Open-set bridge |
|---|---|---|
| `crosshar`, `limubert` | we self-pretrain on our corpus (no released weights) | ConSE |
| `harnet` (ssl-wearables), `unimts`, `normwear`, `imagebind` | frozen, released | ConSE / native text |
| `halo` | ours — frozen Phase-A encoder | ConSE |
| `halo_evidence` | ours — Phase-A encoder + Phase-B evidence engine | candidate label text |

`_pre_corpusmatch_2026-07-14/` and `_pre_vocabfix_2026-07-21/` hold superseded artifacts from
earlier protocols; they are kept for provenance and must never be mixed into a current table.

Roster rationale, verified input contracts, and the frozen-vs-self-train verdicts:
[`../docs/baselines/BASELINES.md`](../docs/baselines/BASELINES.md).
