# HALO docs

Organized into subfolders by concern. Start with **`design/MOTIVATION.md`** for *why* HALO exists.

## `design/` — the contribution: what we're building and why
- [**MOTIVATION.md**](design/MOTIVATION.md) — the load-bearing "why": one language interface for unseen
  labels *and* unseen acquisition configs; open-set-labels-alone is table stakes (ConSE); channel-count
  is not a claim. **Read this first.**
- [LANGUAGE_HIERARCHY.md](design/LANGUAGE_HIERARCHY.md) — second-act design: language at every level
  (conditioning → concepts → labels) + graceful degradation / open-set fallback.
- [AUGMENTATIONS.md](design/AUGMENTATIONS.md) — the augmentation policy + the told-vs-not-told
  conditioning experiment (three-bucket: symmetric robustness / HALO-only-by-necessity / interface).

## `data/` — the corpus and how it's built
- [DATA_PIPELINE.md](data/DATA_PIPELINE.md) — source → curate → unit→g → resample → window → grids;
  the converter contract; download provenance; pre-windowed + gated notes.
- [DATA_HETEROGENEITY.md](data/DATA_HETEROGENEITY.md) — per-dataset normalization decisions (device/
  placement/channels/gravity/unit) and *why*.

## `baselines/` — who we compare against and how
- [BASELINES.md](baselines/BASELINES.md) — the roster, verified input contracts, frozen-vs-self-train
  rationale, the channel/rate defense, the 3-part reporting design, completeness survey.
- [BASELINE_FAIRNESS_POLICY.md](baselines/BASELINE_FAIRNESS_POLICY.md) — the treatment contract: tiered
  heterogeneity framework, faithfulness contract, the fixed-baseline input invariant.

## Conventions
- One concern per folder; add new docs to the matching subfolder and link them here.
- Facts verified against papers/code carry a date; the memory files
  (`~/.claude/.../memory/halo-*.md`) index cross-session context.
