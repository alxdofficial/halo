# HALO weekly — 24 Aug 2026

`HALO_Weekly_20260824.pptx` — 15 slides, matching the house style of
`HALO_Weekly_20260811.pptx` (13.33×7.5 in, Arial, 25 pt bold titles, italic blue deck line,
`#F4F6F9` cards, cream `#FBF3DF` takeaway banners, `#EFEFEF` table headers with an `#EFF4EE`
highlight row).

## Rebuild

```bash
python comms/2026-08-24-weekly/extract.py     # sealed results -> data/deck.json
python comms/2026-08-24-weekly/figures.py     # data/deck.json -> figures/*.png
python comms/2026-08-24-weekly/build_pptx.py  # -> HALO_Weekly_20260824.pptx
```

No number in the deck is typed by hand: every table and figure is generated from
`data/deck.json`, which is extracted from the sealed evaluation artifacts under
`eval/adaptation_results/`. `build_pptx.py` ends with a geometry check that fails loudly if a
shape leaves the slide or a figure lands under its banner.

## Sources

| what | where |
|---|---|
| HALO headline (`long_4h_20260821`) | `eval/adaptation_results/halo_compact_20260822/` |
| baselines, post-fix | `eval/adaptation_results/e2e_compact_35k_20260823/` |
| engine ablation ladder | `.../e2e_recording_rerank_35k_v3_20260824/halo_engine_decomposition.json` |
| architecture trajectory | the three `e2e_*` run directories |

Baselines come from the **2026-08-23** re-run, not `v1_d85761d` (2026-08-17). The older set
predates the LiMU-BERT g-convention fix and the UniMTS E=8 label ensemble, and understates both.
