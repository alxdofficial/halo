# References: datasets and baselines

This tree stores the local primary-source material used to audit every dataset and published baseline
in HALO. A usable entry contains:

- `citation.json`: title, authors/owner, year, venue, stable URLs, licence, and implementation notes.
- `paper.pdf`, `webpage.html`, or another clearly named official technical document.
- `SOURCE.txt`: canonical publication, data, and code URLs.

A `citation.json` plus a URL is not a substitute for the local source document. New dataset integration
is incomplete until its publication or official protocol documentation is present here and readable.

## Current inventory

There are 36 dataset entries and 9 baseline entries, with 45 parseable `citation.json` files, 34 PDFs,
and 5 saved HTML documents. Every dataset entry has a `citation.json` (counted 2026-08-11).

Dataset entries:

`capture24`, `dsads`, `extrasensory`, `hapt`, `harmes`, `harth`, `hhar`, `inclusivehar`, `kuhar`,
`mhealth`, `mobiact`, `motionsense`, `nfi_fared`, `nhanes`, `opportunity`, `pamap2`, `realworld`,
`recgym`, `shoaib`, `sp_sw_har`, `tnda_har`, `uci_har`, `unimib_shar`, `usc_had`, `ut_complex`,
`wisdm`, `xrf_v2`.

Baseline entries:

`crosshar`, `deepconvlstm`, `lanhar`, `limubert`, `llasa`, `moment`, `normwear`, `ssl-wearables`,
`unimts`.

## Scale-source evidence

- **Capture-24:** `references/datasets/capture24/paper.pdf`, the published Scientific Data paper.
- **ExtraSensory:** `references/datasets/extrasensory/paper.pdf`, the canonical phone/watch paper.
- **NHANES PAX80_G:** `references/datasets/nhanes/procedures_manual.pdf` plus the locally saved
  `pax80_g_documentation.html`. The CDC release documentation is authoritative for rate, units,
  placement, file schema, missing labels, and QC intervals.
- **PAAWS:** researched but not integrated. The official release endpoint returns HTTP 403 from this
  machine, so a publication alone does not satisfy the data-access gate.

## Known reference debt

The following pre-existing entries still have `citation.json`/`SOURCE.txt` but no local publication or
saved official page: `harmes`, `nfi_fared`, `sp_sw_har`, `tnda_har`, `usc_had`, `ut_complex`,
`xrf_v2`, and baseline `deepconvlstm`. Their runtime integrations predate this stricter gate. This debt
should be closed before the final paper audit; it must not be copied as the standard for new sources.
