# data/datasets

One subfolder per dataset. Each `<name>/` holds the **dataset-specific** pieces:

- downloads / upstream source (**gitignored** — regenerated from the converter)
- converter / preprocessing script(s) that turn the raw source into per-subject sessions
- metadata, channel descriptions, and per-dataset notes (e.g. which device/placement/channels we keep,
  gravity state, sampling rate, any known data-quality caveats)

Shared, **cross-dataset** logic (unit/gravity canonicalization, the device/channel-selection policy,
harmonised-vs-raw assembly, augmentations, the setup-all entry point) lives in [`../scripts`](../scripts),
not here.

Train (primary): uci_har, hhar, pamap2, wisdm, kuhar, unimib_shar, hapt, mhealth, capture24,
sp_sw_har (phone+watch TUG), nfi_fared (back+forearm), harmes (wrist ADLs), xrf_v2 (dual-wrist +
dual-pocket + head-glasses + AirPods ear, 16 subjects). 12 datasets / 20 streams / 96 labels.
Non-strict ("harmonised") training admits phone + watch + body-strapped `device` placements; the
strict deployment view keeps phone only.
Eval (held out, primary): motionsense, realworld, mobiact, shoaib, inclusivehar, usc_had, tnda_har, ut_complex.
