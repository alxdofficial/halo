# data/datasets

One subfolder per dataset. Each `<name>/` holds the **dataset-specific** pieces:

- downloads / upstream source (**gitignored** — regenerated from the converter)
- converter / preprocessing script(s) that turn the raw source into per-subject sessions
- metadata, channel descriptions, and per-dataset notes (e.g. which device/placement/channels we keep,
  gravity state, sampling rate, any known data-quality caveats)

Shared, **cross-dataset** logic (unit/gravity canonicalization, the device/channel-selection policy,
harmonised-vs-raw assembly, augmentations, the setup-all entry point) lives in [`../scripts`](../scripts),
not here.

Train (primary): uci_har, hhar, pamap2, wisdm, kuhar, unimib_shar, mhealth, capture24,
sp_sw_har (phone+watch TUG), nfi_fared (back+forearm), harmes (wrist ADLs), xrf_v2 (dual-wrist +
dual-pocket + head-glasses + AirPods ear, 16 subjects). HAPT is retained locally but excluded from
Phase A because it is a near-duplicate re-release of UCI HAR.

Optional Phase-A scale sources: ExtraSensory (labelled phone-pocket/hand + watch acceleration), a
bounded NHANES PAX80_G subset (unlabelled non-dominant-wrist acceleration), and H-MOG (phone-in-hand
acceleration + gyroscope during sitting/walking phone use). They are never included by a default grid
build or default paper run; request them explicitly so expanded-data experiments remain attributable.
PAAWS was evaluated but is not integrated because its current repository download returns HTTP 403
from this machine; sample code is not sufficient evidence of access to the released bytes.

Non-strict ("harmonised") training admits phone + watch + body-strapped `device` placements; the
strict deployment view keeps phone only.
Eval (held out, primary): motionsense, realworld, mobiact, shoaib, inclusivehar, usc_had, tnda_har, ut_complex.
