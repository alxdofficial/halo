# data/scripts

Shared, cross-dataset data logic (imported as `data.scripts.*`).

| Script | Role | Status |
|---|---|---|
| `accel_units.py` | unit + gravity canonicalization → accel in `g`, gravity present (single source of truth) | ✅ |
| `deployment_policy.py` | phone/watch device-stream selection — which channels/placement to keep per dataset | ✅ |
| `baseline_view.py` | **harmonised** (fixed 6-ch, zero-pad+mask) vs **non-harmonised** (native) assembly | ✅ |
| `augmentations.py` | augmentation curriculum (physics + text) | ⬜ roadmap |
| `setup_all_datasets.py` | one entry point: build every dataset into harmonised + raw grids | ⬜ roadmap |

The two **training modes** (harmonised / normal) differ only in how the curated stream is assembled
here — see `baseline_view.py` and `../../docs/BASELINE_FAIRNESS_POLICY.md` §2A.
