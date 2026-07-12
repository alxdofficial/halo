# data/scripts

Shared, cross-dataset data logic (imported as `data.scripts.*`).

**Curation & assembly**
| Script | Role | Status |
|---|---|---|
| `accel_units.py` | unit + gravity canonicalization → accel in `g`, gravity present (single source of truth) | ✅ |
| `deployment_policy.py` | phone/watch device-stream selection — which channels/placement per dataset | ✅ |
| `channels.py` | channel-name helpers (`group_channels_by_sensor`, `is_imu_channel`) | ✅ |
| `baseline_view.py` | **harmonised** (fixed 6-ch, zero-pad+mask) vs **non-harmonised** (native) channel views | ✅ |
| `assemble.py` | full pipeline: curate → unit→g → window → harmonised/non-harmonised **grid** | ✅ |
| `windowing.py` | activity-aware variable-length windowing (training) | ✅ |

**Labels & augmentation**
| Script | Role | Status |
|---|---|---|
| `augmentations.py` | physics + text augmentation curriculum (P1–P4, rotation, channel/label text) | ✅ ported |
| `label_augmentation.py` | per-dataset label synonyms + templates | ✅ ported |
| `label_groups.py` | semantic label groups for balanced training sampling | ✅ ported |
| `build_global_label_mapping.py` | build the shared global label vocabulary (for the ConSE baselines) | ✅ ported |

**Orchestration & debug**
| Script | Role | Status |
|---|---|---|
| `setup_all_datasets.py` | one entry point: convert every dataset → sessions → harmonised + raw grids | ⬜ adapting |
| `visualization_utils.py`, `plot_sessions.py` | per-session debug plots | ✅ ported |

Per-dataset pieces (converter, `metadata.json`, `labels.json`) live under [`../datasets/<name>/`](../datasets).
The two **training modes** (harmonised / normal) differ only in how `baseline_view.py` assembles the
curated stream — see `../../docs/BASELINE_FAIRNESS_POLICY.md` §2A.
