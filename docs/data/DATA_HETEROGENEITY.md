# Data heterogeneity — per-dataset reference

> This doc is the *mechanics* of how we normalize heterogeneity. It is **not** a contribution claim:
> reducing every dataset to ≤ 6 channels and resampling is deliberate deployment-realistic
> preprocessing, not the pitch. For *why* HALO exists (language-conditioned open-set + acquisition-config
> generalization — the part that is not cheap preprocessing) see [`MOTIVATION.md`](../design/MOTIVATION.md).

Every non-obvious thing we do to a dataset is recorded here, so we never have to re-derive *why*.
Three modules enforce these decisions in code; this doc is their prose rationale:

- **`data/scripts/deployment_policy.py`** — device/placement/channel selection + gravity reconstruction
- **`data/scripts/accel_units.py`** — accelerometer unit → g (runs *after* deployment_policy)
- **`data/scripts/assembly/baseline_view.py`** — harmonised/non-harmonised views (fixed windowing runs in `build_grids.py`)

Pipeline order (fixed): `raw session → deployment_policy → accel_units → windowing → baseline_view`.
`deployment_policy` owns **gravity**; `accel_units` owns **unit**. Nothing here fabricates data.

## The four heterogeneity axes we normalize (and how)

| Axis | Normalized to | Where |
|---|---|---|
| **Device / placement** | one stream per device: phone (pocket/waist/thigh), watch (wrist), or body-strapped `device` (lower-back, forearm, head-glasses, ear). The **strict** deployment view keeps phone only; the **harmonised** (all-wearable) view keeps phone + watch + `device`. | deployment_policy |
| **Channels** | `[acc_xyz(, gyro_xyz)]`, standard order; ≤ 6 | deployment_policy → baseline_view |
| **Gravity** | present (reconstructed for iOS); kuhar **and** xrf_v2's AirPods stay gravity-removed (never faked) | deployment_policy |
| **Unit** | accelerometer in **g** (|acc| ≈ 1 at rest); gyro untouched | accel_units |

Sampling-rate heterogeneity is **not** flattened in the corpus (it's a first-class comparison axis);
rate is recorded per dataset. `build_grids` materialises **three** regimes (`_ALIGNMENTS`):

- **`harmonised`** — resampled to **60 Hz**, 6-ch pad+mask, canonical labels. The fixed-rate crutch the
  layout-locked baselines (CrossHAR/LiMU-BERT/harnet) require.
- **`non_harmonised`** — native rate, native 3/6-ch, native labels. The raw eval/baseline source.
- **`native`** — **native rate** (no resample), 6-ch pad+mask, canonical labels. **HALO's source:** the
  filterbank tokenizer is rate-invariant, so HALO trains on the corpus's REAL native rates (20/50/100 Hz)
  plus the `rate`/`window_crop` augmentations — not a 60 Hz base with synthetic rate diversity.

> **The "Channels" column below = the dataset's REAL sensors.** The accelerometer is *always* present
> and *never* removed. Some datasets have no usable gyroscope — it is either physically absent
> (capture24/unimib_shar/harth are accelerometer-only devices) or unrecoverable — so they contribute
> 3 real channels. This is **not** a choice about the tensor: the **harmonised**/**native** views are
> *always* 6-channel `[acc_xyz, gyro_xyz]` (acc-only datasets get **zero-padded + masked** gyro slots),
> and the **non-harmonised** view keeps the native 3 or 6. Nothing is ever "taken out".

## Per-dataset table

| Dataset | Role | Device · placement | Native rate | Accel unit | Gravity | Channels | Special treatment (and why) |
|---|---|---|---|---|---|---|---|
| uci_har | train | phone · waist | 50 Hz | g | present | acc+gyro | Uses **`total_acc`** (gravity present, g), **not** `body_acc` (gravity-removed, ≈0.04 g). |
| hhar | train | phone · waist | 50 Hz | m/s² | present | acc+gyro | — |
| pamap2 | train | watch · wrist (hand IMU) | 100 Hz | m/s² | present | acc+gyro | Keep only the **wrist ±16 g** IMU; drop chest, ankle, the ±6 g accel, mag, temp, HR, and the invalid orientation quaternion. |
| wisdm | train | phone · pocket **and** watch · wrist (2 streams) | 20 Hz | m/s² | present | acc(+gyro) | Legacy conversion logs accel and gyro on **disjoint rows**; gyro is optional until the converter emits merged IMU sessions. |
| kuhar | train | phone · waist | 100 Hz | m/s² | **removed** | acc+gyro | Linear acceleration — DC ≈ 0 at rest by design. **Never fabricate gravity.** Gravity-dependent baselines must skip it. |
| unimib_shar | train | phone · pocket | 50 Hz | g | present | acc only | Accelerometer-only dataset. |
| hapt | train | phone · waist | 50 Hz | g | present | acc+gyro | UCI-HAR family (Android, ~1.02 g). |
| mhealth | train | watch · wrist (arm IMU) | 50 Hz | m/s² | present | acc+gyro | Right-lower-arm IMU (co-located acc+gyro). Gyro is somewhat sample-and-hold but **real**, so it is kept as a 6-ch stream. Drop chest, ankle, ECG, mag. |
| capture24 | train | watch · wrist | 100 Hz | g | present | acc only | Free-living Axivity, accelerometer-only. |
| sp_sw_har | train | phone · front pocket **and** watch · wrist (2 streams) | ~102.5 → 100 Hz | g | present | acc+gyro | Paired Timed-Up-and-Go. Per-row labels → **fixed 1.0 s windows** (`turning` is ~1 s; a 6 s window would discard it), `pre_windowed`. Resampled on the real `timestamp` (true ~102.5 Hz, dup timestamps + gaps), not a synthetic clock. |
| nfi_fared | train | body-strapped **lower back** + **dominant forearm** (2 streams) | 100 Hz | g | present | acc+gyro | Forensic activities (incl. transport → `vehicle`, punch/kick/drag as new labels). Gyro **deg/s → rad/s**. SHA-256 dedupe of byte-identical source CSVs (pp10 exp2==exp3). Placement is **forearm** per the Hi-OSCAR paper, not wrist. |
| harmes | train | watch · dominant wrist | 50 Hz | m/s² | present | acc+gyro | 15 fine-grained kitchen/bath hand ADLs. **Right wrist only** — the left Puck.js gyro saturates the int16 rail with an undocumented full-scale (unrecoverable), so we do not ship it. Labels from the start/end event log; corrected a **+1 h clock offset** on 39/71 recordings (DST bug). |
| xrf_v2 | train | 6 streams: L/R **wrist**, L/R **pocket phone**, **head glasses**, **AirPods ear** | 50 Hz (AirPods 25→50) | g | present (**AirPods removed**) | acc+gyro | 16 volunteers, 30 indoor ADLs (arXiv 2501.19034). Device→placement order read from the h5's own `device_order`. Gyro **deg/s → rad/s**. AirPods = iOS **user acceleration (gravity removed)** + gyro; upsampled to 50 Hz. Glasses/ear placements exist nowhere else in the corpus. |
| motionsense | eval | phone · front pocket | 50 Hz | iOS → g | present | acc+gyro | iOS: total accel = **userAcceleration + gravity** (both g), reconstructed in deployment_policy; attitude is QA-only. |
| realworld | eval | phone · waist | 50 Hz | m/s² | present | acc(+gyro) | Gyro retained only when the converted waist stream has a complete finite triad. |
| mobiact | eval | phone · trouser pocket | 50 Hz | m/s² | present | acc+gyro | — |
| shoaib | eval | phone · right pocket (primary) | 50 Hz | m/s² | present | acc+gyro | Left-pocket / belt / wrist-proxy kept as **diagnostic** streams (not in the primary score). |
| inclusivehar | eval | phone · waist | 50 Hz | iOS → g | present | acc+gyro | iOS reconstruction as motionsense; ability-stratified cohort. |
| harth | stress | back / thigh | 50 Hz | g | present | acc only | **Non-deployment** placement — retained only as a placement stress test, never in the primary score. |

## Excluded datasets (kept in `legacy_code`, not in this repo)

| Dataset | Why excluded |
|---|---|
| dsads | Only torso/limb IMUs — no phone-pocket/waist or watch-wrist placement. |
| opportunity | Back + upper/lower-arm IMUs; appendix-only, not phone/watch inputs. |
| recgym | Per-axis min-max [0,1] normalization destroyed physical scale and gravity — non-recoverable. |
| extrasensory | Future add (free-living phone+watch); not yet in the deployment policy. |

## Invariants (asserted by tests)

- Every dataset in `deployment_policy.STREAM_SPECS` is classified in **exactly one** of
  `accel_units.ACC_UNIT_G` / `ACC_UNIT_MS2` (a new dataset cannot land without a unit decision).
- After the pipeline, a still window reads |acc| ≈ 1 g for gravity-present datasets, ≈ 0 for kuhar.
- Gyroscope channels are never scaled by `accel_units`.
