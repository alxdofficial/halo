# sp_sw_har — metadata verification

**Paper/source:** Matey-Sanz et al. (GeoTec 2023) — paired smartphone/smartwatch TUG

https://github.com/GeoTecINIT/sp-sw-har-dataset  (CC-BY-4.0)

## What our pipeline asserts, and on what evidence

_sp = smartphone LEFT FRONT POCKET, _sw = smartwatch LEFT WRIST. acc m/s^2 gravity-present (/9.81 -> g), gyro rad/s. ~102.5 Hz -> resampled to uniform 100 Hz; 1.0 s fixed windows.

Verified 2026-07-24 against the converter's documented derivation (`data/datasets/sp_sw_har/convert.py`) plus empirical checks on the built grids (units, gravity DC, channel order, rate). No paper PDF is redistributed here.
