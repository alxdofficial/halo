# harmes — metadata verification

**Paper/source:** HARMES: Hand Activity Recognition from Multimodal Egocentric Sensing

https://zenodo.org/records/19425719  (HARMES-RAW.zip, CC-BY-4.0)

## What our pipeline asserts, and on what evidence

RIGHT-wrist WearOS smartwatch; acc m/s^2 gravity-present (at-rest |acc|=9.81), gyro rad/s (p99=5.6). Left-wrist Puck.js EXCLUDED (gyro saturates at int16 rail). 50 Hz resampled grid.

Verified 2026-07-24 against the converter's documented derivation (`data/datasets/harmes/convert.py`) plus empirical checks on the built grids (units, gravity DC, channel order, rate). No paper PDF is redistributed here.
