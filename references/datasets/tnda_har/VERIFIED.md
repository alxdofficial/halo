# tnda_har — metadata verification

**Paper/source:** Wu et al., TNDA-HAR

https://ieee-dataport.org/open-access/tnda-har-0  DOI 10.21227/4epb-pg26 (raw behind sign-in; we use the UniMTS bundle: HF xiyuanz/UniMTS)

## What our pipeline asserts, and on what evidence

SECOND-HAND SOURCE: UniMTS preprocessed bundle. 30 channels = 5 IMU positions x (acc,gyro); RIGHT-WRIST = columns 12:18 (acc 12:15, gyro 15:18) per UniMTS skeleton assignment. 50 Hz, acc m/s^2 gravity-present (|acc|~9.8), gyro rad/s. NOTE: bundle ships no per-sample subject id — `subject` records the UniMTS train/test partition, not individual participants.

Verified 2026-07-24 against the converter's documented derivation (`data/datasets/tnda_har/convert.py`) plus empirical checks on the built grids (units, gravity DC, channel order, rate). No paper PDF is redistributed here.
