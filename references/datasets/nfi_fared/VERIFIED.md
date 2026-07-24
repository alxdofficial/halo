# nfi_fared — metadata verification

**Paper/source:** NFI-FARED / Hi-OSCAR (Netherlands Forensic Institute)

https://huggingface.co/datasets/NetherlandsForensicInstitute/NFI_FARED_IMU

## What our pipeline asserts, and on what evidence

TWO strapped IMUs: 'rug'=lower BACK, 'arm'=dominant FOREARM (per paper; NOT the wrist). acc in g gravity-present (|acc|~1.00), gyro deg/s -> rad/s. 100 Hz uniform.

Verified 2026-07-24 against the converter's documented derivation (`data/datasets/nfi_fared/convert.py`) plus empirical checks on the built grids (units, gravity DC, channel order, rate). No paper PDF is redistributed here.
