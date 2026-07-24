# ut_complex — metadata verification

**Paper/source:** Shoaib et al. 2016, Sensors 16(4):426 — Complex Human Activity Recognition

https://www.utwente.nl/en/eemcs/ps/research/dataset/  (ut-data-complex.rar)

## What our pipeline asserts, and on what evidence

Two Samsung Galaxy S2 phones: RIGHT POCKET and RIGHT WRIST (emulating a smartwatch). We keep the WRIST stream only. acc m/s^2 gravity-present (|acc|~9.8), gyro rad/s. Subjects recovered by splitting each 30-min activity block into 10 equal contiguous chunks.

Verified 2026-07-24 against the converter's documented derivation (`data/datasets/ut_complex/convert.py`) plus empirical checks on the built grids (units, gravity DC, channel order, rate). No paper PDF is redistributed here.
