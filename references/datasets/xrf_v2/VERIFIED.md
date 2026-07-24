# xrf_v2 — metadata verification

**Paper/source:** XRF V2, arXiv:2501.19034 (ACM IMWUT 2025)

https://www.kaggle.com/datasets/airslab2020/xrfv2-multimodal-tal-caption-qa-no-rgb

## What our pipeline asserts, and on what evidence

5-position body IMU order read from the h5's OWN `device_order` field (self-describing; immune to the Plus-vs-WWADL_open ordering difference): left_wrist|right_wrist|left_phone(pocket)|right_phone(pocket)|glasses(head). Body IMU 50 Hz, acc g gravity-present, gyro deg/s->rad/s. AirPods Pro 25 Hz USER acceleration (gravity REMOVED, |acc|~0.035 g).

Verified 2026-07-24 against the converter's documented derivation (`data/datasets/xrf_v2/convert.py`) plus empirical checks on the built grids (units, gravity DC, channel order, rate). No paper PDF is redistributed here.
