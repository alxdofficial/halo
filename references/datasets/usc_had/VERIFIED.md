# usc_had — metadata verification

**Paper/source:** Zhang & Sawchuk 2012, USC-HAD

https://sipi.usc.edu/had/  (USC-HAD.zip)

## What our pipeline asserts, and on what evidence

Single MotionNode IMU on the FRONT-RIGHT HIP. 100 Hz, acc +/-6g in g (TOTAL accel, gravity present, still |acc|~1 g), gyro +/-500 dps. Curated as phone_hip per the deployment-realistic phone/watch policy (MOTIVATION §5).

Verified 2026-07-24 against the converter's documented derivation (`data/datasets/usc_had/convert.py`) plus empirical checks on the built grids (units, gravity DC, channel order, rate). No paper PDF is redistributed here.
