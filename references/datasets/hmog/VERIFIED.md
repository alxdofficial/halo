# hmog - metadata verification

**Official source:** H-MOG project page and bundled `data_description.pdf`

https://hmog-dataset.github.io/hmog/

## HALO view

H-MOG contains 100 anonymized participants with up to 24 controlled smartphone-use sessions per
participant. The protocol combines reading, writing, and map-navigation tasks with two motion
conditions. The bundled data dictionary defines `Gesture_scenario = 1` as sitting and `2` as walking;
the odd/even `TaskID` mapping independently encodes the same condition and is checked by the converter.

HALO retains the co-located Samsung Galaxy S4 accelerometer and gyroscope from the phone held in the
hand. The source event clocks are nominally 100 Hz. Accelerometer values are in m/s^2 and empirical
sitting-session DC magnitudes are near 9.81 m/s^2, so gravity is present; gyroscope values follow the
Android rad/s convention. Magnetometer, touch, key, scroll, stroke, and content data are excluded.

The official files contain duplicate `ActivityID` metadata rows, occasional sensor gaps, and one
archive-name typo: `207969.zip` consistently stores internal participant `207696`. The internal ID is
unique and is retained for subject-disjoint splitting; the alias is recorded in `manifest.json`.
The converter checks duplicate protocol fields before collapsing them, synchronizes accelerometer and
gyroscope on monotonic `EventTime`, treats gaps over 250 ms as boundaries, and truncates each
continuous block to complete six-second windows. No window crosses a source gap.

The archive downloaded on 2026-07-26 is 6,132,356,276 bytes and matches the official MD5
`4fd46756abec13815f426c66a58e626f`.
