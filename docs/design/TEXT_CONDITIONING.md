# Acquisition-description conditioning

This is an implementation reference for the optional text-conditioning path in the HALO encoder. It
is not an application contribution on `main`.

## Implemented factorization

The encoder can describe each physical sensor separately from each axis role:

- **sensor identity:** device type, placement, modality, gravity convention, and available partner
  sensor context;
- **axis role:** x, y, or z within that sensor; and
- **numeric metadata:** sampling rate, physical patch duration, temporal position, and validity masks
  through non-language paths.

A frozen sentence encoder produces text features. Learnable pooling/projection components map those
features into the sensor-token space. Repeated descriptions are cached where doing so preserves the
training semantics.

## Application role

Acquisition metadata may help distinguish real movement change from sensor remounting or device
change. That benefit is a hypothesis and must be measured through Task-1 cross-session/remounting
performance and Task-2 false change under controlled remounting.

Prior HAR experiments did not establish that textual acquisition conditioning improved transfer.
The new application comparison therefore includes:

1. full acquisition descriptions;
2. neutral descriptions with the same numeric metadata; and
3. no text conditioning where checkpoint compatibility permits it.

Language never names the target movement in Tasks 1-3. A detection, recurrence, or difference score must come
from sensor evidence, not semantic proximity between activity names.

## Correctness contract

- Every channel maps to exactly one physical sensor and one axis role.
- Left/right and body placement descriptions match converter metadata.
- Acceleration and gyroscope remain distinct modalities.
- Accel-only streams do not invent gyroscope text or tokens.
- Description dropout cannot silently erase numeric rate, duration, masks, or provenance.
- Cached frozen text features are detached; learnable projections still receive gradients when
  encoder training is enabled.

Tests for this contract live in `tests/test_factored_conditioning.py`.
