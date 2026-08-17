# Sensor text conditioning

This file records the current implementation only. Historical per-channel and flat-factored designs
were removed to avoid presenting superseded alternatives as live model paths. The authoritative
end-to-end architecture remains `DESIGN_OF_RECORD.md`.

## Unit of representation

A sensor is one modality triad at one placement: accelerometer xyz or gyroscope xyz. The physical
filterbank first produces per-axis features. `SensorFold` then combines each valid xyz triad into one
token. A six-channel accelerometer-plus-gyroscope stream therefore produces two sensor tokens per
patch; an accelerometer-only stream produces one.

Axis identity is represented by fixed position inside the fold, so sensor-mode training does not use
axis role text. Missing axes remain explicit through the fold's validity input and cannot be confused
with a real channel whose value happens to be zero.

## Conditioning inputs

Each sensor token receives two frozen artifacts through separate gated learnable projections:

1. `text_descriptor`: the frozen MiniLM embedding of modality, device, placement, and gravity state.
2. `sensor_bias`: seven train-only physical measurements plus seven support bits indicating which
   measurements were observable.

The projections and gates train with the encoder. MiniLM and the offline measurements do not. A JEPA
descriptor-mask event hides the text descriptor while leaving signal and bias visible, then asks the
contextual sensor token to retrieve the correct frozen descriptor. This makes text conditioning a
trained source of information instead of an optional residual the model can ignore.

## Data contract

For every sample, these values must agree on sensor count and order:

```text
sensor_texts       list[str], length N
sensor_bias        float tensor, shape (N, 14)
sensor_placement   integer tensor, shape (N,)
sensor_id          integer tensor, shape (6,), mapping live channel slots to [0, N)
```

Channel dropout removes the corresponding sensor text, bias row, and placement row atomically.
Naturally absent modalities never acquire placeholder sensor rows. Ragged batches are zero-padded in
the sensor axis, and `sensor_present` prevents padded rows from entering attention or pooling.

The text descriptor also states whether the partner modality was present during encoding. For
example, an accelerometer row says either `recorded alongside a gyroscope` or `recorded without a
gyroscope`. This matters because the encoder attends across the available sensor set before export.
Channel dropout rewrites this clause after removing a modality; it must never claim that a dropped
sensor was present.

Configuration-changing augmentations (orientation, rate, gravity convention, and channel set) use
one shared draw for both VICReg views. Nuisance augmentations (crop, jitter, small scale, and text
paraphrase) remain independent. This prevents VICReg from being asked to erase the configuration that
the conditioning path is intended to expose.

## Compatibility path

`SetTokenizerEncoder` retains channel granularity only for explicit historical ablations and old
checkpoint inference. The Phase-A CLI constructs `token_granularity="sensor"`; new training and
evaluation reconstruct that value from the checkpoint configuration.
