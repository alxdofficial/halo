# Sensor text conditioning

> ## ⚠️ Half of this is stale — 2026-08-22
> - **`sensor_bias` is NOT a conditioning input.** The current encoder builds
>   `use_sensor_bias_conditioning=False`, and the Phase-B trainer *hard-rejects* any checkpoint that
>   carries it. One artifact reaches the sensor token: the frozen `text_descriptor`, through one
>   gated projection. `SensorRows.bias` survives as a dead field nothing in `model/evidence/` reads.
> - **The descriptor-mask JEPA event is not running.** The recipe builds with
>   `descriptor_prediction=False`, so the mechanism that was supposed to make conditioning
>   non-ignorable is off — which is the direct explanation for the 2026-08-11 parity gate finding
>   the conditioning **inert** (+0.0086 mean, sign flipping on 2 of 4 datasets).
> - **The partner-modality rationale no longer holds.** It is justified here by "the encoder attends
>   across the available sensor set before export"; under the temporal trunk each sensor is encoded
>   in isolation, so a wrist accelerometer row is identical whether or not a gyroscope was present.
> - **Missing: the descriptor's second consumer.** It is carried raw as `SensorRows.descriptor` into
>   the evidence mixer as `ROLE_QUERY_DESC` / `ROLE_EVIDENCE_DESC` tokens — a larger conditioning
>   path than the encoder-side gate this file documents.

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
