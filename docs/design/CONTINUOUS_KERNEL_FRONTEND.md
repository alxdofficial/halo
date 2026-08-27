# Continuous physical-time frontend

> **Implemented encoder arm, 2026-08-24.** Retained because temporal resolution may matter directly
> for sequence matching and phase-local movement comparison.

## Purpose

The fixed HALO filterbank summarizes motion in physical frequency bands. The continuous frontend
adds learnable sub-second temporal structure while defining every analysis kernel and stride in
seconds rather than samples. A 25 Hz and a 100 Hz recording therefore evaluate the same continuous
kernel at different sample locations and emit the same physical output frame rate.

## Current shape

For each accelerometer or gyroscope xyz triad:

```text
native xyz samples
  -> 32 shared continuous physical-time kernels on each axis
  -> 96 response channels at 8 frames/second
  -> dense Conv1d(96 -> 64, kernel 3, stride 2) + LayerNorm + GELU
  -> dense Conv1d(64 -> 128, kernel 3, stride 1) + LayerNorm + GELU
  -> four ordered frames per one-second patch
  -> concatenate observability, edge support, amplitude, signed DC, and axis-validity features
  -> project to one token for the physical sensor
```

The dense CNN mixes all three axes within one sensor but never crosses accelerometer/gyroscope or
device boundaries. Missing axes remain explicit through validity bits.

## Kernel parameterization

Each analysis kernel is a smooth, windowed continuous curve represented by a small Fourier basis and
sampled at the recording's native rate. Span and carrier initialization cover human-motion time
scales. Exact per-harmonic observability masks prevent a low-rate source from pretending to resolve
frequencies above its acquisition bandwidth.

The implementation has 32 active analysis kernels and 135,808 frontend parameters, of which 832 are
the continuous analysis bank. Gradients reach every learnable parameter. Mixed-rate, source-rate,
missing-axis, accel-only, modality-isolation, and end-to-end paths have focused regression tests in
`tests/test_continuous_kernel.py`.

## Measured prior result

Under the previous generic HAR protocol, the dense continuous arm improved HALO's zero-shot point
estimate and the learned reranker but slightly reduced direct 1-NN representation performance. The
dataset-bootstrap intervals crossed zero for the headline frontend differences. That result is mixed
and does not promote this frontend universally.

The application comparison is more diagnostic: evaluate fixed and continuous frontends with the
same Task-1 sequence matcher and Task-2 phase-local score. The continuous arm earns its cost only if
its ordered sub-second features improve event boundaries, same-motion verification, or localization
of known execution changes.

## Cost

The last measured RTX 4090 end-to-end profile used about 129 ms per historical episodic training step
and 3.89 GiB allocated VRAM. In an isolated mixed-rate batch of 512 windows, continuous analysis was
the dominant frontend cost; triad packing, dense CNN, and projection were comparatively small.

Application inference must re-profile complete continuous sessions. Reusing overlapping analysis
frames is likely more important than optimizing the ordinary dense CNN.
