# Continuous-time kernel front end — design brainstorm

> ## IMPLEMENTED AND INTEGRATED — 2026-08-24
>
> `ContinuousKernelTokenizer` has 65,472 parameters (832 in the analysis bank) and is available as
> `--frontend continuous` in Phase A and end-to-end episodic training. It uses the same encoder-facing
> contract as `PhysicalFilterbankTokenizer`. The focused suite has 31 tests; the frontend plus encoder,
> conditioning, sensor-folding, and episodic integration suite has 133 tests.
>
> The integrated arm handles mixed stored/native rates, per-recording reflection boundaries,
> source-rate observability, masked calibration, patch-local amplitude/DC, and per-kernel edge support.
> Analysis parameters use a slower no-weight-decay optimizer group, physical-anchor regularization,
> and dedicated gradient/drift telemetry. The current experiment contract is fixed one-second patches
> with JEPA + VICReg; multi-resolution and physical-feature MAE are rejected rather than silently
> misinterpreted.
>
> **Current measured cross-rate agreement** against 100 Hz on a band-limited signal, anti-alias
> decimated to each rate, is **0.989 at 20 Hz · 0.994 at 25 Hz · 0.999 at 50 Hz** over fully observable
> analysis bands. Final-token cosine agreement is **0.991 · 0.993 · 0.999**, respectively.
>
> The next decision is the matched comparison in S7: physical filterbank versus continuous kernels
> under identical data, encoder, objectives, seeds, and step budgets.
>
> Sections below preserve the design exploration that led to the implementation. Where an exploratory
> proposal conflicts with this status block or the code, the status block and code are authoritative.
> In particular, per-band decimation, ragged per-band frame rates, PCEN, and learnable low-pass pooling
> are unimplemented future ablations. Historical synthetic-probe tables are not paper evidence until
> their harness and artifacts are persisted and rerun against the integrated implementation.

**Original proposal: 2026-08-22; current implementation status is recorded above.** A CNN whose kernels are defined as continuous
curves over *real time* (seconds), sampled at whatever rate the signal arrives at, so one set of
weights convolves 20 Hz and 100 Hz recordings without resampling either.

Companion: `COMPACT_EVIDENCE_ENGINE.md` (what it would replace or sit beside),
`../results/ENCODER_COMPARISON_20260822.md` (why the front end is not currently our measured
bottleneck).

---

## 0. Why consider it at all, given the filterbank works

Be honest about the prior. Our filterbank is **already rate-invariant by construction** — it works
in physical frequency, where rDFT bin *m* of a length-*S* transform at rate *r* is exactly
`m·r/S` Hz, so no interpolation exists in the path at all. And the closest thing to this proposal
that we *have* tested came back inert: the `--frontend learnable` arm moved its adaptive gate from
0.10 to 0.143 over 90k steps and shifted band centres by 0.34% mean. A frozen filterbank probe also
beat a raw-waveform CNN of the same budget (0.444 vs 0.397 accuracy).

So this is **not** "the front end is the bottleneck." Three things a continuous *time-domain* kernel
can represent that a magnitude filterbank structurally cannot:

1. **Waveform shape and phase within a band.** We take band *energy*; the phase inside a band is
   discarded. Heel strike is an asymmetric impulse, not a sinusoid — a learned curve can match its
   shape, a Gaussian band cannot. (Partially addressed already by the `use_phase` cross-channel
   feature; this would be the intra-channel version.)
2. **Non-sinusoidal, non-stationary templates.** A constant-Q band is a sinusoid detector. Impacts,
   ramps and one-shot gestures are not sinusoids.
3. **Temporal localisation.** The DFT is taken over a whole patch and returns one vector; a strided
   convolution returns a short time series, which the trunk can then attend over.

That is the case *for*. The case *against* is that all three are second-order compared to the
measured problem — retrieval ranks by acquisition configuration at ×7.0 while
same-activity/different-device rows sit at the 39th percentile — and none of them is obviously the
cause of that.

---

## 1. What the literature already settled

| work | kernel parametrisation | what we should take |
|---|---|---|
| **CKConv** (Romero et al., ICLR 2022) | an implicit MLP `t → w(t)`; kernel length is continuous | the core trick: sample `w` at `t = n/r` to convolve any rate. Also its warning: a plain ReLU MLP cannot fit high-frequency kernels — they needed **SIREN** (sine activations) |
| **FlexConv** (Romero et al., 2022) | CKConv + a **learned Gaussian mask** on kernel extent | receptive-field width should be *learned*, not fixed per kernel |
| **SincNet** (Ravanelli & Bengio, 2018) | band-pass sinc, **two numbers per filter** (low/high cutoff in Hz) | strongest prior that a heavily constrained, physically-parametrised first layer beats a free one |
| **LEAF / EfficientLEAF** (2021/2022) | learnable Gabor filterbank + learned pooling | and the negative result we already cite: *"a learnable frontend of questionable use"* — learnable frontends often fail to beat fixed mel |
| **S4 / SSM** (Gu et al., 2022) | kernel generated from a continuous state-space | continuous-time by construction, explicitly resample-friendly; also the source of the HiPPO init that makes long kernels trainable |
| **Neural ODE / continuous conv** literature | — | the numerical lesson: **treat convolution as an integral**, not a sum (see §4) |

Two literature lessons that decide our design:
- **SincNet's constraint beats CKConv's freedom** on small data. Our corpus is ~550 h, not
  ImageNet. Favour few parameters per kernel and strong structure.
- **EfficientLEAF's negative result is us.** We have already replicated it internally. So the
  proposal must be justified by what it *adds* (shape/phase/localisation), not by "learnable is
  better".

---

## 2. Answering the four design questions

### 2.1 Should stride be in real-world seconds?

**Yes, and it is not optional.** If stride is in *samples*, the number of output frames per patch
becomes `N/stride = r·T/stride`, which is rate-dependent — the tensor shape downstream would then
depend on the recording device, which is exactly the property we are trying to remove.

The clean formulation: fix the **output frame rate**, not the stride.

```
output_frames_per_patch  F   = 8            # fixed, rate-independent
patch duration           T   = 1.0 s
=> frame period              = T / F = 0.125 s
=> stride in samples         = r * T / F    (20 Hz -> 2.5, 100 Hz -> 12.5)
```

Non-integer strides are the immediate wrinkle. Two safe options, in preference order:

- **(a) Evaluate at arbitrary output times.** Do not stride the input; instead, for each output
  frame centre `t_f = (f + 0.5)·T/F`, sample the kernel at `t_f + n/r` and take the dot product.
  This is a `gather` + `einsum`, it is exact at any rate, and it never rounds. Cost is
  `F · taps · C`, which is small.
- **(b) Integer-stride then resample the output** to `F` frames. Simpler, but introduces an
  interpolation we spent the whole filterbank design avoiding.

**Recommend (a).** It keeps the "no interpolation in the path" property that the filterbank docstring
is proud of.

### 2.2 Should temporal resolution reduce at each layer?

**Yes — but only the first layer needs to be continuous.** This is the biggest simplification
available and it comes straight from SincNet/LEAF practice.

After layer 1 the signal already lives on a **fixed real-time grid** (8 frames per second, say). It
is no longer at the device's sampling rate. So layers 2+ are ordinary, fixed-tap CNN or attention
layers, and downsampling them by a factor of 2 is unambiguous — it halves a *known* frame rate.

```
native signal @ r Hz  ──[continuous kernel bank, output F=8 frames/s]──>  8 Hz feature grid
                       ^^^^ the only rate-aware layer
8 Hz grid ──[ordinary conv, stride 2]──> 4 Hz ──[ordinary conv, stride 2]──> 2 Hz ──> pool
```

Rule to enforce in code: **exactly one module may read `sampling_rate_hz`.** If a second one does,
rate-invariance has leaked.

### 2.3 Multi-span kernels — the good idea, and why it is automatically constant-Q

Your framing (every kernel has 16 numbers; some span 0.5 s, others 2 s) is exactly right, and it
reproduces the constant-Q property for free. With `N_c` control points spanning `T_k` seconds:

```
point spacing   = T_k / N_c  seconds
max representable frequency ≈ N_c / (2 T_k)   Hz     (Nyquist of the kernel's own sampling)
```

So a 16-point kernel over 0.5 s reaches ~16 Hz; over 2 s it reaches ~4 Hz. **Log-spacing the spans
gives log-spaced frequency coverage at constant relative bandwidth** — the same design our
filterbank arrives at from the frequency side. Proposed grid, mirroring `FB_F_MIN_HZ`/`FB_F_MAX_HZ`:

```
spans T_k: log-spaced over [0.08 s, 3.0 s]        # ~= [15 Hz .. 0.3 Hz] at N_c = 16
N_c      : 16 control points, shared by every kernel
K        : 24-32 kernels per channel
```

**Hard constraint we must not miss: a 2 s kernel does not fit in a 1 s patch.** `PATCH_SECONDS` is
1.0. So either the front end runs on the **whole 6 s window before patching** (preferred — it is a
convolution, it is happy with the longer input, and it lets long kernels exist), or long kernels are
dropped. This changes where the module sits in the pipeline and is the main structural consequence
of the proposal.

### 2.4 Nyquist and resolution masking

Both of our existing masks have direct analogues, and there is a **third failure mode unique to
continuous kernels** that the filterbank never had.

**(a) Nyquist — now about aliasing the *kernel*, not the signal.** Sampling a continuous kernel at
`r` samples/second aliases any kernel content above `r/2`. A 16-point 0.5 s kernel wants 16 Hz of
detail; at `r = 20 Hz` you get 10 taps for 16 control points and the kernel's own shape is
undersampled. Mask:

```
observable_k = sigmoid( ( r/2 - N_c/(2 T_k) ) / softness )       # 1 = fully resolved
```

Equivalently `r·T_k >= N_c`. Emit this as a per-kernel feature exactly as `nyquist_mask` is today.

**(b) Resolution — the kernel must fit in the window.** A `T_k` kernel needs at least `T_k` seconds
of signal. With window `D`:

```
resolved_k = clamp( D / T_k, 0, 1 )
```

Flag rather than zero, matching `FB_RESOLUTION_MIN_CYCLES` behaviour.

**(c) NEW — anti-aliasing the kernel before sampling.** This one has no filterbank analogue and is
the most likely source of a silent bug. When `r` is low, naively evaluating the spline at `n/r`
*aliases* the kernel's high-frequency content down into the band, so the same physical filter
behaves differently at 20 Hz and 100 Hz — destroying the rate-invariance the whole design exists
for. Two fixes:

- **Band-limit the basis.** Build the kernel from a basis whose bandwidth you control (e.g. a small
  number of Gaussians, or a truncated Fourier series in `t/T_k`) rather than free control points
  plus a spline. Then you can analytically drop components above `r/2`.
- **Pre-smooth then sample.** Convolve the continuous kernel with a Gaussian of width `1/(2r)`
  before evaluating it. One extra parameter, cheap, and it is the standard mip-map argument.

**Recommend the band-limited basis**, because it makes the aliasing question exact instead of
approximate, and because it is what SincNet effectively does.

**A test that must exist:** feed the *same physical signal* resampled to 20/25/50/100 Hz and assert
the front end's output agrees to a tolerance. That is the single most valuable unit test in the
module, and it is the one that would have caught this class of bug in the baseline-swap harness
earlier today.

---

## 3. Numerical quirks and normalisation

These are where a rate-flexible convolution actually goes wrong.

**3.1 Convolution is an integral, not a sum — this is the big one.**
A naive `sum_n w[n] x[n]` scales with the number of taps, so a 100 Hz signal returns ~5× the response
of the same motion at 20 Hz. Use the Riemann form:

```
y(t) = sum_n w(n/r) x(n/r) * (1/r)        # dt = 1/r
```

Equivalently, normalise the sampled kernel to unit **L1 or L2 norm after sampling**, per rate.
Without this, every downstream statistic is a function of the device's sampling rate and the model
will trivially learn to identify the device — which is precisely the ×7.0 acquisition-configuration
failure we already have. **This is the most important line in the module.**

**3.2 Zero-mean kernels.** Force `sum_n w = 0` (subtract the mean after sampling). Otherwise every
kernel partially measures gravity/DC, which we deliberately handle as a separate signed feature. A
DC-sensitive kernel bank would re-introduce the orientation confound through the back door.

**3.3 Per-kernel gain normalisation.** Kernels with different spans have different natural response
magnitudes. Normalise each sampled kernel to unit L2 norm, then apply a learned scalar gain — the
same split the filterbank uses (fixed analysis, learned projection).

**3.4 Compression and standardisation — reuse what works.** Keep `log1p` compression and the
**frozen per-band standardisation** calibrated over the corpus. This is already load-bearing in the
filterbank (the trainer refuses a checkpoint whose norm stats are unfitted), and there is no reason
the continuous bank should differ. Calibrate over the augmented `(rate, duration)` mix, as now.

**3.5 Padding.** Patches are zero-padded to `S = 256`. A convolution *will* see those zeros as
signal. Either run before padding (on the contiguous window — preferred, and consistent with §2.3)
or carry a validity mask and normalise by the count of valid taps per output frame.

**3.6 Initialisation.** Do not init randomly. **Initialise the kernel bank to Gabor wavelets** at
the same 32 log-spaced centre frequencies the filterbank uses — i.e. start as (an approximation of)
the thing we know already works, and let training deviate. This makes the comparison honest: at step
0 the two front ends are near-equivalent, so any difference is attributable to learning rather than
to a different starting point. It also mirrors the `adaptive_gate_init = 0.1` residual trick we
already use.

**3.7 Precision.** Kernel sampling and the dot product should be FP32 even under autocast, as the
filterbank's own path is. Long kernels at high rate accumulate many terms; bf16 will bleed.

**3.8 Cost.** `K` kernels × `taps` × `C` channels × `F` frames. At `K=32`, `T_max=3 s`, `r=100 Hz`
the longest kernel is 300 taps — 32×300×6×8 ≈ 460k MACs per second of signal per window. Trivial on
GPU, but note the longest kernels dominate; consider evaluating long kernels on a *decimated* copy
of the signal (legitimate — they are band-limited by construction).

---

## 4. What to build, and the decisive test

**Scope: one module, opt-in, drop-in.** `model/tokenizer/continuous_kernel.py`, exposing the same
contract as `PhysicalFilterbankTokenizer.forward(patches, sampling_rate_hz, patch_len_samples,
source_rate_hz)`, selected by extending `--frontend {fixed,learnable,continuous}`. No other file
changes. That keeps it revertible and makes the arm comparison matched by construction.

**The test that matters is NOT an accuracy bake-off.** We already know a learnable front end ties on
accuracy. The question this design uniquely answers is **cross-rate transfer**:

> Train on a rate-restricted subset (e.g. 50 Hz sources only), evaluate on held-out 20 Hz and
> 100 Hz sources. Compare `fixed` vs `continuous`. A kernel defined in seconds should degrade less
> than a bank whose rate-robustness comes from masking.

This is a question our current evidence genuinely cannot answer, and — unlike accuracy — it is the
mechanism form of the config thesis in `MOTIVATION.md`, whose language version measured inert. If
the continuous kernel transfers across rates better, that is a *mechanistic* config-invariance
result, which is a stronger and cleaner claim than the conditioning story we have failed to
demonstrate twice.

**Order of work.**
1. Module + the rate-agreement unit test (§2.4) + zero-mean/integral-normalisation tests (§3.1–3.2).
2. Gabor init (§3.6); assert step-0 output correlates > 0.95 with the fixed filterbank's.
3. Frozen-probe comparison on the existing 24k/6k protocol — cheap sanity, expect a tie.
4. **The cross-rate transfer experiment.** This is the one that decides it.

**Kill criterion, pre-registered.** If the continuous front end does not beat `fixed` on held-out
rate transfer by more than the between-run noise band, it is not worth its complexity, and the
correct action is to record the negative result beside the EfficientLEAF citation and keep the
filterbank.


---

# THE PROPOSED DESIGN (spec — build this)

Everything above is the reasoning. This section is the decision.

## S1. Kernel parametrisation — a truncated Fourier series in normalised time

Each kernel `k` is a continuous function of normalised time `u = t/T_k ∈ [-½, ½]`:

```
w_k(u) = envelope_k(u) · Σ_{m=1..M} [ a_km · cos(2π m u) + b_km · sin(2π m u) ]
envelope_k(u) = exp( -u² / (2 σ_k²) )                       # σ_k learnable (FlexConv's learned extent)
```

**M = 8 harmonics ⇒ 16 numbers per kernel**, which is exactly the "16 numbers forming a curve"
formulation. Why this basis rather than control points + spline:

- **Band-limiting is exact, not approximate.** Harmonic `m` sits at precisely `m/T_k` Hz. Dropping
  the ones above Nyquist is a mask on a coefficient, not a smoothing heuristic. This is the fix for
  §2.4(c), the failure mode with no filterbank analogue.
- **Zero-mean is free.** No `m = 0` term ⇒ `∫w = 0` by construction ⇒ kernels cannot measure
  gravity/DC (§3.2), which stays a separate signed feature.
- **Gabor init is a one-liner.** Set `a_k4 = 1`, everything else 0, and the kernel *is* a Gabor
  wavelet at the carrier — i.e. step 0 ≈ the filterbank we know works (§3.6).

## S2. Span grid — carrier at harmonic 4 makes it constant-Q

```
f_k        : 32 log-spaced centres over [0.3, 15.0] Hz     # identical to FB_F_MIN_HZ/FB_F_MAX_HZ
N_CYCLES   : 4                                             # carrier sits at harmonic m = 4
T_k        : clamp(N_CYCLES / f_k, 0.05 s, 4.0 s)          # span in SECONDS
```

With the carrier at `m = 4`, harmonics 1–8 span `[f_k/4, 2f_k]` — a constant relative bandwidth,
which is the constant-Q property arriving from the time side instead of the frequency side. The
4-second clamp is the window budget; low-frequency kernels that hit it are flagged
resolution-limited exactly as `FB_RESOLUTION_MIN_CYCLES` does today.

## S3. Sampling at the signal's rate

```
taps_k(r) = max(round(T_k · r), 3)
u_n       = (n - (taps_k-1)/2) / taps_k          n = 0 … taps_k-1
w_k[n]    = w_k(u_n)                              evaluated in FP32
```

Then, in order, and none of these is optional:

1. **Per-harmonic band-limit.** Zero coefficient `m` where `m/T_k > 0.9 · r/2`. (Measured: at 20 Hz
   this keeps 84% of harmonics and 27 of 32 kernels — the filterbank keeps 26 of 32 bands at the
   same rate, so the two mask comparably by construction.)
2. **Integral, not sum.** Multiply by `dt = 1/r`, or equivalently renormalise `w_k[n]` to unit L2
   *after* sampling. **Without this the response scales with sampling rate and the model learns to
   identify the device** — the ×7.0 acquisition-configuration failure, re-entered through the front
   door.
3. **Re-zero-mean after sampling.** Discretisation breaks the exact `∫w = 0`; subtract the mean.

## S4. Where it runs, and what it emits

Runs on the **contiguous window before patching** (long kernels do not fit a 1 s patch), then slices
its output back into patches so the encoder contract is unchanged. Reconstruct the window from
`patches` + `patch_len` + `patch_padding_mask` by indexing valid samples — the same `searchsorted`
over cumulative patch lengths already written in `model/tokenizer/baseline_backbone.py`.

```
F = 8 output frames per second (fixed, rate-independent) → 8 frames per 1 s patch
per frame f, channel c, kernel k:
    quadrature response  z = (w_k^cos * x)[t_f] + i·(w_k^sin * x)[t_f]
    magnitude            e = |z|          → log1p compression → frozen per-band standardisation
per (patch, channel) token, concatenate:
    mean_f(e) (K) ‖ max_f(e) (K) ‖ std_f(e) (K) ‖ nyquist_mask (K) ‖ resolution_flag (K)
      ‖ amplitude (1) ‖ signed DC (1)                                  = 5K + 2 = 162 for K = 32
    → shared Linear(162 → d_model)
```

`mean/max/std` over the 8 frames is what buys the **temporal localisation** the DFT path cannot
give, while keeping one token per (patch, sensor) so nothing downstream changes.

## S5. Budget, contract, integration

| | |
|---|---|
| analysis params | 32 × 16 coeffs + 32 σ + 32 gains = **576** |
| projection | `Linear(162 → 128)` = 20,864 |
| **total** | **≈ 21,440** (filterbank: 12,672 + 96) |
| contract | `forward(patches, sampling_rate_hz, patch_len_samples, source_rate_hz) → (B,P,C,d)` — drop-in |
| selection | extend to `--frontend {fixed, learnable, continuous}`; **no other file changes** |
| precision | FP32 island for kernel sampling and the dot product, as the filterbank already does |
| cost note | long kernels dominate; they are band-limited by construction, so evaluate them on a decimated copy of the signal |

## S6. Tests that must exist before any training run

1. **Rate agreement (the important one).** One synthetic signal resampled to 20/25/50/100 Hz →
   outputs agree within tolerance. This is the property the whole design exists for, and it is the
   class of bug that bit the encoder-swap harness on 2026-08-22.
2. **Zero-mean.** A pure DC input produces ~zero response on every kernel.
3. **Amplitude linearity.** Scaling the input scales the pre-compression magnitude proportionally.
4. **Gabor-init equivalence.** At init, per-band output correlates > 0.95 with
   `PhysicalFilterbankTokenizer`'s. If it does not, the two arms do not start from the same place
   and no later comparison is attributable to learning.
5. **Padding isolation.** Poisoning the padded region leaves the output unchanged.

## S7. The experiment, and the pre-registered kill criterion

Order: build + tests (S6) → frozen-probe sanity on the existing 24k/6k protocol (expect a tie) →
**the cross-rate transfer experiment**: train on a rate-restricted subset (50 Hz sources only),
evaluate on held-out 20 Hz and 100 Hz sources, `fixed` vs `continuous`, matched seed and schedule.

**Kill criterion.** If `continuous` does not beat `fixed` on held-out rate transfer by more than the
between-run noise band, it does not earn its complexity: record the negative result beside the
EfficientLEAF citation and keep the filterbank. Given that our learnable-filterbank arm is inert and
the frozen bank already beats a raw CNN, **the prior is that this fails** — which is exactly why the
criterion is written down before the run rather than after it.

---

# S8. Stacking — the decision, and the rules if you stack anyway

## S8.1 Recommendation: do NOT build a deep front end for the first experiment

The instinct to stack (widen channels, non-linearities, pool) is correct CNN practice and wrong
*here*, for a reason specific to our architecture: **HALO already has the depth.** The temporal
trunk is 3 attention layers over per-patch tokens with a receptive field of the whole window. A deep
convolutional front end would be a second temporal model stacked under an existing one.

Two hard numbers make the point:

- A 1 s patch at 8 frames/s supports **at most 3 stride-2 stages** before it is a single frame
  (8 → 4 → 2 → 1). There is very little temporal extent to spend.
- The depth you would be adding is depth the trunk already has, over a longer span.

And a methodological reason that matters more: the experiment is *"do continuous kernels beat a
fixed filterbank"*. If the continuous arm is also 3 layers deeper and 4× wider, a win is
unattributable — the same confound that made this morning's `ours-fixed` vs `ours-learnable` probe
row uninterpretable (different run lengths, not different front ends). **Arm A must differ from the
filterbank in exactly one respect.**

So: **one continuous layer, magnitude, compression, pool to a token.** Depth stays where it is.

## S8.2 If you stack anyway — the rules

Should the shallow arm win and you want to spend depth on the front end, these are the constraints,
and the first one is the only truly novel constraint in the design.

**Rule 1 — exactly one layer is rate-aware.** Layer 1 consumes the native-rate signal and emits a
**fixed 8 Hz real-time grid**. Every later layer sees that grid regardless of whether the recording
was 20 Hz or 100 Hz, so layers 2+ are **ordinary fixed-tap convolutions**. Do not make them
continuous — there is no variable rate left for a continuous kernel to solve, and a second
rate-aware module is how rate-invariance leaks. Enforce by assertion: only the layer-1 module may
read `sampling_rate_hz`.

**Rule 2 — kernel sizes upstairs are still stated in seconds.** A 5-tap kernel on an 8 Hz grid spans
0.62 s. Write it that way in the config so the physical meaning survives a change of `F`. Stacked
5-tap layers with dilation 1/2/4 reach 0.62 s → 1.62 s → 3.62 s of receptive field, which covers a
gait cycle by layer 2 and the whole window by layer 3.

**Rule 3 — channel growth is per input channel, and the fold stays downstream.** Layer 1 emits `K`
kernel responses **per sensor channel**, kept separate (as the filterbank does) so `sensor_fold`
still owns cross-channel mixing. Widen `K → 2K → 4K` within a channel, never across channels. A
front end that mixes accel and gyro breaks the sensor-isolation property the retrieval rows depend
on, and there is a test asserting it.

**Rule 4 — non-linearity: note that layer 1 already has one.** `|z|` (quadrature magnitude) followed
by `log1p` is a compressive non-linearity, and it is the one that makes the features behave like
band energies. Layers 2+ use GELU. Do not put a ReLU directly on the layer-1 complex response — you
would be half-wave rectifying a phase, which is meaningless.

**Rule 5 — normalise over channels, never over time.** LayerNorm/GroupNorm across the feature dim at
each frame is safe. **Any statistic pooled over the time axis is rate- and length-dependent** and
will silently reintroduce a device fingerprint. BatchNorm is doubly forbidden: it mixes statistics
across recordings from different devices in the same batch. Keep the **frozen per-band
standardisation** calibrated over the corpus for layer 1, as the filterbank does.

**Rule 6 — pooling and variable length.** The length problem is already solved by patching and must
stay that way:

- *Within a patch*: a patch is a fixed **1 s**, so it always yields exactly 8 frames. Stride-2
  stages reduce 8 → 4 → 2 → 1 deterministically. **Length-invariant by construction** — variable
  recording length shows up as a variable number of *patches*, which `patch_padding_mask` already
  handles, not as a variable number of frames.
- *Across the window* (the conv runs on the contiguous window for long kernels): pad the window
  edges by `T_max/2`, and use **masked** pooling everywhere so padding never enters a mean.
- *Too short*: ~0.1 % of corpus windows carry a **single sample** (wisdm, capture24, opportunity).
  These must produce a row, not an exception — this exact case crashed all four arms of the
  encoder-swap harness on 2026-08-22. Wrap-pad to the minimum tap count and let the resolution mask
  flag it.
- *Too long*: not reachable through the grid loader (windows cap at 6 s), but masked pooling makes
  it a non-issue if it ever is.

**Rule 7 — keep the token contract.** Whatever the depth, the module still emits **one vector per
(patch, sensor)**. Everything downstream — `sensor_fold`, descriptor conditioning, the trunk, the
retrieval rows, the memory bank — is unchanged. That is what keeps the arm comparison matched and
the change revertible.

## S8.3 Build order

1. **Arm A (shallow, S1–S7).** One continuous layer. This is the experiment.
2. Only if A wins: **Arm B** adds 2 ordinary dilated stages under Rules 1–7, and is compared
   against **A**, not against the filterbank — so depth is isolated from the kernel change.

---

# S9. REVISION — S4/S8.1 were wrong about temporal collapse

**Correcting the record.** S4 pooled the 8 frames of a patch into `mean/max/std`, and S8.1 argued
that depth was unnecessary because "the trunk already has it". Both are wrong for the same reason,
and it defeats the point of the proposal.

**The trunk's depth is at 1-second granularity.** It attends over *patch* tokens. It cannot see
anything below one second. So an order-invariant pool over the 8 frames does not hand sub-second
structure to a deeper model — it **destroys it at the only point in the pipeline where it existed**,
which is the same collapse the filterbank performs, merely one layer later. If the whole motivation
for a convolutional front end is to keep *when* something happened and not only *how much energy*
was present, the design must not do that.

What is actually being lost (measured against a 1 s token boundary):

| structure | scale | visible at 1 s tokens? |
|---|---:|---|
| heel strike / impact transient | ~60 ms | **no** |
| arm-swing reversal | ~150 ms | **no** |
| step (half gait cycle) | ~500 ms | **no** |
| gait cycle | ~1000 ms | yes |

Three of the four are invisible to HALO today. That is the gap this front end exists to close.

## S9.1 Revised S4 — keep the frames ORDERED, and stack across them

Replace the order-invariant pool with an order-preserving one, and put the stack **inside the
patch**, on the fixed 8 Hz grid where it is rate-safe:

```
layer 1 (continuous, rate-aware)   native rate → (K=32, F=8 frames) per patch per channel
layer 2 (ordinary conv, 3 taps)    32 → 64,  frames 8 → 8      GELU, GroupNorm over channels
layer 3 (ordinary conv, 3 taps)    64 → 128, frames 8 → 4      GELU, GroupNorm  (stride 2)
flatten                            128 × 4 = 512 ORDERED features
concat masks                       ‖ nyquist(K) ‖ resolution(K) ‖ amplitude ‖ signed DC
→ Linear(578 → d_model)
```

The flatten is the point: the projection sees frames **in order**, so a rising ramp and a falling
ramp are different inputs. `mean/max/std` could not tell them apart. Layers 2–3 are ordinary convs
because after layer 1 the grid is a fixed 8 Hz regardless of the device rate — Rule 1 still holds,
and this is where stacking genuinely earns its place rather than duplicating the trunk.

`F = 8` (125 ms) is the recommended grid: it resolves the arm-swing and step scales, and it costs a
512-wide flatten. `F = 16` (62 ms) would reach the impact transient at 1024 wide — worth an ablation,
not the default.

## S9.2 The bigger alternative, costed but not recommended yet

The thorough answer to "preserve temporal information" is to stop collapsing at the patch at all:
emit **one token per frame** rather than per patch, and let the trunk attend at 125 ms resolution.
That is architecturally cleaner and it is what a pure CNN-plus-transformer design would do.

It is not free, and the cost lands squarely on Phase B:

```
today      6 patches × 2 sensors  =  12 retrieval rows per window
per-frame  48 frames × 2 sensors  =  96 retrieval rows per window        → 8×
```

Eight times the trunk sequence length, eight times the rows in every episode's memory bank (512
windows ≈ 4,000 rows → ≈ 32,000), eight times the pair-scoring matrix, and a top-64 that now
selects among far more, mostly redundant, neighbours. The bank is already 97% of training compute.

So: **S9.1 first** — it recovers ordered sub-second structure inside the existing token contract, at
no cost to Phase B. Treat per-frame tokens as a separate, later decision that should be argued on
its own evidence, not smuggled in with the front-end change.

## S9.3 Consequence for S8.3

Build order is revised. Arm A is now the S9.1 stack (continuous layer + 2 ordinary layers + ordered
flatten), not the shallow pooled version — because the shallow version does not test the hypothesis.
The filterbank remains the control, and the single respect in which Arm A differs is: *time-domain
kernels with ordered sub-second output*, versus *whole-patch band energies*. That is one claim, and
it is the claim.

---

# S10. Why ordinary convs above layer 1, the shape trace, and the cost correction

## S10.1 Why layers 2+ are ordinary, not continuous

A continuous kernel exists to **decouple the kernel's parametrisation from the sample spacing**,
because the spacing varies from 20 to 100 Hz and is not known until the recording arrives. That
problem is fully solved by layer 1: its output sits on a grid of exactly **125 ms per frame at every
rate**. Above layer 1 the spacing is fixed and known, so:

- An ordinary conv on a known grid **is** a continuous kernel evaluated at that grid; the taps are
  the parametrisation. Continuous parametrisation adds cost and buys nothing.
- What continuous parametrisation *does* still impose is a **smoothness/band-limit prior** — but at
  3 taps there is nothing to be smooth about. A "continuous curve through 3 points" is 3 numbers.
- The compression only pays where taps are many and rate-varying: layer 1's longest kernel is **400
  taps at 100 Hz and 80 at 20 Hz, both described by the same 16 numbers**. That is a 25× compression
  *and* the mechanism that makes one kernel serve both rates. Layer 2 has 3 taps described by 3
  numbers. Nothing to compress, nothing to reconcile.

Same conclusion as Rule 1, now with the reason rather than the assertion.

## S10.2 The time axis, traced (this is the "pixels" question)

`F = 8` frames/s, 1 s patches, 6 s window:

| stage | 20 Hz | 50 Hz | 100 Hz | per patch | ms / pixel |
|---|---:|---:|---:|---:|---:|
| input samples | 120 | 300 | 600 | varies | varies |
| after layer 1 (continuous) | **48** | **48** | **48** | 8 | 125 |
| after layer 2 (3-tap, stride 2) | **24** | **24** | **24** | 4 | 250 |
| after layer 3 (3-tap, stride 1) | **24** | **24** | **24** | 4 | 250 |

**The time axis becomes rate-independent at layer 1 and stays that way** — that is the whole
property, visible as three identical columns. Full tensor trace:

```
(B, C=6, T=120..600)        native, T rate-dependent
  → layer 1 (continuous)    (B, 6, 32, 48)     48 frames at ANY rate
  → layer 2 (dw-sep, s=2)   (B, 6, 64, 24)
  → layer 3 (dw-sep, s=1)   (B, 6, 64, 24)
  → slice to patches        (B, P=6, C=6, 64, 4)
  → flatten frames          (B, 6, 6, 256)     ORDERED
  → ‖ masks ‖ amp ‖ DC      (B, 6, 6, 256 + 2K + 2 = 322)
  → Linear(322 → 128)       (B, P, C, d)       unchanged contract
```

**Receptive field per output pixel:** layer 1 contributes the kernel span itself (0.27 s at the top
band to 4.0 s at the bottom); layers 2–3 add 5 frames = 0.625 s on top of that.

**Short windows work out exactly.** Every window length yields 4 frames per patch, because a patch
is a fixed 1 s and `F = 8` is divisible by the single stride-2:

| window | 1 s | 2 s | 3 s | 5 s | 6 s |
|---|---:|---:|---:|---:|---:|
| frames after L1 | 8 | 16 | 24 | 40 | 48 |
| after stride 2 | 4 | 8 | 12 | 20 | 24 |
| **per patch** | **4** | **4** | **4** | **4** | **4** |

Single-sample windows (~0.1 %) wrap-pad to the minimum tap count and are flagged by the resolution
mask — they must return a row, never raise.

## S10.3 Cost correction — S9.1's stack was too expensive

Measured at a realistic step volume (B ≈ 2,080 windows: bank + queries), against the temporal trunk
at **9.8 GFLOP/step** as the yardstick:

| variant | GFLOP/step | vs trunk |
|---|---:|---:|
| S9.1 as written — dense 32→64→128 | 22.1 | **2.3×** |
| dense, stride 2 moved to layer 2 | 18.4 | 1.9× |
| **depthwise-separable 32→64→128** | 6.3 | 0.64× |
| **depthwise-separable 32→48→64** | 2.9 | **0.30×** |
| one dw-sep layer 32→64 only | 1.3 | 0.13× |

*(layer 1 decimated: 1.2 · filterbank: 0.31)*

**S9.1 as written would cost 2.3× the entire temporal trunk** for a front end — unacceptable for an
arm meant to be a controlled swap. Revised default:

> **Depthwise-separable convolutions, 32 → 64 → 64, stride 2 at layer 2.** ≈2.9 GFLOP/step, under a
> third of the trunk, flatten width 64 × 4 = **256** ordered features.

Depthwise-separable is the right tool here for a structural reason, not only a cost one: the
depthwise stage mixes **time within a kernel band**, and the pointwise stage mixes **across bands at
a fixed time**. That factorisation matches the physics — a band's temporal envelope and the
cross-band pattern at an instant are different kinds of structure — and it is why MobileNet-style
separation costs little accuracy here.

**Also correct the layer-1 cost.** Naively convolving the 4 s kernel at 100 Hz is 400 taps and
16.7 GFLOP/step. But every kernel is band-limited to `8/T_k` Hz *by construction*, so it may be
evaluated on a decimated copy of the signal at `max(2·8/T_k, 4)` Hz with no loss. That is a **13.6×
saving at 100 Hz** (16.7 → 1.2 GFLOP) and it makes the layer-1 cost nearly rate-independent, which
is aesthetically right: the same physical analysis should not cost 5× more because the device
sampled faster.

---

# S11. Kernel spans, overlap, and two consequences worth knowing

Yes — kernels are **variable length by design**, one span per band, spanning a **15× range**.

## S11.1 The span table

`T_k = clamp(N_CYCLES / f_k, 0.05 s, T_MAX)` with `N_CYCLES = 4`:

| band | centre (Hz) | span (s) | cycles in span | taps @20 Hz | taps @100 Hz |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.30 | 4.000 | 1.20 | 80 | 400 |
| 6 | 0.64 | 4.000 | 2.56 | 80 | 400 |
| 12 | 1.36 | 2.933 | 4.00 | 59 | 293 |
| 18 | 2.91 | 1.375 | 4.00 | 28 | 138 |
| 24 | 6.20 | 0.645 | 4.00 | 13 | 65 |
| 31 | 15.00 | 0.267 | 4.00 | 5 | 27 |

Spans run **0.267 s to 4.0 s**. Ten of 32 kernels hit the cap and therefore see *fewer* than 4
cycles — the 0.3 Hz kernel sees only 1.20 — which is precisely the condition the resolution mask
already flags in the filterbank (`FB_RESOLUTION_MIN_CYCLES = 1.0`). Nothing new, but the flag must
be wired or those bands quietly report blur as signal.

## S11.2 Consequence 1 — adjacent frames overlap heavily

With stride 125 ms and spans up to 4 s, successive output frames share most of their input:

| band | span | adjacent-frame overlap |
|---:|---:|---:|
| 15.00 Hz | 0.267 s | 53% |
| 5.47 Hz | 0.732 s | 83% |
| 1.99 Hz | 2.008 s | 94% |
| 0.30 Hz | 4.000 s | **97%** |

Cross-checked against what each band actually *needs*: a band's envelope has bandwidth ≈ `f/Q`, so
it needs ~`f/2` frames/s. The top band needs 7.5 fps and we give 8 — right. The 1 Hz band needs
1 fps and gets 8 — **8× redundant**; the bottom bands are **16× redundant**.

So the low bands carry almost no independent information across the 8 frames of a patch. This is
**not a correctness problem** (it is cheap after decimation, and the projection can learn to ignore
it), but it is wasted flatten width and it should be recorded rather than discovered later. The
principled fix is a **per-band frame rate** — a wavelet/constant-Q pyramid, where low bands emit
1–2 frames per patch and high bands emit 8. That gives ragged tensors and is a natural **ablation**,
not the v1 default.

## S11.3 Consequence 2 — long kernels are mostly edge-padded, and this sets `T_MAX`

A 4 s kernel in a 6 s window is fully supported for only 2 s of frame positions:

| `T_MAX` | frames fully supported (longest kernel) | padded | bands hitting the cap |
|---:|---:|---:|---:|
| 1.0 s | 40 of 48 | 17% | 21 |
| **2.0 s** | **32 of 48** | **33%** | **16** |
| 3.0 s | 24 of 48 | 50% | 12 |
| 4.0 s | 16 of 48 | 67% | 10 |

Per band at `T_MAX = 4`: the 15 Hz kernel is fully supported for 94% of frames, the 2 Hz kernel for
65%, and the sub-1 Hz kernels for only **33%**. Two-thirds of the low bands' output is computed
against padding.

**Recommendation: `T_MAX = 2.0 s`.** It keeps two-thirds of frames fully supported for every kernel,
still reaches 2 cycles at 1 Hz and 1 cycle at 0.5 Hz, and the bands below that were already
resolution-flagged in the filterbank — we are not losing information we currently have, we are
declining to fabricate it. The bands it clamps hardest (0.3–0.9 Hz) are also the ones the signed-DC
feature and the trunk's cross-patch attention already cover.

Whatever `T_MAX` is chosen: **pad by reflection, not zeros** (a zero-padded edge looks like a
step discontinuity to a band-pass kernel and injects broadband energy that is not in the signal),
and emit a per-frame **edge-support fraction** alongside the Nyquist and resolution masks so the
model can discount partially-supported frames rather than trusting them equally.

---

# S12. Cross-rate agreement — MEASURED, and the bug it exposed

S3 said "evaluate at arbitrary output times". That sentence was doing more work than it looked, and
under-specifying it produces a design that silently fails the property it exists for. Simulated
end-to-end (band-limited signal generated at 400 Hz, anti-alias **decimated** to each rate as a real
ADC would, kernel `T = 1.0 s`, 8 harmonics, 8 frames/s output), comparing every rate against 100 Hz:

| implementation | corr @20 Hz | corr @25 Hz | corr @50 Hz |
|---|---:|---:|---:|
| **naive: sample kernel on a symmetric grid, round frame centre to nearest sample** | **0.844** | 0.972 | 0.971 |
| **correct: evaluate the kernel at the exact offsets of the real samples** | **0.986** | 0.987 | 0.998 |

**The bug.** Rounding the output-frame centre to the nearest input sample introduces up to half a
sample of jitter — at 20 Hz that is 25 ms, which is **45° of phase error on a 5 Hz component**. The
fix is the reason to have a continuous kernel at all: for output time `t_f`, take the real sample
times `t_n = n/r` that fall inside the span and evaluate

```
w[n] = w_k( (t_n - t_f) / T_k )
```

so the kernel **absorbs the sub-sample offset** exactly. Never resample the signal, never round the
centre. This is a hard requirement, not an optimisation, and the rate-agreement test in S6 is what
catches it.

**Each of the three S3 rules is load-bearing** — ablated at 20 Hz vs 100 Hz:

| rule removed | correlation | magnitude ratio 20/100 |
|---|---:|---:|
| no `dt = 1/r` (plain sum) | 0.698 | **0.213** |
| no per-harmonic band-limit | 0.644 | 1.067 |
| no re-zero-mean | 0.699 | 1.068 |

Dropping `dt = 1/r` makes a 20 Hz recording return **one fifth** the response of the same physical
motion at 100 Hz — the device fingerprint, straight back in.

**Residual error is genuine, not a defect.** After the fix, ~0.17 relative RMS remains at 20 Hz. It
is quadrature error: a 1 s kernel gets 20 taps at 20 Hz and 100 at 100 Hz, so the Riemann sum
approximating `∫w·x dt` is simply coarser. `L1` normalisation matches `dt`; `L2` is **worse**
(0.556 rel RMS) because it rescales per-rate and destroys amplitude comparability — **use `dt = 1/r`
or `L1`, never `L2`.**

**Two separate causes must not be conflated:**
- *Kernel band-limiting* — a `T = 0.5 s` kernel with 8 harmonics wants 16 Hz, which 20 Hz sampling
  cannot represent, so harmonics 5–8 are dropped and it is **genuinely a different filter** (corr
  0.931). Correct behaviour, and the Nyquist mask flags it.
- *Signal information loss* — a broadband transient sampled at 20 Hz is genuinely not the same
  signal (corr 0.836 even with a perfect front end). Nothing can fix that, and nothing should try.

Design consequence: prefer kernels whose top harmonic is representable at the **lowest** corpus rate
where possible. At 20 Hz the limit is 9 Hz, so `T_k ≥ M/9 ≈ 0.9 s` is fully representable
everywhere; shorter kernels degrade gracefully to fewer harmonics and are masked.

---

# S13. Do we need patches at all? (No — and the design already reflects that)

Correct on both counts, and worth making explicit.

**Patching is an artefact of the filterbank, not a requirement.** The DFT needs a fixed-length
window, so the loader pre-divides into 1 s patches. A convolution has no such need: it consumes a
contiguous signal of any length and emits a sequence whose length scales with duration.

**The design already produces exactly that.** Layer 1 emits 8 frames per second of recording, and
Step 6 groups frames into one token per second. So a 3 s recording yields **3 tokens** and a 6 s
recording yields **6** — the token count already scales with duration, which is the intuitive
behaviour you describe. Variable-length recordings are handled by the existing
`patch_padding_mask`, exactly as now.

What changes is the *status* of the 1 s grid. Today it is **structural** — the DFT window. With a
convolutional front end it becomes a **free hyperparameter**: the token stride is just where we
pool, and 0.5 s (12 tokens per 6 s window) or 2 s (3 tokens) are equally legal. That is a genuine
gain in flexibility, and it makes token stride an ablation we could never run before.

**Keep the patched input format for v1 anyway.** `MultiScaleCollate` already delivers `(B,P,S,C)`,
and the module reconstructs the contiguous window from `patch_len` + `patch_padding_mask` before
convolving (the `searchsorted`-over-cumulative-lengths routine already written in
`baseline_backbone.py`). Reconstructing and re-slicing is mildly redundant, but it means **zero
loader changes**, which keeps the arm a controlled swap. Changing the loader to deliver contiguous
signals is a clean follow-up once the front end has earned it — not a prerequisite.

The one thing to preserve: **token count must remain a function of duration only, never of sampling
rate.** 6 s gives 6 tokens at 20 Hz and at 100 Hz alike. That is the invariant the S6 rate-agreement
test should assert on shapes as well as values.

---

# S14. Debug sweep + capacity audit — 2026-08-23

Swept the built module for numerical behaviour, literature conformance, and capacity. **Two real
defects found and fixed**; both are now regression-tested.

## S14.1 DEFECT — a duration fingerprint (fixed)

`GroupNorm(1, C)` on `(N, C, L)` normalises over **channels *and* time**. It looks like a per-frame
channel norm and is not one. Measured: two windows sharing their first 2 s but differing afterwards
produced tokens for that **shared span** differing by **max |Δ| = 2.43**.

That is a *duration fingerprint* — the same class of leak as a rate fingerprint, and precisely what
Rule 5 forbids. A 2 s and a 6 s recording of the same opening motion would encode differently, and
nothing in a loss curve would ever reveal it. Replaced with `LayerNorm` over the channel dimension
at each frame (delta → 0.0). Test:
`test_features_do_not_depend_on_how_long_the_rest_of_the_recording_was`.

## S14.2 CAPACITY — 8 harmonics were too few for the motivating shape (fixed: M = 8 → 12)

Measured the fraction of an arbitrary kernel shape's variance that `M` harmonics can express:

| target shape | M=4 | **M=8** | **M=12** | M=16 |
|---|---:|---:|---:|---:|
| Gabor (the init) | 0.871 | **1.000** | 1.000 | 1.000 |
| **asymmetric impact (heel strike)** | 0.591 | **0.763** | **0.836** | 0.876 |
| chirp | 0.324 | **0.616** | **0.922** | 0.991 |
| double pulse | 0.846 | 0.979 | 1.000 | 1.000 |
| **sharp spike** | 0.442 | **0.754** | **0.916** | 0.978 |
| random smooth | 0.246 | 0.466 | 0.611 | 0.815 |

The whole argument for a time-domain kernel is *"a heel strike is an asymmetric impulse, not a
sinusoid"* — and M=8 captured only **76%** of that shape. Raised to **M = 12** (24 coefficients per
kernel, bank 576 → 832 params). Test asserts R² > 0.80 on an asymmetric impact.

**The cost, quantified — a genuine tension worth knowing.** A kernel is fully representable only
where `f_k ≤ 1.8·r / M`. More harmonics means sharper shapes but *more rate-fragility*:

| M | bank params | fully-live kernels @20 Hz | corr @20 Hz | corr @50 Hz |
|---:|---:|---:|---:|---:|
| 8 | 576 | 22/32 | 0.99484 | 0.99978 |
| **12** | **832** | **19/32** | **0.99455** | 0.99975 |
| 16 | 1,088 | 16/32 | 0.99494 | 0.99973 |
| 24 | 1,600 | 0/32 | — | 0.99972 |

Cross-rate correlation is *unaffected* for kernels that stay intact; what changes is how many stay
intact. **You cannot have both full shape capacity and full band coverage at 20 Hz** — a shaped
kernel needs more bandwidth than a pure sinusoid, so our high bands are inherently more
rate-fragile than the filterbank's. M=12 trades three Nyquist-masked bands at the lowest rate for
substantially better shape capacity. M=24 is unusable (nothing survives at 20 Hz).

## S14.3 CAPACITY — functional test: can it represent what the filterbank cannot?

Two classes with **power spectra matched to corr 0.9999** (impulsive pulse trains vs their
phase-randomised surrogates), so the discriminating information is *waveform shape only*. Linear
probe on frozen features, held-out split, chance 0.500:

| front end | probe accuracy |
|---|---:|
| filterbank (magnitude per patch) | 0.865 |
| **continuous kernel, Gabor init, frozen** | **0.942** |
| **continuous kernel, after 200 training steps** | **0.981** |

Two things this shows. The architecture beats the filterbank **before any learning** (0.942 vs
0.865) — that gain is the **ordered sub-second frames**, not the kernel shapes. Training then adds
0.04 on top, with coefficients moving 0.019 from init, so the 24 numbers per kernel are genuinely
being used rather than sitting at their Gabor initialisation.

*(An earlier version of this test had both front ends at 1.000 because the construction leaked
spectral cues. The phase-randomised surrogate is the fair version; the first attempt was discarded.)*

## S14.4 Numerical health

- **No dead kernels** (0 of 32 with ~zero variance); per-kernel activation sd spans 0.0007–0.0996,
  which tracks where the probe signal actually has energy rather than indicating collapse.
- **Gradients reach every learnable**, all finite. `sin_coeff` and `gain_logit` start at exactly zero
  by construction (Gabor init), so their gradient/weight ratio is undefined at step 0 — expected,
  and they receive non-zero gradient immediately.
- Token output at init: mean +0.033, sd 0.454, |max| 1.20 — a healthy scale for the trunk.

## S14.5 Literature conformance

| best practice | status |
|---|---|
| Gabor/mel initialisation rather than random | ✅ as SincNet and LEAF both do |
| windowed kernel (avoid ringing) | ✅ Gaussian envelope; SincNet uses Hamming |
| per-band compression + standardisation | ✅ `log1p` + frozen stats — **LEAF reports PCEN beats log**; an untested alternative |
| quadrature magnitude (phase-invariant energy) | ✅ `\|z\|` from a cos/sin pair, as LEAF |
| constrained parametrisation on small data | ✅ 24 numbers/kernel, not a free MLP |
| learnable pooling after the bank | ⚠️ **partial** — strided depthwise conv; LEAF uses a learnable Gaussian low-pass |
| anti-alias before sampling the kernel | ✅ exact per-harmonic mask — **CKConv does not do this** |
| integral (`dt`) normalisation for variable rate | ✅ the rate-comparability requirement |

**Parameter budget against the field:** analysis bank **832** vs SincNet ~160 (2/filter), LEAF ~320,
CKConv 10–50k per layer (an implicit SIREN MLP). We sit between the constrained and free camps,
which is where the literature puts small-data front ends — and we have 2.6× LEAF's analysis
capacity. Note that **88% of the module's 65,472 parameters are the output projection**, though the
fixed filterbank is 99% projection, so this is the more balanced of the two.

**Two open items, neither blocking:** PCEN instead of `log1p` (LEAF's reported win), and a learnable
low-pass pooling instead of the strided conv. Both are ablations for after the front end has earned
its place.
