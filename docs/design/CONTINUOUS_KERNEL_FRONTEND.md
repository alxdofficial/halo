# Continuous-time kernel front end — design brainstorm

**Status: proposal, 2026-08-22. Nothing built.** A CNN whose kernels are defined as continuous
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
