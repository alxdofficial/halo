# HALO Learnable Tokenizer Arm

Status: implementation plan, accepted 2026-07-20.

## Question

Does a mildly adaptive, physically constrained frontend improve HALO Phase-A representations over
the current fixed physical filterbank without learning dataset, subject, placement, or sampling-rate
shortcuts?

The experiment preserves the corpus, augmentation stack, channel-text path, encoder, Phase-A
objectives, optimizer, batch sampler, and training budget. Adaptation is restricted to a small set of
continuous physical frontend parameters. Simultaneous multiresolution tokenization is independently
switchable so its contribution can be measured rather than silently confounded with learnability.

## Experiment Matrix

The two headline arms are:

| Arm | Frontend | Patch presentation |
|---|---|---|
| `fixed` | current fixed physical-Hz filterbank | one duration drawn per batch |
| `learnable` | constrained adaptive physical-Hz filterbank | one short and one long duration per batch |

Two diagnostic runs remain available through independent configuration switches:

| Diagnostic | Purpose |
|---|---|
| fixed frontend + multiresolution | isolates the benefit of simultaneous temporal supports |
| learnable frontend + single resolution | isolates constrained frontend adaptation |

The headline comparison measures the complete proposed learnable frontend package. Claims that
learnability itself caused an improvement require the diagnostic runs.

## Constrained Adaptive Filterbank

The learnable frontend starts exactly at the fixed frontend and shares one parameterization across
all datasets, rates, devices, placements, channels, and resolutions.

1. **Band centers.** Keep the 0.3 Hz and 15 Hz endpoints fixed. Each interior log-frequency center
   receives a bounded displacement of at most 45% of the original adjacent log-band spacing. This
   guarantees ordered centers and prevents bands migrating to arbitrary acquisition artifacts.
2. **Bandwidth.** Each Gaussian bandwidth receives a bounded multiplicative adjustment in
   `[1/1.5, 1.5]`, initialized at 1. The adjustment is shared across configurations but independent
   per physical band.
3. **Compression knee.** Each band receives a bounded pre-`log1p` gain in `[1/2, 2]`, initialized at
   1. This changes the compression knee without replacing the physically meaningful log-energy
   transform.
4. **Filter shape.** One global generalized-Gaussian exponent is learned in `[1.5, 2.5]`, initialized
   at 2 (the current Gaussian). A global value deliberately avoids 32 independently shaped filters.
5. **Fixed/adaptive residual.** Standardized fixed features remain an anchor. A single learned
   convex gate, initialized to 0.1, mixes fixed and adaptive compressed energies. DC, amplitude,
   Nyquist observability, and resolution metadata remain on the fixed physical path.
6. **Regularization.** Center shifts, log-bandwidth factors, log-compression gains, and shape
   displacement receive a small normalized quadratic penalty. The penalty is reported separately.

The Phase-A A1 target remains the frozen, calibrated fixed filterbank. The adaptive frontend cannot
move its own prediction target.

## Simultaneous Multiresolution Tokens

For every augmented recording, draw one duration from each non-overlapping set:

```text
short = {0.4, 0.5, 0.6, 0.7, 0.8} seconds
long  = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5} seconds
```

Accept a pair only when `long >= 1.75 * short`. Draw once per batch for efficient rectangular
tensors. Apply signal augmentation once, then patchify that same view at both resolutions.

Patch intervals are constructed explicitly. A final partial patch is retained with its honest sample
length, center, duration, start, and end; no tail is silently discarded and no end-anchored duplicate
is introduced. Tokens from both scales are sorted by physical center time and presented together to
the existing temporal attention. Cross-channel attention remains within each patch token.

The shared encoder receives more tokens, not separate scale-specific encoders:

```text
same signal -> shared filterbank -> duration-conditioned tokens -> shared factorized transformer
```

## Numerical Metadata

The existing numerical routes remain:

- sampling rate maps DFT bins into physical Hz;
- true patch length controls the Hann window, DC estimate, and resolution;
- physical patch-center seconds drive RoPE;
- observability and resolution values are bounded in `[0, 1]`;
- channel and patch validity are boolean masks.

Multiresolution adds one explicit per-token duration encoding:

```text
u = 2 * (log(duration) - log(0.4)) / (log(1.5) - log(0.4)) - 1
```

`u` is clamped to `[-1, 1]`, projected by a small `1 -> 16 -> d_model` network, and added through a
learned scalar gate initialized to 0.1. A continuous encoding is required so unseen durations remain
valid; there is no duration lookup table. Sampling rate is not embedded directly because doing so
would expose an easy dataset/configuration identifier.

RoPE continues to receive physical seconds rather than normalized session position. The fastest
period becomes 0.4 seconds when multiresolution is enabled; the fixed arm retains its current
0.5-second setting.

## Objective Semantics

- **A1:** choose a physical time interval and mask every token from every resolution whose support
  overlaps it. Whole-channel masks also apply across all resolutions. During the masked forward,
  temporal attention is isolated within each resolution; otherwise the edge of a masked long token
  could still read an overlapping short token outside the initially selected interval. Clean A2/A3
  forwards retain cross-resolution attention. Average masked-prediction loss within each resolution,
  then average active resolutions so the short scale does not dominate through token count.
- **A2:** pool patches within each resolution, average active resolution summaries equally, and apply
  SupCon to the resulting recording embedding.
- **A3:** use that same scale-balanced recording embedding. Targets are still computed once from the
  final augmented recording, not independently from patches.

Short recordings may have one partial token per resolution. They remain valid for A2/A3; A1 skips a
sample when no non-leaking visible/masked split exists.

## Configuration And Reproducibility

Checkpoints and logs must record:

- headline arm;
- frontend kind and every adaptation bound;
- single- versus multiresolution mode and duration choices;
- sampled duration pair per logged batch;
- frontend residual gate, duration gate, center shifts, bandwidth factors, compression gains, shape,
  and adaptation regularizer;
- existing corpus fingerprint, label map, optimizer/scheduler/scaler, RNG, and source metrics.

Validation is deterministic. The fixed arm retains 1.0-second validation. Multiresolution validation
uses 0.5 + 1.5 seconds. Robustness evaluation should additionally report 0.4 + 1.0, 0.6 + 1.2, and
0.8 + 1.5 seconds, but only the canonical pair selects checkpoints.

## Acceptance Gates

Before a full run:

1. Fixed-arm tensors and outputs retain the existing contracts and tests.
2. Every real sample is covered at both resolutions; padded patches remain exactly zero.
3. The filterbank accepts per-token lengths and produces finite forward/backward values.
4. Adaptive centers remain ordered and bounded; all adaptive parameters receive finite gradients.
5. Duration encoding is bounded and changes tokens for equal-center, unequal-duration patches.
6. Channel permutation invariance still holds with multiresolution metadata.
7. A1 masks are overlap-coupled across scales and its reduction weights scales equally.
8. A2/A3 pooling weights active resolutions equally.
9. Save/resume reconstructs the same frontend and multiresolution configuration.
10. A CPU smoke step completes; no full GPU training is launched as part of implementation.

## Results (2026-07-20)

The headline learnable arm (`--arm learnable` = constrained adaptive frontend **+** multiresolution)
was trained on the exact headline budget — 30k steps, batch 512 (64×8), lr 4.2e-4, seed 20260718,
full 12-dataset / 20-stream native corpus — so the arm is the single variable versus the fixed
headline checkpoint (`pretrain_native/best.pt`). Debug sweep first: 122 tests pass, all three arms
smoke green, and the fixed-arm DSP refactor is numerically identical to the pre-refactor path
(≤1e-7 fp32 noise), so the committed headline checkpoint is unchanged.

**Held-out-config transfer — the decisive metric** (`eval_transfer`: subject-disjoint kNN-BA on each
eval dataset's own labels, identical protocol for both arms; each arm scored with its own native
token presentation):

| dataset | fixed (single-res headline) | **fixed + multiresolution** | learnable arm (frontend + MR) |
|---|---|---|---|
| motionsense | 0.897 | **0.940** | 0.916 |
| realworld | 0.852 | **0.863** | 0.857 |
| shoaib | 0.949 | 0.969 | **0.972** |
| inclusivehar | 0.506 | **0.570** | 0.552 |
| **mean** | **0.801** | **0.835** | 0.824 |

**Multiresolution-on-fixed wins outright (0.835), best on 3 of 4 datasets** including +0.064 on the
free-living `inclusivehar` (the weakest cell). The learnable frontend is a small **net negative**
(0.835 → 0.824): making the filterbank adaptive slightly *hurt* held-out transfer while adding cost.
Internal validation agrees: best val kNN-BA 0.681 (fixed+MR) / 0.690 (learnable+MR) / 0.659
(single-res); across all 30 checkpoints the two MR curves are statistically indistinguishable
(mean gap ~+0.003, sign flips), both well above single-res.

**Attribution — the gain is multiresolution, not the learnable filterbank.** Final adaptation
telemetry after 30k steps:

| knob | init → final | bound | verdict |
|---|---|---|---|
| adaptive residual gate | 0.100 → 0.132 | (0,1) | barely opened |
| band-center shift (max) | 0 → 0.019 oct | ±0.45 log-band | ~pinned at fixed init |
| bandwidth factor | 1.0 → 0.99–1.03 | [0.67, 1.5] | ~pinned |
| compression gain | 1.0 → 1.01–1.08 | [0.5, 2.0] | ~pinned |
| filter shape | 2.0 → 1.98 | [1.5, 2.5] | ~pinned |
| **duration gate (multiresolution)** | 0.100 → **0.178** | (0,1) | **opened ~80%** |

The constrained filterbank stayed within a hair of its fixed-physical initialization and its residual
gate barely opened, while the multiresolution duration gate grew ~80%. So the +0.023 transfer gain is
attributable to **simultaneous multiresolution tokenization on the (effectively) fixed physical
frontend**, not to making the filterbank learnable. This is the better outcome for the thesis: two
fixed temporal supports is still a fully physical, non-learned front end.

**Conclusion (confirmed by the diagnostic):** the `--arm fixed --multiresolution` run (same 30k
budget) scored **0.835** held-out transfer — *above* the learnable arm's 0.824 — proving the entire
gain is multiresolution tokenization, not filterbank learnability. **Recommendation: adopt
multiresolution-on-fixed as the new Phase-A default and retire the learnable frontend.** The learnable
code stays in-tree but opt-in (`--frontend learnable`) as a documented negative result; it is a small
net cost. Follow-on work to make it the default: flip `PretrainConfig.multiresolution` on (keep
`frontend="fixed"`), retrain the headline checkpoint, refit the Pipeline-B ConSE head on the new
encoder, and refresh `docs/baselines/RESULTS_V2.md`. Caveat unchanged ([[ceiling probe]]): Phase A is
not the ZS-XD bottleneck (the text bridge is), so this +0.034 encoder gain is real but secondary to
Phase B.

Reproduce:
```bash
PY=/home/alex/code/HALO/legacy_code/.venv/bin/python
# learnable arm (30k, ~2h on a 24 GB card, peak ~20.5 GB):
$PY -m training.tokenizer.pretrain --arm learnable --device cuda --out training/tokenizer/outputs/pretrain_learnable
# fixed + multiresolution attribution diagnostic:
$PY -m training.tokenizer.pretrain --arm fixed --multiresolution --device cuda --out training/tokenizer/outputs/pretrain_fixed_mr
# head-to-head held-out transfer (each checkpoint):
$PY -m training.tokenizer.eval_transfer --checkpoint <out>/best.pt --device cuda
```
