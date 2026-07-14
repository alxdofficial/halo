# HALO as an Evidence Engine — working design

> **Status:** working design, NOT finalized. This is a *separate line of work* from any
> conventional (softmax/cosine-classifier) training path. It reuses the **tokenizer**; the
> **model mechanism and training harness are different** and live in their own tree (see
> "Code organization" below). Name "evidence" is a placeholder — rename freely.
>
> Captured from a design discussion (2026-07-13/14). Prose is deliberately about *why*, not
> code; nothing here is committed as final.

## 0. The shift
From an **activity classifier** (emit one label) to a **human-activity foundation model**
(emit a *supported, calibrated, extensible analysis*). At inference on a dataset it has
never seen, the model behaves like: *"I'm streaming an unknown signal; it resembles these
past labeled things with these confidences; here is the evidence for/against each candidate;
I may abstain."* The structured analysis — not the top-1 label — is the deliverable, and it
survives even when the label is wrong.

## 1. Thesis: salient, NOT invariant (config-conditional)
The field chases **invariance** (augment/normalize away orientation, rate, placement). But
invariance destroys information — an orientation-invariant feature like `|accel|` is robust
*because* it threw away the direction that discriminates activities. **We want features that
stay class-discriminative across heterogeneity because the model is TOLD the config (via
free-form channel text) and normalizes *conditionally*** — not blindly projecting the
nuisance out. Precise objective: same activity across two sensors/placements/times maps
*close* **given each config's description**, while *keeping* the discriminative structure.

Differentiators: vs **UniMTS** (blind orientation-invariance) → config-conditional salience;
vs **ZeroHAR** (fixed attribute schema) → emergent physical primitives + free-form text;
vs **ConSE** baselines (heuristic convex blend of seen labels) → a learned, calibrated,
abstaining, structured evidence decoder.

## 2. Architecture (concern flow)
```
signal ──▶ TOKENIZER (shared, physical filterbank + extensions)  ──▶ multi-view PRIMITIVES
                                                                        (band × channel × cross-channel)
PRIMITIVES ──▶ (config-conditional) ENCODER ──▶ query primitives
query ──▶ ARCHETYPAL MEMORY (curated, non-parametric, evolving) ──▶ retrieved evidence
{retrieved evidence + candidate text hypotheses} ──▶ EVIDENCE DECODER ──▶ ANALYSIS
ANALYSIS = per-candidate {evidence-for, evidence-against, analogues, calibrated confidence, ABSTAIN}
```
The **feature unit is a primitive, not a patch.** A patch is a time-box; the *feature* is
"this band's energy on this channel," "this cross-channel phase pattern," "this cadence."
A never-seen activity is then a **novel combination of familiar primitives** → retrieval
*composes* instead of matching whole embeddings.

## 3. Two evidence views (unified)
- **View 1 — retrieval/analogy:** "resembles labeled memory X, with confidence." (data-driven;
  Matching/Prototypical Nets, kNN-LM/RETRO; a *learned, calibrated* generalization of ConSE.)
- **View 2 — hypothesis/abductive:** "for Y to be true I'd expect A,B,C; seeing them raises Y,
  contradicting them lowers Y." (model-driven; energy-based / analysis-by-synthesis /
  predictive-coding; concrete instantiation = **Evidential Deep Learning** = Dirichlet evidence
  per class → confidence + "I don't know".)
- **Unification:** both are *evidence sources feeding one decoder* that **accumulates across
  patches** into per-candidate beliefs + abstention. Streaming = sequential evidence
  accumulation (Bayesian / drift-diffusion).

## 4. Archetypal memory (non-parametric, curated, evolving)
Confirmed non-parametric: stored real features, *not* free-parameter vectors; a **learnable
GATE** decides what to keep/evict, everything **soft** (differentiable).

**Bootstrapping scheme (proposed):** start with a *vast* bank (≈ every instance), soft-attend
over it, use attention telemetry to surface archetypes; as training proceeds, newer instances
carry *better* features (encoder improved) and roll in alongside retained archetypes; total
size bounded, feature *quality* rises over time.

**Scale reframe:** at our corpus size (~262k windows × 512-d ≈ 0.5 GB; per-primitive ~2–3 GB)
VRAM is **not** the binding constraint — you could hold the whole corpus. Election is a
*quality* decision (clean retrieval neighborhood), not a survival one.

**Three landmines + fixes (this is a curated MoCo queue / DND, with MoCo's known pitfall):**
1. **Representation drift (the big one).** Stored embeddings go stale as the encoder trains;
   matching a v50 query against v5 vectors compares across encoder generations. Fix: store the
   **signal + cached embedding, re-encode (lazily)**, or a **momentum/slow encoder**, or
   periodic re-encode. *Storing primitives (more stable, cheaper to re-encode) mitigates this.*
2. **Attention-mass ≠ utility → head-class collapse.** Most-attended = most-frequent; rare
   classes starve (we already have 0/2-support classes), and generic "hub" entries get kept.
   Fix: **per-class/per-primitive coverage quotas** + a **discriminability-weighted utility**
   ("did retrieving you move the prediction toward truth"), + **diversity** (anti-redundancy).
   Decay telemetry (recent, not cumulative — cumulative mixes encoder eras).
3. **Full softmax is diffuse + slow.** Fix: **soft attention over a hard-retrieved top-k**
   (ANN → k → softmax over k). The *blend* stays differentiable (gradients to query, k values,
   encoder); only *which-k* is non-differentiable, which is fine — we learn an encoder that
   puts good neighbors nearby, not "which neighbor to pick" (kNN-LM / Neural Episodic Control).

## 5. "Useful even when wrong" — two testable properties (the foundation-model claim + its eval)
1. **Analysis-consistency (same-label ⇒ same-analysis).** The evidence/analysis object should
   cluster by *true label* even when top-1 is wrong. A metric-learning objective *on the
   analysis itself*, and a **novel evaluation** decoupled from accuracy (clustering/retrieval
   quality of the analysis vector).
2. **GT-coherence / the "aha" + online write.** When the true label is revealed it should be
   **evidence-consistent** (high evidence-rank, not orthogonal) — measured as GT evidence-rank /
   "surprise." The online version *is* the emergence: on seeing GT, **write (primitives, label)
   into the bank**; the "now it makes sense" is real iff that write (a) reduces future error on
   similar samples and (b) slots in coherently. Same machinery as deployment-time learning.
   (Caveat: "aha" as a feeling is not a metric; GT evidence-rank + online improvement are.)

## 6. Structured positive/negative, located evidence (the frontier — high ceiling, high risk)
"A,B,C present at locations A,B,C **and** D,E,F absent ⇒ Y" — a structured energy / factor
graph over *located* features with **inhibitory** (negative) terms. Negative evidence is
underused in HAR and is what gives clean mutual-exclusion + abstention. Fork: let it **emerge**
from attention with signed weights + a coherence penalty (buildable, less controllable) vs make
it **explicit** (neuro-symbolic; expressive but combinatorial, easy to become a hand-wavy graph
that won't train). **Prototype the emergent version first.**

## 7. Tokenizer direction (shared substrate — brainstorm)
Current: physical-Hz **filterbank** (per-channel spectral, rate-invariant) + signed DC/gravity.
Good, but for *fine-grained, heterogeneity-salient primitives within one patch* it's per-channel,
amplitude-absolute, single-resolution, and monolithic. Candidate extensions, grouped by the
robustness they buy (each must be **ablatable**; physics-grounded hand-crafting is *on-thesis*
because it's more heterogeneity-robust than learned filters):

- **A. Cross-channel relational (biggest salience win — "patterns formed by channels"):**
  per-band **coherence/phase** between channels (amplitude-normalized → gain-robust; encodes
  movement plane/direction); inter-channel **covariance/correlation** structure;
  **orientation-normalized projection** (within-patch PCA of the accel triad → dominant motion
  plane → project → mounting-angle/placement-robust axes = *earned*, config-conditional invariance).
- **B. Amplitude-relative / scale-free (gain & g-vs-m/s² robust):** **relative band energies**
  (band/total); spectral **shape** descriptors (centroid, bandwidth, flatness, roll-off = motion
  "timbre"). Keep absolute DC/gravity too (both views available).
- **C. Multi-resolution / transients:** **wavelet** or multi-window STFT (footfall transient +
  gait rhythm together); **jerk** (d/dt accel) spectral energy (movement sharpness).
- **D. Periodicity / cadence (very discriminative + sensor-robust):** autocorrelation fundamental
  period + harmonic ratios (steps/sec, physical Hz); periodicity strength (regular vs irregular).
- **E. Emit STRUCTURED primitives, not a monolithic patch vector** — a named set the memory
  stores and the decoder attends over → compositional retrieval.
- Guardrails: prune redundancy (coherence/cov/corr overlap — pick minimal); optional small
  **gated learned residual** on top of the physical bank (adapt without losing robustness);
  optional **config-conditional normalization** via channel text (powerful, couples tokenizer
  to language — keep optional).

## 8. Open design forks (to resolve before building)
1. **The load-bearing loss:** write the "pull/push *conditioned on config* + analysis-consistency"
   objective cleanly. If it's coherent, §2–§6 are engineering around it.
2. **Utility signal for the memory:** task-supervised ("retrieval helped predict truth") vs
   self-consistency ("retrieval pulled same-label together"). Lean self-consistency-weighted blend
   (more foundation-model), but this is the real fork.
3. **Structured evidence:** emergent-inhibition (first) vs explicit neuro-symbolic (stretch).
4. **Memory freshness:** re-encode-signal vs momentum-encoder vs periodic-refresh.

## 9. Code organization (keep separate from conventional training)
```
model/tokenizer/     SHARED — port the legacy PhysicalFilterbankTokenizer + the §7 extensions.
model/evidence/      NEW — primitive encoder, archetypal memory, evidence decoder.
training/evidence/    NEW — the retrieval/evidence training harness (its own loop, NOT a
                     conventional classifier trainer). See training/evidence/README.md.
docs/design/EVIDENCE_ENGINE.md   this file (design/rationale).
docs/design/EVIDENCE_ENGINE_BUILD_PLAN.md   the milestoned build plan (how + order + gates).
```
The tokenizer is the only shared component; the mechanism + harness are self-contained so this
line never conflates with a standard softmax/cosine training path.

## 10. Novelty triangulation (honest)
Individually known: Matching/Prototypical Nets, kNN-LM/RETRO, Evidential Deep Learning, OpenMax
(open-set), ConSE, MoCo (memory queue). **The claim is the combination tied to free-form language
conditioning + heterogeneity-salient physical primitives + the analysis-artifact/consistency
evaluation (usefulness ≠ accuracy).** Each piece must be ablated; the one-sentence hook is
*config-conditional salient features feeding a curated-memory evidence engine that produces a
consistent, abstaining analysis — turning HAR from closed classification into an analysis substrate.*
